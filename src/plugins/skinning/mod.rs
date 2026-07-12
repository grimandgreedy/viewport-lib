//! GPU skinning as a deformer-registry plugin.
//!
//! `SkinningPlugin::install` registers a linear-blend-skinning deformer body
//! against the mesh shader family and returns a handle. The handle holds the
//! assigned `DeformerId` and exposes `attach_weights` and `attach_palette`
//! for per-mesh and per-(mesh, instance) data uploads, plus
//! `is_skinned_mesh` so hosts can answer the "does this mesh have skinning
//! data attached" question without consulting the registry.
//!
//! The handle has a symmetric lifecycle: `install` -> `attach_weights` /
//! `attach_palette` (or the `begin_upload_weights` / event forms) ->
//! `detach_weights` / `detach_palette` for individual meshes, or `uninstall`
//! to reclaim every attached buffer at once. The deformer body stays
//! registered for the session (its slot is an append-only registry entry), so
//! re-installing after `uninstall` returns the same `DeformerId` at no cost.
//!
//! Hosts that do not need GPU skinning never call `install` and pay nothing.
//! Static meshes pay nothing either way: the per-object `deform_flags`
//! branch in the composed shader gates the LBS body off.
//!
//! `SkinningPlugin` is not a [`RuntimePlugin`](crate::RuntimePlugin); it is a
//! renderer-side upload handle. Pair it with
//! [`SkeletonPlugin`](crate::plugins::skeleton::SkeletonPlugin) or
//! [`SkinnedActorPlugin`](crate::plugins::skeleton::SkinnedActorPlugin) for the
//! runtime half: those compute the per-frame joint matrices and emit
//! [`SkinnedPoseUpdate`] events on `output.events`; the host drains the events
//! and calls [`SkinningPlugin::attach_palette`] on each one.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::resources::DeviceResources;
use crate::resources::mesh::mesh_store::MeshId;
use crate::resources::mesh_sidecar::registry::{DeformStage, DeformerDesc, DeformerId};

/// A per-mesh deformation update produced by a skinning plugin on the CPU
/// path. Apply by calling `write_mesh_positions_normals`:
///
/// ```rust,ignore
/// for u in output.events.drain::<SkinnedMeshUpdate>() {
///     renderer.resources_mut()
///         .write_mesh_positions_normals(queue, u.mesh_id, &u.positions, &u.normals)
///         .ok();
/// }
/// ```
pub struct SkinnedMeshUpdate {
    /// The mesh to deform.
    pub mesh_id: MeshId,
    /// Skinned vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Skinned vertex normals.
    pub normals: Vec<[f32; 3]>,
}

/// A per-instance joint palette update produced by a skinning plugin on the
/// GPU path. Apply by calling [`SkinningPlugin::attach_palette`]:
///
/// ```rust,ignore
/// for u in output.events.drain::<SkinnedPoseUpdate>() {
///     skinning.attach_palette(
///         renderer.resources_mut(), &device, &queue,
///         u.mesh_id, u.instance_id, &u.joint_matrices,
///     );
/// }
/// ```
pub struct SkinnedPoseUpdate {
    /// The skinned mesh to drive.
    pub mesh_id: MeshId,
    /// Which instance of the mesh this palette is for. Use `0` for single-
    /// instance meshes.
    pub instance_id: u32,
    /// Per-joint skinning matrices in topological order, ready for upload to
    /// the GPU joint palette storage buffer.
    pub joint_matrices: Vec<glam::Mat4>,
}

/// Per-vertex joint influence data for linear blend skinning.
///
/// # Invariants
///
/// - `joint_indices.len() == joint_weights.len() == positions.len()` on the
///   accompanying `MeshData`.
/// - Each vertex carries up to four influences. Unused slots must have weight
///   `0.0` and a valid (any in-range) index; the CPU path skips entries below
///   `1e-6`.
/// - Weights per vertex should sum to `1.0`. The CPU path does not renormalise,
///   so a vertex whose weights sum to less than 1 will deform with reduced
///   magnitude. Importers should normalise before constructing this.
/// - There is no required ordering between the four slots.
///
/// Consumed by both paths: `plugins::skeleton::apply_skin` reads it on the
/// CPU path; [`SkinningPlugin::attach_weights`] packs it into the
/// deformer's per-mesh slot on the GPU path.
#[derive(Clone)]
pub struct SkinWeights {
    /// Joint indices for each vertex: 4 per vertex, parallel to positions.
    pub joint_indices: Vec<[u8; 4]>,
    /// Blend weights for each vertex: 4 per vertex, normalised to sum 1.0.
    pub joint_weights: Vec<[f32; 4]>,
}

// ---------------------------------------------------------------------------
// Packed vertex format and stride constants
// ---------------------------------------------------------------------------

/// Packed per-vertex skin data: four `f32` weights followed by two packed
/// joint-index `u32`s. Total 24 bytes, matching the per-vertex stride
/// expected by the deformer's skinning body.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedSkinVertex {
    weights: [f32; 4],
    /// `joints[0]` in the low 16 bits, `joints[1]` in the high 16 bits.
    joints_01: u32,
    /// `joints[2]` in the low 16 bits, `joints[3]` in the high 16 bits.
    joints_23: u32,
}

/// Per-vertex stride of the skin-weights slot.
const SKIN_WEIGHT_STRIDE_BYTES: u32 = 24;

/// Per-instance stride of the joint-palette slot: one `mat4x4<f32>` per joint.
const SKIN_PALETTE_STRIDE_BYTES: u32 = 64;

// ---------------------------------------------------------------------------
// Deformer body
// ---------------------------------------------------------------------------

/// Priority assigned to the skinning deformer. Negative so morph-target and
/// other object-space deformers, which register at the default priority of
/// 0, run after it.
pub const DEFORM_PRIORITY_SKINNING: i32 = -1000;

const SKINNING_DEFORMER_NAME: &str = "viewport_skin";

const SKINNING_DEFORMER_BODY: &str = r#"
fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    var out = v;
    let slot = ctx.slot;
    if deform_slot_stride(slot) == 0u {
        return out;
    }
    let vi = v.vertex_index;
    let w0 = deform_read_f32(slot, vi, 0u);
    let w1 = deform_read_f32(slot, vi, 1u);
    let w2 = deform_read_f32(slot, vi, 2u);
    let w3 = deform_read_f32(slot, vi, 3u);
    let j01 = deform_read_u32(slot, vi, 4u);
    let j23 = deform_read_u32(slot, vi, 5u);
    let j0 = j01 & 0xFFFFu;
    let j1 = (j01 >> 16u) & 0xFFFFu;
    let j2 = j23 & 0xFFFFu;
    let j3 = (j23 >> 16u) & 0xFFFFu;
    let m = deform_read_instance_mat4(slot, j0) * w0
          + deform_read_instance_mat4(slot, j1) * w1
          + deform_read_instance_mat4(slot, j2) * w2
          + deform_read_instance_mat4(slot, j3) * w3;
    out.position = (m * vec4<f32>(v.position, 1.0)).xyz;
    let m3 = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
    out.normal = m3 * v.normal;
    return out;
}
"#;

// ---------------------------------------------------------------------------
// Plugin handle
// ---------------------------------------------------------------------------

/// Handle to the GPU skinning deformer.
///
/// Returned by [`SkinningPlugin::install`]. Holds the assigned
/// [`DeformerId`] plus a marker set tracking which meshes have had weights
/// attached, so [`Self::is_skinned_mesh`] can answer without going through
/// the registry. Clone-cheap: the marker set is shared.
#[derive(Clone)]
pub struct SkinningPlugin {
    deformer_id: DeformerId,
    skinned_meshes: Arc<Mutex<HashMap<MeshId, ()>>>,
}

impl SkinningPlugin {
    /// Register the skinning deformer with the renderer.
    ///
    /// Call once at startup before uploading any skin data. Composes the LBS
    /// body into the mesh shader family and validates the result through
    /// wgpu's error scope.
    ///
    /// # Errors
    ///
    /// Propagates the registry error if shader composition or validation
    /// fails (extremely unlikely for the shipped body, which is covered by
    /// unit tests).
    pub fn install(
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
    ) -> crate::error::ViewportResult<Self> {
        let deformer_id = match resources.deformer_id_by_name(SKINNING_DEFORMER_NAME) {
            Some(id) => id,
            None => {
                let desc = DeformerDesc {
                    name: SKINNING_DEFORMER_NAME,
                    stage: DeformStage::ObjectSpace,
                    priority: DEFORM_PRIORITY_SKINNING,
                    wgsl_body: SKINNING_DEFORMER_BODY.to_string(),
                    per_vertex_stride: SKIN_WEIGHT_STRIDE_BYTES,
                };
                resources.register_internal_deformer(device, desc)?
            }
        };
        Ok(Self {
            deformer_id,
            skinned_meshes: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// The [`DeformerId`] assigned to skinning on install.
    pub fn deformer_id(&self) -> DeformerId {
        self.deformer_id
    }

    /// Attach per-vertex skin weights to an uploaded mesh.
    ///
    /// Marks the mesh as skinnable and packs the weights into the deformer's
    /// per-mesh slot. Calling again on the same mesh replaces the prior
    /// weights buffer.
    pub fn attach_weights(
        &self,
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        weights: &SkinWeights,
    ) {
        let packed = pack(weights);
        let tight = pack_skin_weights_tight(&packed);
        resources.deform.attach_slot(
            device,
            mesh_id,
            self.deformer_id.slot(),
            SKIN_WEIGHT_STRIDE_BYTES / 4,
            &tight,
        );
        self.skinned_meshes
            .lock()
            .expect("skinning marker poisoned")
            .insert(mesh_id, ());
    }

    /// Start an asynchronous skin-weights upload.
    ///
    /// Returns a `JobId` immediately. The packed weight stream is computed on
    /// a worker thread; buffer creation runs on the main thread during the
    /// next `process_uploads` call after the worker finishes.
    pub fn begin_upload_weights(
        &self,
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        weights: SkinWeights,
    ) -> crate::resources::JobId {
        let device_for_apply = device.clone();
        let slot = self.deformer_id.slot();
        let marker = self.skinned_meshes.clone();
        let mut runner = resources.jobs.lock().expect("upload job runner poisoned");
        runner.submit_cpu(move |progress| {
            progress.set(0.2);
            let packed = pack(&weights);
            let tight = pack_skin_weights_tight(&packed);
            progress.set(0.9);
            Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                Box::new(move |resources: &mut DeviceResources| {
                    resources.deform.attach_slot(
                        &device_for_apply,
                        mesh_id,
                        slot,
                        SKIN_WEIGHT_STRIDE_BYTES / 4,
                        &tight,
                    );
                    marker
                        .lock()
                        .expect("skinning marker poisoned")
                        .insert(mesh_id, ());
                }),
            ))
        })
    }

    /// Upload the joint palette for one instance of a skinned mesh.
    ///
    /// `instance_id` lets multiple skinned instances of one bind-pose mesh
    /// coexist. For single-instance meshes pass `0`. Returns `false` if
    /// [`Self::attach_weights`] has not been called for `mesh_id`.
    ///
    /// `palette[i]` is the object-space skinning matrix for joint `i`
    /// produced by [`crate::JointMatrices::compute`]: the joint's
    /// skeleton-local transform multiplied by its inverse bind. The mesh's
    /// `object.model` is applied separately at draw time, so the palette
    /// composes with the scene node transform rather than replacing it.
    pub fn attach_palette(
        &self,
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        palette: &[glam::Mat4],
    ) -> bool {
        if !self
            .skinned_meshes
            .lock()
            .expect("skinning marker poisoned")
            .contains_key(&mesh_id)
        {
            return false;
        }
        let bytes: Vec<[[f32; 4]; 4]> = palette.iter().map(|m| m.to_cols_array_2d()).collect();
        resources.deform.attach_slot_instance(
            device,
            queue,
            mesh_id,
            instance_id,
            self.deformer_id.slot(),
            SKIN_PALETTE_STRIDE_BYTES / 4,
            bytemuck::cast_slice(&bytes),
        );
        true
    }

    /// Whether `mesh_id` has been marked as skinnable via
    /// [`Self::attach_weights`].
    pub fn is_skinned_mesh(&self, mesh_id: MeshId) -> bool {
        self.skinned_meshes
            .lock()
            .expect("skinning marker poisoned")
            .contains_key(&mesh_id)
    }

    /// Detach the skin weights attached to `mesh_id` and unmark it as skinnable.
    ///
    /// Reverses [`attach_weights`](Self::attach_weights): reclaims the per-mesh
    /// weight buffer and drops the mesh from the skinnable set, so it renders as
    /// a static mesh afterward (the deformer branch gates off once its slot has
    /// no data). Returns `true` if weights were removed, `false` if the mesh had
    /// none. Any joint palettes attached for the mesh become inert; reclaim them
    /// with [`detach_palette`](Self::detach_palette) or by freeing the mesh.
    pub fn detach_weights(
        &self,
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
    ) -> bool {
        let removed = resources.detach_deform_slot(device, mesh_id, self.deformer_id.slot());
        self.skinned_meshes
            .lock()
            .expect("skinning marker poisoned")
            .remove(&mesh_id);
        removed
    }

    /// Detach the joint palette for one instance of a skinned mesh.
    ///
    /// Reverses [`attach_palette`](Self::attach_palette) for a single
    /// `(mesh_id, instance_id)` pair, reclaiming that palette buffer. Returns
    /// `true` if a palette was removed, `false` if none was attached. Leaves the
    /// mesh's weight slot and skinnable marker in place.
    pub fn detach_palette(
        &self,
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
    ) -> bool {
        resources.detach_deform_slot_instance(
            device,
            queue,
            mesh_id,
            instance_id,
            self.deformer_id.slot(),
        )
    }

    /// Tear down every per-mesh skin buffer attached through this handle.
    ///
    /// Detaches the weight slot of every mesh marked skinnable and clears the
    /// marker set, reclaiming that GPU memory. Consumes the handle: its
    /// lifecycle is `install` -> `attach_weights` / `attach_palette` (or the
    /// `begin_upload_weights` / event forms) -> `uninstall`.
    ///
    /// The deformer body itself stays registered for the session: `DeformerId`
    /// is an append-only registry slot, so a later [`install`](Self::install)
    /// returns the same id at no cost. Per-instance palettes are keyed by
    /// `(mesh, instance)` and are not tracked here; reclaim them with
    /// [`detach_palette`](Self::detach_palette) or by freeing the meshes before
    /// uninstalling if their buffers must go immediately.
    pub fn uninstall(self, resources: &mut DeviceResources, device: &crate::gpu::Device) {
        let slot = self.deformer_id.slot();
        let meshes: Vec<MeshId> = {
            let marker = self
                .skinned_meshes
                .lock()
                .expect("skinning marker poisoned");
            marker.keys().copied().collect()
        };
        for mesh_id in meshes {
            resources.detach_deform_slot(device, mesh_id, slot);
        }
        self.skinned_meshes
            .lock()
            .expect("skinning marker poisoned")
            .clear();
    }
}

// ---------------------------------------------------------------------------
// Packing helpers
// ---------------------------------------------------------------------------

fn pack(weights: &SkinWeights) -> Vec<PackedSkinVertex> {
    weights
        .joint_indices
        .iter()
        .zip(weights.joint_weights.iter())
        .map(|(j, w)| {
            let j0 = j[0] as u32;
            let j1 = j[1] as u32;
            let j2 = j[2] as u32;
            let j3 = j[3] as u32;
            PackedSkinVertex {
                weights: *w,
                joints_01: j0 | (j1 << 16),
                joints_23: j2 | (j3 << 16),
            }
        })
        .collect()
}

fn pack_skin_weights_tight(packed: &[PackedSkinVertex]) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed.len() * SKIN_WEIGHT_STRIDE_BYTES as usize);
    for v in packed {
        out.extend_from_slice(bytemuck::bytes_of(&v.weights));
        out.extend_from_slice(bytemuck::bytes_of(&v.joints_01));
        out.extend_from_slice(bytemuck::bytes_of(&v.joints_23));
    }
    out
}

#[cfg(test)]
mod async_skin_tests {
    use super::SkinWeights;
    use super::SkinningPlugin;
    use crate::DeviceResources;
    use crate::geometry::primitives;
    use crate::resources::UploadStatus;

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn unit_weights(vertex_count: usize) -> SkinWeights {
        SkinWeights {
            joint_indices: vec![[0u8; 4]; vertex_count],
            joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; vertex_count],
        }
    }

    #[test]
    fn attach_weights_marks_mesh_skinnable() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let skinning = SkinningPlugin::install(&mut resources, &device).unwrap();
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let weights = unit_weights(plane.positions.len());

        skinning.attach_weights(&mut resources, &device, mesh_id, &weights);
        assert!(skinning.is_skinned_mesh(mesh_id));
    }

    #[test]
    fn detach_weights_unmarks_mesh() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let skinning = SkinningPlugin::install(&mut resources, &device).unwrap();
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        skinning.attach_weights(
            &mut resources,
            &device,
            mesh_id,
            &unit_weights(plane.positions.len()),
        );
        assert!(skinning.is_skinned_mesh(mesh_id));

        assert!(skinning.detach_weights(&mut resources, &device, mesh_id));
        assert!(
            !skinning.is_skinned_mesh(mesh_id),
            "detach must unmark the mesh"
        );
        assert!(
            !skinning.detach_weights(&mut resources, &device, mesh_id),
            "detaching again removes nothing"
        );
    }

    #[test]
    fn uninstall_detaches_all_skinned_meshes() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let skinning = SkinningPlugin::install(&mut resources, &device).unwrap();
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let a = resources.upload_mesh_data(&device, &plane).unwrap();
        let b = resources.upload_mesh_data(&device, &plane).unwrap();
        let w = unit_weights(plane.positions.len());
        skinning.attach_weights(&mut resources, &device, a, &w);
        skinning.attach_weights(&mut resources, &device, b, &w);
        let slot = skinning.deformer_id().slot();

        let probe = skinning.clone();
        skinning.uninstall(&mut resources, &device);
        assert!(!probe.is_skinned_mesh(a));
        assert!(!probe.is_skinned_mesh(b));
        assert!(
            !resources.has_deform_slot(a, slot),
            "weight slot must be freed"
        );
        assert!(
            !resources.has_deform_slot(b, slot),
            "weight slot must be freed"
        );
    }

    #[test]
    fn begin_upload_weights_completes() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let skinning = SkinningPlugin::install(&mut resources, &device).unwrap();
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let weights = unit_weights(plane.positions.len());

        assert!(!skinning.is_skinned_mesh(mesh_id));
        let id = skinning.begin_upload_weights(&mut resources, &device, mesh_id, weights);

        for _ in 0..200 {
            resources.process_uploads(&device, &queue);
            match resources.upload_status(id) {
                UploadStatus::Ready => break,
                UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                UploadStatus::Unknown => panic!("job disappeared"),
            }
        }

        assert!(
            skinning.is_skinned_mesh(mesh_id),
            "skin weights did not install"
        );
    }
}
