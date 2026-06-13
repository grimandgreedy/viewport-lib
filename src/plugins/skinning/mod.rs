//! GPU skinning routed through the deformer registry.
//!
//! Skinning is registered at renderer construction as an internal deformer
//! on a reserved slot. `set_skin_weights` packs per-vertex weights into the
//! deformer's per-mesh slot, `set_skin_palette` writes the joint matrices
//! into the deformer's per-instance slot, and the standard mesh shaders
//! apply LBS during the object-space deform stage. Static meshes pay zero
//! overhead.

use std::collections::HashMap;

use crate::resources::SkinWeights;
use crate::resources::ViewportGpuResources;
use crate::resources::mesh_store::MeshId;

/// Packed per-vertex skin data: four `f32` weights followed by two packed
/// joint-index `u32`s. Total 24 bytes, matching the per-vertex stride
/// expected by the deformer's skinning body.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PackedSkinVertex {
    pub weights: [f32; 4],
    /// `joints[0]` in the low 16 bits, `joints[1]` in the high 16 bits.
    pub joints_01: u32,
    /// `joints[2]` in the low 16 bits, `joints[3]` in the high 16 bits.
    pub joints_23: u32,
}

/// Renderer-side skinning state. Tracks which meshes are skinnable so
/// `is_skinned_mesh` can answer without consulting the deformer registry.
pub(crate) struct SkinningState {
    pub meshes: HashMap<MeshId, ()>,
}

impl SkinningState {
    pub fn new(_device: &wgpu::Device) -> Self {
        Self {
            meshes: HashMap::new(),
        }
    }

    pub(crate) fn pack(weights: &SkinWeights) -> Vec<PackedSkinVertex> {
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
}

// ---------------------------------------------------------------------------
// Internal skinning deformer
// ---------------------------------------------------------------------------

/// Priority assigned to the in-crate skinning deformer. Negative so
/// morph-target and other object-space deformers, which register at the
/// default priority of 0, run after it.
pub const DEFORM_PRIORITY_SKINNING: i32 = -1000;

/// Per-vertex stride of the skin-weights slot: four `f32` weights followed
/// by two packed joint-index `u32`s.
pub const SKIN_WEIGHT_STRIDE_BYTES: u32 = 24;

/// Per-instance stride of the joint-palette slot: one `mat4x4<f32>` per
/// joint.
#[allow(dead_code)]
pub const SKIN_PALETTE_STRIDE_BYTES: u32 = 64;

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

/// Re-pack the legacy `PackedSkinVertex` array as a tight 24-byte stream
/// matching the deformer slot's expected stride: four `f32` weights
/// followed by the two packed joint-index u32s, with no trailing padding.
fn pack_skin_weights_tight(packed: &[PackedSkinVertex]) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed.len() * SKIN_WEIGHT_STRIDE_BYTES as usize);
    for v in packed {
        out.extend_from_slice(bytemuck::bytes_of(&v.weights));
        out.extend_from_slice(bytemuck::bytes_of(&v.joints_01));
        out.extend_from_slice(bytemuck::bytes_of(&v.joints_23));
    }
    out
}

/// Install GPU skinning against the renderer.
///
/// Registers the skinning deformer on a reserved internal slot and wires
/// `set_skin_weights` / `set_skin_palette` to it. Call once at startup
/// before uploading any skin data. Subsequent calls return early without
/// re-registering.
///
/// Returns the assigned [`DeformerId`](crate::DeformerId) on first
/// install, or the previously assigned id on subsequent calls.
///
/// # Errors
///
/// Propagates the underlying registry error if shader composition or
/// validation fails (extremely unlikely for the shipped body, which is
/// covered by unit tests).
pub fn install_skinning(
    resources: &mut ViewportGpuResources,
    device: &wgpu::Device,
) -> crate::error::ViewportResult<crate::DeformerId> {
    if let Some(id) = resources.skinning_slot {
        return Ok(id);
    }
    use crate::resources::mesh_sidecar::registry::{DeformStage, DeformerDesc};
    let desc = DeformerDesc {
        name: SKINNING_DEFORMER_NAME,
        stage: DeformStage::ObjectSpace,
        priority: DEFORM_PRIORITY_SKINNING,
        wgsl_body: SKINNING_DEFORMER_BODY.to_string(),
        per_vertex_stride: SKIN_WEIGHT_STRIDE_BYTES,
    };
    let id = resources.register_internal_deformer(device, desc)?;
    resources.skinning_slot = Some(id);
    Ok(id)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl ViewportGpuResources {
    /// Whether GPU skinning is installed.
    ///
    /// Returns `true` after a successful call to
    /// [`install_skinning`](crate::plugins::skinning::install_skinning),
    /// `false` otherwise. Plugins read this to decide whether to emit
    /// [`crate::SkinnedPoseUpdate`] (GPU path) or
    /// [`crate::SkinnedMeshUpdate`] (CPU path) each frame.
    pub fn supports_gpu_skinning(&self) -> bool {
        self.skinning_slot.is_some()
    }

    /// Attach per-vertex skin weights to an uploaded mesh.
    ///
    /// Creates a sidecar storage buffer holding packed joint indices + weights
    /// (24 bytes per vertex) and the bind group used by the skinned pipeline
    /// variants. The mesh's vertex buffer is not modified.
    ///
    /// Calling this on a `mesh_id` marks the mesh as skinnable: subsequent
    /// draws of any scene node referencing this mesh are routed through the
    /// skinned pipeline variant once a joint palette has been uploaded via
    /// [`Self::set_skin_palette`].
    ///
    /// Calling this again on the same `mesh_id` replaces the weights buffer
    /// and invalidates all previously created instance bind groups; consumers
    /// should re-upload palettes for any active instances afterwards.
    pub fn set_skin_weights(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        weights: &SkinWeights,
    ) {
        let packed = SkinningState::pack(weights);
        self.install_skin_weights(device, mesh_id, &packed);
    }

    /// Start an asynchronous skin-weights upload.
    ///
    /// Returns a `JobId` immediately. The packed weight stream (joint
    /// indices and blend weights, four per vertex) is computed on a worker
    /// thread; buffer creation and the `mesh.skinning` insert run on the
    /// main thread during the next `process_uploads` call after the worker
    /// finishes.
    ///
    /// `weights` transfers into the worker; clone at the call site to
    /// retain ownership.
    ///
    /// As with the synchronous `set_skin_weights`, this replaces any prior
    /// weights for the same `mesh_id` and invalidates active per-instance
    /// bind groups. Re-upload palettes via `set_skin_palette` afterwards.
    pub fn begin_upload_skin_weights(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        weights: SkinWeights,
    ) -> crate::resources::JobId {
        let device_for_apply = device.clone();
        let mut runner = self.jobs.lock().expect("upload job runner poisoned");
        runner.submit_cpu(move |progress| {
            progress.set(0.2);
            let packed = SkinningState::pack(&weights);
            progress.set(0.9);
            Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                Box::new(move |resources: &mut ViewportGpuResources| {
                    resources.install_skin_weights(&device_for_apply, mesh_id, &packed);
                }),
            ))
        })
    }

    /// Shared apply step for both `set_skin_weights` and
    /// `begin_upload_skin_weights`: mark the mesh as skinnable and attach
    /// the tight-packed weights to the internal skinning deformer slot.
    fn install_skin_weights(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        packed: &[PackedSkinVertex],
    ) {
        self.skinning.meshes.insert(mesh_id, ());
        if let Some(slot_id) = self.skinning_slot {
            let tight = pack_skin_weights_tight(packed);
            self.deform.attach_slot(
                device,
                mesh_id,
                slot_id.slot(),
                SKIN_WEIGHT_STRIDE_BYTES / 4,
                &tight,
            );
        }
    }

    /// Upload the joint palette for one instance of a skinned mesh.
    ///
    /// `instance_id` lets multiple skinned instances of one bind-pose mesh
    /// coexist (the crowd case). For single-instance meshes pass `0`.
    ///
    /// Allocates or grows the per-instance palette storage buffer as needed.
    /// Re-binds the per-instance bind group when the buffer is reallocated.
    /// `set_skin_weights` must have been called for `mesh_id` first.
    ///
    /// `palette[i]` is the object-space skinning matrix for joint `i`
    /// produced by [`crate::JointMatrices::compute`]: the joint's
    /// skeleton-local transform multiplied by its inverse bind. The mesh's
    /// `object.model` is applied separately at draw time, so the palette
    /// composes with the scene node transform rather than replacing it.
    pub fn set_skin_palette(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        palette: &[glam::Mat4],
    ) -> bool {
        if !self.skinning.meshes.contains_key(&mesh_id) {
            return false;
        }
        let Some(slot_id) = self.skinning_slot else {
            return false;
        };
        let bytes: Vec<[[f32; 4]; 4]> = palette.iter().map(|m| m.to_cols_array_2d()).collect();
        self.deform.attach_slot_instance(
            device,
            queue,
            mesh_id,
            instance_id,
            slot_id.slot(),
            SKIN_PALETTE_STRIDE_BYTES / 4,
            bytemuck::cast_slice(&bytes),
        );
        true
    }

    /// Whether `mesh_id` has been marked as skinnable via
    /// [`Self::set_skin_weights`].
    pub fn is_skinned_mesh(&self, mesh_id: MeshId) -> bool {
        self.skinning.meshes.contains_key(&mesh_id)
    }
}

#[cfg(test)]
mod async_skin_tests {
    use crate::ViewportGpuResources;
    use crate::geometry::primitives;
    use crate::resources::{SkinWeights, UploadStatus};

    fn try_make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn unit_weights(vertex_count: usize) -> SkinWeights {
        SkinWeights {
            joint_indices: vec![[0u8; 4]; vertex_count],
            joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; vertex_count],
        }
    }

    #[test]
    fn sync_set_skin_weights_marks_mesh_skinnable() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let weights = unit_weights(plane.positions.len());

        resources.set_skin_weights(&device, mesh_id, &weights);
        assert!(resources.is_skinned_mesh(mesh_id));
    }

    #[test]
    fn begin_upload_skin_weights_completes() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let weights = unit_weights(plane.positions.len());

        assert!(!resources.is_skinned_mesh(mesh_id));
        let id = resources.begin_upload_skin_weights(&device, mesh_id, weights);

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
            resources.is_skinned_mesh(mesh_id),
            "skin weights did not install"
        );
    }
}
