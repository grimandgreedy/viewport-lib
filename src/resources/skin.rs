//! GPU skinning sidecar storage.
//!
//! Holds the per-mesh skin-weight storage buffers and the per-(mesh, instance)
//! joint palette storage buffers used by the skinned pipeline variants.
//!
//! Static meshes pay zero overhead: nothing is allocated until
//! [`crate::ViewportGpuResources::set_skin_weights`] is called for a mesh,
//! which is what marks that mesh as skinnable.
//!
//! See `docs/plans/skeletal-animation-plan.md` Phase 5.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::resources::SkinWeights;
use crate::resources::ViewportGpuResources;
use crate::resources::mesh_store::MeshId;

/// Packed per-vertex skin data uploaded to the GPU sidecar storage buffer.
///
/// Layout matches the `SkinVertex` WGSL struct in `mesh_skinned.wgsl`:
/// `weights` first (vec4 aligned to 16), then the two packed joint-index
/// u32s, then 8 bytes of trailing padding. WGSL rounds the struct up to its
/// alignment (16) for array stride, so the stride is 32 bytes even though
/// only 24 bytes carry data. The padding makes Rust agree.
///
/// Total: 32 bytes per vertex (24 bytes data + 8 bytes std430 padding).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PackedSkinVertex {
    pub weights: [f32; 4],
    /// `joints[0]` in the low 16 bits, `joints[1]` in the high 16 bits.
    pub joints_01: u32,
    /// `joints[2]` in the low 16 bits, `joints[3]` in the high 16 bits.
    pub joints_23: u32,
    /// Trailing padding so the struct stride matches WGSL std430.
    pub _pad: [u32; 2],
}

/// Per-instance joint palette: storage buffer plus its bind group.
pub(crate) struct InstancePalette {
    /// Storage buffer of `mat4x4<f32>` joint matrices.
    pub buffer: wgpu::Buffer,
    /// Bind group binding both the mesh's skin weights and this instance's
    /// palette buffer.
    pub bind_group: wgpu::BindGroup,
    /// Number of `Mat4` slots allocated in `buffer`. Used to decide when a
    /// realloc is required.
    pub joint_capacity: u32,
}

/// Per-mesh skinning data: the weights storage buffer plus a map of instance
/// palettes keyed by `instance_id`.
pub(crate) struct MeshSkinning {
    pub weights_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub instances: HashMap<u32, InstancePalette>,
}

/// Renderer-side skinning state.
///
/// Owns the bind group layout used by skinned pipeline variants and a per-mesh
/// map of skinning sidecars.
pub(crate) struct SkinningState {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub meshes: HashMap<MeshId, MeshSkinning>,
}

impl SkinningState {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin_bgl"),
            entries: &[
                // binding 0: skin weights storage buffer (per-vertex joint indices + weights)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: joint palette storage buffer (mat4x4<f32> per joint)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        Self {
            bind_group_layout,
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
                    _pad: [0, 0],
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl ViewportGpuResources {
    /// Always `true`. GPU skinning ships unconditionally as part of the
    /// renderer's capability set.
    ///
    /// Plugins read this to decide whether to emit
    /// [`crate::SkinnedPoseUpdate`] (GPU path) or
    /// [`crate::SkinnedMeshUpdate`] (CPU path) each frame. The flag stays in
    /// the API so a future build that intentionally drops the skinned
    /// pipelines (memory-constrained target, headless test mode) can return
    /// `false` here without changing plugin code.
    pub fn supports_gpu_skinning(&self) -> bool {
        true
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
        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skin_weights_buffer"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.skinning.meshes.insert(
            mesh_id,
            MeshSkinning {
                weights_buffer,
                vertex_count: packed.len() as u32,
                instances: HashMap::new(),
            },
        );
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
    /// `palette[i]` is `world_transform[i] * inverse_bind[i]` for joint `i`,
    /// the LBS-ready matrix produced by
    /// [`crate::JointMatrices::compute`].
    pub fn set_skin_palette(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh_id: MeshId,
        instance_id: u32,
        palette: &[glam::Mat4],
    ) -> bool {
        let bgl = &self.skinning.bind_group_layout;
        let mesh = match self.skinning.meshes.get_mut(&mesh_id) {
            Some(m) => m,
            None => return false,
        };

        let joints_needed = palette.len() as u32;
        let needs_realloc = match mesh.instances.get(&instance_id) {
            Some(inst) => inst.joint_capacity < joints_needed,
            None => true,
        };

        if needs_realloc {
            let capacity = joints_needed.max(1);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skin_palette_buffer"),
                size: (capacity as u64) * std::mem::size_of::<[[f32; 4]; 4]>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("skin_bind_group"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mesh.weights_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
            mesh.instances.insert(
                instance_id,
                InstancePalette {
                    buffer,
                    bind_group,
                    joint_capacity: capacity,
                },
            );
        }

        let inst = mesh.instances.get(&instance_id).unwrap();
        let bytes: Vec<[[f32; 4]; 4]> = palette.iter().map(|m| m.to_cols_array_2d()).collect();
        queue.write_buffer(&inst.buffer, 0, bytemuck::cast_slice(&bytes));
        true
    }

    /// Whether `mesh_id` has been marked as skinnable via
    /// [`Self::set_skin_weights`].
    pub fn is_skinned_mesh(&self, mesh_id: MeshId) -> bool {
        self.skinning.meshes.contains_key(&mesh_id)
    }

    /// Whether the given `(mesh_id, instance_id)` has a palette uploaded and
    /// is ready to be drawn through the skinned pipeline.
    pub(crate) fn skin_instance_bind_group(
        &self,
        mesh_id: MeshId,
        instance_id: u32,
    ) -> Option<&wgpu::BindGroup> {
        self.skinning
            .meshes
            .get(&mesh_id)?
            .instances
            .get(&instance_id)
            .map(|p| &p.bind_group)
    }
}
