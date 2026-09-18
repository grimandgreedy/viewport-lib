//! The tensor glyph sets this item type holds on the consumer's behalf, and
//! the per-frame GPU data every tensor glyph draw is built from.
//!
//! A `TensorGlyphItem` carries its tensors and is rebuilt each frame; a
//! `TensorGlyphSetRefItem` names a set uploaded once through the
//! `*_tensor_glyph_set` methods on
//! [`ViewportRenderer`](crate::renderer::ViewportRenderer). Both end up as the
//! same [`TensorGlyphGpuData`], which is why the builder is shared.
//!
//! The two bind group layouts live here rather than with the pipelines,
//! because an upload builds its bind groups against them and an upload can
//! arrive long before the first frame that draws one. The sphere base mesh does
//! not: it is shared with the glyph item type and stays with the renderer.

use crate::resources::DeviceResources;

pub(crate) use super::types::TensorGlyphSetId;

/// The tensor glyph bind group layouts. Uploads build their bind groups against
/// them, so they live here with the store rather than with the item type's
/// pipelines, and are created up front: they are layouts, not compiled
/// pipelines.
pub(super) struct TensorGlyphResources {
    /// Bind group layout for tensor glyph uniforms (group 1).
    pub(super) bgl: crate::gpu::BindGroupLayout,
    /// Bind group layout for tensor glyph instance storage (group 2).
    pub(super) instance_bgl: crate::gpu::BindGroupLayout,
}

impl TensorGlyphResources {
    pub(super) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = crate::resources::builders::uniform_texture_sampler_bgl(
            device,
            "tensor_glyph_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            crate::gpu::ShaderStages::VERTEX,
        );
        let instance_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("tensor_glyph_instance_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        Self { bgl, instance_bgl }
    }
}

/// The renderer-owned handles one tensor glyph upload binds, resolved from
/// `DeviceResources` before the buffers are built.
#[derive(Clone)]
pub(super) struct TensorGlyphBindings {
    lut_view: crate::gpu::TextureView,
    lut_sampler: crate::gpu::Sampler,
    bgl: crate::gpu::BindGroupLayout,
    instance_bgl: crate::gpu::BindGroupLayout,
    mesh: SphereMesh,
}

/// Clones of the shared sphere base mesh's buffers, so the build never needs a
/// borrow of the mesh cache.
#[derive(Clone)]
pub(super) struct SphereMesh {
    vertex_buffer: crate::gpu::Buffer,
    index_buffer: crate::gpu::Buffer,
    index_count: u32,
    edge_index_buffer: crate::gpu::Buffer,
    edge_index_count: u32,
}

/// Resolve the colourmap LUT, the shared sampler and the sphere base mesh.
///
/// An item that names no colourmap gets Viridis, the same default the draw has
/// always used; a stale id falls back to the neutral LUT.
pub(super) fn resolve_bindings(
    device: &crate::gpu::Device,
    resources: &DeviceResources,
    layouts: &TensorGlyphResources,
    item: &crate::renderer::TensorGlyphItem,
) -> TensorGlyphBindings {
    let lut_view = item
        .colourmap_id
        .and_then(|id| resources.colourmap_view(id))
        .unwrap_or_else(|| {
            resources.builtin_colourmap_view(crate::resources::BuiltinColourmap::Viridis)
        });
    let mesh = resources.ensure_glyph_base_mesh(device, crate::renderer::GlyphType::Sphere);
    TensorGlyphBindings {
        lut_view: lut_view.clone(),
        lut_sampler: resources.material_sampler().clone(),
        bgl: layouts.bgl.clone(),
        instance_bgl: layouts.instance_bgl.clone(),
        mesh: SphereMesh {
            vertex_buffer: mesh.vertex_buffer.clone(),
            index_buffer: mesh.index_buffer.clone(),
            index_count: mesh.index_count,
            edge_index_buffer: mesh.edge_index_buffer.clone(),
            edge_index_count: mesh.edge_index_count,
        },
    }
}

/// Build the GPU data for one tensor glyph set: its buffers and its two bind
/// groups.
///
/// Shared by the per-frame item path and the store, so a reference draw and an
/// inline draw are bit-for-bit the same work.
pub(super) fn build_tensor_glyph_set(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binds: &TensorGlyphBindings,
    item: &crate::renderer::TensorGlyphItem,
    wireframe: bool,
) -> TensorGlyphGpuData {
    {
        let instance_count = item.positions.len() as u32;

        // The shared sphere base mesh, cloned when the bindings were resolved.
        let (mesh_vbuf, mesh_ibuf, mesh_idx_count, mesh_edge_ibuf, mesh_edge_count) = (
            binds.mesh.vertex_buffer.clone(),
            binds.mesh.index_buffer.clone(),
            binds.mesh.index_count,
            binds.mesh.edge_index_buffer.clone(),
            binds.mesh.edge_index_count,
        );

        // Pre-compute per-instance model and normal matrices on the CPU.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TensorInstance {
            model_col0: [f32; 4],
            model_col1: [f32; 4],
            model_col2: [f32; 4],
            model_col3: [f32; 4],
            normal_col0: [f32; 4],
            normal_col1: [f32; 4],
            normal_col2: [f32; 4],
            scalar: f32,
            _pad: [f32; 3],
        }

        // `item.model` is uploaded into `TensorGlyphUniform.model`; the
        // shader composes it on top of the per-instance ellipsoid model so
        // pre-uploaded sets can be moved per frame without rebuilding the
        // instance buffer.

        // Determine scalars for LUT lookup.
        let has_scalars = item.colour_attribute.is_some();
        let (scalar_min, scalar_max) = if let Some(ref scalars) = item.colour_attribute {
            item.scalar_range.unwrap_or_else(|| {
                let mn = scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (mn, mx)
            })
        } else {
            // Sign colouring: map [-1, 1] so LUT midpoint = neutral.
            item.scalar_range.unwrap_or((-1.0, 1.0))
        };

        let instances: Vec<TensorInstance> = (0..item.positions.len())
            .map(|i| {
                let pos = glam::Vec3::from(item.positions[i]);
                let ev = if i < item.eigenvalues.len() {
                    item.eigenvalues[i]
                } else {
                    [1.0, 1.0, 1.0]
                };
                let vecs = if i < item.eigenvectors.len() {
                    item.eigenvectors[i]
                } else {
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                };

                // Scale by |eigenvalue| * global_scale, minimum 1e-6 to avoid degenerate.
                let s0 = (ev[0].abs() * item.scale).max(1e-6_f32);
                let s1 = (ev[1].abs() * item.scale).max(1e-6_f32);
                let s2 = (ev[2].abs() * item.scale).max(1e-6_f32);

                // Rotation matrix: columns are the eigenvectors.
                let col0 = glam::Vec3::from(vecs[0]);
                let col1 = glam::Vec3::from(vecs[1]);
                let col2 = glam::Vec3::from(vecs[2]);

                // Rotation-scale block: RS = R * diag(s0, s1, s2).
                let rs = glam::Mat3::from_cols(col0 * s0, col1 * s1, col2 * s2);

                // 4x4 model matrix.
                let local_model = glam::Mat4::from_mat3(rs) * glam::Mat4::IDENTITY;
                let mut world_model = local_model;
                world_model.w_axis = glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);

                // Normal matrix: R * diag(1/s0, 1/s1, 1/s2).
                let nm = glam::Mat3::from_cols(col0 / s0, col1 / s1, col2 / s2);

                // Scalar for LUT.
                let scalar = if has_scalars {
                    item.colour_attribute
                        .as_ref()
                        .and_then(|sc| sc.get(i))
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    // Sign of dominant eigenvalue.
                    if i < item.eigenvalues.len() {
                        item.eigenvalues[i][0]
                    } else {
                        0.0
                    }
                };

                let mc = world_model.to_cols_array_2d();
                TensorInstance {
                    model_col0: mc[0],
                    model_col1: mc[1],
                    model_col2: mc[2],
                    model_col3: mc[3],
                    normal_col0: [nm.x_axis.x, nm.x_axis.y, nm.x_axis.z, 0.0],
                    normal_col1: [nm.y_axis.x, nm.y_axis.y, nm.y_axis.z, 0.0],
                    normal_col2: [nm.z_axis.x, nm.z_axis.y, nm.z_axis.z, 0.0],
                    scalar,
                    _pad: [0.0; 3],
                }
            })
            .collect();

        let instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tensor_glyph_instance_buf"),
            size: (std::mem::size_of::<TensorInstance>() * instances.len()).max(128) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TensorGlyphUniform {
            model: [[f32; 4]; 4],
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
            _pad1b: f32,
            _pad1c: f32,
            _pad2: [[f32; 4]; 2],
        }
        let uniform_data = TensorGlyphUniform {
            model: item.model,
            has_scalars: if has_scalars { 1 } else { 0 },
            scalar_min,
            scalar_max,
            unlit: item.settings.unlit as u32,
            opacity: item.settings.opacity,
            wireframe: if wireframe { 1 } else { 0 },
            _pad1b: 0.0,
            _pad1c: 0.0,
            _pad2: [[0.0; 4]; 2],
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tensor_glyph_uniform_buf"),
            size: std::mem::size_of::<TensorGlyphUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = &binds.lut_view;
        let lut_sampler = &binds.lut_sampler;

        let uniform_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tensor_glyph_uniform_bg"),
            layout: &binds.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        let instance_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tensor_glyph_instance_bg"),
            layout: &binds.instance_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        TensorGlyphGpuData {
            mesh_vertex_buffer: mesh_vbuf,
            mesh_index_buffer: mesh_ibuf,
            mesh_index_count: mesh_idx_count,
            mesh_edge_index_buffer: mesh_edge_ibuf,
            mesh_edge_index_count: mesh_edge_count,
            instance_count,
            pick_id: item.settings.pick_id,
            wireframe,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }
}

// ---------------------------------------------------------------------------
// The store, and the uploads that fill it
// ---------------------------------------------------------------------------

/// Slotted store of pre-uploaded tensor glyph sets.
pub(super) type TensorGlyphSetStore =
    crate::resources::handle::SlotStore<TensorGlyphGpuData, TensorGlyphSetId>;

impl crate::resources::handle::GpuByteSize for TensorGlyphGpuData {
    fn gpu_bytes(&self) -> u64 {
        self._uniform_buf.size() + self._instance_buf.size()
    }
}

/// GPU data for one tensor glyph draw: the inline items build it each frame,
/// the store holds it across frames.
///
/// The sphere base mesh comes from the shared glyph mesh cache; the buffer
/// handles here are clones of it.
#[derive(Clone)]
pub(crate) struct TensorGlyphGpuData {
    /// Vertex buffer for the sphere base mesh.
    pub(crate) mesh_vertex_buffer: crate::gpu::Buffer,
    /// Triangle index buffer for the sphere base mesh.
    pub(crate) mesh_index_buffer: crate::gpu::Buffer,
    /// Number of triangle mesh indices.
    pub(crate) mesh_index_count: u32,
    /// Edge index buffer for wireframe LineList rendering.
    pub(crate) mesh_edge_index_buffer: crate::gpu::Buffer,
    /// Number of edge indices.
    pub(crate) mesh_edge_index_count: u32,
    /// Number of tensor glyph instances.
    pub(crate) instance_count: u32,
    /// Object-level pick id shared by every instance in the set (from the item's
    /// `settings.pick_id`); `PickId::NONE` when the set is not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Whether this batch should be drawn with the wireframe pipeline.
    pub(crate) wireframe: bool,
    /// Bind group (group 1): uniform + LUT texture + sampler.
    pub(crate) uniform_bind_group: crate::gpu::BindGroup,
    /// Bind group (group 2): per-instance storage buffer.
    pub(crate) instance_bind_group: crate::gpu::BindGroup,
    // Keep buffers alive.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) _instance_buf: crate::gpu::Buffer,
}
