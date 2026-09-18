//! The glyph sets this item type holds on the consumer's behalf, and the
//! per-frame GPU data every glyph draw is built from.
//!
//! A `GlyphItem` carries its samples and is rebuilt each frame; a
//! `GlyphSetRefItem` names a set uploaded once through the `*_glyph_set`
//! methods on [`ViewportRenderer`](crate::renderer::ViewportRenderer). Both end
//! up as the same [`GlyphGpuData`], which is why the builder is shared.
//!
//! The polyline vector decoration builds glyph data through here too: a
//! polyline carrying `node_vectors` or `edge_vectors` draws arrows, and those
//! are glyphs. That is a dependency from the polyline item type onto this one,
//! which is what the decoration actually is.
//!
//! The two bind group layouts live here rather than with the pipelines. The
//! base meshes do not: arrow, sphere and cube are shared with the tensor glyph
//! item type and stay with the renderer, reached through
//! `ensure_glyph_base_mesh`.

use crate::resources::DeviceResources;

pub(crate) use super::types::GlyphSetId;

/// The glyph bind group layouts every upload builds its bind groups against.
pub(crate) struct GlyphLayouts {
    /// Bind group layout for glyph uniforms (group 1).
    pub(crate) bgl: crate::gpu::BindGroupLayout,
    /// Bind group layout for glyph instance storage (group 2).
    pub(crate) instance_bgl: crate::gpu::BindGroupLayout,
}

impl GlyphLayouts {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = crate::resources::builders::uniform_texture_sampler_bgl(
            device,
            "glyph_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            crate::gpu::ShaderStages::VERTEX,
        );
        let instance_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("glyph_instance_bgl"),
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

/// The renderer-owned handles one glyph upload binds, resolved from
/// `DeviceResources` before the buffers are built.
#[derive(Clone)]
pub(crate) struct GlyphBindings {
    lut_view: crate::gpu::TextureView,
    lut_sampler: crate::gpu::Sampler,
    bgl: crate::gpu::BindGroupLayout,
    instance_bgl: crate::gpu::BindGroupLayout,
    mesh: BaseMesh,
}

/// Clones of the shared base mesh's buffers, so the build never needs a borrow
/// of the mesh cache.
#[derive(Clone)]
struct BaseMesh {
    vertex_buffer: crate::gpu::Buffer,
    index_buffer: crate::gpu::Buffer,
    index_count: u32,
    edge_index_buffer: crate::gpu::Buffer,
    edge_index_count: u32,
}

/// Resolve the colourmap LUT, the shared sampler and the item's base mesh.
///
/// An item that names no colourmap gets Viridis, the same default the draw has
/// always used; a stale id falls back to the neutral LUT.
pub(crate) fn resolve_bindings(
    device: &crate::gpu::Device,
    resources: &DeviceResources,
    layouts: &GlyphLayouts,
    item: &crate::renderer::GlyphItem,
) -> GlyphBindings {
    let lut_view = match item.colourmap_id {
        Some(id) => resources.colourmap_view(id),
        None => resources.builtin_colourmap_view(crate::resources::BuiltinColourmap::Viridis),
    }
    .unwrap_or_else(|| resources.fallback_colourmap_view());
    let mesh = resources.ensure_glyph_base_mesh(device, item.glyph_type);
    GlyphBindings {
        lut_view: lut_view.clone(),
        lut_sampler: resources.material_sampler().clone(),
        bgl: layouts.bgl.clone(),
        instance_bgl: layouts.instance_bgl.clone(),
        mesh: BaseMesh {
            vertex_buffer: mesh.vertex_buffer.clone(),
            index_buffer: mesh.index_buffer.clone(),
            index_count: mesh.index_count,
            edge_index_buffer: mesh.edge_index_buffer.clone(),
            edge_index_count: mesh.edge_index_count,
        },
    }
}

/// Build the GPU data for one glyph set: its buffers and its two bind groups.
///
/// Shared by the per-frame item path, the store, and the polyline vector
/// decoration, so all three are bit-for-bit the same work.
pub(crate) fn build_glyph_set(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binds: &GlyphBindings,
    item: &crate::renderer::GlyphItem,
    wireframe: bool,
) -> GlyphGpuData {
    {
        let instance_count = item.positions.len() as u32;

        // The base mesh, cloned when the bindings were resolved.
        let (mesh_vbuf, mesh_ibuf, mesh_idx_count, mesh_edge_ibuf, mesh_edge_count) = (
            binds.mesh.vertex_buffer.clone(),
            binds.mesh.index_buffer.clone(),
            binds.mesh.index_count,
            binds.mesh.edge_index_buffer.clone(),
            binds.mesh.edge_index_count,
        );

        let mags: Vec<f32> = item
            .vectors
            .iter()
            .map(|v| glam::Vec3::from(*v).length())
            .collect();

        let (scalar_min, scalar_max) = if !item.scalars.is_empty() {
            item.scalar_range.unwrap_or_else(|| {
                let min = item.scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = item
                    .scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
        } else {
            item.scalar_range.unwrap_or_else(|| {
                let min = mags.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = mags.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
        };

        let (mag_clamp_min, mag_clamp_max, has_mag_clamp) = item
            .magnitude_clamp
            .map(|(mn, mx)| (mn, mx, 1u32))
            .unwrap_or((0.0, 1.0, 0u32));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GlyphInstance {
            position: [f32; 3],
            _pad0: f32,
            direction: [f32; 3],
            scalar: f32,
        }

        let instances: Vec<GlyphInstance> = (0..item.positions.len())
            .map(|i| GlyphInstance {
                position: item.positions[i],
                _pad0: 0.0,
                direction: item.vectors.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
                scalar: item
                    .scalars
                    .get(i)
                    .copied()
                    .unwrap_or(mags.get(i).copied().unwrap_or(0.0)),
            })
            .collect();

        let instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("glyph_instance_buf"),
            size: (std::mem::size_of::<GlyphInstance>() * instances.len()).max(32) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GlyphUniform {
            model: [[f32; 4]; 4],
            global_scale: f32,
            scale_by_magnitude: u32,
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            mag_clamp_min: f32,
            mag_clamp_max: f32,
            has_mag_clamp: u32,
            default_colour: [f32; 4],
            use_default_colour: u32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
        }
        let uniform_data = GlyphUniform {
            model: item.model,
            global_scale: item.scale,
            scale_by_magnitude: if item.scale_by_magnitude { 1 } else { 0 },
            has_scalars: if !item.scalars.is_empty() { 1 } else { 0 },
            scalar_min,
            scalar_max,
            mag_clamp_min,
            mag_clamp_max,
            has_mag_clamp,
            default_colour: item.default_colour.to_linear_rgba(),
            use_default_colour: if item.default_colour.alpha() > 0.0 && item.use_default_colour {
                1
            } else {
                0
            },
            unlit: if item.settings.unlit { 1 } else { 0 },
            opacity: item.settings.opacity,
            wireframe: if wireframe { 1 } else { 0 },
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("glyph_uniform_buf"),
            size: std::mem::size_of::<GlyphUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = &binds.lut_view;
        let lut_sampler = &binds.lut_sampler;

        let uniform_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("glyph_uniform_bg"),
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
            label: Some("glyph_instance_bg"),
            layout: &binds.instance_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        GlyphGpuData {
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

/// Slotted store of pre-uploaded glyph sets.
pub(super) type GlyphSetStore = crate::resources::handle::SlotStore<GlyphGpuData, GlyphSetId>;

impl crate::resources::handle::GpuByteSize for GlyphGpuData {
    fn gpu_bytes(&self) -> u64 {
        self._uniform_buf.size() + self._instance_buf.size()
    }
}

/// GPU data for one glyph draw: the inline items and the polyline vector
/// decoration build it each frame, the store holds it across frames.
#[derive(Clone)]
pub(crate) struct GlyphGpuData {
    /// Vertex buffer for the glyph base mesh. A cloned handle on the shared
    /// mesh `DeviceResources` caches, so the draw data outlives the borrow the
    /// upload took.
    pub(crate) mesh_vertex_buffer: crate::gpu::Buffer,
    /// Triangle index buffer for the glyph base mesh.
    pub(crate) mesh_index_buffer: crate::gpu::Buffer,
    /// Number of triangle mesh indices.
    pub(crate) mesh_index_count: u32,
    /// Edge index buffer for wireframe LineList rendering.
    pub(crate) mesh_edge_index_buffer: crate::gpu::Buffer,
    /// Number of edge indices.
    pub(crate) mesh_edge_index_count: u32,
    /// Number of glyph instances.
    pub(crate) instance_count: u32,
    /// Object-level pick id shared by every instance in the set (from the item's
    /// `settings.pick_id`); `PickId::NONE` when the set is not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Whether this batch should be drawn with the wireframe pipeline.
    pub(crate) wireframe: bool,
    /// Bind group (group 1): glyph uniform + LUT texture + sampler.
    pub(crate) uniform_bind_group: crate::gpu::BindGroup,
    /// Bind group (group 2): instance storage buffer.
    pub(crate) instance_bind_group: crate::gpu::BindGroup,
    // Keep the buffers alive.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) _instance_buf: crate::gpu::Buffer,
}
