//! The point clouds this item type holds on the consumer's behalf, and the
//! per-frame GPU data every point cloud draw is built from.
//!
//! A `PointCloudItem` carries its points and is rebuilt each frame; a
//! `PointCloudRefItem` names a set uploaded once through the `*_point_cloud`
//! methods on [`ViewportRenderer`](crate::renderer::ViewportRenderer). Both end
//! up as the same [`PointCloudGpuData`], which is why the builder is shared.
//!
//! The group-1 bind group layout lives here rather than with the pipelines,
//! because an upload builds its bind group against it and an upload can arrive
//! long before the first frame that draws one.

use crate::resources::DeviceResources;

pub(crate) use super::types::PointCloudId;

/// Build the point cloud group-1 bind group layout.
pub(super) fn build_bgl(device: &crate::gpu::Device) -> crate::gpu::BindGroupLayout {
    {
        device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}

/// The renderer-owned handles one point cloud upload binds, resolved from
/// `DeviceResources` before the buffers are built.
///
/// They are separated out because they are the only thing the build needs that
/// the plugin does not own, and because wgpu views, samplers and layouts are
/// cheap clonable handles: resolving them up front is what lets the buffer
/// work run on a worker thread, where no `DeviceResources` borrow exists.
#[derive(Clone)]
pub(super) struct PointCloudBindings {
    lut_view: crate::gpu::TextureView,
    lut_sampler: crate::gpu::Sampler,
    bgl: crate::gpu::BindGroupLayout,
}

/// Resolve the colourmap LUT and sampler an item names.
///
/// An item that names no colourmap gets Viridis, the same default the draw has
/// always used; a stale id falls back to the neutral LUT.
pub(super) fn resolve_bindings(
    resources: &DeviceResources,
    bgl: &crate::gpu::BindGroupLayout,
    item: &crate::renderer::PointCloudItem,
) -> PointCloudBindings {
    let lut_view = item
        .colourmap_id
        .and_then(|id| resources.colourmap_view(id))
        .unwrap_or_else(|| {
            resources.builtin_colourmap_view(crate::resources::BuiltinColourmap::Viridis)
        });
    PointCloudBindings {
        lut_view: lut_view.clone(),
        lut_sampler: resources.material_sampler().clone(),
        bgl: bgl.clone(),
    }
}

/// Build the GPU data for one point cloud: its buffers and its group-1 bind
/// group.
///
/// Shared by the per-frame item path and the store, so a reference draw and an
/// inline draw are bit-for-bit the same work.
pub(super) fn build_point_cloud(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binds: &PointCloudBindings,
    item: &crate::renderer::PointCloudItem,
) -> PointCloudGpuData {
    {
        let point_count = item.positions.len() as u32;

        let pos_bytes: Vec<u8> = item
            .positions
            .iter()
            .flat_map(|p| bytemuck::bytes_of(p).iter().copied())
            .collect();
        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pc_vertex_buf"),
            size: pos_bytes.len().max(12) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &pos_bytes);

        let (scalar_buf, has_scalars, scalar_min, scalar_max) = if !item.scalars.is_empty() {
            let min = item
                .scalar_range
                .map(|r| r.0)
                .unwrap_or_else(|| item.scalars.iter().cloned().fold(f32::INFINITY, f32::min));
            let max = item.scalar_range.map(|r| r.1).unwrap_or_else(|| {
                item.scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            });
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_scalar_buf"),
                size: (std::mem::size_of::<f32>() * item.scalars.len()).max(4) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.scalars));
            (buf, 1u32, min, max)
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_scalar_buf_fallback"),
                size: 4,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32, 0.0f32, 1.0f32)
        };

        let (colour_buf, has_colours) = if !item.colours.is_empty() && has_scalars == 0 {
            let bytes: &[u8] = bytemuck::cast_slice(&item.colours);
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_colour_buf"),
                size: bytes.len().max(16) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytes);
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_colour_buf_fallback"),
                size: 16,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        // Radius buffer: radius_scalars (mapped to radius_range) take priority over
        // explicit per-point radii.
        let (radius_buf, has_radius) = if !item.radius_scalars.is_empty() {
            let r_min = item.radius_scalar_range.map(|r| r.0).unwrap_or_else(|| {
                item.radius_scalars
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min)
            });
            let r_max = item.radius_scalar_range.map(|r| r.1).unwrap_or_else(|| {
                item.radius_scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            });
            let range = (r_max - r_min).max(f32::EPSILON);
            let (out_min, out_max) = item.radius_range;
            let mapped: Vec<f32> = item
                .radius_scalars
                .iter()
                .map(|&s| {
                    let t = ((s - r_min) / range).clamp(0.0, 1.0);
                    out_min + t * (out_max - out_min)
                })
                .collect();
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_radius_buf"),
                size: (std::mem::size_of::<f32>() * mapped.len()).max(4) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&mapped));
            (buf, 1u32)
        } else if !item.radii.is_empty() {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_radius_buf"),
                size: (std::mem::size_of::<f32>() * item.radii.len()).max(4) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.radii));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_radius_buf_fallback"),
                size: 4,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        let (transparency_buf, has_transparency) = if !item.transparencies.is_empty() {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_transparency_buf"),
                size: (std::mem::size_of::<f32>() * item.transparencies.len()).max(4) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.transparencies));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_transparency_buf_fallback"),
                size: 4,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PointCloudUniform {
            model: [[f32; 4]; 4],
            default_colour: [f32; 4],
            point_size: f32,
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            has_colours: u32,
            has_radius: u32,
            has_transparency: u32,
            gaussian: u32,
            // 0 = ScreenSpaceCircle, 1 = Sphere
            render_mode: u32,
            _pad: [u32; 3],
        }
        let uniform_data = PointCloudUniform {
            model: item.model,
            default_colour: item.default_colour.to_linear_rgba(),
            point_size: item.point_size,
            has_scalars,
            scalar_min,
            scalar_max,
            has_colours,
            has_radius,
            has_transparency,
            gaussian: if item.gaussian { 1 } else { 0 },
            render_mode: match item.render_mode {
                crate::renderer::PointRenderMode::ScreenSpaceCircle => 0,
                crate::renderer::PointRenderMode::Sphere => 1,
            },
            _pad: [0; 3],
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pc_uniform_buf"),
            size: std::mem::size_of::<PointCloudUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = &binds.lut_view;
        let lut_sampler = &binds.lut_sampler;

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("pc_bind_group"),
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
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: scalar_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: colour_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: radius_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: transparency_buf.as_entire_binding(),
                },
            ],
        });

        PointCloudGpuData {
            vertex_buffer,
            point_count,
            pick_id: item.settings.pick_id,
            bind_group,
            _uniform_buf: uniform_buf,
            _scalar_buf: scalar_buf,
            _colour_buf: colour_buf,
            _radius_buf: radius_buf,
            _transparency_buf: transparency_buf,
        }
    }
}

// ---------------------------------------------------------------------------
// The store, and the uploads that fill it
// ---------------------------------------------------------------------------

/// Slotted store of pre-uploaded point clouds.
///
/// A removed entry leaves an empty slot that a later upload reuses. Each slot
/// carries a generation bumped on removal, so a stale [`PointCloudId`] resolves
/// to nothing rather than aliasing the cloud now in its slot.
pub(super) type PointCloudStore =
    crate::resources::handle::SlotStore<PointCloudGpuData, PointCloudId>;

impl crate::resources::handle::GpuByteSize for PointCloudGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size()
            + self._uniform_buf.size()
            + self._scalar_buf.size()
            + self._colour_buf.size()
            + self._radius_buf.size()
            + self._transparency_buf.size()
    }
}

/// GPU data for one point cloud draw: the inline items build it each frame,
/// the store holds it across frames.
#[derive(Clone)]
pub(crate) struct PointCloudGpuData {
    /// Vertex buffer: one entry per point, packed as `[position: vec3, _pad: f32]` (16 bytes).
    /// The shader reads colour/scalar from storage buffers indexed by `vertex_index`.
    pub(crate) vertex_buffer: crate::gpu::Buffer,
    /// Number of points (= draw count).
    pub(crate) point_count: u32,
    /// The item's pick id (from `settings.pick_id`); `PickId::NONE` when not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Bind group (group 1): uniform + LUT + sampler + scalar + colour + radius + transparency.
    pub(crate) bind_group: crate::gpu::BindGroup,
    // Keep the buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) _scalar_buf: crate::gpu::Buffer,
    pub(crate) _colour_buf: crate::gpu::Buffer,
    pub(crate) _radius_buf: crate::gpu::Buffer,
    pub(crate) _transparency_buf: crate::gpu::Buffer,
}
