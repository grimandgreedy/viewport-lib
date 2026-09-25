//! The point clouds this item type holds on the consumer's behalf, and the
//! per-frame GPU data every point cloud draw is built from.
//!
//! A `PointCloudItem` carries its points and is rebuilt each frame; a
//! `PointCloudRefItem` names a set uploaded once through the `*_point_cloud`
//! methods of [`PointCloudUploads`](crate::PointCloudUploads). Both end up as
//! the same [`PointCloudGpuData`], which is why the builder is shared.
//!
//! The group-1 bind group layout lives here rather than with the pipelines,
//! because an upload builds its bind group against it and an upload can arrive
//! long before the first frame that draws one.

use super::types::{PointCloudId, PointCloudItem, PointRenderMode};
use viewport_lib::gpu;
use viewport_lib::resources::DeviceResources;
use viewport_lib::{Colour, ColourSource, SizeSource};

/// Build the point cloud group-1 bind group layout.
pub(super) fn build_bgl(device: &gpu::Device) -> gpu::BindGroupLayout {
    {
        device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_bgl"),
            entries: &[
                gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: gpu::ShaderStages::VERTEX,
                    ty: gpu::BindingType::Texture {
                        sample_type: gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: gpu::ShaderStages::VERTEX,
                    ty: gpu::BindingType::Sampler(gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: gpu::ShaderStages::VERTEX,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: gpu::ShaderStages::VERTEX,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: gpu::ShaderStages::VERTEX,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: gpu::ShaderStages::VERTEX,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Storage { read_only: true },
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
    lut_view: gpu::TextureView,
    lut_sampler: gpu::Sampler,
    bgl: gpu::BindGroupLayout,
}

/// Resolve the colourmap LUT and the clamping LUT sampler an item names.
///
/// An item that names no colourmap gets Viridis, the same default the draw has
/// always used; a stale id falls back to the neutral LUT.
pub(super) fn resolve_bindings(
    resources: &DeviceResources,
    bgl: &gpu::BindGroupLayout,
    item: &PointCloudItem,
) -> PointCloudBindings {
    let lut_view = crate::sources::requested_colourmap(&item.colour)
        .and_then(|id| resources.colourmap_view(id))
        .unwrap_or_else(|| {
            resources.builtin_colourmap_view(viewport_lib::resources::BuiltinColourmap::Viridis)
        });
    PointCloudBindings {
        lut_view: lut_view.clone(),
        // The LUT sampler clamps; the material sampler repeats, which makes a
        // lookup at exactly 0 wrap onto the far end of the colourmap.
        lut_sampler: resources.lut_sampler().clone(),
        bgl: bgl.clone(),
    }
}

/// What the draw needs out of a [`ColourSource`], in the shape the uniform and
/// the two storage buffers already take.
///
/// The point cloud resolves its own rather than going through
/// `crate::sources::colour_plan`, because a cloud can hold millions of points
/// and `Solid` has to stay one uniform colour rather than becoming a buffer of
/// identical ones.
struct ResolvedColour {
    /// Per-point scalars for the colourmap path; empty otherwise.
    scalars: Vec<f32>,
    /// The scalar domain the colourmap spans.
    scalar_range: (f32, f32),
    /// Per-point RGBA; empty unless the source is per-sample.
    colours: Vec<[f32; 4]>,
    /// The one colour used when neither list is in play.
    flat: [f32; 4],
}

/// Colour for a point past the end of a short per-sample list, and for a source
/// with no natural scalar to read.
const FALLBACK_COLOUR: Colour = Colour::WHITE;

fn resolve_colour(item: &PointCloudItem) -> ResolvedColour {
    let count = item.positions.len();
    let flat = FALLBACK_COLOUR.to_linear_rgba();
    match &item.colour {
        ColourSource::Solid(c) => ResolvedColour {
            scalars: Vec::new(),
            scalar_range: (0.0, 1.0),
            colours: Vec::new(),
            flat: c.to_linear_rgba(),
        },
        ColourSource::PerSample(list) => ResolvedColour {
            scalars: Vec::new(),
            scalar_range: (0.0, 1.0),
            colours: (0..count)
                .map(|i| {
                    list.get(i)
                        .copied()
                        .unwrap_or(FALLBACK_COLOUR)
                        .to_linear_rgba()
                })
                .collect(),
            flat,
        },
        ColourSource::Scalar { values, range, .. } => ResolvedColour {
            scalar_range: crate::sources::resolved_domain(*range, values),
            scalars: (0..count)
                .map(|i| values.get(i).copied().unwrap_or(0.0))
                .collect(),
            colours: Vec::new(),
            flat,
        },
        // A point cloud has no natural scalar, so this is the flat fallback,
        // which is what `ColourSource` documents for an item without one.
        _ => ResolvedColour {
            scalars: Vec::new(),
            scalar_range: (0.0, 1.0),
            colours: Vec::new(),
            flat,
        },
    }
}

/// Each point's pixel radius, or `None` when every point takes the same one.
///
/// `SizeSource::Natural` lands here with an empty natural slice, because a point
/// cloud has none, and resolves to the bottom of its output range.
fn resolve_sizes(item: &PointCloudItem) -> (Option<Vec<f32>>, f32) {
    match &item.size {
        SizeSource::Uniform(px) => (None, *px),
        other => (
            Some(crate::sources::sample_sizes(
                other,
                item.positions.len(),
                &[],
            )),
            0.0,
        ),
    }
}

/// The point cloud's group-1 uniform block.
///
/// A stored cloud keeps its copy so an in-place replace can compare the channel
/// presence flags against a new item without rebuilding anything, and rewrite
/// the block when they match.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PointCloudUniform {
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

/// Build the GPU data for one point cloud: its buffers and its group-1 bind
/// group.
///
/// Shared by the per-frame item path and the store, so a reference draw and an
/// inline draw are bit-for-bit the same work.
pub(super) fn build_point_cloud(
    device: &gpu::Device,
    queue: &gpu::Queue,
    binds: &PointCloudBindings,
    item: &PointCloudItem,
) -> PointCloudGpuData {
    {
        let point_count = item.positions.len() as u32;

        let pos_bytes: &[u8] = bytemuck::cast_slice(&item.positions);
        let vertex_buffer = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("pc_vertex_buf"),
            size: pos_bytes.len().max(12) as u64,
            usage: gpu::BufferUsages::VERTEX | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, pos_bytes);

        let colour = resolve_colour(item);
        let (scalar_min, scalar_max) = colour.scalar_range;

        let (scalar_buf, has_scalars) = if !colour.scalars.is_empty() {
            let buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("pc_scalar_buf"),
                size: (std::mem::size_of::<f32>() * colour.scalars.len()).max(4) as u64,
                usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&colour.scalars));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("pc_scalar_buf_fallback"),
                size: 4,
                usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        let (colour_buf, has_colours) = if !colour.colours.is_empty() {
            let bytes: &[u8] = bytemuck::cast_slice(&colour.colours);
            let buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("pc_colour_buf"),
                size: bytes.len().max(16) as u64,
                usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytes);
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("pc_colour_buf_fallback"),
                size: 16,
                usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        let (per_point_radii, uniform_radius) = resolve_sizes(item);
        let (radius_buf, has_radius) = if let Some(radii) = per_point_radii {
            let buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("pc_radius_buf"),
                size: (std::mem::size_of::<f32>() * radii.len()).max(4) as u64,
                usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&radii));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("pc_radius_buf_fallback"),
                size: 4,
                usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        let (transparency_buf, has_transparency) = if !item.transparencies.is_empty() {
            let buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("pc_transparency_buf"),
                size: (std::mem::size_of::<f32>() * item.transparencies.len()).max(4) as u64,
                usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.transparencies));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("pc_transparency_buf_fallback"),
                size: 4,
                usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        let uniform_data = PointCloudUniform {
            model: item.model,
            default_colour: colour.flat,
            point_size: uniform_radius,
            has_scalars,
            scalar_min,
            scalar_max,
            has_colours,
            has_radius,
            has_transparency,
            gaussian: if item.gaussian { 1 } else { 0 },
            render_mode: match item.render_mode {
                PointRenderMode::ScreenSpaceCircle => 0,
                PointRenderMode::Sphere => 1,
            },
            _pad: [0; 3],
        };
        let uniform_buf = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("pc_uniform_buf"),
            size: std::mem::size_of::<PointCloudUniform>() as u64,
            usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = &binds.lut_view;
        let lut_sampler = &binds.lut_sampler;

        let bind_group = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("pc_bind_group"),
            layout: &binds.bgl,
            entries: &[
                gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu::BindingResource::TextureView(lut_view),
                },
                gpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu::BindingResource::Sampler(lut_sampler),
                },
                gpu::BindGroupEntry {
                    binding: 3,
                    resource: scalar_buf.as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 4,
                    resource: colour_buf.as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 5,
                    resource: radius_buf.as_entire_binding(),
                },
                gpu::BindGroupEntry {
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
            uniform: uniform_data,
            colourmap: crate::sources::requested_colourmap(&item.colour),
            _uniform_buf: uniform_buf,
            _scalar_buf: scalar_buf,
            _colour_buf: colour_buf,
            _radius_buf: radius_buf,
            _transparency_buf: transparency_buf,
        }
    }
}

/// Rewrite a stored cloud's buffers in place, keeping its bind group.
///
/// Returns `false` when the new item does not fit what is already allocated, in
/// which case the caller rebuilds from scratch. The shape has to match exactly:
/// the same point count, the same colourmap behind the LUT binding, and the
/// same set of present channels. All three matter. A different point count
/// means every buffer is the wrong size; a different colourmap means the bind
/// group holds the wrong texture view; and a channel appearing or disappearing
/// swaps a real buffer for a four-byte fallback, which is a different binding
/// even when the counts agree.
///
/// This is what makes a streaming feed cheap: a replace that keeps its shape,
/// which is the normal case when only the values changed, costs six
/// `write_buffer` calls instead of six buffer allocations plus a bind group.
pub(super) fn try_replace_in_place(
    queue: &gpu::Queue,
    gpu: &mut PointCloudGpuData,
    item: &PointCloudItem,
) -> bool {
    let count = item.positions.len() as u32;
    if count != gpu.point_count {
        return false;
    }
    if crate::sources::requested_colourmap(&item.colour) != gpu.colourmap {
        return false;
    }

    let colour = resolve_colour(item);
    let (per_point_radii, uniform_radius) = resolve_sizes(item);
    let (scalar_min, scalar_max) = colour.scalar_range;

    let has_scalars = !colour.scalars.is_empty() as u32;
    let has_colours = !colour.colours.is_empty() as u32;
    let has_radius = per_point_radii.is_some() as u32;
    let has_transparency = !item.transparencies.is_empty() as u32;
    if has_scalars != gpu.uniform.has_scalars
        || has_colours != gpu.uniform.has_colours
        || has_radius != gpu.uniform.has_radius
        || has_transparency != gpu.uniform.has_transparency
    {
        return false;
    }

    queue.write_buffer(&gpu.vertex_buffer, 0, bytemuck::cast_slice(&item.positions));
    if has_scalars == 1 {
        queue.write_buffer(&gpu._scalar_buf, 0, bytemuck::cast_slice(&colour.scalars));
    }
    if has_colours == 1 {
        queue.write_buffer(&gpu._colour_buf, 0, bytemuck::cast_slice(&colour.colours));
    }
    if let Some(radii) = per_point_radii {
        queue.write_buffer(&gpu._radius_buf, 0, bytemuck::cast_slice(&radii));
    }
    if has_transparency == 1 {
        queue.write_buffer(
            &gpu._transparency_buf,
            0,
            bytemuck::cast_slice(&item.transparencies),
        );
    }

    gpu.uniform = PointCloudUniform {
        model: item.model,
        default_colour: colour.flat,
        point_size: uniform_radius,
        has_scalars,
        scalar_min,
        scalar_max,
        has_colours,
        has_radius,
        has_transparency,
        gaussian: if item.gaussian { 1 } else { 0 },
        render_mode: match item.render_mode {
            PointRenderMode::ScreenSpaceCircle => 0,
            PointRenderMode::Sphere => 1,
        },
        _pad: [0; 3],
    };
    queue.write_buffer(&gpu._uniform_buf, 0, bytemuck::bytes_of(&gpu.uniform));
    gpu.pick_id = item.settings.pick_id;
    true
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
    viewport_lib::resources::handle::SlotStore<PointCloudGpuData, PointCloudId>;

impl viewport_lib::resources::handle::GpuByteSize for PointCloudGpuData {
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
    /// Vertex buffer: one tightly packed `[f32; 3]` per point, 12 bytes, matching
    /// the `Float32x3` vertex layout the pipeline declares. The shader reads
    /// colour and scalar from storage buffers indexed by `vertex_index`.
    pub(crate) vertex_buffer: gpu::Buffer,
    /// Number of points (= draw count).
    pub(crate) point_count: u32,
    /// The item's pick id (from `settings.pick_id`); `PickId::NONE` when not pickable.
    pub(crate) pick_id: viewport_lib::PickId,
    /// Bind group (group 1): uniform + LUT + sampler + scalar + colour + radius + transparency.
    pub(crate) bind_group: gpu::BindGroup,
    /// The uniform block as written. Kept so an in-place replace can compare
    /// this cloud's channel presence against a new item's.
    pub(crate) uniform: PointCloudUniform,
    /// The colourmap the bind group's LUT view came from, or `None` for the
    /// default. A replace naming a different one has to rebuild the group.
    pub(crate) colourmap: Option<viewport_lib::resources::ColourmapId>,
    // Keep the buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: gpu::Buffer,
    pub(crate) _scalar_buf: gpu::Buffer,
    pub(crate) _colour_buf: gpu::Buffer,
    pub(crate) _radius_buf: gpu::Buffer,
    pub(crate) _transparency_buf: gpu::Buffer,
}

#[cfg(test)]
mod in_place_tests {
    use super::*;
    use viewport_lib::{Colour, ColourSource, SizeSource};

    fn device() -> Option<(gpu::Device, gpu::Queue)> {
        viewport_lib_testkit::headless_device_with(&viewport_lib_testkit::DeviceProfile::low_power(
            "point_cloud_in_place",
        ))
    }

    fn cloud(n: usize) -> PointCloudItem {
        let mut c = PointCloudItem::default();
        c.positions = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
        c
    }

    /// Build a cloud's GPU data the way an upload would, so the tests below
    /// exercise the same entry a stored cloud holds.
    fn built(
        device: &gpu::Device,
        queue: &gpu::Queue,
        resources: &DeviceResources,
        item: &PointCloudItem,
    ) -> PointCloudGpuData {
        let bgl = build_bgl(device);
        let binds = resolve_bindings(resources, &bgl, item);
        build_point_cloud(device, queue, &binds, item)
    }

    /// The case the streaming feed hits every update: same shape, new values.
    ///
    /// That the buffers are reused rather than rebuilt is structural, not
    /// asserted: `try_replace_in_place` takes no `&Device`, so it cannot
    /// allocate one. What the test pins is that the fast path is *taken* for
    /// this shape, which is the part a future change could silently lose.
    #[test]
    fn a_same_shape_replace_takes_the_in_place_path() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut gpu = built(&device, &queue, &resources, &cloud(8));

        let mut next = cloud(8);
        next.positions[0] = [99.0, 1.0, 2.0];
        next.model = glam::Mat4::from_translation(glam::Vec3::X).to_cols_array_2d();
        assert!(try_replace_in_place(&queue, &mut gpu, &next));
        assert_eq!(
            gpu.uniform.model, next.model,
            "the uniform must carry the new item's model"
        );
    }

    #[test]
    fn a_different_point_count_does_not_fit() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut gpu = built(&device, &queue, &resources, &cloud(8));
        assert!(!try_replace_in_place(&queue, &mut gpu, &cloud(9)));
    }

    /// A channel appearing swaps a four-byte fallback buffer for a real one,
    /// which is a different binding even though the point count is unchanged.
    #[test]
    fn a_channel_appearing_does_not_fit() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let solid = cloud(8);
        let mut gpu = built(&device, &queue, &resources, &solid);

        let mut with_scalars = cloud(8);
        with_scalars.colour = ColourSource::Scalar {
            values: vec![0.5; 8],
            range: Some((0.0, 1.0)),
            colourmap: None,
        };
        assert!(!try_replace_in_place(&queue, &mut gpu, &with_scalars));
    }

    /// Solid to solid is a shape match: the flat colour rides the uniform, not
    /// a buffer, so only the uniform changes.
    #[test]
    fn a_new_solid_colour_fits() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut solid = cloud(8);
        solid.colour = ColourSource::Solid(Colour::WHITE);
        let mut gpu = built(&device, &queue, &resources, &solid);

        let mut recoloured = cloud(8);
        recoloured.colour = ColourSource::Solid(Colour::linear_rgb(1.0, 0.0, 0.0));
        assert!(try_replace_in_place(&queue, &mut gpu, &recoloured));
        assert_eq!(gpu.uniform.default_colour[0], 1.0);
    }

    /// `Uniform` sizes live in the uniform; a per-sample list is a buffer. Going
    /// from one to the other changes the bindings.
    #[test]
    fn a_uniform_to_per_sample_size_does_not_fit() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut fixed = cloud(8);
        fixed.size = SizeSource::Uniform(4.0);
        let mut gpu = built(&device, &queue, &resources, &fixed);

        let mut per_point = cloud(8);
        per_point.size = SizeSource::PerSample(vec![2.0; 8]);
        assert!(!try_replace_in_place(&queue, &mut gpu, &per_point));
    }
}
