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
use viewport_lib::error::{ViewportError, ViewportResult};
use viewport_lib::gpu;
use viewport_lib::plugin_api::Extent;
use viewport_lib::resources::{ContentBuffer, DeviceResources};
use viewport_lib::{Colour, ColourSource, SizeSource};

/// Bytes per element of each channel, in the order the bind group takes them.
///
/// Positions are the vertex stream, tightly packed `[f32; 3]` at stride 12 to
/// match the `Float32x3` layout the pipeline declares. The rest are storage
/// buffers the shader indexes by `vertex_index`.
const POSITION_STRIDE: u32 = 12;
const SCALAR_STRIDE: u32 = 4;
const COLOUR_STRIDE: u32 = 16;
const RADIUS_STRIDE: u32 = 4;
const TRANSPARENCY_STRIDE: u32 = 4;

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

/// Build the GPU data for one point cloud: its channel buffers and its group-1
/// bind group.
///
/// Shared by the per-frame item path and the store, so a reference draw and an
/// inline draw are bit-for-bit the same work. `capacity` is the number of points
/// to allocate room for, which the inline path leaves at the point count and an
/// upload feeding a growing stream sets higher.
pub(super) fn build_point_cloud(
    device: &gpu::Device,
    queue: &gpu::Queue,
    binds: &PointCloudBindings,
    item: &PointCloudItem,
    capacity: u32,
) -> PointCloudGpuData {
    let count = item.positions.len() as u32;
    let capacity = capacity.max(count);

    let colour = resolve_colour(item);
    let (scalar_min, scalar_max) = colour.scalar_range;
    let (per_point_radii, uniform_radius) = resolve_sizes(item);

    let has_scalars = !colour.scalars.is_empty();
    let has_colours = !colour.colours.is_empty();
    let has_radius = per_point_radii.is_some();
    let has_transparency = !item.transparencies.is_empty();

    // An absent channel still needs something in its binding, so it gets a
    // minimum allocation with nothing live in it. `has_*` in the uniform is what
    // the shader reads to know not to index it.
    let mut positions = ContentBuffer::new(
        device,
        "pc_positions",
        gpu::BufferUsages::VERTEX,
        POSITION_STRIDE,
        capacity,
    );
    let mut scalars = ContentBuffer::new(
        device,
        "pc_scalars",
        gpu::BufferUsages::STORAGE,
        SCALAR_STRIDE,
        if has_scalars { capacity } else { 0 },
    );
    let mut colours = ContentBuffer::new(
        device,
        "pc_colours",
        gpu::BufferUsages::STORAGE,
        COLOUR_STRIDE,
        if has_colours { capacity } else { 0 },
    );
    let mut radii = ContentBuffer::new(
        device,
        "pc_radii",
        gpu::BufferUsages::STORAGE,
        RADIUS_STRIDE,
        if has_radius { capacity } else { 0 },
    );
    let mut transparencies = ContentBuffer::new(
        device,
        "pc_transparencies",
        gpu::BufferUsages::STORAGE,
        TRANSPARENCY_STRIDE,
        if has_transparency { capacity } else { 0 },
    );

    // Each write is at element 0 and sets the live count, so a cloud reserved
    // larger than its contents draws its contents and no more.
    let _ = positions.write_range(queue, 0, bytemuck::cast_slice(&item.positions));
    if has_scalars {
        let _ = scalars.write_range(queue, 0, bytemuck::cast_slice(&colour.scalars));
    }
    if has_colours {
        let _ = colours.write_range(queue, 0, bytemuck::cast_slice(&colour.colours));
    }
    if let Some(ref list) = per_point_radii {
        let _ = radii.write_range(queue, 0, bytemuck::cast_slice(list));
    }
    if has_transparency {
        let _ = transparencies.write_range(queue, 0, bytemuck::cast_slice(&item.transparencies));
    }

    let uniform = PointCloudUniform {
        model: item.model,
        default_colour: colour.flat,
        point_size: uniform_radius,
        has_scalars: has_scalars as u32,
        scalar_min,
        scalar_max,
        has_colours: has_colours as u32,
        has_radius: has_radius as u32,
        has_transparency: has_transparency as u32,
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
    queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform));

    let bind_group = build_bind_group(
        device,
        binds,
        &uniform_buf,
        &scalars,
        &colours,
        &radii,
        &transparencies,
    );

    PointCloudGpuData {
        positions,
        scalars,
        colours,
        radii,
        transparencies,
        uniform_buf,
        uniform,
        colourmap: crate::sources::requested_colourmap(&item.colour),
        scalar_domain_fixed: scalar_domain_is_fixed(&item.colour),
        binds: binds.clone(),
        bind_group,
        pick_id: item.settings.pick_id,
    }
}

/// Whether the colourmap domain came from the item rather than from its values.
///
/// A derived domain spans the whole array, so a write that sees part of it
/// cannot maintain it. This is recorded at build time because the item is gone
/// by the time a write arrives.
fn scalar_domain_is_fixed(colour: &ColourSource) -> bool {
    match colour {
        ColourSource::Scalar { range, .. } => range.is_some(),
        // No colourmap domain to maintain, so nothing to refuse.
        _ => true,
    }
}

fn build_bind_group(
    device: &gpu::Device,
    binds: &PointCloudBindings,
    uniform_buf: &gpu::Buffer,
    scalars: &ContentBuffer,
    colours: &ContentBuffer,
    radii: &ContentBuffer,
    transparencies: &ContentBuffer,
) -> gpu::BindGroup {
    device.create_bind_group(&gpu::BindGroupDescriptor {
        label: Some("pc_bind_group"),
        layout: &binds.bgl,
        entries: &[
            gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            gpu::BindGroupEntry {
                binding: 1,
                resource: gpu::BindingResource::TextureView(&binds.lut_view),
            },
            gpu::BindGroupEntry {
                binding: 2,
                resource: gpu::BindingResource::Sampler(&binds.lut_sampler),
            },
            gpu::BindGroupEntry {
                binding: 3,
                resource: scalars.buffer().as_entire_binding(),
            },
            gpu::BindGroupEntry {
                binding: 4,
                resource: colours.buffer().as_entire_binding(),
            },
            gpu::BindGroupEntry {
                binding: 5,
                resource: radii.buffer().as_entire_binding(),
            },
            gpu::BindGroupEntry {
                binding: 6,
                resource: transparencies.buffer().as_entire_binding(),
            },
        ],
    })
}

/// Rewrite a stored cloud's channels in place, keeping its bind group.
///
/// Returns `false` when the new item does not fit what is already allocated, in
/// which case the caller rebuilds from scratch. Two conditions, both about the
/// bindings rather than the contents: the new points must fit the reserved
/// capacity, and the bind group must still be correct, which means the same
/// colourmap behind the LUT view and the same set of present channels. A channel
/// appearing or disappearing swaps a real buffer for a minimum-size one, which
/// is a different binding however the counts compare.
///
/// The point count itself may change, up to the reserved capacity: the live
/// count moves with it and the draw follows. That is what makes a cloud uploaded
/// with headroom absorb a growing feed without reallocating.
pub(super) fn try_replace_in_place(
    queue: &gpu::Queue,
    gpu: &mut PointCloudGpuData,
    item: &PointCloudItem,
) -> bool {
    let count = item.positions.len() as u32;
    if count > gpu.positions.capacity() {
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

    // Lower the live counts first so a shorter cloud does not keep drawing the
    // tail of the longer one it replaces; the writes below raise them again.
    if gpu.set_live_len(count).is_err() {
        return false;
    }
    let ok = gpu
        .positions
        .write_range(queue, 0, bytemuck::cast_slice(&item.positions))
        .is_ok()
        && (has_scalars == 0
            || gpu
                .scalars
                .write_range(queue, 0, bytemuck::cast_slice(&colour.scalars))
                .is_ok())
        && (has_colours == 0
            || gpu
                .colours
                .write_range(queue, 0, bytemuck::cast_slice(&colour.colours))
                .is_ok())
        && match per_point_radii {
            Some(ref list) => gpu
                .radii
                .write_range(queue, 0, bytemuck::cast_slice(list))
                .is_ok(),
            None => true,
        }
        && (has_transparency == 0
            || gpu
                .transparencies
                .write_range(queue, 0, bytemuck::cast_slice(&item.transparencies))
                .is_ok());
    if !ok {
        return false;
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
    gpu.scalar_domain_fixed = scalar_domain_is_fixed(&item.colour);
    queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::bytes_of(&gpu.uniform));
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
    /// What the cloud occupies, reserved capacity included: headroom holds VRAM
    /// whether or not anything draws from it.
    fn gpu_bytes(&self) -> u64 {
        self.uniform_buf.size()
            + self.positions.allocated_bytes()
            + self.scalars.allocated_bytes()
            + self.colours.allocated_bytes()
            + self.radii.allocated_bytes()
            + self.transparencies.allocated_bytes()
    }
}

/// Which of a point cloud's channels a call means.
///
/// Every channel is indexed by point, so they are reserved and sized together
/// and only the writes differ.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum PointChannel {
    Positions,
    Scalars,
    Colours,
    Sizes,
    Transparencies,
}

/// GPU data for one point cloud: the channel buffers, the uniform block and the
/// group-1 bind group.
///
/// The inline items build one each frame and keep only the draw; the store holds
/// one across frames and writes into it.
pub(crate) struct PointCloudGpuData {
    positions: ContentBuffer,
    scalars: ContentBuffer,
    colours: ContentBuffer,
    radii: ContentBuffer,
    transparencies: ContentBuffer,
    uniform_buf: gpu::Buffer,
    /// The uniform block as written. Kept so a replace can compare this cloud's
    /// channel presence against a new item's, and so a write can rewrite one
    /// field without rebuilding the block.
    uniform: PointCloudUniform,
    /// The colourmap the bind group's LUT view came from, or `None` for the
    /// default. A replace naming a different one has to rebuild the group.
    colourmap: Option<viewport_lib::resources::ColourmapId>,
    /// Whether the colourmap domain was supplied rather than derived from the
    /// values. A ranged scalar write is refused when it was derived.
    scalar_domain_fixed: bool,
    /// The LUT view, sampler and layout, kept so a grow can rebuild the bind
    /// group without a `DeviceResources` borrow.
    binds: PointCloudBindings,
    bind_group: gpu::BindGroup,
    pick_id: viewport_lib::PickId,
}

/// What a draw needs out of a point cloud, cheap to clone into a frame list.
///
/// Cloning the whole cloud would mean two owners of one growable buffer, each
/// with its own idea of how much is allocated. The buffers here are wgpu's own
/// reference-counted handles, so this keeps them alive without owning them.
#[derive(Clone)]
pub(crate) struct PointCloudDraw {
    pub(crate) vertex_buffer: gpu::Buffer,
    pub(crate) point_count: u32,
    pub(crate) pick_id: viewport_lib::PickId,
    pub(crate) bind_group: gpu::BindGroup,
}

impl PointCloudGpuData {
    /// This cloud's draw data. The bind group keeps every storage channel alive,
    /// and the cloned vertex buffer keeps the positions.
    pub(crate) fn draw(&self) -> PointCloudDraw {
        PointCloudDraw {
            vertex_buffer: self.positions.buffer().clone(),
            point_count: self.positions.len(),
            pick_id: self.pick_id,
            bind_group: self.bind_group.clone(),
        }
    }

    /// Overwrite the model matrix at offset 0 of the uniform block.
    ///
    /// A reference item re-places a stored cloud without touching its points, so
    /// this is the one field a draw rewrites per frame.
    pub(crate) fn write_model(&self, queue: &gpu::Queue, model: &[[f32; 4]; 4]) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(model));
    }

    fn channel(&self, which: PointChannel) -> &ContentBuffer {
        match which {
            PointChannel::Positions => &self.positions,
            PointChannel::Scalars => &self.scalars,
            PointChannel::Colours => &self.colours,
            PointChannel::Sizes => &self.radii,
            PointChannel::Transparencies => &self.transparencies,
        }
    }

    fn channel_mut(&mut self, which: PointChannel) -> &mut ContentBuffer {
        match which {
            PointChannel::Positions => &mut self.positions,
            PointChannel::Scalars => &mut self.scalars,
            PointChannel::Colours => &mut self.colours,
            PointChannel::Sizes => &mut self.radii,
            PointChannel::Transparencies => &mut self.transparencies,
        }
    }

    /// Whether the channel has a real buffer behind it. Positions always do; the
    /// rest are there only when the uploaded item populated them.
    fn has_channel(&self, which: PointChannel) -> bool {
        match which {
            PointChannel::Positions => true,
            PointChannel::Scalars => self.uniform.has_scalars == 1,
            PointChannel::Colours => self.uniform.has_colours == 1,
            PointChannel::Sizes => self.uniform.has_radius == 1,
            PointChannel::Transparencies => self.uniform.has_transparency == 1,
        }
    }

    /// Refuse a write this cloud cannot serve: an absent channel, or a scalar
    /// channel whose colourmap domain is derived from the values.
    fn check_writable(&self, which: PointChannel, name: &'static str) -> ViewportResult<()> {
        if !self.has_channel(which) {
            return Err(ViewportError::ChannelNotPresent {
                type_name: super::TYPE_NAME,
                channel: name,
            });
        }
        if which == PointChannel::Scalars && !self.scalar_domain_fixed {
            return Err(ViewportError::ChannelDomainNotFixed {
                type_name: super::TYPE_NAME,
                channel: name,
            });
        }
        Ok(())
    }

    /// Write bytes into one channel at an element offset.
    pub(crate) fn write_channel(
        &mut self,
        queue: &gpu::Queue,
        which: PointChannel,
        name: &'static str,
        first_element: u32,
        data: &[u8],
    ) -> ViewportResult<()> {
        self.check_writable(which, name)?;
        self.channel_mut(which)
            .write_range(queue, first_element, data)
    }

    /// Grow every channel this cloud holds to at least `capacity` points, and
    /// rebuild the bind group if any allocation moved.
    ///
    /// All channels together because they share an element index: growing the
    /// positions alone would leave a write to the colours addressing a buffer
    /// that cannot hold it.
    pub(crate) fn reserve(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        capacity: u32,
    ) -> ViewportResult<()> {
        let mut moved = self.positions.reserve(device, queue, capacity);
        for which in [
            PointChannel::Scalars,
            PointChannel::Colours,
            PointChannel::Sizes,
            PointChannel::Transparencies,
        ] {
            if self.has_channel(which) {
                moved |= self.channel_mut(which).reserve(device, queue, capacity);
            }
        }
        if moved {
            self.bind_group = build_bind_group(
                device,
                &self.binds,
                &self.uniform_buf,
                &self.scalars,
                &self.colours,
                &self.radii,
                &self.transparencies,
            );
        }
        Ok(())
    }

    /// Set how many points draw, across every channel this cloud holds.
    pub(crate) fn set_live_len(&mut self, len: u32) -> ViewportResult<()> {
        self.positions.set_len(len)?;
        for which in [
            PointChannel::Scalars,
            PointChannel::Colours,
            PointChannel::Sizes,
            PointChannel::Transparencies,
        ] {
            if self.has_channel(which) {
                self.channel_mut(which).set_len(len)?;
            }
        }
        Ok(())
    }

    /// What one channel holds and how much of it draws.
    pub(crate) fn extent(&self, which: PointChannel) -> Extent {
        let cb = self.channel(which);
        Extent {
            capacity: cb.capacity(),
            len: cb.len(),
        }
    }
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
        build_point_cloud(device, queue, &binds, item, 0)
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

    /// A shorter cloud fits where it used to be refused, because the capacity is
    /// what the bindings depend on and not the count. The live count follows the
    /// new item, so the tail of the old cloud does not keep drawing.
    #[test]
    fn a_shorter_cloud_fits_and_the_draw_count_follows() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut gpu = built(&device, &queue, &resources, &cloud(8));
        assert!(try_replace_in_place(&queue, &mut gpu, &cloud(5)));
        assert_eq!(gpu.extent(PointChannel::Positions).len, 5);
        assert_eq!(gpu.extent(PointChannel::Positions).capacity, 8);
    }

    /// The headroom case: uploaded with capacity, grown into without
    /// reallocating.
    #[test]
    fn a_longer_cloud_fits_inside_reserved_capacity() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let bgl = build_bgl(&device);
        let item = cloud(8);
        let binds = resolve_bindings(&resources, &bgl, &item);
        let mut gpu = build_point_cloud(&device, &queue, &binds, &item, 64);
        assert_eq!(gpu.extent(PointChannel::Positions).capacity, 64);
        assert_eq!(gpu.extent(PointChannel::Positions).len, 8);

        assert!(try_replace_in_place(&queue, &mut gpu, &cloud(40)));
        assert_eq!(gpu.extent(PointChannel::Positions).len, 40);

        // Past the reserve it does not fit, rather than growing behind a call
        // the caller believes is cheap.
        assert!(!try_replace_in_place(&queue, &mut gpu, &cloud(65)));
    }

    /// A channel the upload did not populate has no buffer to write into, and
    /// saying so is better than reallocating under a streaming loop.
    #[test]
    fn writing_an_absent_channel_is_refused() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut solid = cloud(8);
        solid.colour = ColourSource::Solid(Colour::WHITE);
        let mut gpu = built(&device, &queue, &resources, &solid);

        let err = gpu
            .write_channel(
                &queue,
                PointChannel::Colours,
                "colours",
                0,
                bytemuck::cast_slice(&[[1.0f32, 0.0, 0.0, 1.0]]),
            )
            .expect_err("a cloud with one flat colour holds no colour channel");
        assert!(
            matches!(err, ViewportError::ChannelNotPresent { .. }),
            "{err}"
        );

        // Positions are always there, so the same cloud takes a position write.
        assert!(
            gpu.write_channel(
                &queue,
                PointChannel::Positions,
                "positions",
                0,
                bytemuck::cast_slice(&[[1.0f32, 2.0, 3.0]]),
            )
            .is_ok()
        );
    }

    /// A derived colourmap domain cannot survive a write that sees part of the
    /// values, so the write is refused rather than quietly mis-colouring.
    #[test]
    fn writing_scalars_under_a_derived_domain_is_refused() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let mut derived = cloud(8);
        derived.colour = ColourSource::Scalar {
            values: vec![0.5; 8],
            range: None,
            colourmap: None,
        };
        let mut gpu = built(&device, &queue, &resources, &derived);
        let err = gpu
            .write_channel(
                &queue,
                PointChannel::Scalars,
                "scalars",
                0,
                bytemuck::cast_slice(&[0.25f32]),
            )
            .expect_err("a derived domain cannot be maintained by a ranged write");
        assert!(
            matches!(err, ViewportError::ChannelDomainNotFixed { .. }),
            "{err}"
        );

        let mut fixed = cloud(8);
        fixed.colour = ColourSource::Scalar {
            values: vec![0.5; 8],
            range: Some((0.0, 1.0)),
            colourmap: None,
        };
        let mut gpu = built(&device, &queue, &resources, &fixed);
        assert!(
            gpu.write_channel(
                &queue,
                PointChannel::Scalars,
                "scalars",
                0,
                bytemuck::cast_slice(&[0.25f32]),
            )
            .is_ok(),
            "a supplied range is what makes a ranged scalar write well defined"
        );
    }

    /// A window past the reserve is an error, not a grow.
    #[test]
    fn writing_past_the_capacity_is_refused() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut gpu = built(&device, &queue, &resources, &cloud(8));
        let err = gpu
            .write_channel(
                &queue,
                PointChannel::Positions,
                "positions",
                6,
                bytemuck::cast_slice(&[[0.0f32; 3]; 4]),
            )
            .expect_err("[6..10) does not fit 8 points");
        assert!(
            matches!(err, ViewportError::ContentBufferWriteOutOfRange { .. }),
            "{err}"
        );
    }

    /// A write past the live count raises it, which is what makes an appending
    /// feed reserve once and then only write.
    #[test]
    fn a_write_into_reserved_headroom_raises_the_draw_count() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let bgl = build_bgl(&device);
        let item = cloud(4);
        let binds = resolve_bindings(&resources, &bgl, &item);
        let mut gpu = build_point_cloud(&device, &queue, &binds, &item, 16);
        assert_eq!(gpu.extent(PointChannel::Positions).len, 4);

        gpu.write_channel(
            &queue,
            PointChannel::Positions,
            "positions",
            4,
            bytemuck::cast_slice(&[[9.0f32, 0.0, 0.0]; 3]),
        )
        .expect("the window fits the reserve");
        assert_eq!(gpu.extent(PointChannel::Positions).len, 7);
    }

    /// Reserving grows every channel the cloud holds, not just the one named,
    /// because they share an element index.
    #[test]
    fn reserving_grows_every_channel_the_cloud_holds() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut item = cloud(8);
        item.colour = ColourSource::Scalar {
            values: vec![0.5; 8],
            range: Some((0.0, 1.0)),
            colourmap: None,
        };
        item.transparencies = vec![0.5; 8];
        let mut gpu = built(&device, &queue, &resources, &item);

        gpu.reserve(&device, &queue, 100).expect("reserve");
        for which in [
            PointChannel::Positions,
            PointChannel::Scalars,
            PointChannel::Transparencies,
        ] {
            assert!(
                gpu.extent(which).capacity >= 100,
                "{which:?} did not grow with the rest"
            );
            assert_eq!(gpu.extent(which).len, 8, "{which:?} live count moved");
        }
        // Absent channels stay at their minimum: growing them would hold VRAM
        // for a binding the shader never indexes.
        assert_eq!(gpu.extent(PointChannel::Colours).capacity, 1);
    }

    /// The bind group has to follow a grow, or the draw reads the buffer the
    /// reserve replaced.
    #[test]
    fn a_grow_rebuilds_the_bind_group() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut item = cloud(8);
        item.transparencies = vec![0.5; 8];
        let mut gpu = built(&device, &queue, &resources, &item);

        let before = gpu.draw();
        gpu.reserve(&device, &queue, 64).expect("reserve");
        let after = gpu.draw();
        assert!(
            after.vertex_buffer.size() > before.vertex_buffer.size(),
            "a grow past the capacity must replace the positions allocation"
        );
        assert_eq!(
            after.point_count, before.point_count,
            "a grow adds headroom; it does not change what draws"
        );

        // Reserving inside what is already held changes nothing, so the draw
        // data is untouched and no bind group was rebuilt.
        let before = gpu.draw();
        gpu.reserve(&device, &queue, 8).expect("reserve");
        assert_eq!(before.vertex_buffer.size(), gpu.draw().vertex_buffer.size());
    }

    /// Reserved headroom holds VRAM whether or not it draws, so the store's
    /// charge has to count it.
    #[test]
    fn reserved_capacity_is_charged_as_resident() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, gpu::TextureFormat::Rgba8UnormSrgb, 1);
        use viewport_lib::resources::handle::GpuByteSize;
        let bgl = build_bgl(&device);
        let item = cloud(8);
        let binds = resolve_bindings(&resources, &bgl, &item);
        let tight = build_point_cloud(&device, &queue, &binds, &item, 0);
        let roomy = build_point_cloud(&device, &queue, &binds, &item, 1024);
        assert!(
            roomy.gpu_bytes() > tight.gpu_bytes(),
            "headroom must be visible to a consumer budgeting against a ceiling"
        );
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
