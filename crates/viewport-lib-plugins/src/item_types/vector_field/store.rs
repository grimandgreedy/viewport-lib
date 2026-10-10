//! The vector fields this item type holds on the consumer's behalf, and the
//! per-frame GPU data every vector field draw is built from.
//!
//! A `VectorFieldItem` carries its samples and is rebuilt each frame; a
//! `VectorFieldRefItem` names a field uploaded once. Both end up as the same
//! [`VectorFieldGpuData`], which is why the builder is shared.
//!
//! The shape is a consumer-uploaded `MeshId` rather than buffers of this type's
//! own, so the GPU data names the mesh and the draw hooks bind it through
//! `MeshDraw`. The two bind group layouts live here rather than with the
//! pipelines, because an upload builds its bind groups against them and an
//! upload can arrive long before the first frame that draws one.

use crate::item_types::sources::{colour_plan, requested_colourmap};
use viewport_lib::error::ViewportResult;
use viewport_lib::plugin_api::Extent;
use viewport_lib::resources::ContentBuffer;

/// Bytes per interleaved sample record.
const INSTANCE_STRIDE: u32 = std::mem::size_of::<VectorFieldInstance>() as u32;
use viewport_lib::MeshId;
use viewport_lib::resources::DeviceResources;

pub(crate) use super::types::VectorFieldId;

/// The vector field bind group layouts. Uploads build their bind groups against
/// them, so they live here with the store rather than with the item type's
/// pipelines, and are created up front: they are layouts, not compiled
/// pipelines.
pub(super) struct VectorFieldResources {
    /// Bind group layout for the field uniform, LUT and sampler (group 1).
    pub(super) bgl: viewport_lib::gpu::BindGroupLayout,
    /// Bind group layout for the instance storage buffer (group 2).
    pub(super) instance_bgl: viewport_lib::gpu::BindGroupLayout,
}

impl VectorFieldResources {
    pub(super) fn new(device: &viewport_lib::gpu::Device) -> Self {
        let bgl = viewport_lib::plugin_api::builders::uniform_texture_sampler_bgl(
            device,
            "vector_field_bgl",
            viewport_lib::gpu::ShaderStages::VERTEX | viewport_lib::gpu::ShaderStages::FRAGMENT,
            viewport_lib::gpu::ShaderStages::VERTEX,
        );
        let instance_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("vector_field_instance_bgl"),
                entries: &[viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: viewport_lib::gpu::ShaderStages::VERTEX,
                    ty: viewport_lib::gpu::BindingType::Buffer {
                        ty: viewport_lib::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        Self { bgl, instance_bgl }
    }
}

/// The renderer-owned handles one upload binds, resolved from `DeviceResources`
/// before the buffers are built.
#[derive(Clone)]
pub(super) struct VectorFieldBindings {
    lut_view: viewport_lib::gpu::TextureView,
    lut_sampler: viewport_lib::gpu::Sampler,
    bgl: viewport_lib::gpu::BindGroupLayout,
    instance_bgl: viewport_lib::gpu::BindGroupLayout,
}

/// Resolve the colourmap LUT and the clamping LUT sampler.
///
/// A source naming no colourmap gets Viridis; a stale id falls back to the
/// neutral LUT.
pub(super) fn resolve_bindings(
    resources: &DeviceResources,
    layouts: &VectorFieldResources,
    item: &super::types::VectorFieldItem,
) -> VectorFieldBindings {
    let lut_view = requested_colourmap(&item.colour)
        .and_then(|id| resources.colourmap_view(id))
        .unwrap_or_else(|| {
            resources.builtin_colourmap_view(viewport_lib::resources::BuiltinColourmap::Viridis)
        });
    VectorFieldBindings {
        lut_view: lut_view.clone(),
        // The LUT sampler clamps; the material sampler repeats, which makes a
        // lookup at exactly 0 wrap onto the far end of the colourmap.
        lut_sampler: resources.lut_sampler().clone(),
        bgl: layouts.bgl.clone(),
        instance_bgl: layouts.instance_bgl.clone(),
    }
}

/// The magnitude of every sample's vector: the field's natural scalar, and what
/// both `Natural` sources resolve against.
pub(super) fn magnitudes(item: &super::types::VectorFieldItem) -> Vec<f32> {
    (0..item.positions.len())
        .map(|i| {
            item.vectors
                .get(i)
                .map(|v| glam::Vec3::from(*v).length())
                .unwrap_or(0.0)
        })
        .collect()
}

/// Every sample's resolved size, before the item's global `scale`.
///
/// Shared by the buffer build and the CPU pick, so the hit radius matches what
/// was drawn.
pub(super) fn sample_sizes(item: &super::types::VectorFieldItem, mags: &[f32]) -> Vec<f32> {
    crate::item_types::sources::sample_sizes(&item.size, item.positions.len(), mags)
}

/// The per-sample record the shader reads.
///
/// This *is* the public [`Sample`](super::channels::Sample) rather than a
/// parallel copy of it, so the layout a consumer writes through the sample
/// channel cannot drift from the layout the shader indexes.
type VectorFieldInstance = super::channels::Sample;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VectorFieldUniform {
    model: [[f32; 4]; 4],
    global_scale: f32,
    use_lut: u32,
    scalar_min: f32,
    scalar_max: f32,
    unlit: u32,
    opacity: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Build the GPU data for one vector field: its buffers and its two bind
/// groups.
///
/// Shared by the per-frame item path and the store, so a reference draw and an
/// inline draw are bit-for-bit the same work.
pub(super) fn build_vector_field(
    device: &viewport_lib::gpu::Device,
    queue: &viewport_lib::gpu::Queue,
    binds: &VectorFieldBindings,
    item: &super::types::VectorFieldItem,
    capacity: u32,
) -> VectorFieldGpuData {
    let count = item.positions.len();
    let mags = magnitudes(item);
    let sizes = sample_sizes(item, &mags);
    let colours = colour_plan(&item.colour, count, &mags);

    let instances: Vec<VectorFieldInstance> = (0..count)
        .map(|i| VectorFieldInstance {
            position: item.positions[i],
            size: sizes[i],
            vector: item.vectors.get(i).copied().unwrap_or([0.0, 0.0, 0.0]),
            scalar: colours.scalars[i],
            colour: colours.colours[i],
        })
        .collect();

    let mut samples = ContentBuffer::new(
        device,
        "vector_field_instance_buf",
        viewport_lib::gpu::BufferUsages::STORAGE,
        INSTANCE_STRIDE,
        capacity.max(count as u32),
    );
    let _ = samples.write_range(queue, 0, bytemuck::cast_slice(&instances));

    let (scalar_min, scalar_max) = colours.lut_range.unwrap_or((0.0, 1.0));
    let uniform_data = VectorFieldUniform {
        model: item.model,
        global_scale: item.scale,
        use_lut: colours.lut_range.is_some() as u32,
        scalar_min,
        scalar_max,
        unlit: item.settings.unlit as u32,
        opacity: item.settings.opacity,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let uniform_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
        label: Some("vector_field_uniform_buf"),
        size: std::mem::size_of::<VectorFieldUniform>() as u64,
        usage: viewport_lib::gpu::BufferUsages::UNIFORM | viewport_lib::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

    let uniform_bind_group = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
        label: Some("vector_field_uniform_bg"),
        layout: &binds.bgl,
        entries: &[
            viewport_lib::gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            viewport_lib::gpu::BindGroupEntry {
                binding: 1,
                resource: viewport_lib::gpu::BindingResource::TextureView(&binds.lut_view),
            },
            viewport_lib::gpu::BindGroupEntry {
                binding: 2,
                resource: viewport_lib::gpu::BindingResource::Sampler(&binds.lut_sampler),
            },
        ],
    });

    let instance_bind_group = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
        label: Some("vector_field_instance_bg"),
        layout: &binds.instance_bgl,
        entries: &[viewport_lib::gpu::BindGroupEntry {
            binding: 0,
            resource: samples.buffer().as_entire_binding(),
        }],
    });

    VectorFieldGpuData {
        shape: item.shape,
        pick_id: item.settings.pick_id,
        uniform_bind_group,
        instance_bind_group,
        colourmap: crate::item_types::sources::requested_colourmap(&item.colour),
        binds: binds.clone(),
        uniform_buf,
        samples,
    }
}

/// Rewrite a stored field's instance buffer and uniform in place, keeping both
/// bind groups.
///
/// Returns `false` when the new item does not fit what is allocated: a
/// different sample count needs a bigger or smaller instance buffer, and a
/// different colourmap needs a different LUT view in the uniform bind group.
/// The shape `MeshId` may change freely, because the draw binds it by id rather
/// than through a bind group.
pub(super) fn try_replace_in_place(
    queue: &viewport_lib::gpu::Queue,
    gpu: &mut VectorFieldGpuData,
    item: &super::types::VectorFieldItem,
) -> bool {
    let count = item.positions.len() as u32;
    if count > gpu.samples.capacity() {
        return false;
    }
    if crate::item_types::sources::requested_colourmap(&item.colour) != gpu.colourmap {
        return false;
    }
    let count = count as usize;

    let mags = magnitudes(item);
    let sizes = sample_sizes(item, &mags);
    let colours = colour_plan(&item.colour, count, &mags);

    let instances: Vec<VectorFieldInstance> = (0..count)
        .map(|i| VectorFieldInstance {
            position: item.positions[i],
            size: sizes[i],
            vector: item.vectors.get(i).copied().unwrap_or([0.0, 0.0, 0.0]),
            scalar: colours.scalars[i],
            colour: colours.colours[i],
        })
        .collect();
    if gpu.samples.set_len(count as u32).is_err()
        || gpu
            .samples
            .write_range(queue, 0, bytemuck::cast_slice(&instances))
            .is_err()
    {
        return false;
    }

    let (scalar_min, scalar_max) = colours.lut_range.unwrap_or((0.0, 1.0));
    let uniform_data = VectorFieldUniform {
        model: item.model,
        global_scale: item.scale,
        use_lut: colours.lut_range.is_some() as u32,
        scalar_min,
        scalar_max,
        unlit: item.settings.unlit as u32,
        opacity: item.settings.opacity,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

    gpu.shape = item.shape;
    gpu.pick_id = item.settings.pick_id;
    true
}

// ---------------------------------------------------------------------------
// The store, and the uploads that fill it
// ---------------------------------------------------------------------------

/// Slotted store of pre-uploaded vector fields.
pub(super) type VectorFieldStore =
    viewport_lib::resources::handle::SlotStore<VectorFieldGpuData, VectorFieldId>;

impl viewport_lib::resources::handle::GpuByteSize for VectorFieldGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.uniform_buf.size() + self.samples.allocated_bytes()
    }
}

/// GPU data for one vector field: its sample buffer, its uniform block and its
/// two bind groups.
///
/// The shape's vertex and index buffers are not here: they belong to the mesh
/// the consumer uploaded, and the draw hooks bind them by id.
///
/// Not `Clone`: the sample buffer grows, and two owners would each have their own
/// idea of how much is allocated. The frame path takes a [`VectorFieldDraw`].
pub(crate) struct VectorFieldGpuData {
    /// The mesh drawn once per sample.
    pub(crate) shape: MeshId,
    /// Object-level pick id shared by every sample in the field (from the
    /// item's `settings.pick_id`); `PickId::NONE` when it is not pickable.
    pub(crate) pick_id: viewport_lib::PickId,
    /// Bind group (group 1): uniform + LUT texture + sampler.
    pub(crate) uniform_bind_group: viewport_lib::gpu::BindGroup,
    /// Bind group (group 2): per-instance storage buffer.
    pub(crate) instance_bind_group: viewport_lib::gpu::BindGroup,
    /// The colourmap the uniform bind group's LUT view came from, or `None`
    /// for the default. A replace naming a different one has to rebuild it.
    pub(crate) colourmap: Option<viewport_lib::resources::ColourmapId>,
    /// The LUT view, sampler and layouts, kept so a grow can rebuild the
    /// instance bind group without a `DeviceResources` borrow.
    pub(crate) binds: VectorFieldBindings,
    pub(crate) uniform_buf: viewport_lib::gpu::Buffer,
    /// One interleaved record per sample. The whole record is the channel: its
    /// fields are derived together from the item's colour and size sources, and
    /// rewriting one of them would mean reading the others back off the GPU.
    pub(crate) samples: ContentBuffer,
}

/// What a draw needs out of a vector field, cheap to clone into a frame list.
#[derive(Clone)]
pub(crate) struct VectorFieldDraw {
    pub(crate) shape: MeshId,
    pub(crate) instance_count: u32,
    pub(crate) pick_id: viewport_lib::PickId,
    pub(crate) uniform_bind_group: viewport_lib::gpu::BindGroup,
    pub(crate) instance_bind_group: viewport_lib::gpu::BindGroup,
}

impl VectorFieldGpuData {
    /// This field's draw data. The bind groups keep the uniform and sample
    /// buffers alive.
    pub(crate) fn draw(&self) -> VectorFieldDraw {
        VectorFieldDraw {
            shape: self.shape,
            instance_count: self.samples.len(),
            pick_id: self.pick_id,
            uniform_bind_group: self.uniform_bind_group.clone(),
            instance_bind_group: self.instance_bind_group.clone(),
        }
    }

    /// The uniform buffer, for the per-frame model matrix write a reference item
    /// makes.
    pub(crate) fn uniform_buf(&self) -> &viewport_lib::gpu::Buffer {
        &self.uniform_buf
    }

    /// Overwrite `data.len() / stride` samples starting at `first_element`.
    pub(crate) fn write_samples(
        &mut self,
        queue: &viewport_lib::gpu::Queue,
        first_element: u32,
        data: &[u8],
    ) -> ViewportResult<()> {
        self.samples.write_range(queue, first_element, data)
    }

    /// Grow to hold at least `capacity` samples, rebuilding the instance bind
    /// group if the allocation moved. `true` when it did.
    pub(crate) fn reserve(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        capacity: u32,
    ) -> bool {
        let moved = self.samples.reserve(device, queue, capacity);
        if moved {
            self.instance_bind_group =
                device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
                    label: Some("vector_field_instance_bg"),
                    layout: &self.binds.instance_bgl,
                    entries: &[viewport_lib::gpu::BindGroupEntry {
                        binding: 0,
                        resource: self.samples.buffer().as_entire_binding(),
                    }],
                });
        }
        moved
    }

    /// Set how many samples draw.
    pub(crate) fn set_live_len(&mut self, len: u32) -> ViewportResult<()> {
        self.samples.set_len(len)
    }

    /// What the sample buffer holds and how much of it draws.
    pub(crate) fn extent(&self) -> Extent {
        Extent {
            capacity: self.samples.capacity(),
            len: self.samples.len(),
        }
    }
}
