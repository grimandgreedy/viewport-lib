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

use crate::sources::{colour_plan, requested_colourmap};
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
    crate::sources::sample_sizes(&item.size, item.positions.len(), mags)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VectorFieldInstance {
    position: [f32; 3],
    size: f32,
    direction: [f32; 3],
    scalar: f32,
    colour: [f32; 4],
}

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
) -> VectorFieldGpuData {
    let count = item.positions.len();
    let mags = magnitudes(item);
    let sizes = sample_sizes(item, &mags);
    let colours = colour_plan(&item.colour, count, &mags);

    let instances: Vec<VectorFieldInstance> = (0..count)
        .map(|i| VectorFieldInstance {
            position: item.positions[i],
            size: sizes[i],
            direction: item.vectors.get(i).copied().unwrap_or([0.0, 0.0, 0.0]),
            scalar: colours.scalars[i],
            colour: colours.colours[i],
        })
        .collect();

    let instance_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
        label: Some("vector_field_instance_buf"),
        size: (std::mem::size_of::<VectorFieldInstance>() * instances.len()).max(48) as u64,
        usage: viewport_lib::gpu::BufferUsages::STORAGE | viewport_lib::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

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
            resource: instance_buf.as_entire_binding(),
        }],
    });

    VectorFieldGpuData {
        shape: item.shape,
        instance_count: count as u32,
        pick_id: item.settings.pick_id,
        uniform_bind_group,
        instance_bind_group,
        colourmap: crate::sources::requested_colourmap(&item.colour),
        _uniform_buf: uniform_buf,
        _instance_buf: instance_buf,
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
    let count = item.positions.len();
    if count as u32 != gpu.instance_count {
        return false;
    }
    if crate::sources::requested_colourmap(&item.colour) != gpu.colourmap {
        return false;
    }

    let mags = magnitudes(item);
    let sizes = sample_sizes(item, &mags);
    let colours = colour_plan(&item.colour, count, &mags);

    let instances: Vec<VectorFieldInstance> = (0..count)
        .map(|i| VectorFieldInstance {
            position: item.positions[i],
            size: sizes[i],
            direction: item.vectors.get(i).copied().unwrap_or([0.0, 0.0, 0.0]),
            scalar: colours.scalars[i],
            colour: colours.colours[i],
        })
        .collect();
    queue.write_buffer(&gpu._instance_buf, 0, bytemuck::cast_slice(&instances));

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
    queue.write_buffer(&gpu._uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

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
        self._uniform_buf.size() + self._instance_buf.size()
    }
}

/// GPU data for one vector field draw: the inline items build it each frame,
/// the store holds it across frames.
///
/// The shape's vertex and index buffers are not here: they belong to the mesh
/// the consumer uploaded, and the draw hooks bind them by id.
#[derive(Clone)]
pub(crate) struct VectorFieldGpuData {
    /// The mesh drawn once per sample.
    pub(crate) shape: MeshId,
    /// Number of samples.
    pub(crate) instance_count: u32,
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
    // Keep buffers alive.
    pub(crate) _uniform_buf: viewport_lib::gpu::Buffer,
    pub(crate) _instance_buf: viewport_lib::gpu::Buffer,
}
