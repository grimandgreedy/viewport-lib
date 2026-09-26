//! The tensor fields this item type holds on the consumer's behalf, and the
//! per-frame GPU data every tensor field draw is built from.
//!
//! A `TensorFieldItem` carries its tensors and is rebuilt each frame; a
//! `TensorFieldRefItem` names a field uploaded once. Both end up as the same
//! [`TensorFieldGpuData`], which is why the builder is shared.
//!
//! The shape is a consumer-uploaded `MeshId`, so the GPU data names the mesh
//! and the draw hooks bind it through `MeshDraw`. The two bind group layouts
//! live here rather than with the pipelines, because an upload builds its bind
//! groups against them and an upload can arrive long before the first frame
//! that draws one.

use crate::sources::{colour_plan, requested_colourmap};
use viewport_lib::MeshId;
use viewport_lib::error::ViewportResult;
use viewport_lib::plugin_api::Extent;
use viewport_lib::resources::{ContentBuffer, DeviceResources};

/// Bytes per baked sample record.
const INSTANCE_STRIDE: u32 = std::mem::size_of::<TensorFieldInstance>() as u32;

pub(crate) use super::types::TensorFieldId;

/// The tensor field bind group layouts. Uploads build their bind groups against
/// them, so they live here with the store rather than with the item type's
/// pipelines, and are created up front: they are layouts, not compiled
/// pipelines.
pub(super) struct TensorFieldResources {
    /// Bind group layout for the field uniform, LUT and sampler (group 1).
    pub(super) bgl: viewport_lib::gpu::BindGroupLayout,
    /// Bind group layout for the instance storage buffer (group 2).
    pub(super) instance_bgl: viewport_lib::gpu::BindGroupLayout,
}

impl TensorFieldResources {
    pub(super) fn new(device: &viewport_lib::gpu::Device) -> Self {
        let bgl = viewport_lib::plugin_api::builders::uniform_texture_sampler_bgl(
            device,
            "tensor_field_bgl",
            viewport_lib::gpu::ShaderStages::VERTEX | viewport_lib::gpu::ShaderStages::FRAGMENT,
            viewport_lib::gpu::ShaderStages::VERTEX,
        );
        let instance_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("tensor_field_instance_bgl"),
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
pub(super) struct TensorFieldBindings {
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
    layouts: &TensorFieldResources,
    item: &super::types::TensorFieldItem,
) -> TensorFieldBindings {
    let lut_view = requested_colourmap(&item.colour)
        .and_then(|id| resources.colourmap_view(id))
        .unwrap_or_else(|| {
            resources.builtin_colourmap_view(viewport_lib::resources::BuiltinColourmap::Viridis)
        });
    TensorFieldBindings {
        lut_view: lut_view.clone(),
        // The LUT sampler clamps; the material sampler repeats, which makes a
        // lookup at exactly 0 wrap onto the far end of the colourmap.
        lut_sampler: resources.lut_sampler().clone(),
        bgl: layouts.bgl.clone(),
        instance_bgl: layouts.instance_bgl.clone(),
    }
}

/// The dominant eigenvalue of every sample: the field's natural scalar, and
/// what both `Natural` sources resolve against.
///
/// It is signed rather than absolute, so a diverging colourmap separates
/// tension from compression.
pub(super) fn dominant_eigenvalues(item: &super::types::TensorFieldItem) -> Vec<f32> {
    (0..item.positions.len())
        .map(|i| item.tensors.eigen_at(i).values[0])
        .collect()
}

/// Every sample's half-extent along its three eigenvector axes, in the item's
/// own space and already through the size source and the global scale.
///
/// Shared by the buffer build and the CPU pick, so the hit radius matches what
/// was drawn.
pub(super) fn sample_extents(item: &super::types::TensorFieldItem) -> Vec<[f32; 3]> {
    let count = item.positions.len();
    let natural = dominant_eigenvalues(item);
    let sizes = crate::sources::sample_sizes(&item.size, count, &natural);
    (0..count)
        .map(|i| {
            let ev = item.tensors.eigen_at(i).values;
            let s = item.scale * sizes[i];
            // A zero eigenvalue would collapse the instance to a plane and make
            // its normal matrix a division by zero, so it keeps a hair of width.
            [
                (ev[0].abs() * s).max(1e-6),
                (ev[1].abs() * s).max(1e-6),
                (ev[2].abs() * s).max(1e-6),
            ]
        })
        .collect()
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TensorFieldInstance {
    model_col0: [f32; 4],
    model_col1: [f32; 4],
    model_col2: [f32; 4],
    model_col3: [f32; 4],
    normal_col0: [f32; 4],
    normal_col1: [f32; 4],
    normal_col2: [f32; 4],
    scalar: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    colour: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TensorFieldUniform {
    model: [[f32; 4]; 4],
    use_lut: u32,
    scalar_min: f32,
    scalar_max: f32,
    unlit: u32,
    opacity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// This field's per-sample instance records and its uniform block.
///
/// Shared by the build and the in-place replace, so a stored field rewritten in
/// place is bit-for-bit what a fresh build would have produced.
fn build_instances_and_uniform(
    item: &super::types::TensorFieldItem,
) -> (Vec<TensorFieldInstance>, TensorFieldUniform) {
    let count = item.positions.len();
    let natural = dominant_eigenvalues(item);
    let extents = sample_extents(item);
    let colours = colour_plan(&item.colour, count, &natural);

    let instances: Vec<TensorFieldInstance> = (0..count)
        .map(|i| {
            let pos = glam::Vec3::from(item.positions[i]);
            let axes = item.tensors.eigen_at(i).vectors;
            let [s0, s1, s2] = extents[i];

            let col0 = glam::Vec3::from(axes[0]);
            let col1 = glam::Vec3::from(axes[1]);
            let col2 = glam::Vec3::from(axes[2]);

            // Rotation-scale block: RS = R * diag(s0, s1, s2).
            let rs = glam::Mat3::from_cols(col0 * s0, col1 * s1, col2 * s2);
            let mut world_model = glam::Mat4::from_mat3(rs);
            world_model.w_axis = glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);

            // Normal matrix: the inverse transpose of that block, which for an
            // orthonormal basis is R * diag(1/s).
            let nm = glam::Mat3::from_cols(col0 / s0, col1 / s1, col2 / s2);

            let mc = world_model.to_cols_array_2d();
            TensorFieldInstance {
                model_col0: mc[0],
                model_col1: mc[1],
                model_col2: mc[2],
                model_col3: mc[3],
                normal_col0: [nm.x_axis.x, nm.x_axis.y, nm.x_axis.z, 0.0],
                normal_col1: [nm.y_axis.x, nm.y_axis.y, nm.y_axis.z, 0.0],
                normal_col2: [nm.z_axis.x, nm.z_axis.y, nm.z_axis.z, 0.0],
                scalar: colours.scalars[i],
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
                colour: colours.colours[i],
            }
        })
        .collect();

    let (scalar_min, scalar_max) = colours.lut_range.unwrap_or((0.0, 1.0));
    let uniform_data = TensorFieldUniform {
        model: item.model,
        use_lut: colours.lut_range.is_some() as u32,
        scalar_min,
        scalar_max,
        unlit: item.settings.unlit as u32,
        opacity: item.settings.opacity,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };
    (instances, uniform_data)
}

/// Build the GPU data for one tensor field: its buffers and its two bind
/// groups.
///
/// Shared by the per-frame item path and the store, so a reference draw and an
/// inline draw are bit-for-bit the same work.
pub(super) fn build_tensor_field(
    device: &viewport_lib::gpu::Device,
    queue: &viewport_lib::gpu::Queue,
    binds: &TensorFieldBindings,
    item: &super::types::TensorFieldItem,
    capacity: u32,
) -> TensorFieldGpuData {
    let count = item.positions.len();
    let (instances, uniform_data) = build_instances_and_uniform(item);

    let mut samples = ContentBuffer::new(
        device,
        "tensor_field_instance_buf",
        viewport_lib::gpu::BufferUsages::STORAGE,
        INSTANCE_STRIDE,
        capacity.max(count as u32),
    );
    let _ = samples.write_range(queue, 0, bytemuck::cast_slice(&instances));

    let uniform_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
        label: Some("tensor_field_uniform_buf"),
        size: std::mem::size_of::<TensorFieldUniform>() as u64,
        usage: viewport_lib::gpu::BufferUsages::UNIFORM | viewport_lib::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

    let uniform_bind_group = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
        label: Some("tensor_field_uniform_bg"),
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
        label: Some("tensor_field_instance_bg"),
        layout: &binds.instance_bgl,
        entries: &[viewport_lib::gpu::BindGroupEntry {
            binding: 0,
            resource: samples.buffer().as_entire_binding(),
        }],
    });

    TensorFieldGpuData {
        shape: item.shape,
        pick_id: item.settings.pick_id,
        uniform_bind_group,
        instance_bind_group,
        colourmap: crate::sources::requested_colourmap(&item.colour),
        binds: binds.clone(),
        uniform_buf,
        samples,
    }
}

/// Encode one caller-supplied sample into the record the shader reads.
///
/// The rotation-scale block is `R * diag(extents)` and the normal block its
/// inverse transpose, which for an orthonormal `R` is `R * diag(1 / extents)`.
/// Extents are clamped the way the upload path clamps them, so a zero extent
/// leaves a hair of width rather than an infinity in the buffer.
fn encode_sample(sample: &super::channels::Sample) -> TensorFieldInstance {
    let pos = glam::Vec3::from(sample.position);
    let col0 = glam::Vec3::from(sample.axes[0]);
    let col1 = glam::Vec3::from(sample.axes[1]);
    let col2 = glam::Vec3::from(sample.axes[2]);
    let s0 = sample.extents[0].abs().max(1e-6);
    let s1 = sample.extents[1].abs().max(1e-6);
    let s2 = sample.extents[2].abs().max(1e-6);

    let rs = glam::Mat3::from_cols(col0 * s0, col1 * s1, col2 * s2);
    let mut world_model = glam::Mat4::from_mat3(rs);
    world_model.w_axis = glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
    let nm = glam::Mat3::from_cols(col0 / s0, col1 / s1, col2 / s2);
    let mc = world_model.to_cols_array_2d();

    TensorFieldInstance {
        model_col0: mc[0],
        model_col1: mc[1],
        model_col2: mc[2],
        model_col3: mc[3],
        normal_col0: [nm.x_axis.x, nm.x_axis.y, nm.x_axis.z, 0.0],
        normal_col1: [nm.y_axis.x, nm.y_axis.y, nm.y_axis.z, 0.0],
        normal_col2: [nm.z_axis.x, nm.z_axis.y, nm.z_axis.z, 0.0],
        scalar: sample.scalar,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
        colour: sample.colour.to_linear_rgba(),
    }
}

/// Encode a run of caller-supplied samples into the bytes a ranged write sends.
pub(crate) fn encode_samples(samples: &[super::channels::Sample]) -> Vec<u8> {
    let records: Vec<TensorFieldInstance> = samples.iter().map(encode_sample).collect();
    bytemuck::cast_slice(&records).to_vec()
}

/// Rewrite a stored field's instance buffer and uniform in place, keeping both
/// bind groups.
///
/// Returns `false` when the new item does not fit what is allocated: a
/// different sample count needs a different instance buffer, and a different
/// colourmap needs a different LUT view in the uniform bind group. The shape
/// `MeshId` may change freely, because the draw binds it by id rather than
/// through a bind group.
pub(super) fn try_replace_in_place(
    queue: &viewport_lib::gpu::Queue,
    gpu: &mut TensorFieldGpuData,
    item: &super::types::TensorFieldItem,
) -> bool {
    let count = item.positions.len() as u32;
    if count > gpu.samples.capacity() {
        return false;
    }
    if crate::sources::requested_colourmap(&item.colour) != gpu.colourmap {
        return false;
    }

    let (instances, uniform_data) = build_instances_and_uniform(item);
    if gpu.samples.set_len(count).is_err()
        || gpu
            .samples
            .write_range(queue, 0, bytemuck::cast_slice(&instances))
            .is_err()
    {
        return false;
    }
    queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

    gpu.shape = item.shape;
    gpu.pick_id = item.settings.pick_id;
    true
}

// ---------------------------------------------------------------------------
// The store, and the uploads that fill it
// ---------------------------------------------------------------------------

/// Slotted store of pre-uploaded tensor fields.
pub(super) type TensorFieldStore =
    viewport_lib::resources::handle::SlotStore<TensorFieldGpuData, TensorFieldId>;

impl viewport_lib::resources::handle::GpuByteSize for TensorFieldGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.uniform_buf.size() + self.samples.allocated_bytes()
    }
}

/// GPU data for one tensor field: its sample buffer, its uniform block and its
/// two bind groups.
///
/// Not `Clone`: the sample buffer grows, and two owners would each have their own
/// idea of how much is allocated. The frame path takes a [`TensorFieldDraw`].
pub(crate) struct TensorFieldGpuData {
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
    pub(crate) binds: TensorFieldBindings,
    pub(crate) uniform_buf: viewport_lib::gpu::Buffer,
    /// One baked record per sample: a rotation-scale matrix, its inverse
    /// transpose, a scalar and a colour. The whole record is the channel, and a
    /// write supplies the decomposition it is built from.
    pub(crate) samples: ContentBuffer,
}

/// What a draw needs out of a tensor field, cheap to clone into a frame list.
#[derive(Clone)]
pub(crate) struct TensorFieldDraw {
    pub(crate) shape: MeshId,
    pub(crate) instance_count: u32,
    pub(crate) pick_id: viewport_lib::PickId,
    pub(crate) uniform_bind_group: viewport_lib::gpu::BindGroup,
    pub(crate) instance_bind_group: viewport_lib::gpu::BindGroup,
}

impl TensorFieldGpuData {
    /// This field's draw data. The bind groups keep the uniform and sample
    /// buffers alive.
    pub(crate) fn draw(&self) -> TensorFieldDraw {
        TensorFieldDraw {
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

    /// Overwrite the records for `data.len() / stride` samples starting at
    /// `first_element`.
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
                    label: Some("tensor_field_instance_bg"),
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
