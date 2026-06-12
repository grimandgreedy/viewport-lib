//! GPU-driven culling compute dispatch.
//!
//! `CullResources` holds the two compute pipelines used by every cull
//! submission: `cull_instances` tests each AABB against the frustum and
//! claims a slot in the visibility list via atomic add, then
//! `write_indirect_args` packs the per-batch counts into
//! `DrawIndexedIndirect` entries and zeroes the counter for the next call.
//!
//! All callers, internal and plugin, go through one entry point: `dispatch`
//! takes a [`CullSubmission`] and a CPU [`Frustum`], picks the main or a
//! cascade frustum slot, uploads, builds the bind group, and issues both
//! compute passes. wgpu inserts an automatic storage-buffer barrier between
//! compute passes so the second pass sees the first pass's writes.

use crate::camera::frustum::Frustum;
use crate::plugin_api::{BatchMeta, CullSubmission};
use crate::resources::{FrustumPlane, FrustumUniform};

/// Bind group layout entry count for the cull compute pass.
const CULL_BGL_ENTRY_COUNT: usize = 6;

/// Cull compute pipelines and the lib's shared scratch buffers.
pub(super) struct CullResources {
    /// Compute pipeline for `cull_instances` (workgroup 64).
    cull_instances_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for `write_indirect_args` (workgroup 64).
    write_indirect_args_pipeline: wgpu::ComputePipeline,
    /// Shared bind group layout for both pipelines (6 entries, all COMPUTE).
    bgl: wgpu::BindGroupLayout,
    /// Frustum uniform for the main-camera dispatch. One slot, overwritten
    /// each frame.
    pub(super) frustum_buf: wgpu::Buffer,
    /// Per-cascade frustum uniforms. Separate slots so a single frame can
    /// submit the main pass plus every cascade without overwriting an
    /// in-flight upload.
    pub(super) cascade_frustum_bufs: [wgpu::Buffer; 4],
    /// Scratch `BatchMeta` slot for one-mesh submissions that come through
    /// `submit_cull_single_mesh`. One entry, overwritten per call.
    scratch_meta_buf: wgpu::Buffer,
    /// Scratch counter slot paired with `scratch_meta_buf`. One u32,
    /// zeroed per call.
    scratch_counter_buf: wgpu::Buffer,
}

impl CullResources {
    /// Build the pipelines, BGL, and the shared scratch buffers.
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull_bgl"),
            entries: &Self::bgl_entries(),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cull_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/cull.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let cull_instances_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cull_instances_pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("cull_instances"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let write_indirect_args_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("write_indirect_args_pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("write_indirect_args"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let frustum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull_frustum_buf"),
            size: std::mem::size_of::<FrustumUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cascade_frustum_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("cull_cascade_frustum_buf_{i}")),
                size: std::mem::size_of::<FrustumUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let scratch_meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull_scratch_meta_buf"),
            size: std::mem::size_of::<BatchMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull_scratch_counter_buf"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            cull_instances_pipeline,
            write_indirect_args_pipeline,
            bgl,
            frustum_buf,
            cascade_frustum_bufs,
            scratch_meta_buf,
            scratch_counter_buf,
        }
    }

    /// Run the two compute passes for one cull submission.
    ///
    /// `cascade` selects which frustum buffer slot the upload goes to.
    /// `None` is the main-camera dispatch; `Some(idx)` uploads to the
    /// matching cascade slot and forces the cull shader's shadow flag on
    /// (so `InstanceAabb::cast_shadows = 0` entries are skipped).
    pub(super) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frustum: &Frustum,
        cascade: Option<usize>,
        sub: &CullSubmission<'_>,
    ) {
        let frustum_buf = match cascade {
            None => &self.frustum_buf,
            Some(c) => &self.cascade_frustum_bufs[c],
        };
        let shadow_flag: u32 = if cascade.is_some() || sub.shadow_pass {
            1
        } else {
            0
        };
        let frustum_uniform = FrustumUniform {
            planes: std::array::from_fn(|i| FrustumPlane {
                normal: frustum.planes[i].normal.to_array(),
                distance: frustum.planes[i].d,
            }),
            instance_count: sub.instance_count,
            batch_count: sub.batch_count,
            shadow_pass: shadow_flag,
            _pad: 0,
        };
        queue.write_buffer(
            frustum_buf,
            0,
            bytemuck::cast_slice(std::slice::from_ref(&frustum_uniform)),
        );

        let label = match cascade {
            None => "cull_bg".to_string(),
            Some(c) => format!("cull_shadow_bg_{c}"),
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&label),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frustum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sub.instance_aabbs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sub.batch_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sub.counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sub.visible_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sub.indirect_out.as_entire_binding(),
                },
            ],
        });

        let (pass1_label, pass2_label) = match cascade {
            None => (
                "cull_instances_pass".to_string(),
                "write_indirect_args_pass".to_string(),
            ),
            Some(c) => (
                format!("shadow_cull_instances_pass_{c}"),
                format!("shadow_write_indirect_args_pass_{c}"),
            ),
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&pass1_label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_instances_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(sub.instance_count.div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&pass2_label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.write_indirect_args_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(sub.batch_count.div_ceil(64), 1, 1);
        }
    }

    /// Borrow the scratch meta + counter buffers used by
    /// `submit_cull_single_mesh`. The renderer fills these before each
    /// single-mesh dispatch and passes them through as the submission's
    /// `batch_meta` and `counter` buffers.
    pub(super) fn scratch_single_mesh_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        (&self.scratch_meta_buf, &self.scratch_counter_buf)
    }

    fn bgl_entries() -> [wgpu::BindGroupLayoutEntry; CULL_BGL_ENTRY_COUNT] {
        let compute = wgpu::ShaderStages::COMPUTE;
        [
            // binding 0: frustum uniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: compute,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: instance_aabbs (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: compute,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 2: batch_meta (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: compute,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 3: batch counters (atomic, read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: compute,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 4: visibility output (read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: compute,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 5: indirect args (read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: compute,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    }
}
