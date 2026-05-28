//! GPU-driven culling compute dispatch.
//!
//! `CullResources` holds the two compute pipelines and supporting buffers for
//! the cull pass. Call `dispatch` once per frame after uploading instance AABBs
//! and batch metadata to run:
//!
//!   1. `cull_instances`    — one thread per instance, tests AABB vs frustum,
//!                            claims a visibility slot via atomicAdd.
//!   2. `write_indirect_args` — one thread per batch, writes a DrawIndexedIndirect
//!                              entry and resets the counter for the next frame.
//!
//! The two dispatches run in separate wgpu compute passes so the automatic
//! storage-buffer barrier between passes guarantees pass 2 sees pass 1 writes.

use crate::resources::FrustumUniform;

/// Bind group layout entry count for the cull compute pass (group 0).
const CULL_BGL_ENTRY_COUNT: usize = 6;

/// GPU culling infrastructure: pipelines, BGL, and frustum uniform buffer.
pub(super) struct CullResources {
    /// Compute pipeline for the `cull_instances` entry point (workgroup 64).
    cull_instances_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for the `write_indirect_args` entry point (workgroup 64).
    write_indirect_args_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for both cull pipelines (6 bindings, all COMPUTE).
    bgl: wgpu::BindGroupLayout,
    /// 96-byte uniform buffer holding the 6-plane camera frustum for the main pass.
    pub(super) frustum_buf: wgpu::Buffer,
    /// Per-cascade frustum uniform buffers for shadow GPU culling.
    pub(super) cascade_frustum_bufs: [wgpu::Buffer; 4],
}

impl CullResources {
    /// Create the cull pipelines and frustum buffer.
    ///
    /// Called lazily on the first frame where GPU culling is active.
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull_bgl"),
            entries: &Self::bgl_entries(),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cull_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!(concat!(env!("OUT_DIR"), "/cull.wgsl")).into()),
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

        Self {
            cull_instances_pipeline,
            write_indirect_args_pipeline,
            bgl,
            frustum_buf,
            cascade_frustum_bufs,
        }
    }

    /// Dispatch the two cull compute passes into `encoder`.
    ///
    /// - `frustum`: frustum planes to upload this frame.
    /// - `aabb_buf` / `meta_buf` / `counter_buf` / `vis_buf` / `indirect_buf`:
    ///   the GPU buffers allocated by `upload_aabb_and_batch_meta`.
    /// - `instance_count`: total number of instances (drives pass-1 dispatch).
    /// - `batch_count`: number of batches (drives pass-2 dispatch).
    pub(super) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frustum: &FrustumUniform,
        aabb_buf: &wgpu::Buffer,
        meta_buf: &wgpu::Buffer,
        counter_buf: &wgpu::Buffer,
        vis_buf: &wgpu::Buffer,
        indirect_buf: &wgpu::Buffer,
        instance_count: u32,
        batch_count: u32,
    ) {
        // Upload frustum for this frame.
        queue.write_buffer(
            &self.frustum_buf,
            0,
            bytemuck::cast_slice(std::slice::from_ref(frustum)),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.frustum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: aabb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: vis_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indirect_buf.as_entire_binding(),
                },
            ],
        });

        // Pass 1: cull_instances — one thread per instance.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cull_instances_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_instances_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = instance_count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // wgpu inserts an automatic storage-buffer barrier between compute passes,
        // so pass 2 is guaranteed to see all writes from pass 1.

        // Pass 2: write_indirect_args — one thread per batch.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("write_indirect_args_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.write_indirect_args_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = batch_count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    /// Dispatch shadow cascade cull passes into `encoder` for one cascade.
    ///
    /// Reuses the same compute pipelines and BGL as the main pass. The bind group
    /// binds `cascade_frustum_bufs[cascade_idx]` instead of the camera frustum, and
    /// writes into `shadow_vis_buf` / `shadow_indirect_buf` instead of the main-pass
    /// buffers. The shared `counter_buf` is zeroed by `write_indirect_args` at the
    /// end of each cascade dispatch, so cascades can be chained in one encoder.
    pub(super) fn dispatch_shadow(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cascade_idx: usize,
        frustum: &FrustumUniform,
        aabb_buf: &wgpu::Buffer,
        meta_buf: &wgpu::Buffer,
        counter_buf: &wgpu::Buffer,
        shadow_vis_buf: &wgpu::Buffer,
        shadow_indirect_buf: &wgpu::Buffer,
        instance_count: u32,
        batch_count: u32,
    ) {
        queue.write_buffer(
            &self.cascade_frustum_bufs[cascade_idx],
            0,
            bytemuck::cast_slice(std::slice::from_ref(frustum)),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("shadow_cull_bg_{cascade_idx}")),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cascade_frustum_bufs[cascade_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: aabb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shadow_vis_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_indirect_buf.as_entire_binding(),
                },
            ],
        });

        // Pass 1: cull_instances — one thread per instance.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("shadow_cull_instances_pass_{cascade_idx}")),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_instances_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(instance_count.div_ceil(64), 1, 1);
        }

        // wgpu inserts an automatic storage-buffer barrier between compute passes.

        // Pass 2: write_indirect_args — one thread per batch.
        // Also zeroes batch_counters ready for the next cascade or next frame.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("shadow_write_indirect_args_pass_{cascade_idx}")),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.write_indirect_args_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(batch_count.div_ceil(64), 1, 1);
        }
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
            // binding 2: batch_metas (read-only storage)
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
            // binding 3: batch_counters (read-write storage, atomic)
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
            // binding 4: visibility_indices (read-write storage)
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
            // binding 5: indirect_args (read-write storage)
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
