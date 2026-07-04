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
const CULL_BGL_ENTRY_COUNT: usize = 8;

/// Per-frame inputs for the HiZ occlusion test, supplied only by the
/// main-camera cull. Shadow and single-mesh dispatches pass `None`, which
/// binds the fallback HiZ texture and disables the occlusion reject.
pub(super) struct MainCullExtras<'a> {
    /// Camera view-projection, column-major as the shader's `mat4x4` expects.
    pub(super) view_proj: [[f32; 4]; 4],
    /// HiZ mip-0 dimensions in pixels.
    pub(super) viewport: [f32; 2],
    /// Full-mip HiZ view to sample. `None` when no pyramid is available yet
    /// (first frame, or occlusion disabled); the reject is skipped.
    pub(super) hiz_view: Option<&'a wgpu::TextureView>,
    /// Caller's request to run the occlusion test. Ignored when `hiz_view`
    /// is `None`.
    pub(super) do_occlusion: bool,
}

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
    /// 1x1 R32Float texture bound at binding 6 when a dispatch has no HiZ
    /// pyramid (shadow, single-mesh, or occlusion disabled). Keeps the bind
    /// group layout satisfied; never sampled because `do_occlusion` is 0.
    fallback_hiz_view: wgpu::TextureView,
    /// Cull breakdown counters for the main dispatch: [total, frustum_visible].
    /// Cleared each main dispatch, copied to the readback staging buffer.
    main_stats_buf: wgpu::Buffer,
    /// Stats slot for non-main dispatches (shadow, single-mesh). Written but
    /// never read back.
    scratch_stats_buf: wgpu::Buffer,
}

impl CullResources {
    /// Build the pipelines, BGL, and the shared scratch buffers.
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull_bgl"),
            entries: &Self::bgl_entries(),
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "cull_shader",
            crate::resources::builders::wgsl_source!("cull"),
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let cull_instances_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "cull_instances_pipeline",
            &layout,
            &shader,
            "cull_instances",
        );

        let write_indirect_args_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "write_indirect_args_pipeline",
            &layout,
            &shader,
            "write_indirect_args",
        );

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

        let fallback_hiz = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cull_fallback_hiz"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fallback_hiz_view = fallback_hiz.create_view(&wgpu::TextureViewDescriptor::default());

        // Two u32 counters: [total, frustum_visible]. COPY_SRC for the readback
        // copy, COPY_DST for the per-frame clear.
        let main_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull_main_stats_buf"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull_scratch_stats_buf"),
            size: 8,
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
            fallback_hiz_view,
            main_stats_buf,
            scratch_stats_buf,
        }
    }

    /// Borrow the main-cull stats buffer ([total, frustum_visible]) so the
    /// instanced prepare path can copy it into its readback staging buffer.
    pub(super) fn main_stats_buf(&self) -> &wgpu::Buffer {
        &self.main_stats_buf
    }

    /// Run the two compute passes for one cull submission.
    ///
    /// `cascade` selects which frustum buffer slot the upload goes to.
    /// `None` is the main-camera dispatch; `Some(idx)` uploads to the
    /// matching cascade slot and forces the cull shader's shadow flag on
    /// (so `InstanceAabb::cast_shadows = 0` entries are skipped).
    /// `ts` is `Some((query_set, written_mask))` only for the main-camera cull,
    /// which writes a begin/end timestamp pair into the `GPU_TS_CULL` slot
    /// (spanning both compute passes) and sets the slot bit in the mask. Shadow
    /// and single-mesh culls pass `None` and are not timed.
    pub(super) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frustum: &Frustum,
        cascade: Option<usize>,
        sub: &CullSubmission<'_>,
        ts: Option<(&wgpu::QuerySet, &std::sync::atomic::AtomicU32)>,
        extras: Option<&MainCullExtras<'_>>,
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
        // Occlusion runs only when the main cull supplies a HiZ view and asks
        // for it. Without a view, bind the fallback and leave the reject off.
        let do_occlusion: u32 = match extras {
            Some(e) if e.do_occlusion && e.hiz_view.is_some() => 1,
            _ => 0,
        };
        let view_proj = extras.map_or(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            |e| e.view_proj,
        );
        let viewport = extras.map_or([1.0, 1.0], |e| e.viewport);
        let hiz_view = extras
            .and_then(|e| e.hiz_view)
            .unwrap_or(&self.fallback_hiz_view);
        // The main cull records its breakdown; other dispatches scribble into
        // the scratch slot so they do not clobber the readback counters.
        let stats_buf = if extras.is_some() {
            &self.main_stats_buf
        } else {
            &self.scratch_stats_buf
        };

        let frustum_uniform = FrustumUniform {
            planes: std::array::from_fn(|i| FrustumPlane {
                normal: frustum.planes[i].normal.to_array(),
                distance: frustum.planes[i].d,
            }),
            instance_count: sub.instance_count,
            batch_count: sub.batch_count,
            shadow_pass: shadow_flag,
            do_occlusion,
            view_proj,
            viewport,
            _pad0: [0.0, 0.0],
        };
        queue.write_buffer(
            frustum_buf,
            0,
            bytemuck::cast_slice(std::slice::from_ref(&frustum_uniform)),
        );

        // Reset the breakdown counters before cull_instances accumulates.
        encoder.clear_buffer(stats_buf, 0, None);

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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: stats_buf.as_entire_binding(),
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

        // Time the whole cull (begin of pass 1 -> end of pass 2) into the
        // GPU_TS_CULL slot when this is the timed main-camera dispatch.
        let cull_slot = crate::renderer::GPU_TS_CULL;
        let (ts_begin, ts_end) = match ts {
            Some((qs, mask)) => {
                mask.fetch_or(1 << cull_slot, std::sync::atomic::Ordering::Relaxed);
                (
                    Some(wgpu::ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(cull_slot * 2),
                        end_of_pass_write_index: None,
                    }),
                    Some(wgpu::ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: Some(cull_slot * 2 + 1),
                    }),
                )
            }
            None => (None, None),
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&pass1_label),
                timestamp_writes: ts_begin,
            });
            pass.set_pipeline(&self.cull_instances_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(sub.instance_count.div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&pass2_label),
                timestamp_writes: ts_end,
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
            // binding 6: HiZ max-depth pyramid (R32Float, non-filterable, sampled
            // via textureLoad).
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: compute,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // binding 7: cull breakdown counters (read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 7,
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
