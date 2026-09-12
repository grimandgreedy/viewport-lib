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
const CULL_BGL_ENTRY_COUNT: usize = 9;

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
    pub(super) hiz_view: Option<&'a crate::gpu::TextureView>,
    /// Caller's request to run the occlusion test. Ignored when `hiz_view`
    /// is `None`.
    pub(super) do_occlusion: bool,
    /// Per-instance record buffer (the instanced `InstanceData` storage buffer),
    /// read by the cull for each instance's `object_mask`. `None` leaves the
    /// layer-mask reject off and binds the fallback buffer.
    pub(super) instance_data: Option<&'a crate::gpu::Buffer>,
    /// Camera layer mask AND-tested against each instance's `object_mask`.
    /// `!0` (and the default) keeps everything; only consulted when
    /// `instance_data` is `Some`.
    pub(super) cull_mask: u32,
}

/// Per-batch group assignment for GPU draw-list compaction: which pipeline group
/// a batch belongs to (`group_id`, keying the survivor counter) and the batch
/// index at which that group's compacted args start (`group_arg_base`). Opaque
/// and transparent (OIT) batches both get real groups; a batch in neither pass
/// (additive / premultiplied, or when the GPU-driven path is inactive) carries
/// `group_id = NO_GROUP` and is skipped by the compaction pass.
pub(super) const NO_GROUP: u32 = u32::MAX;

/// Cull compute pipelines and the lib's shared scratch buffers.
pub(super) struct CullResources {
    /// Compute pipeline for `cull_instances` (workgroup 64).
    cull_instances_pipeline: crate::gpu::ComputePipeline,
    /// Compute pipeline for `write_indirect_args` (workgroup 64).
    write_indirect_args_pipeline: crate::gpu::ComputePipeline,
    /// Shared bind group layout for both pipelines (6 entries, all COMPUTE).
    bgl: crate::gpu::BindGroupLayout,
    /// Compute pipeline for `compact_draws`: packs each pipeline group's visible
    /// batch draw args to the front of its range and counts the survivors, so the
    /// colour pass issues one multi_draw_indexed_indirect_count per group. Only
    /// used on the GPU-driven submission path (bindless + native multi-draw).
    compact_pipeline: crate::gpu::ComputePipeline,
    /// Bind group layout for `compact_pipeline` (6 entries, all COMPUTE).
    compact_bgl: crate::gpu::BindGroupLayout,
    /// Uniform (batch_count) for the compaction dispatch. One slot, overwritten
    /// each dispatch.
    compact_params_buf: crate::gpu::Buffer,
    /// Frustum uniform for the main-camera dispatch. One slot, overwritten
    /// each frame.
    pub(super) frustum_buf: crate::gpu::Buffer,
    /// Per-cascade frustum uniforms. Separate slots so a single frame can
    /// submit the main pass plus every cascade without overwriting an
    /// in-flight upload.
    pub(super) cascade_frustum_bufs: [crate::gpu::Buffer; 4],
    /// Scratch `BatchMeta` slot for one-mesh submissions that come through
    /// `submit_cull_single_mesh`. One entry, overwritten per call.
    scratch_meta_buf: crate::gpu::Buffer,
    /// Scratch counter slot paired with `scratch_meta_buf`. One u32,
    /// zeroed per call.
    scratch_counter_buf: crate::gpu::Buffer,
    /// 1x1 R32Float texture bound at binding 6 when a dispatch has no HiZ
    /// pyramid (shadow, single-mesh, or occlusion disabled). Keeps the bind
    /// group layout satisfied; never sampled because `do_occlusion` is 0.
    fallback_hiz_view: crate::gpu::TextureView,
    /// One 144-byte `InstanceData` slot bound at binding 8 when a dispatch does
    /// not run the layer-mask reject (shadow, single-mesh, plugin submissions).
    /// Keeps the bind group layout satisfied; never read because `do_mask_cull`
    /// is 0.
    fallback_instance_data: crate::gpu::Buffer,
    /// Cull breakdown counters for the main dispatch: [total, frustum_visible].
    /// Cleared each main dispatch, copied to the readback staging buffer.
    main_stats_buf: crate::gpu::Buffer,
    /// Stats slot for non-main dispatches (shadow, single-mesh). Written but
    /// never read back.
    scratch_stats_buf: crate::gpu::Buffer,
}

impl CullResources {
    /// Build the pipelines, BGL, and the shared scratch buffers.
    pub(super) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("cull_bgl"),
            entries: &Self::bgl_entries(),
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "cull_shader",
            crate::resources::builders::wgsl_source!("cull"),
        );

        let layout =
            crate::resources::builders::pipeline_layout(device, "cull_pipeline_layout", &[&bgl]);

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

        let compact_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("draw_compact_bgl"),
            entries: &Self::compact_bgl_entries(),
        });
        let compact_shader = crate::resources::builders::wgsl_module(
            device,
            "draw_compact_shader",
            crate::resources::builders::wgsl_source!("draw_compact"),
        );
        let compact_layout = crate::resources::builders::pipeline_layout(
            device,
            "draw_compact_pipeline_layout",
            &[&compact_bgl],
        );
        let compact_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "draw_compact_pipeline",
            &compact_layout,
            &compact_shader,
            "compact_draws",
        );
        let compact_params_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("draw_compact_params_buf"),
            size: 16, // vec4-aligned CompactUniform (batch_count + pad)
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frustum_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cull_frustum_buf"),
            size: std::mem::size_of::<FrustumUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cascade_frustum_bufs = std::array::from_fn(|i| {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(&format!("cull_cascade_frustum_buf_{i}")),
                size: std::mem::size_of::<FrustumUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let scratch_meta_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cull_scratch_meta_buf"),
            size: std::mem::size_of::<BatchMeta>() as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_counter_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cull_scratch_counter_buf"),
            size: 4,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fallback_hiz = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("cull_fallback_hiz"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R32Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fallback_hiz_view =
            fallback_hiz.create_view(&crate::gpu::TextureViewDescriptor::default());

        // One InstanceData-sized slot for dispatches that do not run the
        // layer-mask reject. Zeroed; never read (do_mask_cull = 0 for them).
        let fallback_instance_data = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cull_fallback_instance_data"),
            size: std::mem::size_of::<crate::resources::mesh::instancing::InstanceData>() as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Two u32 counters: [total, frustum_visible]. COPY_SRC for the readback
        // copy, COPY_DST for the per-frame clear.
        let main_stats_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cull_main_stats_buf"),
            size: 8,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::COPY_SRC
                | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_stats_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cull_scratch_stats_buf"),
            size: 8,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            cull_instances_pipeline,
            write_indirect_args_pipeline,
            bgl,
            compact_pipeline,
            compact_bgl,
            compact_params_buf,
            frustum_buf,
            cascade_frustum_bufs,
            scratch_meta_buf,
            scratch_counter_buf,
            fallback_hiz_view,
            fallback_instance_data,
            main_stats_buf,
            scratch_stats_buf,
        }
    }

    /// Borrow the main-cull stats buffer ([total, frustum_visible]) so the
    /// instanced prepare path can copy it into its readback staging buffer.
    pub(super) fn main_stats_buf(&self) -> &crate::gpu::Buffer {
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
        encoder: &mut crate::gpu::CommandEncoder,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frustum: &Frustum,
        cascade: Option<usize>,
        sub: &CullSubmission<'_>,
        ts: Option<(&crate::gpu::QuerySet, &std::sync::atomic::AtomicU32)>,
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
        // Layer-mask reject runs only when the main cull supplies the instance
        // buffer. Without it, bind the fallback and leave the reject off.
        let instance_data_buf = extras
            .and_then(|e| e.instance_data)
            .unwrap_or(&self.fallback_instance_data);
        let (cull_mask, do_mask_cull): (u32, u32) = match extras {
            Some(e) if e.instance_data.is_some() => (e.cull_mask, 1),
            _ => (!0, 0),
        };
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
            cull_mask,
            do_mask_cull,
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
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(&label),
            layout: &self.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: frustum_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: sub.instance_aabbs.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: sub.batch_meta.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: sub.counter.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: sub.visible_out.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: sub.indirect_out.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: crate::gpu::BindingResource::TextureView(hiz_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: stats_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: instance_data_buf.as_entire_binding(),
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
                    Some(crate::gpu::ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(cull_slot * 2),
                        end_of_pass_write_index: None,
                    }),
                    Some(crate::gpu::ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: Some(cull_slot * 2 + 1),
                    }),
                )
            }
            None => (None, None),
        };

        {
            let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some(&pass1_label),
                timestamp_writes: ts_begin,
            });
            pass.set_pipeline(&self.cull_instances_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(sub.instance_count.div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
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
    pub(super) fn scratch_single_mesh_buffers(&self) -> (&crate::gpu::Buffer, &crate::gpu::Buffer) {
        (&self.scratch_meta_buf, &self.scratch_counter_buf)
    }

    /// Run the draw-list compaction pass: read the per-batch cull args from
    /// `src_args`, pack each pipeline group's visible batches to the front of its
    /// range in `dst_args`, and write the survivor count per group into
    /// `draw_counts`. `draw_counts` is zeroed first (its atomics accumulate the
    /// per-group counts). Runs in its own compute pass so the automatic
    /// storage barrier orders it after the cull that filled `src_args`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compact_draws(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        batch_count: u32,
        src_args: &crate::gpu::Buffer,
        group_arg_base: &crate::gpu::Buffer,
        group_id: &crate::gpu::Buffer,
        dst_args: &crate::gpu::Buffer,
        draw_counts: &crate::gpu::Buffer,
    ) {
        queue.write_buffer(
            &self.compact_params_buf,
            0,
            bytemuck::cast_slice(&[batch_count, 0u32, 0u32, 0u32]),
        );
        // The survivor counters accumulate via atomicAdd, so start from zero.
        encoder.clear_buffer(draw_counts, 0, None);
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("draw_compact_bg"),
            layout: &self.compact_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: self.compact_params_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: src_args.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: group_arg_base.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: group_id.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: dst_args.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: draw_counts.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
            label: Some("draw_compact_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compact_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(batch_count.div_ceil(64), 1, 1);
    }

    fn compact_bgl_entries() -> [crate::gpu::BindGroupLayoutEntry; 6] {
        let compute = crate::gpu::ShaderStages::COMPUTE;
        let storage = |binding: u32, read_only: bool| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        [
            // binding 0: params uniform (batch_count)
            crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(1, true),  // src_args
            storage(2, true),  // group_arg_base
            storage(3, true),  // group_id
            storage(4, false), // dst_args
            storage(5, false), // draw_counts (atomic)
        ]
    }

    fn bgl_entries() -> [crate::gpu::BindGroupLayoutEntry; CULL_BGL_ENTRY_COUNT] {
        let compute = crate::gpu::ShaderStages::COMPUTE;
        [
            // binding 0: frustum uniform
            crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: instance_aabbs (read-only storage)
            crate::gpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 2: batch_meta (read-only storage)
            crate::gpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 3: batch counters (atomic, read-write storage)
            crate::gpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 4: visibility output (read-write storage)
            crate::gpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 5: indirect args (read-write storage)
            crate::gpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 6: HiZ max-depth pyramid (R32Float, non-filterable, sampled
            // via textureLoad).
            crate::gpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: compute,
                ty: crate::gpu::BindingType::Texture {
                    sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
                    view_dimension: crate::gpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // binding 7: cull breakdown counters (read-write storage)
            crate::gpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 8: per-instance records (read-only storage), read for the
            // per-object layer mask in the camera cull reject.
            crate::gpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: compute,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::CullResources;
    use crate::gpu;

    fn headless_device() -> Option<(gpu::Device, gpu::Queue)> {
        let instance = gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(&gpu::RequestAdapterOptions {
            power_preference: gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            #[cfg(wgpu30)]
            apply_limit_buckets: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&gpu::DeviceDescriptor {
            label: Some("compaction_tests"),
            required_limits: crate::renderer::ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }))
        .ok()?;
        Some((device, queue))
    }

    fn storage_buf(
        device: &gpu::Device,
        queue: &gpu::Queue,
        data: &[u32],
        usage: gpu::BufferUsages,
    ) -> gpu::Buffer {
        let buf = device.create_buffer(&gpu::BufferDescriptor {
            label: None,
            size: (data.len() * 4) as u64,
            usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    fn read_words(device: &gpu::Device, buf: &gpu::Buffer) -> Vec<u32> {
        let slice = buf.slice(..);
        slice.map_async(gpu::MapMode::Read, |_| {});
        let _ = device.poll(gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let words = bytemuck::cast_slice::<u8, u32>(&gpu::mapped_range(slice)).to_vec();
        buf.unmap();
        words
    }

    /// Dispatch `compact_draws` over the crafted inputs and read back
    /// `(per-group survivor counts, compacted dst args as words)`. `batches` are
    /// `DrawIndirect` 5-tuples; `group_id[b]` / `group_arg_base[b]` are the CPU
    /// grouping; `count_slots` sizes the per-group counter buffer.
    fn run_compaction(
        device: &gpu::Device,
        queue: &gpu::Queue,
        cull: &CullResources,
        batches: &[[u32; 5]],
        group_id: &[u32],
        group_arg_base: &[u32],
        count_slots: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let src: Vec<u32> = batches.iter().flatten().copied().collect();
        let storage = gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST;
        let src_args = storage_buf(device, queue, &src, storage);
        let group_arg_base_buf = storage_buf(device, queue, group_arg_base, storage);
        let group_id_buf = storage_buf(device, queue, group_id, storage);
        let dst_args = storage_buf(
            device,
            queue,
            &vec![0u32; batches.len() * 5],
            storage | gpu::BufferUsages::COPY_SRC,
        );
        let draw_counts = device.create_buffer(&gpu::BufferDescriptor {
            label: None,
            size: (count_slots * 4) as u64,
            usage: storage | gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&gpu::CommandEncoderDescriptor {
            label: Some("compaction_test_encoder"),
        });
        cull.compact_draws(
            &mut encoder,
            device,
            queue,
            batches.len() as u32,
            &src_args,
            &group_arg_base_buf,
            &group_id_buf,
            &dst_args,
            &draw_counts,
        );

        let dst_bytes = (batches.len() * 5 * 4) as u64;
        let counts_bytes = (count_slots * 4) as u64;
        let read_usage = gpu::BufferUsages::COPY_DST | gpu::BufferUsages::MAP_READ;
        let dst_staging = device.create_buffer(&gpu::BufferDescriptor {
            label: None,
            size: dst_bytes,
            usage: read_usage,
            mapped_at_creation: false,
        });
        let counts_staging = device.create_buffer(&gpu::BufferDescriptor {
            label: None,
            size: counts_bytes,
            usage: read_usage,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&dst_args, 0, &dst_staging, 0, dst_bytes);
        encoder.copy_buffer_to_buffer(&draw_counts, 0, &counts_staging, 0, counts_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        (
            read_words(device, &counts_staging),
            read_words(device, &dst_staging),
        )
    }

    /// Reconstruct the `DrawIndirect` 5-tuple compacted into `dst` slot `slot`.
    fn entry(dst: &[u32], slot: usize) -> [u32; 5] {
        [
            dst[slot * 5],
            dst[slot * 5 + 1],
            dst[slot * 5 + 2],
            dst[slot * 5 + 3],
            dst[slot * 5 + 4],
        ]
    }

    /// The compaction pass is pure compute, so it runs on any backend (including
    /// Metal, where the `_count` draw path it feeds is dormant). Drive it directly
    /// with crafted per-batch cull args and check it packs each group's visible
    /// batches to the front of the group's arg range and counts the survivors.
    #[test]
    fn compact_draws_packs_survivors_per_group() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping compact_draws_packs_survivors_per_group: no GPU adapter");
            return;
        };
        let cull = CullResources::new(&device);

        // Six batches in two contiguous groups of three. `instance_count == 0`
        // marks a batch the cull emptied (dropped from the draw list). Each batch
        // carries a unique `first_index` so a compacted arg can be traced back to
        // its source batch. Layout matches `DrawIndirect`:
        // (index_count, instance_count, first_index, base_vertex, first_instance).
        let batches: [[u32; 5]; 6] = [
            [100, 5, 0, 0, 0],
            [101, 0, 1, 1, 1], // culled
            [102, 7, 2, 2, 2],
            [103, 0, 3, 3, 3], // culled
            [104, 3, 4, 4, 4],
            [105, 9, 5, 5, 5],
        ];
        // Group 0 = batches 0..3 (arg base 0), group 1 = batches 3..6 (arg base 3).
        let group_id = [0u32, 0, 0, 1, 1, 1];
        let group_arg_base = [0u32, 0, 0, 3, 3, 3];

        let (counts, dst) = run_compaction(
            &device,
            &queue,
            &cull,
            &batches,
            &group_id,
            &group_arg_base,
            2,
        );

        // Two survivors per group (one of each three was culled).
        assert_eq!(counts, vec![2, 2], "per-group survivor counts");

        // Within a group the atomic slot order is unspecified, so compare as sets
        // and trace each survivor back to its source batch by `first_index`.
        let group0: std::collections::HashSet<u32> =
            [entry(&dst, 0)[2], entry(&dst, 1)[2]].into_iter().collect();
        assert_eq!(
            group0,
            [0u32, 2].into_iter().collect(),
            "group 0 survivors packed to [0, 2)"
        );
        let group1: std::collections::HashSet<u32> =
            [entry(&dst, 3)[2], entry(&dst, 4)[2]].into_iter().collect();
        assert_eq!(
            group1,
            [4u32, 5].into_iter().collect(),
            "group 1 survivors packed to [3, 5)"
        );
        // Every compacted arg is a faithful, whole copy of its source batch.
        for slot in [0usize, 1, 3, 4] {
            let e = entry(&dst, slot);
            assert_eq!(
                e, batches[e[2] as usize],
                "arg at slot {slot} copied verbatim"
            );
        }
    }

    /// A batch with `group_id == NO_GROUP` (a transparent or additive batch under
    /// the opaque compaction, say) must be skipped entirely: it must not inflate a
    /// group's survivor count nor scatter its args into `group_arg_base` slot 0.
    #[test]
    fn compact_draws_skips_ungrouped_batches() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping compact_draws_skips_ungrouped_batches: no GPU adapter");
            return;
        };
        let cull = CullResources::new(&device);

        // Batch 1 is ungrouped but visible (instance_count 8). It sits inside
        // group 0's batch range with `group_arg_base == 0`, so a missing skip would
        // corrupt group 0 (inflated count, a stray arg at slot 0).
        let batches: [[u32; 5]; 3] = [
            [100, 5, 0, 0, 0],
            [101, 8, 1, 1, 1], // NO_GROUP: must be dropped despite being visible
            [102, 7, 2, 2, 2],
        ];
        let group_id = [0u32, super::NO_GROUP, 0];
        let group_arg_base = [0u32, 0, 0];

        let (counts, dst) = run_compaction(
            &device,
            &queue,
            &cull,
            &batches,
            &group_id,
            &group_arg_base,
            1,
        );

        // Only batches 0 and 2 survive; the ungrouped batch is not counted.
        assert_eq!(
            counts,
            vec![2],
            "ungrouped batch excluded from the group count"
        );
        let survivors: std::collections::HashSet<u32> =
            [entry(&dst, 0)[2], entry(&dst, 1)[2]].into_iter().collect();
        assert_eq!(
            survivors,
            [0u32, 2].into_iter().collect(),
            "group 0 holds only its own batches"
        );
        assert!(
            !survivors.contains(&1),
            "ungrouped batch 1 must not appear in any group"
        );
    }
}
