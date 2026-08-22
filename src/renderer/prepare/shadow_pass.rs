//! Shadow depth pass: directional cascades and point-light cube-map faces.

use super::*;

impl ViewportRenderer {
    /// Render the shadow depth pass: directional CSM cascades into the atlas
    /// tiles and point-light cube-map faces, including per-cascade plugin draws.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_shadow_pass(
        resources: &mut DeviceResources,
        instancing: &mut InstancingState,
        shadow: &mut crate::renderer::shadow_state::ShadowState,
        compute_filter_results: &[crate::resources::ComputeFilterResult],
        plugins: &std::collections::HashMap<
            &'static str,
            Box<dyn crate::plugin_api::ItemTypePlugin>,
        >,
        plugin_frame_index: u64,
        lighting: &crate::renderer::types::LightingSettings,
        scene_items: &[SceneRenderItem],
        light: &LightingFrame,
        shadows_skipped: bool,
        last_stats: &mut crate::renderer::stats::FrameStats,
        ts_query_set: Option<&crate::gpu::QuerySet>,
        ts_written_mask: &std::sync::atomic::AtomicU32,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        // Shadow-pass instrumentation. The stall reported on some mobile backends
        // shows up at `present` because the shadow depth work is GPU-bound and only
        // forced to completion later. When the `viewport_lib::shadow` target is
        // enabled at debug level, bracket the pass and poll the device to completion
        // so the shadow GPU cost is attributed here instead of hiding inside present.
        // The poll is skipped entirely (zero overhead) when the target is off.
        // Enable with `RUST_LOG=viewport_lib::shadow=debug`. For non-perturbing
        // timing in shipping builds, use GPU timestamp queries instead.
        let shadow_instrument =
            tracing::enabled!(target: "viewport_lib::shadow", tracing::Level::DEBUG);
        let shadow_start = std::time::Instant::now();

        // Shadow depth pass : CSM: render each cascade into its atlas tile.
        // Skip the pass entirely when over budget and shadow reduction is allowed.
        // ------------------------------------------------------------------
        let skip_shadows = shadows_skipped;

        // When skipping the shadow pass (budget pressure or empty scene), clear the
        // atlas to max depth so that stale values from a previous frame or a previous
        // showcase don't produce phantom shadows.
        if lighting.shadows.enabled && (skip_shadows || scene_items.is_empty()) {
            let mut enc = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                label: Some("shadow_clear_encoder"),
            });
            let _ = enc.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("shadow_clear_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: &resources.shadow_map_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(1.0),
                        store: crate::gpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            sink.push(enc.finish());
        }

        // `effective_cascade_count == 0` means the primary light does not cast
        // shadows; the shader skips its CSM sample entirely (the atlas uniform
        // carries cascade_count 0), so the atlas needs neither rendering nor a
        // clear.
        if lighting.shadows.enabled
            && !scene_items.is_empty()
            && !skip_shadows
            && light.effective_cascade_count > 0
        {
            // ------------------------------------------------------------------
            // Shadow GPU cull dispatch
            //
            // For each active cascade, dispatch `cull_instances` + `write_indirect_args`
            // with the cascade frustum. Results land in `shadow_vis_bufs[c]` and
            // `shadow_indirect_bufs[c]`, consumed by the shadow render pass below.
            // All cascade dispatches share the same `batch_counter_buf`; each
            // `write_indirect_args` dispatch resets the counters for the next cascade.
            // ------------------------------------------------------------------
            // Without GPU culling, the same per-cascade cull runs on the CPU:
            // visible instance indices are compacted into `shadow_vis_bufs[c]`
            // and the render pass below draws each batch's sub-range through
            // the `vs_shadow_cull` pipeline. `cpu_cull_ranges[cascade][batch]`
            // holds the (offset, count) of each batch's compacted run.
            let mut cpu_cull_ranges: Option<Vec<Vec<(u32, u32)>>> = None;
            if instancing.gpu_culling_enabled
                && instancing.use_instancing
                && !instancing.batches.is_empty()
                && instancing.cached_instance_count > 0
            {
                // Mutable operations first.
                if instancing.cull_resources.is_none() {
                    instancing.cull_resources =
                        Some(crate::renderer::indirect::CullResources::new(device));
                }
                resources.ensure_cull_instance_pipelines(device);

                let instance_count = instancing.cached_instance_count as u32;
                let batch_count = instancing.batches.len() as u32;
                instancing
                    .shadow_cull
                    .ensure_outputs(device, instance_count, batch_count);
                // Drop shadow cull bind groups whose binding-0 instance storage
                // buffer was rebuilt this frame; ensure_outputs already handles a
                // resized shadow-vis buffer.
                if instancing.shadow_cull.built_gen != instancing.instance_gen {
                    instancing.shadow_cull.shadow_cull_instance_bgs = [None, None, None, None];
                    instancing.shadow_cull.shadow_cutout_cull_bgs.clear();
                    instancing.shadow_cull.built_gen = instancing.instance_gen;
                }
                for c in 0..light.effective_cascade_count {
                    resources.get_shadow_cull_instance_bind_group(
                        &mut instancing.shadow_cull,
                        device,
                        c,
                    );
                }

                if let (Some(aabb_buf), Some(meta_buf), Some(counter_buf)) = (
                    resources.cull.aabb_buf.as_ref(),
                    resources.cull.batch_meta_buf.as_ref(),
                    instancing.shadow_cull.batch_counter_buf.as_ref(),
                ) {
                    let cull = instancing.cull_resources.as_ref().unwrap();
                    let mut shadow_cull_encoder =
                        device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                            label: Some("shadow_cull_encoder"),
                        });
                    for c in 0..light.effective_cascade_count {
                        if let (Some(shadow_vis_buf), Some(shadow_indirect_buf)) = (
                            instancing.shadow_cull.shadow_vis_bufs[c].as_ref(),
                            instancing.shadow_cull.shadow_indirect_bufs[c].as_ref(),
                        ) {
                            let cpu_frustum = crate::camera::frustum::Frustum::from_view_proj(
                                &light.cascade_view_projs[c],
                            );
                            let sub = crate::plugin_api::CullSubmission {
                                instance_aabbs: aabb_buf,
                                instance_count,
                                batch_meta: meta_buf,
                                batch_count,
                                counter: counter_buf,
                                visible_out: shadow_vis_buf,
                                indirect_out: shadow_indirect_buf,
                                shadow_pass: true,
                            };
                            cull.dispatch(
                                &mut shadow_cull_encoder,
                                device,
                                queue,
                                &cpu_frustum,
                                Some(c),
                                &sub,
                                None,
                                None,
                            );
                        }
                    }
                    sink.push(shadow_cull_encoder.finish());
                }
            } else if instancing.use_instancing
                && !instancing.batches.is_empty()
                && instancing.cached_instance_count > 0
                && instancing.cached_aabbs.len() == instancing.cached_instance_count
            {
                // CPU per-cascade shadow cull for devices without GPU-driven
                // culling. Reuses the cull-variant shadow pipeline and the
                // per-cascade visibility buffers; only the index compaction
                // moves to the CPU.
                resources.ensure_cull_instance_pipelines(device);
                if resources.cull.shadow_pipeline.is_some() {
                    let instance_count = instancing.cached_instance_count as u32;
                    let batch_count = instancing.batches.len() as u32;
                    instancing
                        .shadow_cull
                        .ensure_outputs(device, instance_count, batch_count);
                    if instancing.shadow_cull.built_gen != instancing.instance_gen {
                        instancing.shadow_cull.shadow_cull_instance_bgs = [None, None, None, None];
                        instancing.shadow_cull.shadow_cutout_cull_bgs.clear();
                        instancing.shadow_cull.built_gen = instancing.instance_gen;
                    }
                    for c in 0..light.effective_cascade_count {
                        resources.get_shadow_cull_instance_bind_group(
                            &mut instancing.shadow_cull,
                            device,
                            c,
                        );
                    }

                    let mut ranges: Vec<Vec<(u32, u32)>> =
                        Vec::with_capacity(light.effective_cascade_count);
                    let mut indices: Vec<u32> =
                        Vec::with_capacity(instancing.cached_instance_count);
                    for c in 0..light.effective_cascade_count {
                        let frustum = crate::camera::frustum::Frustum::from_view_proj(
                            &light.cascade_view_projs[c],
                        );
                        indices.clear();
                        let mut batch_ranges: Vec<(u32, u32)> =
                            Vec::with_capacity(instancing.batches.len());
                        for batch in &instancing.batches {
                            let start = indices.len() as u32;
                            let lo = batch.instance_offset as usize;
                            let hi = lo + batch.instance_count as usize;
                            for (i, ia) in instancing.cached_aabbs[lo..hi].iter().enumerate() {
                                if ia.cast_shadows == 0 {
                                    continue;
                                }
                                let aabb = crate::Aabb {
                                    min: ia.min.into(),
                                    max: ia.max.into(),
                                };
                                if frustum.cull_aabb(&aabb) {
                                    continue;
                                }
                                indices.push((lo + i) as u32);
                            }
                            batch_ranges.push((start, indices.len() as u32 - start));
                        }
                        if let Some(vis_buf) = instancing.shadow_cull.shadow_vis_bufs[c].as_ref() {
                            if !indices.is_empty() {
                                queue.write_buffer(vis_buf, 0, bytemuck::cast_slice(&indices));
                            }
                        }
                        ranges.push(batch_ranges);
                    }
                    cpu_cull_ranges = Some(ranges);
                }
            }

            let mut encoder =
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("shadow_pass_encoder"),
                });
            {
                let shadow_ts_writes = ts_query_set.map(|qs| {
                    ts_written_mask.fetch_or(
                        1 << crate::renderer::GPU_TS_SHADOW,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    crate::gpu::RenderPassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(crate::renderer::GPU_TS_SHADOW * 2),
                        end_of_pass_write_index: Some(crate::renderer::GPU_TS_SHADOW * 2 + 1),
                    }
                });
                let mut shadow_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("shadow_pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view: &resources.shadow_map_view,
                                depth_ops: Some(crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Clear(1.0),
                                    store: crate::gpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            },
                        ),
                        timestamp_writes: shadow_ts_writes,
                        occlusion_query_set: None,
                    });

                let mut shadow_draws = 0u32;
                // Post-collapse shadow draw commands and geometry binds, for the
                // slab/multi-draw FrameStats counters. `shadow_draws` stays the
                // pre-collapse per-batch count that feeds `shadow_draw_calls`.
                let mut shadow_draw_cmds = 0u32;
                let mut shadow_binds = 0u32;
                let tile_px = light.tile_size as f32;

                if instancing.use_instancing {
                    let use_shadow_indirect = instancing.gpu_culling_enabled
                        && resources.cull.shadow_pipeline.is_some()
                        && instancing.shadow_cull.shadow_vis_bufs[0].is_some();

                    // On backends with native multi-draw the per-cascade shadow
                    // draws collapse into one multi_draw_indexed_indirect per
                    // pipeline/bind-group run, drawn live in the pass; the render
                    // bundle cannot host a multi-draw, so the bundle build and
                    // replay below are skipped in that case.
                    let shadow_multi_draw = instancing.multi_draw_active();
                    if use_shadow_indirect {
                        // GPU-culled indirect shadow path, replayed from cached
                        // per-cascade render bundles. The draw sequence below is
                        // identical every frame for a stable batch list (the
                        // per-frame variation lives in the cascade uniform and
                        // the GPU-cull-written indirect args, which the bundle
                        // references rather than bakes in), and encoding it
                        // costs one set_vertex_buffer/set_index_buffer/draw per
                        // batch per cascade, which dominates prepare() on
                        // many-mesh scenes. Record once, replay until the batch
                        // list or a referenced buffer changes.
                        let bundle_key = (
                            instancing.instance_gen,
                            instancing.batches_gen,
                            instancing.shadow_cull.outputs_gen,
                            light.effective_cascade_count,
                        );
                        if !shadow_multi_draw
                            && instancing.shadow_cull.bundle_key != Some(bundle_key)
                        {
                            instancing.shadow_cull.shadow_bundles = [None, None, None, None];
                            instancing.shadow_cull.bundle_draws = 0;
                            instancing.shadow_cull.bundle_binds = 0;

                            // Build the per-batch alpha-cutout bind groups up
                            // front so the encode loop can look them up with
                            // immutable borrows only.
                            let cutout_keys: Vec<_> = instancing
                                .batches
                                .iter()
                                .filter(|b| b.is_cutout && !b.is_transparent)
                                .map(|b| (b.texture_id, b.normal_map_id, b.ao_map_id))
                                .collect();
                            for cascade in 0..light.effective_cascade_count {
                                for &(t, n, a) in &cutout_keys {
                                    resources.get_shadow_cutout_cull_bind_group(
                                        &mut instancing.shadow_cull,
                                        device,
                                        cascade,
                                        t,
                                        n,
                                        a,
                                    );
                                }
                            }

                            for cascade in 0..light.effective_cascade_count {
                                let Some(pipeline) = resources.cull.shadow_pipeline.as_ref() else {
                                    continue;
                                };
                                let Some(pipeline_two_sided) =
                                    resources.cull.shadow_two_sided_pipeline.as_ref()
                                else {
                                    continue;
                                };
                                let cutout_pipeline =
                                    resources.cull.shadow_cutout_pipeline.as_ref();
                                let cutout_pipeline_two_sided =
                                    resources.cull.shadow_cutout_two_sided_pipeline.as_ref();
                                let Some(cascade_bg) =
                                    resources.instancing.shadow_cascade_bgs[cascade].as_ref()
                                else {
                                    continue;
                                };
                                let Some(inst_cull_bg) =
                                    instancing.shadow_cull.shadow_cull_instance_bgs[cascade]
                                        .as_ref()
                                else {
                                    continue;
                                };
                                let Some(shadow_indirect_buf) =
                                    instancing.shadow_cull.shadow_indirect_bufs[cascade].as_ref()
                                else {
                                    continue;
                                };

                                // Depth-only bundle against the Depth32Float
                                // shadow atlas. Viewport/scissor are render-pass
                                // state, so the per-cascade atlas tile set by
                                // the replay loop below applies to the bundled
                                // draws.
                                let mut bundle_enc =
                                    crate::resources::builders::render_bundle_encoder(
                                        device,
                                        "shadow_cascade_bundle",
                                        &[],
                                        Some(crate::gpu::RenderBundleDepthStencil {
                                            format: crate::gpu::TextureFormat::Depth32Float,
                                            depth_read_only: false,
                                            stencil_read_only: true,
                                        }),
                                        1,
                                    );
                                bundle_enc.set_bind_group(0, cascade_bg, &[]);

                                // Track the currently bound (pipeline, group-1) state. Cutout
                                // batches swap to the cutout pipeline and rebind group 1 to a
                                // bind group that carries the batch albedo texture; opaque
                                // batches use the depth-only pipeline and the shared group 1.
                                let mut cur_pipe: Option<(bool, bool)> = None; // (two_sided, cutout)
                                let mut cur_group1_opaque = false;
                                let mut cur_chunks: Option<(u32, u32)> = None;
                                let mut draws = 0u32;
                                let mut binds = 0u32;
                                for (bi, batch) in instancing.batches.iter().enumerate() {
                                    if batch.is_transparent {
                                        continue;
                                    }
                                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                        continue;
                                    };
                                    // Resolve the cutout bind group; fall back to the opaque
                                    // path if the cutout pipeline or bind group is missing.
                                    let cutout_bg = if batch.is_cutout {
                                        let key = (
                                            cascade,
                                            batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                            batch
                                                .normal_map_id
                                                .map(|t| t.raw())
                                                .unwrap_or(u64::MAX),
                                            batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        );
                                        instancing.shadow_cull.shadow_cutout_cull_bgs.get(&key)
                                    } else {
                                        None
                                    };
                                    let use_cutout = cutout_bg.is_some()
                                        && cutout_pipeline.is_some()
                                        && cutout_pipeline_two_sided.is_some();

                                    if cur_pipe != Some((batch.two_sided, use_cutout)) {
                                        let pipe = match (use_cutout, batch.two_sided) {
                                            (true, true) => cutout_pipeline_two_sided.unwrap(),
                                            (true, false) => cutout_pipeline.unwrap(),
                                            (false, true) => pipeline_two_sided,
                                            (false, false) => pipeline,
                                        };
                                        bundle_enc.set_pipeline(pipe);
                                        cur_pipe = Some((batch.two_sided, use_cutout));
                                    }
                                    if use_cutout {
                                        bundle_enc.set_bind_group(1, cutout_bg.unwrap(), &[]);
                                        cur_group1_opaque = false;
                                    } else if !cur_group1_opaque {
                                        bundle_enc.set_bind_group(1, inst_cull_bg, &[]);
                                        cur_group1_opaque = true;
                                    }
                                    let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                                    if cur_chunks != Some(chunks) {
                                        bundle_enc.set_vertex_buffer(
                                            0,
                                            resources.geometry.vertex_chunk_slice(chunks.0),
                                        );
                                        bundle_enc.set_index_buffer(
                                            resources.geometry.index_chunk_slice(chunks.1),
                                            crate::gpu::IndexFormat::Uint32,
                                        );
                                        binds += 2;
                                        cur_chunks = Some(chunks);
                                    }
                                    bundle_enc
                                        .draw_indexed_indirect(shadow_indirect_buf, bi as u64 * 20);
                                    draws += 1;
                                }
                                let bundle =
                                    bundle_enc.finish(&crate::gpu::RenderBundleDescriptor {
                                        label: Some("shadow_cascade_bundle"),
                                    });
                                instancing.shadow_cull.shadow_bundles[cascade] = Some(bundle);
                                instancing.shadow_cull.bundle_draws += draws;
                                instancing.shadow_cull.bundle_binds += binds;
                            }
                            instancing.shadow_cull.bundle_key = Some(bundle_key);
                        }

                        if shadow_multi_draw {
                            let counts = draw_shadow_cascades_multi_draw(
                                &mut shadow_pass,
                                device,
                                queue,
                                resources,
                                instancing,
                                light,
                                tile_px,
                            );
                            shadow_draws += counts.batch_draws;
                            shadow_draw_cmds += counts.draw_commands;
                            shadow_binds += counts.binds;
                        } else {
                            for cascade in 0..light.effective_cascade_count {
                                let tile_col = (cascade % 2) as f32;
                                let tile_row = (cascade / 2) as f32;
                                shadow_pass.set_viewport(
                                    tile_col * tile_px,
                                    tile_row * tile_px,
                                    tile_px,
                                    tile_px,
                                    0.0,
                                    1.0,
                                );
                                shadow_pass.set_scissor_rect(
                                    (tile_col * tile_px) as u32,
                                    (tile_row * tile_px) as u32,
                                    light.tile_size,
                                    light.tile_size,
                                );

                                // Write cascade view-projection matrix.
                                queue.write_buffer(
                                    resources.instancing.shadow_cascade_bufs[cascade]
                                        .as_ref()
                                        .expect("shadow_instanced_cascade_bufs not allocated"),
                                    0,
                                    bytemuck::cast_slice(
                                        &light.cascade_view_projs[cascade].to_cols_array_2d(),
                                    ),
                                );

                                if let Some(bundle) =
                                    instancing.shadow_cull.shadow_bundles[cascade].as_ref()
                                {
                                    shadow_pass.execute_bundles(std::iter::once(bundle));
                                }
                            }
                            // Bundles replay per-batch draws (no multi-draw), so the
                            // recorded draw and bind counts are the per-frame totals.
                            shadow_draws += instancing.shadow_cull.bundle_draws;
                            shadow_draw_cmds += instancing.shadow_cull.bundle_draws;
                            shadow_binds += instancing.shadow_cull.bundle_binds;
                        }
                    } else if let (Some(ranges), Some(pipeline), Some(pipeline_two_sided)) = (
                        cpu_cull_ranges.as_ref(),
                        resources.cull.shadow_pipeline.as_ref(),
                        resources.cull.shadow_two_sided_pipeline.as_ref(),
                    ) {
                        // CPU-culled direct path: same pipelines and bind groups
                        // as the indirect path, but each batch draws the
                        // compacted sub-range computed on the CPU above.
                        for cascade in 0..light.effective_cascade_count {
                            let tile_col = (cascade % 2) as f32;
                            let tile_row = (cascade / 2) as f32;
                            shadow_pass.set_viewport(
                                tile_col * tile_px,
                                tile_row * tile_px,
                                tile_px,
                                tile_px,
                                0.0,
                                1.0,
                            );
                            shadow_pass.set_scissor_rect(
                                (tile_col * tile_px) as u32,
                                (tile_row * tile_px) as u32,
                                light.tile_size,
                                light.tile_size,
                            );

                            queue.write_buffer(
                                resources.instancing.shadow_cascade_bufs[cascade]
                                    .as_ref()
                                    .expect("shadow_instanced_cascade_bufs not allocated"),
                                0,
                                bytemuck::cast_slice(
                                    &light.cascade_view_projs[cascade].to_cols_array_2d(),
                                ),
                            );

                            let cutout_keys: Vec<_> = instancing
                                .batches
                                .iter()
                                .filter(|b| b.is_cutout && !b.is_transparent)
                                .map(|b| (b.texture_id, b.normal_map_id, b.ao_map_id))
                                .collect();
                            for (t, n, a) in cutout_keys {
                                resources.get_shadow_cutout_cull_bind_group(
                                    &mut instancing.shadow_cull,
                                    device,
                                    cascade,
                                    t,
                                    n,
                                    a,
                                );
                            }
                            let cutout_pipeline = resources.cull.shadow_cutout_pipeline.as_ref();
                            let cutout_pipeline_two_sided =
                                resources.cull.shadow_cutout_two_sided_pipeline.as_ref();

                            let Some(cascade_bg) =
                                resources.instancing.shadow_cascade_bgs[cascade].as_ref()
                            else {
                                continue;
                            };
                            let Some(inst_cull_bg) =
                                instancing.shadow_cull.shadow_cull_instance_bgs[cascade].as_ref()
                            else {
                                continue;
                            };
                            shadow_pass.set_bind_group(0, cascade_bg, &[]);

                            let mut cur_pipe: Option<(bool, bool)> = None;
                            let mut cur_group1_opaque = false;
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            for (bi, batch) in instancing.batches.iter().enumerate() {
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(&(start, count)) = ranges[cascade].get(bi) else {
                                    continue;
                                };
                                if count == 0 {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                let cutout_bg = if batch.is_cutout {
                                    let key = (
                                        cascade,
                                        batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    );
                                    instancing.shadow_cull.shadow_cutout_cull_bgs.get(&key)
                                } else {
                                    None
                                };
                                let use_cutout = cutout_bg.is_some()
                                    && cutout_pipeline.is_some()
                                    && cutout_pipeline_two_sided.is_some();

                                if cur_pipe != Some((batch.two_sided, use_cutout)) {
                                    let pipe = match (use_cutout, batch.two_sided) {
                                        (true, true) => cutout_pipeline_two_sided.unwrap(),
                                        (true, false) => cutout_pipeline.unwrap(),
                                        (false, true) => pipeline_two_sided,
                                        (false, false) => pipeline,
                                    };
                                    shadow_pass.set_pipeline(pipe);
                                    cur_pipe = Some((batch.two_sided, use_cutout));
                                }
                                if use_cutout {
                                    shadow_pass.set_bind_group(1, cutout_bg.unwrap(), &[]);
                                    cur_group1_opaque = false;
                                } else if !cur_group1_opaque {
                                    shadow_pass.set_bind_group(1, inst_cull_bg, &[]);
                                    cur_group1_opaque = true;
                                }
                                let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                                if cur_chunks != Some(chunks) {
                                    shadow_pass.set_vertex_buffer(
                                        0,
                                        resources.geometry.vertex_chunk_slice(chunks.0),
                                    );
                                    shadow_pass.set_index_buffer(
                                        resources.geometry.index_chunk_slice(chunks.1),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    shadow_binds += 2;
                                    cur_chunks = Some(chunks);
                                }
                                let base_vertex = resources.geometry.base_vertex(mesh.vertex_span);
                                let first_index = resources.geometry.first_index(mesh.index_span);
                                shadow_pass.draw_indexed(
                                    first_index..first_index + mesh.index_count,
                                    base_vertex,
                                    start..start + count,
                                );
                                shadow_draws += 1;
                                shadow_draw_cmds += 1;
                            }
                        }
                    } else if let (Some(pipeline), Some(pipeline_two_sided), Some(instance_bg)) = (
                        &resources.instancing.shadow_pipeline,
                        &resources.instancing.shadow_two_sided_pipeline,
                        instancing.batches.first().and_then(|b| {
                            resources.instancing.bind_groups.get(&(
                                b.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                b.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                b.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                            ))
                        }),
                    ) {
                        // Direct draw shadow path (fallback when GPU culling is off).
                        for cascade in 0..light.effective_cascade_count {
                            let tile_col = (cascade % 2) as f32;
                            let tile_row = (cascade / 2) as f32;
                            shadow_pass.set_viewport(
                                tile_col * tile_px,
                                tile_row * tile_px,
                                tile_px,
                                tile_px,
                                0.0,
                                1.0,
                            );
                            shadow_pass.set_scissor_rect(
                                (tile_col * tile_px) as u32,
                                (tile_row * tile_px) as u32,
                                light.tile_size,
                                light.tile_size,
                            );

                            queue.write_buffer(
                                resources.instancing.shadow_cascade_bufs[cascade]
                                    .as_ref()
                                    .expect("shadow_instanced_cascade_bufs not allocated"),
                                0,
                                bytemuck::cast_slice(
                                    &light.cascade_view_projs[cascade].to_cols_array_2d(),
                                ),
                            );

                            let cascade_bg = resources.instancing.shadow_cascade_bgs[cascade]
                                .as_ref()
                                .expect("shadow_instanced_cascade_bgs not allocated");
                            let cutout_pipeline =
                                resources.instancing.shadow_cutout_pipeline.as_ref();
                            let cutout_pipeline_two_sided = resources
                                .instancing
                                .shadow_cutout_two_sided_pipeline
                                .as_ref();
                            shadow_pass.set_bind_group(0, cascade_bg, &[]);

                            let mut cur_pipe: Option<(bool, bool)> = None;
                            let mut cur_group1_opaque = false;
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            for batch in &instancing.batches {
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                // Cutout batches sample the albedo alpha, so they need the
                                // batch's own texture bind group (not the shared first-batch
                                // one) at group 1.
                                let cutout_bg = if batch.is_cutout {
                                    resources.instancing.bind_groups.get(&(
                                        batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    ))
                                } else {
                                    None
                                };
                                let use_cutout = cutout_bg.is_some()
                                    && cutout_pipeline.is_some()
                                    && cutout_pipeline_two_sided.is_some();

                                if cur_pipe != Some((batch.two_sided, use_cutout)) {
                                    let pipe = match (use_cutout, batch.two_sided) {
                                        (true, true) => cutout_pipeline_two_sided.unwrap(),
                                        (true, false) => cutout_pipeline.unwrap(),
                                        (false, true) => pipeline_two_sided,
                                        (false, false) => pipeline,
                                    };
                                    shadow_pass.set_pipeline(pipe);
                                    cur_pipe = Some((batch.two_sided, use_cutout));
                                }
                                if use_cutout {
                                    shadow_pass.set_bind_group(1, cutout_bg.unwrap(), &[]);
                                    cur_group1_opaque = false;
                                } else if !cur_group1_opaque {
                                    shadow_pass.set_bind_group(1, instance_bg, &[]);
                                    cur_group1_opaque = true;
                                }
                                let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                                if cur_chunks != Some(chunks) {
                                    shadow_pass.set_vertex_buffer(
                                        0,
                                        resources.geometry.vertex_chunk_slice(chunks.0),
                                    );
                                    shadow_pass.set_index_buffer(
                                        resources.geometry.index_chunk_slice(chunks.1),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    shadow_binds += 2;
                                    cur_chunks = Some(chunks);
                                }
                                let base_vertex = resources.geometry.base_vertex(mesh.vertex_span);
                                let first_index = resources.geometry.first_index(mesh.index_span);
                                shadow_pass.draw_indexed(
                                    first_index..first_index + mesh.index_count,
                                    base_vertex,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                                shadow_draws += 1;
                                shadow_draw_cmds += 1;
                            }
                        }
                    }

                    // Per-item shadow casters for items excluded from
                    // instanced batches.
                    //
                    // `sorted_items` in the instancing builder filters out
                    // items with: active scalar attributes, two-sided
                    // materials, matcap shading, UV/param visualisation,
                    // per-instance deformer data (skinned), and
                    // position/normal overrides. Those items render through
                    // the per-item lit path; without this loop they would
                    // also be silently dropped from the shadow atlas. Draw
                    // them here with the non-instanced `shadow_pipeline`
                    // (group 0 = `shadow_bind_group` with cascade dynamic
                    // offset, group 1 = mesh.object_bind_group, group 2 =
                    // per-mesh deform sidecar).
                    let filter_results = compute_filter_results;
                    for cascade in 0..light.effective_cascade_count {
                        let tile_col = (cascade % 2) as f32;
                        let tile_row = (cascade / 2) as f32;
                        shadow_pass.set_viewport(
                            tile_col * tile_px,
                            tile_row * tile_px,
                            tile_px,
                            tile_px,
                            0.0,
                            1.0,
                        );
                        shadow_pass.set_scissor_rect(
                            (tile_col * tile_px) as u32,
                            (tile_row * tile_px) as u32,
                            light.tile_size,
                            light.tile_size,
                        );
                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow_bind_group,
                            &[cascade as u32 * 256],
                        );

                        let cascade_frustum = crate::camera::frustum::Frustum::from_view_proj(
                            &light.cascade_view_projs[cascade],
                        );

                        for item in scene_items.iter() {
                            if item.settings.hidden
                                || !item.settings.cast_shadows
                                || item.settings.opacity < 1.0
                            {
                                continue;
                            }
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };

                            // Mirror the inclusion filter from
                            // `sorted_items` (instancing builder). When
                            // every condition holds, the item was drawn by
                            // the instanced shadow path and must not be
                            // drawn again here. Two-sided (`Identical`) meshes
                            // are now in the instanced batches, so they are
                            // excluded here via `backface_needs_per_object`.
                            let in_instanced_batch = item.active_attribute.is_none()
                                && !backface_needs_per_object(item)
                                && item.material.matcap_id().is_none()
                                && item.material.param_vis.is_none()
                                && !filter_results.iter().any(|r| r.mesh_id == item.mesh_id)
                                && !resources.deform.has_per_instance_deform_data(
                                    item.mesh_id,
                                    item.deform_instance,
                                )
                                && mesh.position_override_buffer.is_none()
                                && mesh.normal_override_buffer.is_none();
                            if in_instanced_batch {
                                continue;
                            }

                            let world_aabb = mesh
                                .aabb
                                .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                            if cascade_frustum.cull_aabb(&world_aabb) {
                                continue;
                            }

                            // Two-sided materials cast through the cull-none
                            // pipeline so both faces rasterise; its larger
                            // caster-side bias keeps the surface from
                            // self-shadowing where it is its own receiver.
                            if item.material.is_two_sided() {
                                shadow_pass.set_pipeline(&resources.shadow_pipeline_two_sided);
                            } else {
                                shadow_pass.set_pipeline(&resources.shadow_pipeline);
                            }
                            shadow_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            bind_deform_group!(
                                shadow_pass,
                                resources,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance,)
                            );
                            shadow_pass.set_vertex_buffer(
                                0,
                                resources.geometry.vertex_slice(mesh.vertex_span),
                            );
                            shadow_pass.set_index_buffer(
                                resources.geometry.index_slice(mesh.index_span),
                                crate::gpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            shadow_binds += 2;
                            shadow_draws += 1;
                            shadow_draw_cmds += 1;
                        }
                    }
                } else {
                    for cascade in 0..light.effective_cascade_count {
                        let tile_col = (cascade % 2) as f32;
                        let tile_row = (cascade / 2) as f32;
                        shadow_pass.set_viewport(
                            tile_col * tile_px,
                            tile_row * tile_px,
                            tile_px,
                            tile_px,
                            0.0,
                            1.0,
                        );
                        shadow_pass.set_scissor_rect(
                            (tile_col * tile_px) as u32,
                            (tile_row * tile_px) as u32,
                            light.tile_size,
                            light.tile_size,
                        );

                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow_bind_group,
                            &[cascade as u32 * 256],
                        );

                        let cascade_frustum = crate::camera::frustum::Frustum::from_view_proj(
                            &light.cascade_view_projs[cascade],
                        );

                        for item in scene_items.iter() {
                            if item.settings.hidden {
                                continue;
                            }
                            if !item.settings.cast_shadows {
                                continue;
                            }
                            if item.settings.opacity < 1.0 {
                                continue;
                            }
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };

                            let world_aabb = mesh
                                .aabb
                                .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                            if cascade_frustum.cull_aabb(&world_aabb) {
                                continue;
                            }

                            // Two-sided materials cast through the cull-none
                            // pipeline (see the instanced path above).
                            if item.material.is_two_sided() {
                                shadow_pass.set_pipeline(&resources.shadow_pipeline_two_sided);
                            } else {
                                shadow_pass.set_pipeline(&resources.shadow_pipeline);
                            }
                            shadow_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            bind_deform_group!(
                                shadow_pass,
                                resources,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance,)
                            );
                            shadow_pass.set_vertex_buffer(
                                0,
                                resources.geometry.vertex_slice(mesh.vertex_span),
                            );
                            shadow_pass.set_index_buffer(
                                resources.geometry.index_slice(mesh.index_span),
                                crate::gpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            shadow_binds += 2;
                            shadow_draws += 1;
                            shadow_draw_cmds += 1;
                        }
                    }
                }

                // Item-type plugin shadow casting: per-cascade dispatch
                // with viewport + scissor + cascade bind group set up by
                // the lib. The plugin map is read through a raw pointer
                // captured before the mutable resources borrow split off.
                if !plugins.is_empty() && !frame.scene.plugin_items.is_empty() {
                    for cascade in 0..light.effective_cascade_count {
                        let tile_col = (cascade % 2) as f32;
                        let tile_row = (cascade / 2) as f32;
                        shadow_pass.set_viewport(
                            tile_col * tile_px,
                            tile_row * tile_px,
                            tile_px,
                            tile_px,
                            0.0,
                            1.0,
                        );
                        shadow_pass.set_scissor_rect(
                            (tile_col * tile_px) as u32,
                            (tile_row * tile_px) as u32,
                            light.tile_size,
                            light.tile_size,
                        );
                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow_bind_group,
                            &[cascade as u32 * 256],
                        );
                        let ctx = crate::plugin_api::ShadowCastContext {
                            cascade_idx: cascade as u32,
                            light_view_proj: light.cascade_view_projs[cascade],
                            camera: &frame.camera.render_camera,
                            viewport_index: frame.camera.viewport_index,
                            frame_index: plugin_frame_index,
                        };
                        for (name, plugin) in plugins.iter() {
                            if let Some(items) = frame.scene.plugin_items.get(*name) {
                                plugin.cast_shadow_pass(&mut shadow_pass, &ctx, items.as_ref());
                            }
                        }
                    }
                }

                drop(shadow_pass);
                last_stats.shadow_draw_calls = shadow_draws;
                last_stats.shadow_draw_commands = shadow_draw_cmds;
                last_stats.shadow_buffer_binds = shadow_binds;
            }
            sink.push(encoder.finish());
        }

        // ----------------------------------------------------------------
        // Point-light cubemap shadow passes.
        //
        // One depth-only render pass per (slot, face). Reuses the cascade
        // shadow rasterisation conventions (front-face culling, opaque
        // casters only) but writes linear distance-to-light to frag_depth
        // via `shadow_point_pipeline`. Per-face culling uses the standard
        // CPU frustum from the face's view-projection.
        // ----------------------------------------------------------------
        if lighting.shadows.enabled
            && !scene_items.is_empty()
            && !light.point_shadow_faces.is_empty()
        {
            // Collect the caster list once: item filter, mesh lookup, and
            // world AABB are shared by every slot and face below instead of
            // being recomputed per (face, item).
            struct PointCaster<'a> {
                item: &'a SceneRenderItem,
                mesh: &'a crate::resources::GpuMesh,
                world_aabb: crate::scene::aabb::Aabb,
            }
            let casters: Vec<PointCaster> = scene_items
                .iter()
                .filter(|item| {
                    !item.settings.hidden
                        && item.settings.cast_shadows
                        && item.settings.opacity >= 1.0
                })
                .filter_map(|item| {
                    let mesh = resources.mesh_store.get(item.mesh_id)?;
                    let world_aabb = mesh
                        .aabb
                        .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                    Some(PointCaster {
                        item,
                        mesh,
                        world_aabb,
                    })
                })
                .collect();

            // Decide per pool slot whether its cubemap must re-render this
            // frame. A slot's content is fully determined by its light's
            // position and range plus every in-range caster's mesh identity,
            // content revision, and model matrix; hash those and skip the
            // slot's six face passes when the hash matches what was rendered
            // last time. The pool's depth layers persist between frames, so a
            // skipped slot keeps sampling its existing cubemap. Casters with
            // an active deform instance or a position/normal override animate
            // without changing any hashed input, so their slots re-render
            // every frame.
            let closest_sq = |aabb: &crate::scene::aabb::Aabb, p: glam::Vec3| -> f32 {
                let c = p.clamp(aabb.min, aabb.max);
                (p - c).length_squared()
            };
            let mut slot_dirty = std::collections::HashMap::new();
            for fc in light.point_shadow_faces.iter() {
                if slot_dirty.contains_key(&fc.slot) {
                    continue;
                }
                let mut hasher = crate::renderer::shader_hashes::fnv1a_hash(&[]);
                let mut mix = |bytes: &[u8]| {
                    for &b in bytes {
                        hasher ^= b as u64;
                        hasher = hasher.wrapping_mul(0x0000_0100_0000_01B3);
                    }
                };
                mix(&fc.light_pos.x.to_le_bytes());
                mix(&fc.light_pos.y.to_le_bytes());
                mix(&fc.light_pos.z.to_le_bytes());
                mix(&fc.range.to_le_bytes());
                let mut cacheable = true;
                let range_sq = fc.range * fc.range;
                for c in casters.iter() {
                    if closest_sq(&c.world_aabb, fc.light_pos) > range_sq {
                        continue;
                    }
                    if c.item.deform_instance.is_some()
                        || c.mesh.position_override_buffer.is_some()
                        || c.mesh.normal_override_buffer.is_some()
                    {
                        cacheable = false;
                        break;
                    }
                    mix(&(c.item.mesh_id.index() as u64).to_le_bytes());
                    mix(&c.item.mesh_id.generation.to_le_bytes());
                    mix(&c.mesh.content_rev.to_le_bytes());
                    mix(bytemuck::cast_slice(&c.item.model));
                }
                let slot_idx = fc.slot as usize;
                let dirty = !cacheable
                    || shadow
                        .point_shadow_slot_hashes
                        .get(slot_idx)
                        .copied()
                        .flatten()
                        != Some(hasher);
                if let Some(entry) = shadow.point_shadow_slot_hashes.get_mut(slot_idx) {
                    *entry = cacheable.then_some(hasher);
                }
                slot_dirty.insert(fc.slot, dirty);
            }
            let any_dirty = slot_dirty.values().any(|&d| d);

            // Make sure each casting item's `mesh.object_uniform_buf` carries
            // the item's current world model matrix. When instancing is on,
            // the per-item write-buffer pass earlier in `prepare_scene_internal`
            // skips items that go through the instanced path, leaving the
            // shared per-mesh uniform stale. The cascade shadow pass works
            // around this by drawing instanced items through a different
            // pipeline; the point shadow path here renders every caster
            // through `shadow_point_pipeline` (non-instanced), so it needs
            // a fresh per-mesh write here. Multi-item-per-mesh scenes need a
            // dedicated per-item shadow uniform; the single-item case is the
            // priority bug fix. Skipped entirely when every slot's cubemap is
            // reused this frame.
            if any_dirty {
                for c in casters.iter() {
                    // Only the model matrix (offset 0, 64 bytes) is read by
                    // `shadow_point.wgsl`. Write just that prefix so we don't
                    // clobber the rest of the per-mesh `ObjectUniform`.
                    queue.write_buffer(
                        &c.mesh.object_uniform_buf,
                        0,
                        bytemuck::cast_slice(&c.item.model),
                    );
                }
            }

            let rendered: Vec<&super::PointShadowFace> = light
                .point_shadow_faces
                .iter()
                .filter(|fc| slot_dirty.get(&fc.slot).copied().unwrap_or(true))
                .collect();
            if !rendered.is_empty() {
                let mut enc =
                    device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                        label: Some("point_shadow_encoder"),
                    });
                let last_face = rendered.len() - 1;
                for (face_idx, fc) in rendered.iter().enumerate() {
                    let layer = fc.slot * 6 + fc.face;
                    let view = &resources.point_shadow_face_views[layer as usize];
                    // One slot spans all faces: begin on the first face pass, end
                    // on the last, so the readback sees the whole cubemap cost
                    // (up to casters x 6 depth passes) as one duration.
                    let ts_writes = ts_query_set
                        .filter(|_| face_idx == 0 || face_idx == last_face)
                        .map(|qs| {
                            ts_written_mask.fetch_or(
                                1 << crate::renderer::GPU_TS_POINT_SHADOW,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            let slot = crate::renderer::GPU_TS_POINT_SHADOW;
                            crate::gpu::RenderPassTimestampWrites {
                                query_set: qs,
                                beginning_of_pass_write_index: (face_idx == 0).then_some(slot * 2),
                                end_of_pass_write_index: (face_idx == last_face)
                                    .then_some(slot * 2 + 1),
                            }
                        });
                    let mut pass = enc.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("point_shadow_face_pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view,
                                depth_ops: Some(crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Clear(1.0),
                                    store: crate::gpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            },
                        ),
                        timestamp_writes: ts_writes,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&resources.shadow_point_pipeline);
                    let dyn_offset = layer * POINT_FACE_STRIDE as u32;
                    pass.set_bind_group(0, &resources.shadow_point_face_bind_group, &[dyn_offset]);

                    let face_frustum =
                        crate::camera::frustum::Frustum::from_view_proj(&fc.view_proj);

                    for c in casters.iter() {
                        if face_frustum.cull_aabb(&c.world_aabb) {
                            continue;
                        }
                        pass.set_bind_group(1, &c.mesh.object_bind_group, &[]);
                        bind_deform_group!(
                            pass,
                            resources,
                            resources
                                .deform
                                .instance_bind_group_for(c.item.mesh_id, c.item.deform_instance,)
                        );
                        pass.set_vertex_buffer(
                            0,
                            resources.geometry.vertex_slice(c.mesh.vertex_span),
                        );
                        pass.set_index_buffer(
                            resources.geometry.index_slice(c.mesh.index_span),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..c.mesh.index_count, 0, 0..1);
                    }
                }
                sink.push(enc.finish());
            }
        }

        if shadow_instrument && lighting.shadows.enabled {
            // Force the just-submitted shadow work to finish so the measured time
            // reflects shadow GPU execution rather than landing later at present.
            // Other work submitted before this point in prepare is minor, so this
            // is a good attribution of the shadow cost.
            device
                .poll(crate::gpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_millis(2000)),
                })
                .ok();
            tracing::debug!(
                target: "viewport_lib::shadow",
                ms = shadow_start.elapsed().as_secs_f32() * 1000.0,
                cascades = light.effective_cascade_count,
                atlas = resources.shadow_atlas_size,
                draws = last_stats.shadow_draw_calls,
                point_faces = light.point_shadow_faces.len(),
                "shadow pass + gpu completion"
            );
        }
    }
}

/// Draw the directional shadow cascades live in the pass with per-run
/// `multi_draw_indexed_indirect`, the backend-native alternative to the cached
/// render bundle (which cannot host a multi-draw). Geometry is bound once per
/// slab chunk and the GPU-cull-written args carry each mesh's base_vertex /
/// first_index; a run collapses consecutive batches that share the pipeline
/// variant, the group-1 bind group, and the slab chunk, breaking on a
/// global-index gap so a multi-draw never sweeps in a batch this loop skipped.
/// Returns the number of batch draws issued (pre-collapse, for stats parity
/// with the bundle path).
#[allow(clippy::too_many_arguments)]
fn draw_shadow_cascades_multi_draw(
    pass: &mut crate::gpu::RenderPass<'_>,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    resources: &DeviceResources,
    instancing: &mut InstancingState,
    light: &LightingFrame,
    tile_px: f32,
) -> ShadowDrawCounts {
    let Some(pipeline) = resources.cull.shadow_pipeline.as_ref() else {
        return ShadowDrawCounts::default();
    };
    let Some(pipeline_two_sided) = resources.cull.shadow_two_sided_pipeline.as_ref() else {
        return ShadowDrawCounts::default();
    };
    let cutout_pipeline = resources.cull.shadow_cutout_pipeline.as_ref();
    let cutout_pipeline_two_sided = resources.cull.shadow_cutout_two_sided_pipeline.as_ref();
    let multi_draw = instancing.multi_draw_active();
    let mut drawn = 0u32;
    let mut binds = 0u32;
    let mut draw_cmds = 0u32;

    for cascade in 0..light.effective_cascade_count {
        let tile_col = (cascade % 2) as f32;
        let tile_row = (cascade / 2) as f32;
        pass.set_viewport(
            tile_col * tile_px,
            tile_row * tile_px,
            tile_px,
            tile_px,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            (tile_col * tile_px) as u32,
            (tile_row * tile_px) as u32,
            light.tile_size,
            light.tile_size,
        );

        queue.write_buffer(
            resources.instancing.shadow_cascade_bufs[cascade]
                .as_ref()
                .expect("shadow_instanced_cascade_bufs not allocated"),
            0,
            bytemuck::cast_slice(&light.cascade_view_projs[cascade].to_cols_array_2d()),
        );

        // Build this cascade's alpha-cutout bind groups up front (mutable
        // borrow), so the draw loop below can look them up immutably.
        let cutout_keys: Vec<_> = instancing
            .batches
            .iter()
            .filter(|b| b.is_cutout && !b.is_transparent)
            .map(|b| (b.texture_id, b.normal_map_id, b.ao_map_id))
            .collect();
        for (t, n, a) in cutout_keys {
            resources.get_shadow_cutout_cull_bind_group(
                &mut instancing.shadow_cull,
                device,
                cascade,
                t,
                n,
                a,
            );
        }

        let Some(shadow_indirect_buf) =
            instancing.shadow_cull.shadow_indirect_bufs[cascade].as_ref()
        else {
            continue;
        };
        let Some(cascade_bg) = resources.instancing.shadow_cascade_bgs[cascade].as_ref() else {
            continue;
        };
        let Some(inst_cull_bg) = instancing.shadow_cull.shadow_cull_instance_bgs[cascade].as_ref()
        else {
            continue;
        };
        pass.set_bind_group(0, cascade_bg, &[]);

        let mut cur_pipe: Option<(bool, bool)> = None; // (two_sided, cutout)
        let mut cur_group1: Option<*const crate::gpu::BindGroup> = None;
        let mut cur_chunks: Option<(u32, u32)> = None;
        let mut run_start: u64 = 0;
        let mut run_len: u32 = 0;

        for (bi, batch) in instancing.batches.iter().enumerate() {
            if batch.is_transparent {
                continue;
            }
            let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                continue;
            };
            let cutout_bg = if batch.is_cutout {
                let key = (
                    cascade,
                    batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                    batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                    batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                );
                instancing.shadow_cull.shadow_cutout_cull_bgs.get(&key)
            } else {
                None
            };
            let use_cutout = cutout_bg.is_some()
                && cutout_pipeline.is_some()
                && cutout_pipeline_two_sided.is_some();
            let group1: &crate::gpu::BindGroup = if use_cutout {
                cutout_bg.unwrap()
            } else {
                inst_cull_bg
            };
            let pipe_key = (batch.two_sided, use_cutout);
            let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
            let g1_ptr = group1 as *const crate::gpu::BindGroup;
            let g = bi as u64;

            if run_len > 0
                && g == run_start + run_len as u64
                && cur_pipe == Some(pipe_key)
                && cur_group1 == Some(g1_ptr)
                && cur_chunks == Some(chunks)
            {
                run_len += 1;
                drawn += 1;
                continue;
            }
            if run_len > 0 {
                draw_cmds += crate::renderer::render::emit_indirect_run(
                    pass,
                    shadow_indirect_buf,
                    run_start,
                    run_len,
                    multi_draw,
                );
            }
            if cur_pipe != Some(pipe_key) {
                let pipe = match (use_cutout, batch.two_sided) {
                    (true, true) => cutout_pipeline_two_sided.unwrap(),
                    (true, false) => cutout_pipeline.unwrap(),
                    (false, true) => pipeline_two_sided,
                    (false, false) => pipeline,
                };
                pass.set_pipeline(pipe);
                cur_pipe = Some(pipe_key);
            }
            if cur_group1 != Some(g1_ptr) {
                pass.set_bind_group(1, group1, &[]);
                cur_group1 = Some(g1_ptr);
            }
            if cur_chunks != Some(chunks) {
                pass.set_vertex_buffer(0, resources.geometry.vertex_chunk_slice(chunks.0));
                pass.set_index_buffer(
                    resources.geometry.index_chunk_slice(chunks.1),
                    crate::gpu::IndexFormat::Uint32,
                );
                binds += 2;
                cur_chunks = Some(chunks);
            }
            run_start = g;
            run_len = 1;
            drawn += 1;
        }
        if run_len > 0 {
            draw_cmds += crate::renderer::render::emit_indirect_run(
                pass,
                shadow_indirect_buf,
                run_start,
                run_len,
                multi_draw,
            );
        }
    }

    ShadowDrawCounts {
        batch_draws: drawn,
        binds,
        draw_commands: draw_cmds,
    }
}

/// Shadow instanced-draw tallies for one pass: `batch_draws` is the pre-collapse
/// per-batch draw count (feeds `FrameStats::shadow_draw_calls`), `binds` counts
/// `set_vertex_buffer` + `set_index_buffer` calls, and `draw_commands` is the
/// post-collapse draw count (a multi-draw counts once).
#[derive(Default, Clone, Copy)]
struct ShadowDrawCounts {
    batch_draws: u32,
    binds: u32,
    draw_commands: u32,
}
