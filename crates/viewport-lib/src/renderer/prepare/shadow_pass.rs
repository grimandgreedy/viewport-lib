//! Shadow depth pass: directional cascades and point-light cube-map faces.

use super::*;

/// Whether the shadow cull bind groups no longer match the resources they were
/// built against.
///
/// The same split the main-pass cull uses: a replace swaps a view behind a live
/// id and cannot be detected by checking ids, so it always invalidates; a free
/// removes the id, which checking the batches' resources detects. Under a
/// streaming eviction budget the common case is a free of something these
/// batches never named, and that leaves the bind groups correct.
fn shadow_cull_bind_groups_stale(
    resources: &DeviceResources,
    instancing: &crate::renderer::instancing_state::InstancingState,
) -> bool {
    if instancing.shadow_cull.built_view_epoch != resources.resource_view_epoch {
        return true;
    }
    instancing.shadow_cull.built_free_epoch != resources.resource_free_epoch
        && !crate::renderer::prepare::instanced::cached_batches_resolve(
            resources,
            &instancing.batches,
        )
}

impl ViewportRenderer {
    /// Render the shadow depth pass: directional CSM cascades into the atlas
    /// tiles and point-light cube-map faces, including per-cascade plugin draws.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_shadow_pass(
        resources: &mut DeviceResources,
        instancing: &mut InstancingState,
        shadow: &mut crate::renderer::shadow_state::ShadowState,
        plugins: &crate::renderer::item_plugins::registry::ItemPluginRegistry,
        plugin_frame_index: u64,
        lighting: &crate::renderer::types::LightingSettings,
        scene_items: &[SceneRenderItem],
        // The batches' group-1 bind groups, built by the packer before
        // the shadow pass.
        per_object: &crate::renderer::per_object_state::PerObjectState,
        light: &LightingFrame,
        shadows_skipped: bool,
        last_stats: &mut crate::renderer::stats::FrameStats,
        ts_query_set: Option<&crate::gpu::QuerySet>,
        ts_written_mask: &std::sync::atomic::AtomicU32,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        // The colour family the frame draws with; a caster whose colour
        // pipeline is still compiling casts nothing until it is drawn.
        colour_hdr: bool,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        let clipping_active = DeviceResources::clipping_active(frame);
        // Which batches cast this frame, as a hash: a batch whose colour
        // pipeline is still compiling is left out, and the cached shadow
        // bundles have to be re-recorded when that changes.
        let casters_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for batch in &instancing.batches {
                resources
                    .batch_colour_ready(batch, colour_hdr, clipping_active)
                    .hash(&mut h);
            }
            h.finish()
        };
        // Shadow-pass instrumentation: with the `viewport_lib::shadow` target
        // at debug level, log the pass's CPU encode time and counts
        // (`RUST_LOG=viewport_lib::shadow=debug`). Its GPU time is in the GPU
        // timestamps.
        let shadow_instrument =
            tracing::enabled!(target: "viewport_lib::shadow", tracing::Level::DEBUG);
        let shadow_start = web_time::Instant::now();

        // Shadow depth pass : CSM: render each cascade into its atlas tile.
        // Skip the pass entirely when over budget and shadow reduction is allowed.
        // ------------------------------------------------------------------
        let skip_shadows = shadows_skipped;
        // Item-type plugins cast into the cascades too, so a scene of plugin
        // items alone still has casters.
        let has_casters = !scene_items.is_empty()
            || plugins.iter().any(|(name, _)| {
                !crate::plugin_api::ItemCollections::new(
                    crate::renderer::item_plugins::plugin_collections_slice(frame, name),
                )
                .is_empty()
            });

        // When skipping the shadow pass (budget pressure or no casters), clear the
        // atlas to max depth so that stale values from a previous frame or a previous
        // showcase don't produce phantom shadows.
        //
        // Only once: the atlas holds a constant after the clear, so repeating it
        // every frame is pure cost. An app that draws no casters at all (an
        // overlay-only consumer, or one whose scene is empty this frame) pays
        // for a 4096-square depth clear it never reads otherwise.
        if lighting.shadows.enabled && (skip_shadows || !has_casters) && !shadow.atlas_cleared {
            shadow.atlas_cleared = true;
            let mut enc = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                label: Some("shadow_clear_encoder"),
            });
            let _ = enc.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("shadow_clear_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: &resources.shadow.map_view,
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
            && has_casters
            && !skip_shadows
            && light.effective_cascade_count > 0
        {
            // Cascades are about to be rasterised into the atlas, so this is
            // where the real texture has to exist: until now it has been a 1x1
            // placeholder. Promoting replaces the texture, so whatever the old
            // one held is gone and the clear flag resets either way.
            resources.ensure_shadow_atlas(device);
            resources.ensure_cascade_shadow_pipelines(device);
            // Casters are about to be drawn into the atlas, so it no longer
            // holds the cleared value and the next empty frame must clear again.
            shadow.atlas_cleared = false;
            // ------------------------------------------------------------------
            // Cascade cull
            //
            // The batches that cast culled (`casts_culled`) draw from per-cascade
            // visible lists. With GPU culling, `cull_instances` +
            // `write_indirect_args` run per cascade with the cascade frustum and
            // land in `shadow_vis_bufs[c]` and `shadow_indirect_bufs[c]`; all
            // cascade dispatches share `batch_counter_buf`, which each
            // `write_indirect_args` resets for the next. Without it the same
            // cull runs on the CPU: visible record indices are compacted into
            // `shadow_vis_bufs[c]` and `cpu_cull_ranges[cascade][batch]` holds
            // the (offset, count) of each batch's run. Either way the draws go
            // through the culled entry points of the cascade pipelines, with
            // each batch's group 1 carrying the cascade's list at binding 15.
            // Every other caster draws its instance runs directly below.
            // ------------------------------------------------------------------
            let mut cpu_cull_ranges: Option<Vec<Vec<(u32, u32)>>> = None;
            let cull_inputs = instancing.use_instancing
                && !instancing.batches.is_empty()
                && instancing.cached_instance_count > 0;
            let gpu_cull = cull_inputs && instancing.gpu_culling_enabled;
            let cpu_cull = cull_inputs
                && !instancing.gpu_culling_enabled
                && instancing.cached_aabbs.len() == instancing.cached_instance_count;
            if gpu_cull || cpu_cull {
                let instance_count = instancing.cached_instance_count as u32;
                let batch_count = instancing.batches.len() as u32;
                instancing
                    .shadow_cull
                    .ensure_outputs(device, instance_count, batch_count);
                // The caster groups bind the instance buffer and each batch's
                // textures, so they drop when the buffer is rebuilt, on a
                // replace (which swaps a view under a live id), and on a free
                // only when a resource the batches name has gone.
                let bgs_stale = shadow_cull_bind_groups_stale(resources, instancing);
                if instancing.shadow_cull.built_gen != instancing.instance_gen || bgs_stale {
                    instancing.shadow_cull.clear_bind_groups();
                    instancing.shadow_cull.built_gen = instancing.instance_gen;
                    instancing.shadow_cull.built_free_epoch = resources.resource_free_epoch;
                    instancing.shadow_cull.built_view_epoch = resources.resource_view_epoch;
                }
                prime_caster_groups(resources, instancing, device, light.effective_cascade_count);
            }
            if gpu_cull {
                let instance_count = instancing.cached_instance_count as u32;
                let batch_count = instancing.batches.len() as u32;
                // `gpu_culling_enabled` is only set once the compute is built.
                if let (Some(aabb_buf), Some(meta_buf), Some(counter_buf), Some(cull)) = (
                    resources.cull.aabb_buf.as_ref(),
                    resources.cull.batch_meta_buf.as_ref(),
                    instancing.shadow_cull.batch_counter_buf.as_ref(),
                    instancing.cull_resources.ready(),
                ) {
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
            } else if cpu_cull {
                let mut ranges: Vec<Vec<(u32, u32)>> =
                    Vec::with_capacity(light.effective_cascade_count);
                let mut indices: Vec<u32> = Vec::with_capacity(instancing.cached_instance_count);
                for c in 0..light.effective_cascade_count {
                    let frustum = crate::camera::frustum::Frustum::from_view_proj(
                        &light.cascade_view_projs[c],
                    );
                    indices.clear();
                    let mut batch_ranges: Vec<(u32, u32)> =
                        Vec::with_capacity(instancing.batches.len());
                    for batch in &instancing.batches {
                        let start = indices.len() as u32;
                        if !batch.casts_culled()
                            || !resources.batch_colour_ready(batch, colour_hdr, clipping_active)
                        {
                            batch_ranges.push((start, 0));
                            continue;
                        }
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
                        #[cfg(any(wgpu29, wgpu30))]
                        multiview_mask: None,
                        label: Some("shadow_pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view: &resources.shadow.map_view,
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

                let use_shadow_indirect = gpu_cull
                    && instancing.shadow_cull.shadow_vis_bufs[0].is_some()
                    && instancing.shadow_cull.shadow_indirect_bufs[0].is_some();
                // Whether the casters `casts_culled` names drew from a visible
                // list above, so the direct loop below leaves them out.
                let culled_drawn = use_shadow_indirect || cpu_cull_ranges.is_some();

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
                        casters_hash,
                        instancing.shadow_cull.groups_gen,
                    );
                    if !shadow_multi_draw && instancing.shadow_cull.bundle_key != Some(bundle_key) {
                        instancing.shadow_cull.shadow_bundles = [None, None, None, None];
                        instancing.shadow_cull.bundle_draws = 0;
                        instancing.shadow_cull.bundle_binds = 0;
                        // A bundle records its pipelines, so one left out
                        // while compiling keeps the bundle from being kept.
                        let mut complete = true;
                        for cascade in 0..light.effective_cascade_count {
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
                            let mut bundle_enc = crate::resources::builders::render_bundle_encoder(
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
                            bundle_enc.set_bind_group(
                                0,
                                &resources.shadow.bind_group,
                                &[cascade as u32 * 256],
                            );
                            bind_deform_group!(
                                bundle_enc,
                                resources,
                                &resources.deform.dummy_bind_group
                            );
                            let mut bound = BoundCaster::default();
                            let mut draws = 0u32;
                            let mut binds = 0u32;
                            for (bi, batch) in instancing.batches.iter().enumerate() {
                                let Some(c) = culled_caster(
                                    resources,
                                    instancing,
                                    batch,
                                    cascade,
                                    colour_hdr,
                                    clipping_active,
                                ) else {
                                    continue;
                                };
                                let Some(pipe) = resources.shadow.cascade_culled(c.key) else {
                                    complete = false;
                                    continue;
                                };
                                if bound.pipeline != Some(c.key) {
                                    bundle_enc.set_pipeline(pipe);
                                    bound.pipeline = Some(c.key);
                                }
                                if bound.group1 != Some(c.group1 as *const _) {
                                    bundle_enc.set_bind_group(1, c.group1, &[]);
                                    bound.group1 = Some(c.group1 as *const _);
                                }
                                if bound.chunks != Some(c.chunks) {
                                    bundle_enc.set_vertex_buffer(
                                        0,
                                        resources.geometry.vertex_chunk_slice(c.chunks.0),
                                    );
                                    bundle_enc.set_index_buffer(
                                        resources.geometry.index_chunk_slice(c.chunks.1),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    binds += 2;
                                    bound.chunks = Some(c.chunks);
                                }
                                bundle_enc
                                    .draw_indexed_indirect(shadow_indirect_buf, bi as u64 * 20);
                                draws += 1;
                            }
                            let bundle = bundle_enc.finish(&crate::gpu::RenderBundleDescriptor {
                                label: Some("shadow_cascade_bundle"),
                            });
                            instancing.shadow_cull.shadow_bundles[cascade] = Some(bundle);
                            instancing.shadow_cull.bundle_draws += draws;
                            instancing.shadow_cull.bundle_binds += binds;
                        }
                        if complete {
                            instancing.shadow_cull.bundle_key = Some(bundle_key);
                        }
                    }

                    if shadow_multi_draw {
                        let counts = draw_shadow_cascades_multi_draw(
                            &mut shadow_pass,
                            resources,
                            instancing,
                            light,
                            tile_px,
                            colour_hdr,
                            clipping_active,
                        );
                        shadow_draws += counts.batch_draws;
                        shadow_draw_cmds += counts.draw_commands;
                        shadow_binds += counts.binds;
                    } else {
                        for cascade in 0..light.effective_cascade_count {
                            set_cascade_tile(&mut shadow_pass, cascade, tile_px, light.tile_size);
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
                } else if let Some(ranges) = cpu_cull_ranges.as_ref() {
                    // CPU-culled path: each batch draws the compacted run
                    // computed above through the culled entry points.
                    for cascade in 0..light.effective_cascade_count {
                        set_cascade_tile(&mut shadow_pass, cascade, tile_px, light.tile_size);
                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow.bind_group,
                            &[cascade as u32 * 256],
                        );
                        bind_deform_group!(
                            shadow_pass,
                            resources,
                            &resources.deform.dummy_bind_group
                        );
                        let mut bound = BoundCaster::default();
                        for (bi, batch) in instancing.batches.iter().enumerate() {
                            let Some(&(start, count)) = ranges[cascade].get(bi) else {
                                continue;
                            };
                            if count == 0 {
                                continue;
                            }
                            let Some(c) = culled_caster(
                                resources,
                                instancing,
                                batch,
                                cascade,
                                colour_hdr,
                                clipping_active,
                            ) else {
                                continue;
                            };
                            // Still compiling: the batch casts next frame.
                            let Some(pipe) = resources.shadow.cascade_culled(c.key) else {
                                continue;
                            };
                            if bound.pipeline != Some(c.key) {
                                shadow_pass.set_pipeline(pipe);
                                bound.pipeline = Some(c.key);
                            }
                            if bound.group1 != Some(c.group1 as *const _) {
                                shadow_pass.set_bind_group(1, c.group1, &[]);
                                bound.group1 = Some(c.group1 as *const _);
                            }
                            if bound.chunks != Some(c.chunks) {
                                shadow_pass.set_vertex_buffer(
                                    0,
                                    resources.geometry.vertex_chunk_slice(c.chunks.0),
                                );
                                shadow_pass.set_index_buffer(
                                    resources.geometry.index_chunk_slice(c.chunks.1),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                shadow_binds += 2;
                                bound.chunks = Some(c.chunks);
                            }
                            shadow_pass.draw_indexed(
                                c.indices.clone(),
                                c.base_vertex,
                                start..start + count,
                            );
                            shadow_draws += 1;
                            shadow_draw_cmds += 1;
                        }
                    }
                }

                // Every other caster: drawn through the cascade pipelines with
                // its own group-1 and deform bind groups, by the runs of its
                // instances that cast and sit in the cascade.
                for cascade in 0..light.effective_cascade_count {
                    set_cascade_tile(&mut shadow_pass, cascade, tile_px, light.tile_size);
                    shadow_pass.set_bind_group(
                        0,
                        &resources.shadow.bind_group,
                        &[cascade as u32 * 256],
                    );

                    let cascade_frustum = crate::camera::frustum::Frustum::from_view_proj(
                        &light.cascade_view_projs[cascade],
                    );
                    let mut bound = BoundCaster::default();

                    for batch in &instancing.batches {
                        if batch.is_transparent
                            || (culled_drawn && batch.casts_culled())
                            || matches!(batch.draw, crate::renderer::types::DrawKind::NormalLines)
                            || !resources.batch_colour_ready(batch, colour_hdr, clipping_active)
                        {
                            continue;
                        }
                        let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                            continue;
                        };
                        let Some(group1) = per_object
                            .material_bind_groups
                            .get(&batch.shadow_bind_key)
                            .map(|m| &m.bind_group)
                        else {
                            continue;
                        };
                        // The instances that cast and sit in this cascade,
                        // as contiguous runs.
                        let runs =
                            caster_runs(instancing, batch, |aabb| !cascade_frustum.cull_aabb(aabb));
                        if runs.is_empty() {
                            continue;
                        }
                        let key = caster_key(resources, batch, mesh);
                        // Still compiling: the caster waits a frame.
                        let Some(pl) = resources.shadow.cascade(key) else {
                            continue;
                        };
                        if bound.pipeline != Some(key) {
                            shadow_pass.set_pipeline(pl);
                            bound.pipeline = Some(key);
                        }
                        if bound.group1 != Some(group1 as *const _) {
                            shadow_pass.set_bind_group(1, group1, &[]);
                            bound.group1 = Some(group1 as *const _);
                        }
                        let group2 = resources
                            .deform
                            .instance_bind_group_for(batch.mesh_id, batch.deform_instance);
                        if bound.group2 != Some(group2 as *const _) {
                            bind_deform_group!(shadow_pass, resources, group2);
                            bound.group2 = Some(group2 as *const _);
                        }
                        let geometry = CasterGeometry::of(resources, batch, mesh);
                        shadow_binds +=
                            geometry.bind(&mut shadow_pass, resources, mesh, &mut bound);
                        for run in &runs {
                            shadow_pass.draw_indexed(
                                geometry.indices.clone(),
                                geometry.base_vertex,
                                run.clone(),
                            );
                            shadow_draws += 1;
                            shadow_draw_cmds += 1;
                        }
                    }
                }

                // Item-type plugin shadow casting: per-cascade dispatch
                // with viewport + scissor + cascade bind group set up by
                // the lib. The plugin map is read through a raw pointer
                // captured before the mutable resources borrow split off.
                if !plugins.is_empty() {
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
                            &resources.shadow.bind_group,
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
                            let items = crate::plugin_api::ItemCollections::new(
                                crate::renderer::item_plugins::plugin_collections_slice(
                                    frame, name,
                                ),
                            );
                            if !items.is_empty() {
                                plugin.cast_shadow_pass(&mut shadow_pass, &ctx, &items);
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
        // One depth-only render pass per (slot, face). Opaque casters only,
        // writing linear distance-to-light to frag_depth. Every caster draws
        // through `shadow_point.wgsl` from its records, culled per face
        // against the face frustum. Masked casters use the cutout variant.
        // ----------------------------------------------------------------
        if lighting.shadows.enabled
            && !scene_items.is_empty()
            && !light.point_shadow_faces.is_empty()
            && resources.shadow.point_pipeline.is_some()
        {
            // The batches that cast, each with its own bind group, drawn by
            // instance run.
            struct PointCaster<'a> {
                batch: &'a InstancedBatch,
                mesh: &'a crate::resources::GpuMesh,
                group1: &'a crate::gpu::BindGroup,
                cutout: bool,
            }
            let casters: Vec<PointCaster> = instancing
                .batches
                .iter()
                .filter(|b| {
                    !b.is_transparent
                        && resources.batch_colour_ready(b, colour_hdr, clipping_active)
                })
                .filter_map(|b| {
                    if matches!(b.draw, crate::renderer::types::DrawKind::NormalLines) {
                        return None;
                    }
                    let mesh = resources.mesh_store.get(b.mesh_id)?;
                    let group1 = per_object
                        .material_bind_groups
                        .get(&b.shadow_bind_key)
                        .map(|m| &m.bind_group)?;
                    Some(PointCaster {
                        batch: b,
                        mesh,
                        group1,
                        cutout: b.has_alpha_mask
                            || resources.deform.may_discard(b.mesh_id, b.deform_instance),
                    })
                })
                .collect();

            // The slot hashes below record what was rendered, so while any
            // pipeline these casters need is still on a worker the whole block
            // waits rather than marking a cubemap up to date that was never
            // fully drawn.
            let mut ready = resources.shadow.point(false).is_some();
            if casters.iter().any(|c| c.cutout) {
                ready &= resources.shadow.point(true).is_some();
            }

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
            for fc in light.point_shadow_faces.iter().filter(|_| ready) {
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
                // Every batch contributes the identity of its
                // mesh and the hash of its instance records (which carry each
                // caster's model matrix) when any of its instances is in
                // range. A deform instance or an override animates without
                // changing any hashed input, so such a batch keeps its slot
                // re-rendering every frame.
                for (bi, b) in instancing.batches.iter().enumerate() {
                    if b.is_transparent {
                        continue;
                    }
                    let Some(mesh) = resources.mesh_store.get(b.mesh_id) else {
                        continue;
                    };
                    let start = b.instance_offset as usize;
                    let end = start + b.instance_count as usize;
                    let in_range = instancing.cached_aabbs[start.min(instancing.cached_aabbs.len())
                        ..end.min(instancing.cached_aabbs.len())]
                        .iter()
                        .any(|a| {
                            a.cast_shadows != 0
                                && closest_sq(
                                    &crate::scene::aabb::Aabb {
                                        min: a.min.into(),
                                        max: a.max.into(),
                                    },
                                    fc.light_pos,
                                ) <= range_sq
                        });
                    if !in_range {
                        continue;
                    }
                    if b.deform_instance.is_some()
                        || mesh.position_override_buffer.is_some()
                        || mesh.normal_override_buffer.is_some()
                    {
                        cacheable = false;
                        break;
                    }
                    mix(&(b.mesh_id.index() as u64).to_le_bytes());
                    mix(&b.mesh_id.generation().to_le_bytes());
                    mix(&mesh.content_rev.to_le_bytes());
                    mix(&instancing
                        .cached_instance_hashes
                        .get(bi)
                        .copied()
                        .unwrap_or(0)
                        .to_le_bytes());
                    // A mask cuts the cubemap, so its albedo counts (the
                    // cutoff rides the hashed records).
                    if b.is_cutout {
                        let tex = b.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX);
                        mix(&tex.to_le_bytes());
                    }
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

            let rendered: Vec<&super::PointShadowFace> = light
                .point_shadow_faces
                .iter()
                .filter(|fc| ready && slot_dirty.get(&fc.slot).copied().unwrap_or(true))
                .collect();
            if !rendered.is_empty() {
                let mut enc =
                    device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                        label: Some("point_shadow_encoder"),
                    });
                let last_face = rendered.len() - 1;
                for (face_idx, fc) in rendered.iter().enumerate() {
                    let layer = fc.slot * 6 + fc.face;
                    let view = &resources.shadow.point_face_views[layer as usize];
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
                        #[cfg(any(wgpu29, wgpu30))]
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
                    let dyn_offset = layer * POINT_FACE_STRIDE as u32;
                    pass.set_bind_group(0, &resources.shadow.point_face_bind_group, &[dyn_offset]);

                    let face_frustum =
                        crate::camera::frustum::Frustum::from_view_proj(&fc.view_proj);

                    let mut bound_cutout: Option<bool> = None;
                    let mut bound = BoundCaster::default();
                    for c in casters.iter() {
                        let runs =
                            caster_runs(instancing, c.batch, |aabb| !face_frustum.cull_aabb(aabb));
                        if runs.is_empty() {
                            continue;
                        }
                        if bound_cutout != Some(c.cutout) {
                            let Some(pl) = resources.shadow.point(c.cutout) else {
                                continue;
                            };
                            pass.set_pipeline(pl);
                            bound_cutout = Some(c.cutout);
                        }
                        pass.set_bind_group(1, c.group1, &[]);
                        bind_deform_group!(
                            pass,
                            resources,
                            resources
                                .deform
                                .instance_bind_group_for(c.batch.mesh_id, c.batch.deform_instance)
                        );
                        let geometry = CasterGeometry::of(resources, c.batch, c.mesh);
                        geometry.bind(&mut pass, resources, c.mesh, &mut bound);
                        for run in &runs {
                            pass.draw_indexed(
                                geometry.indices.clone(),
                                geometry.base_vertex,
                                run.clone(),
                            );
                        }
                    }
                }
                sink.push(enc.finish());
            }
        }

        if shadow_instrument && lighting.shadows.enabled {
            // CPU encode time only. The pass's GPU time is in the frame's GPU
            // timestamps; waiting on the device here to measure it would stall
            // every frame for anyone with a permissive subscriber.
            tracing::debug!(
                target: "viewport_lib::shadow",
                ms = shadow_start.elapsed().as_secs_f32() * 1000.0,
                cascades = light.effective_cascade_count,
                atlas = resources.shadow.atlas_size,
                draws = last_stats.shadow_draw_calls,
                point_faces = light.point_shadow_faces.len(),
                "shadow pass encoded"
            );
        }
    }
}

/// Draw the directional shadow cascades live in the pass with per-run
/// `multi_draw_indexed_indirect`, the backend-native alternative to the cached
/// render bundle (which cannot host a multi-draw). Geometry is bound once per
/// slab chunk and the GPU-cull-written args carry each mesh's base_vertex /
/// first_index; a run collapses consecutive batches that share the pipeline,
/// the group-1 bind group, and the slab chunk, breaking on a global-index gap
/// so a multi-draw never sweeps in a batch this loop skipped. Returns the
/// number of batch draws issued (pre-collapse, for stats parity with the
/// bundle path).
fn draw_shadow_cascades_multi_draw(
    pass: &mut crate::gpu::RenderPass<'_>,
    resources: &DeviceResources,
    instancing: &InstancingState,
    light: &LightingFrame,
    tile_px: f32,
    colour_hdr: bool,
    clipping_active: bool,
) -> ShadowDrawCounts {
    let multi_draw = instancing.multi_draw_active();
    let mut drawn = 0u32;
    let mut binds = 0u32;
    let mut draw_cmds = 0u32;

    for cascade in 0..light.effective_cascade_count {
        set_cascade_tile(pass, cascade, tile_px, light.tile_size);
        let Some(shadow_indirect_buf) =
            instancing.shadow_cull.shadow_indirect_bufs[cascade].as_ref()
        else {
            continue;
        };
        pass.set_bind_group(0, &resources.shadow.bind_group, &[cascade as u32 * 256]);
        bind_deform_group!(pass, resources, &resources.deform.dummy_bind_group);

        let mut bound = BoundCaster::default();
        let mut run_start: u64 = 0;
        let mut run_len: u32 = 0;

        for (bi, batch) in instancing.batches.iter().enumerate() {
            let caster = culled_caster(
                resources,
                instancing,
                batch,
                cascade,
                colour_hdr,
                clipping_active,
            );
            // Still compiling (or not a culled caster): checked before the run
            // accounting so a skipped batch breaks the run like a gap.
            let Some((c, pipe)) =
                caster.and_then(|c| resources.shadow.cascade_culled(c.key).map(|p| (c, p)))
            else {
                if run_len > 0 {
                    draw_cmds += crate::renderer::render::emit_indirect_run(
                        pass,
                        shadow_indirect_buf,
                        run_start,
                        run_len,
                        multi_draw,
                    );
                    run_len = 0;
                }
                continue;
            };
            let g = bi as u64;
            if run_len > 0
                && g == run_start + run_len as u64
                && bound.pipeline == Some(c.key)
                && bound.group1 == Some(c.group1 as *const _)
                && bound.chunks == Some(c.chunks)
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
            if bound.pipeline != Some(c.key) {
                pass.set_pipeline(pipe);
                bound.pipeline = Some(c.key);
            }
            if bound.group1 != Some(c.group1 as *const _) {
                pass.set_bind_group(1, c.group1, &[]);
                bound.group1 = Some(c.group1 as *const _);
            }
            if bound.chunks != Some(c.chunks) {
                pass.set_vertex_buffer(0, resources.geometry.vertex_chunk_slice(c.chunks.0));
                pass.set_index_buffer(
                    resources.geometry.index_chunk_slice(c.chunks.1),
                    crate::gpu::IndexFormat::Uint32,
                );
                binds += 2;
                bound.chunks = Some(c.chunks);
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

/// Point the pass at `cascade`'s tile of the shadow atlas.
fn set_cascade_tile(
    pass: &mut crate::gpu::RenderPass<'_>,
    cascade: usize,
    tile_px: f32,
    tile: u32,
) {
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
        tile,
        tile,
    );
}

/// Build each culled caster's group 1 for every cascade: its per-batch member
/// with the cascade's visible list at binding 15, keyed by its
/// `shadow_bind_key`. Groups no batch names any more are dropped.
fn prime_caster_groups(
    resources: &DeviceResources,
    instancing: &mut InstancingState,
    device: &crate::gpu::Device,
    cascades: usize,
) {
    let Some(records) = resources.instancing.storage_buf.as_ref() else {
        return;
    };
    let batches = &instancing.batches;
    let state = &mut instancing.shadow_cull;
    let mut live = std::collections::HashSet::new();
    for batch in batches {
        if batch.is_transparent || !batch.casts_culled() {
            continue;
        }
        live.insert(batch.shadow_bind_key);
        for c in 0..cascades {
            if state.object_bgs[c].contains_key(&batch.shadow_bind_key) {
                continue;
            }
            let Some(visible) = state.shadow_vis_bufs[c].as_ref() else {
                continue;
            };
            let g = &batch.group;
            if let Some((bg, _)) = resources.build_per_item_object_bind_group(
                device,
                batch.mesh_id,
                records,
                g.textures[0],
                g.textures[1],
                g.textures[2],
                g.lut,
                g.matcap,
                g.textures[3],
                g.textures[4],
                g.sampler,
                None,
                false,
                Some(visible),
            ) {
                state.object_bgs[c].insert(batch.shadow_bind_key, bg);
                state.groups_gen = state.groups_gen.wrapping_add(1);
            }
        }
    }
    for m in &mut state.object_bgs {
        m.retain(|k, _| live.contains(k));
    }
}

/// The cascade pipeline key of a caster. Two-sided materials on open surfaces
/// cast through the cull-none pipeline so both faces rasterise; a closed mesh
/// keeps the cull-front pipeline whatever its policy. A masked material (or a
/// deformer that can cut) casts through the cutout pipeline.
fn caster_key(
    resources: &DeviceResources,
    batch: &InstancedBatch,
    mesh: &crate::resources::GpuMesh,
) -> PipelineKey {
    PipelineKey {
        two_sided: batch.two_sided && !mesh.closed,
        cutout: batch.has_alpha_mask
            || resources
                .deform
                .may_discard(batch.mesh_id, batch.deform_instance),
        ..PipelineKey::default()
    }
}

/// What a culled caster draws with in one cascade.
struct CulledCaster<'a> {
    key: PipelineKey,
    group1: &'a crate::gpu::BindGroup,
    chunks: (u32, u32),
    indices: std::ops::Range<u32>,
    base_vertex: i32,
}

/// `batch` as a culled caster in `cascade`, or `None` when it does not cast
/// from the visible lists this frame.
fn culled_caster<'a>(
    resources: &'a DeviceResources,
    instancing: &'a InstancingState,
    batch: &InstancedBatch,
    cascade: usize,
    colour_hdr: bool,
    clipping_active: bool,
) -> Option<CulledCaster<'a>> {
    if batch.is_transparent
        || !batch.casts_culled()
        || !resources.batch_colour_ready(batch, colour_hdr, clipping_active)
    {
        return None;
    }
    let mesh = resources.mesh_store.get(batch.mesh_id)?;
    let group1 = instancing.shadow_cull.object_bgs[cascade].get(&batch.shadow_bind_key)?;
    let geometry = CasterGeometry::of(resources, batch, mesh);
    Some(CulledCaster {
        key: caster_key(resources, batch, mesh),
        group1,
        chunks: (mesh.vertex_span.chunk, mesh.index_span.chunk),
        indices: geometry.indices,
        base_vertex: geometry.base_vertex,
    })
}

/// The geometry a caster binds and the index range it draws. A caster binds
/// what its colour draw binds, since its records' `vertex_base` is that
/// draw's base vertex: the whole slab chunk at the mesh's base, or, for a
/// per-face draw (whose records carry base 0), the mesh's own sub-slice.
struct CasterGeometry {
    whole_chunk: bool,
    indices: std::ops::Range<u32>,
    base_vertex: i32,
}

impl CasterGeometry {
    fn of(
        resources: &DeviceResources,
        batch: &InstancedBatch,
        mesh: &crate::resources::GpuMesh,
    ) -> Self {
        let local = match batch.draw {
            crate::renderer::types::DrawKind::Range(first, count) => first..first + count,
            _ => 0..mesh.index_count,
        };
        if matches!(batch.draw, crate::renderer::types::DrawKind::FaceVertices) {
            return Self {
                whole_chunk: false,
                indices: local,
                base_vertex: 0,
            };
        }
        let first = resources.geometry.first_index(mesh.index_span);
        Self {
            whole_chunk: true,
            indices: first + local.start..first + local.end,
            base_vertex: resources.geometry.base_vertex(mesh.vertex_span),
        }
    }

    /// Bind the geometry unless `bound` already has it. Returns the buffer
    /// binds issued.
    fn bind(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        resources: &DeviceResources,
        mesh: &crate::resources::GpuMesh,
        bound: &mut BoundCaster,
    ) -> u32 {
        let id = (mesh.vertex_span.chunk, mesh.index_span.chunk);
        if self.whole_chunk {
            if bound.chunks == Some(id) {
                return 0;
            }
            pass.set_vertex_buffer(0, resources.geometry.vertex_chunk_slice(id.0));
            pass.set_index_buffer(
                resources.geometry.index_chunk_slice(id.1),
                crate::gpu::IndexFormat::Uint32,
            );
            bound.chunks = Some(id);
            bound.slice = None;
        } else {
            if bound.slice == Some(mesh as *const _) {
                return 0;
            }
            pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
            pass.set_index_buffer(
                resources.geometry.index_slice(mesh.index_span),
                crate::gpu::IndexFormat::Uint32,
            );
            bound.slice = Some(mesh as *const _);
            bound.chunks = None;
        }
        2
    }
}

/// The contiguous runs of a batch's instances whose item casts shadows and
/// whose world AABB passes `visible`, in instance order (the cull inputs hold
/// one AABB per record, in record order).
fn caster_runs(
    instancing: &InstancingState,
    batch: &InstancedBatch,
    visible: impl Fn(&crate::scene::aabb::Aabb) -> bool,
) -> Vec<std::ops::Range<u32>> {
    let start = batch.instance_offset;
    let end = batch.instance_offset + batch.instance_count;
    let mut runs = Vec::new();
    let mut run_start = None;
    for i in start..end {
        let casts = instancing.cached_aabbs.get(i as usize).is_none_or(|a| {
            a.cast_shadows != 0
                && visible(&crate::scene::aabb::Aabb {
                    min: a.min.into(),
                    max: a.max.into(),
                })
        });
        match (casts, run_start) {
            (true, None) => run_start = Some(i),
            (false, Some(r)) => {
                runs.push(r..i);
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(r) = run_start {
        runs.push(r..end);
    }
    runs
}

/// What a per-item caster loop last bound, so a run of casters that share a
/// pipeline, object bind group, deform group and slab chunk binds them once
/// and draws each from the bound chunk with its own base vertex and first
/// index.
#[derive(Default)]
struct BoundCaster {
    pipeline: Option<PipelineKey>,
    group1: Option<*const crate::gpu::BindGroup>,
    group2: Option<*const crate::gpu::BindGroup>,
    chunks: Option<(u32, u32)>,
    /// The mesh whose own sub-slice is bound, for per-face casters.
    slice: Option<*const crate::resources::GpuMesh>,
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
