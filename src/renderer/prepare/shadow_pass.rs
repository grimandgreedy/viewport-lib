//! Shadow depth pass: directional cascades and point-light cube-map faces.

use super::*;

impl ViewportRenderer {
    /// Render the shadow depth pass: directional CSM cascades into the atlas
    /// tiles and point-light cube-map faces, including per-cascade plugin draws.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_shadow_pass(
        resources: &mut ViewportGpuResources,
        instancing: &mut InstancingState,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
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
        if lighting.shadows_enabled && (skip_shadows || scene_items.is_empty()) {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_clear_encoder"),
            });
            let _ = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_clear_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &resources.shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            queue.submit(std::iter::once(enc.finish()));
        }

        if lighting.shadows_enabled && !scene_items.is_empty() && !skip_shadows {
            // ------------------------------------------------------------------
            // Shadow GPU cull dispatch
            //
            // For each active cascade, dispatch `cull_instances` + `write_indirect_args`
            // with the cascade frustum. Results land in `shadow_vis_bufs[c]` and
            // `shadow_indirect_bufs[c]`, consumed by the shadow render pass below.
            // All cascade dispatches share the same `batch_counter_buf`; each
            // `write_indirect_args` dispatch resets the counters for the next cascade.
            // ------------------------------------------------------------------
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
                for c in 0..light.effective_cascade_count {
                    resources.get_shadow_cull_instance_bind_group(device, c);
                }

                let instance_count = instancing.cached_instance_count as u32;
                let batch_count = instancing.batches.len() as u32;

                if let (Some(aabb_buf), Some(meta_buf), Some(counter_buf)) = (
                    resources.instance_aabb_buf.as_ref(),
                    resources.batch_meta_buf.as_ref(),
                    resources.batch_counter_buf.as_ref(),
                ) {
                    let cull = instancing.cull_resources.as_ref().unwrap();
                    let mut shadow_cull_encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("shadow_cull_encoder"),
                        });
                    for c in 0..light.effective_cascade_count {
                        if let (Some(shadow_vis_buf), Some(shadow_indirect_buf)) = (
                            resources.shadow_vis_bufs[c].as_ref(),
                            resources.shadow_indirect_bufs[c].as_ref(),
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
                            );
                        }
                    }
                    queue.submit(std::iter::once(shadow_cull_encoder.finish()));
                }
            }

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_pass_encoder"),
            });
            {
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &resources.shadow_map_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                let mut shadow_draws = 0u32;
                let tile_px = light.tile_size as f32;

                if instancing.use_instancing {
                    let use_shadow_indirect = instancing.gpu_culling_enabled
                        && resources.shadow_instanced_cull_pipeline.is_some()
                        && resources.shadow_vis_bufs[0].is_some();

                    if use_shadow_indirect {
                        // GPU-culled indirect shadow path.
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
                                resources.shadow_instanced_cascade_bufs[cascade]
                                    .as_ref()
                                    .expect("shadow_instanced_cascade_bufs not allocated"),
                                0,
                                bytemuck::cast_slice(
                                    &light.cascade_view_projs[cascade].to_cols_array_2d(),
                                ),
                            );

                            let Some(pipeline) = resources.shadow_instanced_cull_pipeline.as_ref()
                            else {
                                continue;
                            };
                            let Some(cascade_bg) =
                                resources.shadow_instanced_cascade_bgs[cascade].as_ref()
                            else {
                                continue;
                            };
                            let Some(inst_cull_bg) =
                                resources.shadow_cull_instance_bgs[cascade].as_ref()
                            else {
                                continue;
                            };
                            let Some(shadow_indirect_buf) =
                                resources.shadow_indirect_bufs[cascade].as_ref()
                            else {
                                continue;
                            };

                            shadow_pass.set_pipeline(pipeline);
                            shadow_pass.set_bind_group(0, cascade_bg, &[]);
                            shadow_pass.set_bind_group(1, inst_cull_bg, &[]);

                            for (bi, batch) in instancing.batches.iter().enumerate() {
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                shadow_pass
                                    .draw_indexed_indirect(shadow_indirect_buf, bi as u64 * 20);
                                shadow_draws += 1;
                            }
                        }
                    } else if let (Some(pipeline), Some(instance_bg)) = (
                        &resources.shadow_instanced_pipeline,
                        instancing.batches.first().and_then(|b| {
                            resources.instance_bind_groups.get(&(
                                b.texture_id.unwrap_or(u64::MAX),
                                b.normal_map_id.unwrap_or(u64::MAX),
                                b.ao_map_id.unwrap_or(u64::MAX),
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

                            shadow_pass.set_pipeline(pipeline);

                            queue.write_buffer(
                                resources.shadow_instanced_cascade_bufs[cascade]
                                    .as_ref()
                                    .expect("shadow_instanced_cascade_bufs not allocated"),
                                0,
                                bytemuck::cast_slice(
                                    &light.cascade_view_projs[cascade].to_cols_array_2d(),
                                ),
                            );

                            let cascade_bg = resources.shadow_instanced_cascade_bgs[cascade]
                                .as_ref()
                                .expect("shadow_instanced_cascade_bgs not allocated");
                            shadow_pass.set_bind_group(0, cascade_bg, &[]);
                            shadow_pass.set_bind_group(1, instance_bg, &[]);

                            for batch in &instancing.batches {
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                shadow_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                                shadow_draws += 1;
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
                            // drawn again here.
                            let in_instanced_batch = item.active_attribute.is_none()
                                && !item.material.is_two_sided()
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
                            shadow_pass.set_bind_group(
                                2,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance),
                                &[],
                            );
                            shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            shadow_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            shadow_draws += 1;
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
                            shadow_pass.set_bind_group(
                                2,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance),
                                &[],
                            );
                            shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            shadow_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            shadow_draws += 1;
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
            }
            queue.submit(std::iter::once(encoder.finish()));
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
        if lighting.shadows_enabled
            && !scene_items.is_empty()
            && !light.point_shadow_faces.is_empty()
        {
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
            // priority bug fix.
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
                // Only the model matrix (offset 0, 64 bytes) is read by
                // `shadow_point.wgsl`. Write just that prefix so we don't
                // clobber the rest of the per-mesh `ObjectUniform`.
                queue.write_buffer(
                    &mesh.object_uniform_buf,
                    0,
                    bytemuck::cast_slice(&item.model),
                );
            }

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("point_shadow_encoder"),
            });
            for fc in &light.point_shadow_faces {
                let layer = fc.slot * 6 + fc.face;
                let view = &resources.point_shadow_face_views[layer as usize];
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("point_shadow_face_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&resources.shadow_point_pipeline);
                let dyn_offset = layer * POINT_FACE_STRIDE as u32;
                pass.set_bind_group(0, &resources.shadow_point_face_bind_group, &[dyn_offset]);

                let face_frustum = crate::camera::frustum::Frustum::from_view_proj(&fc.view_proj);

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
                    let world_aabb = mesh
                        .aabb
                        .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                    if face_frustum.cull_aabb(&world_aabb) {
                        continue;
                    }
                    pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                    pass.set_bind_group(
                        2,
                        resources
                            .deform
                            .instance_bind_group_for(item.mesh_id, item.deform_instance),
                        &[],
                    );
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
            queue.submit(std::iter::once(enc.finish()));
        }

        if shadow_instrument && lighting.shadows_enabled {
            // Force the just-submitted shadow work to finish so the measured time
            // reflects shadow GPU execution rather than landing later at present.
            // Other work submitted before this point in prepare is minor, so this
            // is a good attribution of the shadow cost.
            device
                .poll(wgpu::PollType::Wait {
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
