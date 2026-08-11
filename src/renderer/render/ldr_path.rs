//! The LDR render path: a single scene pass with optional dynamic-resolution
//! intermediate and backdrop-blur compositing. Used when post-processing is
//! disabled. Builds its own command encoder and returns the finished buffer.

use super::*;

impl ViewportRenderer {
    pub(crate) fn render_frame_ldr(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        vp_idx: usize,
        frame: &FrameData,
        scene_items: &[SceneRenderItem],
        bg_colour: [f32; 4],
        w: u32,
        h: u32,
    ) -> crate::gpu::CommandBuffer {
        // LDR fallback. When dynamic resolution is active and render_scale < 1.0,
        // draw into a scaled intermediate texture and upscale-blit to output_view.
        // Otherwise render directly to output_view.
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("ldr_encoder"),
        });

        let use_dyn_res = self.current_render_scale < 1.0 - 0.001;
        let needs_blur = self.has_backdrop_blur_shapes();

        if use_dyn_res {
            let sw = ((w as f32 * self.current_render_scale) as u32).max(1);
            let sh = ((h as f32 * self.current_render_scale) as u32).max(1);
            self.ensure_dyn_res_target(device, vp_idx, [sw, sh], [w.max(1), h.max(1)]);
        }

        // When blur backdrops are needed and we'd otherwise render directly
        // to the surface (no dyn_res), force an intermediate so it can be
        // sampled for the blur passes.
        if needs_blur && !use_dyn_res {
            self.ensure_backdrop_blur_state(device, w.max(1), h.max(1));
        }

        // When occlusion culling is on, keep the scene depth (instead of
        // discarding it) so it can be copied into the HiZ prev-depth target for
        // next frame's reprojection. The copy happens after the scene pass below.
        let store_scene_depth = self.resources.occlusion_culling_enabled();

        {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot
                .hdr
                .as_ref()
                .expect("HDR state missing in LDR path; ensure_viewport_hdr must have been called");
            let camera_bg = &slot.camera_bind_group;
            let grid_bg = &slot.grid_bind_group;
            // Choose render target: dyn_res intermediate, backdrop intermediate, or output_view.
            let (scene_colour_view, scene_depth_view): (
                &crate::gpu::TextureView,
                &crate::gpu::TextureView,
            ) = if use_dyn_res {
                let dr = slot.dyn_res.as_ref().unwrap();
                (&dr.colour_view, &dr.depth_view)
            } else if needs_blur {
                let bs = self.backdrop_blur_state.as_ref().unwrap();
                (&bs.intermediate_view, &slot_hdr.outline_depth_view)
            } else {
                (output_view, &slot_hdr.outline_depth_view)
            };
            let ts_writes = self.ts_query_set.as_ref().map(|qs| {
                self.ts_written_mask.fetch_or(
                    1 << crate::renderer::GPU_TS_SCENE,
                    std::sync::atomic::Ordering::Relaxed,
                );
                crate::gpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(crate::renderer::GPU_TS_SCENE * 2),
                    end_of_pass_write_index: Some(crate::renderer::GPU_TS_SCENE * 2 + 1),
                }
            });
            let mut render_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("ldr_render_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: scene_colour_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color {
                            r: bg_colour[0] as f64,
                            g: bg_colour[1] as f64,
                            b: bg_colour[2] as f64,
                            a: bg_colour[3] as f64,
                        }),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: scene_depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(1.0),
                        store: if store_scene_depth {
                            crate::gpu::StoreOp::Store
                        } else {
                            crate::gpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: ts_writes,
                occlusion_query_set: None,
            });
            emit_draw_calls!(
                &self.resources,
                &mut render_pass,
                frame,
                self.instancing.use_instancing,
                &self.instancing.batches,
                camera_bg,
                grid_bg,
                &self.compute_filter_results,
                Some(slot),
                &self.mesh_uniforms.wireframe_bind_groups,
                &self.mesh_uniforms.bind_groups,
                &self.mesh_uniforms.submesh_bind_groups,
                scene_items,
                // The cached bundle was recorded from `prepared_surfaces` (the
                // surface submission, with the opaque volume-mesh boundaries
                // truncated off). When those boundaries are appended to
                // `scene_items` the bundle no longer covers the full list, so
                // fall back to the per-item draw path that walks every item.
                if scene_items.len() == self.prepared_surfaces.len() {
                    self.per_object_bundle.as_ref()
                } else {
                    None
                }
            );
            emit_scivis_draw_calls!(
                &self.resources,
                &mut render_pass,
                &self.point_cloud_gpu_data,
                &self.glyph_gpu_data,
                &self.polyline_gpu_data,
                &self.volume_gpu_data,
                &self.streamtube_gpu_data,
                camera_bg,
                &self.tube_gpu_data,
                &self.image_slice_gpu_data,
                &self.tensor_glyph_gpu_data,
                &self.ribbon_gpu_data,
                &self.volume_surface_slice_gpu_data,
                &self.sprite_gpu_data,
                &self.mesh_instance_gpu_data,
                false
            );
            // Gaussian splats (alpha-blended, back-to-front sorted, no depth
            // write). Mirrors the block in `paint_to` and the HDR path so the
            // offscreen LDR path draws splats too.
            if !self.gaussian_splat_draw_data.is_empty() {
                if let Some(ref dual) = self.resources.gaussian_splat.pipeline {
                    render_pass.set_pipeline(dual.for_format(false));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for dd in &self.gaussian_splat_draw_data {
                        if dd.wireframe {
                            continue;
                        }
                        if let Some(set) = self
                            .resources
                            .content
                            .gaussian_splat_store
                            .get_by_index(dd.store_index)
                        {
                            if let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) {
                                render_pass.set_bind_group(1, &vp_sort.render_bg, &[]);
                                render_pass.draw(0..6, 0..dd.count);
                            }
                        }
                    }
                }
            }
            // TransparentVolumeMesh boundary wireframe overlay.
            if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
                if let Some(ref tvm_bg) = self.mesh_uniforms.tvm_wireframe_bg {
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for mesh_id in &self.mesh_uniforms.tvm_wireframe_draws {
                        if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                            render_pass.set_pipeline(&self.resources.wireframe_pipeline);
                            bind_deform_group!(
                                render_pass,
                                self.resources,
                                &self.resources.deform.dummy_bind_group
                            );
                            render_pass.set_bind_group(1, tvm_bg, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            if let Some(edge_buf) = &mesh.edge_index_buffer {
                                render_pass.set_index_buffer(
                                    edge_buf.slice(..),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            }
                        }
                    }
                }
            }
            // GPU implicit surface.
            if !self.implicit_gpu_data.is_empty() {
                if let Some(ref dual) = self.resources.implicit.pipeline {
                    render_pass.set_pipeline(dual.for_format(false));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for gpu in &self.implicit_gpu_data {
                        render_pass.set_bind_group(1, &gpu.bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
            // GPU marching cubes indirect draw.
            if !self.mc_gpu_data.is_empty() {
                if let Some(ref dual) = self.resources.mc.surface_pipeline {
                    render_pass.set_pipeline(dual.for_format(false));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for mc in &self.mc_gpu_data {
                        let vol = &self.resources.mc.volumes[mc.volume_idx];
                        render_pass.set_bind_group(1, &mc.render_bg, &[]);
                        for slab in &vol.slabs {
                            render_pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                            render_pass.draw_indirect(&slab.indirect_buf, 0);
                        }
                    }
                }
            }
            // Outline composite after all scene content.
            emit_outline_composite!(&self.resources, &mut render_pass, Some(slot));
            // Screen-space image overlays.
            // Regular items drawn with depth_compare: Always (always on top).
            // Depth-composite items drawn with depth_compare: LessEqual (occluded by
            // scene geometry whose depth was already written to the depth attachment).
            if !self.screen_image_gpu_data.is_empty() {
                if let Some(overlay_pipeline) = &self.resources.screen_image.pipeline {
                    let dc_pipeline = self.resources.screen_image.dc_pipeline.as_ref();
                    for gpu in &self.screen_image_gpu_data {
                        if let (Some(dc_bg), Some(dc_pipe)) = (&gpu.depth_bind_group, dc_pipeline) {
                            render_pass.set_pipeline(dc_pipe);
                            render_pass.set_bind_group(0, dc_bg, &[]);
                        } else {
                            render_pass.set_pipeline(overlay_pipeline);
                            render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                        }
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
            // When blur backdrops are needed, skip overlays here. They'll
            // be drawn in a second pass after the blur is applied.
            if !needs_blur {
                emit_overlay_2d!(self, render_pass);
            }
        }
        // -- End of scene render pass (dropped above). ---

        // Foreground pass: draw foreground items over the finished scene
        // against a freshly cleared own depth target, into the same colour
        // target the scene pass used. LDR has no post chain, so no coverage
        // mask is needed here. Note the scene pass draws 2D overlays inline
        // (above), so on the LDR path foreground items paint over them.
        // Item-type plugins are not dispatched on the LDR path (their
        // foreground pipelines target the HDR formats), so only the item
        // list gates this pass.
        if !frame.scene.foreground_items.is_empty() {
            let (fw, fh) = if use_dyn_res {
                let dr = self.viewport_slots[vp_idx].dyn_res.as_ref().unwrap();
                (dr.scaled_size[0], dr.scaled_size[1])
            } else {
                (w.max(1), h.max(1))
            };
            {
                let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
                self.resources
                    .ensure_viewport_foreground_depth(device, hdr, fw, fh);
            }

            let resources = &self.resources;
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let scene_colour_view: &crate::gpu::TextureView = if use_dyn_res {
                &slot.dyn_res.as_ref().unwrap().colour_view
            } else if needs_blur {
                &self.backdrop_blur_state.as_ref().unwrap().intermediate_view
            } else {
                output_view
            };
            if let Some(fg_depth_view) = slot_hdr.foreground_depth_view.as_ref() {
                let fg_camera = frame
                    .camera
                    .render_camera
                    .foreground_camera(frame.effects.foreground.as_ref());
                let eye = glam::Vec3::from(fg_camera.eye_position);
                let dist_from_eye = |item: &SceneRenderItem| -> f32 {
                    let pos = glam::Vec3::new(item.model[3][0], item.model[3][1], item.model[3][2]);
                    (pos - eye).length()
                };
                let items = &frame.scene.foreground_items;
                let mut opaque: Vec<(usize, &SceneRenderItem)> = Vec::new();
                let mut transparent: Vec<(usize, &SceneRenderItem)> = Vec::new();
                for (idx, item) in items.iter().enumerate() {
                    if item.settings.hidden || resources.mesh_store.get(item.mesh_id).is_none() {
                        continue;
                    }
                    if item.settings.opacity < 1.0 || item.material.is_blend() {
                        transparent.push((idx, item));
                    } else {
                        opaque.push((idx, item));
                    }
                }
                opaque.sort_by(|a, b| {
                    dist_from_eye(a.1)
                        .partial_cmp(&dist_from_eye(b.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                transparent.sort_by(|a, b| {
                    dist_from_eye(b.1)
                        .partial_cmp(&dist_from_eye(a.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut render_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("ldr_foreground_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: scene_colour_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Load,
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view: fg_depth_view,
                                depth_ops: Some(crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Clear(1.0),
                                    store: crate::gpu::StoreOp::Discard,
                                }),
                                stencil_ops: Some(crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Clear(0),
                                    store: crate::gpu::StoreOp::Discard,
                                }),
                            },
                        ),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                render_pass.set_bind_group(0, &slot.foreground_camera_bind_group, &[]);

                for (idx, item) in opaque.iter().chain(transparent.iter()) {
                    let solid_pl = if item.material.is_two_sided() {
                        &resources.solid_two_sided_pipeline
                    } else {
                        &resources.solid_pipeline
                    };
                    let obj_bg = slot
                        .foreground_objects
                        .get(*idx)
                        .and_then(|e| e.bind_group.as_ref());
                    super::hdr_path::draw_mesh_item(
                        resources,
                        &self.compute_filter_results,
                        &mut render_pass,
                        item,
                        obj_bg,
                        false,
                        false,
                        solid_pl,
                        &resources.solid_two_sided_pipeline,
                        &resources.transparent_pipeline,
                        &resources.wireframe_pipeline,
                        // Foreground items draw through the positional
                        // foreground_objects cache, which has no per-range
                        // entries; they render with the single item material.
                        None,
                        None,
                    );
                }
            }
        }

        // Copy the scene depth just written into this viewport's HiZ prev-depth
        // target so next frame's occlusion cull can reproject it. Mirrors the HDR
        // path's `hdr_store_hiz_depth`; without it, occlusion culling is a silent
        // no-op on the LDR path. Only when occlusion is enabled (the depth was
        // kept via `store_scene_depth`). The depth view is read from `hdr` /
        // `dyn_res` while the pyramid is written into `cull`, disjoint fields of
        // the slot.
        if store_scene_depth {
            let view_proj = frame.camera.render_camera.view_proj().to_cols_array_2d();
            let slot = &mut self.viewport_slots[vp_idx];
            let (depth_only_view, dw, dh) = if use_dyn_res {
                let dr = slot.dyn_res.as_ref().unwrap();
                (&dr.depth_only_view, dr.scaled_size[0], dr.scaled_size[1])
            } else {
                let slot_hdr = slot.hdr.as_ref().unwrap();
                let tex = &slot_hdr.outline_depth_texture;
                (&slot_hdr.outline_depth_only_view, tex.width(), tex.height())
            };
            slot.cull.store_hiz_prev_depth(
                device,
                &mut encoder,
                depth_only_view,
                dw,
                dh,
                view_proj,
            );
        }

        // Backdrop blur: capture scene, run blur, then draw overlays in a
        // second render pass so blur shapes can sample the blurred result.
        if needs_blur {
            let spread = self
                .overlay_shape_gpu_data
                .as_ref()
                .map(|d| d.max_blur_radius)
                .unwrap_or(1.0);
            let blur_bg = {
                let source = if use_dyn_res {
                    &self.viewport_slots[vp_idx]
                        .dyn_res
                        .as_ref()
                        .unwrap()
                        .colour_view
                } else {
                    &self.backdrop_blur_state.as_ref().unwrap().intermediate_view
                };
                self.run_backdrop_blur(&mut encoder, device, queue, source, spread)
            };

            // Second render pass for overlays (Load to preserve scene content).
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let overlay_colour_view: &crate::gpu::TextureView = if use_dyn_res {
                &slot.dyn_res.as_ref().unwrap().colour_view
            } else {
                &self.backdrop_blur_state.as_ref().unwrap().intermediate_view
            };
            let overlay_depth_view: &crate::gpu::TextureView = if use_dyn_res {
                &slot.dyn_res.as_ref().unwrap().depth_view
            } else {
                &slot_hdr.outline_depth_view
            };
            {
                let mut overlay_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("ldr_overlay_blur_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: overlay_colour_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Load,
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view: overlay_depth_view,
                                depth_ops: Some(crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Load,
                                    store: crate::gpu::StoreOp::Discard,
                                }),
                                stencil_ops: None,
                            },
                        ),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                // Draw blur backdrop shapes first.
                self.draw_blur_shapes(&mut overlay_pass, &blur_bg);
                // Then normal shapes.
                emit_overlay_2d!(self, overlay_pass);
            }
        }

        // Resolve last frame's timestamp queries -> staging buffer, but only
        // when the previous readback has finished. Skipping while a map is in
        // flight or the buffer holds unread data keeps the single staging
        // buffer from being overwritten before prepare() has read it. The set
        // resolved here was written during the previous frame; resolving it in
        // this frame's (later) submission is what lets Metal's stage-boundary
        // counters settle, so short passes stop yielding stale equal-timestamp
        // samples.
        if !self.ts_data_ready && !self.ts_map_inflight {
            if let (Some(qs), Some(res_buf), Some(stg_buf)) = (
                self.ts_query_set_prev.as_ref(),
                self.ts_resolve_buf.as_ref(),
                self.ts_staging_buf.as_ref(),
            ) {
                let written = self.ts_prev_mask;
                // Resolve each contiguous run of written slots with its own
                // resolve call (see the HDR path's resolve block for the
                // backend constraints; the old contiguous-prefix scheme
                // silently dropped any slot after the first skipped one).
                if written != 0 {
                    for slot in 0..crate::renderer::GPU_TS_SLOTS {
                        if written & (1 << slot) == 0 {
                            continue;
                        }
                        // Each slot resolves into its own 256-byte region:
                        // resolve destination offsets must be 256-aligned.
                        encoder.resolve_query_set(
                            qs,
                            slot * 2..slot * 2 + 2,
                            res_buf,
                            slot as u64 * 256,
                        );
                    }
                    let ts_bytes = crate::renderer::GPU_TS_SLOTS as u64 * 256;
                    encoder.copy_buffer_to_buffer(res_buf, 0, stg_buf, 0, ts_bytes);
                    self.ts_pending_mask = written;
                    self.ts_data_ready = true;
                    // Consumed: keep a second render call this frame (multi-
                    // viewport) from resolving the same set again.
                    self.ts_prev_mask = 0;
                }
            }
        }

        // Upscale blit from dyn_res intermediate to output_view.
        if use_dyn_res {
            let upscale_bg = &self.viewport_slots[vp_idx]
                .dyn_res
                .as_ref()
                .unwrap()
                .upscale_bind_group;
            let mut upscale_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("dyn_res_upscale_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(pipeline) = &self.resources.post.dyn_res_upscale_pipeline {
                upscale_pass.set_pipeline(pipeline);
                upscale_pass.set_bind_group(0, upscale_bg, &[]);
                upscale_pass.draw(0..3, 0..1);
            }
        } else if needs_blur {
            // Blit backdrop intermediate to the output surface.
            let bs = self.backdrop_blur_state.as_ref().unwrap();
            let blit_bgl = self.resources.post.dyn_res_upscale_bgl.as_ref().unwrap();
            let blit_sampler = self.resources.post.dyn_res_linear_sampler.as_ref().unwrap();
            let blit_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("backdrop_blit_bg"),
                layout: blit_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(&bs.intermediate_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(blit_sampler),
                    },
                ],
            });
            let mut blit_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("backdrop_blit_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(pipeline) = &self.resources.post.dyn_res_upscale_pipeline {
                blit_pass.set_pipeline(pipeline);
                blit_pass.set_bind_group(0, &blit_bg, &[]);
                blit_pass.draw(0..3, 0..1);
            }
        }

        return encoder.finish();
    }
}
