use super::*;

impl ViewportRenderer {
    /// Issue draw calls for the viewport. Call inside a `wgpu::RenderPass`.
    ///
    /// This method requires a `'static` render pass (as provided by egui's
    /// `CallbackTrait`). For non-static render passes (iced, manual wgpu),
    /// use [`paint_to`](Self::paint_to).
    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'static>, frame: &FrameData) {
        let camera_bg = self.viewport_camera_bind_group(frame.camera.viewport_index);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            &self.compute_filter_results
        );
        emit_scivis_draw_calls!(
            &self.resources,
            render_pass,
            &self.point_cloud_gpu_data,
            &self.glyph_gpu_data,
            &self.polyline_gpu_data,
            &self.volume_gpu_data,
            &self.streamtube_gpu_data,
            camera_bg
        );
    }

    /// Issue draw calls into a render pass with any lifetime.
    ///
    /// Identical to [`paint`](Self::paint) but accepts a render pass with a
    /// non-`'static` lifetime, making it usable from iced, raw wgpu, or any
    /// framework that creates its own render pass.
    pub fn paint_to<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>, frame: &FrameData) {
        let camera_bg = self.viewport_camera_bind_group(frame.camera.viewport_index);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            &self.compute_filter_results
        );
        emit_scivis_draw_calls!(
            &self.resources,
            render_pass,
            &self.point_cloud_gpu_data,
            &self.glyph_gpu_data,
            &self.polyline_gpu_data,
            &self.volume_gpu_data,
            &self.streamtube_gpu_data,
            camera_bg
        );
    }

    /// High-level HDR render method. Handles the full post-processing pipeline:
    /// scene -> HDR texture -> (bloom) -> (SSAO) -> tone map -> output_view.
    ///
    /// When `frame.post_process.enabled` is false, falls back to a simple LDR render
    /// pass targeting `output_view` directly.
    ///
    /// Returns a `CommandBuffer` ready to submit.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        // Always run prepare() to upload uniforms and run the shadow pass.
        self.prepare(device, queue, frame);

        // Resolve scene items from the SurfaceSubmission seam.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items,
        };

        let bg_color =
            frame
                .viewport
                .background_color
                .unwrap_or([65.0 / 255.0, 65.0 / 255.0, 65.0 / 255.0, 1.0]);

        if !frame.effects.post_process.enabled {
            // LDR fallback: render directly to output_view.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ldr_encoder"),
            });
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ldr_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: bg_color[0] as f64,
                                g: bg_color[1] as f64,
                                b: bg_color[2] as f64,
                                a: bg_color[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: self.resources.outline_depth_view.as_ref().map(|v| {
                        wgpu::RenderPassDepthStencilAttachment {
                            view: v,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: None,
                        }
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                let camera_bg = self.viewport_camera_bind_group(frame.camera.viewport_index);
                emit_draw_calls!(
                    &self.resources,
                    &mut render_pass,
                    frame,
                    self.use_instancing,
                    &self.instanced_batches,
                    camera_bg,
                    &self.compute_filter_results
                );
                emit_scivis_draw_calls!(
                    &self.resources,
                    &mut render_pass,
                    &self.point_cloud_gpu_data,
                    &self.glyph_gpu_data,
                    &self.polyline_gpu_data,
                    &self.volume_gpu_data,
                    &self.streamtube_gpu_data,
                    camera_bg
                );
            }
            return encoder.finish();
        }

        // HDR path.
        let w = frame.camera.viewport_size[0] as u32;
        let h = frame.camera.viewport_size[1] as u32;

        self.resources.ensure_hdr_target(
            device,
            queue,
            self.resources.target_format,
            w.max(1),
            h.max(1),
        );

        let pp = &frame.effects.post_process;

        // Upload tone map uniform.
        let mode = match pp.tone_mapping {
            crate::renderer::ToneMapping::Reinhard => 0u32,
            crate::renderer::ToneMapping::Aces => 1u32,
            crate::renderer::ToneMapping::KhronosNeutral => 2u32,
        };
        let tm_uniform = crate::resources::ToneMapUniform {
            exposure: pp.exposure,
            mode,
            bloom_enabled: if pp.bloom { 1 } else { 0 },
            ssao_enabled: if pp.ssao { 1 } else { 0 },
            contact_shadows_enabled: if pp.contact_shadows { 1 } else { 0 },
            _pad_tm: [0; 3],
        };
        if let Some(buf) = &self.resources.tone_map_uniform_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&[tm_uniform]));
        }

        // Upload SSAO uniform if needed.
        if pp.ssao {
            if let Some(buf) = &self.resources.ssao_uniform_buf {
                let proj = frame.camera.render_camera.projection;
                let inv_proj = proj.inverse();
                let ssao_uniform = crate::resources::SsaoUniform {
                    inv_proj: inv_proj.to_cols_array_2d(),
                    proj: proj.to_cols_array_2d(),
                    radius: 0.5,
                    bias: 0.025,
                    _pad: [0.0; 2],
                };
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&[ssao_uniform]));
            }
        }

        // Upload contact shadow uniform if needed.
        if pp.contact_shadows {
            if let Some(buf) = &self.resources.contact_shadow_uniform_buf {
                let proj = frame.camera.render_camera.projection;
                let inv_proj = proj.inverse();
                // Transform first light direction to view space.
                let light_dir_world: glam::Vec3 = if let Some(l) = frame.effects.lighting.lights.first() {
                    match l.kind {
                        LightKind::Directional { direction } => {
                            glam::Vec3::from(direction).normalize()
                        }
                        LightKind::Spot { direction, .. } => {
                            glam::Vec3::from(direction).normalize()
                        }
                        _ => glam::Vec3::new(0.0, -1.0, 0.0),
                    }
                } else {
                    glam::Vec3::new(0.0, -1.0, 0.0)
                };
                let view = frame.camera.render_camera.view;
                let light_dir_view = view.transform_vector3(light_dir_world).normalize();
                let world_up_view = view.transform_vector3(glam::Vec3::Y).normalize();
                let cs_uniform = crate::resources::ContactShadowUniform {
                    inv_proj: inv_proj.to_cols_array_2d(),
                    proj: proj.to_cols_array_2d(),
                    light_dir_view: [
                        light_dir_view.x,
                        light_dir_view.y,
                        light_dir_view.z,
                        0.0,
                    ],
                    world_up_view: [world_up_view.x, world_up_view.y, world_up_view.z, 0.0],
                    params: [
                        pp.contact_shadow_max_distance,
                        pp.contact_shadow_steps as f32,
                        pp.contact_shadow_thickness,
                        0.0,
                    ],
                };
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&[cs_uniform]));
            }
        }

        // Upload bloom threshold/intensity uniform (horizontal flag is ignored for threshold pass).
        if pp.bloom {
            if let Some(buf) = &self.resources.bloom_uniform_buf {
                let bloom_u = crate::resources::BloomUniform {
                    threshold: pp.bloom_threshold,
                    intensity: pp.bloom_intensity,
                    horizontal: 0,
                    _pad: 0,
                };
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&[bloom_u]));
            }
        }

        // Rebuild tone-map bind group with correct bloom/AO texture views.
        self.resources.rebuild_tone_map_bind_group_with_device(
            device,
            pp.bloom,
            pp.ssao,
            pp.contact_shadows,
        );

        // -----------------------------------------------------------------------
        // Pre-allocate OIT targets if any transparent items exist.
        // Must happen before camera_bg is borrowed (borrow-checker constraint).
        // -----------------------------------------------------------------------
        {
            let needs_oit = if self.use_instancing && !self.instanced_batches.is_empty() {
                self.instanced_batches.iter().any(|b| b.is_transparent)
            } else {
                scene_items
                    .iter()
                    .any(|i| i.visible && i.material.opacity < 1.0)
            };
            if needs_oit {
                let w = (frame.camera.viewport_size[0] as u32).max(1);
                let h = (frame.camera.viewport_size[1] as u32).max(1);
                self.resources.ensure_oit_targets(device, w, h);
            }
        }

        // -----------------------------------------------------------------------
        // Build the command encoder.
        // -----------------------------------------------------------------------
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdr_encoder"),
        });

        // Per-viewport camera bind group for the HDR path.
        let camera_bg = self.viewport_camera_bind_group(frame.camera.viewport_index);

        // -----------------------------------------------------------------------
        // HDR scene pass: render geometry into the HDR texture.
        // -----------------------------------------------------------------------
        {
            let hdr_view = match &self.resources.hdr_view {
                Some(v) => v,
                None => {
                    return encoder.finish();
                }
            };
            let hdr_depth_view = match &self.resources.hdr_depth_view {
                Some(v) => v,
                None => {
                    return encoder.finish();
                }
            };

            let clear_wgpu = wgpu::Color {
                r: bg_color[0] as f64,
                g: bg_color[1] as f64,
                b: bg_color[2] as f64,
                a: bg_color[3] as f64,
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_scene_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_wgpu),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: hdr_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let resources = &self.resources;
            render_pass.set_bind_group(0, camera_bg, &[]);

            // Draw skybox background if environment map is loaded and show_skybox is enabled.
            let show_skybox = frame
                .effects
                .environment
                .as_ref()
                .is_some_and(|e| e.show_skybox)
                && resources.ibl_skybox_view.is_some();
            if show_skybox {
                render_pass.set_pipeline(&resources.skybox_pipeline);
                render_pass.draw(0..3, 0..1);
            }

            let use_instancing = self.use_instancing;
            let batches = &self.instanced_batches;

            if !scene_items.is_empty() {
                if use_instancing && !batches.is_empty() {
                    let excluded_items: Vec<&SceneRenderItem> = scene_items
                        .iter()
                        .filter(|item| {
                            item.visible
                                && (item.active_attribute.is_some() || item.two_sided)
                                && resources
                                    .mesh_store
                                    .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                    .is_some()
                        })
                        .collect();

                    // Separate opaque and transparent batches.
                    let mut opaque_batches: Vec<&InstancedBatch> = Vec::new();
                    let mut transparent_batches: Vec<&InstancedBatch> = Vec::new();
                    for batch in batches {
                        if batch.is_transparent {
                            transparent_batches.push(batch);
                        } else {
                            opaque_batches.push(batch);
                        }
                    }

                    if !opaque_batches.is_empty() && !frame.viewport.wireframe_mode {
                        if let Some(ref pipeline) = resources.hdr_solid_instanced_pipeline {
                            render_pass.set_pipeline(pipeline);
                            for batch in &opaque_batches {
                                let Some(mesh) = resources
                                    .mesh_store
                                    .get(crate::resources::mesh_store::MeshId(batch.mesh_index))
                                else {
                                    continue;
                                };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) =
                                    resources.instance_bind_groups.get(&mat_key)
                                else {
                                    continue;
                                };
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                            }
                        }
                    }

                    // NOTE: transparent_batches are now rendered in the OIT pass below,
                    // not in the HDR scene pass. This block intentionally left empty.
                    let _ = &transparent_batches; // suppress unused warning

                    if frame.viewport.wireframe_mode {
                        if let Some(ref hdr_wf) = resources.hdr_wireframe_pipeline {
                            render_pass.set_pipeline(hdr_wf);
                            for item in scene_items {
                                if !item.visible {
                                    continue;
                                }
                                let Some(mesh) = resources
                                    .mesh_store
                                    .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                else {
                                    continue;
                                };
                                render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.edge_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            }
                        }
                    } else if let (Some(hdr_solid), Some(hdr_solid_two_sided)) = (
                        &resources.hdr_solid_pipeline,
                        &resources.hdr_solid_two_sided_pipeline,
                    ) {
                        for item in excluded_items
                            .into_iter()
                            .filter(|item| item.material.opacity >= 1.0)
                        {
                            let Some(mesh) = resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                            else {
                                continue;
                            };
                            let pipeline = if item.two_sided {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
                } else {
                    // Per-object path.
                    let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
                    let dist_from_eye = |item: &&SceneRenderItem| -> f32 {
                        let pos =
                            glam::Vec3::new(item.model[3][0], item.model[3][1], item.model[3][2]);
                        (pos - eye).length()
                    };

                    let mut opaque: Vec<&SceneRenderItem> = Vec::new();
                    let mut transparent: Vec<&SceneRenderItem> = Vec::new();
                    for item in scene_items {
                        if !item.visible
                            || resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                .is_none()
                        {
                            continue;
                        }
                        if item.material.opacity < 1.0 {
                            transparent.push(item);
                        } else {
                            opaque.push(item);
                        }
                    }
                    opaque.sort_by(|a, b| {
                        dist_from_eye(a)
                            .partial_cmp(&dist_from_eye(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    transparent.sort_by(|a, b| {
                        dist_from_eye(b)
                            .partial_cmp(&dist_from_eye(a))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    let draw_item_hdr =
                        |render_pass: &mut wgpu::RenderPass<'_>,
                         item: &SceneRenderItem,
                         solid_pl: &wgpu::RenderPipeline,
                         trans_pl: &wgpu::RenderPipeline,
                         wf_pl: &wgpu::RenderPipeline| {
                            let mesh = resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                .unwrap();
                            // mesh.object_bind_group (group 1) already carries the object uniform
                            // and the correct texture views.
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            if frame.viewport.wireframe_mode {
                                render_pass.set_pipeline(wf_pl);
                                render_pass.set_index_buffer(
                                    mesh.edge_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            } else if item.material.opacity < 1.0 {
                                render_pass.set_pipeline(trans_pl);
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            } else {
                                render_pass.set_pipeline(solid_pl);
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            }
                        };

                    // NOTE: only opaque items are drawn here. Transparent items are
                    // routed to the OIT pass below.
                    let _ = &transparent; // suppress unused warning
                    if let (
                        Some(hdr_solid),
                        Some(hdr_solid_two_sided),
                        Some(hdr_trans),
                        Some(hdr_wf),
                    ) = (
                        &resources.hdr_solid_pipeline,
                        &resources.hdr_solid_two_sided_pipeline,
                        &resources.hdr_transparent_pipeline,
                        &resources.hdr_wireframe_pipeline,
                    ) {
                        for item in &opaque {
                            let solid_pl = if item.two_sided {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            draw_item_hdr(&mut render_pass, item, solid_pl, hdr_trans, hdr_wf);
                        }
                    }
                }
            }

            // Cap fill pass (HDR path — section view cross-section fill).
            if !resources.cap_buffers.is_empty() {
                if let Some(ref hdr_overlay) = resources.hdr_overlay_pipeline {
                    render_pass.set_pipeline(hdr_overlay);
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &resources.cap_buffers {
                        render_pass.set_bind_group(1, bg, &[]);
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }
            }

            // SciVis Phase B+D+M8+M: point cloud, glyph, polyline, volume, streamtube (HDR path).
            emit_scivis_draw_calls!(
                &self.resources,
                &mut render_pass,
                &self.point_cloud_gpu_data,
                &self.glyph_gpu_data,
                &self.polyline_gpu_data,
                &self.volume_gpu_data,
                &self.streamtube_gpu_data,
                camera_bg
            );
        }

        // -----------------------------------------------------------------------
        // OIT pass: render transparent items into accum + reveal textures.
        // Completely skipped when no transparent items exist (zero overhead).
        // -----------------------------------------------------------------------
        let has_transparent = if self.use_instancing && !self.instanced_batches.is_empty() {
            self.instanced_batches.iter().any(|b| b.is_transparent)
        } else {
            scene_items
                .iter()
                .any(|i| i.visible && i.material.opacity < 1.0)
        };

        if has_transparent {
            // OIT targets already allocated in the pre-pass above.
            if let (Some(accum_view), Some(reveal_view), Some(hdr_depth_view)) = (
                self.resources.oit_accum_view.as_ref(),
                self.resources.oit_reveal_view.as_ref(),
                self.resources.hdr_depth_view.as_ref(),
            ) {
                // Clear accum to (0,0,0,0), reveal to 1.0 (no contribution yet).
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: accum_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: reveal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 1.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // reuse opaque depth
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                oit_pass.set_bind_group(0, camera_bg, &[]);

                if self.use_instancing && !self.instanced_batches.is_empty() {
                    if let Some(ref pipeline) = self.resources.oit_instanced_pipeline {
                        oit_pass.set_pipeline(pipeline);
                        for batch in &self.instanced_batches {
                            if !batch.is_transparent {
                                continue;
                            }
                            let Some(mesh) = self
                                .resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(batch.mesh_index))
                            else {
                                continue;
                            };
                            let mat_key = (
                                batch.texture_id.unwrap_or(u64::MAX),
                                batch.normal_map_id.unwrap_or(u64::MAX),
                                batch.ao_map_id.unwrap_or(u64::MAX),
                            );
                            let Some(inst_tex_bg) =
                                self.resources.instance_bind_groups.get(&mat_key)
                            else {
                                continue;
                            };
                            oit_pass.set_bind_group(1, inst_tex_bg, &[]);
                            oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            oit_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            oit_pass.draw_indexed(
                                0..mesh.index_count,
                                0,
                                batch.instance_offset..batch.instance_offset + batch.instance_count,
                            );
                        }
                    }
                } else if let Some(ref pipeline) = self.resources.oit_pipeline {
                    oit_pass.set_pipeline(pipeline);
                    for item in scene_items {
                        if !item.visible || item.material.opacity >= 1.0 {
                            continue;
                        }
                        let Some(mesh) = self
                            .resources
                            .mesh_store
                            .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                        else {
                            continue;
                        };
                        oit_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                        oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        oit_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        oit_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // OIT composite pass: blend accum/reveal into HDR buffer.
        // Only executes when transparent items were present.
        // -----------------------------------------------------------------------
        if has_transparent {
            if let (Some(pipeline), Some(bg), Some(hdr_view)) = (
                self.resources.oit_composite_pipeline.as_ref(),
                self.resources.oit_composite_bind_group.as_ref(),
                self.resources.hdr_view.as_ref(),
            ) {
                let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_composite_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                composite_pass.set_pipeline(pipeline);
                composite_pass.set_bind_group(0, bg, &[]);
                composite_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Outline composite pass (HDR path): blit offscreen outline onto hdr_view.
        // Runs after the HDR scene pass (which has depth+stencil) in a separate
        // pass with no depth attachment, so the composite pipeline is compatible.
        // -----------------------------------------------------------------------
        if !self.resources.outline_object_buffers.is_empty() {
            if let (Some(pipeline), Some(bg), Some(hdr_view), Some(hdr_depth_view)) = (
                &self.resources.outline_composite_pipeline_single,
                &self.resources.outline_composite_bind_group,
                &self.resources.hdr_view,
                &self.resources.hdr_depth_view,
            ) {
                let mut outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hdr_outline_composite_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                outline_pass.set_pipeline(pipeline);
                outline_pass.set_bind_group(0, bg, &[]);
                outline_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // SSAO pass.
        // -----------------------------------------------------------------------
        if pp.ssao {
            if let (Some(ssao_bg), Some(ssao_pipeline), Some(ssao_view)) = (
                &self.resources.ssao_bg,
                &self.resources.ssao_pipeline,
                &self.resources.ssao_view,
            ) {
                {
                    let mut ssao_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ssao_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: ssao_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    ssao_pass.set_pipeline(ssao_pipeline);
                    ssao_pass.set_bind_group(0, ssao_bg, &[]);
                    ssao_pass.draw(0..3, 0..1);
                }

                // SSAO blur pass.
                if let (Some(ssao_blur_bg), Some(ssao_blur_pipeline), Some(ssao_blur_view)) = (
                    &self.resources.ssao_blur_bg,
                    &self.resources.ssao_blur_pipeline,
                    &self.resources.ssao_blur_view,
                ) {
                    let mut ssao_blur_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("ssao_blur_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: ssao_blur_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    ssao_blur_pass.set_pipeline(ssao_blur_pipeline);
                    ssao_blur_pass.set_bind_group(0, ssao_blur_bg, &[]);
                    ssao_blur_pass.draw(0..3, 0..1);
                }
            }
        }

        // -----------------------------------------------------------------------
        // Contact shadow pass.
        // -----------------------------------------------------------------------
        if pp.contact_shadows {
            if let (Some(cs_bg), Some(cs_pipeline), Some(cs_view)) = (
                &self.resources.contact_shadow_bg,
                &self.resources.contact_shadow_pipeline,
                &self.resources.contact_shadow_view,
            ) {
                let mut cs_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("contact_shadow_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: cs_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                cs_pass.set_pipeline(cs_pipeline);
                cs_pass.set_bind_group(0, cs_bg, &[]);
                cs_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Bloom passes.
        // -----------------------------------------------------------------------
        if pp.bloom {
            // Threshold pass: extract bright pixels into bloom_threshold_texture.
            if let (
                Some(bloom_threshold_bg),
                Some(bloom_threshold_pipeline),
                Some(bloom_threshold_view),
            ) = (
                &self.resources.bloom_threshold_bg,
                &self.resources.bloom_threshold_pipeline,
                &self.resources.bloom_threshold_view,
            ) {
                {
                    let mut threshold_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("bloom_threshold_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: bloom_threshold_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    threshold_pass.set_pipeline(bloom_threshold_pipeline);
                    threshold_pass.set_bind_group(0, bloom_threshold_bg, &[]);
                    threshold_pass.draw(0..3, 0..1);
                }

                // 4 ping-pong H+V blur passes for a wide glow.
                // Pass 1: threshold -> ping -> pong. Passes 2-4: pong -> ping -> pong.
                if let (
                    Some(blur_h_bg),
                    Some(blur_h_pong_bg),
                    Some(blur_v_bg),
                    Some(blur_pipeline),
                    Some(bloom_ping_view),
                    Some(bloom_pong_view),
                ) = (
                    &self.resources.bloom_blur_h_bg,
                    &self.resources.bloom_blur_h_pong_bg,
                    &self.resources.bloom_blur_v_bg,
                    &self.resources.bloom_blur_pipeline,
                    &self.resources.bloom_ping_view,
                    &self.resources.bloom_pong_view,
                ) {
                    const BLUR_ITERATIONS: usize = 4;
                    for i in 0..BLUR_ITERATIONS {
                        // H pass: pass 0 reads threshold, subsequent passes read pong.
                        let h_bg = if i == 0 { blur_h_bg } else { blur_h_pong_bg };
                        {
                            let mut h_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("bloom_blur_h_pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: bloom_ping_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                        depth_slice: None,
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            h_pass.set_pipeline(blur_pipeline);
                            h_pass.set_bind_group(0, h_bg, &[]);
                            h_pass.draw(0..3, 0..1);
                        }
                        // V pass: ping -> pong.
                        {
                            let mut v_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("bloom_blur_v_pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: bloom_pong_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                        depth_slice: None,
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            v_pass.set_pipeline(blur_pipeline);
                            v_pass.set_bind_group(0, blur_v_bg, &[]);
                            v_pass.draw(0..3, 0..1);
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Tone map pass: HDR + bloom + AO -> (fxaa_texture if FXAA) or output_view.
        // -----------------------------------------------------------------------
        let use_fxaa = pp.fxaa && self.resources.fxaa_view.is_some();
        if let (Some(tone_map_pipeline), Some(tone_map_bg)) = (
            &self.resources.tone_map_pipeline,
            &self.resources.tone_map_bind_group,
        ) {
            let tone_target: &wgpu::TextureView = if use_fxaa {
                self.resources.fxaa_view.as_ref().unwrap()
            } else {
                output_view
            };
            let mut tone_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tone_map_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: tone_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            tone_pass.set_pipeline(tone_map_pipeline);
            tone_pass.set_bind_group(0, tone_map_bg, &[]);
            tone_pass.draw(0..3, 0..1);
        }

        // -----------------------------------------------------------------------
        // FXAA pass: fxaa_texture -> output_view (only when FXAA is enabled).
        // -----------------------------------------------------------------------
        if use_fxaa {
            if let (Some(fxaa_pipeline), Some(fxaa_bg)) = (
                &self.resources.fxaa_pipeline,
                &self.resources.fxaa_bind_group,
            ) {
                let mut fxaa_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("fxaa_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                fxaa_pass.set_pipeline(fxaa_pipeline);
                fxaa_pass.set_bind_group(0, fxaa_bg, &[]);
                fxaa_pass.draw(0..3, 0..1);
            }
        }

        encoder.finish()
    }

    /// Render a frame to an offscreen texture and return raw RGBA bytes.
    ///
    /// Creates a temporary [`wgpu::Texture`] render target of the given dimensions,
    /// runs all render passes (shadow, scene, post-processing) into it via
    /// [`render()`](Self::render), then copies the result back to CPU memory.
    ///
    /// No OS window or [`wgpu::Surface`] is required. The caller is responsible for
    /// initialising the wgpu adapter with `compatible_surface: None` and for
    /// constructing a valid [`FrameData`] (including `viewport_size` matching
    /// `width`/`height`).
    ///
    /// Returns `width * height * 4` bytes in RGBA8 layout. The caller encodes to
    /// PNG/EXR independently — no image codec dependency in this crate.
    pub fn render_offscreen(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        // 1. Create offscreen texture with RENDER_ATTACHMENT | COPY_SRC usage.
        let target_format = self.resources.target_format;
        let offscreen_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // 2. Create a texture view for rendering into.
        let output_view = offscreen_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 3. Ensure a depth-stencil buffer exists for the given dimensions.
        //    The LDR render pass uses `resources.outline_depth_view` as its depth
        //    attachment. If none exists, `solid_pipeline` (which expects
        //    Depth24PlusStencil8) would produce a wgpu validation error.
        self.resources
            .ensure_outline_target(device, width.max(1), height.max(1));

        // 4. Render the scene into the offscreen texture.
        //    The caller must set `frame.camera.viewport_size` to `[width as f32, height as f32]`
        //    and `frame.camera.render_camera.aspect` to `width as f32 / height as f32`
        //    for correct HDR target allocation and scissor rects.
        let cmd_buf = self.render(device, queue, &output_view, frame);
        queue.submit(std::iter::once(cmd_buf));

        // 5. Copy texture -> staging buffer (wgpu requires row alignment to 256 bytes).
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) & !(align - 1);
        let buffer_size = (padded_row * height.max(1)) as u64;

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen_copy_encoder"),
        });
        copy_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &offscreen_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height.max(1)),
                },
            },
            wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(copy_encoder.finish()));

        // 6. Map buffer and extract tightly-packed RGBA pixels.
        let (tx, rx) = std::sync::mpsc::channel();
        staging_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        let _ = rx.recv().unwrap_or(Err(wgpu::BufferAsyncError));

        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        {
            let mapped = staging_buf.slice(..).get_mapped_range();
            let data: &[u8] = &mapped;
            if padded_row == unpadded_row {
                // No padding — copy entire slice directly.
                pixels.extend_from_slice(data);
            } else {
                // Strip row padding.
                for row in 0..height as usize {
                    let start = row * padded_row as usize;
                    let end = start + unpadded_row as usize;
                    pixels.extend_from_slice(&data[start..end]);
                }
            }
        }
        staging_buf.unmap();

        // 7. Swizzle BGRA -> RGBA if the format stores bytes in BGRA order.
        let is_bgra = matches!(
            target_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        if is_bgra {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2); // B ↔ R
            }
        }

        pixels
    }
}
