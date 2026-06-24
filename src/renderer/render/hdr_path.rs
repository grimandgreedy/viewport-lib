//! The HDR render path: the full post-processing pipeline (scene, sprites,
//! decals, transparency, scatter, flow, bloom, tone mapping, overlays). Builds
//! its own command encoder and returns the finished buffer.

use super::*;

/// Per-frame context shared by the HDR pass-group methods. Holds the
/// preamble-computed values each pass needs. It borrows only frame-level
/// data, never `self`, so the pass methods can take `&mut self` freely.
struct HdrFrameCtx<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    frame: &'a FrameData,
    scene_items: &'a [SceneRenderItem],
    output_view: &'a wgpu::TextureView,
    vp_idx: usize,
    w: u32,
    h: u32,
    ssaa_factor: u32,
    hdr_clear_rgb: [f32; 3],
}

impl ViewportRenderer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_frame_hdr(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        vp_idx: usize,
        frame: &FrameData,
        scene_items: &[SceneRenderItem],
        bg_colour: [f32; 4],
        w: u32,
        h: u32,
        ssaa_factor: u32,
    ) -> wgpu::CommandBuffer {
        // HDR path.
        let pp = &frame.effects.post_process;

        let hdr_clear_rgb = [
            bg_colour[0].powf(2.2),
            bg_colour[1].powf(2.2),
            bg_colour[2].powf(2.2),
        ];

        // Upload tone map uniform into the per-viewport buffer.
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
            edl_enabled: if pp.edl_enabled { 1 } else { 0 },
            edl_radius: pp.edl_radius,
            edl_strength: pp.edl_strength,
            background_colour: bg_colour,
            near_plane: frame.camera.render_camera.near,
            far_plane: frame.camera.render_camera.far,
            lic_enabled: if scene_items
                .iter()
                .any(|i| i.lic.is_some() && !i.settings.hidden)
            {
                1
            } else {
                0
            },
            lic_strength: scene_items
                .iter()
                .filter(|i| !i.settings.hidden)
                .find_map(|i| i.lic.as_ref().map(|l| l.config.strength))
                .unwrap_or(0.5),
        };
        {
            let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            queue.write_buffer(
                &hdr.tone_map_uniform_buf,
                0,
                bytemuck::cast_slice(&[tm_uniform]),
            );

            // Upload SSAO uniform if needed.
            if pp.ssao {
                let proj = frame.camera.render_camera.projection;
                let inv_proj = proj.inverse();
                let ssao_uniform = crate::resources::SsaoUniform {
                    inv_proj: inv_proj.to_cols_array_2d(),
                    proj: proj.to_cols_array_2d(),
                    radius: 0.5,
                    bias: 0.025,
                    _pad: [0.0; 2],
                };
                queue.write_buffer(
                    &hdr.ssao_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[ssao_uniform]),
                );
            }

            // Upload contact shadow uniform if needed.
            if pp.contact_shadows {
                let proj = frame.camera.render_camera.projection;
                let inv_proj = proj.inverse();
                let light_dir_world: glam::Vec3 =
                    if let Some(l) = frame.effects.lighting.lights.first() {
                        match l.kind {
                            LightKind::Directional { direction } => {
                                glam::Vec3::from(direction).normalize()
                            }
                            LightKind::Spot { direction, .. } => {
                                // Spot::direction is the shining direction
                                // (light -> scene); the march needs the
                                // surface -> light direction, so negate.
                                -glam::Vec3::from(direction).normalize()
                            }
                            _ => glam::Vec3::new(0.0, -1.0, 0.0),
                        }
                    } else {
                        glam::Vec3::new(0.0, -1.0, 0.0)
                    };
                let view = frame.camera.render_camera.view;
                let light_dir_view = view.transform_vector3(light_dir_world).normalize();
                let world_up_view = view.transform_vector3(glam::Vec3::Z).normalize();
                let cs_uniform = crate::resources::ContactShadowUniform {
                    inv_proj: inv_proj.to_cols_array_2d(),
                    proj: proj.to_cols_array_2d(),
                    light_dir_view: [light_dir_view.x, light_dir_view.y, light_dir_view.z, 0.0],
                    world_up_view: [world_up_view.x, world_up_view.y, world_up_view.z, 0.0],
                    params: [
                        pp.contact_shadow_max_distance,
                        pp.contact_shadow_steps as f32,
                        pp.contact_shadow_thickness,
                        0.0,
                    ],
                };
                queue.write_buffer(
                    &hdr.contact_shadow_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[cs_uniform]),
                );
            }

            // Upload bloom uniform if needed.
            if pp.bloom {
                let bloom_u = crate::resources::BloomUniform {
                    threshold: pp.bloom_threshold,
                    intensity: pp.bloom_intensity,
                    horizontal: 0,
                    _pad: 0,
                };
                queue.write_buffer(&hdr.bloom_uniform_buf, 0, bytemuck::cast_slice(&[bloom_u]));
            }
        }

        // Upload DoF uniform when enabled.
        if pp.dof_enabled {
            let (w, h) = {
                let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                (hdr.scene_size[0] as f32, hdr.scene_size[1] as f32)
            };
            let dof_uniform = crate::resources::DofUniform {
                focal_distance: pp.dof_focal_distance,
                focal_range: pp.dof_focal_range,
                max_blur_radius: pp.dof_max_blur_radius,
                near_plane: frame.camera.render_camera.near,
                far_plane: frame.camera.render_camera.far,
                viewport_width: w,
                viewport_height: h,
                _pad: 0.0,
            };
            let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            queue.write_buffer(
                &hdr.dof_uniform_buf,
                0,
                bytemuck::cast_slice(&[dof_uniform]),
            );
        }

        // Rebuild tone-map bind group with correct bloom/AO/DoF texture views.
        {
            let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
            self.resources.rebuild_tone_map_bind_group(
                device,
                hdr,
                pp.bloom,
                pp.ssao,
                pp.contact_shadows,
                scene_items
                    .iter()
                    .any(|i| i.lic.is_some() && !i.settings.hidden),
                pp.dof_enabled,
            );
        }

        // -----------------------------------------------------------------------
        // Pre-allocate OIT targets if any transparent items exist.
        // Must happen before camera_bg is borrowed (borrow-checker constraint).
        // -----------------------------------------------------------------------
        {
            let needs_oit = if self.instancing.use_instancing && !self.instancing.batches.is_empty()
            {
                self.instancing.batches.iter().any(|b| b.is_transparent)
            } else {
                scene_items
                    .iter()
                    .any(|i| !i.settings.hidden && i.settings.opacity < 1.0)
            } || frame
                .scene
                .volume_meshes
                .iter()
                .any(|i| !i.settings.hidden && i.transparency.is_some());
            if needs_oit {
                let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
                let [sw, sh] = hdr.scene_size;
                self.resources.ensure_viewport_oit(device, hdr, sw, sh);
            }
        }

        // -----------------------------------------------------------------------
        // Build the command encoder.
        // -----------------------------------------------------------------------
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdr_encoder"),
        });

        let ctx = HdrFrameCtx {
            device,
            queue,
            frame,
            scene_items,
            output_view,
            vp_idx,
            w,
            h,
            ssaa_factor,
            hdr_clear_rgb,
        };

        self.hdr_scene_pass(&ctx, &mut encoder);
        self.hdr_sprite_passes(&ctx, &mut encoder);
        self.hdr_ssaa_refraction(&ctx, &mut encoder);
        self.hdr_decals(&ctx, &mut encoder);
        self.hdr_sub_highlight(&ctx, &mut encoder);
        self.hdr_oit(&ctx, &mut encoder);
        self.hdr_scatter(&ctx, &mut encoder);
        self.hdr_lic(&ctx, &mut encoder);
        self.hdr_outline_and_post(&ctx, &mut encoder);
        self.hdr_tonemap_resolve(&ctx, &mut encoder);
        self.hdr_scene_overlays(&ctx, &mut encoder);
        self.hdr_final_overlay(&ctx, &mut encoder);
        // Resolve timestamp queries -> staging buffer (HDR path). Skip while a
        // readback is unread or in flight so the single staging buffer is not
        // overwritten before prepare() reads it.
        if !self.ts_data_ready && !self.ts_map_inflight {
            if let (Some(qs), Some(res_buf), Some(stg_buf)) = (
                self.ts_query_set.as_ref(),
                self.ts_resolve_buf.as_ref(),
                self.ts_staging_buf.as_ref(),
            ) {
                let ts_count = 2 * crate::renderer::GPU_TS_SLOTS;
                encoder.resolve_query_set(qs, 0..ts_count, res_buf, 0);
                encoder.copy_buffer_to_buffer(res_buf, 0, stg_buf, 0, ts_count as u64 * 8);
                self.ts_pending_mask = self
                    .ts_written_mask
                    .load(std::sync::atomic::Ordering::Relaxed);
                self.ts_data_ready = true;
            }
        }

        encoder.finish()
    }

    fn hdr_scene_pass(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let frame = ctx.frame;
        let scene_items = ctx.scene_items;
        let vp_idx = ctx.vp_idx;
        let ssaa_factor = ctx.ssaa_factor;
        let hdr_clear_rgb = ctx.hdr_clear_rgb;
        // Per-viewport camera bind group and HDR state for the HDR path.
        let slot = &self.viewport_slots[vp_idx];
        let camera_bg = &slot.camera_bind_group;
        let slot_hdr = slot.hdr.as_ref().expect(
            "HDR state missing; ensure_viewport_hdr must be called before render_frame_internal",
        );

        // -----------------------------------------------------------------------
        // HDR scene pass: render geometry into the HDR texture.
        // -----------------------------------------------------------------------
        {
            // Use SSAA target if enabled, otherwise render directly to hdr_texture.
            let use_ssaa = ssaa_factor > 1
                && slot_hdr.ssaa_colour_view.is_some()
                && slot_hdr.ssaa_depth_view.is_some();
            let scene_colour_view = if use_ssaa {
                slot_hdr.ssaa_colour_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_view
            };
            let scene_depth_view = if use_ssaa {
                slot_hdr.ssaa_depth_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_depth_view
            };

            let clear_wgpu = wgpu::Color {
                r: hdr_clear_rgb[0] as f64,
                g: hdr_clear_rgb[1] as f64,
                b: hdr_clear_rgb[2] as f64,
                // Clear alpha to 0.0 so OIT composite can signal presence via alpha > 0.
                // Background pixels remain at alpha=0 and are detected in tone_map.wgsl.
                a: 0.0,
            };

            let hdr_ts_writes = self.ts_query_set.as_ref().map(|qs| {
                self.ts_written_mask.fetch_or(
                    1 << crate::renderer::GPU_TS_SCENE,
                    std::sync::atomic::Ordering::Relaxed,
                );
                wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(crate::renderer::GPU_TS_SCENE * 2),
                    end_of_pass_write_index: Some(crate::renderer::GPU_TS_SCENE * 2 + 1),
                }
            });
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_scene_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_colour_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_wgpu),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: scene_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: hdr_ts_writes,
                occlusion_query_set: None,
            });

            let resources = &self.resources;
            render_pass.set_bind_group(0, camera_bg, &[]);

            // Check skybox eligibility early; drawn after all opaques below.
            let show_skybox = frame
                .effects
                .environment
                .as_ref()
                .is_some_and(|e| e.show_skybox)
                && resources.ibl_skybox_view.is_some();

            let use_instancing = self.instancing.use_instancing;
            let batches = &self.instancing.batches;
            let compute_filter_results = &self.compute_filter_results;

            if !scene_items.is_empty() {
                if use_instancing && !batches.is_empty() {
                    let excluded_items: Vec<(usize, &SceneRenderItem)> = scene_items
                        .iter()
                        .enumerate()
                        .filter(|(_, item)| {
                            // The per-object set is exactly the visible items that were
                            // not admitted to an instanced batch. Reuse `is_instanceable`
                            // (the single source of truth used in prepare) instead of
                            // re-listing its conditions, so this filter cannot drift from
                            // it -- a past drift dropped position-override and
                            // compute-filter items from the scene pass entirely.
                            !item.settings.hidden
                                && resources.mesh_store.get(item.mesh_id).is_some()
                                && !crate::renderer::prepare::is_instanceable(
                                    item,
                                    resources,
                                    compute_filter_results,
                                )
                        })
                        .collect();

                    // Separate opaque and transparent batches.
                    // Carry the global batch index (position in `batches`) alongside each batch
                    // so draw_indexed_indirect can compute the correct buffer offset.
                    let mut opaque_batches: Vec<(usize, &InstancedBatch)> = Vec::new();
                    let mut transparent_batches: Vec<(usize, &InstancedBatch)> = Vec::new();
                    for (batch_global_idx, batch) in batches.iter().enumerate() {
                        if batch.is_transparent {
                            transparent_batches.push((batch_global_idx, batch));
                        } else {
                            opaque_batches.push((batch_global_idx, batch));
                        }
                    }

                    if !opaque_batches.is_empty() && !frame.viewport.wireframe_mode {
                        let use_indirect = self.instancing.gpu_culling_enabled
                            && resources.hdr_solid_instanced_cull_pipeline.is_some()
                            && resources.indirect_args_buf.is_some();

                        if use_indirect {
                            if let (Some(pipeline), Some(pipeline_two_sided), Some(indirect_buf)) = (
                                &resources.hdr_solid_instanced_cull_pipeline,
                                &resources.hdr_solid_instanced_cull_two_sided_pipeline,
                                &resources.indirect_args_buf,
                            ) {
                                render_pass.set_bind_group(
                                    2,
                                    &resources.deform.dummy_bind_group,
                                    &[],
                                );
                                // Batches are sorted with two_sided in the key, so
                                // one- and two-sided runs are contiguous; switch the
                                // pipeline only when the flag changes.
                                let mut cur_two_sided: Option<bool> = None;
                                for (batch_global_idx, batch) in &opaque_batches {
                                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                        continue;
                                    };
                                    let mat_key = (
                                        batch.texture_id.unwrap_or(u64::MAX),
                                        batch.normal_map_id.unwrap_or(u64::MAX),
                                        batch.ao_map_id.unwrap_or(u64::MAX),
                                    );
                                    let Some(inst_tex_bg) =
                                        resources.instance_cull_bind_groups.get(&mat_key)
                                    else {
                                        continue;
                                    };
                                    if cur_two_sided != Some(batch.two_sided) {
                                        render_pass.set_pipeline(if batch.two_sided {
                                            pipeline_two_sided
                                        } else {
                                            pipeline
                                        });
                                        cur_two_sided = Some(batch.two_sided);
                                    }
                                    render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                    render_pass.set_index_buffer(
                                        mesh.index_buffer.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    // Each DrawIndexedIndirect entry is 20 bytes; index by global
                                    // batch position so the offset matches write_indirect_args output.
                                    render_pass.draw_indexed_indirect(
                                        indirect_buf,
                                        *batch_global_idx as u64 * 20,
                                    );
                                }
                            }
                        } else if let (Some(pipeline), Some(pipeline_two_sided)) = (
                            &resources.hdr_solid_instanced_pipeline,
                            &resources.hdr_solid_two_sided_instanced_pipeline,
                        ) {
                            render_pass.set_bind_group(2, &resources.deform.dummy_bind_group, &[]);
                            let mut cur_two_sided: Option<bool> = None;
                            for (_, batch) in &opaque_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
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
                                if cur_two_sided != Some(batch.two_sided) {
                                    render_pass.set_pipeline(if batch.two_sided {
                                        pipeline_two_sided
                                    } else {
                                        pipeline
                                    });
                                    cur_two_sided = Some(batch.two_sided);
                                }
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
                            let mut wf_idx = 0usize;
                            for item in scene_items {
                                if item.settings.hidden {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                    continue;
                                };
                                render_pass.set_pipeline(hdr_wf);
                                render_pass.set_bind_group(
                                    2,
                                    resources.deform.instance_bind_group_for(
                                        item.mesh_id,
                                        item.deform_instance,
                                    ),
                                    &[],
                                );
                                let bg = self
                                    .mesh_uniforms
                                    .wireframe_bind_groups
                                    .get(wf_idx)
                                    .unwrap_or(&mesh.object_bind_group);
                                render_pass.set_bind_group(1, bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.edge_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                                wf_idx += 1;
                            }
                        }
                    } else if let (Some(hdr_solid), Some(hdr_solid_two_sided)) = (
                        &resources.hdr_solid_pipeline,
                        &resources.hdr_solid_two_sided_pipeline,
                    ) {
                        // Only opaque excluded items are drawn in the scene pass; transparent
                        // excluded items go to the OIT pass below. LDR draws all excluded
                        // items inline (including transparent ones) using the transparent
                        // pipeline -- an intentional divergence since HDR uses OIT for
                        // transparency throughout.
                        for (item_idx, item) in
                            excluded_items.iter().copied().filter(|(_, item)| {
                                item.settings.opacity >= 1.0 && !item.material.is_blend()
                            })
                        {
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };
                            let pipeline = if item.material.is_two_sided() {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(
                                2,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance),
                                &[],
                            );
                            let obj_bg = self
                                .mesh_uniforms
                                .bind_groups
                                .get(item_idx)
                                .and_then(|opt| opt.as_ref())
                                .unwrap_or(&mesh.object_bind_group);
                            render_pass.set_bind_group(1, obj_bg, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            let filter = compute_filter_results
                                .iter()
                                .find(|r| r.mesh_id == item.mesh_id);
                            if let Some(fr) = filter {
                                render_pass.set_index_buffer(
                                    fr.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..fr.index_count, 0, 0..1);
                            } else {
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            }
                        }
                    }

                    // Normal-line overlays for instanced items with show_normals set.
                    // Instanced batch draws skip per-item logic, so these are drawn
                    // here after all batches finish.
                    if let Some(hdr_wf) = &resources.hdr_wireframe_pipeline {
                        for item in scene_items
                            .iter()
                            .filter(|i| i.show_normals && !i.settings.hidden)
                        {
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };
                            if let Some(ref nl_buf) = mesh.normal_line_buffer {
                                if mesh.normal_line_count > 0 {
                                    render_pass.set_pipeline(hdr_wf);
                                    render_pass.set_bind_group(
                                        2,
                                        &resources.deform.dummy_bind_group,
                                        &[],
                                    );
                                    render_pass.set_bind_group(1, &mesh.normal_bind_group, &[]);
                                    render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                                    render_pass.draw(0..mesh.normal_line_count, 0..1);
                                }
                            }
                        }
                    }
                } else {
                    // Per-object path.
                    let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
                    let dist_from_eye = |entry: &(usize, &SceneRenderItem)| -> f32 {
                        let item = entry.1;
                        let pos =
                            glam::Vec3::new(item.model[3][0], item.model[3][1], item.model[3][2]);
                        (pos - eye).length()
                    };

                    let mut opaque: Vec<(usize, &SceneRenderItem)> = Vec::new();
                    let mut transparent: Vec<(usize, &SceneRenderItem)> = Vec::new();
                    for (idx, item) in scene_items.iter().enumerate() {
                        if item.settings.hidden || resources.mesh_store.get(item.mesh_id).is_none()
                        {
                            continue;
                        }
                        if item.settings.opacity < 1.0 || item.material.is_blend() {
                            transparent.push((idx, item));
                        } else {
                            opaque.push((idx, item));
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

                    let per_item_bgs = &self.mesh_uniforms.bind_groups;
                    let draw_item_hdr =
                        |render_pass: &mut wgpu::RenderPass<'_>,
                         item_idx: usize,
                         item: &SceneRenderItem,
                         solid_pl: &wgpu::RenderPipeline,
                         trans_pl: &wgpu::RenderPipeline,
                         wf_pl: &wgpu::RenderPipeline| {
                            let mesh = resources.mesh_store.get(item.mesh_id).unwrap();
                            let obj_bg = per_item_bgs
                                .get(item_idx)
                                .and_then(|opt| opt.as_ref())
                                .unwrap_or(&mesh.object_bind_group);
                            render_pass.set_bind_group(1, obj_bg, &[]);

                            let deform_bg = resources
                                .deform
                                .instance_bind_group_for(item.mesh_id, item.deform_instance);
                            let is_face_attr = item.active_attribute.as_ref().map_or(false, |a| {
                                matches!(
                                    a.kind,
                                    crate::resources::AttributeKind::Face
                                        | crate::resources::AttributeKind::FaceColour
                                        | crate::resources::AttributeKind::Halfedge
                                        | crate::resources::AttributeKind::Corner
                                )
                            });
                            if frame.viewport.wireframe_mode {
                                render_pass.set_pipeline(wf_pl);
                                render_pass.set_bind_group(2, deform_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.edge_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            } else if is_face_attr {
                                if let Some(ref fvb) = mesh.face_vertex_buffer {
                                    let pl = if item.settings.opacity < 1.0 {
                                        trans_pl
                                    } else {
                                        solid_pl
                                    };
                                    render_pass.set_pipeline(pl);
                                    render_pass.set_bind_group(2, deform_bg, &[]);
                                    render_pass.set_vertex_buffer(0, fvb.slice(..));
                                    render_pass.draw(0..mesh.index_count, 0..1);
                                }
                            } else {
                                let filter = compute_filter_results
                                    .iter()
                                    .find(|r| r.mesh_id == item.mesh_id);
                                let pl = if item.settings.opacity < 1.0 {
                                    trans_pl
                                } else {
                                    solid_pl
                                };
                                render_pass.set_pipeline(pl);
                                render_pass.set_bind_group(2, deform_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                if let Some(fr) = filter {
                                    render_pass.set_index_buffer(
                                        fr.index_buffer.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    render_pass.draw_indexed(0..fr.index_count, 0, 0..1);
                                } else {
                                    render_pass.set_index_buffer(
                                        mesh.index_buffer.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                                }
                            }
                            if item.show_normals {
                                if let Some(ref nl_buf) = mesh.normal_line_buffer {
                                    if mesh.normal_line_count > 0 {
                                        render_pass.set_pipeline(wf_pl);
                                        render_pass.set_bind_group(
                                            2,
                                            &resources.deform.dummy_bind_group,
                                            &[],
                                        );
                                        render_pass.set_bind_group(1, &mesh.normal_bind_group, &[]);
                                        render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                                        render_pass.draw(0..mesh.normal_line_count, 0..1);
                                    }
                                }
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
                        for (item_idx, item) in &opaque {
                            let solid_pl = if item.material.is_two_sided() {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            draw_item_hdr(
                                &mut render_pass,
                                *item_idx,
                                item,
                                solid_pl,
                                hdr_trans,
                                hdr_wf,
                            );
                        }
                    }
                }
            }

            // Cap fill pass (HDR path : section view cross-section fill).
            if !slot.cap_buffers.is_empty() {
                if let Some(ref hdr_overlay) = resources.hdr_overlay_pipeline {
                    render_pass.set_pipeline(hdr_overlay);
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.cap_buffers {
                        render_pass.set_bind_group(1, bg, &[]);
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }
            }

            // Scivis layers (point cloud, glyph, polyline, volume, streamtube,
            // image slice, tensor glyph, ribbon, volume surface slice, sprites).
            //
            // Sprites are routed through a separate post-pass below when SSAA is
            // depth. The post-pass targets the ssaa_* attachments and samples
            // ssaa_depth_only_view when SSAA is active, the hdr_* attachments
            // otherwise. Sprites are always skipped inline here.
            let sprite_slice_for_inline: &[crate::resources::SpriteGpuData] = &[];
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
                sprite_slice_for_inline,
                &self.mesh_instance_gpu_data,
                true
            );

            // GPU implicit surface (HDR path, before skybox).
            if !self.implicit_gpu_data.is_empty() {
                if let Some(ref dual) = self.resources.implicit_pipeline {
                    render_pass.set_pipeline(dual.for_format(true));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for gpu in &self.implicit_gpu_data {
                        render_pass.set_bind_group(1, &gpu.bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
            // GPU marching cubes indirect draw (HDR path).
            if !self.mc_gpu_data.is_empty() {
                render_pass.set_bind_group(0, camera_bg, &[]);
                for mc in &self.mc_gpu_data {
                    let vol = &self.resources.mc_volumes[mc.volume_idx];
                    if mc.wireframe || frame.viewport.wireframe_mode {
                        if let Some(ref dual) = self.resources.mc_wireframe_pipeline {
                            render_pass.set_pipeline(dual.for_format(true));
                            for (slab, wire_bg) in vol.slabs.iter().zip(mc.wire_slab_bgs.iter()) {
                                render_pass.set_bind_group(1, wire_bg, &[]);
                                render_pass.draw_indirect(&slab.wire_indirect_buf, 0);
                            }
                        }
                    } else if let Some(ref dual) = self.resources.mc_surface_pipeline {
                        render_pass.set_pipeline(dual.for_format(true));
                        render_pass.set_bind_group(1, &mc.render_bg, &[]);
                        for slab in &vol.slabs {
                            render_pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                            render_pass.draw_indirect(&slab.indirect_buf, 0);
                        }
                    }
                }
            }

            // Gaussian splats (HDR path).
            if !self.gaussian_splat_draw_data.is_empty() {
                if let Some(ref dual) = self.resources.gaussian_splat_pipeline {
                    render_pass.set_pipeline(dual.for_format(true));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for dd in &self.gaussian_splat_draw_data {
                        if dd.wireframe {
                            continue;
                        }
                        if let Some(set) = self.resources.gaussian_splat_store.get(dd.store_index) {
                            if let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) {
                                render_pass.set_bind_group(1, &vp_sort.render_bg, &[]);
                                render_pass.draw(0..6, 0..dd.count);
                            }
                        }
                    }
                }
            }
            // TransparentVolumeMesh boundary wireframe overlay (HDR path).
            if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
                if let (Some(tvm_bg), Some(hdr_wf)) = (
                    &self.mesh_uniforms.tvm_wireframe_bg,
                    &resources.hdr_wireframe_pipeline,
                ) {
                    for mesh_id in &self.mesh_uniforms.tvm_wireframe_draws {
                        if let Some(mesh) = resources.mesh_store.get(*mesh_id) {
                            render_pass.set_pipeline(hdr_wf);
                            render_pass.set_bind_group(2, &resources.deform.dummy_bind_group, &[]);
                            render_pass.set_bind_group(1, tvm_bg, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(
                                mesh.edge_index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                        }
                    }
                }
            }

            // Item-type plugin paint: after built-in opaques, before
            // skybox. Standard group-0 bindings are already bound.
            self.dispatch_plugin_paint(&mut render_pass, frame);

            // Draw skybox last among opaques : only uncovered sky pixels pass depth == 1.0.
            if show_skybox {
                render_pass.set_bind_group(0, camera_bg, &[]);
                render_pass.set_pipeline(&resources.skybox_pipeline);
                render_pass.draw(0..3, 0..1);
            }
        }
    }

    fn hdr_sprite_passes(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let device = ctx.device;
        let vp_idx = ctx.vp_idx;
        let ssaa_factor = ctx.ssaa_factor;
        // -----------------------------------------------------------------------
        // Sprite post-pass: redraws sprite batches in two passes so that the
        // soft-particle shader path can sample resolved scene depth.
        //
        // The depth-write batch runs first with the depth attachment writable
        // and the fallback group-2 bind group, since the live depth view is
        // also the attachment and cannot be sampled at the same time.
        //
        // The transparent batch runs after with the depth attachment in
        // read-only mode and the per-viewport bind group, which lets the
        // sprite shader sample the live scene depth and apply the soft fade.
        //
        // Selects the ssaa_* colour/depth/sample views when SSAA is active so
        // sprites draw at supersampled resolution and resolve with the rest of
        // the scene; otherwise targets the hdr_* views directly.
        // -----------------------------------------------------------------------
        if !self.sprite_gpu_data.is_empty() {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
            let resources = &self.resources;

            let use_ssaa = ssaa_factor > 1
                && slot_hdr.ssaa_colour_view.is_some()
                && slot_hdr.ssaa_depth_view.is_some()
                && slot_hdr.ssaa_depth_only_view.is_some();
            let colour_view = if use_ssaa {
                slot_hdr.ssaa_colour_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_view
            };
            let depth_view = if use_ssaa {
                slot_hdr.ssaa_depth_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_depth_view
            };
            let depth_only_view = if use_ssaa {
                slot_hdr.ssaa_depth_only_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_depth_only_view
            };

            let any_depth_write = self.sprite_gpu_data.iter().any(|s| s.depth_write);
            let any_transparent = self.sprite_gpu_data.iter().any(|s| !s.depth_write);

            let buckets = [
                // (depth_write, blend, lit, pipeline)
                (
                    true,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    false,
                    resources.sprite_pipeline_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Additive,
                    false,
                    resources.sprite_pipeline_additive_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Premultiplied,
                    false,
                    resources.sprite_pipeline_premultiplied_depth_write.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    false,
                    resources.sprite_pipeline.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Additive,
                    false,
                    resources.sprite_pipeline_additive.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Premultiplied,
                    false,
                    resources.sprite_pipeline_premultiplied.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    true,
                    resources.sprite_lit_pipeline_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Additive,
                    true,
                    resources.sprite_lit_pipeline_additive_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Premultiplied,
                    true,
                    resources
                        .sprite_lit_pipeline_premultiplied_depth_write
                        .as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    true,
                    resources.sprite_lit_pipeline.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Additive,
                    true,
                    resources.sprite_lit_pipeline_additive.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Premultiplied,
                    true,
                    resources.sprite_lit_pipeline_premultiplied.as_ref(),
                ),
            ];
            let lit_fallback_bg = resources.sprite_lit_fallback_bg.as_ref();

            let fallback_soft_bg = resources.sprite_soft_fallback_bg.as_ref();

            // Pass 1: depth-write sprites, depth attachment writable, fallback
            // bound at group 2 (the live depth view is aliased to the
            // attachment in this pass and cannot also be sampled).
            if any_depth_write {
                if let Some(fallback_soft_bg) = fallback_soft_bg {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("sprite_depth_write_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: colour_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    for (depth_write, blend, lit, pipeline) in &buckets {
                        if !*depth_write {
                            continue;
                        }
                        let Some(dual) = pipeline else { continue };
                        let mut bound = false;
                        for sprite in self.sprite_gpu_data.iter() {
                            if sprite.wireframe
                                || !sprite.depth_write
                                || sprite.blend != *blend
                                || sprite.lit != *lit
                                || sprite.refraction_strength > 0.0
                            {
                                continue;
                            }
                            if !bound {
                                pass.set_pipeline(dual.for_format(true));
                                pass.set_bind_group(0, camera_bg, &[]);
                                pass.set_bind_group(2, fallback_soft_bg, &[]);
                                bound = true;
                            }
                            pass.set_bind_group(1, &sprite.bind_group, &[]);
                            if *lit {
                                let normal_bg = sprite.lit_normal_bg.as_ref().or(lit_fallback_bg);
                                if let Some(bg) = normal_bg {
                                    pass.set_bind_group(3, bg, &[]);
                                }
                            }
                            pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
                            pass.draw(0..6, 0..sprite.sprite_count);
                        }
                    }
                }
            }

            // Pass 2: transparent sprites, depth attachment read-only so the
            // live depth view can be sampled by the sprite shader for fade.
            if any_transparent {
                let real_soft_bg = if let (Some(bgl), Some(sampler)) = (
                    resources.sprite_soft_bgl.as_ref(),
                    resources.sprite_soft_sampler.as_ref(),
                ) {
                    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sprite_soft_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(depth_only_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    }))
                } else {
                    None
                };

                if let Some(real_soft_bg) = real_soft_bg.as_ref() {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("sprite_transparent_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: colour_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: None,
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    for (depth_write, blend, lit, pipeline) in &buckets {
                        if *depth_write {
                            continue;
                        }
                        let Some(dual) = pipeline else { continue };
                        let mut bound = false;
                        for sprite in self.sprite_gpu_data.iter() {
                            if sprite.wireframe
                                || sprite.depth_write
                                || sprite.blend != *blend
                                || sprite.lit != *lit
                                || sprite.refraction_strength > 0.0
                            {
                                continue;
                            }
                            if !bound {
                                pass.set_pipeline(dual.for_format(true));
                                pass.set_bind_group(0, camera_bg, &[]);
                                pass.set_bind_group(2, real_soft_bg, &[]);
                                bound = true;
                            }
                            pass.set_bind_group(1, &sprite.bind_group, &[]);
                            if *lit {
                                let normal_bg = sprite.lit_normal_bg.as_ref().or(lit_fallback_bg);
                                if let Some(bg) = normal_bg {
                                    pass.set_bind_group(3, bg, &[]);
                                }
                            }
                            pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
                            pass.draw(0..6, 0..sprite.sprite_count);
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // GPU particle sprite pass: each particle system draws its full
        // capacity as billboards through a sprite-shader variant that reads
        // positions and per-instance data from the system's GPU buffer. Dead
        // particles emit a degenerate vertex and contribute no fragments.
        // -----------------------------------------------------------------------
        if !self.particle_gpu_data.is_empty() {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
            let resources = &self.resources;

            let use_ssaa = ssaa_factor > 1
                && slot_hdr.ssaa_colour_view.is_some()
                && slot_hdr.ssaa_depth_view.is_some();
            let colour_view = if use_ssaa {
                slot_hdr.ssaa_colour_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_view
            };
            let depth_view = if use_ssaa {
                slot_hdr.ssaa_depth_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_depth_view
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpu_particle_sprite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: colour_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, camera_bg, &[]);
            let particle_lit_fallback = resources.particle_sprite_lit_fallback_bg.as_ref();
            for pd in &self.particle_gpu_data {
                if pd.hidden {
                    continue;
                }
                let Some(system) = resources
                    .particle_systems
                    .get(pd.system_idx)
                    .and_then(|s| s.as_ref())
                    .filter(|s| s.alive)
                else {
                    continue;
                };
                match pd.route {
                    crate::resources::gpu_particles::ParticleDrawRoute::Sprite { lit } => {
                        let dual = match (pd.blend, lit) {
                            (crate::renderer::SpriteBlend::Additive, false) => {
                                resources.particle_sprite_pipeline_additive.as_ref()
                            }
                            (crate::renderer::SpriteBlend::Premultiplied, false) => {
                                resources.particle_sprite_pipeline_premultiplied.as_ref()
                            }
                            (crate::renderer::SpriteBlend::AlphaBlend, false) => {
                                resources.particle_sprite_pipeline_alpha.as_ref()
                            }
                            (crate::renderer::SpriteBlend::Additive, true) => {
                                resources.particle_sprite_lit_pipeline_additive.as_ref()
                            }
                            (crate::renderer::SpriteBlend::Premultiplied, true) => resources
                                .particle_sprite_lit_pipeline_premultiplied
                                .as_ref(),
                            (crate::renderer::SpriteBlend::AlphaBlend, true) => {
                                resources.particle_sprite_lit_pipeline_alpha.as_ref()
                            }
                        };
                        let Some(dual) = dual else { continue };
                        let Some(draw_bg) = system.draw_bg.as_ref() else {
                            continue;
                        };
                        pass.set_pipeline(dual.for_format(true));
                        pass.set_bind_group(1, draw_bg, &[]);
                        if lit {
                            let normal_bg =
                                system.draw_lit_normal_bg.as_ref().or(particle_lit_fallback);
                            if let Some(bg) = normal_bg {
                                pass.set_bind_group(2, bg, &[]);
                            }
                        }
                        pass.draw(0..6, 0..system.capacity);
                    }
                    crate::resources::gpu_particles::ParticleDrawRoute::Mesh { mesh_id } => {
                        let dual = match pd.blend {
                            crate::renderer::SpriteBlend::Additive => {
                                resources.particle_mesh_pipeline_additive.as_ref()
                            }
                            crate::renderer::SpriteBlend::Premultiplied => {
                                resources.particle_mesh_pipeline_premultiplied.as_ref()
                            }
                            crate::renderer::SpriteBlend::AlphaBlend => {
                                resources.particle_mesh_pipeline_alpha.as_ref()
                            }
                        };
                        let Some(dual) = dual else { continue };
                        let Some(draw_bg) = system.draw_bg_mesh.as_ref() else {
                            continue;
                        };
                        let Some(mesh) = resources.mesh_store.get(
                            crate::resources::mesh_store::MeshId::from_index(mesh_id as usize),
                        ) else {
                            continue;
                        };
                        pass.set_pipeline(dual.for_format(true));
                        pass.set_bind_group(1, draw_bg, &[]);
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..mesh.index_count, 0, 0..system.capacity);
                    }
                }
            }
        }
    }

    fn hdr_ssaa_refraction(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let device = ctx.device;
        let vp_idx = ctx.vp_idx;
        let ssaa_factor = ctx.ssaa_factor;
        // -----------------------------------------------------------------------
        // Refractive sprite pass.
        //
        // Sprites flagged with `refraction_strength` skip the normal sprite
        // pass and draw here instead. The renderer copies the HDR colour into
        // a separate resolve texture, then each refractive sprite samples
        // that texture at an offset driven by its own texture (R/G channels
        // as signed displacement, alpha as mask).
        //
        // Non-SSAA HDR path only: SSAA would need a resolve at supersampled
        // resolution and the soft-particle post-pass machinery does not yet
        // share its resolve. This matches the soft-particle constraint.
        // -----------------------------------------------------------------------
        let has_refractive = self
            .sprite_gpu_data
            .iter()
            .any(|s| s.refraction_strength > 0.0 && !s.wireframe);
        if has_refractive && ssaa_factor == 1 {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
            let resources = &self.resources;
            let physical_w = slot_hdr.hdr_texture.size().width;
            let physical_h = slot_hdr.hdr_texture.size().height;

            // Allocate or resize the resolve texture so it matches the HDR
            // attachment exactly. The copy depends on identical dimensions
            // and format (Rgba16Float). Lives in side-storage indexed by
            // viewport so the outer borrows on `viewport_slots` stay
            // immutable.
            while self.sprite_refraction_resolves.len() <= vp_idx {
                self.sprite_refraction_resolves.push(None);
            }
            let need_realloc = match self.sprite_refraction_resolves[vp_idx].as_ref() {
                Some(r) => r.size != [physical_w, physical_h],
                None => true,
            };
            if need_realloc {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("sprite_refraction_resolve"),
                    size: wgpu::Extent3d {
                        width: physical_w.max(1),
                        height: physical_h.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                self.sprite_refraction_resolves[vp_idx] = Some(SpriteRefractionResolve {
                    texture: tex,
                    view,
                    size: [physical_w, physical_h],
                });
            }
            let resolve = self.sprite_refraction_resolves[vp_idx].as_ref().unwrap();
            let resolve_tex = &resolve.texture;
            let resolve_view = &resolve.view;

            // Copy current HDR colour -> resolve texture so the refraction
            // shader can sample the scene without reading from its render
            // attachment.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &slot_hdr.hdr_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: resolve_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: physical_w.max(1),
                    height: physical_h.max(1),
                    depth_or_array_layers: 1,
                },
            );

            // Build the group-2 bind group: sampled resolve view + sampler.
            let refraction_bgl = resources.sprite_refraction_bgl.as_ref().unwrap();
            let refraction_sampler = resources.sprite_refraction_sampler.as_ref().unwrap();
            let refraction_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sprite_refraction_bg"),
                layout: refraction_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(resolve_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(refraction_sampler),
                    },
                ],
            });

            if let Some(pipeline) = resources.sprite_refraction_pipeline.as_ref() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sprite_refraction_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, camera_bg, &[]);
                pass.set_bind_group(2, &refraction_bg, &[]);
                for sprite in self.sprite_gpu_data.iter() {
                    if sprite.refraction_strength <= 0.0 || sprite.wireframe {
                        continue;
                    }
                    pass.set_bind_group(1, &sprite.bind_group, &[]);
                    pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
                    pass.draw(0..6, 0..sprite.sprite_count);
                }
            }
        }

        // -----------------------------------------------------------------------
        // SSAA resolve pass: downsample supersampled scene -> hdr_texture.
        // Only runs when ssaa_factor > 1 and the resolve pipeline is available.
        // -----------------------------------------------------------------------
        if ssaa_factor > 1 {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let (Some(pipeline), Some(bg)) = (
                &self.resources.ssaa_resolve_pipeline,
                &slot_hdr.ssaa_resolve_bind_group,
            ) {
                let mut resolve_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ssaa_resolve_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
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
                resolve_pass.set_pipeline(pipeline);
                resolve_pass.set_bind_group(0, bg, &[]);
                resolve_pass.draw(0..3, 0..1);
            }
        }
    }

    fn hdr_decals(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let vp_idx = ctx.vp_idx;
        // -----------------------------------------------------------------------
        // Decal exclude pass (D5): stamp stencil = 0 on non-receiver surfaces.
        // Runs after the opaque pass, before the decal pass.
        // -----------------------------------------------------------------------
        if !self.decal_exclude_items.is_empty() {
            if let Some(exclude_pl) = self.resources.decal_exclude_pipeline.as_ref() {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("decal_exclude_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(exclude_pl);
                pass.set_stencil_reference(0);
                pass.set_bind_group(0, camera_bg, &[]);
                for item in &self.decal_exclude_items {
                    if let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) {
                        pass.set_bind_group(1, &item.bind_group, &[]);
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        if mesh.index_count > 0 {
                            pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Decal pass (D1): projects each decal texture onto opaque surfaces.
        // Reads scene depth as a texture; no depth attachment.
        // Runs after opaque geometry and SSAA resolve, before transparent passes.
        // -----------------------------------------------------------------------
        if !self.decal_gpu_data.is_empty() {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
            let depth_bg = &slot_hdr.decal_depth_bg;
            let replace_pipeline = self.resources.decal_replace_pipeline.as_ref();
            let multiply_pipeline = self.resources.decal_multiply_pipeline.as_ref();
            let additive_pipeline = self.resources.decal_additive_pipeline.as_ref();
            if replace_pipeline.is_some()
                || multiply_pipeline.is_some()
                || additive_pipeline.is_some()
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("decal_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
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
                pass.set_bind_group(0, camera_bg, &[]);
                pass.set_bind_group(1, depth_bg, &[]);
                for gpu in &self.decal_gpu_data {
                    let pipeline = match gpu.blend_mode {
                        crate::renderer::DecalBlendMode::Replace => replace_pipeline,
                        crate::renderer::DecalBlendMode::Multiply => multiply_pipeline,
                        crate::renderer::DecalBlendMode::Additive => additive_pipeline,
                    };
                    if let Some(pl) = pipeline {
                        pass.set_pipeline(&pl.hdr);
                        pass.set_bind_group(2, &gpu.bind_group, &[]);
                        pass.draw(0..6, 0..1);
                    }
                }
            }
        }
    }

    fn hdr_sub_highlight(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let vp_idx = ctx.vp_idx;
        // -----------------------------------------------------------------------
        // Sub-object highlight pass: face fill, edge lines, vertex sprites.
        // Runs after opaque geometry (depth buffer is ready) and before OIT so
        // highlights are not occluded by opaque surfaces.
        // -----------------------------------------------------------------------
        if let Some(sub_hl) = self.viewport_slots[vp_idx].sub_highlight.as_ref() {
            let resources = &self.resources;
            if let (Some(fill_pl), Some(edge_pl), Some(sprite_pl)) = (
                &resources.sub_highlight_fill_pipeline,
                &resources.sub_highlight_edge_pipeline,
                &resources.sub_highlight_sprite_pipeline,
            ) {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sub_highlight_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            // Store even though depth_write_enabled=false on all
                            // sub-highlight pipelines: the values are unchanged, but
                            // StoreOp::Discard would invalidate the tile on Metal and
                            // cause subsequent passes (tone_map, grid, screen images)
                            // to read 0.0, making the background go black.
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if sub_hl.fill_vertex_count > 0 {
                    pass.set_pipeline(fill_pl);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(1, &sub_hl.fill_bind_group, &[]);
                    pass.set_vertex_buffer(0, sub_hl.fill_vertex_buf.slice(..));
                    pass.draw(0..sub_hl.fill_vertex_count, 0..1);
                }
                if sub_hl.edge_segment_count > 0 {
                    pass.set_pipeline(edge_pl);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(1, &sub_hl.edge_bind_group, &[]);
                    pass.set_vertex_buffer(0, sub_hl.edge_vertex_buf.slice(..));
                    pass.draw(0..6, 0..sub_hl.edge_segment_count);
                }
                if sub_hl.sprite_point_count > 0 {
                    pass.set_pipeline(sprite_pl);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(1, &sub_hl.sprite_bind_group, &[]);
                    pass.set_vertex_buffer(0, sub_hl.sprite_vertex_buf.slice(..));
                    pass.draw(0..6, 0..sub_hl.sprite_point_count);
                }
            }
        }
    }

    fn hdr_oit(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let device = ctx.device;
        let queue = ctx.queue;
        let frame = ctx.frame;
        let scene_items = ctx.scene_items;
        let vp_idx = ctx.vp_idx;
        let slot = &self.viewport_slots[vp_idx];
        let camera_bg = &slot.camera_bind_group;
        let slot_hdr = slot.hdr.as_ref().unwrap();
        // -----------------------------------------------------------------------
        // OIT pass: render transparent items into accum + reveal textures.
        // Completely skipped when no transparent items exist (zero overhead).
        // -----------------------------------------------------------------------
        let has_transparent =
            if self.instancing.use_instancing && !self.instancing.batches.is_empty() {
                // Transparent instanced batches go through OIT. Transparent excluded items
                // (two-sided, active-attribute, matcap) are not in any instanced batch, so
                // they must also be checked here -- otherwise the OIT pass is skipped and
                // those items are invisible.
                self.instancing.batches.iter().any(|b| b.is_transparent)
                    || scene_items.iter().any(|i| {
                        // A transparent item that is not instanceable is drawn per-object
                        // in the OIT pass below; if any exists the pass must run.
                        !i.settings.hidden
                            && (i.settings.opacity < 1.0 || i.material.is_blend())
                            && !crate::renderer::prepare::is_instanceable(
                                i,
                                &self.resources,
                                &self.compute_filter_results,
                            )
                    })
            } else {
                scene_items.iter().any(|i| {
                    !i.settings.hidden && (i.settings.opacity < 1.0 || i.material.is_blend())
                })
            } || frame
                .scene
                .volume_meshes
                .iter()
                .any(|i| !i.settings.hidden && i.transparency.is_some());

        if has_transparent {
            // OIT targets already allocated in the pre-pass above.
            if let (Some(accum_view), Some(reveal_view)) = (
                slot_hdr.oit_accum_view.as_ref(),
                slot_hdr.oit_reveal_view.as_ref(),
            ) {
                let hdr_depth_view = &slot_hdr.hdr_depth_view;
                let oit_ts_writes = self.ts_query_set.as_ref().map(|qs| {
                    self.ts_written_mask.fetch_or(
                        1 << crate::renderer::GPU_TS_OIT,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    wgpu::RenderPassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(crate::renderer::GPU_TS_OIT * 2),
                        end_of_pass_write_index: Some(crate::renderer::GPU_TS_OIT * 2 + 1),
                    }
                });
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
                    timestamp_writes: oit_ts_writes,
                    occlusion_query_set: None,
                });

                oit_pass.set_bind_group(0, camera_bg, &[]);

                if self.instancing.use_instancing && !self.instancing.batches.is_empty() {
                    let use_indirect_oit = self.instancing.gpu_culling_enabled
                        && self.resources.oit_instanced_cull_pipeline.is_some()
                        && self.resources.indirect_args_buf.is_some();

                    if use_indirect_oit {
                        if let (Some(pipeline), Some(indirect_buf)) = (
                            &self.resources.oit_instanced_cull_pipeline,
                            &self.resources.indirect_args_buf,
                        ) {
                            oit_pass.set_pipeline(pipeline);
                            oit_pass.set_bind_group(
                                2,
                                &self.resources.deform.dummy_bind_group,
                                &[],
                            );
                            for (batch_global_idx, batch) in
                                self.instancing.batches.iter().enumerate()
                            {
                                if !batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = self.resources.mesh_store.get(batch.mesh_id)
                                else {
                                    continue;
                                };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) =
                                    self.resources.instance_cull_bind_groups.get(&mat_key)
                                else {
                                    continue;
                                };
                                oit_pass.set_bind_group(1, inst_tex_bg, &[]);
                                oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                oit_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                oit_pass.draw_indexed_indirect(
                                    indirect_buf,
                                    batch_global_idx as u64 * 20,
                                );
                            }
                        }
                    } else if let Some(ref pipeline) = self.resources.oit_instanced_pipeline {
                        oit_pass.set_pipeline(pipeline);
                        oit_pass.set_bind_group(2, &self.resources.deform.dummy_bind_group, &[]);
                        for batch in &self.instancing.batches {
                            if !batch.is_transparent {
                                continue;
                            }
                            let Some(mesh) = self.resources.mesh_store.get(batch.mesh_id) else {
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

                    // Transparent excluded items (two-sided, active attribute, matcap) are not
                    // in any instanced batch, so the instanced OIT loop above skips them.
                    // Render them here individually so they are not invisible at opacity < 1.
                    if let Some(ref pipeline) = self.resources.oit_pipeline {
                        oit_pass.set_pipeline(pipeline);
                        for (item_idx, item) in scene_items.iter().enumerate() {
                            if item.settings.hidden
                                || (item.settings.opacity >= 1.0 && !item.material.is_blend())
                            {
                                continue;
                            }
                            // Instanceable transparent items go through the instanced OIT
                            // path; only the per-object (non-instanceable) ones draw here.
                            if crate::renderer::prepare::is_instanceable(
                                item,
                                &self.resources,
                                &self.compute_filter_results,
                            ) {
                                continue;
                            }
                            let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };
                            let deform_bg = self
                                .resources
                                .deform
                                .instance_bind_group_for(item.mesh_id, item.deform_instance);
                            let obj_bg = self
                                .mesh_uniforms
                                .bind_groups
                                .get(item_idx)
                                .and_then(|opt| opt.as_ref())
                                .unwrap_or(&mesh.object_bind_group);
                            oit_pass.set_bind_group(1, obj_bg, &[]);
                            oit_pass.set_bind_group(2, deform_bg, &[]);
                            oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            oit_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            oit_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
                } else if let Some(ref pipeline) = self.resources.oit_pipeline {
                    oit_pass.set_pipeline(pipeline);
                    for (item_idx, item) in scene_items.iter().enumerate() {
                        if item.settings.hidden
                            || (item.settings.opacity >= 1.0 && !item.material.is_blend())
                        {
                            continue;
                        }
                        let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                            continue;
                        };
                        let deform_bg = self
                            .resources
                            .deform
                            .instance_bind_group_for(item.mesh_id, item.deform_instance);
                        let obj_bg = self
                            .mesh_uniforms
                            .bind_groups
                            .get(item_idx)
                            .and_then(|opt| opt.as_ref())
                            .unwrap_or(&mesh.object_bind_group);
                        oit_pass.set_bind_group(1, obj_bg, &[]);
                        oit_pass.set_bind_group(2, deform_bg, &[]);
                        oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        oit_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        oit_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }

                // -----------------------------------------------------------
                // Projected tetrahedra transparent volume meshes.
                // Items with `transparency: Some(_)` route here; opaque items
                // already drew through the surface pipeline.
                // -----------------------------------------------------------
                let any_transparent = frame
                    .scene
                    .volume_meshes
                    .iter()
                    .any(|i| !i.settings.hidden && i.transparency.is_some());
                if any_transparent {
                    self.resources.ensure_pt_pipeline(device);
                    // Pre-build LUT bind groups for every unique colourmap the
                    // current frame's transparent items reference, so the draw
                    // loop below can borrow them immutably from the cache.
                    for item in &frame.scene.volume_meshes {
                        if item.settings.hidden || item.transparency.is_none() {
                            continue;
                        }
                        self.resources
                            .ensure_pt_lut_bind_group(device, item.colourmap_id);
                    }
                    if let Some(pipeline) = self.resources.pt_pipeline.as_ref() {
                        oit_pass.set_pipeline(pipeline);
                        oit_pass.set_bind_group(0, camera_bg, &[]);
                        let resources = &self.resources;
                        for item in &frame.scene.volume_meshes {
                            if item.settings.hidden {
                                continue;
                            }
                            let Some(transparency) = item.transparency else {
                                continue;
                            };
                            if item.settings.wireframe || frame.viewport.wireframe_mode {
                                continue;
                            }
                            let Some(pt_id) = item.projected_tet_id else {
                                continue;
                            };
                            let Some(gpu) = resources.projected_tet_store.get(pt_id.0) else {
                                continue;
                            };
                            let (scalar_min, scalar_max) =
                                item.scalar_range.unwrap_or(gpu.scalar_range);
                            let uniform = crate::resources::ProjectedTetUniform {
                                density: transparency.density,
                                scalar_min,
                                scalar_max,
                                threshold_min: transparency.threshold_min,
                                threshold_max: transparency.threshold_max,
                                unlit: if item.settings.unlit { 1 } else { 0 },
                                opacity: item.settings.opacity,
                                _pad: 0.0,
                            };
                            queue.write_buffer(
                                &gpu.uniform_buffer,
                                0,
                                bytemuck::bytes_of(&uniform),
                            );
                            // Look up the pre-built LUT bind group (cache miss
                            // is impossible because we populated it above).
                            let lut_bg = item
                                .colourmap_id
                                .and_then(|id| {
                                    resources
                                        .colourmap_views
                                        .get(id.0)
                                        .and(resources.pt_lut_bind_groups.get(&id.0))
                                })
                                .or(resources.pt_fallback_lut_bind_group.as_ref());
                            let Some(lut_bg) = lut_bg else { continue };
                            oit_pass.set_bind_group(2, lut_bg, &[]);
                            for chunk in &gpu.chunks {
                                oit_pass.set_bind_group(1, &chunk.bind_group, &[]);
                                oit_pass.draw(0..6, 0..chunk.tet_count);
                            }
                        }
                    }
                }

                // Item-type plugin transparent draws.
                self.dispatch_plugin_paint_transparent(&mut oit_pass, frame);
            }
        }

        // -----------------------------------------------------------------------
        // OIT composite pass: blend accum/reveal into HDR buffer.
        // Only executes when transparent items were present.
        // -----------------------------------------------------------------------
        if has_transparent {
            if let (Some(pipeline), Some(bg)) = (
                self.resources.oit_composite_pipeline.as_ref(),
                slot_hdr.oit_composite_bind_group.as_ref(),
            ) {
                let hdr_view = &slot_hdr.hdr_view;
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
    }

    fn hdr_scatter(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let device = ctx.device;
        let queue = ctx.queue;
        let frame = ctx.frame;
        let vp_idx = ctx.vp_idx;
        let slot = &self.viewport_slots[vp_idx];
        let camera_bg = &slot.camera_bind_group;
        let slot_hdr = slot.hdr.as_ref().unwrap();
        // -----------------------------------------------------------------------
        // Scatter-volume pass: render each visible volume as an instanced
        // draw whose vertex shader projects the world bounding box and emits
        // a screen-space quad. Per-volume draws accumulate into
        // `raw_current` in back-to-front order. Optional temporal-resolve
        // and composite passes follow.
        // -----------------------------------------------------------------------
        if !self.prepared_scatter_volumes.is_empty() {
            let scatter = &frame.effects.scatter;
            let [sw, sh] = slot_hdr.scene_size;
            let want_downsample = scatter.downsample;
            let (tw, th) = if want_downsample {
                ((sw / 2).max(1), (sh / 2).max(1))
            } else {
                (sw.max(1), sh.max(1))
            };

            // Grow the per-viewport scatter state slot lazily.
            while self.scatter_viewport_states.len() <= vp_idx {
                self.scatter_viewport_states.push(None);
            }

            // Pipelines.
            self.resources
                .ensure_scatter_pipeline(device, wgpu::TextureFormat::Rgba16Float);
            self.resources
                .ensure_scatter_composite_pipeline(device, wgpu::TextureFormat::Rgba16Float);
            self.resources
                .ensure_scatter_temporal_resolve_pipeline(device);

            // (Re)allocate intermediates if requested size / mode changed.
            let needs_alloc = match self.scatter_viewport_states[vp_idx].as_ref() {
                None => true,
                Some(s) => s.size != [tw, th] || s.downsampled != want_downsample,
            };
            if needs_alloc {
                let make_tex = |label: &str| {
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width: tw,
                            height: th,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                };
                let raw_tex = make_tex("scatter_raw_current");
                let hist_a_tex = make_tex("scatter_history_a");
                let hist_b_tex = make_tex("scatter_history_b");
                let raw_view = raw_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let hist_a_view = hist_a_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let hist_b_view = hist_b_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let composite_bg_raw = self.resources.make_scatter_composite_bg(device, &raw_view);
                let composite_bg_history_a = self
                    .resources
                    .make_scatter_composite_bg(device, &hist_a_view);
                let composite_bg_history_b = self
                    .resources
                    .make_scatter_composite_bg(device, &hist_b_view);
                let temporal_resolve_bg_read_a = self.resources.make_scatter_temporal_resolve_bg(
                    device,
                    queue,
                    &raw_view,
                    &hist_a_view,
                    &slot_hdr.hdr_depth_only_view,
                );
                let temporal_resolve_bg_read_b = self.resources.make_scatter_temporal_resolve_bg(
                    device,
                    queue,
                    &raw_view,
                    &hist_b_view,
                    &slot_hdr.hdr_depth_only_view,
                );
                self.scatter_viewport_states[vp_idx] =
                    Some(crate::resources::ScatterViewportState {
                        raw_current_texture: raw_tex,
                        raw_current_view: raw_view,
                        history_a_texture: hist_a_tex,
                        history_a_view: hist_a_view,
                        history_b_texture: hist_b_tex,
                        history_b_view: hist_b_view,
                        composite_bg_raw,
                        composite_bg_history_a,
                        composite_bg_history_b,
                        temporal_resolve_bg_read_a,
                        temporal_resolve_bg_read_b,
                        size: [tw, th],
                        downsampled: want_downsample,
                        parity: 0,
                        history_valid: false,
                        prev_view_proj: [[0.0; 4]; 4],
                        refraction_source_texture: None,
                        refraction_source_view: None,
                        refraction_source_bg: None,
                        refraction_blit_bg: None,
                        refraction_source_size: [0, 0],
                    });
            }

            let (parity, history_valid, prev_view_proj) = {
                let s = self.scatter_viewport_states[vp_idx].as_ref().unwrap();
                (s.parity, s.history_valid, s.prev_view_proj)
            };

            // -----------------------------------------------------------------
            // Refraction pass: distort the scene colour behind each refractive
            // volume. Runs before the scatter pass so absorption and
            // in-scattering apply on top of the shimmered scene. Skipped
            // entirely when no volume has `refraction = Some(...)`.
            // -----------------------------------------------------------------
            if !self.prepared_refraction_volumes.is_empty() {
                self.resources
                    .ensure_scatter_refraction_pipeline(device, wgpu::TextureFormat::Rgba16Float);
                self.resources.ensure_scatter_refraction_blit_pipeline(
                    device,
                    wgpu::TextureFormat::Rgba16Float,
                );

                // Allocate (or resize) the refraction source texture at HDR
                // resolution. Replaces the per-viewport entry's view when the
                // scene size changes.
                let need_realloc = {
                    let s = self.scatter_viewport_states[vp_idx].as_ref().unwrap();
                    s.refraction_source_view.is_none() || s.refraction_source_size != [sw, sh]
                };
                if need_realloc {
                    let src_tex = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("scatter_refraction_source"),
                        size: wgpu::Extent3d {
                            width: sw.max(1),
                            height: sh.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let blit_bg = self
                        .resources
                        .make_scatter_composite_bg(device, &slot_hdr.hdr_view);
                    let source_bg = self.resources.make_scatter_refraction_source_bg(
                        device,
                        &src_view,
                        &slot_hdr.hdr_depth_only_view,
                    );
                    let s = self.scatter_viewport_states[vp_idx].as_mut().unwrap();
                    s.refraction_source_texture = Some(src_tex);
                    s.refraction_source_view = Some(src_view);
                    s.refraction_source_bg = Some(source_bg);
                    s.refraction_blit_bg = Some(blit_bg);
                    s.refraction_source_size = [sw, sh];
                }

                let time_seconds = self.start_instant.elapsed().as_secs_f32();
                let n_ref = self.resources.write_scatter_refraction_per_volume_buffer(
                    device,
                    queue,
                    &self.prepared_refraction_volumes,
                    time_seconds,
                );
                let ref_stride = self.resources.scatter_refraction_per_volume_stride();

                if n_ref > 0 {
                    let s = self.scatter_viewport_states[vp_idx].as_ref().unwrap();
                    let src_view = s.refraction_source_view.as_ref().unwrap();
                    let blit_bg = s.refraction_blit_bg.as_ref().unwrap();
                    let source_bg = s.refraction_source_bg.as_ref().unwrap();

                    // Blit-copy HDR -> refraction source (replace blend).
                    if let Some(blit_pipeline) =
                        self.resources.scatter_refraction_blit_pipeline.as_ref()
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("scatter_refraction_blit_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: src_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        pass.set_pipeline(blit_pipeline);
                        pass.set_bind_group(0, blit_bg, &[]);
                        pass.draw(0..3, 0..1);
                    }

                    // Per-volume distortion pass: write distorted samples into
                    // the HDR target.
                    if let (Some(pipeline), Some(per_vol_bg)) = (
                        self.resources.scatter_refraction_pipeline.as_ref(),
                        self.resources.scatter_refraction_per_volume_bg.as_ref(),
                    ) {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("scatter_refraction_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &slot_hdr.hdr_view,
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
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, camera_bg, &[]);
                        pass.set_bind_group(2, source_bg, &[]);
                        for i in 0..n_ref {
                            let dyn_offset = i * ref_stride;
                            pass.set_bind_group(1, per_vol_bg, &[dyn_offset]);
                            pass.draw(0..6, 0..1);
                        }
                    }
                }
            }

            // Per-volume uniform buffer.
            self.resources.clear_scatter_per_volume_tex_cache();
            let n = self.resources.write_scatter_per_volume_buffer(
                device,
                queue,
                &self.prepared_scatter_volumes,
            );
            let stride = self.resources.scatter_per_volume_stride();

            // Per-frame uniform + frame bind group.
            let time_seconds = self.start_instant.elapsed().as_secs_f32();
            let global_steps = scatter.quality.default_steps();
            let depth_token = (vp_idx as u64).wrapping_mul(1_000_003)
                ^ (tw as u64).wrapping_mul(7919)
                ^ (th as u64).wrapping_mul(31);
            self.resources.write_scatter_frame_uniform(
                device,
                queue,
                &slot_hdr.hdr_depth_only_view,
                depth_token,
                time_seconds,
                global_steps,
                scatter.blue_noise_jitter,
                self.frame_counter,
            );

            // Temporal uniform (used by the optional resolve pass).
            if scatter.temporal {
                self.resources.write_scatter_temporal_uniform(
                    device,
                    queue,
                    prev_view_proj,
                    scatter.temporal_blend,
                    history_valid,
                );
            }

            // Pre-build per-volume texture bind groups (cached by ids).
            let mut per_vol_tex_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(n as usize);
            for (volume, _, _) in self.prepared_scatter_volumes.iter().take(n as usize) {
                let (lut_id, density_id) =
                    crate::resources::ViewportGpuResources::scatter_volume_tex_ids(volume);
                let bg = self
                    .resources
                    .ensure_scatter_per_volume_tex_bg(device, queue, lut_id, density_id);
                per_vol_tex_bgs.push(bg);
            }

            if n > 0 {
                let s = self.scatter_viewport_states[vp_idx].as_ref().unwrap();
                let raw_view = &s.raw_current_view;
                if let (Some(pipeline), Some(per_vol_bg), Some(frame_bg)) = (
                    self.resources.scatter_pipeline.as_ref(),
                    self.resources.scatter_per_volume_bg.as_ref(),
                    self.resources.scatter_frame_bg.as_ref(),
                ) {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("scatter_volume_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: raw_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(3, frame_bg, &[]);
                    for i in 0..n {
                        let dyn_offset = i * stride;
                        pass.set_bind_group(1, per_vol_bg, &[dyn_offset]);
                        pass.set_bind_group(2, &per_vol_tex_bgs[i as usize], &[]);
                        pass.draw(0..6, 0..1);
                    }
                }

                // Optional temporal-resolve pass.
                let source_for_composite: &wgpu::BindGroup = if scatter.temporal {
                    let s = self.scatter_viewport_states[vp_idx].as_ref().unwrap();
                    // parity = which slot to write next (history_new).
                    let (resolve_bg, target_view, source_composite) = if parity == 0 {
                        // history_prev = B, write history_new = A.
                        (
                            &s.temporal_resolve_bg_read_b,
                            &s.history_a_view,
                            &s.composite_bg_history_a,
                        )
                    } else {
                        (
                            &s.temporal_resolve_bg_read_a,
                            &s.history_b_view,
                            &s.composite_bg_history_b,
                        )
                    };
                    if let Some(resolve_pipeline) =
                        self.resources.scatter_temporal_resolve_pipeline.as_ref()
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("scatter_temporal_resolve_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        pass.set_pipeline(resolve_pipeline);
                        pass.set_bind_group(0, resolve_bg, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    source_composite
                } else {
                    &s.composite_bg_raw
                };

                // Composite pass: source -> HDR with premultiplied alpha-over.
                if let Some(composite_pipeline) = self.resources.scatter_composite_pipeline.as_ref()
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("scatter_composite_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &slot_hdr.hdr_view,
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
                    pass.set_pipeline(composite_pipeline);
                    pass.set_bind_group(0, source_for_composite, &[]);
                    pass.draw(0..3, 0..1);
                }

                // Advance ping-pong + history for next frame.
                let s = self.scatter_viewport_states[vp_idx].as_mut().unwrap();
                s.prev_view_proj = frame.camera.render_camera.view_proj().to_cols_array_2d();
                s.parity = 1 - s.parity;
                s.history_valid = scatter.temporal;
            } else if let Some(s) = self.scatter_viewport_states[vp_idx].as_mut() {
                s.history_valid = false;
            }
        }
    }

    fn hdr_lic(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let vp_idx = ctx.vp_idx;
        let slot = &self.viewport_slots[vp_idx];
        let slot_hdr = slot.hdr.as_ref().unwrap();
        // -----------------------------------------------------------------------
        // Surface LIC passes.
        // Pass 1: render each LIC mesh into lic_vector_texture (Rgba8Unorm).
        // Pass 2: advect fullscreen triangle into lic_output_texture (R8Unorm).
        // -----------------------------------------------------------------------
        if !self.lic_gpu_data.is_empty() {
            if let (Some(surface_pipeline), Some(advect_pipeline)) = (
                self.resources.lic_surface_pipeline.as_ref(),
                self.resources.lic_advect_pipeline.as_ref(),
            ) {
                let camera_bg = &slot.camera_bind_group;
                // Pass 1: surface vector pass (clears lic_vector_texture first).
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("lic_surface_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &slot_hdr.lic_vector_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(surface_pipeline);
                    pass.set_bind_group(0, camera_bg, &[]);
                    for gpu in &self.lic_gpu_data {
                        let Some(mesh) = self.resources.mesh_store.get(gpu.mesh_id) else {
                            continue;
                        };
                        let Some(vec_buf) =
                            mesh.vector_attribute_buffers.get(&gpu.vector_attribute)
                        else {
                            continue;
                        };
                        pass.set_bind_group(1, &gpu.bind_group, &[]);
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pass.set_vertex_buffer(1, vec_buf.slice(..));
                        pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
                // Pass 2: advect pass (fullscreen, writes LIC intensity to lic_output_texture).
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("lic_advect_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &slot_hdr.lic_output_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.5,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(advect_pipeline);
                    pass.set_bind_group(0, &slot_hdr.lic_advect_bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        }
    }

    fn hdr_outline_and_post(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let vp_idx = ctx.vp_idx;
        let frame = ctx.frame;
        let pp = &frame.effects.post_process;
        let slot = &self.viewport_slots[vp_idx];
        let slot_hdr = slot.hdr.as_ref().unwrap();
        // -----------------------------------------------------------------------
        // Outline composite pass (HDR path): blit offscreen outline onto hdr_view.
        // Runs after the HDR scene pass (which has depth+stencil) in a separate
        // pass with no depth attachment, so the composite pipeline is compatible.
        // -----------------------------------------------------------------------
        if !slot.outline_object_buffers.is_empty()
            || !slot.splat_outline_buffers.is_empty()
            || !slot.streamtube_outline_items.is_empty()
            || !slot.tube_outline_items.is_empty()
            || !slot.ribbon_outline_items.is_empty()
            || !slot.polyline_outline_indices.is_empty()
            || !slot.volume_outline_indices.is_empty()
            || !slot.glyph_outline_indices.is_empty()
            || !slot.tensor_glyph_outline_indices.is_empty()
            || !slot.sprite_outline_indices.is_empty()
            || !slot.raw_geom_outline_buffers.is_empty()
            || !slot.screen_rect_outline_buffers.is_empty()
            || !slot.implicit_outline_indices.is_empty()
            || !slot.mc_outline_data.is_empty()
        {
            // Prefer the HDR-format pipeline; fall back to LDR single-sample.
            let hdr_pipeline = self
                .resources
                .outline_composite_pipeline_hdr
                .as_ref()
                .or(self.resources.outline_composite_pipeline_single.as_ref());
            if let Some(pipeline) = hdr_pipeline {
                let bg = &slot_hdr.outline_composite_bind_group;
                let hdr_view = &slot_hdr.hdr_view;
                let hdr_depth_view = &slot_hdr.hdr_depth_view;
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
                            store: wgpu::StoreOp::Store,
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

        // Effect throttling. Flag was computed in prepare() so that
        // FrameStats reports exactly what fired rather than an approximation.
        let throttle_effects = self.degradation_effects_throttled;

        // -----------------------------------------------------------------------
        // SSAO pass.
        // -----------------------------------------------------------------------
        if pp.ssao && !throttle_effects {
            if let Some(ssao_pipeline) = &self.resources.ssao_pipeline {
                {
                    let mut ssao_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ssao_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &slot_hdr.ssao_view,
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
                    ssao_pass.set_bind_group(0, &slot_hdr.ssao_bg, &[]);
                    ssao_pass.draw(0..3, 0..1);
                }

                // SSAO blur pass.
                if let Some(ssao_blur_pipeline) = &self.resources.ssao_blur_pipeline {
                    let mut ssao_blur_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("ssao_blur_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &slot_hdr.ssao_blur_view,
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
                    ssao_blur_pass.set_bind_group(0, &slot_hdr.ssao_blur_bg, &[]);
                    ssao_blur_pass.draw(0..3, 0..1);
                }
            }
        }

        // -----------------------------------------------------------------------
        // Contact shadow pass.
        // -----------------------------------------------------------------------
        if pp.contact_shadows && !throttle_effects {
            if let Some(cs_pipeline) = &self.resources.contact_shadow_pipeline {
                let mut cs_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("contact_shadow_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.contact_shadow_view,
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
                cs_pass.set_bind_group(0, &slot_hdr.contact_shadow_bg, &[]);
                cs_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Bloom passes.
        // -----------------------------------------------------------------------
        if pp.bloom && !throttle_effects {
            // Threshold pass: extract bright pixels into bloom_threshold_texture.
            if let Some(bloom_threshold_pipeline) = &self.resources.bloom_threshold_pipeline {
                {
                    let mut threshold_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("bloom_threshold_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &slot_hdr.bloom_threshold_view,
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
                    threshold_pass.set_bind_group(0, &slot_hdr.bloom_threshold_bg, &[]);
                    threshold_pass.draw(0..3, 0..1);
                }

                // 4 ping-pong H+V blur passes for a wide glow.
                // Pass 1: threshold -> ping -> pong. Passes 2-4: pong -> ping -> pong.
                if let Some(blur_pipeline) = &self.resources.bloom_blur_pipeline {
                    let blur_h_bg = &slot_hdr.bloom_blur_h_bg;
                    let blur_h_pong_bg = &slot_hdr.bloom_blur_h_pong_bg;
                    let blur_v_bg = &slot_hdr.bloom_blur_v_bg;
                    let bloom_ping_view = &slot_hdr.bloom_ping_view;
                    let bloom_pong_view = &slot_hdr.bloom_pong_view;
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
        // Depth of field pass: HDR + depth -> dof_texture (when enabled).
        // -----------------------------------------------------------------------
        if pp.dof_enabled && !throttle_effects {
            if let Some(dof_pipeline) = &self.resources.dof_pipeline {
                let mut dof_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("dof_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.dof_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                dof_pass.set_pipeline(dof_pipeline);
                dof_pass.set_bind_group(0, &slot_hdr.dof_bg, &[]);
                dof_pass.draw(0..3, 0..1);
            }
        }
    }

    fn hdr_tonemap_resolve(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        let frame = ctx.frame;
        let pp = &frame.effects.post_process;
        let slot = &self.viewport_slots[vp_idx];
        let slot_hdr = slot.hdr.as_ref().unwrap();
        // -----------------------------------------------------------------------
        // Tone map pass: HDR + bloom + AO -> tone-mapped LDR.
        //
        // When render_scale < 1.0 the entire post-process chain runs at scene
        // resolution. The result lands in upscale_view (scene-res) and is then
        // upscale-blitted to output_view at native resolution.
        // -----------------------------------------------------------------------
        let use_fxaa = pp.fxaa;
        let use_hdr_upscale = slot_hdr.upscale_bind_group.is_some();
        if let Some(tone_map_pipeline) = &self.resources.tone_map_pipeline {
            let tone_target: &wgpu::TextureView = if use_fxaa {
                &slot_hdr.fxaa_view
            } else if use_hdr_upscale {
                slot_hdr.upscale_view.as_ref().unwrap()
            } else {
                output_view
            };
            let tone_ts_writes = self.ts_query_set.as_ref().map(|qs| {
                self.ts_written_mask.fetch_or(
                    1 << crate::renderer::GPU_TS_POST,
                    std::sync::atomic::Ordering::Relaxed,
                );
                wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(crate::renderer::GPU_TS_POST * 2),
                    end_of_pass_write_index: Some(crate::renderer::GPU_TS_POST * 2 + 1),
                }
            });
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
                timestamp_writes: tone_ts_writes,
                occlusion_query_set: None,
            });
            tone_pass.set_pipeline(tone_map_pipeline);
            tone_pass.set_bind_group(0, &slot_hdr.tone_map_bind_group, &[]);
            tone_pass.draw(0..3, 0..1);
        }

        // -----------------------------------------------------------------------
        // FXAA pass: fxaa_texture -> upscale_view (scaled) or output_view (1:1).
        // -----------------------------------------------------------------------
        if use_fxaa {
            if let Some(fxaa_pipeline) = &self.resources.fxaa_pipeline {
                let fxaa_target: &wgpu::TextureView = if use_hdr_upscale {
                    slot_hdr.upscale_view.as_ref().unwrap()
                } else {
                    output_view
                };
                let mut fxaa_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("fxaa_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: fxaa_target,
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
                fxaa_pass.set_bind_group(0, &slot_hdr.fxaa_bind_group, &[]);
                fxaa_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // HDR upscale pass: blit scene-resolution post-processed output to native
        // output_view. Only runs when render_scale < 1.0.
        // -----------------------------------------------------------------------
        if use_hdr_upscale {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let Some(upscale_bg) = &slot_hdr.upscale_bind_group {
                if let Some(pipeline) = &self.resources.dyn_res_upscale_pipeline {
                    let mut upscale_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("hdr_upscale_pass"),
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
                    upscale_pass.set_pipeline(pipeline);
                    upscale_pass.set_bind_group(0, upscale_bg, &[]);
                    upscale_pass.draw(0..3, 0..1);
                }
            }
        }

        // Depth blit pass: when render_scale < 1.0, the scene depth texture is
        // smaller than the output surface. Copy it to output_depth_texture (native
        // resolution) so the post-tone-map passes below can attach output_depth_view
        // alongside output_view without a size mismatch. Skipped when render_scale
        // is 1.0 (output_depth_view is just a second view of hdr_depth_texture).
        {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let Some(blit_bg) = &slot_hdr.depth_blit_bind_group {
                if let Some(blit_pipeline) = &self.resources.depth_blit_pipeline {
                    let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("depth_blit_pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &slot_hdr.output_depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    blit_pass.set_pipeline(blit_pipeline);
                    blit_pass.set_bind_group(0, blit_bg, &[]);
                    blit_pass.draw(0..3, 0..1);
                }
            }
        }
    }

    fn hdr_scene_overlays(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let frame = ctx.frame;
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        // Grid pass (HDR path): draw the existing analytical grid on the final
        // output after tone mapping / FXAA, reusing the scene depth buffer so
        // scene geometry still occludes the grid exactly as in the LDR path.
        if frame.viewport.show_grid {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let grid_bg = &slot.grid_bind_group;
            let mut grid_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_grid_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &slot_hdr.output_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            grid_pass.set_pipeline(&self.resources.grid_pipeline);
            grid_pass.set_bind_group(0, grid_bg, &[]);
            grid_pass.draw(0..3, 0..1);
        }

        // Ground plane pass (HDR path): drawn after grid, before editor overlays.
        // Uses the scene depth buffer for correct occlusion against geometry.
        if !matches!(
            frame.effects.ground_plane.mode,
            crate::renderer::types::GroundPlaneMode::None
        ) {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let mut gp_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_ground_plane_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &slot_hdr.output_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            gp_pass.set_pipeline(&self.resources.ground_plane_pipeline);
            gp_pass.set_bind_group(0, &self.resources.ground_plane_bind_group, &[]);
            gp_pass.draw(0..3, 0..1);
        }

        // Screen-space image overlay pass (HDR path).
        // Must run before the editor overlay and axes passes because those
        // discard hdr_depth_view (StoreOp::Discard). The DC pipeline compares
        // per-pixel image depth against the scene depth buffer; if the buffer
        // has been discarded, Metal returns zeros and all DC fragments fail.
        // Plain overlay items (depth_compare: Always) are unaffected by depth,
        // but ordering them here keeps depth-composite correct.
        if !self.screen_image_gpu_data.is_empty() {
            if let Some(overlay_pipeline) = &self.resources.screen_image_pipeline {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let dc_pipeline = self.resources.screen_image_dc_pipeline.as_ref();
                let mut img_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("screen_image_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                for gpu in &self.screen_image_gpu_data {
                    if let (Some(dc_bg), Some(dc_pipe)) = (&gpu.depth_bind_group, dc_pipeline) {
                        img_pass.set_pipeline(dc_pipe);
                        img_pass.set_bind_group(0, dc_bg, &[]);
                    } else {
                        img_pass.set_pipeline(overlay_pipeline);
                        img_pass.set_bind_group(0, &gpu.bind_group, &[]);
                    }
                    img_pass.draw(0..6, 0..1);
                }
            }
        }

        // Editor overlay pass (HDR path): draw viewport/editor overlays on the
        // final output after tone mapping / FXAA, reusing the scene depth
        // buffer so depth-tested helpers still behave correctly.
        {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let has_editor_overlays = (frame.interaction.gizmo_model.is_some()
                && slot.gizmo_index_count > 0)
                || !slot.constraint_line_buffers.is_empty()
                || !slot.clip_plane_fill_buffers.is_empty()
                || !slot.clip_plane_line_buffers.is_empty()
                || !slot.xray_object_buffers.is_empty();
            if has_editor_overlays {
                let camera_bg = &slot.camera_bind_group;
                let mut overlay_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hdr_editor_overlay_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if frame.interaction.gizmo_model.is_some() && slot.gizmo_index_count > 0 {
                    overlay_pass.set_pipeline(&self.resources.gizmo_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    overlay_pass.set_bind_group(1, &slot.gizmo_bind_group, &[]);
                    overlay_pass.set_vertex_buffer(0, slot.gizmo_vertex_buffer.slice(..));
                    overlay_pass.set_index_buffer(
                        slot.gizmo_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    overlay_pass.draw_indexed(0..slot.gizmo_index_count, 0, 0..1);
                }

                if !slot.constraint_line_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.overlay_line_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, index_count, _ubuf, bg) in &slot.constraint_line_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*index_count, 0, 0..1);
                    }
                }

                if !slot.clip_plane_fill_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.overlay_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_fill_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }

                if !slot.clip_plane_line_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.overlay_line_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_line_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }

                if !slot.xray_object_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.xray_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (mesh_id, _buf, bg) in &slot.xray_object_buffers {
                        let Some(mesh) = self.resources.mesh_store.get(*mesh_id) else {
                            continue;
                        };
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        overlay_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        overlay_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }
        }

        // Axes indicator pass (HDR path): draw in screen space on the final
        // output after tone mapping / FXAA so it stays visible in PBR mode.
        if frame.viewport.show_axes_indicator {
            let slot = &self.viewport_slots[vp_idx];
            if slot.axes_vertex_count > 0 {
                let slot_hdr = slot.hdr.as_ref().unwrap();
                let mut axes_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hdr_axes_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                slot.draw_axes_indicator(&mut axes_pass, &self.resources, true);
            }
        }
    }

    fn hdr_final_overlay(&mut self, ctx: &HdrFrameCtx, encoder: &mut wgpu::CommandEncoder) {
        let device = ctx.device;
        let queue = ctx.queue;
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        let w = ctx.w;
        let h = ctx.h;
        // Overlay shapes, rects, labels, scalar bars, rulers, and overlay images (HDR path): drawn last.
        let has_overlay = self.overlay_shape_gpu_data.is_some()
            || self.overlay_rect_gpu_data.is_some()
            || self.label_gpu_data.is_some()
            || self.scalar_bar_gpu_data.is_some()
            || self.ruler_gpu_data.is_some()
            || self.loading_bar_gpu_data.is_some()
            || !self.overlay_image_gpu_data.is_empty();

        // HDR backdrop blur: the tonemapped scene is on an intermediate so we
        // can sample it for the blur. When blur is needed we redirect the
        // tonemapped output to a managed intermediate, blur it, then draw
        // overlays there and blit to output_view at the end.
        let needs_hdr_blur = self.has_backdrop_blur_shapes();
        let hdr_blur_bg: Option<wgpu::BindGroup> = if needs_hdr_blur && has_overlay {
            self.ensure_backdrop_blur_state(device, w.max(1), h.max(1));
            // Blit output_view content to the intermediate so we have a
            // samplable copy. We already have the tonemapped result on
            // output_view. Unfortunately surface textures can't be sampled,
            // so we blit to our intermediate first.
            //
            // Actually for HDR: the tone-map wrote to output_view (or
            // upscale_view). We need the scene in a samplable texture. The
            // HDR colour texture (hdr_colour_view) is samplable but it's HDR.
            // For simplicity, use the HDR colour texture as the blur source;
            // the blur result will be HDR-ish but clamped by the LDR target
            // format of the blur textures. This looks acceptable in practice.
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            let source = &slot_hdr.hdr_view;
            let spread = self
                .overlay_shape_gpu_data
                .as_ref()
                .map(|d| d.max_blur_radius)
                .unwrap_or(8.0);
            Some(self.run_backdrop_blur(encoder, device, queue, source, spread))
        } else {
            None
        };

        if has_overlay {
            let hdr_depth_view = &self.viewport_slots[vp_idx]
                .hdr
                .as_ref()
                .unwrap()
                .output_depth_view;
            let mut overlay_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
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
            // Blur backdrop shapes drawn first (behind normal shapes).
            if let Some(ref bg) = hdr_blur_bg {
                self.draw_blur_shapes(&mut overlay_pass, bg);
            }
            emit_overlay_2d!(self, overlay_pass);
        }
    }
}
