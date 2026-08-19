//! The HDR render path: the full post-processing pipeline (scene, sprites,
//! decals, transparency, scatter, flow, bloom, tone mapping, overlays). Builds
//! its own command encoder and returns the finished buffer.

use super::*;

/// Per-frame context shared by the HDR pass-group methods. Holds the
/// preamble-computed values each pass needs. It borrows only frame-level
/// data, never `self`, so the pass methods can take `&mut self` freely.
struct HdrFrameCtx<'a> {
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
    frame: &'a FrameData,
    scene_items: &'a [SceneRenderItem],
    output_view: &'a crate::gpu::TextureView,
    vp_idx: usize,
    w: u32,
    h: u32,
    ssaa_factor: u32,
    hdr_clear_rgb: [f32; 3],
}

/// Screen-space scissor for one decal's fullscreen quad.
enum DecalScissor {
    /// The decal projects entirely off screen: skip the draw.
    Skip,
    /// A box corner is at or behind the near plane (camera inside/straddling
    /// the decal), so the screen bound is unreliable: use the full framebuffer.
    Full,
    /// Tight scissor rect (x, y, w, h) in framebuffer pixels.
    Rect(u32, u32, u32, u32),
}

/// Project the decal's unit-cube corners and return a scissor rect bounding
/// their screen extent. The decal fragment shader still runs a fullscreen quad,
/// but the scissor confines rasterization to the decal's actual footprint,
/// removing the per-decal fullscreen overdraw. `vp_w`/`vp_h` are the decal-pass
/// target dimensions.
fn decal_scissor(model: &glam::Mat4, view_proj: &glam::Mat4, vp_w: u32, vp_h: u32) -> DecalScissor {
    let mvp = *view_proj * *model;
    let mut min = glam::Vec2::splat(f32::MAX);
    let mut max = glam::Vec2::splat(f32::MIN);
    for cz in [-0.5f32, 0.5] {
        for cy in [-0.5f32, 0.5] {
            for cx in [-0.5f32, 0.5] {
                let clip = mvp * glam::Vec4::new(cx, cy, cz, 1.0);
                if clip.w <= 1e-4 {
                    return DecalScissor::Full;
                }
                let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
                min = min.min(ndc);
                max = max.max(ndc);
            }
        }
    }
    // NDC (y up) -> framebuffer pixels (y down), clamped to the target.
    let (fw, fh) = (vp_w as f32, vp_h as f32);
    let x0 = ((min.x * 0.5 + 0.5) * fw).floor().clamp(0.0, fw);
    let x1 = ((max.x * 0.5 + 0.5) * fw).ceil().clamp(0.0, fw);
    let y0 = ((0.5 - max.y * 0.5) * fh).floor().clamp(0.0, fh);
    let y1 = ((0.5 - min.y * 0.5) * fh).ceil().clamp(0.0, fh);
    let w = (x1 - x0) as u32;
    let h = (y1 - y0) as u32;
    if w == 0 || h == 0 {
        DecalScissor::Skip
    } else {
        DecalScissor::Rect(x0 as u32, y0 as u32, w, h)
    }
}

/// Encode one non-instanced mesh item's draws: bind the object (group 1),
/// material plugin (group 3), and deform (group 2) groups, pick the pipeline,
/// and draw. `obj_bg_override` is the item's per-item bind group when one was
/// prepared; `None` falls back to the mesh's shared object bind group. `hdr`
/// selects the material-plugin pipeline family; the built-in pipelines are
/// passed in by the caller. Shared by the HDR scene pass and the HDR/LDR
/// foreground passes; group 0 must already be bound by the caller.
///
/// `submesh_bgs` carries the per-range bind groups for an item drawn with
/// per-submesh materials; with it set the indexed path issues one draw per
/// range. `submesh_transparent` filters which ranges draw: `Some(false)`
/// draws only opaque-material ranges (the HDR scene pass, whose transparent
/// ranges go to OIT), `Some(true)` only blend ranges, `None` all of them
/// (foreground passes, which draw transparency inline).
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_mesh_item(
    resources: &DeviceResources,
    compute_filter_results: &[crate::resources::ComputeFilterResult],
    render_pass: &mut crate::gpu::RenderPass<'_>,
    item: &SceneRenderItem,
    obj_bg_override: Option<&crate::gpu::BindGroup>,
    // Object-data element index the whole-mesh draw selects with
    // @builtin(instance_index). Must be 0 when `obj_bg_override` is None (the
    // mesh's single-element fallback buffer).
    obj_index: u32,
    wireframe_mode: bool,
    hdr: bool,
    solid_pl: &crate::gpu::RenderPipeline,
    solid_two_sided_pl: &crate::gpu::RenderPipeline,
    trans_pl: &crate::gpu::RenderPipeline,
    wf_pl: &crate::gpu::RenderPipeline,
    submesh_bgs: Option<&[Option<crate::gpu::BindGroup>]>,
    // Object-data indices parallel to `submesh_bgs`; a range with its own bind
    // group selects its element here.
    submesh_indices: Option<&[u32]>,
    submesh_transparent: Option<bool>,
) {
    let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
        return;
    };
    let obj_bg = obj_bg_override.unwrap_or(&mesh.object_bind_group);
    render_pass.set_bind_group(1, obj_bg, &[]);
    let plug = resources.material_plugin_draw(item.material.shading_plugin);
    if let Some((_, mat_bg)) = plug {
        bind_material_group!(render_pass, mat_bg);
    }

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
    if wireframe_mode {
        if let Some(edge_buf) = &mesh.edge_index_buffer {
            render_pass.set_pipeline(wf_pl);
            bind_deform_group!(render_pass, resources, deform_bg);
            render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
            render_pass.set_index_buffer(edge_buf.slice(..), crate::gpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..mesh.edge_index_count, 0, obj_index..obj_index + 1);
        }
    } else if is_face_attr {
        if let Some(ref fvb) = mesh.face_vertex_buffer {
            let pl = if let Some((pp, _)) = plug {
                match (
                    hdr,
                    item.settings.opacity < 1.0,
                    item.material.is_two_sided(),
                ) {
                    (true, true, _) => &pp.hdr.transparent,
                    (true, false, true) => &pp.hdr.solid_two_sided,
                    (true, false, false) => &pp.hdr.solid,
                    (false, true, _) => &pp.ldr.transparent,
                    (false, false, true) => &pp.ldr.solid_two_sided,
                    (false, false, false) => &pp.ldr.solid,
                }
            } else if item.settings.opacity < 1.0 {
                trans_pl
            } else {
                solid_pl
            };
            render_pass.set_pipeline(pl);
            bind_deform_group!(render_pass, resources, deform_bg);
            render_pass.set_vertex_buffer(0, fvb.slice(..));
            render_pass.draw(0..mesh.index_count, obj_index..obj_index + 1);
        }
    } else {
        let filter = compute_filter_results
            .iter()
            .find(|r| r.mesh_id == item.mesh_id);
        let ranges = if filter.is_none() {
            // A compute-filtered index buffer is compacted, so the mesh's
            // ranges no longer address it; the filter branch below draws the
            // whole filtered mesh with the item material instead.
            crate::renderer::prepare::active_submesh_materials(item, mesh).zip(submesh_bgs)
        } else {
            None
        };
        if let Some((mats, bgs)) = ranges {
            bind_deform_group!(render_pass, resources, deform_bg);
            render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
            render_pass.set_index_buffer(
                resources.geometry.index_slice(mesh.index_span),
                crate::gpu::IndexFormat::Uint32,
            );
            for (r, (mat, range)) in mats.iter().zip(&mesh.submeshes).enumerate() {
                let is_trans = item.settings.opacity < 1.0 || mat.is_blend();
                if let Some(want) = submesh_transparent {
                    if is_trans != want {
                        continue;
                    }
                }
                let plug_r = resources.material_plugin_draw(mat.shading_plugin);
                let pl = if let Some((pp, _)) = plug_r {
                    match (hdr, is_trans, mat.is_two_sided()) {
                        (true, true, _) => &pp.hdr.transparent,
                        (true, false, true) => &pp.hdr.solid_two_sided,
                        (true, false, false) => &pp.hdr.solid,
                        (false, true, _) => &pp.ldr.transparent,
                        (false, false, true) => &pp.ldr.solid_two_sided,
                        (false, false, false) => &pp.ldr.solid,
                    }
                } else if is_trans {
                    trans_pl
                } else if mat.is_two_sided() {
                    solid_two_sided_pl
                } else {
                    solid_pl
                };
                render_pass.set_pipeline(pl);
                let (bg, inst) = match bgs.get(r).and_then(|b| b.as_ref()) {
                    Some(rbg) => (
                        rbg,
                        submesh_indices
                            .and_then(|v| v.get(r))
                            .copied()
                            .unwrap_or(obj_index),
                    ),
                    None => (obj_bg, obj_index),
                };
                render_pass.set_bind_group(1, bg, &[]);
                if let Some((_, mat_bg)) = plug_r {
                    bind_material_group!(render_pass, mat_bg);
                }
                render_pass.draw_indexed(
                    range.first_index..range.first_index + range.index_count,
                    0,
                    inst..inst + 1,
                );
            }
        } else {
            let pl = if let Some((pp, _)) = plug {
                if item.settings.opacity < 1.0 {
                    &pp.hdr.transparent
                } else if item.material.is_two_sided() {
                    &pp.hdr.solid_two_sided
                } else {
                    &pp.hdr.solid
                }
            } else if item.settings.opacity < 1.0 {
                trans_pl
            } else {
                solid_pl
            };
            render_pass.set_pipeline(pl);
            bind_deform_group!(render_pass, resources, deform_bg);
            render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
            if let Some(fr) = filter {
                render_pass
                    .set_index_buffer(fr.index_buffer.slice(..), crate::gpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..fr.index_count, 0, obj_index..obj_index + 1);
            } else {
                render_pass.set_index_buffer(
                    resources.geometry.index_slice(mesh.index_span),
                    crate::gpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..mesh.index_count, 0, obj_index..obj_index + 1);
            }
        }
    }
    if item.show_normals {
        if let Some(ref nl_buf) = mesh.normal_line_buffer {
            if mesh.normal_line_count > 0 {
                render_pass.set_pipeline(wf_pl);
                bind_deform_group!(render_pass, resources, &resources.deform.dummy_bind_group);
                render_pass.set_bind_group(1, &mesh.normal_bind_group, &[]);
                render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                render_pass.draw(0..mesh.normal_line_count, 0..1);
            }
        }
    }
}

impl ViewportRenderer {
    /// Timestamp writes for one measured pass. `begin` and `end` select
    /// which boundary of the slot's begin/end pair this pass writes, so a
    /// multi-pass effect can begin on its first pass and end on its last
    /// (each query index must be written at most once per frame).
    fn ts_writes_for(
        &self,
        slot: u32,
        begin: bool,
        end: bool,
    ) -> Option<crate::gpu::RenderPassTimestampWrites<'_>> {
        self.ts_query_set.as_ref().map(|qs| {
            self.ts_written_mask
                .fetch_or(1 << slot, std::sync::atomic::Ordering::Relaxed);
            crate::gpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: begin.then_some(slot * 2),
                end_of_pass_write_index: end.then_some(slot * 2 + 1),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_frame_hdr(
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
        ssaa_factor: u32,
    ) -> crate::gpu::CommandBuffer {
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
            foreground_enabled: if self.foreground_active(frame) { 1 } else { 0 },
            _pad: [0; 3],
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
                    max_brightness: pp.bloom_max_brightness,
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
                foreground_enabled: if self.foreground_active(frame) {
                    1.0
                } else {
                    0.0
                },
            };
            let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            queue.write_buffer(
                &hdr.dof_uniform_buf,
                0,
                bytemuck::cast_slice(&[dof_uniform]),
            );
        }

        // Pre-allocate the foreground depth target so the tone-map / DOF bind
        // groups rebuilt below can reference it as the coverage mask. The pass
        // draws into hdr_view after the SSAA resolve, so the depth target is
        // scene-sized (matching hdr_view, OIT, and the other post-resolve
        // passes), not SSAA-sized.
        let use_foreground = self.foreground_active(frame);
        if use_foreground {
            let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
            let [sw, sh] = hdr.scene_size;
            self.resources
                .ensure_viewport_foreground_depth(device, hdr, sw, sh);
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
                use_foreground,
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
                scene_items.iter().any(|i| {
                    !i.settings.hidden
                        && crate::renderer::prepare::has_transparent_draws(i, &self.resources)
                })
            } || frame
                .scene
                .volume_meshes
                .iter()
                .any(|i| !i.settings.hidden && i.transparency.is_some())
                // Item-type plugins may draw into the OIT pass through
                // `paint_transparent` (mirrors `has_transparent` below).
                || self.any_plugin_items_submitted(frame);
            if needs_oit {
                let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
                let [sw, sh] = hdr.scene_size;
                self.resources.ensure_viewport_oit(device, hdr, sw, sh);
            }
        }

        // -----------------------------------------------------------------------
        // Build the command encoder.
        // -----------------------------------------------------------------------
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
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
        self.hdr_store_hiz_depth(&ctx, &mut encoder);
        self.hdr_external_instances(&ctx, &mut encoder);
        self.hdr_sprite_passes(&ctx, &mut encoder);
        self.hdr_ssaa_refraction(&ctx, &mut encoder);
        self.hdr_decals(&ctx, &mut encoder);
        self.hdr_decal_outline(&ctx, &mut encoder);
        self.hdr_sub_highlight(&ctx, &mut encoder);
        self.hdr_depth_read_pass(&ctx, &mut encoder);
        self.hdr_oit(&ctx, &mut encoder);
        self.hdr_scatter(&ctx, &mut encoder);
        self.hdr_lic(&ctx, &mut encoder);
        self.hdr_outline_composite(&ctx, &mut encoder);
        self.hdr_foreground(&ctx, &mut encoder);
        self.hdr_post_effects(&ctx, &mut encoder);
        self.hdr_tonemap_resolve(&ctx, &mut encoder);
        self.hdr_scene_overlays(&ctx, &mut encoder);
        self.hdr_final_overlay(&ctx, &mut encoder);
        // Resolve last frame's timestamp queries -> staging buffer (HDR path).
        // Skip while a readback is unread or in flight so the single staging
        // buffer is not overwritten before prepare() reads it. The set resolved
        // here was written during the previous frame; resolving it in this
        // frame's (later) submission is what lets Metal's stage-boundary
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
                // resolve call. Resolving a slot no pass wrote is undefined in
                // Vulkan and corrupts the command stream on some drivers
                // (NVIDIA Linux); filling skipped slots with extra
                // write_timestamp calls between passes hangs Apple Metal,
                // whose stage-boundary counters cannot sample inside encoders.
                // Per-run resolves touch only written queries, so every
                // written slot is read regardless of which optional passes ran
                // (the old contiguous-prefix scheme silently dropped any slot
                // after the first skipped one, e.g. post and cull on frames
                // with no OIT).
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

        encoder.finish()
    }

    fn hdr_scene_pass(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
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

            let clear_wgpu = crate::gpu::Color {
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
                crate::gpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(crate::renderer::GPU_TS_SCENE * 2),
                    end_of_pass_write_index: Some(crate::renderer::GPU_TS_SCENE * 2 + 1),
                }
            });
            let mut render_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("hdr_scene_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: scene_colour_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(clear_wgpu),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: scene_depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(1.0),
                        store: crate::gpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(1),
                        store: crate::gpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: hdr_ts_writes,
                occlusion_query_set: None,
            });

            let resources = &self.resources;
            // This viewport's own cull outputs (indirect args and cull bind
            // groups), written by run_viewport_cull against this slot's camera.
            let cull0 = &self.viewport_slots[vp_idx].cull;
            render_pass.set_bind_group(0, camera_bg, &[]);

            // Check skybox eligibility early; drawn after all opaques below.
            let show_skybox = frame
                .effects
                .environment
                .as_ref()
                .is_some_and(|e| e.show_skybox)
                && resources.ibl.skybox_view.is_some();

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
                            && resources.cull.hdr_solid_pipeline.is_some()
                            && cull0.indirect_args_buf.is_some();

                        // Early-Z fast path: when no clip object can discard a
                        // fragment this frame, opaque batches without alpha-mask
                        // instances draw with the discard-free pipeline twin so
                        // hidden fragments are depth-rejected before shading.
                        let clipping_active = frame
                            .effects
                            .clip_objects
                            .iter()
                            .any(|o| o.enabled && o.clip_geometry);

                        if use_indirect {
                            if let (Some(pipeline), Some(pipeline_two_sided), Some(indirect_buf)) = (
                                &resources.cull.hdr_solid_pipeline,
                                &resources.cull.hdr_solid_two_sided_pipeline,
                                &cull0.indirect_args_buf,
                            ) {
                                let nodiscard_pipes = (
                                    resources.cull.hdr_solid_nodiscard_pipeline.as_ref(),
                                    resources
                                        .cull
                                        .hdr_solid_two_sided_nodiscard_pipeline
                                        .as_ref(),
                                );
                                bind_deform_group!(
                                    render_pass,
                                    resources,
                                    &resources.deform.dummy_bind_group
                                );
                                // Geometry lives in the shared slab, so the chunk
                                // buffers bind once and each batch's indirect args
                                // carry the mesh's base_vertex / first_index (written
                                // by the cull kernel). Consecutive batches that share
                                // the pipeline variant, the instance+texture bind
                                // group, and the slab chunk form a run drawn with one
                                // multi_draw_indexed_indirect where the backend
                                // supports it; runs break on a global-index gap (a
                                // transparent batch sits between two opaque ones) so a
                                // multi-draw never sweeps in an entry the CPU skipped.
                                let multi_draw = self.instancing.multi_draw_active();
                                let mut cur_pipe: Option<(bool, bool)> = None;
                                let mut cur_bg: Option<*const crate::gpu::BindGroup> = None;
                                let mut cur_chunks: Option<(u32, u32)> = None;
                                let mut run_start: u64 = 0;
                                let mut run_len: u32 = 0;
                                for (batch_global_idx, batch) in &opaque_batches {
                                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                        continue;
                                    };
                                    let mat_key = (
                                        batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    );
                                    let Some(inst_tex_bg) =
                                        cull0.instance_cull_bind_groups.get(&mat_key)
                                    else {
                                        continue;
                                    };
                                    let no_discard = !clipping_active
                                        && !batch.has_alpha_mask
                                        && nodiscard_pipes.0.is_some()
                                        && nodiscard_pipes.1.is_some();
                                    let pipe_key = (batch.two_sided, no_discard);
                                    let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                                    let bg_ptr = inst_tex_bg as *const crate::gpu::BindGroup;
                                    let g = *batch_global_idx as u64;
                                    if run_len > 0
                                        && g == run_start + run_len as u64
                                        && cur_pipe == Some(pipe_key)
                                        && cur_bg == Some(bg_ptr)
                                        && cur_chunks == Some(chunks)
                                    {
                                        run_len += 1;
                                        continue;
                                    }
                                    if run_len > 0 {
                                        let dc = crate::renderer::render::emit_indirect_run(
                                            &mut render_pass,
                                            indirect_buf,
                                            run_start,
                                            run_len,
                                            multi_draw,
                                        );
                                        self.frame_main_draw_commands
                                            .fetch_add(dc, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    if cur_pipe != Some(pipe_key) {
                                        render_pass.set_pipeline(
                                            match (no_discard, batch.two_sided) {
                                                (true, true) => nodiscard_pipes.1.unwrap(),
                                                (true, false) => nodiscard_pipes.0.unwrap(),
                                                (false, true) => pipeline_two_sided,
                                                (false, false) => pipeline,
                                            },
                                        );
                                        cur_pipe = Some(pipe_key);
                                    }
                                    if cur_bg != Some(bg_ptr) {
                                        render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                        cur_bg = Some(bg_ptr);
                                    }
                                    if cur_chunks != Some(chunks) {
                                        render_pass.set_vertex_buffer(
                                            0,
                                            resources.geometry.vertex_chunk_slice(chunks.0),
                                        );
                                        render_pass.set_index_buffer(
                                            resources.geometry.index_chunk_slice(chunks.1),
                                            crate::gpu::IndexFormat::Uint32,
                                        );
                                        self.frame_main_buffer_binds
                                            .fetch_add(2, std::sync::atomic::Ordering::Relaxed);
                                        cur_chunks = Some(chunks);
                                    }
                                    run_start = g;
                                    run_len = 1;
                                }
                                if run_len > 0 {
                                    let dc = crate::renderer::render::emit_indirect_run(
                                        &mut render_pass,
                                        indirect_buf,
                                        run_start,
                                        run_len,
                                        multi_draw,
                                    );
                                    self.frame_main_draw_commands
                                        .fetch_add(dc, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                        } else if let (Some(pipeline), Some(pipeline_two_sided)) = (
                            &resources.instancing.hdr_solid_pipeline,
                            &resources.instancing.hdr_solid_two_sided_pipeline,
                        ) {
                            let nodiscard_pipes = (
                                resources.instancing.hdr_solid_nodiscard_pipeline.as_ref(),
                                resources
                                    .instancing
                                    .hdr_solid_two_sided_nodiscard_pipeline
                                    .as_ref(),
                            );
                            bind_deform_group!(
                                render_pass,
                                resources,
                                &resources.deform.dummy_bind_group
                            );
                            let mut cur_pipe: Option<(bool, bool)> = None;
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            for (_, batch) in &opaque_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                let mat_key = (
                                    batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) =
                                    resources.instancing.bind_groups.get(&mat_key)
                                else {
                                    continue;
                                };
                                let no_discard = !clipping_active
                                    && !batch.has_alpha_mask
                                    && nodiscard_pipes.0.is_some()
                                    && nodiscard_pipes.1.is_some();
                                if cur_pipe != Some((batch.two_sided, no_discard)) {
                                    render_pass.set_pipeline(match (no_discard, batch.two_sided) {
                                        (true, true) => nodiscard_pipes.1.unwrap(),
                                        (true, false) => nodiscard_pipes.0.unwrap(),
                                        (false, true) => pipeline_two_sided,
                                        (false, false) => pipeline,
                                    });
                                    cur_pipe = Some((batch.two_sided, no_discard));
                                }
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                                if cur_chunks != Some(chunks) {
                                    render_pass.set_vertex_buffer(
                                        0,
                                        resources.geometry.vertex_chunk_slice(chunks.0),
                                    );
                                    render_pass.set_index_buffer(
                                        resources.geometry.index_chunk_slice(chunks.1),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    self.frame_main_buffer_binds
                                        .fetch_add(2, std::sync::atomic::Ordering::Relaxed);
                                    cur_chunks = Some(chunks);
                                }
                                let base_vertex = resources.geometry.base_vertex(mesh.vertex_span);
                                let first_index = resources.geometry.first_index(mesh.index_span);
                                render_pass.draw_indexed(
                                    first_index..first_index + mesh.index_count,
                                    base_vertex,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                                self.frame_main_draw_commands
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }

                    // NOTE: transparent_batches are now rendered in the OIT pass below,
                    // not in the HDR scene pass. This block intentionally left empty.
                    let _ = &transparent_batches; // suppress unused warning

                    if frame.viewport.wireframe_mode {
                        if let Some(ref hdr_wf) = resources.scene.hdr_wireframe {
                            let mut wf_idx = 0usize;
                            for item in scene_items {
                                if item.settings.hidden {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                    continue;
                                };
                                render_pass.set_pipeline(hdr_wf);
                                bind_deform_group!(
                                    render_pass,
                                    resources,
                                    resources.deform.instance_bind_group_for(
                                        item.mesh_id,
                                        item.deform_instance,
                                    )
                                );
                                let bg = self
                                    .mesh_uniforms
                                    .wireframe_bind_groups
                                    .get(wf_idx)
                                    .unwrap_or(&mesh.object_bind_group);
                                render_pass.set_bind_group(1, bg, &[]);
                                render_pass.set_vertex_buffer(
                                    0,
                                    resources.geometry.vertex_slice(mesh.vertex_span),
                                );
                                if let Some(edge_buf) = &mesh.edge_index_buffer {
                                    render_pass.set_index_buffer(
                                        edge_buf.slice(..),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                                }
                                wf_idx += 1;
                            }
                        }
                    } else if let (Some(hdr_solid), Some(hdr_solid_two_sided)) = (
                        &resources.scene.hdr_solid,
                        &resources.scene.hdr_solid_two_sided,
                    ) {
                        // Clip geometry disables the discard-free early-Z twin
                        // (the clip discards would be stripped). Computed here
                        // because this per-object branch is the `else` of the
                        // instanced path where the same check lives.
                        let clipping_active = frame
                            .effects
                            .clip_objects
                            .iter()
                            .any(|o| o.enabled && o.clip_geometry);
                        // Only opaque excluded items are drawn in the scene pass; transparent
                        // excluded items go to the OIT pass below. LDR draws all excluded
                        // items inline (including transparent ones) using the transparent
                        // pipeline -- an intentional divergence since HDR uses OIT for
                        // transparency throughout.
                        for (item_idx, item) in
                            excluded_items.iter().copied().filter(|(_, item)| {
                                crate::renderer::prepare::has_opaque_draws(item, resources)
                            })
                        {
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };
                            let plug = resources.material_plugin_draw(item.material.shading_plugin);
                            // Early-Z fast path: a plain-opaque, single-material,
                            // non-scalar item in a frame with no clip geometry can
                            // never hit a `discard`, so draw it with the
                            // discard-free twin and let hidden fragments be
                            // depth-rejected before shading. Plugin, alpha-mask,
                            // submesh-material, and scalar-attribute (NaN-discard)
                            // draws keep the discarding pipeline.
                            let no_discard = plug.is_none()
                                && !clipping_active
                                && !resources.force_po_discard
                                && matches!(
                                    item.material.alpha_mode,
                                    crate::scene::material::AlphaMode::Opaque
                                )
                                && item.active_attribute.is_none()
                                && item.submesh_materials.is_none()
                                && resources.scene.hdr_solid_nodiscard.is_some()
                                && resources.scene.hdr_solid_two_sided_nodiscard.is_some();
                            let pipeline = if let Some((pp, _)) = plug {
                                if item.material.is_two_sided() {
                                    &pp.hdr.solid_two_sided
                                } else {
                                    &pp.hdr.solid
                                }
                            } else if no_discard {
                                if item.material.is_two_sided() {
                                    resources
                                        .scene
                                        .hdr_solid_two_sided_nodiscard
                                        .as_ref()
                                        .unwrap()
                                } else {
                                    resources.scene.hdr_solid_nodiscard.as_ref().unwrap()
                                }
                            } else if item.material.is_two_sided() {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            render_pass.set_pipeline(pipeline);
                            bind_deform_group!(
                                render_pass,
                                resources,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance,)
                            );
                            // A per-item slot draws with the shared material bind
                            // group at object_indices[item_idx]; a None slot uses
                            // the mesh's single-element buffer at instance 0.
                            let (obj_bg, obj_inst) = match self
                                .mesh_uniforms
                                .bind_groups
                                .get(item_idx)
                                .and_then(|opt| opt.as_ref())
                            {
                                Some(bg) => (bg, self.mesh_uniforms.object_indices[item_idx]),
                                None => (&mesh.object_bind_group, 0),
                            };
                            render_pass.set_bind_group(1, obj_bg, &[]);
                            if let Some((_, mat_bg)) = plug {
                                bind_material_group!(render_pass, mat_bg);
                            }
                            render_pass.set_vertex_buffer(
                                0,
                                resources.geometry.vertex_slice(mesh.vertex_span),
                            );
                            let filter = compute_filter_results
                                .iter()
                                .find(|r| r.mesh_id == item.mesh_id);
                            let ranges = if filter.is_none() {
                                crate::renderer::prepare::active_submesh_materials(item, mesh)
                                    .zip(self.mesh_uniforms.submesh_bind_groups.get(&item_idx))
                            } else {
                                None
                            };
                            if let Some(fr) = filter {
                                render_pass.set_index_buffer(
                                    fr.index_buffer.slice(..),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(
                                    0..fr.index_count,
                                    0,
                                    obj_inst..obj_inst + 1,
                                );
                            } else if let Some((mats, bgs)) = ranges {
                                // One draw per opaque-material range; blend
                                // ranges go to the OIT pass with the other
                                // transparent excluded items.
                                render_pass.set_index_buffer(
                                    resources.geometry.index_slice(mesh.index_span),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                for (r, (mat, range)) in
                                    mats.iter().zip(&mesh.submeshes).enumerate()
                                {
                                    if mat.is_blend() {
                                        continue;
                                    }
                                    let plug_r = resources.material_plugin_draw(mat.shading_plugin);
                                    let pl = if let Some((pp, _)) = plug_r {
                                        if mat.is_two_sided() {
                                            &pp.hdr.solid_two_sided
                                        } else {
                                            &pp.hdr.solid
                                        }
                                    } else if mat.is_two_sided() {
                                        hdr_solid_two_sided
                                    } else {
                                        hdr_solid
                                    };
                                    render_pass.set_pipeline(pl);
                                    let (bg, inst) = match bgs.get(r).and_then(|b| b.as_ref()) {
                                        Some(rbg) => (
                                            rbg,
                                            self.mesh_uniforms
                                                .submesh_indices
                                                .get(&item_idx)
                                                .and_then(|v| v.get(r))
                                                .copied()
                                                .unwrap_or(0),
                                        ),
                                        None => (obj_bg, obj_inst),
                                    };
                                    render_pass.set_bind_group(1, bg, &[]);
                                    if let Some((_, mat_bg)) = plug_r {
                                        bind_material_group!(render_pass, mat_bg);
                                    }
                                    render_pass.draw_indexed(
                                        range.first_index..range.first_index + range.index_count,
                                        0,
                                        inst..inst + 1,
                                    );
                                }
                            } else {
                                render_pass.set_index_buffer(
                                    resources.geometry.index_slice(mesh.index_span),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    obj_inst..obj_inst + 1,
                                );
                            }
                        }
                    }

                    // Normal-line overlays for instanced items with show_normals set.
                    // Instanced batch draws skip per-item logic, so these are drawn
                    // here after all batches finish.
                    if let Some(hdr_wf) = &resources.scene.hdr_wireframe {
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
                                    bind_deform_group!(
                                        render_pass,
                                        resources,
                                        &resources.deform.dummy_bind_group
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

                    // When prepare cached an HDR render bundle for this item
                    // set and this pass's camera bind group, replay it instead
                    // of encoding one draw per opaque item. Bundled draws run
                    // in submission order rather than the front-to-back sort
                    // below: encode savings traded against early-z, the same
                    // trade the LDR bundle makes. Transparent items are routed
                    // to the OIT pass either way.
                    let bundle_hit = self
                        .per_object_bundle
                        .as_ref()
                        .filter(|pb| pb.hdr && pb.camera_bg == *camera_bg);

                    let mut opaque: Vec<(usize, &SceneRenderItem)> = Vec::new();
                    let mut transparent: Vec<(usize, &SceneRenderItem)> = Vec::new();
                    for (idx, item) in scene_items.iter().enumerate() {
                        if item.settings.hidden || resources.mesh_store.get(item.mesh_id).is_none()
                        {
                            continue;
                        }
                        // A per-submesh-material item can hold both opaque and
                        // blend ranges, so it may appear in both lists: its
                        // opaque ranges draw here, its blend ranges in OIT.
                        if crate::renderer::prepare::has_transparent_draws(item, resources) {
                            transparent.push((idx, item));
                        }
                        if bundle_hit.is_none()
                            && crate::renderer::prepare::has_opaque_draws(item, resources)
                        {
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

                    if let Some(pb) = bundle_hit {
                        render_pass.execute_bundles(std::iter::once(&pb.bundle));
                        // Bundle execution resets all render-pass state;
                        // restore the camera bind group for the draws below.
                        render_pass.set_bind_group(0, camera_bg, &[]);
                    }

                    let per_item_bgs = &self.mesh_uniforms.bind_groups;

                    // NOTE: only opaque items are drawn here. Transparent items are
                    // routed to the OIT pass below.
                    let _ = &transparent; // suppress unused warning
                    if let (
                        Some(hdr_solid),
                        Some(hdr_solid_two_sided),
                        Some(hdr_trans),
                        Some(hdr_wf),
                    ) = (
                        &resources.scene.hdr_solid,
                        &resources.scene.hdr_solid_two_sided,
                        &resources.scene.hdr_transparent,
                        &resources.scene.hdr_wireframe,
                    ) {
                        for (item_idx, item) in &opaque {
                            let solid_pl = if item.material.is_two_sided() {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            let obj_bg = per_item_bgs.get(*item_idx).and_then(|opt| opt.as_ref());
                            draw_mesh_item(
                                resources,
                                compute_filter_results,
                                &mut render_pass,
                                item,
                                obj_bg,
                                obj_bg.map_or(0, |_| self.mesh_uniforms.object_indices[*item_idx]),
                                frame.viewport.wireframe_mode,
                                true,
                                solid_pl,
                                hdr_solid_two_sided,
                                hdr_trans,
                                hdr_wf,
                                self.mesh_uniforms
                                    .submesh_bind_groups
                                    .get(item_idx)
                                    .map(|v| v.as_slice()),
                                self.mesh_uniforms
                                    .submesh_indices
                                    .get(item_idx)
                                    .map(|v| v.as_slice()),
                                Some(false),
                            );
                        }
                    }
                }
            }

            // Cap fill pass (HDR path : section view cross-section fill).
            if !slot.cap_buffers.is_empty() {
                if let Some(ref hdr_overlay) = resources.scene.hdr_overlay {
                    render_pass.set_pipeline(hdr_overlay);
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.cap_buffers {
                        render_pass.set_bind_group(1, bg, &[]);
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass
                            .set_index_buffer(ibuf.slice(..), crate::gpu::IndexFormat::Uint32);
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
                if let Some(ref dual) = self.resources.implicit.pipeline {
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
                    let vol = &self.resources.mc.volumes[mc.volume_idx];
                    if mc.wireframe || frame.viewport.wireframe_mode {
                        if let Some(ref dual) = self.resources.mc.wireframe_pipeline {
                            render_pass.set_pipeline(dual.for_format(true));
                            for (slab, wire_bg) in vol.slabs.iter().zip(mc.wire_slab_bgs.iter()) {
                                render_pass.set_bind_group(1, wire_bg, &[]);
                                render_pass.draw_indirect(&slab.wire_indirect_buf, 0);
                            }
                        }
                    } else if let Some(ref dual) = self.resources.mc.surface_pipeline {
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
                if let Some(ref dual) = self.resources.gaussian_splat.pipeline {
                    render_pass.set_pipeline(dual.for_format(true));
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
            // TransparentVolumeMesh boundary wireframe overlay (HDR path).
            if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
                if let (Some(tvm_bg), Some(hdr_wf)) = (
                    &self.mesh_uniforms.tvm_wireframe_bg,
                    &resources.scene.hdr_wireframe,
                ) {
                    for mesh_id in &self.mesh_uniforms.tvm_wireframe_draws {
                        if let Some(mesh) = resources.mesh_store.get(*mesh_id) {
                            render_pass.set_pipeline(hdr_wf);
                            bind_deform_group!(
                                render_pass,
                                resources,
                                &resources.deform.dummy_bind_group
                            );
                            render_pass.set_bind_group(1, tvm_bg, &[]);
                            render_pass.set_vertex_buffer(
                                0,
                                resources.geometry.vertex_slice(mesh.vertex_span),
                            );
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

            // Draw skybox after built-in opaques : only uncovered sky pixels
            // pass depth == 1.0. Drawn before plugin paint so blended plugin
            // content (additive/alpha particles that do not write depth)
            // composites over the sky instead of being painted over by it.
            if show_skybox {
                render_pass.set_bind_group(0, camera_bg, &[]);
                render_pass.set_pipeline(&resources.ibl.skybox_pipeline);
                render_pass.draw(0..3, 0..1);
            }

            // Item-type plugin paint: after built-in opaques and the skybox.
            // Standard group-0 bindings are already bound.
            self.dispatch_plugin_paint(&mut render_pass, frame);
        }
    }

    /// Copy this viewport's scene depth into its HiZ prev-depth target for next
    /// frame's occlusion cull, which reprojects it into the new camera before
    /// building the pyramid. Runs right after the scene pass so it captures the
    /// opaque depth, and only when occlusion culling is enabled (the copy is
    /// otherwise unused).
    fn hdr_store_hiz_depth(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        if !self.resources.occlusion_culling_enabled() {
            return;
        }
        // A derivative render (capture / bake) must not overwrite the presented
        // frame's prev-depth: that would feed next frame's occlusion
        // reprojection the probe camera's depth instead of the shown view's.
        if !self.render_advances_state() {
            return;
        }
        let view_proj = ctx
            .frame
            .camera
            .render_camera
            .view_proj()
            .to_cols_array_2d();
        // Borrow the slot mutably and split its fields: the depth view is read
        // from `hdr` while the pyramid is written into `cull`.
        let slot = &mut self.viewport_slots[ctx.vp_idx];
        let Some(slot_hdr) = slot.hdr.as_ref() else {
            return;
        };
        let use_ssaa = ctx.ssaa_factor > 1
            && slot_hdr.ssaa_depth_texture.is_some()
            && slot_hdr.ssaa_depth_only_view.is_some();
        let (depth_view, depth_tex) = if use_ssaa {
            (
                slot_hdr.ssaa_depth_only_view.as_ref().unwrap(),
                slot_hdr.ssaa_depth_texture.as_ref().unwrap(),
            )
        } else {
            (&slot_hdr.hdr_depth_only_view, &slot_hdr.hdr_depth_texture)
        };
        let w = depth_tex.width();
        let h = depth_tex.height();
        slot.cull
            .store_hiz_prev_depth(ctx.device, encoder, depth_view, w, h, view_proj);
    }

    /// Draw this frame's external instance sets: opaque depth-tested meshes
    /// instanced off consumer-owned positions buffers. Runs right after the
    /// opaque scene pass so the instances occlude and are occluded like
    /// ordinary opaque geometry; transparents composite over them later.
    fn hdr_external_instances(
        &mut self,
        ctx: &HdrFrameCtx,
        encoder: &mut crate::gpu::CommandEncoder,
    ) {
        let vp_idx = ctx.vp_idx;
        let ssaa_factor = ctx.ssaa_factor;
        if self.external_instances_gpu_data.is_empty() {
            return;
        }
        let resources = &self.resources;
        let Some(pipeline) = self.resources.external_instances.pipeline.as_ref() else {
            return;
        };
        let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
        let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;

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

        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(feature = "wgpu29")]
            multiview_mask: None,
            label: Some("external_instances_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: colour_view,
                resolve_target: None,
                ops: crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline.for_format(true));
        pass.set_bind_group(0, camera_bg, &[]);
        for gd in &self.external_instances_gpu_data {
            let Some(mesh) = self.resources.mesh_store.get(gd.mesh_id) else {
                continue;
            };
            pass.set_bind_group(1, &gd.bind_group, &[]);
            pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
            pass.set_index_buffer(
                resources.geometry.index_slice(mesh.index_span),
                crate::gpu::IndexFormat::Uint32,
            );
            // The instance range is the buffer window: `instance_index` in
            // the shader starts at `first_instance` for direct draws.
            pass.draw_indexed(
                0..mesh.index_count,
                0,
                gd.first_instance..gd.first_instance + gd.instance_count,
            );
        }
    }

    fn hdr_sprite_passes(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
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
                    resources.sprite.pipeline_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Additive,
                    false,
                    resources.sprite.pipeline_additive_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Premultiplied,
                    false,
                    resources.sprite.pipeline_premultiplied_depth_write.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    false,
                    resources.sprite.pipeline.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Additive,
                    false,
                    resources.sprite.pipeline_additive.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Premultiplied,
                    false,
                    resources.sprite.pipeline_premultiplied.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    true,
                    resources.sprite.lit_pipeline_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Additive,
                    true,
                    resources.sprite.lit_pipeline_additive_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Premultiplied,
                    true,
                    resources
                        .sprite
                        .lit_pipeline_premultiplied_depth_write
                        .as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    true,
                    resources.sprite.lit_pipeline.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Additive,
                    true,
                    resources.sprite.lit_pipeline_additive.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Premultiplied,
                    true,
                    resources.sprite.lit_pipeline_premultiplied.as_ref(),
                ),
            ];
            let lit_fallback_bg = resources.sprite.lit_fallback_bg.as_ref();

            let fallback_soft_bg = resources.sprite.soft_fallback_bg.as_ref();

            // Pass 1: depth-write sprites, depth attachment writable, fallback
            // bound at group 2 (the live depth view is aliased to the
            // attachment in this pass and cannot also be sampled).
            if any_depth_write {
                if let Some(fallback_soft_bg) = fallback_soft_bg {
                    let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("sprite_depth_write_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: colour_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Load,
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view: depth_view,
                                depth_ops: Some(crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Load,
                                    store: crate::gpu::StoreOp::Store,
                                }),
                                stencil_ops: Some(crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Load,
                                    store: crate::gpu::StoreOp::Store,
                                }),
                            },
                        ),
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
                    resources.sprite.soft_bgl.as_ref(),
                    resources.sprite.soft_sampler.as_ref(),
                ) {
                    Some(device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("sprite_soft_bg"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::gpu::BindingResource::TextureView(depth_only_view),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: crate::gpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    }))
                } else {
                    None
                };

                if let Some(real_soft_bg) = real_soft_bg.as_ref() {
                    let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("sprite_transparent_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: colour_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Load,
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view: depth_view,
                                depth_ops: None,
                                stencil_ops: None,
                            },
                        ),
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

            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("gpu_particle_sprite_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: colour_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, camera_bg, &[]);
            let particle_lit_fallback = resources.particle.sprite_lit_fallback_bg.as_ref();
            for pd in &self.particle_gpu_data {
                if pd.hidden {
                    continue;
                }
                let Some(system) = resources
                    .particle
                    .systems
                    .get(pd.system_idx)
                    .and_then(|s| s.as_ref())
                    .filter(|s| s.alive)
                else {
                    continue;
                };
                match pd.route {
                    crate::resources::gpu::gpu_particles::ParticleDrawRoute::Sprite { lit } => {
                        let dual = match (pd.blend, lit) {
                            (crate::renderer::SpriteBlend::Additive, false) => {
                                resources.particle.sprite_pipeline_additive.as_ref()
                            }
                            (crate::renderer::SpriteBlend::Premultiplied, false) => {
                                resources.particle.sprite_pipeline_premultiplied.as_ref()
                            }
                            (crate::renderer::SpriteBlend::AlphaBlend, false) => {
                                resources.particle.sprite_pipeline_alpha.as_ref()
                            }
                            (crate::renderer::SpriteBlend::Additive, true) => {
                                resources.particle.sprite_lit_pipeline_additive.as_ref()
                            }
                            (crate::renderer::SpriteBlend::Premultiplied, true) => resources
                                .particle
                                .sprite_lit_pipeline_premultiplied
                                .as_ref(),
                            (crate::renderer::SpriteBlend::AlphaBlend, true) => {
                                resources.particle.sprite_lit_pipeline_alpha.as_ref()
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
                    crate::resources::gpu::gpu_particles::ParticleDrawRoute::Mesh { mesh_id } => {
                        let dual = match pd.blend {
                            crate::renderer::SpriteBlend::Additive => {
                                resources.particle.mesh_pipeline_additive.as_ref()
                            }
                            crate::renderer::SpriteBlend::Premultiplied => {
                                resources.particle.mesh_pipeline_premultiplied.as_ref()
                            }
                            crate::renderer::SpriteBlend::AlphaBlend => {
                                resources.particle.mesh_pipeline_alpha.as_ref()
                            }
                        };
                        let Some(dual) = dual else { continue };
                        let Some(draw_bg) = system.draw_bg_mesh.as_ref() else {
                            continue;
                        };
                        let Some(mesh) = resources.mesh_store.get(mesh_id) else {
                            continue;
                        };
                        pass.set_pipeline(dual.for_format(true));
                        pass.set_bind_group(1, draw_bg, &[]);
                        pass.set_vertex_buffer(
                            0,
                            resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        pass.set_index_buffer(
                            resources.geometry.index_slice(mesh.index_span),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..mesh.index_count, 0, 0..system.capacity);
                    }
                }
            }
        }
    }

    fn hdr_ssaa_refraction(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
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
                let tex = device.create_texture(&crate::gpu::TextureDescriptor {
                    label: Some("sprite_refraction_resolve"),
                    size: crate::gpu::Extent3d {
                        width: physical_w.max(1),
                        height: physical_h.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: crate::gpu::TextureDimension::D2,
                    format: crate::gpu::TextureFormat::Rgba16Float,
                    usage: crate::gpu::TextureUsages::COPY_DST
                        | crate::gpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
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
                crate::gpu::TexelCopyTextureInfo {
                    texture: &slot_hdr.hdr_texture,
                    mip_level: 0,
                    origin: crate::gpu::Origin3d::ZERO,
                    aspect: crate::gpu::TextureAspect::All,
                },
                crate::gpu::TexelCopyTextureInfo {
                    texture: resolve_tex,
                    mip_level: 0,
                    origin: crate::gpu::Origin3d::ZERO,
                    aspect: crate::gpu::TextureAspect::All,
                },
                crate::gpu::Extent3d {
                    width: physical_w.max(1),
                    height: physical_h.max(1),
                    depth_or_array_layers: 1,
                },
            );

            // Build the group-2 bind group: sampled resolve view + sampler.
            let refraction_bgl = resources.sprite.refraction_bgl.as_ref().unwrap();
            let refraction_sampler = resources.sprite.refraction_sampler.as_ref().unwrap();
            let refraction_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("sprite_refraction_bg"),
                layout: refraction_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(resolve_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(refraction_sampler),
                    },
                ],
            });

            if let Some(pipeline) = resources.sprite.refraction_pipeline.as_ref() {
                let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("sprite_refraction_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
                        resolve_target: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.hdr_depth_view,
                        depth_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Store,
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
                &self.resources.post.ssaa_resolve_pipeline,
                &slot_hdr.ssaa_resolve_bind_group,
            ) {
                let mut resolve_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("ssaa_resolve_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: &slot_hdr.hdr_view,
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
                resolve_pass.set_pipeline(pipeline);
                resolve_pass.set_bind_group(0, bg, &[]);
                resolve_pass.draw(0..3, 0..1);
            }
        }
    }

    fn hdr_decals(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let resources = &self.resources;
        let vp_idx = ctx.vp_idx;
        // -----------------------------------------------------------------------
        // Decal exclude pass (D5): stamp stencil = 0 on non-receiver surfaces.
        // Runs after the opaque pass, before the decal pass.
        // -----------------------------------------------------------------------
        if !self.decal_exclude_items.is_empty() {
            if let Some(exclude_pl) = self.resources.decal.exclude_pipeline.as_ref() {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
                let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("decal_exclude_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.hdr_depth_view,
                        depth_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Store,
                        }),
                        stencil_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Store,
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
                        pass.set_vertex_buffer(
                            0,
                            resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        if mesh.index_count > 0 {
                            pass.set_index_buffer(
                                resources.geometry.index_slice(mesh.index_span),
                                crate::gpu::IndexFormat::Uint32,
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
            let replace_pipeline = self.resources.decal.replace_pipeline.as_ref();
            let multiply_pipeline = self.resources.decal.multiply_pipeline.as_ref();
            let additive_pipeline = self.resources.decal.additive_pipeline.as_ref();
            // Scissor rects are in the decal render-target's pixel space. Read
            // the resolved HDR target size directly (ctx.w/ctx.h are the
            // supersampled/physical dimensions, which differ under SSAA or DPI
            // scaling).
            let target_w = slot_hdr.hdr_texture.width();
            let target_h = slot_hdr.hdr_texture.height();
            if replace_pipeline.is_some()
                || multiply_pipeline.is_some()
                || additive_pipeline.is_some()
            {
                let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("decal_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
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
                pass.set_bind_group(0, camera_bg, &[]);
                pass.set_bind_group(1, depth_bg, &[]);
                let view_proj = ctx.frame.camera.render_camera.view_proj();
                for gpu in &self.decal_gpu_data {
                    let pipeline = match gpu.blend_mode {
                        crate::renderer::DecalBlendMode::Replace => replace_pipeline,
                        crate::renderer::DecalBlendMode::Multiply => multiply_pipeline,
                        crate::renderer::DecalBlendMode::Additive => additive_pipeline,
                    };
                    if let Some(pl) = pipeline {
                        // Confine each decal's fullscreen quad to its screen
                        // footprint to avoid 173x fullscreen overdraw.
                        match decal_scissor(&gpu.model, &view_proj, target_w, target_h) {
                            DecalScissor::Skip => continue,
                            DecalScissor::Full => pass.set_scissor_rect(0, 0, target_w, target_h),
                            DecalScissor::Rect(x, y, w, h) => pass.set_scissor_rect(x, y, w, h),
                        }
                        pass.set_pipeline(&pl.hdr);
                        pass.set_bind_group(2, &gpu.bind_group, &[]);
                        pass.draw(0..6, 0..1);
                    }
                }
            }
        }
    }

    /// Trace an anti-aliased ring around the footprint of selected decals.
    ///
    /// Runs after the decal colour pass so the scene depth the decal projects
    /// against already exists. Selected decals are stamped into a transient R8
    /// mask (reusing the colour pass's coverage math and bind groups), then a
    /// fullscreen edge-detect blends the outline ring onto the HDR target. Does
    /// nothing when no decal is selected, so the common case pays no cost.
    fn hdr_decal_outline(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        if !self.decal_gpu_data.iter().any(|g| g.selected) {
            return;
        }

        let vp_idx = ctx.vp_idx;
        let device = ctx.device;

        let (target_w, target_h) = {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            (
                slot_hdr.hdr_texture.width().max(1),
                slot_hdr.hdr_texture.height().max(1),
            )
        };

        self.resources.ensure_decal_outline_pipelines(device);
        self.resources
            .ensure_decal_outline_targets(device, target_w, target_h);

        let (Some(mask_pl), Some(edge_pl), Some(targets)) = (
            self.resources.decal.outline_mask_pipeline.as_ref(),
            self.resources.decal.outline_edge_pipeline.as_ref(),
            self.resources.decal.outline_targets.as_ref(),
        ) else {
            return;
        };

        // Refresh the edge uniform in place; the target set (mask texture, view,
        // buffer, bind group) is reused frame to frame and rebuilt only when the
        // viewport size changes, so the pass allocates nothing per frame.
        let edge_uniform = crate::resources::OutlineEdgeUniform {
            colour: ctx.frame.interaction.outline_colour,
            radius: ctx.frame.interaction.outline_width_px,
            viewport_w: target_w as f32,
            viewport_h: target_h as f32,
            _pad: 0.0,
        };
        ctx.queue.write_buffer(
            &targets.edge_uniform_buf,
            0,
            bytemuck::cast_slice(&[edge_uniform]),
        );

        let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
        let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
        let depth_bg = &slot_hdr.decal_depth_bg;

        // Accumulate the selected decals' screen AABB while stamping the mask, so
        // the edge pass runs only over that region instead of the whole frame.
        let mut union: Option<(i32, i32, i32, i32)> = None;
        let mut any_full = false;

        // Mask pass: stamp each selected decal's footprint.
        {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("decal_outline_mask_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: &targets.mask_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(mask_pl);
            pass.set_bind_group(0, camera_bg, &[]);
            pass.set_bind_group(1, depth_bg, &[]);
            let view_proj = ctx.frame.camera.render_camera.view_proj();
            for gpu in &self.decal_gpu_data {
                if !gpu.selected {
                    continue;
                }
                match decal_scissor(&gpu.model, &view_proj, target_w, target_h) {
                    DecalScissor::Skip => continue,
                    DecalScissor::Full => {
                        any_full = true;
                        pass.set_scissor_rect(0, 0, target_w, target_h);
                    }
                    DecalScissor::Rect(x, y, w, h) => {
                        let (x0, y0, x1, y1) = (x as i32, y as i32, (x + w) as i32, (y + h) as i32);
                        union = Some(match union {
                            Some((ux0, uy0, ux1, uy1)) => {
                                (ux0.min(x0), uy0.min(y0), ux1.max(x1), uy1.max(y1))
                            }
                            None => (x0, y0, x1, y1),
                        });
                        pass.set_scissor_rect(x, y, w, h);
                    }
                }
                pass.set_bind_group(2, &gpu.bind_group, &[]);
                pass.draw(0..6, 0..1);
            }
        }

        // Every selected decal projected off screen: nothing to outline.
        if !any_full && union.is_none() {
            return;
        }
        // Bound the edge pass to the decals' screen AABB, expanded by the outline
        // width. A decal straddling the near plane (Full) falls back to fullscreen.
        let edge_rect = if any_full {
            None
        } else {
            union.map(|(x0, y0, x1, y1)| {
                let m = ctx.frame.interaction.outline_width_px.ceil() as i32 + 2;
                let cx0 = (x0 - m).clamp(0, target_w as i32);
                let cy0 = (y0 - m).clamp(0, target_h as i32);
                let cx1 = (x1 + m).clamp(0, target_w as i32);
                let cy1 = (y1 + m).clamp(0, target_h as i32);
                (
                    cx0 as u32,
                    cy0 as u32,
                    (cx1 - cx0).max(0) as u32,
                    (cy1 - cy0).max(0) as u32,
                )
            })
        };

        // Edge pass: ring edge-detect blended onto the HDR colour target.
        {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("decal_outline_edge_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: &slot_hdr.hdr_view,
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
            pass.set_pipeline(edge_pl);
            pass.set_bind_group(0, &targets.edge_bind_group, &[]);
            if let Some((x, y, w, h)) = edge_rect {
                if w == 0 || h == 0 {
                    return;
                }
                pass.set_scissor_rect(x, y, w, h);
            }
            pass.draw(0..3, 0..1);
        }
    }

    fn hdr_sub_highlight(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let vp_idx = ctx.vp_idx;
        // -----------------------------------------------------------------------
        // Sub-object highlight pass: face fill, edge lines, vertex sprites.
        // Runs after opaque geometry (depth buffer is ready) and before OIT so
        // highlights are not occluded by opaque surfaces.
        // -----------------------------------------------------------------------
        if let Some(sub_hl) = self.viewport_slots[vp_idx].sub_highlight.as_ref() {
            let resources = &self.resources;
            if let (Some(fill_pl), Some(edge_pl), Some(sprite_pl)) = (
                &resources.sub_highlight.fill_pipeline,
                &resources.sub_highlight.edge_pipeline,
                &resources.sub_highlight.sprite_pipeline,
            ) {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
                let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("sub_highlight_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
                        resolve_target: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.hdr_depth_view,
                        depth_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            // Store even though depth_write_enabled=false on all
                            // sub-highlight pipelines: the values are unchanged, but
                            // StoreOp::Discard would invalidate the tile on Metal and
                            // cause subsequent passes (tone_map, grid, screen images)
                            // to read 0.0, making the background go black.
                            store: crate::gpu::StoreOp::Store,
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

    /// Read-only-depth plugin pass. Runs after the opaque scene (and the
    /// built-in sprite passes) and before OIT, with the scene depth attachment
    /// bound read-only so opted-in `ItemTypePlugin`s can sample it while they
    /// draw (soft particles, contact effects, depth-aware fog). Blends over the
    /// HDR scene colour; tests against opaque depth but writes none.
    ///
    /// Fully skipped when no plugin returns `draws_depth_read()`: no render
    /// pass begins and the depth attachment sees no transition.
    fn hdr_depth_read_pass(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let frame = ctx.frame;
        if !self.any_plugin_draws_depth_read(frame) {
            return;
        }
        let device = ctx.device;
        let vp_idx = ctx.vp_idx;
        let ssaa_factor = ctx.ssaa_factor;
        let resources = &self.resources;
        let slot = &self.viewport_slots[vp_idx];
        let camera_bg = &slot.camera_bind_group;
        let slot_hdr = slot.hdr.as_ref().unwrap();

        // Match the sprite soft path: target the ssaa_* colour/depth views when
        // SSAA is active so plugin draws land at supersampled resolution and
        // resolve with the rest of the scene; otherwise the hdr_* views.
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

        // Prebuilt group handed to plugins that have a spare bind group. Plugins
        // at the four-group limit ignore it and bake the same depth-only view +
        // sampler into a group of their own.
        let depth_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("plugin_depth_read_bg"),
            layout: &resources.material.depth_read_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(
                        &resources.material.depth_read_sampler,
                    ),
                },
            ],
        });

        // Depth attachment read-only (`depth_ops: None`) so `depth_only_view`,
        // a depth-aspect view of the same buffer, can be sampled in the pass.
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(feature = "wgpu29")]
            multiview_mask: None,
            label: Some("hdr_depth_read_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: colour_view,
                resolve_target: None,
                ops: crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_bind_group(0, camera_bg, &[]);

        self.dispatch_plugin_paint_depth_read(
            &mut pass,
            frame,
            depth_only_view,
            &resources.material.depth_read_sampler,
            &depth_bg,
        );
    }

    fn hdr_oit(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let resources = &self.resources;
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
        let has_transparent = if self.instancing.use_instancing && !self.instancing.batches.is_empty() {
                // Transparent instanced batches go through OIT. Transparent excluded items
                // (two-sided, active-attribute, matcap) are not in any instanced batch, so
                // they must also be checked here -- otherwise the OIT pass is skipped and
                // those items are invisible.
                self.instancing.batches.iter().any(|b| b.is_transparent)
                    || scene_items.iter().any(|i| {
                        // A transparent item that is not instanceable is drawn per-object
                        // in the OIT pass below; if any exists the pass must run.
                        !i.settings.hidden
                            && crate::renderer::prepare::has_transparent_draws(i, &self.resources)
                            && !crate::renderer::prepare::is_instanceable(
                                i,
                                &self.resources,
                                &self.compute_filter_results,
                            )
                    })
            } else {
                scene_items.iter().any(|i| {
                    !i.settings.hidden
                        && crate::renderer::prepare::has_transparent_draws(i, &self.resources)
                })
            } || frame
                .scene
                .volume_meshes
                .iter()
                .any(|i| !i.settings.hidden && i.transparency.is_some())
                // Item-type plugins draw into the OIT pass through
                // `paint_transparent` for any registered plugin with a
                // non-empty submitted collection (mirrors `needs_oit` above).
                || self.any_plugin_items_submitted(frame);

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
                    crate::gpu::RenderPassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(crate::renderer::GPU_TS_OIT * 2),
                        end_of_pass_write_index: Some(crate::renderer::GPU_TS_OIT * 2 + 1),
                    }
                });
                // Clear accum to (0,0,0,0), reveal to 1.0 (no contribution yet).
                let mut oit_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("oit_pass"),
                    color_attachments: &[
                        Some(crate::gpu::RenderPassColorAttachment {
                            view: accum_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Clear(crate::gpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(crate::gpu::RenderPassColorAttachment {
                            view: reveal_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Clear(crate::gpu::Color {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 1.0,
                                    a: 1.0,
                                }),
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                        view: hdr_depth_view,
                        depth_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load, // reuse opaque depth
                            store: crate::gpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: oit_ts_writes,
                    occlusion_query_set: None,
                });

                oit_pass.set_bind_group(0, camera_bg, &[]);

                if self.instancing.use_instancing && !self.instancing.batches.is_empty() {
                    // This viewport's own cull outputs.
                    let cull0 = &self.viewport_slots[vp_idx].cull;
                    let use_indirect_oit = self.instancing.gpu_culling_enabled
                        && self.resources.cull.oit_pipeline.is_some()
                        && cull0.indirect_args_buf.is_some();

                    if use_indirect_oit {
                        if let (Some(pipeline), Some(indirect_buf)) =
                            (&self.resources.cull.oit_pipeline, &cull0.indirect_args_buf)
                        {
                            oit_pass.set_pipeline(pipeline);
                            bind_deform_group!(
                                oit_pass,
                                self.resources,
                                &self.resources.deform.dummy_bind_group
                            );
                            // Transparent batches all use the single OIT pipeline, so
                            // a run collapses when the bind group and slab chunk hold
                            // across consecutive global indices (see the opaque path).
                            let multi_draw = self.instancing.multi_draw_active();
                            let mut cur_bg: Option<*const crate::gpu::BindGroup> = None;
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            let mut run_start: u64 = 0;
                            let mut run_len: u32 = 0;
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
                                    batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) =
                                    cull0.instance_cull_bind_groups.get(&mat_key)
                                else {
                                    continue;
                                };
                                let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                                let bg_ptr = inst_tex_bg as *const crate::gpu::BindGroup;
                                let g = batch_global_idx as u64;
                                if run_len > 0
                                    && g == run_start + run_len as u64
                                    && cur_bg == Some(bg_ptr)
                                    && cur_chunks == Some(chunks)
                                {
                                    run_len += 1;
                                    continue;
                                }
                                if run_len > 0 {
                                    let dc = crate::renderer::render::emit_indirect_run(
                                        &mut oit_pass,
                                        indirect_buf,
                                        run_start,
                                        run_len,
                                        multi_draw,
                                    );
                                    self.frame_main_draw_commands
                                        .fetch_add(dc, std::sync::atomic::Ordering::Relaxed);
                                }
                                if cur_bg != Some(bg_ptr) {
                                    oit_pass.set_bind_group(1, inst_tex_bg, &[]);
                                    cur_bg = Some(bg_ptr);
                                }
                                if cur_chunks != Some(chunks) {
                                    oit_pass.set_vertex_buffer(
                                        0,
                                        resources.geometry.vertex_chunk_slice(chunks.0),
                                    );
                                    oit_pass.set_index_buffer(
                                        resources.geometry.index_chunk_slice(chunks.1),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    self.frame_main_buffer_binds
                                        .fetch_add(2, std::sync::atomic::Ordering::Relaxed);
                                    cur_chunks = Some(chunks);
                                }
                                run_start = g;
                                run_len = 1;
                            }
                            if run_len > 0 {
                                let dc = crate::renderer::render::emit_indirect_run(
                                    &mut oit_pass,
                                    indirect_buf,
                                    run_start,
                                    run_len,
                                    multi_draw,
                                );
                                self.frame_main_draw_commands
                                    .fetch_add(dc, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    } else if let Some(ref pipeline) = self.resources.oit.instanced_pipeline {
                        oit_pass.set_pipeline(pipeline);
                        bind_deform_group!(
                            oit_pass,
                            self.resources,
                            &self.resources.deform.dummy_bind_group
                        );
                        let mut cur_chunks: Option<(u32, u32)> = None;
                        for batch in &self.instancing.batches {
                            if !batch.is_transparent {
                                continue;
                            }
                            let Some(mesh) = self.resources.mesh_store.get(batch.mesh_id) else {
                                continue;
                            };
                            let mat_key = (
                                batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                            );
                            let Some(inst_tex_bg) =
                                self.resources.instancing.bind_groups.get(&mat_key)
                            else {
                                continue;
                            };
                            oit_pass.set_bind_group(1, inst_tex_bg, &[]);
                            let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                            if cur_chunks != Some(chunks) {
                                oit_pass.set_vertex_buffer(
                                    0,
                                    resources.geometry.vertex_chunk_slice(chunks.0),
                                );
                                oit_pass.set_index_buffer(
                                    resources.geometry.index_chunk_slice(chunks.1),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                self.frame_main_buffer_binds
                                    .fetch_add(2, std::sync::atomic::Ordering::Relaxed);
                                cur_chunks = Some(chunks);
                            }
                            let base_vertex = resources.geometry.base_vertex(mesh.vertex_span);
                            let first_index = resources.geometry.first_index(mesh.index_span);
                            oit_pass.draw_indexed(
                                first_index..first_index + mesh.index_count,
                                base_vertex,
                                batch.instance_offset..batch.instance_offset + batch.instance_count,
                            );
                            self.frame_main_draw_commands
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }

                    // Transparent excluded items (two-sided, active attribute, matcap) are not
                    // in any instanced batch, so the instanced OIT loop above skips them.
                    // Render them here individually so they are not invisible at opacity < 1.
                    if let Some(ref pipeline) = self.resources.oit.pipeline {
                        oit_pass.set_pipeline(pipeline);
                        let mut plugin_pipeline_active = false;
                        for (item_idx, item) in scene_items.iter().enumerate() {
                            if item.settings.hidden
                                || !crate::renderer::prepare::has_transparent_draws(
                                    item,
                                    &self.resources,
                                )
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
                            let (obj_bg, obj_inst) = match self
                                .mesh_uniforms
                                .bind_groups
                                .get(item_idx)
                                .and_then(|opt| opt.as_ref())
                            {
                                Some(bg) => (bg, self.mesh_uniforms.object_indices[item_idx]),
                                None => (&mesh.object_bind_group, 0),
                            };
                            bind_deform_group!(oit_pass, self.resources, deform_bg);
                            oit_pass.set_vertex_buffer(
                                0,
                                resources.geometry.vertex_slice(mesh.vertex_span),
                            );
                            oit_pass.set_index_buffer(
                                resources.geometry.index_slice(mesh.index_span),
                                crate::gpu::IndexFormat::Uint32,
                            );
                            if let Some((mats, bgs)) =
                                crate::renderer::prepare::active_submesh_materials(item, mesh)
                                    .zip(self.mesh_uniforms.submesh_bind_groups.get(&item_idx))
                            {
                                // Blend-material ranges only; the item's opaque
                                // ranges drew in the scene pass.
                                for (r, (mat, range)) in
                                    mats.iter().zip(&mesh.submeshes).enumerate()
                                {
                                    if item.settings.opacity >= 1.0 && !mat.is_blend() {
                                        continue;
                                    }
                                    match self.resources.material_plugin_draw(mat.shading_plugin) {
                                        Some((pp, mat_bg)) => {
                                            oit_pass.set_pipeline(&pp.oit);
                                            bind_material_group!(oit_pass, mat_bg);
                                            plugin_pipeline_active = true;
                                        }
                                        None if plugin_pipeline_active => {
                                            oit_pass.set_pipeline(pipeline);
                                            plugin_pipeline_active = false;
                                        }
                                        None => {}
                                    }
                                    let (bg, inst) = match bgs.get(r).and_then(|b| b.as_ref()) {
                                        Some(rbg) => (
                                            rbg,
                                            self.mesh_uniforms
                                                .submesh_indices
                                                .get(&item_idx)
                                                .and_then(|v| v.get(r))
                                                .copied()
                                                .unwrap_or(obj_inst),
                                        ),
                                        None => (obj_bg, obj_inst),
                                    };
                                    oit_pass.set_bind_group(1, bg, &[]);
                                    oit_pass.draw_indexed(
                                        range.first_index..range.first_index + range.index_count,
                                        0,
                                        inst..inst + 1,
                                    );
                                }
                                continue;
                            }
                            match self
                                .resources
                                .material_plugin_draw(item.material.shading_plugin)
                            {
                                Some((pp, mat_bg)) => {
                                    oit_pass.set_pipeline(&pp.oit);
                                    bind_material_group!(oit_pass, mat_bg);
                                    plugin_pipeline_active = true;
                                }
                                None if plugin_pipeline_active => {
                                    oit_pass.set_pipeline(pipeline);
                                    plugin_pipeline_active = false;
                                }
                                None => {}
                            }
                            oit_pass.set_bind_group(1, obj_bg, &[]);
                            oit_pass.draw_indexed(0..mesh.index_count, 0, obj_inst..obj_inst + 1);
                        }
                    }
                } else if let Some(ref pipeline) = self.resources.oit.pipeline {
                    oit_pass.set_pipeline(pipeline);
                    let mut plugin_pipeline_active = false;
                    for (item_idx, item) in scene_items.iter().enumerate() {
                        if item.settings.hidden
                            || !crate::renderer::prepare::has_transparent_draws(
                                item,
                                &self.resources,
                            )
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
                        let (obj_bg, obj_inst) = match self
                            .mesh_uniforms
                            .bind_groups
                            .get(item_idx)
                            .and_then(|opt| opt.as_ref())
                        {
                            Some(bg) => (bg, self.mesh_uniforms.object_indices[item_idx]),
                            None => (&mesh.object_bind_group, 0),
                        };
                        bind_deform_group!(oit_pass, self.resources, deform_bg);
                        oit_pass.set_vertex_buffer(
                            0,
                            resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        oit_pass.set_index_buffer(
                            resources.geometry.index_slice(mesh.index_span),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        if let Some((mats, bgs)) =
                            crate::renderer::prepare::active_submesh_materials(item, mesh)
                                .zip(self.mesh_uniforms.submesh_bind_groups.get(&item_idx))
                        {
                            // Blend-material ranges only; the item's opaque
                            // ranges drew in the scene pass.
                            for (r, (mat, range)) in mats.iter().zip(&mesh.submeshes).enumerate() {
                                if item.settings.opacity >= 1.0 && !mat.is_blend() {
                                    continue;
                                }
                                match self.resources.material_plugin_draw(mat.shading_plugin) {
                                    Some((pp, mat_bg)) => {
                                        oit_pass.set_pipeline(&pp.oit);
                                        bind_material_group!(oit_pass, mat_bg);
                                        plugin_pipeline_active = true;
                                    }
                                    None if plugin_pipeline_active => {
                                        oit_pass.set_pipeline(pipeline);
                                        plugin_pipeline_active = false;
                                    }
                                    None => {}
                                }
                                let (bg, inst) = match bgs.get(r).and_then(|b| b.as_ref()) {
                                    Some(rbg) => (
                                        rbg,
                                        self.mesh_uniforms
                                            .submesh_indices
                                            .get(&item_idx)
                                            .and_then(|v| v.get(r))
                                            .copied()
                                            .unwrap_or(obj_inst),
                                    ),
                                    None => (obj_bg, obj_inst),
                                };
                                oit_pass.set_bind_group(1, bg, &[]);
                                oit_pass.draw_indexed(
                                    range.first_index..range.first_index + range.index_count,
                                    0,
                                    inst..inst + 1,
                                );
                            }
                            continue;
                        }
                        match self
                            .resources
                            .material_plugin_draw(item.material.shading_plugin)
                        {
                            Some((pp, mat_bg)) => {
                                oit_pass.set_pipeline(&pp.oit);
                                bind_material_group!(oit_pass, mat_bg);
                                plugin_pipeline_active = true;
                            }
                            None if plugin_pipeline_active => {
                                oit_pass.set_pipeline(pipeline);
                                plugin_pipeline_active = false;
                            }
                            None => {}
                        }
                        oit_pass.set_bind_group(1, obj_bg, &[]);
                        oit_pass.draw_indexed(0..mesh.index_count, 0, obj_inst..obj_inst + 1);
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
                    if let Some(pipeline) = self.resources.pt.pipeline.as_ref() {
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
                            let Some(gpu) = resources.content.projected_tet_store.get(pt_id) else {
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
                                        .content
                                        .colourmap_views
                                        .get(id.0)
                                        .and(resources.pt.lut_bind_groups.get(&id.0))
                                })
                                .or(resources.pt.fallback_lut_bind_group.as_ref());
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
                self.resources.oit.composite_pipeline.as_ref(),
                slot_hdr.oit_composite_bind_group.as_ref(),
            ) {
                let hdr_view = &slot_hdr.hdr_view;
                let mut composite_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("oit_composite_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: hdr_view,
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
                composite_pass.set_pipeline(pipeline);
                composite_pass.set_bind_group(0, bg, &[]);
                composite_pass.draw(0..3, 0..1);
            }
        }
    }

    fn hdr_scatter(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
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
                .ensure_scatter_pipeline(device, crate::gpu::TextureFormat::Rgba16Float);
            self.resources
                .ensure_scatter_composite_pipeline(device, crate::gpu::TextureFormat::Rgba16Float);
            self.resources
                .ensure_scatter_temporal_resolve_pipeline(device);

            // (Re)allocate intermediates if requested size / mode changed.
            let needs_alloc = match self.scatter_viewport_states[vp_idx].as_ref() {
                None => true,
                Some(s) => s.size != [tw, th] || s.downsampled != want_downsample,
            };
            if needs_alloc {
                let make_tex = |label: &str| {
                    device.create_texture(&crate::gpu::TextureDescriptor {
                        label: Some(label),
                        size: crate::gpu::Extent3d {
                            width: tw,
                            height: th,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: crate::gpu::TextureDimension::D2,
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                            | crate::gpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                };
                let raw_tex = make_tex("scatter_raw_current");
                let hist_a_tex = make_tex("scatter_history_a");
                let hist_b_tex = make_tex("scatter_history_b");
                let raw_view = raw_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
                let hist_a_view =
                    hist_a_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
                let hist_b_view =
                    hist_b_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
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
                self.resources.ensure_scatter_refraction_pipeline(
                    device,
                    crate::gpu::TextureFormat::Rgba16Float,
                );
                self.resources.ensure_scatter_refraction_blit_pipeline(
                    device,
                    crate::gpu::TextureFormat::Rgba16Float,
                );

                // Allocate (or resize) the refraction source texture at HDR
                // resolution. Replaces the per-viewport entry's view when the
                // scene size changes.
                let need_realloc = {
                    let s = self.scatter_viewport_states[vp_idx].as_ref().unwrap();
                    s.refraction_source_view.is_none() || s.refraction_source_size != [sw, sh]
                };
                if need_realloc {
                    let src_tex = device.create_texture(&crate::gpu::TextureDescriptor {
                        label: Some("scatter_refraction_source"),
                        size: crate::gpu::Extent3d {
                            width: sw.max(1),
                            height: sh.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: crate::gpu::TextureDimension::D2,
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                            | crate::gpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let src_view =
                        src_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
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
                        self.resources.scatter.refraction_blit_pipeline.as_ref()
                    {
                        let mut pass =
                            encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                                #[cfg(feature = "wgpu29")]
                                multiview_mask: None,
                                label: Some("scatter_refraction_blit_pass"),
                                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                                    view: src_view,
                                    resolve_target: None,
                                    ops: crate::gpu::Operations {
                                        load: crate::gpu::LoadOp::Clear(
                                            crate::gpu::Color::TRANSPARENT,
                                        ),
                                        store: crate::gpu::StoreOp::Store,
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
                        self.resources.scatter.refraction_pipeline.as_ref(),
                        self.resources.scatter.refraction_per_volume_bg.as_ref(),
                    ) {
                        let mut pass =
                            encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                                #[cfg(feature = "wgpu29")]
                                multiview_mask: None,
                                label: Some("scatter_refraction_pass"),
                                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                                    view: &slot_hdr.hdr_view,
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
            let mut per_vol_tex_bgs: Vec<crate::gpu::BindGroup> = Vec::with_capacity(n as usize);
            for (volume, _, _) in self.prepared_scatter_volumes.iter().take(n as usize) {
                let (lut_id, density_id) =
                    crate::resources::DeviceResources::scatter_volume_tex_ids(volume);
                let bg = self
                    .resources
                    .ensure_scatter_per_volume_tex_bg(device, queue, lut_id, density_id);
                per_vol_tex_bgs.push(bg);
            }

            if n > 0 {
                let s = self.scatter_viewport_states[vp_idx].as_ref().unwrap();
                let raw_view = &s.raw_current_view;
                if let (Some(pipeline), Some(per_vol_bg), Some(frame_bg)) = (
                    self.resources.scatter.pipeline.as_ref(),
                    self.resources.scatter.per_volume_bg.as_ref(),
                    self.resources.scatter.frame_bg.as_ref(),
                ) {
                    let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("scatter_volume_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: raw_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                                store: crate::gpu::StoreOp::Store,
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
                let source_for_composite: &crate::gpu::BindGroup = if scatter.temporal {
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
                        self.resources.scatter.temporal_resolve_pipeline.as_ref()
                    {
                        let mut pass =
                            encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                                #[cfg(feature = "wgpu29")]
                                multiview_mask: None,
                                label: Some("scatter_temporal_resolve_pass"),
                                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                                    view: target_view,
                                    resolve_target: None,
                                    ops: crate::gpu::Operations {
                                        load: crate::gpu::LoadOp::Clear(
                                            crate::gpu::Color::TRANSPARENT,
                                        ),
                                        store: crate::gpu::StoreOp::Store,
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
                if let Some(composite_pipeline) = self.resources.scatter.composite_pipeline.as_ref()
                {
                    let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("scatter_composite_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: &slot_hdr.hdr_view,
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

    fn hdr_lic(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let resources = &self.resources;
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
                self.resources.lic.surface_pipeline.as_ref(),
                self.resources.lic.advect_pipeline.as_ref(),
            ) {
                let camera_bg = &slot.camera_bind_group;
                // Pass 1: surface vector pass (clears lic_vector_texture first).
                {
                    let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("lic_surface_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: &slot_hdr.lic_vector_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                                store: crate::gpu::StoreOp::Store,
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
                        pass.set_vertex_buffer(
                            0,
                            resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        pass.set_vertex_buffer(1, vec_buf.slice(..));
                        pass.set_index_buffer(
                            resources.geometry.index_slice(mesh.index_span),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
                // Pass 2: advect pass (fullscreen, writes LIC intensity to lic_output_texture).
                {
                    let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("lic_advect_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: &slot_hdr.lic_output_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Clear(crate::gpu::Color {
                                    r: 0.5,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: crate::gpu::StoreOp::Store,
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

    fn hdr_outline_composite(
        &mut self,
        ctx: &HdrFrameCtx,
        encoder: &mut crate::gpu::CommandEncoder,
    ) {
        let vp_idx = ctx.vp_idx;
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
            || slot.plugin_outline_present
        {
            // Prefer the HDR-format pipeline; fall back to LDR single-sample.
            let hdr_pipeline = self
                .resources
                .outline
                .composite_pipeline_hdr
                .as_ref()
                .or(self.resources.outline.composite_pipeline_single.as_ref());
            if let Some(pipeline) = hdr_pipeline {
                let bg = &slot_hdr.outline_composite_bind_group;
                let hdr_view = &slot_hdr.hdr_view;
                let hdr_depth_view = &slot_hdr.hdr_depth_view;
                let mut outline_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("hdr_outline_composite_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: hdr_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Load,
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view: hdr_depth_view,
                                depth_ops: Some(crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Load,
                                    store: crate::gpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            },
                        ),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                outline_pass.set_pipeline(pipeline);
                outline_pass.set_bind_group(0, bg, &[]);
                outline_pass.draw(0..3, 0..1);
            }
        }
    }

    /// Foreground pass: draw `SceneFrame::foreground_items` (and foreground
    /// plugin items) over the composited scene, against a freshly cleared
    /// depth target. Runs after the outline composite and before the
    /// post-effect sub-passes, so foreground emissives feed bloom and DOF
    /// sees the foreground colour (its coverage mask keeps covered pixels
    /// sharp). The cleared own depth is what makes foreground geometry
    /// neither occluded by nor clipped into world geometry; the group-0
    /// bind group carries the foreground camera and disabled clip planes.
    fn hdr_foreground(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let frame = ctx.frame;
        if !self.foreground_active(frame) {
            return;
        }
        let vp_idx = ctx.vp_idx;
        let resources = &self.resources;
        let slot = &self.viewport_slots[vp_idx];
        let slot_hdr = slot.hdr.as_ref().unwrap();
        let Some(fg_depth_view) = slot_hdr.foreground_depth_view.as_ref() else {
            return;
        };
        let (Some(hdr_solid), Some(hdr_solid_two_sided), Some(hdr_trans), Some(hdr_wf)) = (
            &resources.scene.hdr_solid,
            &resources.scene.hdr_solid_two_sided,
            &resources.scene.hdr_transparent,
            &resources.scene.hdr_wireframe,
        ) else {
            return;
        };

        let fg_camera = frame
            .camera
            .render_camera
            .foreground_camera(frame.effects.foreground.as_ref());

        // Opaque front-to-back, then blended back-to-front. Foreground
        // transparency is plain sorted alpha blending against the foreground
        // depth, not OIT.
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

        let mut render_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(feature = "wgpu29")]
            multiview_mask: None,
            label: Some("hdr_foreground_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: &slot_hdr.hdr_view,
                resolve_target: None,
                ops: crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: fg_depth_view,
                depth_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Clear(1.0),
                    store: crate::gpu::StoreOp::Store,
                }),
                stencil_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Clear(0),
                    store: crate::gpu::StoreOp::Discard,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_bind_group(0, &slot.foreground_camera_bind_group, &[]);

        for (idx, item) in opaque.iter().chain(transparent.iter()) {
            let solid_pl = if item.material.is_two_sided() {
                hdr_solid_two_sided
            } else {
                hdr_solid
            };
            let obj_bg = slot
                .foreground_objects
                .get(*idx)
                .and_then(|e| e.bind_group.as_ref());
            draw_mesh_item(
                resources,
                &self.compute_filter_results,
                &mut render_pass,
                item,
                obj_bg,
                // Foreground buffers hold one element each: draw at instance 0.
                0,
                false,
                true,
                solid_pl,
                hdr_solid_two_sided,
                hdr_trans,
                hdr_wf,
                // Foreground items draw through the positional
                // foreground_objects cache, which has no per-range entries;
                // they render with the single item material.
                None,
                None,
                None,
            );
        }

        self.dispatch_plugin_paint_foreground(&mut render_pass, frame, &fg_camera);
    }

    fn hdr_post_effects(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let vp_idx = ctx.vp_idx;
        let frame = ctx.frame;
        let pp = &frame.effects.post_process;
        let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();

        // Effect throttling. Flag was computed in prepare() so that
        // FrameStats reports exactly what fired rather than an approximation.
        let throttle_effects = self.degradation_effects_throttled;

        // -----------------------------------------------------------------------
        // SSAO pass.
        // -----------------------------------------------------------------------
        if pp.ssao && !throttle_effects {
            if let Some(ssao_pipeline) = &self.resources.post.ssao_pipeline {
                // The SSAO slot begins on the occlusion pass and ends on the
                // blur pass (or on the occlusion pass when there is no blur).
                let has_blur = self.resources.post.ssao_blur_pipeline.is_some();
                {
                    let ts = self.ts_writes_for(crate::renderer::GPU_TS_SSAO, true, !has_blur);
                    let mut ssao_pass =
                        encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                            #[cfg(feature = "wgpu29")]
                            multiview_mask: None,
                            label: Some("ssao_pass"),
                            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                                view: &slot_hdr.ssao_view,
                                resolve_target: None,
                                ops: crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Clear(crate::gpu::Color::WHITE),
                                    store: crate::gpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: ts,
                            occlusion_query_set: None,
                        });
                    ssao_pass.set_pipeline(ssao_pipeline);
                    ssao_pass.set_bind_group(0, &slot_hdr.ssao_bg, &[]);
                    ssao_pass.draw(0..3, 0..1);
                }

                // SSAO blur pass.
                if let Some(ssao_blur_pipeline) = &self.resources.post.ssao_blur_pipeline {
                    let ts = self.ts_writes_for(crate::renderer::GPU_TS_SSAO, false, true);
                    let mut ssao_blur_pass =
                        encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                            #[cfg(feature = "wgpu29")]
                            multiview_mask: None,
                            label: Some("ssao_blur_pass"),
                            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                                view: &slot_hdr.ssao_blur_view,
                                resolve_target: None,
                                ops: crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Clear(crate::gpu::Color::WHITE),
                                    store: crate::gpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: ts,
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
            if let Some(cs_pipeline) = &self.resources.post.contact_shadow_pipeline {
                let mut cs_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("contact_shadow_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: &slot_hdr.contact_shadow_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(crate::gpu::Color::WHITE),
                            store: crate::gpu::StoreOp::Store,
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
            if let Some(bloom_threshold_pipeline) = &self.resources.post.bloom_threshold_pipeline {
                // The bloom slot begins on the threshold pass and ends on the
                // last blur pass (or here when there is no blur pipeline).
                let has_blur = self.resources.post.bloom_blur_pipeline.is_some();
                {
                    let ts = self.ts_writes_for(crate::renderer::GPU_TS_BLOOM, true, !has_blur);
                    let mut threshold_pass =
                        encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                            #[cfg(feature = "wgpu29")]
                            multiview_mask: None,
                            label: Some("bloom_threshold_pass"),
                            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                                view: &slot_hdr.bloom_threshold_view,
                                resolve_target: None,
                                ops: crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Clear(crate::gpu::Color::BLACK),
                                    store: crate::gpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: ts,
                            occlusion_query_set: None,
                        });
                    threshold_pass.set_pipeline(bloom_threshold_pipeline);
                    threshold_pass.set_bind_group(0, &slot_hdr.bloom_threshold_bg, &[]);
                    threshold_pass.draw(0..3, 0..1);
                }

                // 4 ping-pong H+V blur passes for a wide glow.
                // Pass 1: threshold -> ping -> pong. Passes 2-4: pong -> ping -> pong.
                if let Some(blur_pipeline) = &self.resources.post.bloom_blur_pipeline {
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
                                encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                                    #[cfg(feature = "wgpu29")]
                                    multiview_mask: None,
                                    label: Some("bloom_blur_h_pass"),
                                    color_attachments: &[Some(
                                        crate::gpu::RenderPassColorAttachment {
                                            view: bloom_ping_view,
                                            resolve_target: None,
                                            ops: crate::gpu::Operations {
                                                load: crate::gpu::LoadOp::Clear(
                                                    crate::gpu::Color::BLACK,
                                                ),
                                                store: crate::gpu::StoreOp::Store,
                                            },
                                            depth_slice: None,
                                        },
                                    )],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            h_pass.set_pipeline(blur_pipeline);
                            h_pass.set_bind_group(0, h_bg, &[]);
                            h_pass.draw(0..3, 0..1);
                        }
                        // V pass: ping -> pong. The last iteration closes the
                        // bloom timing slot.
                        {
                            let ts = (i == BLUR_ITERATIONS - 1)
                                .then(|| {
                                    self.ts_writes_for(crate::renderer::GPU_TS_BLOOM, false, true)
                                })
                                .flatten();
                            let mut v_pass =
                                encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                                    #[cfg(feature = "wgpu29")]
                                    multiview_mask: None,
                                    label: Some("bloom_blur_v_pass"),
                                    color_attachments: &[Some(
                                        crate::gpu::RenderPassColorAttachment {
                                            view: bloom_pong_view,
                                            resolve_target: None,
                                            ops: crate::gpu::Operations {
                                                load: crate::gpu::LoadOp::Clear(
                                                    crate::gpu::Color::BLACK,
                                                ),
                                                store: crate::gpu::StoreOp::Store,
                                            },
                                            depth_slice: None,
                                        },
                                    )],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: ts,
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
            if let Some(dof_pipeline) = &self.resources.post.dof_pipeline {
                let mut dof_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("dof_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: &slot_hdr.dof_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(crate::gpu::Color::BLACK),
                            store: crate::gpu::StoreOp::Store,
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

    fn hdr_tonemap_resolve(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
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
        if let Some(tone_map_pipeline) = &self.resources.post.tone_map_pipeline {
            let tone_target: &crate::gpu::TextureView = if use_fxaa {
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
                crate::gpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(crate::renderer::GPU_TS_POST * 2),
                    end_of_pass_write_index: Some(crate::renderer::GPU_TS_POST * 2 + 1),
                }
            });
            let mut tone_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("tone_map_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: tone_target,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color::BLACK),
                        store: crate::gpu::StoreOp::Store,
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
            if let Some(fxaa_pipeline) = &self.resources.post.fxaa_pipeline {
                let fxaa_target: &crate::gpu::TextureView = if use_hdr_upscale {
                    slot_hdr.upscale_view.as_ref().unwrap()
                } else {
                    output_view
                };
                let ts = self.ts_writes_for(crate::renderer::GPU_TS_FXAA, true, true);
                let mut fxaa_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("fxaa_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: fxaa_target,
                        resolve_target: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(crate::gpu::Color::BLACK),
                            store: crate::gpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: ts,
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
                if let Some(pipeline) = &self.resources.post.dyn_res_upscale_pipeline {
                    let mut upscale_pass =
                        encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                            #[cfg(feature = "wgpu29")]
                            multiview_mask: None,
                            label: Some("hdr_upscale_pass"),
                            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                                view: output_view,
                                resolve_target: None,
                                ops: crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Clear(crate::gpu::Color::BLACK),
                                    store: crate::gpu::StoreOp::Store,
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
                if let Some(blit_pipeline) = &self.resources.post.depth_blit_pipeline {
                    let mut blit_pass =
                        encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                            #[cfg(feature = "wgpu29")]
                            multiview_mask: None,
                            label: Some("depth_blit_pass"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(
                                crate::gpu::RenderPassDepthStencilAttachment {
                                    view: &slot_hdr.output_depth_view,
                                    depth_ops: Some(crate::gpu::Operations {
                                        load: crate::gpu::LoadOp::Clear(1.0),
                                        store: crate::gpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    blit_pass.set_pipeline(blit_pipeline);
                    blit_pass.set_bind_group(0, blit_bg, &[]);
                    blit_pass.draw(0..3, 0..1);
                }
            }
        }

        // Foreground depth stamp: write near depth into output_depth_view
        // wherever the foreground pass drew, so the post-tone-map passes
        // below (grid, ground plane, gizmos) are occluded by foreground
        // geometry. Runs after the depth blit in both render-scale cases
        // (at scale 1.0 output_depth_view aliases the scene depth, which the
        // tone map pass has already consumed).
        if self.foreground_active(ctx.frame) {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let (Some(fg_view), Some(pipeline), Some(bgl)) = (
                slot_hdr.foreground_depth_only_view.as_ref(),
                self.resources.post.foreground_stamp_pipeline.as_ref(),
                self.resources.post.foreground_stamp_bgl.as_ref(),
            ) {
                let stamp_bg = ctx
                    .device
                    .create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("foreground_stamp_bg"),
                        layout: bgl,
                        entries: &[crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: crate::gpu::BindingResource::TextureView(fg_view),
                        }],
                    });
                let mut stamp_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("foreground_depth_stamp_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                stamp_pass.set_pipeline(pipeline);
                stamp_pass.set_bind_group(0, &stamp_bg, &[]);
                stamp_pass.draw(0..3, 0..1);
            }
        }
    }

    fn hdr_scene_overlays(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let resources = &self.resources;
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
            let mut grid_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("hdr_grid_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: &slot_hdr.output_depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            grid_pass.set_pipeline(&self.resources.guides.grid_pipeline);
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
            let mut gp_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("hdr_ground_plane_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: &slot_hdr.output_depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
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
            if let Some(overlay_pipeline) = &self.resources.screen_image.pipeline {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let dc_pipeline = self.resources.screen_image.dc_pipeline.as_ref();
                let mut img_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("screen_image_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Discard,
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
                let mut overlay_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(feature = "wgpu29")]
                        multiview_mask: None,
                        label: Some("hdr_editor_overlay_pass"),
                        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                            view: output_view,
                            resolve_target: None,
                            ops: crate::gpu::Operations {
                                load: crate::gpu::LoadOp::Load,
                                store: crate::gpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(
                            crate::gpu::RenderPassDepthStencilAttachment {
                                view: &slot_hdr.output_depth_view,
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

                if frame.interaction.gizmo_model.is_some() && slot.gizmo_index_count > 0 {
                    overlay_pass.set_pipeline(&self.resources.gizmo.pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    overlay_pass.set_bind_group(1, &slot.gizmo_bind_group, &[]);
                    overlay_pass.set_vertex_buffer(0, slot.gizmo_vertex_buffer.slice(..));
                    overlay_pass.set_index_buffer(
                        slot.gizmo_index_buffer.slice(..),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    overlay_pass.draw_indexed(0..slot.gizmo_index_count, 0, 0..1);
                }

                if !slot.constraint_line_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.guides.overlay_line_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, index_count, _ubuf, bg) in &slot.constraint_line_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass
                            .set_index_buffer(ibuf.slice(..), crate::gpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*index_count, 0, 0..1);
                    }
                }

                if !slot.clip_plane_fill_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.guides.overlay_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_fill_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass
                            .set_index_buffer(ibuf.slice(..), crate::gpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }

                if !slot.clip_plane_line_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.guides.overlay_line_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_line_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass
                            .set_index_buffer(ibuf.slice(..), crate::gpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }

                if !slot.xray_object_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.outline.xray_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (mesh_id, _buf, bg) in &slot.xray_object_buffers {
                        let Some(mesh) = self.resources.mesh_store.get(*mesh_id) else {
                            continue;
                        };
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(
                            0,
                            resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        overlay_pass.set_index_buffer(
                            resources.geometry.index_slice(mesh.index_span),
                            crate::gpu::IndexFormat::Uint32,
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
                let mut axes_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(feature = "wgpu29")]
                    multiview_mask: None,
                    label: Some("hdr_axes_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Load,
                            store: crate::gpu::StoreOp::Discard,
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

    fn hdr_final_overlay(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
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
        let hdr_blur_bg: Option<crate::gpu::BindGroup> = if needs_hdr_blur && has_overlay {
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
            let mut overlay_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(feature = "wgpu29")]
                multiview_mask: None,
                label: Some("overlay_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: hdr_depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Discard,
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

#[cfg(test)]
mod decal_scissor_tests {
    use super::{DecalScissor, decal_scissor};
    use glam::{Mat4, Vec3};

    fn view_proj() -> Mat4 {
        let proj = Mat4::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        // Z-up camera 10 units back on -Y, looking at the origin.
        let view = Mat4::look_at_rh(Vec3::new(0.0, -10.0, 2.0), Vec3::ZERO, Vec3::Z);
        proj * view
    }

    #[test]
    fn centered_box_yields_subrect() {
        let model = Mat4::from_scale(Vec3::splat(1.0));
        match decal_scissor(&model, &view_proj(), 1920, 1080) {
            DecalScissor::Rect(x, y, w, h) => {
                assert!(w > 0 && h > 0);
                assert!(
                    w < 1920 && h < 1080,
                    "small distant box should not fill the screen"
                );
                assert!(
                    x + w <= 1920 && y + h <= 1080,
                    "rect must stay within the target"
                );
            }
            _ => panic!("expected a sub-rect for a centered box"),
        }
    }

    #[test]
    fn far_offscreen_box_skips() {
        // Far off to the +X side but still in front of the camera.
        let model = Mat4::from_translation(Vec3::new(500.0, 100.0, 0.0));
        assert!(matches!(
            decal_scissor(&model, &view_proj(), 1920, 1080),
            DecalScissor::Skip
        ));
    }

    #[test]
    fn box_enclosing_camera_falls_back_to_full() {
        // A large box centred on the camera puts a corner behind the near plane.
        let model =
            Mat4::from_translation(Vec3::new(0.0, -10.0, 2.0)) * Mat4::from_scale(Vec3::splat(4.0));
        assert!(matches!(
            decal_scissor(&model, &view_proj(), 1920, 1080),
            DecalScissor::Full
        ));
    }
}
