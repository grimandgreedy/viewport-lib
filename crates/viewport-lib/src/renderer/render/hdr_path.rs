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
    /// The frame's composite inputs and tone-map uniform as computed in the
    /// preamble, kept so the tone-map stage can re-derive them with external
    /// slot contributions applied.
    composite_inputs: crate::resources::CompositeInputs,
    tm_uniform: crate::resources::ToneMapUniform,
}

/// Build the per-frame, per-viewport context handed to external post-effect
/// producers.
fn post_effect_ctx<'a>(
    device: &'a crate::gpu::Device,
    slot_hdr: &'a crate::resources::ViewportHdrState,
    frame: &'a FrameData,
    viewport_index: usize,
) -> crate::plugin_api::PostEffectContext<'a> {
    crate::plugin_api::PostEffectContext {
        device,
        viewport_index,
        scene_size: slot_hdr.scene_size,
        output_size: slot_hdr.output_size,
        proj: frame.camera.render_camera.projection,
        view: frame.camera.render_camera.view,
        near: frame.camera.render_camera.near,
        far: frame.camera.render_camera.far,
        scene_colour: &slot_hdr.hdr_view,
        scene_depth: &slot_hdr.hdr_depth_only_view,
        post: &frame.effects.post_process,
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
            let key = PipelineKey::two_sided(item.material.is_two_sided());
            let pl = if let Some((pp, _)) = plug {
                if item.settings.opacity < 1.0 {
                    if hdr {
                        &pp.hdr_transparent
                    } else {
                        &pp.ldr.transparent
                    }
                } else if hdr {
                    pp.hdr_opaque.get(key)
                } else {
                    select_two_sided(key, &pp.ldr.solid, &pp.ldr.solid_two_sided)
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
                let range_key = PipelineKey::two_sided(mat.is_two_sided());
                let pl = if let Some((pp, _)) = plug_r {
                    if is_trans {
                        if hdr {
                            &pp.hdr_transparent
                        } else {
                            &pp.ldr.transparent
                        }
                    } else if hdr {
                        pp.hdr_opaque.get(range_key)
                    } else {
                        select_two_sided(range_key, &pp.ldr.solid, &pp.ldr.solid_two_sided)
                    }
                } else if is_trans {
                    trans_pl
                } else {
                    select_two_sided(range_key, solid_pl, solid_two_sided_pl)
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
                    &pp.hdr_transparent
                } else {
                    pp.hdr_opaque
                        .get(PipelineKey::two_sided(item.material.is_two_sided()))
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
    ///
    /// Shared with the LDR path, which times the same overlay slot when a
    /// backdrop-blur shape forces the overlay into its own pass.
    pub(crate) fn ts_writes_for(
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

        // The background colour is linear at the pipeline boundary. Clear the
        // linear HDR scene texture with it directly, matching the tone-map
        // uniform below (which composites the same linear value) and the LDR
        // path's clear. An earlier powf(2.2) here treated the value as sRGB and
        // decoded it a second time, so the HDR and LDR paths disagreed on the
        // background for the same scene.
        let hdr_clear_rgb = [bg_colour[0], bg_colour[1], bg_colour[2]];

        // Which effect inputs feed the tone-map composite this frame. Built
        // once here so the uniform's enable lanes below and the bind group's
        // view selection share one source and cannot disagree.
        // The grade LUT is validated against the texture store up front: the
        // renderer needs an owned texture to read the LUT's height, so ids
        // registered as external views are ignored.
        let grade_lut = pp.grade_lut.filter(|id| {
            self.resources
                .content
                .textures
                .get(*id)
                .is_some_and(|t| t.texture.is_some())
        });
        let grade_lut_size = grade_lut
            .and_then(|id| self.resources.content.textures.get(id))
            .and_then(|t| t.texture.as_ref())
            .map(|t| t.height() as f32)
            .unwrap_or(0.0);
        let composite_inputs = crate::resources::CompositeInputs {
            bloom: pp.bloom.enabled,
            ssao: pp.ssao,
            contact_shadows: pp.contact_shadows.enabled,
            lic: scene_items
                .iter()
                .any(|i| i.lic.is_some() && !i.settings.hidden),
            dof: pp.dof.enabled,
            foreground: self.foreground_active(frame),
            grade_lut,
        };

        // Upload tone map uniform into the per-viewport buffer.
        let mode = match frame.effects.display.operator {
            crate::renderer::ToneMapping::Reinhard => 0u32,
            crate::renderer::ToneMapping::Aces => 1u32,
            crate::renderer::ToneMapping::KhronosNeutral => 2u32,
        };
        let tm_uniform = crate::resources::ToneMapUniform {
            // Exposure is applied from the per-viewport exposure state buffer
            // (the composite's exposure slot), not this field; kept at 1.0 for
            // layout stability.
            exposure: 1.0,
            mode,
            bloom_enabled: composite_inputs.bloom as u32,
            ssao_enabled: composite_inputs.ssao as u32,
            contact_shadows_enabled: composite_inputs.contact_shadows as u32,
            edl_enabled: if pp.edl.enabled { 1 } else { 0 },
            edl_radius: pp.edl.radius,
            edl_strength: pp.edl.strength,
            background_colour: bg_colour,
            near_plane: frame.camera.render_camera.near,
            far_plane: frame.camera.render_camera.far,
            lic_enabled: composite_inputs.lic as u32,
            _pad_lic: 0.0,
            foreground_enabled: composite_inputs.foreground as u32,
            vignette_amount: if pp.vignette.enabled {
                pp.vignette.amount.clamp(0.0, 1.0)
            } else {
                0.0
            },
            vignette_radius: pp.vignette.radius,
            vignette_softness: pp.vignette.softness,
            grade_enabled: composite_inputs.grade_lut.is_some() as u32,
            grade_lut_size,
            _pad: [0; 2],
        };
        {
            let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            queue.write_buffer(
                &hdr.tone_map_uniform_buf,
                0,
                bytemuck::cast_slice(&[tm_uniform]),
            );

            // Producer uniforms: every composite-input producer derives its
            // per-frame uniform from the same inputs.
            let inputs = crate::resources::ProducerFrameInputs {
                post: pp,
                proj: frame.camera.render_camera.projection,
                view: frame.camera.render_camera.view,
                near: frame.camera.render_camera.near,
                far: frame.camera.render_camera.far,
                first_light: frame.effects.lighting.lights.first(),
                foreground_active: composite_inputs.foreground,
                exposure: frame.effects.display.exposure,
            };
            for producer in self.resources.post_producers() {
                if producer.enabled(&inputs) {
                    producer.upload(queue, hdr, &inputs);
                }
            }
        }

        // External post-effect producers and stages: run any deferred GPU
        // init, then this frame's uniform writes.
        self.init_pending_post_effect_producers(device);
        self.frame_external_slot_views.clear();
        if !self.post_effect_producers.is_empty() || !self.post_effect_stages.is_empty() {
            let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            let ctx = post_effect_ctx(device, hdr, frame, vp_idx);
            for entry in &mut self.post_effect_producers {
                if entry.gpu_ready && entry.producer.enabled() {
                    entry.producer.prepare(queue, &ctx);
                }
            }
            for entry in &mut self.post_effect_stages {
                if entry.gpu_ready && entry.stage.enabled() {
                    entry.stage.prepare(queue, &ctx);
                }
            }
        }

        // Pre-allocate the foreground depth target so the tone-map / DOF bind
        // groups rebuilt below can reference it as the coverage mask. The pass
        // draws into hdr_view after the SSAA resolve, so the depth target is
        // scene-sized (matching hdr_view, OIT, and the other post-resolve
        // passes), not SSAA-sized.
        let use_foreground = composite_inputs.foreground;
        if use_foreground {
            let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
            let [sw, sh] = hdr.scene_size;
            self.resources
                .ensure_viewport_foreground_depth(device, hdr, sw, sh);
        }

        // Rebuild the tone-map bind group with this frame's composite inputs.
        // External producers have not encoded yet; if any contribute a slot
        // view this frame, the tone-map stage rebuilds again with overrides.
        {
            let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
            self.resources
                .rebuild_tone_map_bind_group(device, hdr, composite_inputs, &[]);
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
            composite_inputs,
            tm_uniform,
        };

        self.hdr_scene_pass(&ctx, &mut encoder);
        self.hdr_store_hiz_depth(&ctx, &mut encoder);
        self.hdr_external_instances(&ctx, &mut encoder);
        self.hdr_ssaa_refraction(&ctx, &mut encoder);
        // Item-type plugins that composite onto the finished opaque surfaces:
        // decals stamp here, underneath the selection affordances and the
        // depth-read transparency that follow.
        self.dispatch_plugin_encode(
            &mut encoder,
            frame,
            crate::plugin_api::EncoderScope::OnOpaqueSurfaces,
            device,
            queue,
            vp_idx,
        );
        self.hdr_sub_highlight(&ctx, &mut encoder);
        self.hdr_depth_read_pass(&ctx, &mut encoder);
        // Item-type plugins that own passes rather than draws get the encoder
        // here, with the opaque image and final opaque depth in hand.
        self.dispatch_plugin_encode(
            &mut encoder,
            frame,
            crate::plugin_api::EncoderScope::AfterOpaque,
            device,
            queue,
            vp_idx,
        );
        self.hdr_oit(&ctx, &mut encoder);
        // The second scope: opaque plus resolved transparency, which is what a
        // volumetric effect composites over. The scatter-volume plugin encodes
        // its ray-march, temporal blend and composite here.
        self.dispatch_plugin_encode(
            &mut encoder,
            frame,
            crate::plugin_api::EncoderScope::AfterTransparent,
            device,
            queue,
            vp_idx,
        );
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
                #[cfg(any(wgpu29, wgpu30))]
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
                            .clip
                            .objects
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
                                // GPU-driven submission: when the compaction pass
                                // precomputed draw groups (bindless + native
                                // multi-draw-count), issue one
                                // multi_draw_indexed_indirect_count per group and
                                // skip the CPU run-forming entirely.
                                let mut did_groups = false;
                                if !self.instancing.draw_groups.is_empty() {
                                    if let (Some(compacted), Some(counts)) = (
                                        cull0.compacted_args_buf.as_ref(),
                                        cull0.draw_counts_buf.as_ref(),
                                    ) {
                                        let mut cur_pipe: Option<(bool, bool)> = None;
                                        let mut cur_chunks: Option<(u32, u32)> = None;
                                        for group in self.instancing.draw_groups.iter() {
                                            // The group carries its pipeline
                                            // selectors and geometry chunk, so no
                                            // per-group batch / mesh-store lookup
                                            // is needed (the count path draws every
                                            // batch's args from the compacted
                                            // buffer the GPU wrote).
                                            let pipe_key = (group.two_sided, group.no_discard);
                                            if cur_pipe != Some(pipe_key) {
                                                let key = PipelineKey {
                                                    two_sided: group.two_sided,
                                                    no_discard_eligible: group.no_discard,
                                                    ..PipelineKey::default()
                                                };
                                                render_pass.set_pipeline(select_opaque_solid(
                                                    key,
                                                    pipeline,
                                                    pipeline_two_sided,
                                                    nodiscard_pipes.0,
                                                    nodiscard_pipes.1,
                                                ));
                                                cur_pipe = Some(pipe_key);
                                            }
                                            let chunks = (group.vertex_chunk, group.index_chunk);
                                            if cur_chunks != Some(chunks) {
                                                // The colour bind group carries this
                                                // chunk's uv1 buffer, so rebind group 1
                                                // whenever the slab chunk changes.
                                                let Some(bg) = cull0
                                                    .bindless_cull_bind_groups
                                                    .get(&resources.uv1_chunk_key(chunks.0))
                                                else {
                                                    continue;
                                                };
                                                render_pass.set_bind_group(1, bg, &[]);
                                                render_pass.set_vertex_buffer(
                                                    0,
                                                    resources.geometry.vertex_chunk_slice(chunks.0),
                                                );
                                                render_pass.set_index_buffer(
                                                    resources.geometry.index_chunk_slice(chunks.1),
                                                    crate::gpu::IndexFormat::Uint32,
                                                );
                                                cur_chunks = Some(chunks);
                                            }
                                            render_pass.multi_draw_indexed_indirect_count(
                                                compacted,
                                                group.arg_base as u64 * 20,
                                                counts,
                                                group.count_index as u64 * 4,
                                                group.size,
                                            );
                                            self.frame_main_draw_commands
                                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        }
                                        did_groups = true;
                                    }
                                }
                                let mut cur_pipe: Option<(bool, bool)> = None;
                                let mut cur_bg: Option<*const crate::gpu::BindGroup> = None;
                                let mut cur_chunks: Option<(u32, u32)> = None;
                                let mut run_start: u64 = 0;
                                let mut run_len: u32 = 0;
                                for (batch_global_idx, batch) in opaque_batches
                                    .iter()
                                    .take(if did_groups { 0 } else { usize::MAX })
                                {
                                    // Plugin batches draw in the dedicated plugin
                                    // sub-loop below; skipping here breaks the
                                    // current run on the global-index gap, exactly
                                    // as an interleaved transparent batch does.
                                    if batch.shading_plugin.is_some() {
                                        continue;
                                    }
                                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                        continue;
                                    };
                                    let mat_key = (
                                        batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        batch
                                            .metallic_roughness_id
                                            .map(|t| t.raw())
                                            .unwrap_or(u64::MAX),
                                        batch.emissive_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                        resources.uv1_chunk_key(mesh.vertex_span.chunk),
                                    );
                                    let Some(inst_tex_bg) =
                                        resources.instanced_cull_colour_bind_group(cull0, mat_key)
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
                                        let key = PipelineKey {
                                            two_sided: batch.two_sided,
                                            no_discard_eligible: no_discard,
                                            ..PipelineKey::default()
                                        };
                                        render_pass.set_pipeline(select_opaque_solid(
                                            key,
                                            pipeline,
                                            pipeline_two_sided,
                                            nodiscard_pipes.0,
                                            nodiscard_pipes.1,
                                        ));
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
                                // Plugin batches draw in the plugin sub-loop below.
                                if batch.shading_plugin.is_some() {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                let mat_key = (
                                    batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch
                                        .metallic_roughness_id
                                        .map(|t| t.raw())
                                        .unwrap_or(u64::MAX),
                                    batch.emissive_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    resources.uv1_chunk_key(mesh.vertex_span.chunk),
                                );
                                let Some(inst_tex_bg) =
                                    resources.instanced_colour_bind_group(mat_key)
                                else {
                                    continue;
                                };
                                let no_discard = !clipping_active
                                    && !batch.has_alpha_mask
                                    && nodiscard_pipes.0.is_some()
                                    && nodiscard_pipes.1.is_some();
                                if cur_pipe != Some((batch.two_sided, no_discard)) {
                                    let key = PipelineKey {
                                        two_sided: batch.two_sided,
                                        no_discard_eligible: no_discard,
                                        ..PipelineKey::default()
                                    };
                                    render_pass.set_pipeline(select_opaque_solid(
                                        key,
                                        pipeline,
                                        pipeline_two_sided,
                                        nodiscard_pipes.0,
                                        nodiscard_pipes.1,
                                    ));
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

                        // Material-plugin opaque batches: one instanced call each,
                        // through the plugin's composed instanced pipeline plus its
                        // group-3 params bind. The loops above skip plugin batches
                        // (they sit outside GPU-cull / count-multi-draw run-forming),
                        // so this covers them whether or not culling drew the rest.
                        if opaque_batches
                            .iter()
                            .any(|(_, b)| b.shading_plugin.is_some())
                        {
                            bind_deform_group!(
                                render_pass,
                                resources,
                                &resources.deform.dummy_bind_group
                            );
                            // When GPU culling ran this frame, a plugin batch draws
                            // its culled instances from the cull-written indirect
                            // args through the plugin's `vs_main_cull` pipeline; the
                            // cull kernel wrote args + visibility for every batch,
                            // including plugin ones (they are only excluded from the
                            // count-multi-draw compaction, not the cull). Otherwise
                            // it draws every instance directly.
                            let plugin_indirect = (self.instancing.gpu_culling_enabled
                                && resources.cull.hdr_solid_pipeline.is_some())
                            .then(|| cull0.indirect_args_buf.as_ref())
                            .flatten();
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            for (batch_global_idx, batch) in &opaque_batches {
                                if batch.shading_plugin.is_none() {
                                    continue;
                                }
                                let Some((plug_pipes, mat_bg)) =
                                    resources.material_plugin_instanced_draw(batch.shading_plugin)
                                else {
                                    continue;
                                };
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                let mat_key = (
                                    batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch
                                        .metallic_roughness_id
                                        .map(|t| t.raw())
                                        .unwrap_or(u64::MAX),
                                    batch.emissive_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    resources.uv1_chunk_key(mesh.vertex_span.chunk),
                                );
                                let no_discard = !clipping_active && !batch.has_alpha_mask;
                                let key = PipelineKey {
                                    two_sided: batch.two_sided,
                                    no_discard_eligible: no_discard,
                                    ..PipelineKey::default()
                                };
                                // Geometry chunk binds once per change; both draw
                                // paths read the same slab chunk.
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
                                let culled = plugin_indirect
                                    .zip(plug_pipes.cull.as_ref())
                                    .and_then(|(indirect_buf, cull_set)| {
                                        resources
                                            .instanced_cull_colour_bind_group(cull0, mat_key)
                                            .map(|bg| (indirect_buf, cull_set, bg))
                                    });
                                if let Some((indirect_buf, cull_set, cull_bg)) = culled {
                                    render_pass.set_pipeline(cull_set.get(key));
                                    render_pass.set_bind_group(1, cull_bg, &[]);
                                    bind_material_group!(render_pass, mat_bg);
                                    render_pass.draw_indexed_indirect(
                                        indirect_buf,
                                        *batch_global_idx as u64 * 20,
                                    );
                                } else {
                                    let Some(inst_tex_bg) =
                                        resources.instanced_colour_bind_group(mat_key)
                                    else {
                                        continue;
                                    };
                                    render_pass.set_pipeline(plug_pipes.hdr_opaque.get(key));
                                    render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                    bind_material_group!(render_pass, mat_bg);
                                    let base_vertex =
                                        resources.geometry.base_vertex(mesh.vertex_span);
                                    let first_index =
                                        resources.geometry.first_index(mesh.index_span);
                                    render_pass.draw_indexed(
                                        first_index..first_index + mesh.index_count,
                                        base_vertex,
                                        batch.instance_offset
                                            ..batch.instance_offset + batch.instance_count,
                                    );
                                }
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
                    } else if let Some(hdr_opaque) = &resources.scene.hdr_opaque {
                        // Clip geometry disables the discard-free early-Z twin
                        // (the clip discards would be stripped). Computed here
                        // because this per-object branch is the `else` of the
                        // instanced path where the same check lives.
                        let clipping_active = frame
                            .effects
                            .clip
                            .objects
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
                            // depth-rejected before shading. Alpha-mask and
                            // submesh-material draws keep the discarding pipeline.
                            // Material-plugin items are just as eligible: their
                            // `hdr_opaque` set carries the same discard-free twin.
                            let key = PipelineKey {
                                two_sided: item.material.is_two_sided(),
                                no_discard_eligible: !clipping_active
                                    && !resources.force_po_discard
                                    && matches!(
                                        item.material.alpha_mode,
                                        crate::scene::material::AlphaMode::Opaque
                                    )
                                    && item.active_attribute.is_none()
                                    && item.submesh_materials.is_none(),
                                ..PipelineKey::default()
                            };
                            let pipeline = if let Some((pp, _)) = plug {
                                pp.hdr_opaque.get(key)
                            } else {
                                hdr_opaque.get(key)
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
                                    let range_key = PipelineKey::two_sided(mat.is_two_sided());
                                    let pl = if let Some((pp, _)) = plug_r {
                                        pp.hdr_opaque.get(range_key)
                                    } else {
                                        hdr_opaque.get(range_key)
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
                    if let (Some(hdr_opaque), Some(hdr_trans), Some(hdr_wf)) = (
                        &resources.scene.hdr_opaque,
                        &resources.scene.hdr_transparent,
                        &resources.scene.hdr_wireframe,
                    ) {
                        let hdr_solid_two_sided = hdr_opaque.get(PipelineKey::two_sided(true));
                        for (item_idx, item) in &opaque {
                            let solid_pl = hdr_opaque
                                .get(PipelineKey::two_sided(item.material.is_two_sided()));
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

            // The shared line substrate, mesh instances and sprites.
            //
            // depth. The post-pass targets the ssaa_* attachments and samples
            // ssaa_depth_only_view when SSAA is active, the hdr_* attachments
            // otherwise. Sprites are always skipped inline here.
            self.draw_line_and_instance_layers(&mut render_pass, camera_bg, true);

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
            self.dispatch_plugin_paint(&mut render_pass, frame, true);
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
            #[cfg(any(wgpu29, wgpu30))]
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

    fn hdr_ssaa_refraction(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let vp_idx = ctx.vp_idx;
        let ssaa_factor = ctx.ssaa_factor;
        // -----------------------------------------------------------------------
        // SSAA resolve pass: downsample supersampled scene -> hdr_texture.
        // Only runs when ssaa_factor > 1 and the resolve pipeline is available.
        //
        // Both halves matter. The colour resolve produces the image; the depth
        // blit after it writes the supersampled depth down into hdr_depth, which
        // every pass from here on attaches and depth-tests against. Without the
        // depth half that buffer is never written all frame under SSAA, and
        // decals, the sub-highlight, OIT, scatter, the foreground pass and the
        // plugin encode hook all fail their depth test and draw nothing.
        // -----------------------------------------------------------------------
        if ssaa_factor > 1 {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let (Some(pipeline), Some(bg)) = (
                &self.resources.post.ssaa_resolve_pipeline,
                &slot_hdr.ssaa_resolve_bind_group,
            ) {
                let mut resolve_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(any(wgpu29, wgpu30))]
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

            // Depth half: a fullscreen depth-only pass taking the nearest
            // sample of each block. See `ssaa_depth_resolve.wgsl` for why the
            // reduction has to be min rather than an arbitrary sub-sample:
            // the decal pass reconstructs its receiver normal from screen-space
            // derivatives of this buffer.
            if let (Some(blit_pipeline), Some(blit_bg)) = (
                &self.resources.post.ssaa_depth_resolve_pipeline,
                &slot_hdr.ssaa_depth_blit_bind_group,
            ) {
                let mut depth_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(any(wgpu29, wgpu30))]
                    multiview_mask: None,
                    label: Some("ssaa_depth_resolve_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.hdr_depth_view,
                        depth_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(1.0),
                            store: crate::gpu::StoreOp::Store,
                        }),
                        // 1, the value the scene pass clears stencil to when
                        // it owns this attachment. The decal exclude pass then
                        // stamps 0 on non-receivers as usual.
                        stencil_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(1),
                            store: crate::gpu::StoreOp::Store,
                        }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                depth_pass.set_pipeline(blit_pipeline);
                depth_pass.set_bind_group(0, blit_bg, &[]);
                depth_pass.draw(0..3, 0..1);
            }
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
                    #[cfg(any(wgpu29, wgpu30))]
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
                            // cause subsequent passes (tone_map, grid, overlays)
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
        let resources = &self.resources;
        let slot = &self.viewport_slots[vp_idx];
        let camera_bg = &slot.camera_bind_group;
        let slot_hdr = slot.hdr.as_ref().unwrap();

        // The HDR views, not the supersampled ones. This pass runs after the
        // SSAA resolve, which is encoded once per frame and never again, so a
        // draw into the supersampled colour here would land in a texture
        // nothing reads afterwards and vanish from the frame. It used to select
        // the ssaa_* views by copying the sprite passes, which make the same
        // choice correctly because it runs *before* the resolve.
        let colour_view = &slot_hdr.hdr_view;
        let depth_view = &slot_hdr.hdr_depth_view;
        let depth_only_view = &slot_hdr.hdr_depth_only_view;

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
            #[cfg(any(wgpu29, wgpu30))]
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
                    #[cfg(any(wgpu29, wgpu30))]
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
                            // Two-sided transparent batches draw through the
                            // cull-none twin; fall back to the culled pipeline if
                            // the twin is missing.
                            let pipeline_two_sided = self
                                .resources
                                .cull
                                .oit_two_sided_pipeline
                                .as_ref()
                                .unwrap_or(pipeline);
                            bind_deform_group!(
                                oit_pass,
                                self.resources,
                                &self.resources.deform.dummy_bind_group
                            );
                            // GPU-driven submission: iterate the precomputed
                            // transparent groups and issue one
                            // multi_draw_indexed_indirect_count per group over the
                            // shared compacted args + counts, skipping the CPU
                            // run-forming below. Active only under the same
                            // bindless + native-multi-draw gate as the opaque path.
                            let mut did_oit_groups = false;
                            if !self.instancing.oit_draw_groups.is_empty() {
                                if let (Some(compacted), Some(counts)) = (
                                    cull0.compacted_args_buf.as_ref(),
                                    cull0.draw_counts_buf.as_ref(),
                                ) {
                                    let mut cur_two_sided: Option<bool> = None;
                                    let mut cur_chunks: Option<(u32, u32)> = None;
                                    for group in self.instancing.oit_draw_groups.iter() {
                                        // Pipeline selector and geometry chunk come
                                        // from the group itself; the per-batch args
                                        // are drawn from the compacted buffer, so no
                                        // batch / mesh-store lookup is needed here.
                                        if cur_two_sided != Some(group.two_sided) {
                                            oit_pass.set_pipeline(if group.two_sided {
                                                pipeline_two_sided
                                            } else {
                                                pipeline
                                            });
                                            cur_two_sided = Some(group.two_sided);
                                        }
                                        let chunks = (group.vertex_chunk, group.index_chunk);
                                        if cur_chunks != Some(chunks) {
                                            // The colour bind group carries this
                                            // chunk's uv1 buffer, so rebind group 1
                                            // whenever the slab chunk changes.
                                            let Some(bg) = cull0
                                                .bindless_cull_bind_groups
                                                .get(&resources.uv1_chunk_key(chunks.0))
                                            else {
                                                continue;
                                            };
                                            oit_pass.set_bind_group(1, bg, &[]);
                                            oit_pass.set_vertex_buffer(
                                                0,
                                                resources.geometry.vertex_chunk_slice(chunks.0),
                                            );
                                            oit_pass.set_index_buffer(
                                                resources.geometry.index_chunk_slice(chunks.1),
                                                crate::gpu::IndexFormat::Uint32,
                                            );
                                            cur_chunks = Some(chunks);
                                        }
                                        oit_pass.multi_draw_indexed_indirect_count(
                                            compacted,
                                            group.arg_base as u64 * 20,
                                            counts,
                                            group.count_index as u64 * 4,
                                            group.size,
                                        );
                                        self.frame_main_draw_commands
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    did_oit_groups = true;
                                }
                            }
                            // Transparent batches pick the OIT pipeline by their
                            // two-sidedness, so a run collapses when the pipeline,
                            // bind group, and slab chunk hold across consecutive
                            // global indices (see the opaque path). Skipped when the
                            // group path above already submitted the draws.
                            let multi_draw = self.instancing.multi_draw_active();
                            let mut cur_bg: Option<*const crate::gpu::BindGroup> = None;
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            let mut cur_two_sided: Option<bool> = None;
                            let mut run_start: u64 = 0;
                            let mut run_len: u32 = 0;
                            // `take(0)` when the group path already drew, so the
                            // CPU run-forming is skipped without duplicating draws.
                            let cpu_run_limit = if did_oit_groups { 0 } else { usize::MAX };
                            for (batch_global_idx, batch) in self
                                .instancing
                                .batches
                                .iter()
                                .enumerate()
                                .take(cpu_run_limit)
                            {
                                // Plugin transparent batches draw in the plugin OIT
                                // sub-loop below; skipping breaks the run on the
                                // global-index gap like an opaque batch does.
                                if !batch.is_transparent || batch.shading_plugin.is_some() {
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
                                    batch
                                        .metallic_roughness_id
                                        .map(|t| t.raw())
                                        .unwrap_or(u64::MAX),
                                    batch.emissive_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    self.resources.uv1_chunk_key(mesh.vertex_span.chunk),
                                );
                                let Some(inst_tex_bg) = self
                                    .resources
                                    .instanced_cull_colour_bind_group(cull0, mat_key)
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
                                    && cur_two_sided == Some(batch.two_sided)
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
                                if cur_two_sided != Some(batch.two_sided) {
                                    oit_pass.set_pipeline(if batch.two_sided {
                                        pipeline_two_sided
                                    } else {
                                        pipeline
                                    });
                                    cur_two_sided = Some(batch.two_sided);
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
                        // Two-sided transparent batches draw through the cull-none
                        // twin; fall back to the culled pipeline if it is missing.
                        let pipeline_two_sided = self
                            .resources
                            .oit
                            .instanced_pipeline_two_sided
                            .as_ref()
                            .unwrap_or(pipeline);
                        bind_deform_group!(
                            oit_pass,
                            self.resources,
                            &self.resources.deform.dummy_bind_group
                        );
                        let mut cur_chunks: Option<(u32, u32)> = None;
                        let mut cur_two_sided: Option<bool> = None;
                        for batch in &self.instancing.batches {
                            // Plugin transparent batches draw in the plugin OIT
                            // sub-loop below.
                            if !batch.is_transparent || batch.shading_plugin.is_some() {
                                continue;
                            }
                            let Some(mesh) = self.resources.mesh_store.get(batch.mesh_id) else {
                                continue;
                            };
                            let mat_key = (
                                batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                batch
                                    .metallic_roughness_id
                                    .map(|t| t.raw())
                                    .unwrap_or(u64::MAX),
                                batch.emissive_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                self.resources.uv1_chunk_key(mesh.vertex_span.chunk),
                            );
                            let Some(inst_tex_bg) =
                                self.resources.instanced_colour_bind_group(mat_key)
                            else {
                                continue;
                            };
                            if cur_two_sided != Some(batch.two_sided) {
                                oit_pass.set_pipeline(if batch.two_sided {
                                    pipeline_two_sided
                                } else {
                                    pipeline
                                });
                                cur_two_sided = Some(batch.two_sided);
                            }
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

                    // Material-plugin transparent batches: one instanced OIT call
                    // each, through the plugin's composed instanced OIT pipeline
                    // plus its group-3 params bind. The OIT loops above skip plugin
                    // batches, so this covers them under both the culled and direct
                    // transparent paths.
                    if self
                        .instancing
                        .batches
                        .iter()
                        .any(|b| b.is_transparent && b.shading_plugin.is_some())
                    {
                        bind_deform_group!(oit_pass, resources, &resources.deform.dummy_bind_group);
                        // As on the opaque plugin path: a plugin batch draws its
                        // culled instances from the cull-written indirect args
                        // through the plugin's `vs_main_cull` OIT pipeline when
                        // culling ran, else every instance directly.
                        let plugin_indirect = (self.instancing.gpu_culling_enabled
                            && resources.cull.hdr_solid_pipeline.is_some())
                        .then(|| cull0.indirect_args_buf.as_ref())
                        .flatten();
                        let mut cur_chunks: Option<(u32, u32)> = None;
                        for (batch_global_idx, batch) in self.instancing.batches.iter().enumerate()
                        {
                            if !batch.is_transparent || batch.shading_plugin.is_none() {
                                continue;
                            }
                            let Some((plug_pipes, mat_bg)) =
                                resources.material_plugin_instanced_draw(batch.shading_plugin)
                            else {
                                continue;
                            };
                            let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                continue;
                            };
                            let mat_key = (
                                batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                batch
                                    .metallic_roughness_id
                                    .map(|t| t.raw())
                                    .unwrap_or(u64::MAX),
                                batch.emissive_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                resources.uv1_chunk_key(mesh.vertex_span.chunk),
                            );
                            let key = PipelineKey {
                                two_sided: batch.two_sided,
                                ..PipelineKey::default()
                            };
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
                            let culled = plugin_indirect
                                .zip(plug_pipes.oit_cull.as_ref())
                                .and_then(|(indirect_buf, cull_set)| {
                                    resources
                                        .instanced_cull_colour_bind_group(cull0, mat_key)
                                        .map(|bg| (indirect_buf, cull_set, bg))
                                });
                            if let Some((indirect_buf, cull_set, cull_bg)) = culled {
                                oit_pass.set_pipeline(cull_set.get(key));
                                oit_pass.set_bind_group(1, cull_bg, &[]);
                                bind_material_group!(oit_pass, mat_bg);
                                oit_pass.draw_indexed_indirect(
                                    indirect_buf,
                                    batch_global_idx as u64 * 20,
                                );
                            } else {
                                let Some(inst_tex_bg) =
                                    resources.instanced_colour_bind_group(mat_key)
                                else {
                                    continue;
                                };
                                oit_pass.set_pipeline(plug_pipes.oit.get(key));
                                oit_pass.set_bind_group(1, inst_tex_bg, &[]);
                                bind_material_group!(oit_pass, mat_bg);
                                let base_vertex = resources.geometry.base_vertex(mesh.vertex_span);
                                let first_index = resources.geometry.first_index(mesh.index_span);
                                oit_pass.draw_indexed(
                                    first_index..first_index + mesh.index_count,
                                    base_vertex,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                            }
                            self.frame_main_draw_commands
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }

                    // Transparent excluded items (two-sided, active attribute, matcap) are not
                    // in any instanced batch, so the instanced OIT loop above skips them.
                    // Render them here individually so they are not invisible at opacity < 1.
                    if let Some(ref oit_variants) = self.resources.oit.pipeline {
                        oit_pass.set_pipeline(oit_variants.get(PipelineKey::default()));
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
                                    let range_key = PipelineKey::two_sided(mat.is_two_sided());
                                    match self.resources.material_plugin_draw(mat.shading_plugin) {
                                        Some((pp, mat_bg)) => {
                                            oit_pass.set_pipeline(pp.oit.get(range_key));
                                            bind_material_group!(oit_pass, mat_bg);
                                        }
                                        // Two-sided per-range material draws back
                                        // faces through the cull-none OIT pipeline.
                                        None => {
                                            oit_pass.set_pipeline(oit_variants.get(range_key));
                                        }
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
                            let item_key = PipelineKey::two_sided(item.material.is_two_sided());
                            match self
                                .resources
                                .material_plugin_draw(item.material.shading_plugin)
                            {
                                Some((pp, mat_bg)) => {
                                    oit_pass.set_pipeline(pp.oit.get(item_key));
                                    bind_material_group!(oit_pass, mat_bg);
                                }
                                // Select the two-sided OIT pipeline for a
                                // non-`Cull` material so its back faces draw.
                                None => {
                                    oit_pass.set_pipeline(oit_variants.get(item_key));
                                }
                            }
                            oit_pass.set_bind_group(1, obj_bg, &[]);
                            oit_pass.draw_indexed(0..mesh.index_count, 0, obj_inst..obj_inst + 1);
                        }
                    }
                } else if let Some(ref oit_variants) = self.resources.oit.pipeline {
                    oit_pass.set_pipeline(oit_variants.get(PipelineKey::default()));
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
                                let range_key = PipelineKey::two_sided(mat.is_two_sided());
                                match self.resources.material_plugin_draw(mat.shading_plugin) {
                                    Some((pp, mat_bg)) => {
                                        oit_pass.set_pipeline(pp.oit.get(range_key));
                                        bind_material_group!(oit_pass, mat_bg);
                                    }
                                    // Two-sided per-range material draws back
                                    // faces through the cull-none OIT pipeline.
                                    None => {
                                        oit_pass.set_pipeline(oit_variants.get(range_key));
                                    }
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
                        let item_key = PipelineKey::two_sided(item.material.is_two_sided());
                        match self
                            .resources
                            .material_plugin_draw(item.material.shading_plugin)
                        {
                            Some((pp, mat_bg)) => {
                                oit_pass.set_pipeline(pp.oit.get(item_key));
                                bind_material_group!(oit_pass, mat_bg);
                            }
                            // Select the two-sided OIT pipeline for a non-`Cull`
                            // material so its back faces draw.
                            None => {
                                oit_pass.set_pipeline(oit_variants.get(item_key));
                            }
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
                        #[cfg(any(wgpu29, wgpu30))]
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
                        #[cfg(any(wgpu29, wgpu30))]
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
                        #[cfg(any(wgpu29, wgpu30))]
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
        if !slot.selection_outlines.outline_object_buffers.is_empty()
            || !slot.selection_outlines.polyline_outline_indices.is_empty()
            || slot.selection_outlines.plugin_outline_present
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
                        #[cfg(any(wgpu29, wgpu30))]
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
        let (Some(hdr_opaque), Some(hdr_trans), Some(hdr_wf)) = (
            &resources.scene.hdr_opaque,
            &resources.scene.hdr_transparent,
            &resources.scene.hdr_wireframe,
        ) else {
            return;
        };
        let hdr_solid_two_sided = hdr_opaque.get(PipelineKey::two_sided(true));

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
            // Per-camera layer cull, as in the scene pass: drop a foreground item
            // whose visibility mask shares no bit with this viewport's cull_mask.
            // Default masks (`!0`) keep it.
            if (item.settings.visibility_mask & frame.camera.cull_mask) == 0 {
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
            #[cfg(any(wgpu29, wgpu30))]
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
            let solid_pl = hdr_opaque.get(PipelineKey::two_sided(item.material.is_two_sided()));
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
        // Composite-input producers (SSAO, contact shadows, bloom, DoF,
        // exposure), in the fixed encode order. The throttle skips the
        // throttleable producers only; exposure always resolves.
        // -----------------------------------------------------------------------
        let inputs = crate::resources::ProducerFrameInputs {
            post: pp,
            proj: frame.camera.render_camera.projection,
            view: frame.camera.render_camera.view,
            near: frame.camera.render_camera.near,
            far: frame.camera.render_camera.far,
            first_light: frame.effects.lighting.lights.first(),
            foreground_active: self.foreground_active(frame),
            exposure: frame.effects.display.exposure,
        };
        let timing = crate::resources::ProducerTiming {
            query_set: self.ts_query_set.as_ref(),
            written_mask: &self.ts_written_mask,
        };
        for producer in self.resources.post_producers() {
            if producer.enabled(&inputs) && (!throttle_effects || !producer.throttleable()) {
                producer.encode(slot_hdr, encoder, &inputs, &timing);
            }
        }

        // External post-effect producers run after the built-ins, in
        // registration order, under the same throttle. Returned views are
        // collected and bound at their slots when the tone-map stage
        // rebuilds the composite bind group; the last producer for a slot
        // wins, and an external view over an enabled built-in logs once.
        if !self.post_effect_producers.is_empty() && !throttle_effects {
            let ci = ctx.composite_inputs;
            let ctx = post_effect_ctx(ctx.device, slot_hdr, frame, vp_idx);
            let mut collected: Vec<(crate::plugin_api::PostEffectSlot, crate::gpu::TextureView)> =
                std::mem::take(&mut self.frame_external_slot_views);
            for entry in &mut self.post_effect_producers {
                if !(entry.gpu_ready && entry.producer.enabled()) {
                    continue;
                }
                let slot = entry.producer.slot();
                let Some(view) = entry.producer.encode(encoder, &ctx) else {
                    continue;
                };
                let view = view.clone();
                let (builtin_on, slot_bit) = match slot {
                    crate::plugin_api::PostEffectSlot::Bloom => (ci.bloom, 1u8),
                    crate::plugin_api::PostEffectSlot::AmbientOcclusion => (ci.ssao, 2),
                    crate::plugin_api::PostEffectSlot::ContactShadow => (ci.contact_shadows, 4),
                    crate::plugin_api::PostEffectSlot::SurfaceLic => (ci.lic, 8),
                };
                if builtin_on && self.post_effect_slot_warned & slot_bit == 0 {
                    self.post_effect_slot_warned |= slot_bit;
                    tracing::debug!(
                        "post-effect producer '{}' overrides the enabled built-in {:?} slot; \
                         switch the built-in off when replacing it",
                        entry.producer.type_name(),
                        slot,
                    );
                }
                collected.push((slot, view));
            }
            self.frame_external_slot_views = collected;
        }
    }

    fn hdr_tonemap_resolve(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        let frame = ctx.frame;
        let pp = &frame.effects.post_process;
        // Bind this frame's external composite contributions: force the
        // matching enable lanes on and rebuild the tone-map bind group with
        // the producer views at their slots. The uniform rewrite is staged
        // before this encoder's submission, so it wins over the preamble's
        // write of the same buffer.
        if !self.frame_external_slot_views.is_empty() {
            let mut inputs = ctx.composite_inputs;
            let mut uniform = ctx.tm_uniform;
            for (slot, _) in &self.frame_external_slot_views {
                match slot {
                    crate::plugin_api::PostEffectSlot::Bloom => {
                        inputs.bloom = true;
                        uniform.bloom_enabled = 1;
                    }
                    crate::plugin_api::PostEffectSlot::AmbientOcclusion => {
                        inputs.ssao = true;
                        uniform.ssao_enabled = 1;
                    }
                    crate::plugin_api::PostEffectSlot::ContactShadow => {
                        inputs.contact_shadows = true;
                        uniform.contact_shadows_enabled = 1;
                    }
                    crate::plugin_api::PostEffectSlot::SurfaceLic => {
                        inputs.lic = true;
                        uniform.lic_enabled = 1;
                    }
                }
            }
            let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
            ctx.queue.write_buffer(
                &hdr.tone_map_uniform_buf,
                0,
                bytemuck::cast_slice(&[uniform]),
            );
            self.resources.rebuild_tone_map_bind_group(
                ctx.device,
                hdr,
                inputs,
                &self.frame_external_slot_views,
            );
        }
        let slot = &self.viewport_slots[vp_idx];
        let slot_hdr = slot.hdr.as_ref().unwrap();
        // -----------------------------------------------------------------------
        // Tone map pass: HDR + bloom + AO -> tone-mapped LDR.
        //
        // When render_scale < 1.0 the entire post-process chain runs at scene
        // resolution. The result lands in upscale_view (scene-res) and is then
        // upscale-blitted to output_view at native resolution.
        // -----------------------------------------------------------------------
        let use_hdr_upscale = slot_hdr.upscale_bind_group.is_some();
        // The post-composite stage chain: built-in stages (FXAA) and external
        // stages merged in ascending order-key order (stable: ties keep
        // built-ins first, then registration order). The composite renders
        // into the first stage's input, each stage into the next stage's
        // input, and the last into the frame's final target.
        enum ChainEntry<'a> {
            Builtin(&'a dyn crate::resources::PostStage),
            External(usize),
        }
        let mut chain: Vec<(i32, ChainEntry<'_>)> = Vec::new();
        for s in self.resources.post_stages() {
            if s.enabled(pp) {
                chain.push((
                    crate::plugin_api::post_effect::stage_order::ANTI_ALIASING,
                    ChainEntry::Builtin(s),
                ));
            }
        }
        for (i, entry) in self.post_effect_stages.iter().enumerate() {
            if entry.gpu_ready && entry.stage.enabled() {
                chain.push((entry.order, ChainEntry::External(i)));
            }
        }
        chain.sort_by_key(|(order, _)| *order);
        let final_target: crate::gpu::TextureView = if use_hdr_upscale {
            slot_hdr.upscale_view.as_ref().unwrap().clone()
        } else {
            output_view.clone()
        };
        if let Some(tone_map_pipeline) = &self.resources.post.tone_map_pipeline {
            let tone_target: crate::gpu::TextureView = match chain.first() {
                Some((_, ChainEntry::Builtin(s))) => s.input_view(slot_hdr).clone(),
                Some((_, ChainEntry::External(j))) => {
                    self.post_effect_stages[*j].stage.input_view(vp_idx).clone()
                }
                None => final_target.clone(),
            };
            let tone_target = &tone_target;
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
                #[cfg(any(wgpu29, wgpu30))]
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
        // Post-composite stages, chained toward the final target.
        // -----------------------------------------------------------------------
        {
            let timing = crate::resources::ProducerTiming {
                query_set: self.ts_query_set.as_ref(),
                written_mask: &self.ts_written_mask,
            };
            for i in 0..chain.len() {
                let target: crate::gpu::TextureView = match chain.get(i + 1) {
                    Some((_, ChainEntry::Builtin(s))) => s.input_view(slot_hdr).clone(),
                    Some((_, ChainEntry::External(j))) => {
                        self.post_effect_stages[*j].stage.input_view(vp_idx).clone()
                    }
                    None => final_target.clone(),
                };
                match &chain[i].1 {
                    ChainEntry::Builtin(s) => s.encode(slot_hdr, encoder, &target, &timing),
                    ChainEntry::External(j) => {
                        let stage_ctx = post_effect_ctx(ctx.device, slot_hdr, frame, vp_idx);
                        self.post_effect_stages[*j]
                            .stage
                            .encode(encoder, &target, &stage_ctx);
                    }
                }
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
                            #[cfg(any(wgpu29, wgpu30))]
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
                            #[cfg(any(wgpu29, wgpu30))]
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
                    #[cfg(any(wgpu29, wgpu30))]
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
                #[cfg(any(wgpu29, wgpu30))]
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
                #[cfg(any(wgpu29, wgpu30))]
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
            gp_pass.set_pipeline(&self.resources.ground.pipeline);
            gp_pass.set_bind_group(0, &self.resources.ground.bind_group, &[]);
            gp_pass.draw(0..3, 0..1);
        }

        // Editor overlay pass (HDR path): draw viewport/editor overlays on the
        // final output after tone mapping / FXAA, reusing the scene depth
        // buffer so depth-tested helpers still behave correctly.
        {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let has_editor_overlays =
                !slot.constraint_line_buffers.is_empty() || !slot.xray_object_buffers.is_empty();
            if has_editor_overlays {
                let camera_bg = &slot.camera_bind_group;
                let mut overlay_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(any(wgpu29, wgpu30))]
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

        // The axes orientation indicator draws as screen-space overlay shapes in
        // the shared overlay pass (see `axes_overlay_items`), not here.
    }

    fn hdr_final_overlay(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let device = ctx.device;
        let queue = ctx.queue;
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        let w = ctx.w;
        let h = ctx.h;
        // Overlay shapes and the merged text batch (HDR path): drawn last.
        let has_overlay = self.overlay_shape_gpu_data.is_some() || self.label_gpu_data.is_some();

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
            let overlay_ts_writes = self.ts_writes_for(crate::renderer::GPU_TS_OVERLAY, true, true);
            let mut overlay_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
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
                timestamp_writes: overlay_ts_writes,
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
