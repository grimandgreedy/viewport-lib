//! The HDR render path: the full post-processing pipeline (scene, sprites,
//! decals, transparency, scatter, flow, bloom, tone mapping, overlays). Builds
//! its own command encoder and returns the finished buffer.

use super::pass_table::{HdrPass, MAX_PASSES};
use super::*;
use crate::resources::PostProducer;
use crate::resources::scene_pipelines::MeshColourFamily;

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

/// One entry of the post-composite stage chain.
enum ChainEntry<'a> {
    Builtin(&'a dyn crate::resources::PostStage),
    External(usize),
}

/// The post-composite stage chain: built-in stages (FXAA) and external stages
/// merged in ascending order-key order (stable: ties keep built-ins first,
/// then registration order). The composite renders into the first stage's
/// input, each stage into the next stage's input, and the last into the
/// frame's final target.
fn post_stage_chain<'a>(
    resources: &'a DeviceResources,
    stages: &[crate::renderer::RegisteredPostEffectStage],
    pp: &crate::PostProcessSettings,
) -> Vec<(i32, ChainEntry<'a>)> {
    let mut chain: Vec<(i32, ChainEntry<'a>)> = Vec::new();
    for s in resources.post_stages() {
        if s.enabled(pp) {
            chain.push((
                crate::plugin_api::post_effect::stage_order::ANTI_ALIASING,
                ChainEntry::Builtin(s),
            ));
        }
    }
    for (i, entry) in stages.iter().enumerate() {
        if entry.gpu_ready && entry.stage.enabled() {
            chain.push((entry.order, ChainEntry::External(i)));
        }
    }
    chain.sort_by_key(|(order, _)| *order);
    chain
}

/// Build the per-frame, per-viewport context handed to external post-effect
/// producers.
fn post_effect_ctx<'a>(
    device: &'a crate::gpu::Device,
    slot_hdr: &'a crate::resources::ViewportHdrState,
    frame: &'a FrameData,
    viewport_index: usize,
    origin: &crate::renderer::origin::OriginState,
) -> crate::plugin_api::PostEffectContext<'a> {
    crate::plugin_api::PostEffectContext {
        device,
        viewport_index,
        scene_size: slot_hdr.scene_size,
        output_size: slot_hdr.output_size,
        proj: frame.camera.render_camera.projection(),
        view: origin.camera(&frame.camera.render_camera).view(),
        near: frame.camera.render_camera.depth_range().0,
        far: frame.camera.render_camera.depth_range().1,
        scene_colour: &slot_hdr.hdr_view,
        scene_depth: &slot_hdr.hdr_depth_only_view,
        post: &frame.effects.post_process,
    }
}

/// Encode one non-instanced mesh item's draws: bind the object (group 1) and
/// material plugin (group 2) groups, pick the pipeline, and draw. `obj_override` is the item's per-item bind group and the record
/// buffer it binds, when one was prepared; `None` falls back to the mesh's
/// shared object bind group and template record. `hdr`
/// selects the material-plugin pipeline family and `family` the built-in
/// one. A draw whose pipeline is still compiling is skipped. Shared by the
/// HDR scene pass and the HDR/LDR foreground passes; group 0 must already be
/// bound by the caller.
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
    render_pass: &mut crate::gpu::RenderPass<'_>,
    item: &SceneRenderItem,
    obj_override: Option<(&crate::gpu::BindGroup, &crate::gpu::Buffer)>,
    // Object-data element index the whole-mesh draw selects with
    // @builtin(instance_index). Must be 0 when `obj_override` is None (the
    // mesh's single-element fallback buffer).
    obj_index: u32,
    wireframe_mode: bool,
    hdr: bool,
    family: &dyn MeshColourFamily,
    submesh_bgs: Option<&[Option<crate::gpu::BindGroup>]>,
    // Object-data indices parallel to `submesh_bgs`; a range with its own bind
    // group selects its element here.
    submesh_indices: Option<&[u32]>,
    submesh_transparent: Option<bool>,
) {
    let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
        return;
    };
    let (obj_bg, records) =
        obj_override.unwrap_or((&mesh.object_bind_group, &mesh.object_uniform_buf));
    render_pass.set_bind_group(1, obj_bg, &[]);
    render_pass.set_vertex_buffer(1, records.slice(..));
    // A prepared record carries the mesh's base vertex and draws from the
    // whole slab chunk; the template record carries 0 and draws the mesh's
    // own slice. Either way the record's `vertex_base` is the base drawn with.
    let (vertices, indices, base_vertex, first_index) = if obj_override.is_some() {
        (
            resources
                .geometry
                .vertex_chunk_slice(mesh.vertex_span.chunk),
            resources.geometry.index_chunk_slice(mesh.index_span.chunk),
            resources.geometry.base_vertex(mesh.vertex_span),
            resources.geometry.first_index(mesh.index_span),
        )
    } else {
        (
            resources.geometry.vertex_slice(mesh.vertex_span),
            resources.geometry.index_slice(mesh.index_span),
            0,
            0,
        )
    };
    let material = resources.material_or_default(item.material);
    let plug =
        resources.material_plugin_draw(material.shading_plugin, &material.plugin_textures, false);
    if let Some((_, mat_bg)) = plug {
        bind_material_group!(render_pass, mat_bg);
    }

    let is_face_attr = item.active_attribute.as_ref().map_or(false, |a| {
        matches!(
            a.kind,
            crate::resources::AttributeKind::Face
                | crate::resources::AttributeKind::FaceColour
                | crate::resources::AttributeKind::Halfedge
                | crate::resources::AttributeKind::Corner
        )
    });
    let item_key = PipelineKey::two_sided(material.is_two_sided());
    if wireframe_mode || item.settings.wireframe {
        if let Some(edge_buf) = &mesh.edge_index_buffer {
            let Some(wf_pl) = family.wireframe() else {
                return;
            };
            render_pass.set_pipeline(wf_pl);
            render_pass.set_vertex_buffer(0, vertices);
            render_pass.set_index_buffer(edge_buf.slice(..), crate::gpu::IndexFormat::Uint32);
            render_pass.draw_indexed(
                0..mesh.edge_index_count,
                base_vertex,
                obj_index..obj_index + 1,
            );
        }
    } else if is_face_attr {
        if let Some(ref fvb) = mesh.face_vertex_buffer {
            let pl = if let Some((pp, _)) = plug {
                if item.settings.opacity < 1.0 {
                    pp.transparent(hdr, crate::renderer::pipeline_key::FeatureSet::UNIVERSAL)
                        .pipeline
                } else {
                    pp.opaque(
                        hdr,
                        item_key,
                        crate::renderer::pipeline_key::FeatureSet::UNIVERSAL,
                    )
                    .pipeline
                }
            } else if item.settings.opacity < 1.0 {
                family.transparent()
            } else {
                family.opaque(item_key)
            };
            // A pipeline still compiling skips the draw this frame.
            let Some(pl) = pl else {
                return;
            };
            render_pass.set_pipeline(pl);
            render_pass.set_vertex_buffer(0, fvb.slice(..));
            render_pass.draw(0..mesh.index_count, obj_index..obj_index + 1);
        }
    } else {
        let ranges =
            crate::renderer::prepare::active_submesh_materials(item, mesh).zip(submesh_bgs);
        if let Some((mats, bgs)) = ranges {
            render_pass.set_vertex_buffer(0, vertices);
            render_pass.set_index_buffer(indices, crate::gpu::IndexFormat::Uint32);
            for (r, (mat, range)) in mats.iter().zip(&mesh.submeshes).enumerate() {
                let mat = resources.material_or_default(*mat);
                let is_trans = item.settings.opacity < 1.0 || mat.is_blend();
                if let Some(want) = submesh_transparent {
                    if is_trans != want {
                        continue;
                    }
                }
                let plug_r =
                    resources.material_plugin_draw(mat.shading_plugin, &mat.plugin_textures, false);
                let range_key = PipelineKey::two_sided(mat.is_two_sided());
                // A range draws two-sided when its own material or the item's
                // says so.
                let builtin_key = PipelineKey::two_sided(item_key.two_sided || range_key.two_sided);
                let pl = if let Some((pp, _)) = plug_r {
                    if is_trans {
                        pp.transparent(hdr, crate::renderer::pipeline_key::FeatureSet::UNIVERSAL)
                            .pipeline
                    } else {
                        pp.opaque(
                            hdr,
                            range_key,
                            crate::renderer::pipeline_key::FeatureSet::UNIVERSAL,
                        )
                        .pipeline
                    }
                } else if is_trans {
                    family.transparent()
                } else {
                    family.opaque(builtin_key)
                };
                let Some(pl) = pl else {
                    continue;
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
                let first = first_index + range.first_index;
                render_pass.draw_indexed(
                    first..first + range.index_count,
                    base_vertex,
                    inst..inst + 1,
                );
            }
        } else {
            let pl = if let Some((pp, _)) = plug {
                if item.settings.opacity < 1.0 {
                    pp.transparent(hdr, crate::renderer::pipeline_key::FeatureSet::UNIVERSAL)
                        .pipeline
                } else {
                    pp.opaque(
                        hdr,
                        item_key,
                        crate::renderer::pipeline_key::FeatureSet::UNIVERSAL,
                    )
                    .pipeline
                }
            } else if item.settings.opacity < 1.0 {
                family.transparent()
            } else {
                family.opaque(item_key)
            };
            let Some(pl) = pl else {
                return;
            };
            render_pass.set_pipeline(pl);
            render_pass.set_vertex_buffer(0, vertices);
            render_pass.set_index_buffer(indices, crate::gpu::IndexFormat::Uint32);
            render_pass.draw_indexed(
                first_index..first_index + mesh.index_count,
                base_vertex,
                obj_index..obj_index + 1,
            );
        }
    }
    if item.show_normals {
        if let (Some(nl_buf), Some(wf_pl)) = (&mesh.normal_line_buffer, family.wireframe()) {
            if mesh.normal_line_count > 0 {
                render_pass.set_pipeline(wf_pl);
                render_pass.set_bind_group(1, &mesh.normal_bind_group, &[]);
                render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                render_pass.set_vertex_buffer(1, mesh.normal_uniform_buf.slice(..));
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
        // HDR path. `bg_colour` is premultiplied linear RGBA; it reaches the
        // tone-map uniform below and is composited under transparent content
        // there, in display space, exactly once.
        let pp = &frame.effects.post_process;

        // Clear the linear HDR scene texture to nothing, not to the background.
        // The buffer is a premultiplied accumulation whose alpha is scene
        // coverage, so the background cannot live in it: clearing to the
        // background put `bg * (1 - coverage)` into every partially covered
        // pixel, which the composite then added a second time, measuring at
        // exactly twice what one application leaves. The background is the
        // composite's job alone.
        let hdr_clear_rgb = [0.0f32; 3];

        // Which effect inputs feed the tone-map composite this frame. Built
        // once here so the uniform's enable lanes below and the bind group's
        // view selection share one source and cannot disagree.
        // The grade LUT is validated against the texture store up front: the
        // renderer needs an owned texture to read the LUT's height, so ids
        // registered as external views are ignored.
        // The custom operator's view LUT is validated the same way.
        let owned_lut_size = |id: crate::resources::TextureId| {
            self.resources
                .content
                .textures
                .get(id)
                .and_then(|t| t.texture.as_ref())
                .map(|t| t.height() as f32)
        };
        let grade_lut = pp.grade_lut.filter(|id| owned_lut_size(*id).is_some());
        let grade_lut_size = grade_lut.and_then(owned_lut_size).unwrap_or(0.0);
        let view_lut = match frame.effects.display.operator {
            crate::renderer::ToneMapping::Custom { lut } => {
                Some(lut).filter(|id| owned_lut_size(*id).is_some())
            }
            _ => None,
        };
        let view_lut_size = view_lut.and_then(owned_lut_size).unwrap_or(0.0);
        let mut composite_inputs = self.viewport_slots[vp_idx]
            .pass_table
            .composite_inputs(grade_lut, view_lut);

        // Build what this frame binds and nothing else. Each group is built by
        // the first frame that asks for it, on the same condition its pass
        // tests below, so an effect that stays off never compiles.
        {
            let format = self.resources.target_format;
            let res = &mut self.resources;
            res.ensure_tone_map_pipeline(device, format);
            // The OIT resolve is one small pipeline, and an item-type plugin
            // can draw into the OIT pass on any frame. Built up front so the
            // first frame a plugin item appears compiles nothing of ours.
            res.ensure_oit_composite_pipeline(device);
            if pp.bloom.enabled {
                res.ensure_bloom_pipelines(device);
            }
            if pp.ssao {
                res.ensure_ssao_pipelines(device);
            }
            if pp.contact_shadows.enabled {
                res.ensure_contact_shadow_pipeline(device);
            }
            if pp.dof.enabled {
                res.ensure_dof_pipeline(device);
            }
            if pp.fxaa {
                res.ensure_fxaa_pipeline(device, format);
            }
            if composite_inputs.foreground {
                res.ensure_foreground_stamp_pipeline(device);
            }
            if ssaa_factor > 1 {
                res.ensure_ssaa_resolve_pipelines(device);
            }
            // Manual and physical-camera exposure write the state buffer
            // directly; only automatic exposure dispatches the metering passes.
            if frame.effects.display.exposure.manual_multiplier().is_none() {
                res.scene_tier
                    .exposure
                    .ensure_pipelines(device, &res.pipeline_compiler);
            }
            let slot = &self.viewport_slots[vp_idx];
            if slot
                .hdr
                .as_ref()
                .is_some_and(|hdr| hdr.depth_blit_bind_group.is_some())
            {
                res.ensure_depth_blit_pipeline(device);
            }
            // The mesh family, and the cap fill that shares its group. The
            // foreground pass sets up with these pipelines before it draws
            // anything, so a foreground drawn only by a plugin needs them too.
            if !scene_items.is_empty()
                || !frame.scene.volume_meshes.is_empty()
                || composite_inputs.foreground
                || !slot.cap_buffers.is_empty()
                || !self.mesh_uniforms.tvm_wireframe_draws.is_empty()
            {
                res.ensure_hdr_mesh_pipelines(device);
            }
            // Batches and explicit mesh instances draw with the HDR mesh
            // family and its culled twin. Prepare has normally built them;
            // this covers a frame prepared for a direct paint and then
            // rendered here.
            if (self.instancing.use_instancing && !self.instancing.batches.is_empty())
                || !self.mesh_instance_gpu_data.is_empty()
            {
                res.ensure_hdr_mesh_pipelines(device);
                res.ensure_hdr_cull_pipelines(device);
            }
        }
        // An effect whose pipelines are still compiling is off for the frame,
        // so the composite does not read a target nothing wrote. Asking also
        // starts the compile; under `Blocking` the block above built them.
        {
            let post = &self.resources.post;
            composite_inputs.bloom &= post.bloom.ready();
            composite_inputs.ssao &= post.ssao.ready();
            composite_inputs.contact_shadows &= post.contact_shadow.ready();
            composite_inputs.dof &= post.dof.ready();
        }

        // Upload tone map uniform into the per-viewport buffer.
        // A custom operator whose LUT is missing applies no operator.
        let mode = match frame.effects.display.operator {
            crate::renderer::ToneMapping::Custom { .. } if composite_inputs.view_lut.is_none() => {
                u32::MAX
            }
            op => op.composite_index(),
        };
        let (contrast, input_scale, output_scale) = pp.generic_tone_curve.scales();
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
            near_plane: frame.camera.render_camera.depth_range().0,
            far_plane: frame.camera.render_camera.depth_range().1,
            orthographic: u32::from(frame.camera.render_camera.is_orthographic()),
            projection_far: {
                let rc = &frame.camera.render_camera;
                let (near, far) = rc.depth_range();
                crate::plugin_api::depth::clip_lane(rc.projection(), near, far)[1]
            },
            foreground_enabled: composite_inputs.foreground as u32,
            vignette_amount: if pp.vignette.enabled {
                pp.vignette.amount.clamp(0.0, 1.0)
            } else {
                0.0
            },
            vignette_radius: pp.vignette.radius,
            vignette_softness: pp.vignette.softness,
            grade_mode: match (composite_inputs.grade_lut, pp.grading_space) {
                (None, _) => 0,
                (Some(_), crate::renderer::GradingSpace::Display) => 1,
                (Some(_), _) => 2,
            },
            grade_lut_size,
            working_space: match pp.working_space {
                crate::renderer::WorkingSpace::Rec2020 => 1,
                _ => 0,
            },
            agx_look: match pp.agx_look {
                crate::renderer::AgxLook::Punchy => 1,
                crate::renderer::AgxLook::Golden => 2,
                _ => 0,
            },
            generic_curve: [contrast, input_scale, output_scale, view_lut_size],
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
                proj: frame.camera.render_camera.projection(),
                view: self.origin.camera(&frame.camera.render_camera).view(),
                near: frame.camera.render_camera.depth_range().0,
                far: frame.camera.render_camera.depth_range().1,
                first_light: frame.effects.lighting.lights.first(),
                foreground_active: composite_inputs.foreground,
                exposure: frame.effects.display.exposure,
                presented: self.render_advances_state(),
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
            let ctx = post_effect_ctx(device, hdr, frame, vp_idx, &self.origin);
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
            // A superset of the OIT pass's own test, so the pass never finds its
            // targets or pipelines missing.
            let mesh_transparency = (self.instancing.use_instancing
                && self.instancing.batches.iter().any(|b| b.is_transparent))
                || scene_items.iter().any(|i| {
                    !i.settings.hidden
                        && crate::renderer::prepare::has_transparent_draws(i, &self.resources)
                })
                || frame
                    .scene
                    .volume_meshes
                    .iter()
                    .any(|i| !i.settings.hidden && i.transparency.is_some());
            // Item-type plugins may draw into the OIT pass through
            // `paint_transparent`.
            let needs_oit = mesh_transparency || self.any_plugin_items_submitted(frame);
            if needs_oit {
                if mesh_transparency {
                    self.resources.ensure_oit_mesh_pipelines(device);
                }
                if self.instancing.use_instancing
                    && self.instancing.batches.iter().any(|b| b.is_transparent)
                {
                    self.resources.ensure_oit_mesh_pipelines(device);
                    self.resources.ensure_oit_cull_pipelines(device);
                }
                let slot = &mut self.viewport_slots[vp_idx];
                slot.target_idle.use_oit();
                let hdr = slot.hdr.as_mut().unwrap();
                let [sw, sh] = hdr.scene_size;
                self.resources.ensure_viewport_oit(device, hdr, sw, sh);
            }
        }

        self.ensure_transmission_target(device, vp_idx);

        // -----------------------------------------------------------------------
        // Build the command encoder.
        // -----------------------------------------------------------------------
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("hdr_encoder"),
        });

        // An automatic exposure lives on the GPU: copy the one the last frame
        // resolved into this viewport's camera copies ahead of every draw, so
        // the frame shades with it and the meter can divide it back out. The
        // fixed modes wrote theirs at prepare.
        if frame.effects.display.exposure.manual_multiplier().is_none() {
            let slot = &self.viewport_slots[vp_idx];
            if let Some(hdr) = slot.hdr.as_ref() {
                for camera in [
                    &slot.camera_buf,
                    &slot.foreground_camera_buf,
                    &self.resources.scene_tier.binds.camera_uniform_buf,
                ] {
                    encoder.copy_buffer_to_buffer(
                        &hdr.exposure_state_buf,
                        0,
                        camera,
                        crate::resources::CAMERA_EXPOSURE_OFFSET,
                        4,
                    );
                }
            }
        }

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

        // The passes the table runs, in declaration order. Copied out so the
        // passes can borrow the renderer mutably.
        let mut passes = [HdrPass::Scene; MAX_PASSES];
        let mut count = 0;
        for pass in self.viewport_slots[vp_idx].pass_table.live_passes() {
            passes[count] = pass;
            count += 1;
        }
        for pass in &passes[..count] {
            match *pass {
                HdrPass::Scene => self.hdr_scene_pass(&ctx, &mut encoder),
                HdrPass::HizStore => self.hdr_store_hiz_depth(&ctx, &mut encoder),
                HdrPass::SsaaResolve => self.hdr_ssaa_refraction(&ctx, &mut encoder),
                // Stamp the surface mask before anything reads it. After the
                // resolve, so the stencil written is the one the passes below
                // attach.
                HdrPass::SurfaceMask => self.hdr_surface_mask(&ctx, &mut encoder),
                // Item-type plugins that composite onto the finished opaque
                // surfaces: decals stamp here, underneath the selection
                // affordances and the depth-read transparency that follow.
                HdrPass::ScopeOnOpaqueSurfaces => self.encode_scope(
                    &ctx,
                    &mut encoder,
                    crate::plugin_api::EncoderScope::OnOpaqueSurfaces,
                ),
                HdrPass::SubHighlight => self.hdr_sub_highlight(&ctx, &mut encoder),
                HdrPass::DepthRead => self.hdr_depth_read_pass(&ctx, &mut encoder),
                HdrPass::OpaqueCopy => self.hdr_opaque_copy(device, &mut encoder, vp_idx),
                HdrPass::Transmission => self.hdr_transmission(&mut encoder, vp_idx, frame),
                // Item-type plugins that own passes rather than draws get the
                // encoder here, with the opaque image and final opaque depth
                // in hand.
                HdrPass::ScopeAfterOpaque => self.encode_scope(
                    &ctx,
                    &mut encoder,
                    crate::plugin_api::EncoderScope::AfterOpaque,
                ),
                HdrPass::Oit => self.hdr_oit(&ctx, &mut encoder),
                // The second scope: opaque plus resolved transparency, which is
                // what a volumetric effect composites over. The scatter-volume
                // plugin encodes its ray-march, temporal blend and composite
                // here.
                HdrPass::ScopeAfterTransparent => self.encode_scope(
                    &ctx,
                    &mut encoder,
                    crate::plugin_api::EncoderScope::AfterTransparent,
                ),
                HdrPass::OutlineComposite => self.hdr_outline_composite(&ctx, &mut encoder),
                HdrPass::Foreground => self.hdr_foreground(&ctx, &mut encoder),
                HdrPass::Producer(i) => self.hdr_builtin_producer(&ctx, &mut encoder, i as usize),
                HdrPass::ExternalProducers => self.hdr_external_producers(&ctx, &mut encoder),
                HdrPass::ToneMap => self.hdr_tone_map(&ctx, &mut encoder),
                HdrPass::PostStages => self.hdr_post_stages(&ctx, &mut encoder),
                HdrPass::Upscale => self.hdr_upscale(&ctx, &mut encoder),
                HdrPass::OutputDepthBlit => self.hdr_output_depth_blit(&ctx, &mut encoder),
                HdrPass::ForegroundStamp => self.hdr_foreground_stamp(&ctx, &mut encoder),
                HdrPass::SceneOverlays => self.hdr_scene_overlays(&ctx, &mut encoder),
                HdrPass::Overlay => self.hdr_final_overlay(&ctx, &mut encoder),
            }
        }
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

    fn hdr_surface_mask(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        if self.surface_mask.active() {
            self.mark_pass_ran(ctx.vp_idx, HdrPass::SurfaceMask);
        }
        self.encode_surface_mask(encoder, ctx.frame, ctx.vp_idx);
    }

    fn hdr_scene_pass(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let _span = tracing::debug_span!(target: "viewport_lib::frame", "scene_pass").entered();
        self.mark_pass_ran(ctx.vp_idx, HdrPass::Scene);
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
            let scene_fraction_view = match &slot_hdr.ssaa_indirect_fraction_view {
                Some(view) if use_ssaa => view,
                _ => &slot_hdr.indirect_fraction_view,
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
            // The indirect fraction starts at 1, wholly indirect, where nothing
            // draws, and is kept only for a frame whose composite reads it.
            let fraction_store = if slot.pass_table.reads_indirect_fraction() {
                crate::gpu::StoreOp::Store
            } else {
                crate::gpu::StoreOp::Discard
            };
            let mut render_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("hdr_scene_pass"),
                color_attachments: &[
                    Some(crate::gpu::RenderPassColorAttachment {
                        view: scene_colour_view,
                        resolve_target: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(clear_wgpu),
                            store: crate::gpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(crate::gpu::RenderPassColorAttachment {
                        view: scene_fraction_view,
                        resolve_target: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(crate::gpu::Color::WHITE),
                            store: fraction_store,
                        },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: scene_depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::plugin_api::depth::DEPTH_CLEAR),
                        store: crate::gpu::StoreOp::Store,
                    }),
                    // Every pixel starts as a member of every layer; the
                    // surface mask pass overwrites the ones that are not.
                    stencil_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::plugin_api::SURFACE_MASK_LAYERS),
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
            let skybox_bg = self.viewport_slots[vp_idx]
                .skybox
                .as_ref()
                .filter(|_| self.viewport_slots[vp_idx].draw_skybox)
                .map(|b| &b.bind_group);

            let batches = &self.instancing.batches;

            if !scene_items.is_empty() && !batches.is_empty() {
                let clipping_active = DeviceResources::clipping_active(frame);
                // Under GPU culling the culled batches draw from this
                // viewport's cull-written indirect args and the rest directly.
                let use_indirect = self.instancing.gpu_culling_enabled
                    && resources.cull.hdr.is_some()
                    && cull0.indirect_args_buf.is_some();
                // A cached bundle holds the opaque batches this pass draws
                // directly; under GPU culling, only those the cull does not
                // draw.
                let bundle_hit = self.per_object_bundle.as_ref().filter(|pb| {
                    pb.hdr && pb.direct_only == use_indirect && pb.camera_bg == *camera_bg
                });
                if use_indirect {
                    let multi_draw = self.instancing.multi_draw_active();
                    // GPU-driven submission: when the compaction pass formed
                    // draw groups (bindless + native multi-draw-count), one
                    // multi_draw_indexed_indirect_count per group; the batches
                    // left out of the groups take the run path below.
                    let mut grouped = false;
                    if let (false, Some(compacted), Some(counts), Some(words)) = (
                        self.instancing.draw_groups.is_empty(),
                        cull0.compacted_args_buf.as_ref(),
                        cull0.draw_counts_buf.as_ref(),
                        cull0.words_slice(),
                    ) {
                        grouped = true;
                        render_pass.set_vertex_buffer(1, words);
                        let mut cur_pipe: Option<*const crate::gpu::RenderPipeline> = None;
                        let mut cur_bg: Option<*const crate::gpu::BindGroup> = None;
                        let mut cur_chunks: Option<(u32, u32)> = None;
                        for group in &self.instancing.draw_groups {
                            let key = PipelineKey {
                                two_sided: group.two_sided,
                                no_discard_eligible: group.no_discard,
                                ..PipelineKey::default()
                            };
                            // Still compiling: the group waits a frame.
                            let Some(pl) = resources
                                .cull
                                .hdr_opaque_features(key, group.features, group.bindless)
                                .pipeline
                            else {
                                continue;
                            };
                            let Some(bg) = cull0.object_cull_bind_groups.get(&group.bind_key)
                            else {
                                continue;
                            };
                            if cur_pipe != Some(pl as *const _) {
                                render_pass.set_pipeline(pl);
                                cur_pipe = Some(pl as *const _);
                            }
                            if cur_bg != Some(bg as *const _) {
                                render_pass.set_bind_group(1, bg, &[]);
                                cur_bg = Some(bg as *const _);
                            }
                            let chunks = (group.vertex_chunk, group.index_chunk);
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
                    }
                    let (commands, binds) = crate::renderer::types::draw_culled_batches(
                        resources,
                        cull0,
                        &mut render_pass,
                        batches,
                        crate::renderer::types::CulledPass::Opaque { clipping_active },
                        multi_draw,
                        |b| grouped && b.draws_grouped(),
                    );
                    self.frame_main_draw_commands
                        .fetch_add(commands, std::sync::atomic::Ordering::Relaxed);
                    self.frame_main_buffer_binds
                        .fetch_add(binds, std::sync::atomic::Ordering::Relaxed);
                }
                if let Some(pb) = bundle_hit {
                    render_pass.execute_bundles(std::iter::once(&pb.bundle));
                    // Bundle execution resets all render-pass state; restore
                    // the camera bind group for the draws below.
                    render_pass.set_bind_group(0, camera_bg, &[]);
                }

                // Every opaque batch neither path above drew, then the normal
                // lines, which overlay everything.
                if resources.scene.hdr.is_some() {
                    let hdr_family = resources.scene.hdr_family();
                    let hdr_bindless_family = resources.scene.hdr_bindless_family();
                    let batch_pass = crate::renderer::types::BatchPass::Colour {
                        family: &hdr_family,
                        bindless_family: &hdr_bindless_family,
                        hdr: true,
                        clipping_active,
                    };
                    let mut batch_bound = crate::renderer::types::BatchBound::default();
                    let batch_bgs = &self.mesh_uniforms.material_bind_groups;
                    for batch in batches {
                        if batch.is_transparent
                            || batch.is_transmissive
                            || matches!(batch.draw, crate::renderer::types::DrawKind::NormalLines)
                            || (use_indirect && batch.draws_culled())
                            || bundle_hit.is_some()
                        {
                            continue;
                        }
                        if crate::renderer::types::draw_batch(
                            resources,
                            batch_bgs,
                            &mut render_pass,
                            batch,
                            &batch_pass,
                            &mut batch_bound,
                        ) {
                            self.frame_main_draw_commands
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    for batch in batches {
                        if !matches!(batch.draw, crate::renderer::types::DrawKind::NormalLines) {
                            continue;
                        }
                        crate::renderer::types::draw_batch(
                            resources,
                            batch_bgs,
                            &mut render_pass,
                            batch,
                            &batch_pass,
                            &mut batch_bound,
                        );
                    }
                    self.frame_main_buffer_binds.fetch_add(
                        batch_bound.buffer_binds,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }

            // Section caps, clipped by the other planes and the clip volumes.
            if !slot.cap_buffers.is_empty() {
                if let Some(cap_pipeline) = resources.section_cap_pipeline(true) {
                    render_pass.set_pipeline(cap_pipeline);
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
            self.draw_mesh_instance_batches(&mut render_pass, camera_bg);

            // TransparentVolumeMesh boundary wireframe overlay (HDR path).
            if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
                if let (Some(tvm_bg), Some(hdr_wf)) = (
                    &self.mesh_uniforms.tvm_wireframe_bg,
                    resources.scene.hdr_wireframe(),
                ) {
                    for (slot, mesh_id) in self.mesh_uniforms.tvm_wireframe_draws.iter().enumerate()
                    {
                        if let Some(mesh) = resources.mesh_store.get(*mesh_id) {
                            render_pass.set_pipeline(hdr_wf);
                            render_pass.set_bind_group(1, tvm_bg, &[]);
                            if let Some(records) = &self.mesh_uniforms.tvm_wireframe_buf {
                                render_pass.set_vertex_buffer(1, records.slice(..));
                            }
                            render_pass.set_vertex_buffer(
                                0,
                                resources.geometry.vertex_slice(mesh.vertex_span),
                            );
                            if let Some(edge_buf) = &mesh.edge_index_buffer {
                                render_pass.set_index_buffer(
                                    edge_buf.slice(..),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                let slot = slot as u32;
                                render_pass.draw_indexed(
                                    0..mesh.edge_index_count,
                                    0,
                                    slot..slot + 1,
                                );
                            }
                        }
                    }
                }
            }

            // Draw skybox after built-in opaques : only uncovered sky pixels
            // pass the depth-equals-clear test. Drawn before plugin paint so blended plugin
            // content (additive/alpha particles that do not write depth)
            // composites over the sky instead of being painted over by it.
            if let Some(skybox_bg) = skybox_bg {
                render_pass.set_bind_group(0, camera_bg, &[]);
                render_pass.set_bind_group(1, skybox_bg, &[]);
                render_pass.set_pipeline(resources.scene_tier.ibl.skybox_pipeline());
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
        self.mark_pass_ran(ctx.vp_idx, HdrPass::HizStore);
        let view_proj = self
            .origin
            .camera(&ctx.frame.camera.render_camera)
            .view_proj()
            .to_cols_array_2d();
        let origin = self.origin.origin();
        let hiz = self.resources.hiz_pipelines(ctx.device);
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
        slot.cull.store_hiz_prev_depth(
            ctx.device, &hiz, encoder, depth_view, w, h, view_proj, origin,
        );
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
            self.mark_pass_ran(vp_idx, HdrPass::SsaaResolve);
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let fraction_store = if slot.pass_table.reads_indirect_fraction() {
                crate::gpu::StoreOp::Store
            } else {
                crate::gpu::StoreOp::Discard
            };
            if let (Some(pipeline), Some(bg)) = (
                &self.resources.post.ssaa_resolve_pipeline,
                &slot_hdr.ssaa_resolve_bind_group,
            ) {
                let mut resolve_pass =
                    encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                        #[cfg(any(wgpu29, wgpu30))]
                        multiview_mask: None,
                        label: Some("ssaa_resolve_pass"),
                        color_attachments: &[
                            Some(crate::gpu::RenderPassColorAttachment {
                                view: &slot_hdr.hdr_view,
                                resolve_target: None,
                                ops: crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Load,
                                    store: crate::gpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            }),
                            Some(crate::gpu::RenderPassColorAttachment {
                                view: &slot_hdr.indirect_fraction_view,
                                resolve_target: None,
                                ops: crate::gpu::Operations {
                                    load: crate::gpu::LoadOp::Load,
                                    store: fraction_store,
                                },
                                depth_slice: None,
                            }),
                        ],
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
                            load: crate::gpu::LoadOp::Clear(crate::plugin_api::depth::DEPTH_CLEAR),
                            store: crate::gpu::StoreOp::Store,
                        }),
                        // The value the scene pass clears stencil to when it
                        // owns this attachment. The surface mask pass then
                        // stamps over it as usual.
                        stencil_ops: Some(crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(crate::plugin_api::SURFACE_MASK_LAYERS),
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
                self.mark_pass_ran(vp_idx, HdrPass::SubHighlight);
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
        self.mark_pass_ran(ctx.vp_idx, HdrPass::DepthRead);
        // The depth plugins sample. Where a read-only depth attachment cannot
        // also be sampled (wgpu's GL backend counts it as written), that is a
        // copy taken just before the pass; elsewhere the scene depth itself.
        let sample_view = if self.resources.capabilities.direct_depth_sampling {
            self.viewport_slots[ctx.vp_idx]
                .hdr
                .as_ref()
                .unwrap()
                .hdr_depth_only_view
                .clone()
        } else {
            self.copy_depth_for_read(ctx, encoder)
        };
        // Prebuilt group handed to plugins that have a spare bind group. Plugins
        // at the four-group limit ignore it and bake the same depth-only view +
        // sampler into a group of their own.
        let depth_bg = {
            let resources = &self.resources;
            let hdr = self.viewport_slots[ctx.vp_idx].hdr.as_mut().unwrap();
            let view = sample_view.clone();
            hdr.depth_read_bg.get(view.clone(), || {
                ctx.device
                    .create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("plugin_depth_read_bg"),
                        layout: &resources.scene_tier.material.depth_read_bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::gpu::BindingResource::TextureView(&view),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: crate::gpu::BindingResource::Sampler(
                                    &resources.scene_tier.material.depth_read_sampler,
                                ),
                            },
                        ],
                    })
            })
        };
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

        // Depth attachment read-only (`depth_ops: None`) so `sample_view`, a
        // depth-aspect view of the same buffer where the device allows it, can
        // be sampled in the pass.
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
            &sample_view,
            &resources.scene_tier.material.depth_read_sampler,
            &depth_bg,
        );
    }

    /// Hand the encoder to the plugins that encode at `scope`. Without direct
    /// depth sampling their `scene_depth_only` is the depth-read copy, taken
    /// here so it holds the depth as of this scope.
    fn encode_scope(
        &mut self,
        ctx: &HdrFrameCtx,
        encoder: &mut crate::gpu::CommandEncoder,
        scope: crate::plugin_api::EncoderScope,
    ) {
        if !self.resources.capabilities.direct_depth_sampling
            && self
                .item_type_plugins
                .values()
                .any(|p| p.encoder_scopes().contains(&scope))
        {
            self.copy_depth_for_read(ctx, encoder);
        }
        if self.any_plugin_encodes_at(scope) {
            self.mark_pass_ran(
                ctx.vp_idx,
                match scope {
                    crate::plugin_api::EncoderScope::OnOpaqueSurfaces => {
                        HdrPass::ScopeOnOpaqueSurfaces
                    }
                    crate::plugin_api::EncoderScope::AfterOpaque => HdrPass::ScopeAfterOpaque,
                    _ => HdrPass::ScopeAfterTransparent,
                },
            );
        }
        self.dispatch_plugin_encode(encoder, ctx.frame, scope, ctx.device, ctx.queue, ctx.vp_idx);
    }

    /// Copy the scene depth into the viewport's depth-read copy with the depth
    /// blit, and return the copy's depth-aspect view. For a device whose
    /// read-only depth attachments cannot be sampled in the same pass.
    fn copy_depth_for_read(
        &mut self,
        ctx: &HdrFrameCtx,
        encoder: &mut crate::gpu::CommandEncoder,
    ) -> crate::gpu::TextureView {
        self.resources.ensure_depth_blit_pipeline(ctx.device);
        let resources = &self.resources;
        let hdr = self.viewport_slots[ctx.vp_idx].hdr.as_mut().unwrap();
        let size = hdr.hdr_depth_texture.size();
        if hdr
            .depth_read_copy
            .as_ref()
            .is_none_or(|c| c.texture.size() != size)
        {
            let texture = ctx.device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("depth_read_copy"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: hdr.hdr_depth_texture.format(),
                usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                    | crate::gpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
            let depth_only_view = texture.create_view(&crate::gpu::TextureViewDescriptor {
                aspect: crate::gpu::TextureAspect::DepthOnly,
                ..Default::default()
            });
            hdr.depth_read_copy = Some(crate::resources::postprocess::targets::DepthTarget {
                texture,
                view,
                depth_only_view,
            });
        }
        let source = hdr.hdr_depth_only_view.clone();
        let blit_bg = hdr.depth_read_copy_bg.get(source.clone(), || {
            ctx.device
                .create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("depth_read_copy_bg"),
                    layout: resources
                        .post
                        .depth_blit_bgl
                        .as_ref()
                        .expect("ensure_hdr_infra not called"),
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: crate::gpu::BindingResource::TextureView(&source),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::Sampler(
                                resources
                                    .post
                                    .pp_nearest_sampler
                                    .as_ref()
                                    .expect("ensure_hdr_infra not called"),
                            ),
                        },
                    ],
                })
        });
        let copy = hdr.depth_read_copy.as_ref().unwrap();
        {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("depth_read_copy_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: &copy.view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::plugin_api::depth::DEPTH_CLEAR),
                        store: crate::gpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(0),
                        store: crate::gpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(
                resources
                    .post
                    .depth_blit_pipeline
                    .as_ref()
                    .expect("built above"),
            );
            pass.set_bind_group(0, &blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        copy.depth_only_view.clone()
    }

    fn hdr_oit(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let _span = tracing::debug_span!(target: "viewport_lib::frame", "oit_pass").entered();
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
                // Every transparent item is in a blended batch, which goes through OIT.
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
                self.mark_pass_ran(vp_idx, HdrPass::Oit);
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

                if !self.instancing.batches.is_empty() {
                    // This viewport's own cull outputs.
                    let cull0 = &self.viewport_slots[vp_idx].cull;
                    let resources = &self.resources;
                    let batches = &self.instancing.batches;
                    let use_indirect_oit = self.instancing.gpu_culling_enabled
                        && resources.cull.oit.is_some()
                        && cull0.indirect_args_buf.is_some();

                    if use_indirect_oit {
                        // GPU-driven submission: one
                        // multi_draw_indexed_indirect_count per precomputed
                        // transparent group, as on the opaque path; the batches
                        // left out of the groups take the run path below.
                        let mut grouped = false;
                        if let (false, Some(compacted), Some(counts), Some(words)) = (
                            self.instancing.oit_draw_groups.is_empty(),
                            cull0.compacted_args_buf.as_ref(),
                            cull0.draw_counts_buf.as_ref(),
                            cull0.words_slice(),
                        ) {
                            grouped = true;
                            oit_pass.set_vertex_buffer(1, words);
                            let mut cur_pipe: Option<*const crate::gpu::RenderPipeline> = None;
                            let mut cur_bg: Option<*const crate::gpu::BindGroup> = None;
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            for group in &self.instancing.oit_draw_groups {
                                // Still compiling: the group waits a frame.
                                let Some(pl) = resources
                                    .cull
                                    .oit_features(
                                        PipelineKey::two_sided(group.two_sided),
                                        group.features,
                                        group.bindless,
                                    )
                                    .pipeline
                                else {
                                    continue;
                                };
                                let Some(bg) = cull0.object_cull_bind_groups.get(&group.bind_key)
                                else {
                                    continue;
                                };
                                if cur_pipe != Some(pl as *const _) {
                                    oit_pass.set_pipeline(pl);
                                    cur_pipe = Some(pl as *const _);
                                }
                                if cur_bg != Some(bg as *const _) {
                                    oit_pass.set_bind_group(1, bg, &[]);
                                    cur_bg = Some(bg as *const _);
                                }
                                let chunks = (group.vertex_chunk, group.index_chunk);
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
                        }
                        let (commands, binds) = crate::renderer::types::draw_culled_batches(
                            resources,
                            cull0,
                            &mut oit_pass,
                            batches,
                            crate::renderer::types::CulledPass::Oit,
                            self.instancing.multi_draw_active(),
                            |b| grouped && b.draws_grouped(),
                        );
                        self.frame_main_draw_commands
                            .fetch_add(commands, std::sync::atomic::Ordering::Relaxed);
                        self.frame_main_buffer_binds
                            .fetch_add(binds, std::sync::atomic::Ordering::Relaxed);
                    }

                    // Every blended batch the cull did not draw: the per-object
                    // OIT pipelines, by instance range.
                    if resources.oit.pipeline.is_some() {
                        let batch_pass = crate::renderer::types::BatchPass::Oit;
                        let mut batch_bound = crate::renderer::types::BatchBound::default();
                        let batch_bgs = &self.mesh_uniforms.material_bind_groups;
                        for batch in batches {
                            if !batch.is_transparent || (use_indirect_oit && batch.draws_culled()) {
                                continue;
                            }
                            if crate::renderer::types::draw_batch(
                                resources,
                                batch_bgs,
                                &mut oit_pass,
                                batch,
                                &batch_pass,
                                &mut batch_bound,
                            ) {
                                self.frame_main_draw_commands
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                        self.frame_main_buffer_binds.fetch_add(
                            batch_bound.buffer_binds,
                            std::sync::atomic::Ordering::Relaxed,
                        );
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
                                model: self.origin.rel_model(item.model),
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
            // Skipped while the pipeline is on a worker: transparency lands a
            // frame or two late.
            if let (Some(pipeline), Some(bg)) = (
                self.resources
                    .oit
                    .composite_pipeline
                    .as_ref()
                    .and_then(|p| p.get(0)),
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
            || slot.selection_outlines.plugin_outline_present
        {
            // Skipped while the pipeline is on a worker: the outline appears
            // a frame or two late.
            if let (Some(pipeline), Some(targets)) = (
                self.resources
                    .outline
                    .as_ref()
                    .and_then(|o| o.composite_hdr()),
                slot.outline_targets.as_ref(),
            ) {
                self.mark_pass_ran(vp_idx, HdrPass::OutlineComposite);
                let bg = &targets.composite_bind_group;
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
        if resources.scene.hdr.is_none() {
            return;
        }
        self.mark_pass_ran(vp_idx, HdrPass::Foreground);
        let family = resources.scene.hdr_family();

        let fg_camera = frame
            .camera
            .render_camera
            .foreground_camera(frame.effects.foreground.as_ref());

        // Opaque by priority then front-to-back, then blended by priority then
        // back-to-front (or blend order). Foreground transparency is plain
        // sorted alpha blending against the foreground depth, not OIT.
        let eye = fg_camera.eye_position();
        let dist_from_eye = |item: &SceneRenderItem| -> f32 {
            (item.model.translation - glam::DVec3::from(eye)).length() as f32
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
            if item.settings.opacity < 1.0
                || resources.material_or_default(item.material).is_blend()
            {
                transparent.push((idx, item));
            } else {
                opaque.push((idx, item));
            }
        }
        opaque.sort_by(|a, b| {
            a.1.settings
                .draw_priority
                .cmp(&b.1.settings.draw_priority)
                .then_with(|| {
                    dist_from_eye(a.1)
                        .partial_cmp(&dist_from_eye(b.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        transparent.sort_by_key(|(_, item)| {
            crate::renderer::types::draw_order::blend_key(
                item.settings.draw_priority,
                item.settings.blend_order,
                dist_from_eye(item),
            )
        });

        let mut render_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("hdr_foreground_pass"),
            // The foreground draws with the scene pass's pipelines, so it
            // attaches the indirect fraction as well. The composite skips
            // the occlusion terms where the foreground covers, so what it
            // writes there is never read; the rest must survive.
            color_attachments: &[
                Some(crate::gpu::RenderPassColorAttachment {
                    view: &slot_hdr.hdr_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(crate::gpu::RenderPassColorAttachment {
                    view: &slot_hdr.indirect_fraction_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: if slot.pass_table.reads_indirect_fraction() {
                            crate::gpu::StoreOp::Store
                        } else {
                            crate::gpu::StoreOp::Discard
                        },
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: fg_depth_view,
                depth_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Clear(crate::plugin_api::depth::DEPTH_CLEAR),
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
            let obj_bg = slot
                .foreground_objects
                .get(*idx)
                .and_then(|e| e.bind_group.as_ref().map(|bg| (bg, &e.uniform_buf)));
            draw_mesh_item(
                resources,
                &mut render_pass,
                item,
                obj_bg,
                // Foreground buffers hold one element each: draw at instance 0.
                0,
                false,
                true,
                &family,
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

    /// Encode built-in composite-input producer `index` (see
    /// `DeviceResources::post_producers`).
    fn hdr_builtin_producer(
        &mut self,
        ctx: &HdrFrameCtx,
        encoder: &mut crate::gpu::CommandEncoder,
        index: usize,
    ) {
        let _span = tracing::debug_span!(target: "viewport_lib::frame", "post_effects").entered();
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
            proj: frame.camera.render_camera.projection(),
            view: self.origin.camera(&frame.camera.render_camera).view(),
            near: frame.camera.render_camera.depth_range().0,
            far: frame.camera.render_camera.depth_range().1,
            first_light: frame.effects.lighting.lights.first(),
            foreground_active: self.foreground_active(frame),
            exposure: frame.effects.display.exposure,
            presented: self.render_advances_state(),
        };
        let timing = crate::resources::ProducerTiming {
            query_set: self.ts_query_set.as_ref(),
            written_mask: &self.ts_written_mask,
        };
        let producer = self.resources.post_producers()[index];
        if producer.enabled(&inputs)
            && producer.ready()
            && (!throttle_effects || !producer.throttleable())
        {
            self.mark_pass_ran(vp_idx, HdrPass::Producer(index as u8));
            producer.encode(slot_hdr, encoder, &inputs, &timing);
        }
    }

    fn hdr_external_producers(
        &mut self,
        ctx: &HdrFrameCtx,
        encoder: &mut crate::gpu::CommandEncoder,
    ) {
        let _span = tracing::debug_span!(target: "viewport_lib::frame", "post_effects").entered();
        let vp_idx = ctx.vp_idx;
        let frame = ctx.frame;
        let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
        let throttle_effects = self.degradation_effects_throttled;

        // External post-effect producers run after the built-ins, in
        // registration order, under the same throttle. Returned views are
        // collected and bound at their slots when the tone-map stage
        // rebuilds the composite bind group; the last producer for a slot
        // wins, and an external view over an enabled built-in logs once.
        if !self.post_effect_producers.is_empty() && !throttle_effects {
            self.mark_pass_ran(vp_idx, HdrPass::ExternalProducers);
            let ci = ctx.composite_inputs;
            let ctx = post_effect_ctx(ctx.device, slot_hdr, frame, vp_idx, &self.origin);
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
                let (builtin_on, slot_name) = match slot {
                    crate::plugin_api::PostEffectSlot::Bloom => (ci.bloom, "Bloom"),
                    crate::plugin_api::PostEffectSlot::AmbientOcclusion => {
                        (ci.ssao, "AmbientOcclusion")
                    }
                    crate::plugin_api::PostEffectSlot::ContactShadow => {
                        (ci.contact_shadows, "ContactShadow")
                    }
                };
                if builtin_on {
                    let producer = entry.producer.type_name();
                    self.resources.notices.report(
                        crate::notice::NoticeKey::new(
                            crate::NoticeCategory::Pipeline,
                            "post_effect_slot_taken",
                            [
                                crate::notice::name_id(slot_name),
                                crate::notice::name_id(producer),
                            ],
                        ),
                        crate::Severity::Info,
                        || crate::NoticeDetail::PostEffectSlotTaken {
                            slot: slot_name,
                            producer: producer.to_string(),
                        },
                    );
                }
                collected.push((slot, view));
            }
            self.frame_external_slot_views = collected;
        }
    }

    fn hdr_tone_map(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let _span = tracing::debug_span!(target: "viewport_lib::frame", "post_pass").entered();
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
        let chain = post_stage_chain(&self.resources, &self.post_effect_stages, pp);
        let final_target: crate::gpu::TextureView = if use_hdr_upscale {
            slot_hdr.upscale_view.as_ref().unwrap().clone()
        } else {
            output_view.clone()
        };
        if let Some(tone_map_pipeline) = &self.resources.post.tone_map_pipeline {
            self.mark_pass_ran(vp_idx, HdrPass::ToneMap);
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
    }

    /// The post-composite stages, chained from the composite's output toward
    /// the final target.
    fn hdr_post_stages(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let _span = tracing::debug_span!(target: "viewport_lib::frame", "post_pass").entered();
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        let frame = ctx.frame;
        let pp = &frame.effects.post_process;
        let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
        let use_hdr_upscale = slot_hdr.upscale_bind_group.is_some();
        let chain = post_stage_chain(&self.resources, &self.post_effect_stages, pp);
        let final_target: crate::gpu::TextureView = if use_hdr_upscale {
            slot_hdr.upscale_view.as_ref().unwrap().clone()
        } else {
            output_view.clone()
        };
        // -----------------------------------------------------------------------
        // Post-composite stages, chained toward the final target.
        // -----------------------------------------------------------------------
        {
            let timing = crate::resources::ProducerTiming {
                query_set: self.ts_query_set.as_ref(),
                written_mask: &self.ts_written_mask,
            };
            if !chain.is_empty() {
                self.mark_pass_ran(vp_idx, HdrPass::PostStages);
            }
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
                        let stage_ctx =
                            post_effect_ctx(ctx.device, slot_hdr, frame, vp_idx, &self.origin);
                        self.post_effect_stages[*j]
                            .stage
                            .encode(encoder, &target, &stage_ctx);
                    }
                }
            }
        }
    }

    fn hdr_upscale(&mut self, ctx: &HdrFrameCtx, encoder: &mut crate::gpu::CommandEncoder) {
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        let use_hdr_upscale = self.viewport_slots[vp_idx]
            .hdr
            .as_ref()
            .is_some_and(|hdr| hdr.upscale_bind_group.is_some());
        // -----------------------------------------------------------------------
        // HDR upscale pass: blit scene-resolution post-processed output to native
        // output_view. Only runs when render_scale < 1.0.
        // -----------------------------------------------------------------------
        if use_hdr_upscale {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let Some(upscale_bg) = &slot_hdr.upscale_bind_group {
                if let Some(pipeline) = &self.resources.post.dyn_res_upscale_pipeline {
                    self.mark_pass_ran(vp_idx, HdrPass::Upscale);
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
    }

    fn hdr_output_depth_blit(
        &mut self,
        ctx: &HdrFrameCtx,
        encoder: &mut crate::gpu::CommandEncoder,
    ) {
        let vp_idx = ctx.vp_idx;
        // Depth blit pass: when render_scale < 1.0, the scene depth texture is
        // smaller than the output surface. Copy it to output_depth_texture (native
        // resolution) so the post-tone-map passes below can attach output_depth_view
        // alongside output_view without a size mismatch. Skipped when render_scale
        // is 1.0 (output_depth_view is just a second view of hdr_depth_texture).
        {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let Some(blit_bg) = &slot_hdr.depth_blit_bind_group {
                if let Some(blit_pipeline) = &self.resources.post.depth_blit_pipeline {
                    self.mark_pass_ran(vp_idx, HdrPass::OutputDepthBlit);
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
                                        load: crate::gpu::LoadOp::Clear(
                                            crate::plugin_api::depth::DEPTH_CLEAR,
                                        ),
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
    }

    fn hdr_foreground_stamp(
        &mut self,
        ctx: &HdrFrameCtx,
        encoder: &mut crate::gpu::CommandEncoder,
    ) {
        let vp_idx = ctx.vp_idx;
        // Foreground depth stamp: write near depth into output_depth_view
        // wherever the foreground pass drew, so the post-tone-map passes
        // below (grid, ground plane, gizmos) are occluded by foreground
        // geometry. Runs after the depth blit in both render-scale cases
        // (at scale 1.0 output_depth_view aliases the scene depth, which the
        // tone map pass has already consumed).
        if self.foreground_active(ctx.frame) {
            self.mark_pass_ran(vp_idx, HdrPass::ForegroundStamp);
            let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
            if let (Some(fg_view), Some(pipeline), Some(bgl), Some(nearest)) = (
                hdr.foreground_depth_only_view.clone(),
                self.resources.post.foreground_stamp_pipeline.as_ref(),
                self.resources.post.foreground_stamp_bgl.as_ref(),
                self.resources.post.pp_nearest_sampler.as_ref(),
            ) {
                let stamp_bg = hdr.foreground_stamp_bg.get(fg_view.clone(), || {
                    ctx.device
                        .create_bind_group(&crate::gpu::BindGroupDescriptor {
                            label: Some("foreground_stamp_bg"),
                            layout: bgl,
                            entries: &[
                                crate::gpu::BindGroupEntry {
                                    binding: 0,
                                    resource: crate::gpu::BindingResource::TextureView(&fg_view),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 1,
                                    resource: crate::gpu::BindingResource::Sampler(nearest),
                                },
                            ],
                        })
                });
                let slot_hdr = &*hdr;
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
        let _span = tracing::debug_span!(target: "viewport_lib::frame", "overlay_pass").entered();
        let resources = &self.resources;
        let frame = ctx.frame;
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        // Grid pass (HDR path): draw the existing analytical grid on the final
        // output after tone mapping / FXAA, reusing the scene depth buffer so
        // scene geometry still occludes the grid exactly as in the LDR path.
        if let (true, Some(grid_pipeline)) = (
            frame.viewport.show_grid,
            self.resources
                .guide_pipeline(crate::resources::overlay::guides::GUIDE_GRID),
        ) {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            self.mark_pass_ran(vp_idx, HdrPass::SceneOverlays);
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
            grid_pass.set_pipeline(grid_pipeline);
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
            self.mark_pass_ran(vp_idx, HdrPass::SceneOverlays);
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
            gp_pass.set_pipeline(self.resources.scene_tier.ground.pipeline());
            let ground_bg = self.viewport_slots[vp_idx]
                .ground_bind_group
                .current()
                .unwrap_or(&self.resources.scene_tier.ground.bind_group);
            gp_pass.set_bind_group(0, ground_bg, &[]);
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
                self.mark_pass_ran(vp_idx, HdrPass::SceneOverlays);
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

                if let (false, Some(pipeline)) = (
                    slot.constraint_line_buffers.is_empty(),
                    self.resources
                        .guide_pipeline(crate::resources::overlay::guides::GUIDE_LINES),
                ) {
                    overlay_pass.set_pipeline(pipeline);
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
                    overlay_pass.set_pipeline(self.resources.scene_tier.outline.xray_pipeline());
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (mesh_id, xray) in &slot.xray_object_buffers {
                        let Some(mesh) = self.resources.mesh_store.get(*mesh_id) else {
                            continue;
                        };
                        overlay_pass.set_bind_group(1, &xray.bind_group, &[]);
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
        let _span = tracing::debug_span!(target: "viewport_lib::frame", "overlay_pass").entered();
        let device = ctx.device;
        let queue = ctx.queue;
        let output_view = ctx.output_view;
        let vp_idx = ctx.vp_idx;
        let w = ctx.w;
        let h = ctx.h;
        // Overlay shapes and the merged text batch (HDR path): drawn last.
        let has_overlay = {
            let ov = &self.viewport_slots[vp_idx].overlay;
            ov.overlay_shape_gpu_data.is_some() || ov.label_gpu_data.is_some()
        };

        // HDR backdrop blur: the tonemapped scene is on an intermediate so we
        // can sample it for the blur. When blur is needed we redirect the
        // tonemapped output to a managed intermediate, blur it, then draw
        // overlays there and blit to output_view at the end.
        let needs_hdr_blur = self.has_backdrop_blur_shapes(vp_idx);
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
            let spread = self.viewport_slots[vp_idx]
                .overlay
                .overlay_shape_gpu_data
                .as_ref()
                .map(|d| d.max_blur_radius)
                .unwrap_or(8.0);
            Some(self.run_backdrop_blur(encoder, device, queue, source, spread))
        } else {
            None
        };

        if has_overlay {
            self.mark_pass_ran(vp_idx, HdrPass::Overlay);
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
                self.draw_blur_shapes(&mut overlay_pass, bg, vp_idx);
            }
            let ov = &self.viewport_slots[vp_idx].overlay;
            emit_overlay_2d!(self, ov, overlay_pass);
        }
    }
}
