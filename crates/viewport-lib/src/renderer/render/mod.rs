use super::*;

/// Default background #3b3b40, in linear light: the renderer outputs linear and
/// the sRGB target encodes on write.
const DEFAULT_BACKGROUND: [f32; 4] = [0.0437, 0.0437, 0.0513, 1.0];

/// The frame's background as premultiplied linear RGBA.
///
/// The background is a colour the scene composites over, so every consumer of
/// it wants it premultiplied: at alpha 1 the straight value unchanged, at alpha
/// 0 nothing rather than a colour with no coverage, and in between a translucent
/// plate. `Colour` keeps storing alpha straight, as its own contract says;
/// premultiplication belongs here at the pipeline boundary.
fn background_premultiplied(frame: &FrameData) -> [f32; 4] {
    let c = frame
        .viewport
        .background_colour
        .map(|c| c.to_linear_rgba())
        .unwrap_or(DEFAULT_BACKGROUND);
    [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]]
}

/// Emit one indirect draw run against the shared args buffer. `start` is the
/// run's first global batch index and `len` the number of consecutive batches
/// in it (all sharing pipeline, bind group, and geometry chunk). Each
/// `DrawIndexedIndirect` entry is 20 bytes. When native multi-draw is
/// available a run longer than one collapses into a single
/// `multi_draw_indexed_indirect`; otherwise (and for single-batch runs) it
/// falls back to one `draw_indexed_indirect` per entry over the same buffer.
///
/// Returns the number of draw commands issued: `1` for a collapsed multi-draw,
/// `len` for the per-entry fallback. Callers sum this for the draw-command
/// stats counters.
pub(crate) fn emit_indirect_run(
    render_pass: &mut crate::gpu::RenderPass<'_>,
    indirect_buf: &crate::gpu::Buffer,
    start: u64,
    len: u32,
    multi_draw: bool,
) -> u32 {
    if multi_draw && len > 1 {
        render_pass.multi_draw_indexed_indirect(indirect_buf, start * 20, len);
        1
    } else {
        for k in 0..len as u64 {
            render_pass.draw_indexed_indirect(indirect_buf, (start + k) * 20);
        }
        len
    }
}

/// Emit the frame's 2D overlay draw list (the main segments, not a layer's
/// members) into `$render_pass`. A pass the renderer cannot interrupt reads no
/// destination: layer composites draw with their fallback blend and backdrops
/// are skipped. Run after all scene content so the overlays sit on top. The
/// overlay pipelines are format-neutral (no separate LDR/HDR variant), so this
/// is shared verbatim across paths.
macro_rules! emit_overlay_2d {
    ($this:ident, $ov:ident, $render_pass:ident) => {{
        let segs = $ov.main_segments();
        let pass: &mut crate::gpu::RenderPass<'_> = &mut $render_pass;
        crate::renderer::render::emit_overlay_segments(
            &$this.resources,
            $ov,
            pass,
            segs,
            &crate::renderer::render::OverlayEmit::NO_DESTINATION,
        );
    }};
}

/// What an overlay draw can read of the destination.
pub(crate) struct OverlayEmit<'a> {
    /// Layer composites can blend by formula against the snapshot.
    pub dst: bool,
    /// The backdrop pipeline's group 0 over the blurred destination; `None`
    /// skips every backdrop draw.
    pub backdrop: Option<&'a crate::gpu::BindGroup>,
}

impl OverlayEmit<'_> {
    /// A pass that cannot be interrupted, or a layer's members.
    pub(crate) const NO_DESTINATION: OverlayEmit<'static> = OverlayEmit {
        dst: false,
        backdrop: None,
    };
    /// A layer's members, drawn into its scratch target.
    pub(crate) const MEMBERS: OverlayEmit<'static> = Self::NO_DESTINATION;
}

/// The pipeline a segment draws with, to skip redundant `set_pipeline` calls.
#[derive(Clone, Copy, PartialEq)]
enum OverlayPipe {
    Shape(crate::renderer::types::OverlayBlend),
    Text(crate::renderer::types::OverlayBlend),
    Backdrop,
    Layer(crate::resources::overlay::overlay_layer::LayerPipelineKey),
}

/// Emit `segs` of `ov`'s draw list. Each segment draws through the shape,
/// text, backdrop or layer pipeline for its blend mode, and the pipeline is
/// set only when it changes between consecutive segments.
pub(crate) fn emit_overlay_segments(
    resources: &crate::resources::DeviceResources,
    ov: &crate::renderer::ViewportOverlay,
    render_pass: &mut crate::gpu::RenderPass<'_>,
    segs: &[crate::renderer::overlay_draw_order::OverlayDrawSegment],
    emit: &OverlayEmit<'_>,
) {
    use crate::renderer::overlay_draw_order::OverlayDrawSource;
    let mut last_pipe: Option<OverlayPipe> = None;
    let mut set = |pass: &mut crate::gpu::RenderPass<'_>,
                   key: OverlayPipe,
                   pipeline: &crate::gpu::RenderPipeline| {
        if last_pipe != Some(key) {
            pass.set_pipeline(pipeline);
            last_pipe = Some(key);
        }
    };
    // The per-draw instance stream the text pipeline reads at vertex slot
    // 1: slot 0 is the identity, retained groups their own.
    let instances = ov.overlay_bindings.instances.buffer();
    for seg in segs {
        let shape_pipeline = resources.overlay_shape.pipeline_for(seg.blend);
        let text_pipeline = resources.overlay_text.pipeline_for(seg.blend);
        match seg.source {
            OverlayDrawSource::Shape {
                instance_start,
                instance_count,
                fill,
            } => {
                if let Some(ref sd) = ov.overlay_shape_gpu_data {
                    if let (Some(pipeline), Some(ibuf), Some(shadow_bg), Some(fill_bg)) = (
                        shape_pipeline,
                        &sd.instance_buf,
                        &sd.shadow_bind_group,
                        sd.fill_bind_groups.get(fill as usize),
                    ) {
                        set(render_pass, OverlayPipe::Shape(seg.blend), pipeline);
                        // The run is bound by its byte offset rather than a
                        // first-instance index, which not every backend
                        // carries into `instance_index`.
                        render_pass.set_bind_group(0, shadow_bg, &[0]);
                        render_pass.set_bind_group(1, fill_bg, &[]);
                        render_pass.set_vertex_buffer(
                            0,
                            ibuf.slice(
                                crate::resources::OverlayShapeInstance::byte_offset(
                                    instance_start,
                                )..,
                            ),
                        );
                        render_pass.draw(0..6, 0..instance_count);
                    }
                }
            }
            OverlayDrawSource::Text {
                vertex_start,
                vertex_count,
                fill,
            } => {
                if let (Some(pipeline), Some(td), Some(inst)) =
                    (text_pipeline, ov.label_gpu_data.as_ref(), instances)
                {
                    if let (Some(fill_bg), Some(vbuf)) =
                        (td.fill_bind_groups.get(fill as usize), &td.vertex_buf)
                    {
                        set(render_pass, OverlayPipe::Text(seg.blend), pipeline);
                        render_pass.set_bind_group(0, &td.bind_group, &[]);
                        render_pass.set_bind_group(1, fill_bg, &[]);
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass.set_vertex_buffer(1, inst.slice(..));
                        render_pass.draw(vertex_start..vertex_start + vertex_count, 0..1);
                    }
                }
            }
            OverlayDrawSource::Retained { draw_index } => {
                // A retained group draws through the text pipeline, sharing its
                // bind group (atlas, clip, viewport) with the immediate text
                // batch, from its own cached buffer, fill records and
                // per-frame instance slot.
                if let (Some(pipeline), Some(td), Some(rd), Some(inst)) = (
                    text_pipeline,
                    ov.label_gpu_data.as_ref(),
                    ov.overlay_retained_draws.get(draw_index as usize),
                    instances,
                ) {
                    set(render_pass, OverlayPipe::Text(seg.blend), pipeline);
                    render_pass.set_bind_group(0, &td.bind_group, &[]);
                    render_pass.set_bind_group(1, &rd.fill_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, rd.vertex_buf.slice(..));
                    render_pass.set_vertex_buffer(1, inst.slice(..));
                    render_pass.draw(0..rd.vertex_count, rd.instance_index..rd.instance_index + 1);
                }
            }
            OverlayDrawSource::RetainedShape { draw_index } => {
                // A retained group's SDF shape stream, through the shape
                // pipeline, from its cached instance buffer, shadow and
                // fill bind groups, with its per-frame group instance
                // selected by the dynamic offset.
                if let (Some(pipeline), Some(rd)) = (
                    shape_pipeline,
                    ov.overlay_retained_shape_draws.get(draw_index as usize),
                ) {
                    set(render_pass, OverlayPipe::Shape(seg.blend), pipeline);
                    let offset = rd.instance_index
                        * crate::resources::overlay::geometry::INSTANCE_UNIFORM_STRIDE as u32;
                    render_pass.set_bind_group(0, &rd.bind_group, &[offset]);
                    render_pass.set_bind_group(1, &rd.fill_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, rd.instance_buf.slice(..));
                    render_pass.draw(0..6, 0..rd.instance_count);
                }
            }
            OverlayDrawSource::Backdrop {
                vertex_start,
                vertex_count,
            } => {
                // A backdrop-blur shape over the blurred destination. Without
                // a blurred destination there is nothing to draw it over.
                if let (Some(bg), Some(sd), Some(pipeline)) = (
                    emit.backdrop,
                    ov.overlay_shape_gpu_data.as_ref(),
                    resources.overlay_shape.tex_pipeline.as_ref(),
                ) {
                    if let (Some(clip_bg), Some(vbuf)) =
                        (sd.tex_clip_bind_group.as_ref(), sd.blur_vertex_buf.as_ref())
                    {
                        set(render_pass, OverlayPipe::Backdrop, pipeline);
                        render_pass.set_bind_group(0, bg, &[]);
                        render_pass.set_bind_group(1, clip_bg, &[]);
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass.draw(vertex_start..vertex_start + vertex_count, 0..1);
                    }
                }
            }
            OverlayDrawSource::Layer { index } => {
                let lt = &ov.layer_targets;
                let (Some(d), Some(ibuf), Some(clip_bg)) = (
                    lt.draws.get(index as usize),
                    lt.instance_buf.as_ref(),
                    lt.composite_clip_bg.as_ref(),
                ) else {
                    continue;
                };
                let Some(bg) = lt.composite_bg(d.scratch) else {
                    continue;
                };
                let layer = &resources.overlay_layer;
                let slice = ibuf.slice(
                    crate::resources::overlay::overlay_layer::OverlayLayerInstance::byte_offset(
                        index,
                    )..,
                );
                use crate::resources::overlay::overlay_layer::LayerPipelineKey as K;
                // The backdrop under the content, when the destination has
                // been blurred.
                if d.style.backdrop.is_active() && emit.backdrop.is_some() {
                    if let Some(pipeline) = layer.pipeline(K::Backdrop) {
                        set(render_pass, OverlayPipe::Layer(K::Backdrop), pipeline);
                        render_pass.set_bind_group(0, bg, &[]);
                        render_pass.set_bind_group(1, clip_bg, &[]);
                        render_pass.set_vertex_buffer(0, slice);
                        render_pass.draw(0..6, 0..1);
                    }
                }
                let key = if emit.dst { d.key() } else { d.fallback_key() };
                if let Some(pipeline) = layer.pipeline(key) {
                    set(render_pass, OverlayPipe::Layer(key), pipeline);
                    render_pass.set_bind_group(0, bg, &[]);
                    render_pass.set_bind_group(1, clip_bg, &[]);
                    render_pass.set_vertex_buffer(0, slice);
                    render_pass.draw(0..6, 0..1);
                }
            }
        }
    }
}

mod hdr_path;
mod ldr_path;
mod paint_direct;
pub(crate) mod pass_table;
mod transmission;

impl ViewportRenderer {
    /// Render the scene into an intermediate dyn-res texture for the LDR callback
    /// render path (e.g. eframe's `CallbackTrait`).
    ///
    /// Call from `CallbackTrait::prepare` after [`prepare`](Self::prepare), passing the
    /// `egui_encoder`. If `current_render_scale < 1.0`, the full scene is drawn into a
    /// scaled intermediate texture and `true` is returned. Call
    /// [`paint_dyn_res_blit`](Self::paint_dyn_res_blit) from `CallbackTrait::paint`
    /// instead of [`paint`](Self::paint).
    ///
    /// If scale is 1.0 or above, nothing is encoded and `false` is returned. Call
    /// [`paint`](Self::paint) as normal.
    ///
    /// The `egui_encoder` is submitted before the surface render pass begins, so the
    /// intermediate texture is fully written before the blit reads it.
    pub(crate) fn prepare_ldr_dyn_res(
        &mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        device: &crate::gpu::Device,
        frame: &FrameData,
    ) -> bool {
        if self.current_render_scale >= 1.0 - 0.001 {
            return false;
        }

        let vp_idx = frame.camera.viewport_index;
        let w = (frame.camera.viewport_size[0] as u32).max(1);
        let h = (frame.camera.viewport_size[1] as u32).max(1);
        let sw = ((w as f32 * self.current_render_scale) as u32).max(1);
        let sh = ((h as f32 * self.current_render_scale) as u32).max(1);

        self.ensure_dyn_res_target(device, vp_idx, [sw, sh], [w, h]);
        self.resources.ensure_dyn_res_ds_pipeline(device);

        let bg_colour = background_premultiplied(frame);

        {
            let slot = &self.viewport_slots[vp_idx];
            let dr = slot.dyn_res.as_ref().unwrap();
            let colour_view = &dr.colour_view;
            let depth_view = &dr.depth_view;
            let camera_bg = &slot.camera_bind_group;
            let grid_bg = &slot.grid_bind_group;

            let mut render_pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("ldr_dyn_res_render_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: colour_view,
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
                    view: depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::plugin_api::depth::DEPTH_CLEAR),
                        store: crate::gpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            emit_draw_calls!(
                &self.resources,
                &mut render_pass,
                frame,
                &self.instancing.batches,
                camera_bg,
                grid_bg,
                Some(slot),
                &self.mesh_uniforms.material_bind_groups,
                &self.prepared_surfaces,
                self.per_object_bundle.as_ref()
            );
            self.draw_mesh_instance_batches(&mut render_pass, camera_bg);
            // TransparentVolumeMesh boundary wireframe overlay.
            if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
                if let Some(ref tvm_bg) = self.mesh_uniforms.tvm_wireframe_bg {
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for (slot, mesh_id) in self.mesh_uniforms.tvm_wireframe_draws.iter().enumerate()
                    {
                        if let (Some(mesh), Some(wf)) = (
                            self.resources.mesh_store.get(*mesh_id),
                            self.resources.scene.wireframe(),
                        ) {
                            render_pass.set_pipeline(wf);
                            render_pass.set_bind_group(1, tvm_bg, &[]);
                            if let Some(records) = &self.mesh_uniforms.tvm_wireframe_buf {
                                render_pass.set_vertex_buffer(1, records.slice(..));
                            }
                            render_pass.set_vertex_buffer(
                                0,
                                self.resources.geometry.vertex_slice(mesh.vertex_span),
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
            // Item-type plugin paint (LDR opt-in only): after all built-in
            // scene content, mirroring the HDR scene-pass position.
            self.dispatch_plugin_paint(&mut render_pass, frame, false);
            // Outline composite after all scene content.
            emit_outline_composite!(&self.resources, &mut render_pass, Some(slot));
            // Sub-object highlight (LDR path).
            self.paint_ldr_sub_highlight(&mut render_pass, vp_idx, camera_bg);
            let ov = &self.viewport_slots[vp_idx].overlay;
            emit_overlay_2d!(self, ov, render_pass);
        }

        true
    }

    /// Blit the dyn-res intermediate texture into the provided render pass.
    ///
    /// Call from `CallbackTrait::paint` when
    /// [`prepare_ldr_dyn_res`](Self::prepare_ldr_dyn_res) returned `true` for the same
    /// frame. Emits a fullscreen upscale quad into `render_pass`.
    pub(crate) fn paint_dyn_res_blit<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        let vp_idx = frame.camera.viewport_index;
        if let Some(dr) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.dyn_res.as_ref())
        {
            if let Some(pipeline) = &self.resources.post.dyn_res_upscale_ds_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &dr.upscale_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
    }

    /// Run the full HDR pipeline (OIT, EDL, tone-map) for the eframe callback model.
    ///
    /// This is the HDR counterpart of
    /// [`prepare_ldr_dyn_res`](Self::prepare_ldr_dyn_res) for use when
    /// `frame.effects.display.mode` is `PipelineMode::Hdr`.
    ///
    /// Internally this method:
    /// 1. Calls [`prepare`](Self::prepare) to upload uniforms and run the shadow pass.
    /// 2. Ensures a per-viewport intermediate texture at the viewport's native resolution.
    /// 3. Calls the full render pipeline (including OIT and EDL) into that texture.
    ///
    /// The returned [`wgpu::CommandBuffer`] must be returned from
    /// `CallbackTrait::prepare` so eframe submits it **before** the egui render pass.
    ///
    /// Call [`paint_hdr_blit`](Self::paint_hdr_blit) from `CallbackTrait::paint` to
    /// composite the intermediate texture into the egui render pass.
    pub(crate) fn prepare_hdr_callback(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> crate::gpu::CommandBuffer {
        self.direct_paint = false;
        self.prepare(device, queue, frame);

        let vp_idx = frame.camera.viewport_index;
        // Intermediate texture must be at physical pixel size so it matches the
        // HDR depth buffer allocated inside render_frame_internal (which also
        // uses physical pixels). Using logical size here produces a mismatch on
        // hidpi displays between the colour attachment (this texture) and the
        // depth attachment (hdr_depth_view) in the grid/overlay passes.
        let ppp = frame.camera.pixels_per_point;
        let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        // Ensure the blit pipeline (required by create_hdr_callback_target).
        self.resources.ensure_dyn_res_pipeline(device);
        self.resources.ensure_dyn_res_ds_pipeline(device);

        // Create or resize the per-viewport intermediate texture.
        self.ensure_viewport_slot(device, vp_idx);
        let needs_create = match self.viewport_slots[vp_idx].hdr_callback.as_ref() {
            None => true,
            Some(t) => t.size != [w, h],
        };
        if needs_create {
            let target = self.resources.create_hdr_callback_target(device, [w, h]);
            self.viewport_slots[vp_idx].hdr_callback = Some(target);
        }

        // Create a fresh TextureView from the stored Texture.
        // This owned view does not borrow viewport_slots, allowing the subsequent
        // mutable call to render_frame_internal without a borrow conflict.
        let output_view = self.viewport_slots[vp_idx]
            .hdr_callback
            .as_ref()
            .unwrap()
            .texture
            .create_view(&crate::gpu::TextureViewDescriptor::default());

        self.render_frame_internal(device, queue, &output_view, vp_idx, frame)
    }

    /// HDR encode for a single viewport in the multi-viewport eframe callback model.
    ///
    /// Like [`prepare_hdr_callback`](Self::prepare_hdr_callback) but skips the internal
    /// [`prepare`](Self::prepare) call. The caller must have already called
    /// [`prepare_scene`](Self::prepare_scene) and [`prepare_viewport`](Self::prepare_viewport)
    /// for `id` before invoking this.
    ///
    /// Multi-viewport HDR sequence:
    /// 1. Call `prepare_scene` once.
    /// 2. Call `prepare_viewport` for each viewport.
    /// 3. Call this method for each viewport; collect the returned `CommandBuffer`s.
    /// 4. Return them from `CallbackTrait::prepare`.
    ///
    /// Call [`paint_hdr_blit`](Self::paint_hdr_blit) for each viewport from
    /// `CallbackTrait::paint` with the scissor/viewport rect set first.
    pub(crate) fn prepare_hdr_callback_viewport(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: ViewportId,
        frame: &FrameData,
    ) -> crate::gpu::CommandBuffer {
        if self.device_lost_now() {
            return crate::renderer::empty_command_buffer(device);
        }
        let vp_idx = id.0;
        let ppp = frame.camera.pixels_per_point;
        let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        self.resources.ensure_dyn_res_pipeline(device);
        self.resources.ensure_dyn_res_ds_pipeline(device);

        self.ensure_viewport_slot(device, vp_idx);
        let needs_create = match self.viewport_slots[vp_idx].hdr_callback.as_ref() {
            None => true,
            Some(t) => t.size != [w, h],
        };
        if needs_create {
            let target = self.resources.create_hdr_callback_target(device, [w, h]);
            self.viewport_slots[vp_idx].hdr_callback = Some(target);
        }

        let output_view = self.viewport_slots[vp_idx]
            .hdr_callback
            .as_ref()
            .unwrap()
            .texture
            .create_view(&crate::gpu::TextureViewDescriptor::default());

        self.render_frame_internal(device, queue, &output_view, vp_idx, frame)
    }

    /// Blit the HDR intermediate texture into the egui render pass.
    ///
    /// Call from `CallbackTrait::paint` after
    /// [`prepare_hdr_callback`](Self::prepare_hdr_callback) has been called for the
    /// same frame and viewport. Emits a fullscreen triangle into `render_pass`.
    pub(crate) fn paint_hdr_blit<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        if self.device_lost_now() {
            return;
        }
        let vp_idx = frame.camera.viewport_index;
        if let Some(hc) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.hdr_callback.as_ref())
        {
            if let Some(pipeline) = &self.resources.post.dyn_res_upscale_ds_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &hc.blit_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
        // Shadow atlas viewer overlay.
        if let (true, true, Some(pipeline)) = (
            self.resources.scene_tier.is_built(),
            frame.effects.debug.show_shadow_atlas,
            self.resources
                .guide_pipeline(crate::resources::overlay::guides::GUIDE_ATLAS_VIEWER),
        ) {
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.resources.scene_tier.shadow.atlas_viewer_bg, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }

    /// Like [`paint_hdr_blit`](Self::paint_hdr_blit) but for render passes without a
    /// depth-stencil attachment. Use this when you create the blit render pass yourself
    /// (e.g. winit) and omit the depth attachment.
    pub(crate) fn paint_hdr_blit_no_ds<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        if self.device_lost_now() {
            return;
        }
        let vp_idx = frame.camera.viewport_index;
        if let Some(hc) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.hdr_callback.as_ref())
        {
            if let Some(pipeline) = &self.resources.post.dyn_res_upscale_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &hc.blit_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
    }

    /// Unified prepare step for the eframe `CallbackTrait::prepare` method.
    ///
    /// Replaces manual `prepare` + `prepare_ldr_dyn_res` or `prepare_hdr_callback`
    /// calls. Dispatches internally based on `frame.effects.display.mode`:
    ///
    /// - HDR path (`display.mode = PipelineMode::Hdr`): runs the full HDR pipeline (OIT,
    ///   EDL, tone-map) and returns the resulting `CommandBuffer` for eframe to
    ///   submit before the egui render pass.
    /// - LDR path: calls `prepare`, and if dynamic resolution is active, encodes the
    ///   scene into a separate `CommandBuffer` (also submitted before the render
    ///   pass). Returns an empty `Vec` when dyn-res is inactive.
    ///
    /// Call [`paint_callback`](Self::paint_callback) from `CallbackTrait::paint`.
    pub(crate) fn prepare_callback(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> Vec<crate::gpu::CommandBuffer> {
        if self.skip_frame_device_lost() {
            return Vec::new();
        }
        if self.renders_hdr(frame) {
            let cb = self.prepare_hdr_callback(device, queue, frame);
            vec![cb]
        } else {
            self.direct_paint = true;
            self.prepare(device, queue, frame);
            if self.current_render_scale < 1.0 - 0.001 {
                let mut encoder =
                    device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                        label: Some("ldr_dyn_res_callback_encoder"),
                    });
                self.prepare_ldr_dyn_res(&mut encoder, device, frame);
                vec![encoder.finish()]
            } else {
                Vec::new()
            }
        }
    }

    /// Unified paint step for the eframe `CallbackTrait::paint` method.
    ///
    /// Call after [`prepare_callback`](Self::prepare_callback) for the same frame.
    /// Dispatches internally to `paint_hdr_blit`, `paint_dyn_res_blit`, or `paint`
    /// based on which path `prepare_callback` activated.
    pub(crate) fn paint_callback<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        if self.device_lost_now() {
            return;
        }
        let vp_idx = frame.camera.viewport_index;
        if self.renders_hdr(frame) {
            if self
                .viewport_slots
                .get(vp_idx)
                .and_then(|s| s.hdr_callback.as_ref())
                .is_some()
            {
                self.paint_hdr_blit(render_pass, frame);
                return;
            }
        }
        if self.current_render_scale < 1.0 - 0.001
            && self
                .viewport_slots
                .get(vp_idx)
                .and_then(|s| s.dyn_res.as_ref())
                .is_some()
        {
            self.paint_dyn_res_blit(render_pass, frame);
        } else {
            self.paint_to(render_pass, frame);
        }
    }

    /// High-level HDR render for a single viewport identified by `id`.
    ///
    /// Unlike [`render`](Self::render), this method does **not** call
    /// [`prepare`](Self::prepare) internally.  The caller must have already called
    /// [`prepare_scene`](Self::prepare_scene) and
    /// [`prepare_viewport`](Self::prepare_viewport) for `id` before invoking this.
    ///
    /// This is the right entry point for multi-viewport frames:
    /// 1. Call `prepare_scene` once.
    /// 2. Call `prepare_viewport` for each viewport.
    /// 3. Call `render_viewport` for each viewport with its own `output_view`.
    ///
    /// Returns a [`wgpu::CommandBuffer`] ready to submit.
    pub(crate) fn render_viewport(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        id: ViewportId,
        frame: &FrameData,
    ) -> crate::gpu::CommandBuffer {
        let _cache = self.resources.pipeline_compiler.cache_scope();
        if self.device_lost_now() {
            return crate::renderer::empty_command_buffer(device);
        }
        self.render_frame_internal(device, queue, output_view, id.0, frame)
    }

    /// High-level HDR render method. Handles the full post-processing pipeline:
    /// scene -> HDR texture -> (bloom) -> (SSAO) -> tone map -> output_view.
    ///
    /// When `frame.effects.display.mode` is `PipelineMode::Direct`, falls back to a simple LDR render
    /// pass targeting `output_view` directly.
    ///
    /// Returns a `CommandBuffer` ready to submit.
    pub(crate) fn render(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        frame: &FrameData,
    ) -> crate::gpu::CommandBuffer {
        let _cache = self.resources.pipeline_compiler.cache_scope();
        if self.skip_frame_device_lost() {
            return crate::renderer::empty_command_buffer(device);
        }
        // Always run prepare() to upload uniforms and run the shadow pass.
        self.direct_paint = false;
        self.prepare(device, queue, frame);
        self.render_frame_internal(
            device,
            queue,
            output_view,
            frame.camera.viewport_index,
            frame,
        )
    }

    /// Deferred-submit counterpart of [`render`](Self::render): runs prepare in
    /// deferred mode, encodes the scene pass, and returns every command buffer
    /// (prepare passes first, scene pass last) without submitting. The caller
    /// submits them, in order, on the device-driving thread.
    pub(crate) fn render_deferred(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        frame: &FrameData,
    ) -> Vec<crate::gpu::CommandBuffer> {
        if self.skip_frame_device_lost() {
            return Vec::new();
        }
        self.direct_paint = false;
        let (_stats, mut buffers) = self.prepare_deferred(device, queue, frame);
        buffers.push(self.render_frame_internal(
            device,
            queue,
            output_view,
            frame.camera.viewport_index,
            frame,
        ));
        buffers
    }

    /// Render-only path shared by `render()` and `render_viewport()`.
    ///
    /// `vp_idx` selects the per-viewport slot to use for camera/HDR state,
    /// independent of `frame.camera.viewport_index`.
    fn render_frame_internal(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        vp_idx: usize,
        frame: &FrameData,
    ) -> crate::gpu::CommandBuffer {
        let _span =
            tracing::debug_span!(target: "viewport_lib::frame", "render", vp = vp_idx).entered();
        self.check_output_format(output_view);
        // Draw the LOD-resolved surfaces from prepare (level mesh chosen,
        // culled items hidden), extended with the boundary draws contributed
        // by opaque volume meshes (see the matching construction in
        // `prepare.rs`). Reading the resolved list rather than the raw
        // `frame.scene.surfaces` is what carries the LOD swap and cull into
        // the scene pass. The list is moved out for the frame rather than
        // cloned, then trimmed back and returned.
        let mut items = std::mem::take(&mut self.prepared_surfaces);
        let prepared = items.len();
        let origin = self.origin;
        items.extend(
            frame
                .scene
                .volume_meshes
                .iter()
                .filter(|item| item.transparency.is_none())
                .map(|item| {
                    let mut item = item.to_render_item();
                    item.model.translation -= origin.origin();
                    item
                }),
        );
        self.prepared_surface_count = prepared;
        let cmd = self.render_frame_body(device, queue, output_view, vp_idx, frame, &items);
        items.truncate(prepared);
        self.prepared_surfaces = items;
        cmd
    }

    fn render_frame_body(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        vp_idx: usize,
        frame: &FrameData,
        scene_items: &[SceneRenderItem],
    ) -> crate::gpu::CommandBuffer {
        let paint_start = web_time::Instant::now();
        self.last_frame_kept_scene_depth = self.keep_scene_depth;
        // Reset the per-frame main-pass draw counters; the instanced draw loops
        // bump them through `&self` during encode and they are latched into
        // `FrameStats` after paint.
        self.frame_main_buffer_binds
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.frame_main_draw_commands
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.resources
            .frame_feature_sets_capped
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.resources
            .frame_stand_in_draws
            .store(0, std::sync::atomic::Ordering::Relaxed);

        let bg_colour = background_premultiplied(frame);
        let ppp = frame.camera.pixels_per_point;
        let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        // Ensure the per-viewport targets this frame draws into. The LDR path
        // needs its depth buffer and nothing else; the HDR path needs the
        // targets its planned passes read and write. A group
        // an earlier frame promoted stays live until it has gone unused for
        // `target_release_frames` frames.
        let ssaa_factor = frame.effects.post_process.ssaa_factor.max(1);
        let needed = if self.renders_hdr(frame) {
            self.ensure_viewport_slot(device, vp_idx);
            if self.has_transmissive_batches() {
                self.resources.ensure_transmission(device);
            }
            self.plan_hdr_passes(frame, scene_items, vp_idx, w.max(1), h.max(1), ssaa_factor);
            self.viewport_slots[vp_idx].pass_table.groups()
        } else {
            crate::resources::TargetGroups::LDR_DEPTH
        };
        if self.resources.scene_tier.is_built() {
            self.ensure_viewport_hdr(
                device,
                queue,
                vp_idx,
                w.max(1),
                h.max(1),
                ssaa_factor,
                self.current_render_scale,
                needed,
            );
            if self.foreground_active(frame) {
                self.viewport_slots[vp_idx].target_idle.use_foreground();
            }
            self.release_idle_viewport_targets(device, queue, vp_idx);
        } else {
            self.ensure_overlay_depth(device, vp_idx, w.max(1), h.max(1));
        }

        // Lazy-initialize GPU timestamp resources on first render call when supported.
        if self.ts_query_set.is_none()
            && device
                .features()
                .contains(crate::gpu::Features::TIMESTAMP_QUERY)
        {
            // One begin/end timestamp pair per measured pass. The resolve
            // buffer gives each slot its own 256-byte region because
            // `resolve_query_set` destination offsets must be 256-aligned and
            // written slots are resolved individually (see the resolve blocks).
            let ts_count = 2 * super::GPU_TS_SLOTS;
            let ts_bytes = super::GPU_TS_SLOTS as u64 * 256;
            self.ts_query_set = Some(device.create_query_set(&crate::gpu::QuerySetDescriptor {
                label: Some("ts_query_set"),
                ty: crate::gpu::QueryType::Timestamp,
                count: ts_count,
            }));
            // Second set of the double buffer: passes write one set while the
            // other (written last frame) is resolved from this frame's encoder.
            self.ts_query_set_prev =
                Some(device.create_query_set(&crate::gpu::QuerySetDescriptor {
                    label: Some("ts_query_set_prev"),
                    ty: crate::gpu::QueryType::Timestamp,
                    count: ts_count,
                }));
            self.ts_resolve_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("ts_resolve_buf"),
                size: ts_bytes,
                usage: crate::gpu::BufferUsages::QUERY_RESOLVE | crate::gpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.ts_staging_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("ts_staging_buf"),
                size: ts_bytes,
                usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            self.ts_period = queue.get_timestamp_period();
        }

        if frame.effects.display.is_hdr() && !self.renders_hdr(frame) {
            self.last_stats.hdr_drawn_as_ldr = true;
        }
        let cmd_buf = if !self.renders_hdr(frame) {
            // The LDR path binds the LDR mesh family. Prepare has normally
            // built it; this covers a frame prepared for the other family.
            if self.resources.scene_tier.is_built()
                && (!scene_items.is_empty() || Self::has_mesh_content(frame))
            {
                self.resources.ensure_ldr_mesh_pipelines(device);
            }
            self.render_frame_ldr(
                device,
                queue,
                output_view,
                vp_idx,
                frame,
                scene_items,
                bg_colour,
                w,
                h,
            )
        } else {
            // The HDR path builds the post-chain pipelines it binds itself,
            // group by group, as each frame asks for them.
            self.render_frame_hdr(
                device,
                queue,
                output_view,
                vp_idx,
                frame,
                scene_items,
                bg_colour,
                ssaa_factor,
            )
        };
        // CPU time spent encoding the paint pass (draw-call recording), separate
        // from prepare. Latched so last_frame_stats() reflects it after render.
        self.last_stats.cpu_paint_ms = paint_start.elapsed().as_secs_f32() * 1000.0;
        // Latch the main-pass geometry-bind and draw-command counters the
        // instanced draw loops accumulated during encode.
        self.last_stats.main_buffer_binds = self
            .frame_main_buffer_binds
            .load(std::sync::atomic::Ordering::Relaxed);
        self.last_stats.main_draw_commands = self
            .frame_main_draw_commands
            .load(std::sync::atomic::Ordering::Relaxed);
        // Pipelines compiled during the render phase (e.g. the shared HDR set on
        // the first HDR frame) land after prepare() snapshotted the counter; fold
        // them into this frame's stats rather than the next frame's.
        self.last_stats.pipelines_built_this_frame += self
            .resources
            .frame_pipelines_built
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        self.last_stats.pipelines_pending = self.resources.pipeline_compiler.pending() as u32;
        self.last_stats.feature_sets_capped = self
            .resources
            .frame_feature_sets_capped
            .load(std::sync::atomic::Ordering::Relaxed);
        self.last_stats.stand_in_draws = self
            .resources
            .frame_stand_in_draws
            .load(std::sync::atomic::Ordering::Relaxed);
        self.last_stats.feature_pipelines_built = self.resources.feature_sets() as u32;
        cmd_buf
    }

    /// Whether `frame` draws through the HDR path. A frame that asks for it on
    /// an overlay-only renderer draws through the LDR path instead, which is
    /// all the overlay family needs.
    pub(crate) fn renders_hdr(&self, frame: &FrameData) -> bool {
        frame.effects.display.is_hdr() && self.resources.scene_tier.is_built()
    }

    /// The depth store op for a pass that ends the frame on the scene depth
    /// (at render scale 1 the output depth is a view of it): kept when
    /// `keep_scene_depth` is set, discarded otherwise.
    pub(crate) fn scene_depth_store_op(&self) -> crate::gpu::StoreOp {
        if self.keep_scene_depth {
            crate::gpu::StoreOp::Store
        } else {
            crate::gpu::StoreOp::Discard
        }
    }

    /// Report that GPU plugins' `post_paint` was skipped this frame, once per
    /// reason (repeats are counted on the same notice).
    pub(crate) fn report_post_paint_skipped(&self, reason: &'static str) {
        self.resources.notices.report(
            crate::notice::NoticeKey::new(
                crate::NoticeCategory::Pipeline,
                "post_paint_skipped",
                [crate::notice::name_id(reason), 0],
            ),
            // A plugin that only uses `pre_prepare` lands here too: the runner
            // cannot tell whether a plugin implements `post_paint`.
            crate::Severity::Info,
            || crate::NoticeDetail::PostPaintSkipped { reason },
        );
    }

    /// Why `frame`, rendered into `output_view`, leaves no scene targets a
    /// GPU plugin can read, or `None` when it does.
    pub(crate) fn post_paint_skip_reason(
        &self,
        frame: &FrameData,
        output_view: Option<&crate::gpu::TextureView>,
    ) -> Option<&'static str> {
        if self.renders_hdr(frame) {
            return None;
        }
        let Some(output_view) = output_view else {
            return Some("the LDR frame was drawn into the host's render pass");
        };
        if !output_view
            .texture()
            .usage()
            .contains(crate::gpu::TextureUsages::TEXTURE_BINDING)
        {
            return Some("the LDR frame's output texture lacks TEXTURE_BINDING");
        }
        if self.current_render_scale < 1.0 - 0.001 {
            return Some("the LDR frame was drawn at a reduced render scale");
        }
        if self.ldr_overlay_intermediate(frame.camera.viewport_index, Some(output_view)) {
            return Some("the LDR frame drew its overlays into an intermediate target");
        }
        None
    }

    /// Views of the scene colour and depth `frame` was last rendered into, for
    /// sampling after the frame (a GPU plugin's `post_paint`).
    ///
    /// On an HDR frame these are the viewport's HDR colour target and its
    /// depth, the pair post effects read. On an LDR frame the colour is
    /// `output_view` and the depth is the viewport's LDR depth target. `None`
    /// when the viewport has not drawn; on an LDR frame also when
    /// `output_view` is missing or its texture lacks `TEXTURE_BINDING`, and
    /// when the frame drew into an intermediate target (dynamic resolution,
    /// or overlays that read the destination of an output that cannot be
    /// copied) instead of `output_view`.
    pub(crate) fn scene_target_views<'a>(
        &'a self,
        frame: &FrameData,
        output_view: Option<&'a crate::gpu::TextureView>,
    ) -> Option<crate::runtime::PostPaintTargets<'a>> {
        let vp_idx = frame.camera.viewport_index;
        let slot = self.viewport_slots.get(vp_idx)?;
        if self.renders_hdr(frame) {
            let hdr = slot.hdr.as_ref()?;
            return Some(crate::runtime::PostPaintTargets::new(
                &hdr.hdr_view,
                &hdr.hdr_depth_only_view,
                hdr.hdr_texture.format(),
            ));
        }
        let colour = output_view?;
        if !colour
            .texture()
            .usage()
            .contains(crate::gpu::TextureUsages::TEXTURE_BINDING)
        {
            return None;
        }
        if self.current_render_scale < 1.0 - 0.001
            || self.ldr_overlay_intermediate(vp_idx, Some(colour))
        {
            return None;
        }
        let depth = match &slot.hdr {
            Some(hdr) => &hdr.outline_depth_only_view,
            None => &slot.overlay_depth.as_ref()?.depth_only_view,
        };
        Some(crate::runtime::PostPaintTargets::new(
            colour,
            depth,
            self.resources.target_format,
        ))
    }

    /// Panic, naming both formats, when `output_view`'s texture cannot be drawn
    /// by pipelines compiled for the renderer's format. Without this the first
    /// pass to bind a pipeline fails wgpu validation with a message about that
    /// pass, not about the target.
    ///
    /// A view does not expose its own format, only its texture's, so one
    /// reinterpretation is allowed: a linear texture rendered through an sRGB
    /// view by a renderer created for the sRGB format.
    fn check_output_format(&self, output_view: &crate::gpu::TextureView) {
        let texture = output_view.texture().format();
        let target = self.resources.target_format;
        if texture != target && texture.add_srgb_suffix() != target {
            panic!(
                "the output view's texture is {texture:?}, but this renderer was \
                 created for {target:?} and every pipeline is compiled for that format. Create \
                 the renderer with the format of the target it draws into, or draw into a \
                 texture of the renderer's format."
            );
        }
    }

    /// Render a frame into `output_view` and submit it, without reading anything
    /// back. `output_view` must be a `RENDER_ATTACHMENT` view in the format the
    /// renderer was created with, sized to `frame.camera.viewport_size`.
    ///
    /// # Panics
    ///
    /// When `output_view`'s texture is not in the renderer's format (a linear
    /// texture viewed as the renderer's sRGB format is allowed).
    ///
    /// Unlike [`render_offscreen`](Self::render_offscreen), this neither copies
    /// the result to the CPU nor blocks on the GPU, so the caller can reuse one
    /// target and drive many frames back to back. Intended for headless loops
    /// (perf measurement, capture pipelines) that read results back through GPU
    /// timestamps or their own copy on a cadence rather than every frame.
    pub fn render_to_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        frame: &FrameData,
    ) {
        let cmd_buf = self.render(device, queue, output_view, frame);
        queue.submit(std::iter::once(cmd_buf));
    }

    /// Encode a full frame (prepare passes + scene pass) into command buffers
    /// without submitting any of them: the deferred-submit counterpart of
    /// [`render_to_texture`](Self::render_to_texture). Buffers are returned in
    /// submission order (prepare first, scene pass last).
    ///
    /// A render worker can call this off the device-driving thread, then hand the
    /// buffers back for the driving thread to submit via
    /// [`submit_frame`](Self::submit_frame). Submitting from a non-driving thread
    /// corrupts some drivers (see [`crate::renderer::SubmitSink`]).
    pub fn render_to_texture_deferred(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        frame: &FrameData,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.render_deferred(device, queue, output_view, frame)
    }

    /// Submit a batch of command buffers produced by the `*_deferred` encoders,
    /// in order, on the device-driving thread. A no-op for an empty batch.
    pub fn submit_frame(queue: &crate::gpu::Queue, buffers: Vec<crate::gpu::CommandBuffer>) {
        if !buffers.is_empty() {
            queue.submit(buffers);
        }
    }

    /// Render a frame to an offscreen texture and return raw RGBA bytes.
    ///
    /// Creates a temporary [`wgpu::Texture`] render target of the given dimensions,
    /// runs all render passes (shadow, scene, post-processing) into it via
    /// [`OwnedPath::render`](crate::OwnedPath::render), then copies the result back to CPU memory.
    ///
    /// No OS window or [`wgpu::Surface`] is required. The caller is responsible for
    /// initialising the wgpu adapter with `compatible_surface: None` and for
    /// constructing a valid [`FrameData`] (including `viewport_size` matching
    /// `width`/`height`).
    ///
    /// Returns `width * height * 4` bytes in RGBA8 layout. The caller encodes to
    /// PNG/EXR independently : no image codec dependency in this crate. Returns
    /// an empty vec once the device is lost.
    ///
    /// This is a *presented* render: it advances per-frame state (it pumps the
    /// upload pipeline, bumps the frame counter, stores HiZ prev-depth, and
    /// writes `FrameStats`), because a headless app commonly uses it as its only
    /// render and relies on uploads promoting here. If instead you render a
    /// preview / thumbnail *alongside* a main presented frame and do not want it
    /// to touch that frame's state, use
    /// [`render_offscreen_snapshot`](Self::render_offscreen_snapshot).
    ///
    /// A frame that leaves `FrameData::time` at `None` renders at
    /// `FrameTime::fixed(0.0)` here, not the renderer's clock, so two captures
    /// of one frame match.
    pub fn render_offscreen(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        if self.skip_frame_device_lost() {
            return Vec::new();
        }
        // Hands the result back, so nothing may be skipped while it compiles.
        let _blocking = self.resources.pipeline_compiler.blocking_scope();
        // 1. Create offscreen texture with RENDER_ATTACHMENT | COPY_SRC usage.
        let target_format = self.resources.target_format;
        let offscreen_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("offscreen_target"),
            size: crate::gpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: target_format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // 2. Create a texture view for rendering into.
        let output_view =
            offscreen_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        // 3. render() calls ensure_viewport_hdr which provides the depth-stencil buffer
        //    for both LDR and HDR paths, so no separate ensure_outline_target is needed.

        // 4. Render the scene into the offscreen texture.
        //    The caller must set `frame.camera.viewport_size` to `[width as f32, height as f32]`
        //    and build `frame.camera.render_camera.projection` with aspect `width / height`
        //    for correct HDR target allocation and scissor rects.
        //    A frame without its own time renders at `FrameTime::fixed(0.0)`
        //    so captures are reproducible.
        let prev_fixed = self.frame_clock.set_fixed_fallback(true);
        let cmd_buf = self.render(device, queue, &output_view, frame);
        self.frame_clock.set_fixed_fallback(prev_fixed);
        queue.submit(std::iter::once(cmd_buf));

        // 5. Copy texture -> staging buffer (wgpu requires row alignment to 256 bytes).
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = crate::gpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) & !(align - 1);
        let buffer_size = (padded_row * height.max(1)) as u64;

        let staging_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("offscreen_staging"),
            size: buffer_size,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut copy_encoder =
            device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                label: Some("offscreen_copy_encoder"),
            });
        copy_encoder.copy_texture_to_buffer(
            crate::gpu::TexelCopyTextureInfo {
                texture: &offscreen_texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyBufferInfo {
                buffer: &staging_buf,
                layout: crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height.max(1)),
                },
            },
            crate::gpu::Extent3d {
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
            .map_async(crate::gpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(crate::gpu::PollType::wait_indefinitely())
            .unwrap();
        let _ = rx.recv().unwrap_or(Err(crate::gpu::BufferAsyncError));

        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        {
            let mapped = crate::gpu::mapped_range(staging_buf.slice(..));
            let data: &[u8] = &mapped;
            if padded_row == unpadded_row {
                // No padding : copy entire slice directly.
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
            crate::gpu::TextureFormat::Bgra8Unorm | crate::gpu::TextureFormat::Bgra8UnormSrgb
        );
        if is_bgra {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2); // B <-> R
            }
        }

        pixels
    }

    /// [`render_offscreen`](Self::render_offscreen) as an auxiliary (snapshot)
    /// render: it reads the currently resident scene without advancing shared
    /// per-frame state, so it does not pump the upload pipeline, bump the frame
    /// counter, store HiZ prev-depth, clobber `FrameStats`, or run item-type
    /// plugins.
    ///
    /// Use this for a preview / thumbnail rendered alongside a main presented
    /// frame, where the offscreen render must not perturb that frame's timeline.
    /// Because it does not pump uploads, the scene must already be resident (gate
    /// on [`frame_fully_resident`](Self::frame_fully_resident) if unsure);
    /// item-type plugin geometry is absent, as in a probe bake. If this is your
    /// only render, use [`render_offscreen`](Self::render_offscreen) instead so
    /// uploads promote.
    pub fn render_offscreen_snapshot(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        // Hands the result back, so nothing may be skipped while it compiles.
        let _blocking = self.resources.pipeline_compiler.blocking_scope();
        let saved_mode = self.render_mode;
        let saved_stats = self.last_stats;
        self.render_mode = crate::renderer::RenderMode::Derivative;
        let pixels = self.render_offscreen(device, queue, frame, width, height);
        self.render_mode = saved_mode;
        self.last_stats = saved_stats;
        pixels
    }

    // ------------------------------------------------------------------
    // Overlay destination reads
    // ------------------------------------------------------------------

    /// Ensure the LDR intermediate target exists at the right size.
    fn ensure_backdrop_blur_state(&mut self, device: &crate::gpu::Device, w: u32, h: u32) {
        let need_recreate = match &self.backdrop_blur_state {
            Some(s) => s.size != [w, h] || s.format != self.resources.target_format,
            None => true,
        };
        if !need_recreate {
            return;
        }
        let format = self.resources.target_format;
        let intermediate_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("backdrop_intermediate"),
            size: crate::gpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING
                | crate::gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let intermediate_view = intermediate_texture.create_view(&Default::default());
        self.backdrop_blur_state = Some(crate::resources::BackdropBlurState {
            intermediate_texture,
            intermediate_view,
            size: [w, h],
            format,
            blit: Default::default(),
        });
    }

    /// Whether the LDR path draws `vp_idx`'s frame into its intermediate
    /// target: its overlays read the destination, and `output` cannot be
    /// copied from at a break.
    fn ldr_overlay_intermediate(
        &self,
        vp_idx: usize,
        output: Option<&crate::gpu::TextureView>,
    ) -> bool {
        let Some(slot) = self.viewport_slots.get(vp_idx) else {
            return false;
        };
        let ov = &slot.overlay;
        ov.has_breaks()
            && self.current_render_scale >= 1.0 - 0.001
            && !output.is_some_and(|o| ov.layer_targets.can_snapshot(&o.texture()))
    }

    /// Draw `vp_idx`'s overlay list into `target` in passes of its own.
    ///
    /// When the target can be copied, the pass ends at each break: what is
    /// drawn so far is copied to the snapshot (and blurred for backdrops)
    /// before the items that read it. Otherwise the list draws in one pass,
    /// layers with their fallback blend, and backdrops over `scene` blurred
    /// (the scene without the overlays) when it is given, or not at all.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw_overlay_passes(
        &mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        device: &crate::gpu::Device,
        vp_idx: usize,
        target: &crate::gpu::TextureView,
        depth: &crate::gpu::TextureView,
        depth_store: crate::gpu::StoreOp,
        scene: Option<&crate::gpu::TextureView>,
        ppp: f32,
        label: &'static str,
    ) {
        let target_texture = target.texture();
        let can_copy = self.viewport_slots[vp_idx]
            .overlay
            .layer_targets
            .can_snapshot(&target_texture);
        // Without a copy, backdrops draw over the scene blurred on its own.
        let blurred = !can_copy
            && match scene {
                Some(src) if self.viewport_slots[vp_idx].overlay.has_backdrops() => self
                    .viewport_slots[vp_idx]
                    .overlay
                    .layer_targets
                    .blur_scene(encoder, device, &self.resources, src),
                _ => false,
            };
        let mut ts = self.ts_writes_for(crate::renderer::GPU_TS_OVERLAY, true, true);
        fn begin<'e>(
            encoder: &'e mut crate::gpu::CommandEncoder,
            label: &'static str,
            target: &crate::gpu::TextureView,
            depth: &crate::gpu::TextureView,
            ts: Option<crate::gpu::RenderPassTimestampWrites<'_>>,
            store: crate::gpu::StoreOp,
        ) -> crate::gpu::RenderPass<'e> {
            encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some(label),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Load,
                        store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: ts,
                occlusion_query_set: None,
            })
        }
        if !can_copy {
            let ov = &self.viewport_slots[vp_idx].overlay;
            let mut pass = begin(encoder, label, target, depth, ts.take(), depth_store);
            emit_overlay_segments(
                &self.resources,
                ov,
                &mut pass,
                ov.main_segments(),
                &OverlayEmit {
                    dst: false,
                    backdrop: if blurred {
                        ov.layer_targets.item_backdrop_bg()
                    } else {
                        None
                    },
                },
            );
            return;
        }
        let ov = &self.viewport_slots[vp_idx].overlay;
        let segs = ov.main_segments();
        let breaks = &ov.overlay_breaks;
        let emit = OverlayEmit {
            dst: true,
            backdrop: ov.layer_targets.item_backdrop_bg(),
        };
        let mut start = 0usize;
        for k in 0..=breaks.len() {
            let end = breaks.get(k).map_or(segs.len(), |b| b.segment as usize);
            let last = k == breaks.len();
            if end > start || last {
                let store = if last {
                    depth_store
                } else {
                    crate::gpu::StoreOp::Store
                };
                let mut pass = begin(encoder, label, target, depth, ts.take(), store);
                emit_overlay_segments(&self.resources, ov, &mut pass, &segs[start..end], &emit);
            }
            if let Some(b) = breaks.get(k) {
                let region = ov.layer_targets.break_region(b.bounds, b.blur, ppp);
                ov.layer_targets.snapshot(
                    encoder,
                    &self.resources,
                    &target_texture,
                    region,
                    b.blur,
                );
            }
            start = end;
        }
    }
}

impl crate::renderer::ViewportRenderer {
    /// Draw the mesh-instance batches from the per-frame GPU data `prepare()`
    /// built.
    ///
    /// Core rather than an item-type plugin, and deliberately so: a
    /// mesh-instance batch rides the scene graph's own instanced pipeline.
    /// Lines, the other layer that used to draw from here, belong to the
    /// polyline item type, which also draws the other types' wireframe
    /// overlays.
    pub(crate) fn draw_mesh_instance_batches(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'_>,
        camera_bg: &crate::gpu::BindGroup,
    ) {
        // Mesh-instance pass: one draw call per host-built batch, routed by
        // blend mode, through the HDR mesh family's blended members.
        if !self.mesh_instance_gpu_data.is_empty() {
            use crate::renderer::SpriteBlend;
            for blend in [
                SpriteBlend::AlphaBlend,
                SpriteBlend::Additive,
                SpriteBlend::Premultiplied,
            ] {
                // Resolved by the first batch that needs it, so a blend mode
                // nothing uses never compiles.
                let mut pipeline = None;
                for batch in self.mesh_instance_gpu_data.iter() {
                    if batch.blend != blend {
                        continue;
                    }
                    let Some(mesh) = self.resources.mesh_store.get(batch.mesh_id) else {
                        continue;
                    };
                    if mesh.index_count == 0 {
                        continue;
                    }
                    if pipeline.is_none() {
                        let Some(pl) = self.resources.scene.hdr_blend(blend) else {
                            // Still compiling: this blend mode waits a frame.
                            break;
                        };
                        render_pass.set_pipeline(pl);
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        pipeline = Some(pl);
                    }
                    render_pass.set_bind_group(1, &batch.bind_group, &[]);
                    render_pass.set_vertex_buffer(1, batch.instance_buf.slice(..));
                    render_pass.set_vertex_buffer(
                        0,
                        self.resources.geometry.vertex_slice(mesh.vertex_span),
                    );
                    render_pass.set_index_buffer(
                        self.resources.geometry.index_slice(mesh.index_span),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..batch.instance_count);
                }
            }
        }
    }
}
