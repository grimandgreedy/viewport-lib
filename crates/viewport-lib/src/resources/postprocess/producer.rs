//! Composite-input producers: the pre-tone-map effects that compute an
//! offscreen texture the composite samples.
//!
//! Each producer owns its shared GPU state (pipelines, layouts, static
//! resources) and a per-viewport state struct on `ViewportHdrState`, and
//! implements [`PostProducer`]: a settings gate, a per-frame uniform upload,
//! and pass encoding at the post-effects position in the HDR frame. The
//! frame loop drives every producer through the same three calls, so adding
//! an effect of this shape means adding a producer, not editing the loop.

use crate::resources::ViewportHdrState;

/// Per-frame inputs a producer may read: the active settings plus the camera
/// and lighting data the uniform uploads derive from.
pub(crate) struct ProducerFrameInputs<'a> {
    pub(crate) post: &'a crate::PostProcessSettings,
    pub(crate) proj: glam::Mat4,
    pub(crate) view: glam::Mat4,
    pub(crate) first_light: Option<&'a crate::LightSource>,
}

/// GPU timestamp plumbing for producer passes, mirroring the renderer's
/// per-slot begin/end pairs.
pub(crate) struct ProducerTiming<'a> {
    pub(crate) query_set: Option<&'a crate::gpu::QuerySet>,
    pub(crate) written_mask: &'a std::sync::atomic::AtomicU32,
}

impl ProducerTiming<'_> {
    /// Timestamp writes for one measured pass. `begin` and `end` select which
    /// boundary of the slot's begin/end pair this pass writes, so a
    /// multi-pass producer can begin on its first pass and end on its last
    /// (each query index must be written at most once per frame).
    pub(crate) fn writes(
        &self,
        slot: u32,
        begin: bool,
        end: bool,
    ) -> Option<crate::gpu::RenderPassTimestampWrites<'_>> {
        self.query_set.map(|qs| {
            self.written_mask
                .fetch_or(1 << slot, std::sync::atomic::Ordering::Relaxed);
            crate::gpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: begin.then_some(slot * 2),
                end_of_pass_write_index: end.then_some(slot * 2 + 1),
            }
        })
    }
}

/// Encode one fullscreen-triangle pass: clear the target, bind one group,
/// draw three vertices.
pub(crate) fn fullscreen_pass(
    encoder: &mut crate::gpu::CommandEncoder,
    label: &str,
    view: &crate::gpu::TextureView,
    clear: crate::gpu::Color,
    pipeline: &crate::gpu::RenderPipeline,
    bind_group: &crate::gpu::BindGroup,
    timestamp_writes: Option<crate::gpu::RenderPassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
        #[cfg(any(wgpu29, wgpu30))]
        multiview_mask: None,
        label: Some(label),
        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: crate::gpu::Operations {
                load: crate::gpu::LoadOp::Clear(clear),
                store: crate::gpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes,
        occlusion_query_set: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// A pre-tone-map effect that fills a composite input slot.
pub(crate) trait PostProducer {
    /// Whether the effect is switched on this frame (settings only; the
    /// degradation throttle is applied by the caller at encode time).
    fn enabled(&self, inputs: &ProducerFrameInputs<'_>) -> bool;
    /// Write this frame's uniforms into the viewport's buffers.
    fn upload(
        &self,
        queue: &crate::gpu::Queue,
        hdr: &ViewportHdrState,
        inputs: &ProducerFrameInputs<'_>,
    );
    /// Encode the producer's passes into the frame encoder.
    fn encode(
        &self,
        hdr: &ViewportHdrState,
        encoder: &mut crate::gpu::CommandEncoder,
        timing: &ProducerTiming<'_>,
    );
}

// --- SSAO ---

/// Shared SSAO state: occlusion + blur pipelines, their layouts, and the
/// static noise texture and hemisphere kernel.
#[derive(Default)]
pub(crate) struct SsaoProducer {
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) blur_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) blur_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) noise_texture: Option<crate::gpu::Texture>,
    pub(crate) noise_view: Option<crate::gpu::TextureView>,
    pub(crate) kernel_buf: Option<crate::gpu::Buffer>,
}

/// Per-viewport SSAO state: the occlusion and blur targets plus their bind
/// groups and the uniform buffer.
// Texture fields keep the GPU allocations alive; the passes bind the views.
#[allow(dead_code)]
pub(crate) struct SsaoViewport {
    pub(crate) texture: crate::gpu::Texture,
    pub(crate) view: crate::gpu::TextureView,
    pub(crate) blur_texture: crate::gpu::Texture,
    pub(crate) blur_view: crate::gpu::TextureView,
    pub(crate) bg: crate::gpu::BindGroup,
    pub(crate) blur_bg: crate::gpu::BindGroup,
    pub(crate) uniform_buf: crate::gpu::Buffer,
}

impl PostProducer for SsaoProducer {
    fn enabled(&self, inputs: &ProducerFrameInputs<'_>) -> bool {
        inputs.post.ssao
    }

    fn upload(
        &self,
        queue: &crate::gpu::Queue,
        hdr: &ViewportHdrState,
        inputs: &ProducerFrameInputs<'_>,
    ) {
        let uniform = super::uniforms::SsaoUniform {
            inv_proj: inputs.proj.inverse().to_cols_array_2d(),
            proj: inputs.proj.to_cols_array_2d(),
            radius: 0.5,
            bias: 0.025,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&hdr.ssao.uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
    }

    fn encode(
        &self,
        hdr: &ViewportHdrState,
        encoder: &mut crate::gpu::CommandEncoder,
        timing: &ProducerTiming<'_>,
    ) {
        let Some(pipeline) = &self.pipeline else {
            return;
        };
        // The SSAO slot begins on the occlusion pass and ends on the blur
        // pass (or on the occlusion pass when there is no blur).
        let has_blur = self.blur_pipeline.is_some();
        fullscreen_pass(
            encoder,
            "ssao_pass",
            &hdr.ssao.view,
            crate::gpu::Color::WHITE,
            pipeline,
            &hdr.ssao.bg,
            timing.writes(crate::renderer::GPU_TS_SSAO, true, !has_blur),
        );
        if let Some(blur_pipeline) = &self.blur_pipeline {
            fullscreen_pass(
                encoder,
                "ssao_blur_pass",
                &hdr.ssao.blur_view,
                crate::gpu::Color::WHITE,
                blur_pipeline,
                &hdr.ssao.blur_bg,
                timing.writes(crate::renderer::GPU_TS_SSAO, false, true),
            );
        }
    }
}

// --- Contact shadows ---

/// Shared contact-shadow state: the screen-space march pipeline + layout.
#[derive(Default)]
pub(crate) struct ContactShadowProducer {
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
}

/// Per-viewport contact-shadow state.
// The texture field keeps the GPU allocation alive; the pass binds the view.
#[allow(dead_code)]
pub(crate) struct ContactShadowViewport {
    pub(crate) texture: crate::gpu::Texture,
    pub(crate) view: crate::gpu::TextureView,
    pub(crate) bg: crate::gpu::BindGroup,
    pub(crate) uniform_buf: crate::gpu::Buffer,
}

impl PostProducer for ContactShadowProducer {
    fn enabled(&self, inputs: &ProducerFrameInputs<'_>) -> bool {
        inputs.post.contact_shadows.enabled
    }

    fn upload(
        &self,
        queue: &crate::gpu::Queue,
        hdr: &ViewportHdrState,
        inputs: &ProducerFrameInputs<'_>,
    ) {
        let settings = &inputs.post.contact_shadows;
        let light_dir_world: glam::Vec3 = if let Some(l) = inputs.first_light {
            match l.kind {
                crate::LightKind::Directional { direction } => {
                    glam::Vec3::from(direction).normalize()
                }
                crate::LightKind::Spot { direction, .. } => {
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
        let light_dir_view = inputs.view.transform_vector3(light_dir_world).normalize();
        let world_up_view = inputs.view.transform_vector3(glam::Vec3::Z).normalize();
        let uniform = super::uniforms::ContactShadowUniform {
            inv_proj: inputs.proj.inverse().to_cols_array_2d(),
            proj: inputs.proj.to_cols_array_2d(),
            light_dir_view: [light_dir_view.x, light_dir_view.y, light_dir_view.z, 0.0],
            world_up_view: [world_up_view.x, world_up_view.y, world_up_view.z, 0.0],
            params: [
                settings.max_distance,
                settings.steps as f32,
                settings.thickness,
                0.0,
            ],
        };
        queue.write_buffer(
            &hdr.contact_shadow.uniform_buf,
            0,
            bytemuck::cast_slice(&[uniform]),
        );
    }

    fn encode(
        &self,
        hdr: &ViewportHdrState,
        encoder: &mut crate::gpu::CommandEncoder,
        _timing: &ProducerTiming<'_>,
    ) {
        let Some(pipeline) = &self.pipeline else {
            return;
        };
        fullscreen_pass(
            encoder,
            "contact_shadow_pass",
            &hdr.contact_shadow.view,
            crate::gpu::Color::WHITE,
            pipeline,
            &hdr.contact_shadow.bg,
            None,
        );
    }
}

// --- Bloom ---

/// Shared bloom state: the threshold and blur pipelines and their common
/// layout.
#[derive(Default)]
pub(crate) struct BloomProducer {
    pub(crate) threshold_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) blur_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
}

/// Per-viewport bloom state: the scene-resolution threshold target, the
/// half-resolution ping/pong pair, the four blur bind groups, and the
/// threshold + constant H/V uniform buffers.
// Texture fields keep the GPU allocations alive; the passes bind the views.
#[allow(dead_code)]
pub(crate) struct BloomViewport {
    pub(crate) threshold_texture: crate::gpu::Texture,
    pub(crate) threshold_view: crate::gpu::TextureView,
    pub(crate) ping_texture: crate::gpu::Texture,
    pub(crate) ping_view: crate::gpu::TextureView,
    pub(crate) pong_texture: crate::gpu::Texture,
    pub(crate) pong_view: crate::gpu::TextureView,
    pub(crate) threshold_bg: crate::gpu::BindGroup,
    /// H-blur bind group that reads from the threshold target (pass 0 only).
    pub(crate) blur_h_bg: crate::gpu::BindGroup,
    /// V-blur bind group that reads from ping.
    pub(crate) blur_v_bg: crate::gpu::BindGroup,
    /// H-blur bind group that reads from pong (passes 1+).
    pub(crate) blur_h_pong_bg: crate::gpu::BindGroup,
    pub(crate) uniform_buf: crate::gpu::Buffer,
    /// Constant H-blur uniform buffer (horizontal=1, written once at creation).
    pub(crate) h_uniform_buf: crate::gpu::Buffer,
    /// Constant V-blur uniform buffer (horizontal=0, written once at creation).
    pub(crate) v_uniform_buf: crate::gpu::Buffer,
}

impl PostProducer for BloomProducer {
    fn enabled(&self, inputs: &ProducerFrameInputs<'_>) -> bool {
        inputs.post.bloom.enabled
    }

    fn upload(
        &self,
        queue: &crate::gpu::Queue,
        hdr: &ViewportHdrState,
        inputs: &ProducerFrameInputs<'_>,
    ) {
        let settings = &inputs.post.bloom;
        let uniform = super::uniforms::BloomUniform {
            threshold: settings.threshold,
            intensity: settings.intensity,
            horizontal: 0,
            max_brightness: settings.max_brightness,
        };
        queue.write_buffer(&hdr.bloom.uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
    }

    fn encode(
        &self,
        hdr: &ViewportHdrState,
        encoder: &mut crate::gpu::CommandEncoder,
        timing: &ProducerTiming<'_>,
    ) {
        let Some(threshold_pipeline) = &self.threshold_pipeline else {
            return;
        };
        // The bloom slot begins on the threshold pass and ends on the last
        // blur pass (or on the threshold pass when there is no blur).
        let has_blur = self.blur_pipeline.is_some();
        fullscreen_pass(
            encoder,
            "bloom_threshold_pass",
            &hdr.bloom.threshold_view,
            crate::gpu::Color::BLACK,
            threshold_pipeline,
            &hdr.bloom.threshold_bg,
            timing.writes(crate::renderer::GPU_TS_BLOOM, true, !has_blur),
        );

        // 4 ping-pong H+V blur passes for a wide glow.
        // Pass 1: threshold -> ping -> pong. Passes 2-4: pong -> ping -> pong.
        if let Some(blur_pipeline) = &self.blur_pipeline {
            const BLUR_ITERATIONS: usize = 4;
            for i in 0..BLUR_ITERATIONS {
                // H pass: pass 0 reads threshold, subsequent passes read pong.
                let h_bg = if i == 0 {
                    &hdr.bloom.blur_h_bg
                } else {
                    &hdr.bloom.blur_h_pong_bg
                };
                fullscreen_pass(
                    encoder,
                    "bloom_blur_h_pass",
                    &hdr.bloom.ping_view,
                    crate::gpu::Color::BLACK,
                    blur_pipeline,
                    h_bg,
                    None,
                );
                // V pass: ping -> pong. The last iteration closes the bloom
                // timing slot.
                let ts = (i == BLUR_ITERATIONS - 1)
                    .then(|| timing.writes(crate::renderer::GPU_TS_BLOOM, false, true))
                    .flatten();
                fullscreen_pass(
                    encoder,
                    "bloom_blur_v_pass",
                    &hdr.bloom.pong_view,
                    crate::gpu::Color::BLACK,
                    blur_pipeline,
                    &hdr.bloom.blur_v_bg,
                    ts,
                );
            }
        }
    }
}
