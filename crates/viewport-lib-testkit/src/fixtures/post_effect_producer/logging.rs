//! A post-effect producer that records its lifecycle and can fill its slot
//! with a flat value.

use crate::fixtures::CallLog;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use viewport_lib::wgpu;
use viewport_lib::{
    PostEffectContext, PostEffectProducer, PostEffectResizeContext, PostEffectSlot,
};

/// Records `init_gpu`, `on_viewport_resized` (with the viewport index and
/// scene size), `prepare` and `encode`, and returns either nothing (leaving
/// the slot to the built-in effect) or its own texture cleared to a flat
/// value.
///
/// The enable gate is shared through an `Arc<AtomicBool>` so a test can flip
/// it between frames and assert the self-gate skips `prepare` and `encode`.
pub struct LoggingPostEffectProducer {
    log: CallLog,
    slot: PostEffectSlot,
    enabled: Arc<AtomicBool>,
    /// `Some(value)` clears the producer's own texture to that value each
    /// frame and returns it for the slot; `None` returns no view.
    fill: Option<f32>,
    textures: HashMap<usize, (wgpu::Texture, wgpu::TextureView)>,
}

impl LoggingPostEffectProducer {
    /// A producer for `slot` that logs but contributes no view, so the
    /// built-in effect keeps the slot.
    pub fn new(log: CallLog, slot: PostEffectSlot, enabled: Arc<AtomicBool>) -> Self {
        Self {
            log,
            slot,
            enabled,
            fill: None,
            textures: HashMap::new(),
        }
    }

    /// Also fill the slot: allocate a per-viewport texture and clear it to
    /// `value` each frame, returning it from `encode`.
    pub fn filling(mut self, value: f32) -> Self {
        self.fill = Some(value);
        self
    }

    /// The single-channel format the fill texture uses. The composite accepts
    /// any filterable float format; `R8Unorm` matches the built-ins for the
    /// single-channel slots.
    fn fill_format(&self) -> wgpu::TextureFormat {
        match self.slot {
            PostEffectSlot::Bloom => wgpu::TextureFormat::Rgba16Float,
            _ => wgpu::TextureFormat::R8Unorm,
        }
    }
}

impl PostEffectProducer for LoggingPostEffectProducer {
    fn type_name(&self) -> &'static str {
        "logging_post_effect_producer"
    }

    fn slot(&self) -> PostEffectSlot {
        self.slot
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn init_gpu(&mut self, _device: &wgpu::Device) {
        self.log.record("init_gpu");
    }

    fn on_device_recreated(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        self.log.record("on_device_recreated");
        self.textures.clear();
    }

    fn on_viewport_resized(&mut self, device: &wgpu::Device, ctx: &PostEffectResizeContext<'_>) {
        self.log.record(format!(
            "resize:{}:{}x{}",
            ctx.viewport_index, ctx.scene_size[0], ctx.scene_size[1]
        ));
        if self.fill.is_none() {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("logging_post_effect_producer_fill"),
            size: wgpu::Extent3d {
                width: ctx.scene_size[0],
                height: ctx.scene_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.fill_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.textures.insert(ctx.viewport_index, (texture, view));
    }

    fn prepare(&mut self, _queue: &wgpu::Queue, ctx: &PostEffectContext<'_>) {
        self.log.record(format!("prepare:{}", ctx.viewport_index));
    }

    fn encode<'a>(
        &'a mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PostEffectContext<'_>,
    ) -> Option<&'a wgpu::TextureView> {
        self.log.record(format!(
            "encode:{}:{}x{}",
            ctx.viewport_index, ctx.scene_size[0], ctx.scene_size[1]
        ));
        let fill = self.fill?;
        let (_, view) = self.textures.get(&ctx.viewport_index)?;
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("logging_post_effect_producer_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: fill as f64,
                        g: fill as f64,
                        b: fill as f64,
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
        Some(view)
    }
}
