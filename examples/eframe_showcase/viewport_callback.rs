//! egui wgpu paint callback : thin adapter that delegates to `ViewportRenderer`.

use std::sync::atomic::{AtomicBool, Ordering};

use viewport_lib::{FrameData, ViewportRenderer};

pub struct ViewportCallback {
    pub frame: FrameData,
    /// Set by `prepare` to indicate whether dyn-res was active this frame.
    /// `paint` reads it to decide whether to blit or paint directly.
    dyn_res_active: AtomicBool,
}

impl ViewportCallback {
    pub fn new(frame: FrameData) -> Self {
        Self { frame, dyn_res_active: AtomicBool::new(false) }
    }
}

impl eframe::egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut eframe::wgpu::CommandEncoder,
        callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        if let Some(renderer) = callback_resources.get_mut::<ViewportRenderer>() {
            renderer.prepare(device, queue, &self.frame);
            let used = renderer.prepare_ldr_dyn_res(egui_encoder, device, &self.frame);
            self.dyn_res_active.store(used, Ordering::Relaxed);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = callback_resources.get::<ViewportRenderer>() {
            if self.dyn_res_active.load(Ordering::Relaxed) {
                renderer.paint_dyn_res_blit(render_pass, &self.frame);
            } else {
                renderer.paint(render_pass, &self.frame);
            }
        }
    }
}
