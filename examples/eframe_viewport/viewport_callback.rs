//! egui wgpu paint callback - thin adapter that delegates to `ViewportRenderer`.
//!
//! This implements `eframe::egui_wgpu::CallbackTrait` so that the viewport
//! renderer can be invoked from an egui paint callback. The pattern is:
//!   1. Register `ViewportRenderer` in `CallbackResources` at startup
//!   2. Each frame, create a `ViewportCallback` with the current `FrameData`
//!   3. egui calls `prepare()` then `paint()` during its render pass

use viewport_lib::{FrameData, ViewportRenderer};

pub struct ViewportCallback {
    pub frame: FrameData,
}

impl eframe::egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder,
        callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        if let Some(renderer) = callback_resources.get_mut::<ViewportRenderer>() {
            renderer.prepare(device, queue, &self.frame);
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
            renderer.paint(render_pass, &self.frame);
        }
    }
}
