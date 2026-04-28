use std::sync::{Arc, Mutex};
use viewport_lib::{FrameData, ViewportRenderer};

pub struct ViewportCallback {
    pub frame: FrameData,
    /// Cursor position to GPU-pick at this frame, set on click.
    pub pick_cursor: Option<glam::Vec2>,
    /// Shared slot: prepare writes the raw pick_id (0 = miss) after GPU pick.
    pub pick_result: Arc<Mutex<Option<u64>>>,
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

            if let Some(cursor) = self.pick_cursor {
                let hit = renderer.pick_scene_gpu(device, queue, cursor, &self.frame);
                let id = hit.map(|h| h.object_id.0).unwrap_or(0);
                if let Ok(mut slot) = self.pick_result.lock() {
                    *slot = Some(id);
                }
            }
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
