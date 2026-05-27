use viewport_lib::{FrameData, ViewportRenderer};

pub struct ViewportCallback {
    pub frame: FrameData,
    pub instancing_status: std::sync::Arc<std::sync::Mutex<(bool, usize)>>,
    pub pixel_read_req: std::sync::Arc<std::sync::Mutex<Option<(u32, u32)>>>,
    pub pixel_read_res: std::sync::Arc<std::sync::Mutex<Option<[f32; 4]>>>,
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
            // Service any pending pixel readback from the previous frame.
            if let Ok(mut req) = self.pixel_read_req.lock() {
                if let Some((x, y)) = req.take() {
                    let values = renderer.read_debug_pixel(device, queue, x, y);
                    if let Ok(mut res) = self.pixel_read_res.lock() {
                        *res = values;
                    }
                }
            }

            let cmds = renderer.pass().prepare(device, queue, &self.frame);
            if let Ok(mut status) = self.instancing_status.lock() {
                *status = (renderer.is_using_instanced_path(), renderer.instanced_batch_count());
            }
            return cmds;
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
            renderer.pass_view().paint(render_pass, &self.frame);
        }
    }
}
