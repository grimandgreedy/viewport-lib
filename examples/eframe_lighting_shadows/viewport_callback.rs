use viewport_lib::{ExposureReadback, FrameData, ShadowDebugStats, ViewportId, ViewportRenderer};

pub struct ViewportCallback {
    pub frame: FrameData,
    pub vp_id: ViewportId,
    /// Whether to read back the metered EV this frame (auto-exposure only; the
    /// readback does a blocking device poll, so it is gated).
    pub read_exposure: bool,
    pub instancing_status: std::sync::Arc<std::sync::Mutex<(bool, usize)>>,
    pub pixel_read_req: std::sync::Arc<std::sync::Mutex<Option<(u32, u32)>>>,
    pub pixel_read_res: std::sync::Arc<std::sync::Mutex<Option<[f32; 4]>>>,
    pub shadow_stats: std::sync::Arc<std::sync::Mutex<Option<ShadowDebugStats>>>,
    pub exposure_readback: std::sync::Arc<std::sync::Mutex<Option<ExposureReadback>>>,
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
                *status = (
                    renderer.is_using_instanced_path(),
                    renderer.instanced_batch_count(),
                );
            }
            if let Ok(mut stats) = self.shadow_stats.lock() {
                *stats = Some(renderer.shadow_debug_stats());
            }
            // Read back the GPU-metered EV for the auto-exposure readout. Gated by
            // read_exposure since it does a blocking device poll.
            if self.read_exposure {
                if let Some(rb) = renderer.exposure_state(device, queue, self.vp_id) {
                    if let Ok(mut slot) = self.exposure_readback.lock() {
                        *slot = Some(rb);
                    }
                }
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
