use viewport_lib::{FrameData, ViewportId, ViewportRenderer};

pub struct MultiViewportCallback {
    pub frames: [FrameData; 4],
    pub viewports: [ViewportId; 4],
}

impl eframe::egui_wgpu::CallbackTrait for MultiViewportCallback {
    fn prepare(
        &self,
        device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder,
        callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        if let Some(renderer) = callback_resources.get_mut::<ViewportRenderer>() {
            let (scene_fx, _) = self.frames[0].effects.split();
            renderer.pass().prepare_scene(device, queue, &self.frames[0], &scene_fx);
            for (i, frame) in self.frames.iter().enumerate() {
                renderer.pass().prepare_viewport(device, queue, self.viewports[i], frame);
            }
            return self
                .frames
                .iter()
                .enumerate()
                .map(|(i, frame)| {
                    renderer
                        .pass()
                        .prepare_hdr_viewport(device, queue, self.viewports[i], frame)
                })
                .collect();
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = callback_resources.get::<ViewportRenderer>() {
            let vp = info.viewport_in_pixels();
            let fx = vp.left_px.max(0) as u32;
            let fy = vp.top_px.max(0) as u32;
            let fw = vp.width_px.max(0) as u32;
            let fh = vp.height_px.max(0) as u32;

            let hw = fw / 2;
            let hh = fh / 2;
            let hw2 = fw - hw;
            let hh2 = fh - hh;

            // (x, y, w, h) in physical pixels : TL, TR, BL, BR.
            let quads: [(u32, u32, u32, u32); 4] = [
                (fx, fy, hw, hh),
                (fx + hw, fy, hw2, hh),
                (fx, fy + hh, hw, hh2),
                (fx + hw, fy + hh, hw2, hh2),
            ];

            for (i, &(qx, qy, qw, qh)) in quads.iter().enumerate() {
                if qw == 0 || qh == 0 {
                    continue;
                }
                render_pass.set_viewport(qx as f32, qy as f32, qw as f32, qh as f32, 0.0, 1.0);
                render_pass.set_scissor_rect(qx, qy, qw, qh);
                renderer.pass_view().paint_hdr_blit(render_pass, &self.frames[i]);
            }
        }
    }
}
