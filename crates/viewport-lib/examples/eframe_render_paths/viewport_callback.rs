use viewport_lib as vpl;
use vpl::{FrameData, MeshId, ViewportRenderer, plugins::skinning::SkinningPlugin};

/// Per-frame data handed from the egui update to the wgpu paint callback.
///
/// Everything that needs the renderer's `&mut` state, the device, or the queue
/// happens in `prepare`: toggling GPU culling, uploading the skinned mesh's joint
/// palette, and the prepare/paint calls themselves.
pub struct ViewportCallback {
    pub frame: FrameData,
    /// Desired GPU-culling state (applied if the device supports it).
    pub gpu_culling: bool,
    /// Skinning handle + mesh + this frame's two-joint palette, uploaded before prepare.
    pub skinning: Option<SkinningPlugin>,
    pub skinned_mesh: MeshId,
    pub skin_palette: [glam::Mat4; 2],
    /// Reports back whether the instanced path ran and how many batches, for the status line.
    pub instancing_status: std::sync::Arc<std::sync::Mutex<(bool, usize)>>,
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
            // GPU culling is a renderer-wide setting: flip the instanced batches
            // between the direct-draw and indirect (GPU-cull) pipelines.
            if self.gpu_culling && renderer.is_gpu_culling_supported() {
                renderer.enable_gpu_driven_culling();
            } else {
                renderer.disable_gpu_driven_culling();
            }

            // Upload the skinned mesh's joint palette for this frame. This is what
            // gives the item per-instance deform data, routing it down the per-object
            // skinned path. Instance 0, since there is a single skinned object.
            if let Some(skinning) = &self.skinning {
                skinning.attach_palette(
                    renderer.resources_mut(),
                    device,
                    queue,
                    self.skinned_mesh,
                    0,
                    &self.skin_palette,
                );
            }

            let cmds = renderer.pass().prepare(device, queue, &self.frame);
            if let Ok(mut status) = self.instancing_status.lock() {
                *status = (
                    renderer.is_using_instanced_path(),
                    renderer.instanced_batch_count(),
                );
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
