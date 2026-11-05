//! The attached runtime's GPU plugin hooks, called from the render forwards.

use super::ViewportInstance;

impl ViewportInstance {
    /// The scene colour and depth the last frame drew, as a GPU plugin's
    /// `post_paint` receives them. `output_view` is the view the frame was
    /// rendered into, for an LDR frame.
    #[doc(hidden)]
    pub fn scene_target_views<'a>(
        &'a self,
        output_view: Option<&'a crate::gpu::TextureView>,
    ) -> Option<crate::runtime::PostPaintTargets<'a>> {
        self.renderer.scene_target_views(&self.frame, output_view)
    }
}
