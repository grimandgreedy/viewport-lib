//! Per-viewport render-target allocation for the post-process chain.
//!
//! `ViewportTargetAllocator` owns the sizing rules for viewport-sized
//! targets: the resolution classes (output, scene, half-scene, SSAA-scaled
//! scene), the minimum-1 clamps, and the base usage every render target
//! carries. An effect declares a label, format, and resolution class instead
//! of hand-computing extents, so every target allocated for a viewport goes
//! through one place and resizes with one set of rules.

/// Resolution class of a per-viewport target.
#[derive(Clone, Copy)]
pub(crate) enum TargetSize {
    /// Native output resolution.
    Output,
    /// Scene resolution (output scaled by the render scale).
    Scene,
    /// Half the scene resolution (the bloom ping/pong class).
    HalfScene,
    /// Scene resolution multiplied by the SSAA factor.
    SsaaScene,
}

/// A depth target plus the views its consumers need.
pub(crate) struct DepthTarget {
    pub(crate) texture: crate::gpu::Texture,
    /// Full (depth + stencil) view, used as a render attachment.
    pub(crate) view: crate::gpu::TextureView,
    /// Depth-aspect-only view for sampling.
    pub(crate) depth_only_view: crate::gpu::TextureView,
}

/// Allocates the viewport-sized textures for one viewport's post state.
pub(crate) struct ViewportTargetAllocator<'a> {
    device: &'a crate::gpu::Device,
    output: [u32; 2],
    scene: [u32; 2],
    ssaa_factor: u32,
}

impl<'a> ViewportTargetAllocator<'a> {
    pub(crate) fn new(
        device: &'a crate::gpu::Device,
        output_w: u32,
        output_h: u32,
        scene_w: u32,
        scene_h: u32,
        ssaa_factor: u32,
    ) -> Self {
        Self {
            device,
            output: [output_w.max(1), output_h.max(1)],
            scene: [scene_w.max(1), scene_h.max(1)],
            ssaa_factor: ssaa_factor.max(1),
        }
    }

    fn extent(&self, size: TargetSize) -> crate::gpu::Extent3d {
        let [w, h] = match size {
            TargetSize::Output => self.output,
            TargetSize::Scene => self.scene,
            TargetSize::HalfScene => [(self.scene[0] / 2).max(1), (self.scene[1] / 2).max(1)],
            TargetSize::SsaaScene => [
                self.scene[0] * self.ssaa_factor,
                self.scene[1] * self.ssaa_factor,
            ],
        };
        crate::gpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        }
    }

    /// Allocate a texture with an explicit usage set, plus its default view.
    pub(crate) fn texture(
        &self,
        label: &str,
        format: crate::gpu::TextureFormat,
        size: TargetSize,
        usage: crate::gpu::TextureUsages,
    ) -> (crate::gpu::Texture, crate::gpu::TextureView) {
        let tex = self.device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some(label),
            size: self.extent(size),
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        (tex, view)
    }

    /// Allocate a colour render target: `RENDER_ATTACHMENT | TEXTURE_BINDING`
    /// plus any extra usage, with its default view.
    pub(crate) fn colour(
        &self,
        label: &str,
        format: crate::gpu::TextureFormat,
        size: TargetSize,
        extra_usage: crate::gpu::TextureUsages,
    ) -> (crate::gpu::Texture, crate::gpu::TextureView) {
        self.texture(
            label,
            format,
            size,
            crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING
                | extra_usage,
        )
    }

    /// Allocate a `Depth24PlusStencil8` depth target with its attachment view
    /// and a depth-aspect-only sampling view.
    pub(crate) fn depth(&self, label: &str, size: TargetSize) -> DepthTarget {
        let (texture, view) = self.texture(
            label,
            crate::gpu::TextureFormat::Depth24PlusStencil8,
            size,
            crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
        );
        let depth_only_view = texture.create_view(&crate::gpu::TextureViewDescriptor {
            aspect: crate::gpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        DepthTarget {
            texture,
            view,
            depth_only_view,
        }
    }
}
