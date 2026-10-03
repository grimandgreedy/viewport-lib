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
    /// One texel: the stand-in for a target whose group is not live yet.
    Placeholder,
}

/// The groups a viewport's targets are allocated in, one per consumer.
///
/// A viewport holds every target at full size only for the groups a frame has
/// asked for; the rest are one-texel stand-ins, so the bind groups that name
/// them stay valid and cost nothing. A group is promoted to full size by the
/// first frame that uses it and stays live after that.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct TargetGroups(u16);

impl TargetGroups {
    /// HDR scene colour and depth: every frame on the HDR path.
    pub(crate) const SCENE: Self = Self(1 << 0);
    /// The depth buffer the LDR path renders against, which the outline mask
    /// pass also tests.
    pub(crate) const LDR_DEPTH: Self = Self(1 << 1);
    /// Outline mask and colour.
    pub(crate) const OUTLINE: Self = Self(1 << 2);
    /// Bloom threshold, ping and pong.
    pub(crate) const BLOOM: Self = Self(1 << 3);
    /// SSAO and its blur.
    pub(crate) const SSAO: Self = Self(1 << 4);
    /// Depth of field.
    pub(crate) const DOF: Self = Self(1 << 5);
    /// Contact shadows.
    pub(crate) const CONTACT_SHADOW: Self = Self(1 << 6);
    /// The FXAA input.
    pub(crate) const FXAA: Self = Self(1 << 7);

    pub(crate) const fn empty() -> Self {
        Self(0)
    }

    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// `group` when `on`, nothing otherwise.
    pub(crate) const fn when(group: Self, on: bool) -> Self {
        if on { group } else { Self(0) }
    }

    /// `live` when `group` is in the set, the one-texel stand-in otherwise.
    pub(crate) fn size(self, group: Self, live: TargetSize) -> TargetSize {
        if self.contains(group) {
            live
        } else {
            TargetSize::Placeholder
        }
    }
}

impl std::ops::BitOr for TargetGroups {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
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
            TargetSize::Placeholder => [1, 1],
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
        if crate::resources::build_log::enabled() {
            let extent = self.extent(size);
            // `block_copy_size` is None for the depth-stencil formats, which are
            // not copyable; they are 4 bytes per texel here.
            let texel = format.block_copy_size(None).unwrap_or(4) as u64;
            crate::resources::build_log::record_texture(
                label,
                extent.width as u64 * extent.height as u64 * texel,
            );
        }
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
