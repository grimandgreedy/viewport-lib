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
/// first frame that uses it. It stays live after that, or, with
/// `RendererConfig::target_release_frames` set, until it has gone unused for
/// that many frames (see [`TargetIdle`]).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct TargetGroups(u16);

impl TargetGroups {
    /// HDR scene colour and depth: every frame on the HDR path.
    pub(crate) const SCENE: Self = Self(1 << 0);
    /// The depth buffer the LDR path renders against, which the outline mask
    /// pass also tests.
    pub(crate) const LDR_DEPTH: Self = Self(1 << 1);
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

    /// The groups in `self` that are not in `other`.
    pub(crate) const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// The groups this set holds, one bit each, with their names.
    pub(crate) fn names(self) -> impl Iterator<Item = &'static str> {
        const NAMES: [(TargetGroups, &str); 7] = [
            (TargetGroups::SCENE, "scene"),
            (TargetGroups::LDR_DEPTH, "ldr_depth"),
            (TargetGroups::BLOOM, "bloom"),
            (TargetGroups::SSAO, "ssao"),
            (TargetGroups::DOF, "dof"),
            (TargetGroups::CONTACT_SHADOW, "contact_shadow"),
            (TargetGroups::FXAA, "fxaa"),
        ];
        NAMES
            .into_iter()
            .filter(move |(group, _)| self.contains(*group))
            .map(|(_, name)| name)
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

/// How long each of one viewport's targets has gone unused, so the ones a
/// viewport stopped asking for can return to their placeholders.
///
/// Every frame records what it asked for; [`age`](Self::age) then counts one
/// frame and reports which held targets have been idle long enough to
/// release. The OIT and foreground depth targets are not in a group, so they
/// are counted alongside.
#[derive(Default)]
pub(crate) struct TargetIdle {
    /// Groups asked for since the last `age`.
    asked: TargetGroups,
    /// Frames each group has gone unused, indexed by its bit.
    group_frames: [u32; 16],
    oit_used: bool,
    oit_frames: u32,
    foreground_used: bool,
    foreground_frames: u32,
}

/// The held targets [`TargetIdle::age`] found idle long enough to release.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct IdleTargets {
    pub(crate) groups: TargetGroups,
    pub(crate) oit: bool,
    pub(crate) foreground: bool,
}

impl TargetIdle {
    /// Record that a frame asked for `groups`.
    pub(crate) fn ask(&mut self, groups: TargetGroups) {
        self.asked = self.asked | groups;
    }

    /// Record that a frame drew order-independent transparency.
    pub(crate) fn use_oit(&mut self) {
        self.oit_used = true;
    }

    /// Record that a frame drew the foreground pass.
    pub(crate) fn use_foreground(&mut self) {
        self.foreground_used = true;
    }

    /// Count one frame. `held` is what the viewport holds and `oit` /
    /// `foreground` whether it holds those targets; everything recorded since
    /// the last call counts as used. Returns what has now gone unused for
    /// `after` frames or more, and resets the count of whatever it returns.
    pub(crate) fn age(
        &mut self,
        held: TargetGroups,
        oit: bool,
        foreground: bool,
        after: u32,
    ) -> IdleTargets {
        let mut idle = IdleTargets::default();
        let asked = std::mem::take(&mut self.asked);
        for (bit, frames) in self.group_frames.iter_mut().enumerate() {
            let group = TargetGroups(1 << bit);
            if !held.contains(group) || asked.contains(group) {
                *frames = 0;
                continue;
            }
            *frames += 1;
            if *frames >= after {
                *frames = 0;
                idle.groups = idle.groups | group;
            }
        }
        let count = |held: bool, used: bool, frames: &mut u32| -> bool {
            if !held || used {
                *frames = 0;
                return false;
            }
            *frames += 1;
            if *frames >= after {
                *frames = 0;
                return true;
            }
            false
        };
        idle.oit = count(
            oit,
            std::mem::take(&mut self.oit_used),
            &mut self.oit_frames,
        );
        idle.foreground = count(
            foreground,
            std::mem::take(&mut self.foreground_used),
            &mut self.foreground_frames,
        );
        idle
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

#[cfg(test)]
mod tests {
    use super::{IdleTargets, TargetGroups as G, TargetIdle};

    /// A group asked for once is released after `after` frames without it,
    /// and a group asked for again restarts its count.
    #[test]
    fn idle_groups_release_after_the_threshold() {
        let mut idle = TargetIdle::default();
        let held = G::SCENE | G::BLOOM;
        idle.ask(held);
        assert_eq!(idle.age(held, false, false, 3), IdleTargets::default());
        for _ in 0..2 {
            idle.ask(G::SCENE);
            assert_eq!(idle.age(held, false, false, 3).groups, G::empty());
        }
        idle.ask(G::SCENE);
        assert_eq!(idle.age(held, false, false, 3).groups, G::BLOOM);

        idle.ask(G::SCENE);
        idle.age(held, false, false, 3);
        idle.ask(G::SCENE | G::BLOOM);
        idle.age(held, false, false, 3);
        idle.ask(G::SCENE);
        assert_eq!(idle.age(held, false, false, 3).groups, G::empty());
    }

    /// The OIT and foreground targets follow the same count.
    #[test]
    fn idle_oit_and_foreground_release_after_the_threshold() {
        let mut idle = TargetIdle::default();
        idle.use_oit();
        idle.use_foreground();
        assert_eq!(idle.age(G::empty(), true, true, 2), IdleTargets::default());
        idle.use_foreground();
        assert_eq!(idle.age(G::empty(), true, true, 2), IdleTargets::default());
        let released = idle.age(G::empty(), true, true, 2);
        assert!(released.oit && !released.foreground);
    }
}
