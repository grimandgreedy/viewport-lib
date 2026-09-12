//! Headless rendering harness shared by the counter tests, snapshot tests, and
//! benches.
//!
//! [`Harness`] owns a headless device (from [`crate::device::headless_device`])
//! plus a `ViewportRenderer` and offers the operations the drivers need: build a
//! catalogue scene, render it offscreen, and read back the resulting
//! [`FrameStats`].

#[cfg(feature = "scenes")]
use crate::scenes::{BuildCtx, BuiltScene, NamedScene};
use viewport_lib::wgpu;
use viewport_lib::{FrameData, FrameStats, ViewportRenderer};

/// A renderer plus its device, ready to build and render catalogue scenes.
pub struct Harness {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The wgpu queue.
    pub queue: wgpu::Queue,
    /// The viewport renderer (LDR target format).
    pub renderer: ViewportRenderer,
}

impl Harness {
    /// The target format `new` builds its renderer with.
    pub const DEFAULT_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

    /// Build a harness on a headless device, or `None` if no adapter exists.
    pub fn new() -> Option<Self> {
        Self::with_profile(&crate::device::DeviceProfile::harness())
    }

    /// Build a harness on a device requested with `profile`, or `None` when no
    /// adapter matches it.
    ///
    /// `new`'s default profile takes wgpu's default limits, which is below what
    /// some paths need: per-vertex deformers and material plugins require the
    /// renderer's recommended limits (three and four bind groups respectively,
    /// plus the storage-buffer headroom), and register calls fail cleanly
    /// without them. Pass `DeviceProfile::low_power(..)` or
    /// `high_performance(..)` to get those limits.
    pub fn with_profile(profile: &crate::device::DeviceProfile) -> Option<Self> {
        let (device, queue) = crate::device::headless_device_with(profile)?;
        Some(Self::from_device(
            device,
            queue,
            Self::DEFAULT_TARGET_FORMAT,
        ))
    }

    /// Build a harness around a device the caller already owns, with a
    /// renderer targeting `target_format`.
    ///
    /// For tests that need several renderers on one device (an A/B render
    /// where each side registers different plugins), or a target format other
    /// than the default.
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let renderer = ViewportRenderer::new(&device, target_format);
        Self {
            device,
            queue,
            renderer,
        }
    }

    /// Upload a scene's assets into this harness's renderer and return the built
    /// scene. Assets accumulate in the renderer; build each scene on a fresh
    /// harness when isolated counts matter.
    #[cfg(feature = "scenes")]
    pub fn build_scene(&mut self, scene: &NamedScene) -> BuiltScene {
        let mut ctx = BuildCtx {
            res: self.renderer.resources_mut(),
            device: &self.device,
            queue: &self.queue,
        };
        (scene.build)(&mut ctx)
    }

    /// Render a frame offscreen and return the RGBA pixels (row-major,
    /// `width * height * 4` bytes).
    pub fn render(&mut self, frame: &FrameData, width: u32, height: u32) -> Vec<u8> {
        self.renderer
            .render_offscreen(&self.device, &self.queue, frame, width, height)
    }

    /// The most recent frame's statistics.
    pub fn stats(&self) -> FrameStats {
        self.renderer.last_frame_stats()
    }

    /// Render the same frame twice and return the second frame's stats. Counter
    /// tests use this so cache-warmup effects (first-frame uploads, batch
    /// building) settle before the counters are asserted.
    pub fn render_two_frames(&mut self, frame: &FrameData, width: u32, height: u32) -> FrameStats {
        let _ = self.render(frame, width, height);
        let _ = self.render(frame, width, height);
        self.stats()
    }
}
