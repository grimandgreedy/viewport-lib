//! Headless rendering harness shared by the counter tests, snapshot tests, and
//! benches.
//!
//! [`Harness`] owns a headless device (from [`crate::device::headless_device`])
//! plus a `ViewportRenderer` and offers the operations the drivers need: build a
//! catalogue scene, render it offscreen, and read back the resulting
//! [`FrameStats`].

use crate::device::headless_device;
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
    /// Build a harness on a headless device, or `None` if no adapter exists.
    pub fn new() -> Option<Self> {
        let (device, queue) = headless_device()?;
        let renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
        Some(Self {
            device,
            queue,
            renderer,
        })
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
