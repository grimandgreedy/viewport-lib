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
    /// Info for the adapter the device was requested from. `None` when the
    /// harness was built around a caller-owned device via
    /// [`from_device`](Self::from_device).
    pub adapter_info: Option<wgpu::AdapterInfo>,
    /// Scenes built so far; stamps each built scene's `generation` so the
    /// renderer's scene-content caches see consecutive catalogue scenes as
    /// different content.
    scenes_built: u64,
}

impl Harness {
    /// The target format `new` builds its renderer with.
    pub const DEFAULT_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

    /// The target format [`Harness::render_float`] needs the renderer built with.
    pub const FLOAT_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    /// Build a harness on a headless device, or `None` if no adapter exists.
    pub fn new() -> Option<Self> {
        Self::with_profile(&crate::device::DeviceProfile::harness())
    }

    /// Build a harness on a headless device whose renderer targets
    /// `target_format`, or `None` if no adapter exists.
    ///
    /// Pass [`Harness::FLOAT_TARGET_FORMAT`] to render into a half-float target
    /// and read frames back with [`render_float`](Self::render_float).
    pub fn with_target_format(target_format: wgpu::TextureFormat) -> Option<Self> {
        let (device, queue, info) =
            crate::device::headless_device_with_info(&crate::device::DeviceProfile::harness())?;
        let mut harness = Self::from_device(device, queue, target_format);
        harness.adapter_info = Some(info);
        Some(harness)
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
        let (device, queue, info) = crate::device::headless_device_with_info(profile)?;
        let mut harness = Self::from_device(device, queue, Self::DEFAULT_TARGET_FORMAT);
        harness.adapter_info = Some(info);
        Some(harness)
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
            adapter_info: None,
            scenes_built: 0,
        }
    }

    /// Directory name golden references for this harness's backend live under:
    /// `"metal"`, `"vulkan"`, `"dx12"`, `"gl"`. References are blessed per
    /// backend because different backends rasterise the same frame differently.
    /// Falls back to `"unknown"` when the harness was built around a
    /// caller-owned device.
    pub fn backend_dir_name(&self) -> String {
        match &self.adapter_info {
            Some(info) => format!("{:?}", info.backend).to_lowercase(),
            None => "unknown".to_string(),
        }
    }

    /// Upload a scene's assets into this harness's renderer and return the built
    /// scene. Assets accumulate in the renderer; build each scene on a fresh
    /// harness when isolated counts matter.
    #[cfg(feature = "scenes")]
    pub fn build_scene(&mut self, scene: &NamedScene) -> BuiltScene {
        let mut ctx = BuildCtx {
            renderer: &mut self.renderer,
            device: &self.device,
            queue: &self.queue,
        };
        let mut built = (scene.build)(&mut ctx);
        // Each build gets a fresh scene generation. The renderer's
        // instanced-batch cache keys on `SceneFrame::generation` plus item
        // counts, so two different scenes built back to back with the default
        // generation and the same item count would silently reuse the first
        // scene's batches.
        self.scenes_built += 1;
        built.generation = self.scenes_built;
        built
    }

    /// Render a frame offscreen and return the RGBA pixels (row-major,
    /// `width * height * 4` bytes).
    pub fn render(&mut self, frame: &FrameData, width: u32, height: u32) -> Vec<u8> {
        self.renderer
            .render_offscreen(&self.device, &self.queue, frame, width, height)
    }

    /// Render a frame into a half-float target and return the pixels as `f32`,
    /// row-major, four channels per pixel.
    ///
    /// The renderer must have been built with [`FLOAT_TARGET_FORMAT`](Self::FLOAT_TARGET_FORMAT)
    /// (see [`with_target_format`](Self::with_target_format)) or the pipelines
    /// will not match the target this creates.
    ///
    /// Use this instead of [`render`](Self::render) when comparing two frames for
    /// equality. An 8-bit readback only reports a pixel whose value crosses a
    /// quantisation step, so it misses most of a small difference and reports the
    /// rest intermittently, and the size of the step a pixel crosses says nothing
    /// about the size of the difference that moved it.
    pub fn render_float(&mut self, frame: &FrameData, width: u32, height: u32) -> Vec<[f32; 4]> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("harness_float_target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FLOAT_TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render_to_texture(&self.device, &self.queue, &view, frame);

        let bytes_per_pixel = 8u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) & !(align - 1);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("harness_float_staging"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("harness_float_copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            size,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .expect("float readback poll");
        let _ = rx.recv();

        let mut pixels = Vec::with_capacity((width * height) as usize);
        {
            let mapped = staging.slice(..).get_mapped_range();
            for row in 0..height as usize {
                let start = row * padded_row as usize;
                let bytes = &mapped[start..start + unpadded_row as usize];
                for texel in bytes.chunks_exact(bytes_per_pixel as usize) {
                    let mut channels = [0f32; 4];
                    for (c, slot) in channels.iter_mut().enumerate() {
                        *slot = half_to_f32(u16::from_le_bytes([texel[c * 2], texel[c * 2 + 1]]));
                    }
                    pixels.push(channels);
                }
            }
        }
        staging.unmap();
        pixels
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

/// Decode an IEEE half into an `f32`.
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) & 1;
    let exponent = u32::from(bits >> 10) & 0x1f;
    let fraction = u32::from(bits) & 0x3ff;
    let out = if exponent == 0 {
        if fraction == 0 {
            sign << 31
        } else {
            // Subnormal half, normal f32: shift the fraction up until the
            // implicit leading bit appears, paying for it in the exponent.
            let mut shift = 0i32;
            let mut f = fraction;
            while f & 0x400 == 0 {
                f <<= 1;
                shift -= 1;
            }
            (sign << 31) | (((127 - 15 + shift) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exponent == 0x1f {
        (sign << 31) | (0xff << 23) | (fraction << 13)
    } else {
        (sign << 31) | ((exponent + 127 - 15) << 23) | (fraction << 13)
    };
    f32::from_bits(out)
}
