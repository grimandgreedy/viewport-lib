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
    #[cfg(feature = "scenes")]
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
        let backend = info.backend;
        let mut harness = Self::build(device, queue, target_format, None, None, Some(backend));
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
    ///
    /// The renderer reads the adapter's downlevel capabilities, less the
    /// profile's [`remove_downlevel`](crate::device::DeviceProfile::remove_downlevel).
    ///
    /// # Panics
    ///
    /// When the profile puts the device below the tier the renderer can be
    /// built on.
    pub fn with_profile(profile: &crate::device::DeviceProfile) -> Option<Self> {
        let d = crate::device::headless_device_full(profile)?;
        let mut harness = Self::build(
            d.device,
            d.queue,
            Self::DEFAULT_TARGET_FORMAT,
            None,
            Some(d.downlevel),
            Some(d.info.backend),
        );
        harness.adapter_info = Some(d.info);
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
        Self::from_device_with_pipeline_cache(device, queue, target_format, None)
    }

    /// Like [`from_device`](Self::from_device), with the renderer's pipeline
    /// cache seeded from `pipeline_cache_data`. Read the cache back with
    /// `harness.renderer.pipeline_cache_data()`. The data is ignored on a
    /// device without `Features::PIPELINE_CACHE`.
    pub fn from_device_with_pipeline_cache(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
        pipeline_cache_data: Option<&[u8]>,
    ) -> Self {
        Self::build(
            device,
            queue,
            target_format,
            pipeline_cache_data,
            None,
            None,
        )
    }

    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
        pipeline_cache_data: Option<&[u8]>,
        downlevel: Option<wgpu::DownlevelCapabilities>,
        backend: Option<wgpu::Backend>,
    ) -> Self {
        // A test reads its frame back at once, so every pipeline has to be
        // there when the frame draws, whatever the platform's default.
        let mut config = viewport_lib::RendererConfig::new(target_format)
            .with_pipeline_cache_data(pipeline_cache_data.map(<[u8]>::to_vec))
            .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
        if let Some(downlevel) = downlevel {
            config = config.with_downlevel_capabilities(downlevel);
        }
        // wgpu 27 cannot read the backend from a device, and the GL backend's
        // shaders cannot load from depth textures.
        config.backend = backend;
        let mut renderer = ViewportRenderer::with_config(&device, &config);
        // The item types viewport-lib ships with live in their own crate and
        // register like any other plugin, so a harness that renders them has
        // to install them the way a consumer does.
        viewport_lib_plugins::item_types::install(&mut renderer, &device);
        Self {
            device,
            queue,
            renderer,
            adapter_info: None,
            #[cfg(feature = "scenes")]
            scenes_built: 0,
        }
    }

    /// Directory name golden references for this harness's adapter live under:
    /// `"metal"`, `"vulkan"`, `"dx12"`, `"gl"` for hardware, with the
    /// rasteriser appended for a software adapter (`"vulkan-lavapipe"`,
    /// `"gl-llvmpipe"`, `"dx12-warp"`). References are blessed per adapter
    /// because each rasterises the same frame differently, and a software
    /// rasteriser differently again from hardware on the same backend. Falls
    /// back to `"unknown"` when the harness was built around a caller-owned
    /// device. See [`reference_dir_name`].
    pub fn backend_dir_name(&self) -> String {
        match &self.adapter_info {
            Some(info) => {
                reference_dir_name(info.backend, info.device_type, &info.name, &info.driver)
            }
            None => "unknown".to_string(),
        }
    }

    /// The adapter's `driver_info` (for Mesa, the version string `vulkaninfo`
    /// prints as `driverInfo`), or `None` for a caller-owned device.
    pub fn driver_info(&self) -> Option<&str> {
        self.adapter_info.as_ref().map(|i| i.driver_info.as_str())
    }

    /// Upload a scene's assets into this harness's renderer and return the built
    /// scene. Assets accumulate in the renderer; build each scene on a fresh
    /// harness when isolated counts matter. Any environment zones are cleared
    /// first, so only the scene built last has its zones in effect.
    #[cfg(feature = "scenes")]
    pub fn build_scene(&mut self, scene: &NamedScene) -> BuiltScene {
        // Environment zones are renderer state rather than frame data: a scene
        // that sets them does so in its build, and the next build starts clear.
        // Only the full tier holds them.
        if self.renderer.tier() == viewport_lib::Tier::Full {
            self.renderer.clear_environment_zones();
        }
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
            let mapped = viewport_lib::gpu::mapped_range(staging.slice(..));
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
            // A subnormal's exponent is -14, one above the biased zero.
            (sign << 31) | (((127 - 14 + shift) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exponent == 0x1f {
        (sign << 31) | (0xff << 23) | (fraction << 13)
    } else {
        (sign << 31) | ((exponent + 127 - 15) << 23) | (fraction << 13)
    };
    f32::from_bits(out)
}

/// The golden reference directory name for an adapter.
///
/// The backend's name in lower case (`metal`, `vulkan`, `dx12`, `gl`), plus a
/// suffix naming the rasteriser when `device_type` is
/// [`DeviceType::Cpu`](wgpu::DeviceType::Cpu): `lavapipe` for Mesa's Vulkan
/// driver, `llvmpipe` for Mesa's GL driver, `warp` for Direct3D's, `softpipe`
/// and `swiftshader` by name, and `cpu` for any other software adapter.
/// `name` and `driver` are the adapter's `AdapterInfo` fields.
pub fn reference_dir_name(
    backend: wgpu::Backend,
    device_type: wgpu::DeviceType,
    name: &str,
    driver: &str,
) -> String {
    let base = format!("{backend:?}").to_lowercase();
    if device_type != wgpu::DeviceType::Cpu {
        return base;
    }
    let id = format!("{name} {driver}").to_lowercase();
    // lavapipe reports itself as "llvmpipe" in its device name, so the backend
    // decides which Mesa driver it is.
    let rasteriser = if id.contains("lavapipe")
        || (backend == wgpu::Backend::Vulkan && id.contains("llvmpipe"))
    {
        "lavapipe"
    } else if id.contains("llvmpipe") {
        "llvmpipe"
    } else if id.contains("basic render driver") || id.contains("warp") {
        "warp"
    } else if id.contains("softpipe") {
        "softpipe"
    } else if id.contains("swiftshader") {
        "swiftshader"
    } else {
        "cpu"
    };
    format!("{base}-{rasteriser}")
}

#[cfg(test)]
mod tests {
    use super::{half_to_f32, reference_dir_name};
    use viewport_lib::wgpu::{Backend, DeviceType};

    #[test]
    fn halves_decode_across_the_subnormal_boundary() {
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(half_to_f32(0x0200), 2f32.powi(-15));
        assert_eq!(half_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(half_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
    }

    #[test]
    fn hardware_adapters_keep_the_backend_name() {
        assert_eq!(
            reference_dir_name(Backend::Metal, DeviceType::IntegratedGpu, "Apple M4", ""),
            "metal"
        );
        assert_eq!(
            reference_dir_name(
                Backend::Vulkan,
                DeviceType::DiscreteGpu,
                "NVIDIA GeForce RTX 3080",
                "NVIDIA"
            ),
            "vulkan"
        );
        assert_eq!(
            reference_dir_name(Backend::Gl, DeviceType::IntegratedGpu, "Mesa Intel", ""),
            "gl"
        );
    }

    #[test]
    fn software_adapters_name_their_rasteriser() {
        assert_eq!(
            reference_dir_name(
                Backend::Vulkan,
                DeviceType::Cpu,
                "llvmpipe (LLVM 17.0.6, 256 bits)",
                "llvmpipe"
            ),
            "vulkan-lavapipe"
        );
        assert_eq!(
            reference_dir_name(
                Backend::Gl,
                DeviceType::Cpu,
                "llvmpipe (LLVM 17.0.6, 256 bits)",
                ""
            ),
            "gl-llvmpipe"
        );
        assert_eq!(
            reference_dir_name(
                Backend::Dx12,
                DeviceType::Cpu,
                "Microsoft Basic Render Driver",
                ""
            ),
            "dx12-warp"
        );
        assert_eq!(
            reference_dir_name(Backend::Vulkan, DeviceType::Cpu, "SwiftShader Device", ""),
            "vulkan-swiftshader"
        );
        assert_eq!(
            reference_dir_name(Backend::Vulkan, DeviceType::Cpu, "Some Emulator", ""),
            "vulkan-cpu"
        );
    }
}
