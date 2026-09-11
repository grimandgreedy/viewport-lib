//! Bloom as an external [`PostEffectProducer`]: a faithful copy of the
//! built-in threshold + separable-blur bloom, filling the
//! [`PostEffectSlot::Bloom`] composite slot.
//!
//! This is the stress case for the producer surface: it reads the HDR
//! scene colour, runs a scene-resolution threshold pass into its own
//! `Rgba16Float` target, then four horizontal + vertical blur iterations
//! over a half-resolution ping/pong pair, and contributes the final pong
//! texture to the additive bloom slot. Shaders, formats, sizes, iteration
//! count, and uniform layout match the built-in exactly, so with
//! `PostProcessSettings.bloom.enabled = false` and this producer
//! registered, output is pixel-identical to the built-in.
//!
//! [`PostEffectProducer`]: viewport_lib::PostEffectProducer
//! [`PostEffectSlot::Bloom`]: viewport_lib::PostEffectSlot::Bloom

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use viewport_lib::wgpu;
use viewport_lib::{
    PostEffectContext, PostEffectProducer, PostEffectResizeContext, PostEffectSlot,
};

use crate::{SettingsHandle, clamp_sampler, colour_target, fullscreen_pass, fullscreen_pipeline};

/// Host-driven settings, matching the built-in `BloomSettings` fields.
#[derive(Clone, Copy, Debug)]
pub struct BloomEffectSettings {
    pub enabled: bool,
    /// Luminance threshold above which pixels bloom.
    pub threshold: f32,
    /// Bloom contribution strength, applied in the threshold pass.
    pub intensity: f32,
    /// Firefly cap: a pixel's luminance is clamped to this before
    /// thresholding.
    pub max_brightness: f32,
}

impl Default for BloomEffectSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 1.0,
            intensity: 0.5,
            max_brightness: 8.0,
        }
    }
}

/// Matches the built-in `BloomUniform` layout (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniform {
    threshold: f32,
    intensity: f32,
    horizontal: u32,
    max_brightness: f32,
}

/// Matches the built-in blur iteration count.
const BLUR_ITERATIONS: usize = 4;

struct BloomViewport {
    _threshold_texture: wgpu::Texture,
    threshold_view: wgpu::TextureView,
    _ping_texture: wgpu::Texture,
    _pong_texture: wgpu::Texture,
    ping_view: wgpu::TextureView,
    pong_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    /// Rebuilt each `prepare`: it binds the frame's scene colour view,
    /// which the resize signal does not carry.
    threshold_bg: Option<wgpu::BindGroup>,
    /// H-blur reading the threshold target (iteration 0 only).
    blur_h_bg: wgpu::BindGroup,
    /// V-blur reading ping.
    blur_v_bg: wgpu::BindGroup,
    /// H-blur reading pong (iterations 1+).
    blur_h_pong_bg: wgpu::BindGroup,
}

/// The external bloom producer. Construct with [`BloomEffect::new`],
/// register the effect, keep the handle.
pub struct BloomEffect {
    settings: SettingsHandle<BloomEffectSettings>,
    device: Option<wgpu::Device>,
    threshold_pipeline: Option<wgpu::RenderPipeline>,
    blur_pipeline: Option<wgpu::RenderPipeline>,
    bgl: Option<wgpu::BindGroupLayout>,
    sampler: Option<wgpu::Sampler>,
    per_viewport: HashMap<usize, BloomViewport>,
}

impl BloomEffect {
    pub fn new(settings: BloomEffectSettings) -> (Self, SettingsHandle<BloomEffectSettings>) {
        let handle = Arc::new(Mutex::new(settings));
        (
            Self {
                settings: handle.clone(),
                device: None,
                threshold_pipeline: None,
                blur_pipeline: None,
                bgl: None,
                sampler: None,
                per_viewport: HashMap::new(),
            },
            handle,
        )
    }
}

/// A uniform buffer whose contents are written at creation, so no queue is
/// needed (the resize signal carries only the device).
fn const_uniform(device: &wgpu::Device, label: &str, value: BloomUniform) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<BloomUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    // 30 made mapping fallible; a buffer created mapped cannot fail here.
    #[cfg(not(feature = "wgpu30"))]
    let mut range = buf.slice(..).get_mapped_range_mut();
    #[cfg(feature = "wgpu30")]
    let mut range = buf.slice(..).get_mapped_range_mut().unwrap();
    range.copy_from_slice(bytemuck::cast_slice(&[value]));
    drop(range);
    buf.unmap();
    buf
}

impl PostEffectProducer for BloomEffect {
    fn type_name(&self) -> &'static str {
        "external_bloom"
    }

    fn slot(&self) -> PostEffectSlot {
        PostEffectSlot::Bloom
    }

    fn enabled(&self) -> bool {
        self.settings.lock().unwrap().enabled
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        // Filterable colour texture + filtering sampler + uniform: the
        // built-in bloom layout, shared by threshold and blur.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("external_bloom_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        self.threshold_pipeline = Some(fullscreen_pipeline(
            device,
            "external_bloom_threshold_pipeline",
            include_str!("shaders/bloom_threshold.wgsl"),
            &bgl,
            wgpu::TextureFormat::Rgba16Float,
        ));
        self.blur_pipeline = Some(fullscreen_pipeline(
            device,
            "external_bloom_blur_pipeline",
            include_str!("shaders/bloom_blur.wgsl"),
            &bgl,
            wgpu::TextureFormat::Rgba16Float,
        ));
        self.sampler = Some(clamp_sampler(
            device,
            "external_bloom_sampler",
            wgpu::FilterMode::Linear,
        ));
        self.bgl = Some(bgl);
        self.device = Some(device.clone());
        self.per_viewport.clear();
    }

    fn on_viewport_resized(&mut self, device: &wgpu::Device, ctx: &PostEffectResizeContext<'_>) {
        let (Some(bgl), Some(sampler)) = (&self.bgl, &self.sampler) else {
            return;
        };
        // Threshold at scene resolution, ping/pong at half: the built-in
        // sizing.
        let half = [
            (ctx.scene_size[0] / 2).max(1),
            (ctx.scene_size[1] / 2).max(1),
        ];
        let (threshold_texture, threshold_view) = colour_target(
            device,
            "external_bloom_threshold",
            ctx.scene_size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let (ping_texture, ping_view) = colour_target(
            device,
            "external_bloom_ping",
            half,
            wgpu::TextureFormat::Rgba16Float,
        );
        let (pong_texture, pong_view) = colour_target(
            device,
            "external_bloom_pong",
            half,
            wgpu::TextureFormat::Rgba16Float,
        );
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("external_bloom_uniform"),
            size: std::mem::size_of::<BloomUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let h_uniform = const_uniform(
            device,
            "external_bloom_h_uniform",
            BloomUniform {
                threshold: 0.0,
                intensity: 0.0,
                horizontal: 1,
                max_brightness: 0.0,
            },
        );
        let v_uniform = const_uniform(
            device,
            "external_bloom_v_uniform",
            BloomUniform {
                threshold: 0.0,
                intensity: 0.0,
                horizontal: 0,
                max_brightness: 0.0,
            },
        );
        let blur_bg = |label: &str, input: &wgpu::TextureView, uniform: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            })
        };
        let blur_h_bg = blur_bg("external_bloom_blur_h_bg", &threshold_view, &h_uniform);
        let blur_v_bg = blur_bg("external_bloom_blur_v_bg", &ping_view, &v_uniform);
        let blur_h_pong_bg = blur_bg("external_bloom_blur_h_pong_bg", &pong_view, &h_uniform);
        self.per_viewport.insert(
            ctx.viewport_index,
            BloomViewport {
                _threshold_texture: threshold_texture,
                threshold_view,
                _ping_texture: ping_texture,
                _pong_texture: pong_texture,
                ping_view,
                pong_view,
                uniform_buf,
                threshold_bg: None,
                blur_h_bg,
                blur_v_bg,
                blur_h_pong_bg,
            },
        );
    }

    fn prepare(&mut self, queue: &wgpu::Queue, ctx: &PostEffectContext<'_>) {
        let (Some(device), Some(bgl), Some(sampler)) = (&self.device, &self.bgl, &self.sampler)
        else {
            return;
        };
        let Some(vp) = self.per_viewport.get_mut(&ctx.viewport_index) else {
            return;
        };
        let settings = *self.settings.lock().unwrap();
        let uniform = BloomUniform {
            threshold: settings.threshold,
            intensity: settings.intensity,
            horizontal: 0,
            max_brightness: settings.max_brightness,
        };
        queue.write_buffer(&vp.uniform_buf, 0, bytemuck::cast_slice(&[uniform]));

        // The scene colour view can change when the viewport's targets are
        // recreated, so bind it fresh each frame.
        vp.threshold_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("external_bloom_threshold_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ctx.scene_colour),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vp.uniform_buf.as_entire_binding(),
                },
            ],
        }));
    }

    fn encode<'a>(
        &'a mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PostEffectContext<'_>,
    ) -> Option<&'a wgpu::TextureView> {
        let threshold_pipeline = self.threshold_pipeline.as_ref()?;
        let blur_pipeline = self.blur_pipeline.as_ref()?;
        let vp = self.per_viewport.get(&ctx.viewport_index)?;
        let threshold_bg = vp.threshold_bg.as_ref()?;

        fullscreen_pass(
            encoder,
            "external_bloom_threshold_pass",
            &vp.threshold_view,
            wgpu::Color::BLACK,
            threshold_pipeline,
            threshold_bg,
        );
        // Iteration 0 reads the threshold target; later iterations read the
        // previous vertical result from pong. Matches the built-in order.
        for i in 0..BLUR_ITERATIONS {
            let h_bg = if i == 0 {
                &vp.blur_h_bg
            } else {
                &vp.blur_h_pong_bg
            };
            fullscreen_pass(
                encoder,
                "external_bloom_blur_h_pass",
                &vp.ping_view,
                wgpu::Color::BLACK,
                blur_pipeline,
                h_bg,
            );
            fullscreen_pass(
                encoder,
                "external_bloom_blur_v_pass",
                &vp.pong_view,
                wgpu::Color::BLACK,
                blur_pipeline,
                &vp.blur_v_bg,
            );
        }
        Some(&vp.pong_view)
    }
}
