//! The `viewport-lib-vfx` kit rebuilt as a [`PostEffectStage`] stack:
//! colour grade, depth fog, and edge detect as three chained display-space
//! stages behind one settings handle. (The kit's fourth effect, bloom, is
//! covered HDR-correctly by [`crate::BloomEffect`].)
//!
//! The original kit could not compose back through `GpuPlugin::post_paint`,
//! so it owned a parallel `VfxRenderer` with its own scene colour/depth
//! targets, a Scene/TempA/TempB ping-pong, resize tracking, and a final
//! blit. On the stage chain none of that exists: each stage owns exactly
//! one input texture per viewport, the chain routes the previous writer
//! into it, and the last stage renders into the frame's final target. The
//! depth-reading stages take the scene depth from the effect context
//! instead of owning a depth target.
//!
//! Register with [`vfx_stack`]:
//!
//! ```ignore
//! let (stages, settings) = vfx_stack();
//! for (stage, order) in stages {
//!     renderer.add_post_effect_stage(stage, order);
//! }
//! settings.lock().unwrap().depth_fog.enabled = true;
//! ```
//!
//! [`PostEffectStage`]: viewport_lib::PostEffectStage

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use viewport_lib::plugin_api::post_effect::stage_order;
use viewport_lib::wgpu;
use viewport_lib::{PostEffectContext, PostEffectResizeContext, PostEffectStage};

use viewport_lib::plugin_api::post_effect::build_post_effect_pipeline;

use crate::{SettingsHandle, clamp_sampler, colour_target, fullscreen_pass, wgsl_module};

/// Colour-grade parameters (exposure in stops, contrast and saturation as
/// multipliers around mid-grey, multiplicative tint).
#[derive(Clone, Copy, Debug)]
pub struct ColourGrade {
    pub enabled: bool,
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub tint: [f32; 3],
}

impl Default for ColourGrade {
    fn default() -> Self {
        Self {
            enabled: true,
            exposure: 0.08,
            contrast: 1.12,
            saturation: 1.05,
            tint: [1.0, 0.97, 0.92],
        }
    }
}

/// Depth-fog parameters: fog ramps between the `near` and `far` depth-buffer
/// values (non-linear device depth, as in the original kit).
#[derive(Clone, Copy, Debug)]
pub struct DepthFog {
    pub enabled: bool,
    pub near: f32,
    pub far: f32,
    pub amount: f32,
    pub colour: [f32; 3],
}

impl Default for DepthFog {
    fn default() -> Self {
        Self {
            enabled: false,
            near: 0.45,
            far: 1.0,
            amount: 0.55,
            colour: [0.62, 0.72, 0.82],
        }
    }
}

/// Edge-detect parameters: colour- and depth-gradient edges above the
/// threshold are drawn in `colour`.
#[derive(Clone, Copy, Debug)]
pub struct EdgeDetect {
    pub enabled: bool,
    pub colour_strength: f32,
    pub depth_strength: f32,
    pub threshold: f32,
    pub colour: [f32; 3],
}

impl Default for EdgeDetect {
    fn default() -> Self {
        Self {
            enabled: false,
            colour_strength: 1.2,
            depth_strength: 4.0,
            threshold: 0.06,
            colour: [0.04, 0.05, 0.06],
        }
    }
}

/// The stack's settings, shared by all three stages through one handle.
#[derive(Clone, Copy, Debug, Default)]
pub struct VfxSettings {
    pub colour_grade: ColourGrade,
    pub depth_fog: DepthFog,
    pub edge_detect: EdgeDetect,
}

/// Matches the kit's `VfxUniform` layout (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VfxUniform {
    viewport_size: [f32; 2],
    inv_viewport_size: [f32; 2],
    params0: [f32; 4],
    params1: [f32; 4],
}

impl VfxUniform {
    fn new(size: [u32; 2], params0: [f32; 4], params1: [f32; 4]) -> Self {
        let w = size[0].max(1) as f32;
        let h = size[1].max(1) as f32;
        Self {
            viewport_size: [w, h],
            inv_viewport_size: [1.0 / w, 1.0 / h],
            params0,
            params1,
        }
    }
}

/// Which of the three passes a [`VfxStage`] instance runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    ColourGrade,
    DepthFog,
    EdgeDetect,
}

impl Pass {
    fn shader(self) -> &'static str {
        match self {
            Pass::ColourGrade => include_str!("shaders/colour_grade.wgsl"),
            Pass::DepthFog => include_str!("shaders/depth_fog.wgsl"),
            Pass::EdgeDetect => include_str!("shaders/edge_detect.wgsl"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Pass::ColourGrade => "vfx_colour_grade",
            Pass::DepthFog => "vfx_depth_fog",
            Pass::EdgeDetect => "vfx_edge_detect",
        }
    }

    fn reads_depth(self) -> bool {
        !matches!(self, Pass::ColourGrade)
    }
}

struct VfxViewport {
    _texture: wgpu::Texture,
    input_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    /// Binds the stage input (and, for the depth passes, the viewport's
    /// scene depth from the resize signal); valid until the next signal.
    bind_group: wgpu::BindGroup,
}

/// One stage of the stack. All three passes share this implementation,
/// differing in shader, bind layout (the fog and edge passes read the
/// scene depth), and which settings block gates and parameterises them.
pub struct VfxStage {
    pass: Pass,
    settings: SettingsHandle<VfxSettings>,
    /// Built on the first resize signal: the pipeline's target format is
    /// the renderer's LDR format, which arrives with that signal.
    pipeline: Option<wgpu::RenderPipeline>,
    bgl: Option<wgpu::BindGroupLayout>,
    sampler: Option<wgpu::Sampler>,
    per_viewport: HashMap<usize, VfxViewport>,
}

/// Build the three-stage stack. Returns the stages paired with ascending
/// chain order keys in the external band, and the shared settings handle.
pub fn vfx_stack() -> (
    Vec<(Box<dyn PostEffectStage>, i32)>,
    SettingsHandle<VfxSettings>,
) {
    let settings: SettingsHandle<VfxSettings> = Arc::new(Mutex::new(VfxSettings::default()));
    let stage = |pass: Pass| -> Box<dyn PostEffectStage> {
        Box::new(VfxStage {
            pass,
            settings: settings.clone(),
            pipeline: None,
            bgl: None,
            sampler: None,
            per_viewport: HashMap::new(),
        })
    };
    let stages = vec![
        (stage(Pass::ColourGrade), stage_order::EXTERNAL_DEFAULT),
        (stage(Pass::DepthFog), stage_order::EXTERNAL_DEFAULT + 1),
        (stage(Pass::EdgeDetect), stage_order::EXTERNAL_DEFAULT + 2),
    ];
    (stages, settings)
}

impl VfxStage {
    fn params(&self) -> ([f32; 4], [f32; 4]) {
        let s = self.settings.lock().unwrap();
        match self.pass {
            Pass::ColourGrade => (
                [
                    s.colour_grade.exposure,
                    s.colour_grade.contrast,
                    s.colour_grade.saturation,
                    0.0,
                ],
                [
                    s.colour_grade.tint[0],
                    s.colour_grade.tint[1],
                    s.colour_grade.tint[2],
                    0.0,
                ],
            ),
            Pass::DepthFog => (
                [s.depth_fog.near, s.depth_fog.far, s.depth_fog.amount, 0.0],
                [
                    s.depth_fog.colour[0],
                    s.depth_fog.colour[1],
                    s.depth_fog.colour[2],
                    0.0,
                ],
            ),
            Pass::EdgeDetect => (
                [
                    s.edge_detect.colour_strength,
                    s.edge_detect.depth_strength,
                    s.edge_detect.threshold,
                    0.0,
                ],
                [
                    s.edge_detect.colour[0],
                    s.edge_detect.colour[1],
                    s.edge_detect.colour[2],
                    0.0,
                ],
            ),
        }
    }
}

impl PostEffectStage for VfxStage {
    fn type_name(&self) -> &'static str {
        self.pass.label()
    }

    fn enabled(&self) -> bool {
        let s = self.settings.lock().unwrap();
        match self.pass {
            Pass::ColourGrade => s.colour_grade.enabled,
            Pass::DepthFog => s.depth_fog.enabled,
            Pass::EdgeDetect => s.edge_detect.enabled,
        }
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        if self.pass.reads_depth() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        let next = entries.len() as u32;
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: next + 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(self.pass.label()),
            entries: &entries,
        });
        self.sampler = Some(clamp_sampler(
            device,
            self.pass.label(),
            wgpu::FilterMode::Linear,
        ));
        self.bgl = Some(bgl);
        // The pipeline waits for the first resize signal, which carries the
        // target format.
        self.pipeline = None;
        self.per_viewport.clear();
    }

    fn on_viewport_resized(&mut self, device: &wgpu::Device, ctx: &PostEffectResizeContext<'_>) {
        let (Some(bgl), Some(sampler)) = (&self.bgl, &self.sampler) else {
            return;
        };
        if self.pipeline.is_none() {
            let shader = wgsl_module(device, self.pass.label(), self.pass.shader());
            self.pipeline = Some(build_post_effect_pipeline(
                device,
                self.pass.label(),
                &shader,
                bgl,
                ctx.target_format,
                None,
            ));
        }
        let (texture, input_view) =
            colour_target(device, self.pass.label(), ctx.scene_size, ctx.target_format);
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.pass.label()),
            size: std::mem::size_of::<VfxUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&input_view),
        }];
        if self.pass.reads_depth() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(ctx.scene_depth),
            });
        }
        let next = entries.len() as u32;
        entries.push(wgpu::BindGroupEntry {
            binding: next,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: next + 1,
            resource: uniform_buf.as_entire_binding(),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.pass.label()),
            layout: bgl,
            entries: &entries,
        });
        self.per_viewport.insert(
            ctx.viewport_index,
            VfxViewport {
                _texture: texture,
                input_view,
                uniform_buf,
                bind_group,
            },
        );
    }

    fn prepare(&mut self, queue: &wgpu::Queue, ctx: &PostEffectContext<'_>) {
        let (params0, params1) = self.params();
        let Some(vp) = self.per_viewport.get(&ctx.viewport_index) else {
            return;
        };
        let uniform = VfxUniform::new(ctx.scene_size, params0, params1);
        queue.write_buffer(&vp.uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
    }

    fn input_view(&self, viewport_index: usize) -> &wgpu::TextureView {
        &self.per_viewport[&viewport_index].input_view
    }

    fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        ctx: &PostEffectContext<'_>,
    ) {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return;
        };
        let Some(vp) = self.per_viewport.get(&ctx.viewport_index) else {
            return;
        };
        fullscreen_pass(
            encoder,
            self.pass.label(),
            target,
            wgpu::Color::BLACK,
            pipeline,
            &vp.bind_group,
        );
    }
}
