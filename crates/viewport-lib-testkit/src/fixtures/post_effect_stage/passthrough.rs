//! A post-effect stage that copies its input through, scaled by a constant.

use crate::fixtures::CallLog;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use viewport_lib::plugin_api::build_post_effect_pipeline;
use viewport_lib::plugin_api::shared_wgsl::POST_EFFECT_VS_WGSL;
use viewport_lib::wgpu;
use viewport_lib::{PostEffectContext, PostEffectResizeContext, PostEffectStage};

/// Samples [`input_view`](PostEffectStage::input_view) and writes it to the
/// target multiplied by a constant scale, recording `init_gpu`,
/// `on_viewport_resized`, `prepare` and `encode`.
///
/// A scale below 1 darkens the frame, so a test can assert the stage was in
/// the image path at all; two stages with different scales and different
/// order keys multiply in chain order, which is how the ordering assertion
/// works.
pub struct PassthroughPostEffectStage {
    log: CallLog,
    label: &'static str,
    scale: f32,
    enabled: Arc<AtomicBool>,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Built on the first resize signal: the pipeline needs the renderer's
    /// LDR target format, which only the resize context carries.
    pipeline: Option<wgpu::RenderPipeline>,
    per_viewport: HashMap<usize, (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)>,
}

impl PassthroughPostEffectStage {
    /// A stage logging under `label` that scales its input by `scale`.
    ///
    /// `device` is needed up front for the layout and sampler, which do not
    /// depend on the target format.
    pub fn new(
        device: &wgpu::Device,
        log: CallLog,
        label: &'static str,
        scale: f32,
        enabled: Arc<AtomicBool>,
    ) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("passthrough_stage_bgl"),
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
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("passthrough_stage_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            log,
            label,
            scale,
            enabled,
            bind_group_layout,
            sampler,
            pipeline: None,
            per_viewport: HashMap::new(),
        }
    }

    fn record(&self, entry: String) {
        self.log.record(format!("{}:{entry}", self.label));
    }

    fn build_pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let source = format!(
            "{POST_EFFECT_VS_WGSL}
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

@fragment
fn fs_main(in: ViewportPostVsOut) -> @location(0) vec4<f32> {{
    let c = textureSample(input_texture, input_sampler, in.uv);
    return vec4<f32>(c.rgb * {scale:?}, 1.0);
}}
",
            scale = self.scale
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("passthrough_stage_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        build_post_effect_pipeline(
            device,
            "passthrough_stage_pipeline",
            &shader,
            &self.bind_group_layout,
            format,
            None,
        )
    }
}

impl PostEffectStage for PassthroughPostEffectStage {
    fn type_name(&self) -> &'static str {
        self.label
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn init_gpu(&mut self, _device: &wgpu::Device) {
        self.record("init_gpu".into());
    }

    fn on_device_recreated(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        self.record("on_device_recreated".into());
        self.pipeline = None;
        self.per_viewport.clear();
    }

    fn on_viewport_resized(&mut self, device: &wgpu::Device, ctx: &PostEffectResizeContext<'_>) {
        self.record(format!(
            "resize:{}:{}x{}",
            ctx.viewport_index, ctx.scene_size[0], ctx.scene_size[1]
        ));
        if self.pipeline.is_none() {
            self.pipeline = Some(self.build_pipeline(device, ctx.target_format));
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("passthrough_stage_input"),
            size: wgpu::Extent3d {
                width: ctx.scene_size[0],
                height: ctx.scene_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ctx.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("passthrough_stage_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.per_viewport
            .insert(ctx.viewport_index, (texture, view, bind_group));
    }

    fn prepare(&mut self, _queue: &wgpu::Queue, ctx: &PostEffectContext<'_>) {
        self.record(format!("prepare:{}", ctx.viewport_index));
    }

    fn input_view(&self, viewport_index: usize) -> &wgpu::TextureView {
        &self.per_viewport[&viewport_index].1
    }

    fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        ctx: &PostEffectContext<'_>,
    ) {
        self.record(format!("encode:{}", ctx.viewport_index));
        let Some(pipeline) = self.pipeline.as_ref() else {
            return;
        };
        let (_, _, bind_group) = &self.per_viewport[&ctx.viewport_index];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("passthrough_stage_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
