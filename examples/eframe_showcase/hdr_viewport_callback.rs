//! HDR viewport callback — renders through the full HDR pipeline into an
//! intermediate texture, then blits to the eframe render pass.
//!
//! The standard [`ViewportCallback`](super::viewport_callback::ViewportCallback)
//! calls `renderer.prepare()` + `renderer.paint()` (LDR path). That path bypasses
//! all post-processing (SSAO, bloom, FXAA, SSAA). This callback instead calls
//! `renderer.render()` which runs the full HDR pipeline, including SSAA, and
//! writes the finished frame into an intermediate texture. A simple fullscreen
//! blit then copies that texture into whatever render pass eframe provides.
//!
//! ## Usage
//! 1. At app startup, register [`HdrBlitResources`] in the egui-wgpu callback
//!    resources (see `init_hdr_blit_resources` below).
//! 2. In the showcase's render loop, use `HdrViewportCallback` instead of
//!    `ViewportCallback`.

use eframe::{
    egui_wgpu,
    wgpu::{self, include_wgsl},
};
use viewport_lib::{FrameData, ViewportRenderer};

// ---------------------------------------------------------------------------
// Blit resources (stored once in callback_resources)
// ---------------------------------------------------------------------------

/// GPU resources for the HDR→surface blit pass.
/// Stored in `callback_resources` under this type.
pub struct HdrBlitResources {
    pub texture: wgpu::Texture,
    pub blit_view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    pub format: wgpu::TextureFormat,
    pub size: [u32; 2],
}

impl HdrBlitResources {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let (bgl, pipeline) = create_blit_pipeline(device, format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr_blit_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let texture = create_intermediate_texture(device, format, width.max(1), height.max(1));
        let blit_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = create_blit_bind_group(device, &bgl, &blit_view, &sampler);

        Self {
            texture,
            blit_view,
            bind_group,
            bgl,
            pipeline,
            sampler,
            format,
            size: [width, height],
        }
    }

    /// Recreate the intermediate texture (and dependent bind group) if the
    /// viewport size has changed since last call. Idempotent if size matches.
    pub fn resize_if_needed(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.size == [w, h] {
            return;
        }
        self.texture = create_intermediate_texture(device, self.format, w, h);
        self.blit_view = self.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.bind_group =
            create_blit_bind_group(device, &self.bgl, &self.blit_view, &self.sampler);
        self.size = [w, h];
    }
}

fn create_intermediate_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr_intermediate_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn create_blit_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let shader = device.create_shader_module(include_wgsl!("shaders/hdr_blit.wgsl"));

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hdr_blit_bgl"),
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

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hdr_blit_pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hdr_blit_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (bgl, pipeline)
}

fn create_blit_bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hdr_blit_bind_group"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// Public init helper
// ---------------------------------------------------------------------------

/// Create and register [`HdrBlitResources`] into the egui-wgpu callback
/// resources. Call once at app startup after the renderer is set up.
pub fn init_hdr_blit_resources(
    render_state: &eframe::egui_wgpu::RenderState,
    format: wgpu::TextureFormat,
) {
    let res = HdrBlitResources::new(&render_state.device, format, 1, 1);
    render_state
        .renderer
        .write()
        .callback_resources
        .insert(res);
}

// ---------------------------------------------------------------------------
// Callback
// ---------------------------------------------------------------------------

pub struct HdrViewportCallback {
    pub frame: FrameData,
    pub viewport_size: [u32; 2],
}

impl egui_wgpu::CallbackTrait for HdrViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let [w, h] = self.viewport_size;

        // Step 1 — resize intermediate texture if needed.
        // Borrow scoped so it's fully released before we borrow the renderer.
        {
            let blit = callback_resources
                .get_mut::<HdrBlitResources>()
                .expect("HdrBlitResources must be registered at startup");
            blit.resize_if_needed(device, w, h);
        }

        // Step 2 — create a fresh view into the (possibly resized) texture.
        // create_view() returns an owned handle; does not extend the borrow above.
        let render_view = {
            let blit = callback_resources
                .get::<HdrBlitResources>()
                .unwrap();
            blit.texture.create_view(&wgpu::TextureViewDescriptor::default())
        };

        // Step 3 — render the full HDR frame into the intermediate texture.
        // renderer.render() calls prepare() internally, so we don't call it
        // separately (unlike ViewportCallback).
        let cmd = callback_resources
            .get_mut::<ViewportRenderer>()
            .expect("ViewportRenderer must be registered at startup")
            .render(device, queue, &render_view, &self.frame);

        vec![cmd]
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        // Blit the rendered intermediate texture to the eframe surface.
        let blit = callback_resources
            .get::<HdrBlitResources>()
            .expect("HdrBlitResources must be registered at startup");
        render_pass.set_pipeline(&blit.pipeline);
        render_pass.set_bind_group(0, &blit.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
