//! Lazy pipeline creation for overlay text and solid screen-space quads.

/// Screen-space overlay text pipeline (glyph atlas quads), its layout, and sampler.
/// Lazily built on the first frame with non-empty labels.
#[derive(Default)]
pub(crate) struct OverlayTextResources {
    /// Render pipeline for screen-space text and solid overlay quads.
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout (group 0: atlas texture + sampler).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Linear sampler for the glyph atlas texture.
    pub(crate) sampler: Option<wgpu::Sampler>,
}

impl crate::resources::DeviceResources {
    /// Lazily create the overlay text render pipeline.
    ///
    /// No-op if already created.  Called from `prepare_viewport_internal()` when
    /// `frame.overlays.labels` is non-empty.
    pub(crate) fn ensure_overlay_text_pipeline(&mut self, device: &wgpu::Device) {
        if self.overlay_text.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // binding 0: glyph atlas texture, binding 1: sampler.
        let bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "overlay_text_bgl",
            wgpu::ShaderStages::FRAGMENT,
        );

        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "overlay_text_sampler");

        let layout =
            crate::resources::builders::pipeline_layout(device, "overlay_text_layout", &[&bgl]);

        let shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_text.wgsl",
            crate::resources::builders::wgsl_source!("overlay_text"),
        );

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_text_pipeline",
                layout: &layout,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[OverlayTextVertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    wgpu::CompareFunction::Always,
                )),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                cache: None,
            },
        );

        self.overlay_text.bgl = Some(bgl);
        self.overlay_text.sampler = Some(sampler);
        self.overlay_text.pipeline = Some(pipeline);
    }
}

/// Per-vertex data for overlay text and solid screen-space quads (labels, backgrounds, leader lines).
///
/// All fields are packed into a single vertex to allow batching every overlay
/// quad into one draw call per frame.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayTextVertex {
    /// NDC position (xy, z=0 w=1 in shader).
    pub position: [f32; 2],
    /// Atlas UV coordinates.  Ignored when `use_texture` is 0.
    pub uv: [f32; 2],
    /// RGBA tint colour.
    pub colour: [f32; 4],
    /// 1.0 = sample glyph atlas alpha, 0.0 = solid colour.
    pub use_texture: f32,
    pub _pad: f32,
}

impl OverlayTextVertex {
    /// wgpu vertex buffer layout matching `overlay_text.wgsl`.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayTextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position vec2f
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 1: uv vec2f
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 2: colour vec4f
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 3: use_texture f32
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Per-frame GPU data for batched overlay label rendering.
pub(crate) struct LabelGpuData {
    /// Vertex buffer containing all label geometry for this frame.
    pub vertex_buf: wgpu::Buffer,
    /// Number of vertices to draw.
    pub vertex_count: u32,
    /// Bind group referencing the glyph atlas texture and sampler.
    pub bind_group: wgpu::BindGroup,
}
