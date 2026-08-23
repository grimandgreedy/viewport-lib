//! Lazy pipeline creation for overlay text and solid screen-space quads.

/// Screen-space overlay text pipeline (glyph atlas quads), its layout, and sampler.
/// Lazily built on the first frame with non-empty labels.
#[derive(Default)]
pub(crate) struct OverlayTextResources {
    /// Render pipeline for screen-space text and solid overlay quads.
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout (group 0: atlas texture + sampler).
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    /// Linear sampler for the glyph atlas texture.
    pub(crate) sampler: Option<crate::gpu::Sampler>,
}

impl crate::resources::DeviceResources {
    /// Lazily create the overlay text render pipeline.
    ///
    /// No-op if already created.  Called from `prepare_viewport_internal()` when
    /// `frame.overlays.labels` is non-empty.
    pub(crate) fn ensure_overlay_text_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.overlay_text.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // binding 0: glyph atlas texture, binding 1: sampler, binding 2: clip-shape
        // storage buffer (shared clip model, evaluated per fragment).
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("overlay_text_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(
                                std::mem::size_of::<crate::resources::ClipShapeGpu>() as u64,
                            )
                            .unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

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
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[OverlayTextVertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState::default(),
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(crate::gpu::BlendState {
                            color: crate::gpu::BlendComponent {
                                src_factor: crate::gpu::BlendFactor::SrcAlpha,
                                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                                operation: crate::gpu::BlendOperation::Add,
                            },
                            alpha: crate::gpu::BlendComponent {
                                src_factor: crate::gpu::BlendFactor::One,
                                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                                operation: crate::gpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
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
    /// Index into the frame's clip-shape storage buffer, or `-1.0` for no clip.
    /// Selects the mask whose SDF (and parent chain) clips this vertex.
    pub clip_index: f32,
    /// Bounding-box clip in framebuffer pixels `[x0, y0, x1, y1]`; all-zero means
    /// no box clip. A cheap pre-clip that contains rectangular masks exactly and
    /// bounds the SDF test for the rest.
    pub clip_rect: [f32; 4],
}

impl OverlayTextVertex {
    /// wgpu vertex buffer layout matching `overlay_text.wgsl`.
    pub fn buffer_layout() -> crate::gpu::VertexBufferLayout<'static> {
        crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayTextVertex>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position vec2f
                crate::gpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: crate::gpu::VertexFormat::Float32x2,
                },
                // location 1: uv vec2f
                crate::gpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: crate::gpu::VertexFormat::Float32x2,
                },
                // location 2: colour vec4f
                crate::gpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 3: use_texture f32
                crate::gpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: crate::gpu::VertexFormat::Float32,
                },
                // location 4: clip_index f32
                crate::gpu::VertexAttribute {
                    offset: 36,
                    shader_location: 4,
                    format: crate::gpu::VertexFormat::Float32,
                },
                // location 5: clip_rect vec4f (framebuffer px x0,y0,x1,y1)
                crate::gpu::VertexAttribute {
                    offset: 40,
                    shader_location: 5,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Per-frame GPU data for batched overlay label rendering.
pub(crate) struct LabelGpuData {
    /// Vertex buffer containing all label geometry for this frame.
    pub vertex_buf: crate::gpu::Buffer,
    /// Number of vertices to draw.
    pub vertex_count: u32,
    /// Bind group referencing the glyph atlas texture, sampler, and clip buffer.
    pub bind_group: crate::gpu::BindGroup,
    /// Backing clip-shape storage buffer for `bind_group` (binding 2). Kept alive
    /// alongside it.
    pub _clip_buf: crate::gpu::Buffer,
}
