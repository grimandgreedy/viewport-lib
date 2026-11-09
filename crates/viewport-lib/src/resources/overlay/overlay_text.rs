//! Lazy pipeline creation for overlay text and solid screen-space quads.

/// Screen-space overlay text pipeline (glyph atlas quads), its layout, and sampler.
/// Lazily built on the first frame with non-empty labels.
#[derive(Default)]
pub(crate) struct OverlayTextResources {
    /// Render pipeline for screen-space text and solid overlay quads.
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// The text pipeline for each blend mode other than source-over that a
    /// frame has used.
    pub(crate) variants: crate::resources::overlay::overlay_shape::OverlayBlendVariants,
    /// The text pipeline's layout and shader, kept to build blend variants.
    pub(crate) layout: Option<crate::gpu::PipelineLayout>,
    pub(crate) shader: Option<crate::gpu::ShaderModule>,
    /// Bind group layout (group 0: atlas texture + sampler).
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    /// Linear sampler for the glyph atlas texture.
    pub(crate) sampler: Option<crate::gpu::Sampler>,
}

impl OverlayTextResources {
    /// The text pipeline that draws with `blend`, if it has been built.
    pub(crate) fn pipeline_for(
        &self,
        blend: crate::renderer::OverlayBlend,
    ) -> Option<&crate::gpu::RenderPipeline> {
        if blend == crate::renderer::OverlayBlend::SourceOver {
            self.pipeline.as_ref()
        } else {
            self.variants.get(blend)
        }
    }
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

        // binding 0: glyph atlas texture, binding 1: sampler, binding 2: the
        // clip-shape uniform array (shared clip model, evaluated per fragment).
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
                crate::resources::overlay::overlay_shape::clip_shapes_entry(2),
                // binding 3: viewport-size uniform (logical w, h). The vertex
                // shader maps local-pixel positions to NDC with it.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: the coverage atlas vector-shape cells sample.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "overlay_text_sampler");

        // Group 1: the shared fill module's records, image and sampler.
        let fill_bgl = self.overlay_fill_bgl(device).clone();
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "overlay_text_layout",
            &[&bgl, &fill_bgl],
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_text.wgsl",
            crate::resources::builders::wgsl_source!("overlay_text"),
        );

        let pipeline = self.overlay_text_variant(
            device,
            &layout,
            &shader,
            crate::renderer::OverlayBlend::SourceOver,
        );

        self.overlay_text.bgl = Some(bgl);
        self.overlay_text.sampler = Some(sampler);
        self.overlay_text.pipeline = Some(pipeline);
        self.overlay_text.layout = Some(layout);
        self.overlay_text.shader = Some(shader);
    }

    /// The text pipeline for `blend`, built on first use, as for the shape
    /// pipeline.
    pub(crate) fn ensure_overlay_text_blend(
        &mut self,
        device: &crate::gpu::Device,
        blend: crate::renderer::OverlayBlend,
    ) {
        self.ensure_overlay_text_pipeline(device);
        if self.overlay_text.variants.get(blend).is_some()
            || blend == crate::renderer::OverlayBlend::SourceOver
        {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let (Some(layout), Some(shader)) = (
            self.overlay_text.layout.clone(),
            self.overlay_text.shader.clone(),
        ) else {
            return;
        };
        let pipeline = self.overlay_text_variant(device, &layout, &shader, blend);
        self.overlay_text.variants.insert(blend, pipeline);
    }

    fn overlay_text_variant(
        &self,
        device: &crate::gpu::Device,
        layout: &crate::gpu::PipelineLayout,
        shader: &crate::gpu::ShaderModule,
        blend: crate::renderer::OverlayBlend,
    ) -> crate::gpu::RenderPipeline {
        let (blend_state, fs_entry) =
            crate::resources::overlay::overlay_shape::overlay_blend_target(blend);
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_text_pipeline",
                layout,
                vertex_module: shader,
                vertex_entry: "vs_main",
                // The per-draw instance is a second, per-instance stream: slot 0
                // is the identity immediate draws use, retained groups draw over
                // their own slot.
                vertex_buffers: &[
                    OverlayTextVertex::buffer_layout(),
                    crate::resources::OverlayInstance::buffer_layout(
                        &crate::resources::overlay::geometry::TEXT_INSTANCE_ATTRIBUTES,
                    ),
                ],
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    self.scene_depth_format(),
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState::default(),
                fragment: Some(crate::gpu::FragmentState {
                    module: shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(blend_state),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                cache: None,
            },
        )
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
    /// 0.0 = solid colour, 1.0 = tint the glyph atlas's coverage, 2.0 = the
    /// glyph atlas's colour as-is, 3.0 = tint the coverage atlas's cell.
    pub use_texture: f32,
    /// Index into the frame's clip-shape array, or `-1.0` for no clip.
    /// Selects the mask whose SDF (and parent chain) clips this vertex.
    pub clip_index: f32,
    /// Bounding-box clip in framebuffer pixels `[x0, y0, x1, y1]`; all-zero means
    /// no box clip. A cheap pre-clip that contains rectangular masks exactly and
    /// bounds the SDF test for the rest.
    pub clip_rect: [f32; 4],
    /// Whether a retained group's per-frame tint reaches this vertex: `1.0` for
    /// the item's own content, `0.0` for a shadow layer baked into the same
    /// stream.
    ///
    /// A tint never reaches a shadow layer, and on the retained path the
    /// shadows are geometry in this very buffer, so the rule has to ride the
    /// vertex. Immediate draws take the identity instance, whose tint changes
    /// nothing either way.
    pub group_tint: f32,
    /// `[local_x, local_y, fill_index, 0]`: the vertex's offset from the centre
    /// of the box its item's fill spans, in the item's own frame, and the first
    /// texel of that fill's record, or `-1` when the colour alone is the fill.
    pub fill: [f32; 4],
    /// `[dir_x, dir_y, side, max_inset]`: the vertex's place in an
    /// anti-aliasing fringe, which the vertex shader resolves in device pixels.
    ///
    /// `side` is `0` for a vertex outside any fringe (glyphs, quads), which
    /// draws at full coverage where it is. `-1` is the inner row of a fringe:
    /// the vertex moves half a device pixel against `dir` (never further than
    /// `max_inset`, in the same units as `position`) and draws at full
    /// coverage, scaled down when `max_inset` is under half a pixel so a
    /// hairline keeps its weight. `+1` is the outer row: half a device pixel
    /// along `dir`, at zero coverage. `dir` is the edge's outward normal, with
    /// a mitre's length at a join.
    pub aa: [f32; 4],
}

impl OverlayTextVertex {
    /// The `fill` of a vertex that reads no fill record.
    pub const NO_FILL: [f32; 4] = [0.0, 0.0, -1.0, 0.0];
    /// The `aa` of a vertex outside any fringe.
    pub const NO_AA: [f32; 4] = [0.0; 4];
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
                // location 6: group_tint f32
                crate::gpu::VertexAttribute {
                    offset: 56,
                    shader_location: 6,
                    format: crate::gpu::VertexFormat::Float32,
                },
                // location 12 (7..11 are the instance stream): fill vec4f
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayTextVertex, fill) as u64,
                    shader_location: 12,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
                // location 13: aa vec4f (fringe direction, side, max inset)
                crate::gpu::VertexAttribute {
                    offset: std::mem::offset_of!(OverlayTextVertex, aa) as u64,
                    shader_location: 13,
                    format: crate::gpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Per-frame GPU data for batched overlay label rendering.
pub(crate) struct LabelGpuData {
    /// The frame's text-stream vertices in draw order, written once the draw
    /// list is built.
    pub vertex_buf: Option<crate::gpu::Buffer>,
    /// Group 1 bind groups: the frame's fill records with the blank image at
    /// index 0, then one per image a text-stream item is filled from.
    pub fill_bind_groups: Vec<crate::gpu::BindGroup>,
    /// Bind group referencing the glyph atlas texture, sampler, and clip buffer.
    pub bind_group: crate::gpu::BindGroup,
    /// Backing clip-shape uniform for `bind_group` (binding 2). Kept alive
    /// alongside it.
    pub _clip_buf: crate::gpu::Buffer,
}
