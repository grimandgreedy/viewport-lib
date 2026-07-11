//! Lazy pipeline creation for SDF overlay shapes.

use crate::renderer::OverlayTextureId;

/// SDF overlay shape pipelines (solid + textured fill) and their shared sampler.
/// Lazily built on the first frame with non-empty shapes.
#[derive(Default)]
pub(crate) struct OverlayShapeResources {
    /// Render pipeline for screen-space SDF shapes (rounded rects, circles, etc.).
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    /// Render pipeline for SDF shapes with texture fill.
    pub(crate) tex_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the texture pipeline (group 0: texture + sampler).
    pub(crate) tex_bgl: Option<wgpu::BindGroupLayout>,
    /// Clamp-to-edge linear sampler shared across all texture shape bind groups.
    pub(crate) tex_sampler: Option<wgpu::Sampler>,
}

/// Fullscreen separable Gaussian blur pipeline used to produce the blurred
/// scene texture for `backdrop_blur` overlay shapes.
#[derive(Default)]
pub(crate) struct BackdropBlurResources {
    /// Fullscreen separable Gaussian blur pipeline.
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout (group 0: source texture + sampler + uniforms).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Linear clamp sampler shared by blur passes.
    pub(crate) sampler: Option<wgpu::Sampler>,
}

impl crate::resources::DeviceResources {
    /// Lazily create the SDF overlay shape render pipeline.
    ///
    /// No-op if already created. Called from `prepare_viewport_internal()` when
    /// `frame.overlays.shapes` is non-empty.
    pub(crate) fn ensure_overlay_shape_pipeline(&mut self, device: &wgpu::Device) {
        if self.overlay_shape.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_shape_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_shape.wgsl",
            crate::resources::builders::wgsl_source!("overlay_shape"),
        );

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_shape_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[OverlayShapeVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
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
            multiview: None,
            cache: None,
        });

        self.overlay_shape.pipeline = Some(pipeline);
    }

    /// Lazily create the SDF overlay shape render pipeline with texture fill.
    ///
    /// No-op if already created. Called from `prepare_viewport_internal()` when
    /// any shape in `OverlayFrame.shapes` carries an `OverlayTextureId`.
    pub(crate) fn ensure_overlay_shape_tex_pipeline(&mut self, device: &wgpu::Device) {
        if self.overlay_shape.tex_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "overlay_shape_tex_bgl",
            wgpu::ShaderStages::FRAGMENT,
        );

        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "overlay_shape_tex_sampler");

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_shape_tex_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_shape_tex.wgsl",
            crate::resources::builders::wgsl_source!("overlay_shape_tex"),
        );

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_shape_tex_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[OverlayShapeTexVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
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
            multiview: None,
            cache: None,
        });

        self.overlay_shape.tex_bgl = Some(bgl);
        self.overlay_shape.tex_sampler = Some(sampler);
        self.overlay_shape.tex_pipeline = Some(pipeline);
    }

    /// Upload RGBA8 pixel data as a persistent texture for overlay shape fills.
    ///
    /// Returns an `OverlayTextureId` that can be stored in
    /// `OverlayShapeItem::texture`. The texture persists for the lifetime of
    /// this `DeviceResources`.
    ///
    /// `rgba_data` must contain exactly `width * height * 4` bytes in
    /// row-major, top-to-bottom order. The data is treated as sRGB-encoded
    /// (standard 8-bit image data).
    pub fn upload_overlay_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> OverlayTextureId {
        assert_eq!(
            rgba_data.len(),
            (width * height * 4) as usize,
            "upload_overlay_texture: rgba_data length does not match width * height * 4"
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay_shape_tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = self
            .content
            .overlay_textures
            .push(OverlayShapeTextureEntry {
                _texture: texture,
                view,
            });
        OverlayTextureId(id as u64)
    }

    /// Start an asynchronous overlay texture upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Texture
    /// creation and `queue.write_texture` run on a worker thread on cloned
    /// `Device` and `Queue` handles; the apply step inserts the prepared
    /// `OverlayShapeTextureEntry` into the store. Take the resulting
    /// [`OverlayTextureId`] via
    /// [`upload_result_overlay_texture`](Self::upload_result_overlay_texture).
    ///
    /// Ownership of `rgba_data` transfers into the worker.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// before submission when `rgba_data.len() != width * height * 4`.
    pub fn begin_upload_overlay_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: Vec<u8>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        let expected = (width * height * 4) as usize;
        if rgba_data.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba_data.len(),
            });
        }

        let slot = crate::resources::ResultSlot::<OverlayTextureId>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let queue_for_worker = queue.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let texture = device_for_worker.create_texture(&wgpu::TextureDescriptor {
                    label: Some("overlay_shape_tex"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue_for_worker.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rgba_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut crate::resources::DeviceResources| {
                        let id =
                            resources
                                .content
                                .overlay_textures
                                .push(OverlayShapeTextureEntry {
                                    _texture: texture,
                                    view,
                                });
                        slot_for_apply.set(OverlayTextureId(id as u64));
                    }),
                ))
            })
        };

        self.job_results
            .overlay_texture
            .lock()
            .expect("overlay texture result map poisoned")
            .insert(id, slot);
        Ok(id)
    }

    /// Take the [`OverlayTextureId`] produced by a completed
    /// [`begin_upload_overlay_texture`](Self::begin_upload_overlay_texture) job.
    pub fn upload_result_overlay_texture(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<OverlayTextureId> {
        let mut map = self
            .job_results
            .overlay_texture
            .lock()
            .expect("overlay texture result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(tid) => {
                map.remove(&id);
                Ok(tid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Lazily create the separable Gaussian blur pipeline used to blur the
    /// scene texture for `backdrop_blur` overlay shapes.
    ///
    /// No-op if already created.
    pub(crate) fn ensure_backdrop_blur_pipeline(&mut self, device: &wgpu::Device) {
        if self.backdrop_blur.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop_blur_bgl"),
            entries: &[
                // binding 0: source texture
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
                // binding 1: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: blur parameters uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "backdrop_blur_sampler");

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("backdrop_blur_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "backdrop_blur.wgsl",
            crate::resources::builders::wgsl_source!("backdrop_blur"),
        );

        let pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "backdrop_blur_pipeline",
            &layout,
            &shader,
            self.target_format,
            None,
        );

        self.backdrop_blur.bgl = Some(bgl);
        self.backdrop_blur.sampler = Some(sampler);
        self.backdrop_blur.pipeline = Some(pipeline);
    }
}

/// Per-vertex data for SDF overlay shapes.
///
/// Each shape is a bounding quad (6 vertices). The fragment shader uses
/// `local_pos` and `half_size` to evaluate a signed-distance function,
/// producing anti-aliased fill and border.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayShapeVertex {
    /// NDC position (xy).
    pub position: [f32; 2],
    /// Position relative to shape centre, in logical pixels.
    pub local_pos: [f32; 2],
    /// RGBA fill colour (pre-multiplied opacity).
    pub fill_colour: [f32; 4],
    /// RGBA border colour (pre-multiplied opacity).
    pub border_colour: [f32; 4],
    /// Half-extents of the shape bounding box in logical pixels.
    pub half_size: [f32; 2],
    /// Shape-specific radii. For RoundedRect: [top-left, top-right,
    /// bottom-right, bottom-left]. For Rect: uniform radius in [0].
    /// Unused components are zero.
    pub radii: [f32; 4],
    /// Border thickness in logical pixels.
    pub border_width: f32,
    /// Encoded shape type: 0 = Rect/RoundedRect, 1 = Circle, 2 = Ellipse, 3 = Capsule.
    pub shape_type: f32,
    /// RGBA end colour for linear gradient (equals fill_colour for solid fill).
    pub fill_colour2: [f32; 4],
    /// Gradient parameters: `[type, angle, stop_count, _pad]`.
    /// `type` selects solid/linear/radial/conical; `stop_count` is the
    /// number of active gradient stops (0 for solid, 2..4 otherwise).
    pub gradient_params: [f32; 4],
    /// RGBA shadow colour (pre-multiplied opacity).
    pub shadow_colour: [f32; 4],
    /// Shadow parameters: x = radius (pixels), y = offset_x, z = offset_y.
    pub shadow_params: [f32; 4],
    /// Clip rectangle in framebuffer pixels (x0, y0, x1, y1). All-zero means
    /// no clipping. Fragments outside the rect are discarded.
    pub clip_rect: [f32; 4],
    /// Rotation around the shape centre in radians. Applied to `local_pos`
    /// before SDF evaluation so the shape rotates inside its axis-aligned
    /// bounding box.
    pub rotation: f32,
    /// Third gradient colour stop. Unused for 2-stop and solid fills.
    pub stop_colour_c: [f32; 4],
    /// Fourth gradient colour stop. Unused for 2-stop and solid fills.
    pub stop_colour_d: [f32; 4],
    /// Stop positions in `[0, 1]` along the gradient axis, sorted ascending.
    /// For 2-stop fills only `[0]` and `[1]` are valid; remaining entries
    /// are 1.0 sentinels. The active stop count lives in
    /// `gradient_params[2]`.
    pub stop_positions: [f32; 4],
}

impl OverlayShapeVertex {
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayShapeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position vec2f
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 1: local_pos vec2f
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 2: fill_colour vec4f
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 3: border_colour vec4f
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 4: half_size vec2f
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 5: radii vec4f
                wgpu::VertexAttribute {
                    offset: 56,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 6: shape_meta vec2f (border_width, shape_type) -- combined
                wgpu::VertexAttribute {
                    offset: 72,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 7: stop_positions vec4f (gradient stop positions)
                wgpu::VertexAttribute {
                    offset: 196,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 8: fill_colour2 vec4f (end colour for gradient)
                wgpu::VertexAttribute {
                    offset: 80,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 9: gradient_params vec4f (type, angle, stop_count, pad)
                wgpu::VertexAttribute {
                    offset: 96,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 10: shadow_colour vec4f
                wgpu::VertexAttribute {
                    offset: 112,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 11: shadow_params vec4f (radius, offset_x, offset_y, border_mode)
                wgpu::VertexAttribute {
                    offset: 128,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 12: clip_rect vec4f (x0, y0, x1, y1 in framebuffer pixels)
                wgpu::VertexAttribute {
                    offset: 144,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 13: rotation f32 (radians)
                wgpu::VertexAttribute {
                    offset: 160,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Float32,
                },
                // location 14: stop_colour_c vec4f
                wgpu::VertexAttribute {
                    offset: 164,
                    shader_location: 14,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 15: stop_colour_d vec4f
                wgpu::VertexAttribute {
                    offset: 180,
                    shader_location: 15,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Per-vertex data for SDF textured overlay shapes.
///
/// Same layout as `OverlayShapeVertex` with an additional UV field at the end.
/// Used by the texture pipeline; `fill_colour` acts as a tint multiplied with
/// the sampled texel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayShapeTexVertex {
    /// NDC position (xy).
    pub position: [f32; 2],
    /// Position relative to shape centre, in logical pixels.
    pub local_pos: [f32; 2],
    /// RGBA tint colour (pre-multiplied opacity). Multiplied with texture sample.
    pub fill_colour: [f32; 4],
    /// RGBA border colour (pre-multiplied opacity).
    pub border_colour: [f32; 4],
    /// Half-extents of the shape bounding box in logical pixels.
    pub half_size: [f32; 2],
    /// Shape-specific radii (same encoding as `OverlayShapeVertex`).
    pub radii: [f32; 4],
    /// Border thickness in logical pixels.
    pub border_width: f32,
    /// Encoded shape type (same values as `OverlayShapeVertex`).
    pub shape_type: f32,
    /// Texture UV coordinates. (0,0) = top-left of image, (1,1) = bottom-right.
    /// Slightly outside [0,1] in the border/AA padding region.
    pub uv: [f32; 2],
    /// RGBA shadow colour (pre-multiplied opacity).
    pub shadow_colour: [f32; 4],
    /// Shadow parameters: x = radius (pixels), y = offset_x, z = offset_y, w = border_mode.
    pub shadow_params: [f32; 4],
    /// Per-shape flags.
    /// - `x` = is_backdrop_blur (0.0 = regular tinted sample; 1.0 = the bound
    ///   texture is the scene-blur output composited under the tint).
    /// - `y` = nine-slice centre tile mode (0 = stretch, 1 = tile).
    /// - `z` = nine-slice edge tile mode (0 = stretch, 1 = tile).
    /// - `w` = nine-slice enabled flag (0 = disabled, 1 = enabled).
    pub extras: [f32; 4],
    /// Nine-slice insets in texture UV space, `[top, right, bottom, left]`.
    /// Unused when `extras.w == 0`.
    pub nine_slice_uv: [f32; 4],
    /// Nine-slice insets as fractions of the shape's bounding box,
    /// `[top, right, bottom, left]`. Unused when `extras.w == 0`.
    pub nine_slice_frac: [f32; 4],
    /// Texture transform part A: `[offset_x, offset_y, scale_x, scale_y]`.
    /// `scale = 1.0` is 1:1. Applies when 9-slice is disabled.
    pub texture_transform_a: [f32; 4],
    /// Texture transform part B:
    /// `[rotation_radians, tile_mode (0/1/2), flip_x (0/1), flip_y (0/1)]`.
    pub texture_transform_b: [f32; 4],
}

impl OverlayShapeTexVertex {
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayShapeTexVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position vec2f
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 1: local_pos vec2f
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 2: fill_colour vec4f (tint)
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 3: border_colour vec4f
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 4: half_size vec2f
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 5: radii vec4f
                wgpu::VertexAttribute {
                    offset: 56,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 6: border_width f32
                wgpu::VertexAttribute {
                    offset: 72,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
                // location 7: shape_type f32
                wgpu::VertexAttribute {
                    offset: 76,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
                // location 8: uv vec2f
                wgpu::VertexAttribute {
                    offset: 80,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // location 9: shadow_colour vec4f
                wgpu::VertexAttribute {
                    offset: 88,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 10: shadow_params vec4f (radius, offset_x, offset_y, border_mode)
                wgpu::VertexAttribute {
                    offset: 104,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 11: extras vec4f (blur, centre_mode, edge_mode, nine_slice_enabled)
                wgpu::VertexAttribute {
                    offset: 120,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 12: nine_slice_uv vec4f
                wgpu::VertexAttribute {
                    offset: 136,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 13: nine_slice_frac vec4f
                wgpu::VertexAttribute {
                    offset: 152,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 14: texture_transform_a vec4f (offset.xy, scale.xy)
                wgpu::VertexAttribute {
                    offset: 168,
                    shader_location: 14,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // location 15: texture_transform_b vec4f (rotation, tile_mode, flip_x, flip_y)
                wgpu::VertexAttribute {
                    offset: 184,
                    shader_location: 15,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// One batch of textured SDF overlay shapes sharing a single texture.
pub(crate) struct OverlayShapeTexBatch {
    pub vertex_buf: wgpu::Buffer,
    pub vertex_count: u32,
    pub bind_group: wgpu::BindGroup,
}

/// Persistent texture entry for an overlay shape texture fill.
///
/// Stored in `DeviceResources::overlay_textures`.
pub(crate) struct OverlayShapeTextureEntry {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// Per-frame GPU data for batched SDF overlay shape rendering.
pub(crate) struct OverlayShapeGpuData {
    /// Vertex buffer for solid (non-textured) shapes. `None` when all shapes are textured.
    pub vertex_buf: Option<wgpu::Buffer>,
    /// Number of solid vertices. Zero when all shapes are textured.
    pub vertex_count: u32,
    /// One batch per unique texture, drawn after solid shapes.
    pub tex_batches: Vec<OverlayShapeTexBatch>,
    /// Vertex buffer for backdrop-blur shapes (frosted glass). Uses the same
    /// vertex layout as `OverlayShapeTexVertex` with screen-space UVs.
    /// The bind group is created at render time once the blurred scene texture
    /// is available.
    pub blur_vertex_buf: Option<wgpu::Buffer>,
    /// Number of blur backdrop vertices.
    pub blur_vertex_count: u32,
    /// Maximum `backdrop_blur` value across all blur shapes this frame.
    /// Used as the spread parameter for the Gaussian blur passes.
    pub max_blur_radius: f32,
}
