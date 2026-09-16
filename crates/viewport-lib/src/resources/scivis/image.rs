use super::*;

impl DeviceResources {
    // -------------------------------------------------------------------------
    // Screen-space image overlays
    // -------------------------------------------------------------------------

    /// Lazily create the screen-space image render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when
    /// `frame.scene.screen_images` is non-empty.
    pub(crate) fn ensure_screen_image_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.screen_image.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let shader = crate::resources::builders::wgsl_module(
            device,
            "screen_image_shader",
            crate::resources::builders::wgsl_source!("screen_image"),
        );

        // binding 0: ScreenImageUniform, binding 1: texture_2d<f32>, binding 2: sampler.
        let bgl = crate::resources::builders::uniform_texture_sampler_bgl(
            device,
            "screen_image_bgl",
            crate::gpu::ShaderStages::VERTEX_FRAGMENT,
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let layout =
            crate::resources::builders::pipeline_layout(device, "screen_image_layout", &[&bgl]);

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "screen_image_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[],
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(crate::gpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                // Use Always depth compare (never test) so screen images are always on top.
                // No depth writes. Format must match the depth attachment of the render pass.
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
            },
        );

        self.screen_image.bgl = Some(bgl);
        self.screen_image.pipeline = Some(pipeline);
    }

    /// Lazily create the depth-composite screen-image render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when any submitted
    /// `ScreenImageItem` carries per-pixel depth data.
    pub(crate) fn ensure_screen_image_dc_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.screen_image.dc_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let shader = crate::resources::builders::wgsl_module(
            device,
            "screen_image_dc_shader",
            crate::resources::builders::wgsl_source!("screen_image_dc"),
        );

        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("screen_image_dc_bgl"),
            entries: &[
                // binding 0: ScreenImageUniform
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: colour texture_2d<f32>
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 2: sampler (filtering, for colour texture)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: R32Float depth texture (non-filterable, read via textureLoad)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
            ],
        });

        let layout =
            crate::resources::builders::pipeline_layout(device, "screen_image_dc_layout", &[&bgl]);

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "screen_image_dc_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[],
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(crate::gpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                // Depth test: discard fragments whose image depth exceeds scene depth.
                // depth_write_enabled: false so the scene depth buffer is not modified.
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::LessEqual,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
            },
        );

        self.screen_image.dc_bgl = Some(bgl);
        self.screen_image.dc_pipeline = Some(pipeline);
    }

    /// Upload one [`ScreenImageItem`] to the GPU and return its per-frame GPU data.
    ///
    /// Creates a new RGBA8Unorm texture each call : intended for per-frame data.
    /// The returned [`ScreenImageGpuData`] is valid only for one frame.
    pub(crate) fn upload_screen_image(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::ScreenImageItem,
        viewport_w: f32,
        viewport_h: f32,
    ) -> ScreenImageGpuData {
        // Infer the physical texture dimensions from the pixel buffer.
        // item.width/height are in logical pixels (the visual size); callers may
        // supply a higher-resolution buffer (e.g. width*ppp x height*ppp) for
        // crisp HiDPI rendering. The integer scale factor is derived from the
        // ratio of pixel count to logical area.
        let logical_area = (item.width * item.height) as usize;
        let tex_scale = if logical_area > 0 {
            let ratio = item.pixels.len() / logical_area;
            (ratio as f32).sqrt().round() as u32
        } else {
            1
        }
        .max(1);
        let tex_w = (item.width * tex_scale).max(1);
        let tex_h = (item.height * tex_scale).max(1);

        // Create texture from pixel data.
        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("screen_image_tex"),
            size: crate::gpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8UnormSrgb,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        if !item.pixels.is_empty() && item.width > 0 && item.height > 0 {
            let raw: Vec<u8> = item.pixels.iter().flat_map(|p| p.iter().copied()).collect();
            let needed = (tex_w as usize) * (tex_h as usize) * 4;
            if raw.len() < needed {
                tracing::warn!(
                    target: "viewport_lib::screen_image",
                    width = item.width,
                    height = item.height,
                    pixels_len = item.pixels.len(),
                    inferred_tex_w = tex_w,
                    inferred_tex_h = tex_h,
                    expected_bytes = needed,
                    actual_bytes = raw.len(),
                    "ScreenImageItem pixel buffer is smaller than the inferred texture size \
                     (item.width * item.height does not divide pixels.len() into a square scale \
                     factor). Skipping upload; the image will render blank. Resize the buffer or \
                     set item.width / item.height to match the buffer's actual physical resolution."
                );
            } else {
                queue.write_texture(
                    crate::gpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: crate::gpu::Origin3d::ZERO,
                        aspect: crate::gpu::TextureAspect::All,
                    },
                    &raw,
                    crate::gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(tex_w * 4),
                        rows_per_image: Some(tex_h),
                    },
                    crate::gpu::Extent3d {
                        width: tex_w,
                        height: tex_h,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "screen_image_sampler");

        // Compute NDC extents from anchor, image size, and scale.
        let [ndc_min_x, ndc_max_x, ndc_min_y, ndc_max_y] = crate::renderer::viewport_anchored_ndc(
            item.anchor_x,
            item.anchor_y,
            [
                item.width as f32 * item.scale,
                item.height as f32 * item.scale,
            ],
            [viewport_w, viewport_h],
        );

        // ScreenImageUniform: ndc_min(vec2) + ndc_max(vec2) + alpha(f32) + pad(3xf32) = 32 bytes
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct ScreenImageUniform {
            ndc_min: [f32; 2],
            ndc_max: [f32; 2],
            alpha: f32,
            _pad: [f32; 3],
        }

        let uniform_data = ScreenImageUniform {
            ndc_min: [ndc_min_x, ndc_min_y],
            ndc_max: [ndc_max_x, ndc_max_y],
            alpha: item.alpha,
            _pad: [0.0; 3],
        };

        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("screen_image_uniform"),
            size: std::mem::size_of::<ScreenImageUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .screen_image
            .bgl
            .as_ref()
            .expect("ensure_screen_image_pipeline not called");

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("screen_image_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(&view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // If the item carries per-pixel depth data, upload a R32Float depth texture
        // and create a second bind group for the depth-composite pipeline.
        let (depth_texture_opt, depth_bind_group_opt) = if let Some(depth_values) = &item.depth {
            let dc_bgl =
                self.screen_image.dc_bgl.as_ref().expect(
                    "ensure_screen_image_dc_pipeline not called before upload_screen_image",
                );

            let dtex = device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("screen_image_depth_tex"),
                size: crate::gpu::Extent3d {
                    width: item.width.max(1),
                    height: item.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: crate::gpu::TextureFormat::R32Float,
                usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                    | crate::gpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Upload depth values as raw bytes (each f32 = 4 bytes).
            let pixel_count = (item.width * item.height) as usize;
            let safe_depth: Vec<f32> = if depth_values.len() >= pixel_count {
                depth_values[..pixel_count].to_vec()
            } else {
                // Pad with far-plane depth (1.0) if caller supplied too few values.
                let mut v = depth_values.clone();
                v.resize(pixel_count, 1.0);
                v
            };

            if item.width > 0 && item.height > 0 {
                queue.write_texture(
                    crate::gpu::TexelCopyTextureInfo {
                        texture: &dtex,
                        mip_level: 0,
                        origin: crate::gpu::Origin3d::ZERO,
                        aspect: crate::gpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&safe_depth),
                    crate::gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(item.width * 4),
                        rows_per_image: Some(item.height),
                    },
                    crate::gpu::Extent3d {
                        width: item.width,
                        height: item.height,
                        depth_or_array_layers: 1,
                    },
                );
            }

            let dview = dtex.create_view(&crate::gpu::TextureViewDescriptor::default());

            let dc_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("screen_image_dc_bg"),
                layout: dc_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureView(&view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(&sampler),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: crate::gpu::BindingResource::TextureView(&dview),
                    },
                ],
            });

            (Some(dtex), Some(dc_bg))
        } else {
            (None, None)
        };

        ScreenImageGpuData {
            _uniform_buf: uniform_buf,
            _texture: texture,
            bind_group,
            _depth_texture: depth_texture_opt,
            depth_bind_group: depth_bind_group_opt,
        }
    }

    // -------------------------------------------------------------------------
    // Volume Surface Slice representation
    // -------------------------------------------------------------------------

    /// Lazily create the screen-rect outline mask pipeline.
    ///
    /// Renders an NDC-space quad into the R8Unorm outline mask. Uses a single
    /// bind group (group 0) with one uniform binding (NdcRectUniform, 16 bytes).
    /// No camera bind group needed. No-op if already created.
    pub(crate) fn ensure_screen_rect_outline_mask_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.screen_image.rect_outline_mask_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = crate::resources::builders::uniform_bgl(
            device,
            "screen_rect_outline_bgl",
            crate::gpu::ShaderStages::VERTEX,
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "screen_rect_outline_mask_shader",
            crate::resources::builders::wgsl_source!("outline_mask_ndc"),
        );

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "screen_rect_outline_mask_pipeline_layout",
            &[&bgl],
        );

        let pipeline = crate::resources::builders::build_outline_mask_pipeline(
            device,
            "screen_rect_outline_mask_pipeline",
            &layout,
            &shader,
            crate::gpu::TextureFormat::R8Unorm,
            &[],
            None,
            false,
            crate::gpu::CompareFunction::Always,
        );

        self.screen_image.rect_outline_bgl = Some(bgl);
        self.screen_image.rect_outline_mask_pipeline = Some(pipeline);
    }
}

/// Per-frame GPU data for one screen-space image overlay, created in `prepare()`.
pub struct ScreenImageGpuData {
    /// Uniform buffer: `ScreenImageUniform` (32 bytes) with NDC extents and alpha.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    /// Uploaded RGBA8 texture for this image (recreated each frame).
    pub(crate) _texture: crate::gpu::Texture,
    /// Bind group (group 0): uniform + colour texture + sampler.
    /// Used by the regular pipeline (no depth test).
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Uploaded R32Float depth texture. `None` when the item has no depth data.
    pub(crate) _depth_texture: Option<crate::gpu::Texture>,
    /// Bind group for the depth-composite pipeline (group 0: uniform + colour + sampler + depth).
    /// `Some` only when the item carries per-pixel depth data.
    pub(crate) depth_bind_group: Option<crate::gpu::BindGroup>,
}
