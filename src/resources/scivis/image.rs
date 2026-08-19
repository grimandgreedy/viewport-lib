use super::*;
use crate::resources::Vertex;

impl DeviceResources {
    // -------------------------------------------------------------------------
    // 2D Image Slice representation
    // -------------------------------------------------------------------------

    /// Lazily create the image slice render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.image_slices` is non-empty.
    pub(crate) fn ensure_image_slice_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.image_slice.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("image_slice_bgl"),
            entries: &[
                // binding 0: ImageSliceUniform
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
                // binding 1: the scalar field texture_3d<f32> (filterable:
                // R16Float, or R32Float with FLOAT32_FILTERABLE), so the slice
                // samples it trilinearly instead of nearest-neighbor.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: crate::gpu::TextureViewDimension::D3,
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 2: vol_sampler (linear)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: lut_tex (colourmap texture_2d)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "image_slice_shader",
            crate::resources::builders::wgsl_source!("image_slice"),
        );

        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "image_slice_pipeline_layout",
            &self.camera_bind_group_layout,
            &bgl,
        );

        self.image_slice.bgl = Some(bgl);
        self.image_slice.pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "image_slice_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[], // no vertex buffer: generates quad from vertex_index
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: crate::gpu::CompareFunction::LessEqual,
                sample_count: self.sample_count,
                ldr_format: self.target_format,
            },
        ));
    }

    /// Upload one [`ImageSliceItem`] to the GPU and return draw data.
    ///
    /// Creates a uniform buffer describing the slice parameters and a bind group
    /// referencing the existing uploaded volume texture.  No vertex buffer is needed:
    /// the shader generates a quad from `vertex_index`.
    pub(crate) fn upload_image_slice(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::ImageSliceItem,
    ) -> Option<crate::resources::ImageSliceGpuData> {
        // Check volume exists before allocating anything.
        if self.content.volume_textures.get(item.volume_id).is_none() {
            return None;
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ImageSliceUniform {
            bbox_min: [f32; 3],
            axis: u32,
            bbox_max: [f32; 3],
            offset: f32,
            scalar_min: f32,
            scalar_max: f32,
            opacity: f32,
            _pad: f32,
        }

        let axis_u32 = match item.axis {
            crate::renderer::SliceAxis::X => 0u32,
            crate::renderer::SliceAxis::Y => 1u32,
            crate::renderer::SliceAxis::Z => 2u32,
        };

        let uniform_data = ImageSliceUniform {
            bbox_min: item.bbox_min,
            axis: axis_u32,
            bbox_max: item.bbox_max,
            offset: item.offset.clamp(0.0, 1.0),
            scalar_min: item.scalar_range.0,
            scalar_max: item.scalar_range.1,
            opacity: item.opacity,
            _pad: 0.0,
        };

        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("image_slice_uniform_buf"),
            size: std::mem::size_of::<ImageSliceUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // Linear sampler so the slice reconstructs the field trilinearly.
        let vol_sampler =
            crate::resources::builders::clamp_linear_sampler(device, "image_slice_vol_sampler");

        // Resolve LUT view index before creating any bind group references.
        let lut_view_idx: Option<usize> = self.content.builtin_colourmap_ids.and_then(|ids| {
            let preset_id = item
                .colour_lut
                .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
            if preset_id.0 < self.content.colourmap_views.len() {
                Some(preset_id.0)
            } else {
                None
            }
        });

        let bgl = self
            .image_slice
            .bgl
            .as_ref()
            .expect("ensure_image_slice_pipeline not called");

        // Borrow vol_view and lut_view after all mutable references are resolved.
        let vol_view = &self
            .content
            .volume_textures
            .get(item.volume_id)
            .expect("volume existence checked above")
            .1;
        let lut_view = lut_view_idx
            .map(|i| &self.content.colourmap_views[i])
            .unwrap_or(&self.content.fallback_lut_view);

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("image_slice_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(vol_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&vol_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::Sampler(&self.material.sampler),
                },
            ],
        });

        Some(crate::resources::ImageSliceGpuData {
            bind_group,
            _uniform_buf: uniform_buf,
            pick_id: item.settings.pick_id,
        })
    }

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
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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
        use crate::ImageAnchor;

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
        let img_w_ndc = 2.0 * item.width as f32 * item.scale / viewport_w.max(1.0);
        let img_h_ndc = 2.0 * item.height as f32 * item.scale / viewport_h.max(1.0);

        let (ndc_min_x, ndc_max_x, ndc_min_y, ndc_max_y) = match item.anchor {
            ImageAnchor::TopLeft => (-1.0, -1.0 + img_w_ndc, 1.0 - img_h_ndc, 1.0),
            ImageAnchor::TopRight => (1.0 - img_w_ndc, 1.0, 1.0 - img_h_ndc, 1.0),
            ImageAnchor::BottomLeft => (-1.0, -1.0 + img_w_ndc, -1.0, -1.0 + img_h_ndc),
            ImageAnchor::BottomRight => (1.0 - img_w_ndc, 1.0, -1.0, -1.0 + img_h_ndc),
            _ => (
                -img_w_ndc * 0.5,
                img_w_ndc * 0.5,
                -img_h_ndc * 0.5,
                img_h_ndc * 0.5,
            ),
        };

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

    /// Upload one [`OverlayImageItem`] to the GPU and return its render data.
    ///
    /// Reuses the `screen_image_pipeline` and its bind group layout; the shaders and
    /// uniform layout are identical. No depth path: `OverlayImageItem` has no depth field.
    ///
    /// Caller must have called [`ensure_screen_image_pipeline`] first.
    pub(crate) fn upload_overlay_image(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::OverlayImageItem,
        viewport_w: f32,
        viewport_h: f32,
    ) -> ScreenImageGpuData {
        use crate::ImageAnchor;

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

        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("overlay_image_tex"),
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
                    target: "viewport_lib::overlay_image",
                    width = item.width,
                    height = item.height,
                    pixels_len = item.pixels.len(),
                    inferred_tex_w = tex_w,
                    inferred_tex_h = tex_h,
                    expected_bytes = needed,
                    actual_bytes = raw.len(),
                    "OverlayImageItem pixel buffer is smaller than the inferred texture size \
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
            crate::resources::builders::clamp_linear_sampler(device, "overlay_image_sampler");

        let img_w_ndc = 2.0 * item.width as f32 * item.scale / viewport_w.max(1.0);
        let img_h_ndc = 2.0 * item.height as f32 * item.scale / viewport_h.max(1.0);

        let (ndc_min_x, ndc_max_x, ndc_min_y, ndc_max_y) = match item.anchor {
            ImageAnchor::TopLeft => (-1.0, -1.0 + img_w_ndc, 1.0 - img_h_ndc, 1.0),
            ImageAnchor::TopRight => (1.0 - img_w_ndc, 1.0, 1.0 - img_h_ndc, 1.0),
            ImageAnchor::BottomLeft => (-1.0, -1.0 + img_w_ndc, -1.0, -1.0 + img_h_ndc),
            ImageAnchor::BottomRight => (1.0 - img_w_ndc, 1.0, -1.0, -1.0 + img_h_ndc),
            _ => (
                -img_w_ndc * 0.5,
                img_w_ndc * 0.5,
                -img_h_ndc * 0.5,
                img_h_ndc * 0.5,
            ),
        };

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
            label: Some("overlay_image_uniform"),
            size: std::mem::size_of::<ScreenImageUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .screen_image
            .bgl
            .as_ref()
            .expect("ensure_screen_image_pipeline not called before upload_overlay_image");

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("overlay_image_bg"),
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

        ScreenImageGpuData {
            _uniform_buf: uniform_buf,
            _texture: texture,
            bind_group,
            _depth_texture: None,
            depth_bind_group: None,
        }
    }

    // -------------------------------------------------------------------------
    // Volume Surface Slice representation
    // -------------------------------------------------------------------------

    /// Lazily create the volume surface slice render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when
    /// `frame.scene.volume_surface_slices` is non-empty.
    pub(crate) fn ensure_volume_surface_slice_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.volume.surface_slice_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_surface_slice_bgl"),
            entries: &[
                // binding 0: VolumeSurfaceSliceUniform
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
                // binding 1: the scalar field texture_3d<f32> (filterable:
                // R16Float, or R32Float with FLOAT32_FILTERABLE), so the slice
                // samples it trilinearly instead of nearest-neighbor.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: crate::gpu::TextureViewDimension::D3,
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 2: vol_sampler (linear)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: lut_tex (colourmap texture_2d)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "volume_surface_slice_shader",
            crate::resources::builders::wgsl_source!("volume_surface_slice"),
        );

        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "volume_surface_slice_layout",
            &self.camera_bind_group_layout,
            &bgl,
        );

        self.volume.surface_slice_bgl = Some(bgl);
        self.volume.surface_slice_pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "volume_surface_slice_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::LessEqual,
                sample_count: self.sample_count,
                ldr_format: self.target_format,
            },
        ));
    }

    /// Upload one [`VolumeSurfaceSliceItem`] and return per-frame GPU data.
    ///
    /// Creates a uniform buffer and a bind group pointing at the existing uploaded
    /// volume texture and colourmap LUT. The mesh vertex/index buffers are referenced
    /// by `MeshId` and looked up from the mesh store at draw time.
    pub(crate) fn upload_volume_surface_slice(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::VolumeSurfaceSliceItem,
    ) -> Option<crate::resources::VolumeSurfaceSliceGpuData> {
        if self.content.volume_textures.get(item.volume_id).is_none() {
            return None;
        }
        // Verify the mesh exists.
        if self.mesh_store.get(item.mesh_id).is_none() {
            return None;
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct VolumeSurfaceSliceUniform {
            model: [[f32; 4]; 4],
            bbox_min: [f32; 3],
            scalar_min: f32,
            bbox_max: [f32; 3],
            scalar_max: f32,
            opacity: f32,
            _pad: [f32; 3],
        }

        let uniform_data = VolumeSurfaceSliceUniform {
            model: item.model,
            bbox_min: item.bbox_min,
            scalar_min: item.scalar_range.0,
            bbox_max: item.bbox_max,
            scalar_max: item.scalar_range.1,
            // ItemSettings.opacity multiplies into the type's own opacity field
            // so consumers can drive transparency through the standard per-item
            // settings without abandoning the existing field.
            opacity: item.opacity * item.settings.opacity,
            _pad: [0.0; 3],
        };

        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_surface_slice_uniform"),
            size: std::mem::size_of::<VolumeSurfaceSliceUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let vol_sampler = crate::resources::builders::clamp_linear_sampler(
            device,
            "volume_surface_slice_vol_sampler",
        );

        let lut_view_idx: Option<usize> = self.content.builtin_colourmap_ids.and_then(|ids| {
            let preset_id = item
                .colour_lut
                .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
            if preset_id.0 < self.content.colourmap_views.len() {
                Some(preset_id.0)
            } else {
                None
            }
        });

        let bgl = self
            .volume
            .surface_slice_bgl
            .as_ref()
            .expect("ensure_volume_surface_slice_pipeline not called");

        let vol_view = &self
            .content
            .volume_textures
            .get(item.volume_id)
            .expect("volume existence checked above")
            .1;
        let lut_view = lut_view_idx
            .map(|i| &self.content.colourmap_views[i])
            .unwrap_or(&self.content.fallback_lut_view);

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("volume_surface_slice_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(vol_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&vol_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::Sampler(&self.material.sampler),
                },
            ],
        });

        Some(crate::resources::VolumeSurfaceSliceGpuData {
            bind_group,
            _uniform_buf: uniform_buf,
            mesh_id: item.mesh_id,
            pick_id: item.settings.pick_id,
        })
    }

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
