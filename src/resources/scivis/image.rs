use super::*;
use crate::resources::Vertex;

impl ViewportGpuResources {
    // -------------------------------------------------------------------------
    // Phase 3.2 : 2D Image Slice representation
    // -------------------------------------------------------------------------

    /// Lazily create the image slice render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.image_slices` is non-empty.
    pub(crate) fn ensure_image_slice_pipeline(&mut self, device: &wgpu::Device) {
        if self.image_slice_pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image_slice_bgl"),
            entries: &[
                // binding 0: ImageSliceUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: texture_3d<f32> — R32Float is not filterable on most hardware,
                // so declare as non-filterable and use a NonFiltering sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // binding 2: vol_sampler (non-filtering nearest — matches R32Float)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 3: lut_tex (colourmap texture_2d)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image_slice_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/image_slice.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image_slice_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &bgl],
            push_constant_ranges: &[],
        });

        let sample_count = self.sample_count;
        let make = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("image_slice_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[], // no vertex buffer: generates quad from vertex_index
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        self.image_slice_bgl = Some(bgl);
        self.image_slice_pipeline = Some(DualPipeline {
            ldr: make(self.target_format),
            hdr: make(wgpu::TextureFormat::Rgba16Float),
        });
    }

    /// Upload one [`ImageSliceItem`] to the GPU and return draw data.
    ///
    /// Creates a uniform buffer describing the slice parameters and a bind group
    /// referencing the existing uploaded volume texture.  No vertex buffer is needed:
    /// the shader generates a quad from `vertex_index`.
    pub(crate) fn upload_image_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::ImageSliceItem,
    ) -> Option<crate::resources::ImageSliceGpuData> {
        // Check volume exists before allocating anything.
        if item.volume_id.0 >= self.volume_textures.len() {
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

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image_slice_uniform_buf"),
            size: std::mem::size_of::<ImageSliceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // Nearest-neighbor sampler for crisp slice sampling.
        let vol_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("image_slice_vol_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Resolve LUT view index before creating any bind group references.
        let lut_view_idx: Option<usize> = self.builtin_colourmap_ids.and_then(|ids| {
            let preset_id = item
                .colour_lut
                .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
            if preset_id.0 < self.colourmap_views.len() {
                Some(preset_id.0)
            } else {
                None
            }
        });

        let bgl = self
            .image_slice_bgl
            .as_ref()
            .expect("ensure_image_slice_pipeline not called");

        // Borrow vol_view and lut_view after all mutable references are resolved.
        let vol_view = &self.volume_textures[item.volume_id.0].1;
        let lut_view = lut_view_idx
            .map(|i| &self.colourmap_views[i])
            .unwrap_or(&self.fallback_lut_view);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image_slice_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&vol_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
            ],
        });

        Some(crate::resources::ImageSliceGpuData {
            bind_group,
            _uniform_buf: uniform_buf,
        })
    }

    // -------------------------------------------------------------------------
    // Phase 10B : screen-space image overlays
    // -------------------------------------------------------------------------

    /// Lazily create the screen-space image render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when
    /// `frame.scene.screen_images` is non-empty.
    pub(crate) fn ensure_screen_image_pipeline(&mut self, device: &wgpu::Device) {
        if self.screen_image_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("screen_image_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/screen_image.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("screen_image_bgl"),
            entries: &[
                // binding 0: ScreenImageUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 2: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("screen_image_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screen_image_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Use Always depth compare (never test) so screen images are always on top.
            // No depth writes. Format must match the depth attachment of the render pass.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        self.screen_image_bgl = Some(bgl);
        self.screen_image_pipeline = Some(pipeline);
    }

    /// Lazily create the depth-composite screen-image render pipeline (Phase 12).
    ///
    /// No-op if already created. Called from `prepare()` when any submitted
    /// `ScreenImageItem` carries per-pixel depth data.
    pub(crate) fn ensure_screen_image_dc_pipeline(&mut self, device: &wgpu::Device) {
        if self.screen_image_dc_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("screen_image_dc_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/screen_image_dc.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("screen_image_dc_bgl"),
            entries: &[
                // binding 0: ScreenImageUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: colour texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 2: sampler (filtering, for colour texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: R32Float depth texture (non-filterable, read via textureLoad)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("screen_image_dc_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screen_image_dc_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Depth test: discard fragments whose image depth exceeds scene depth.
            // depth_write_enabled: false so the scene depth buffer is not modified.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        self.screen_image_dc_bgl = Some(bgl);
        self.screen_image_dc_pipeline = Some(pipeline);
    }

    /// Upload one [`ScreenImageItem`] to the GPU and return its per-frame GPU data.
    ///
    /// Creates a new RGBA8Unorm texture each call : intended for per-frame data.
    /// The returned [`ScreenImageGpuData`] is valid only for one frame.
    pub(crate) fn upload_screen_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::ScreenImageItem,
        viewport_w: f32,
        viewport_h: f32,
    ) -> ScreenImageGpuData {
        use crate::ImageAnchor;

        // Create texture from pixel data.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screen_image_tex"),
            size: wgpu::Extent3d {
                width: item.width.max(1),
                height: item.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        if !item.pixels.is_empty() && item.width > 0 && item.height > 0 {
            let raw: Vec<u8> = item.pixels.iter().flat_map(|p| p.iter().copied()).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &raw,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(item.width * 4),
                    rows_per_image: Some(item.height),
                },
                wgpu::Extent3d {
                    width: item.width,
                    height: item.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("screen_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

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

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screen_image_uniform"),
            size: std::mem::size_of::<ScreenImageUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .screen_image_bgl
            .as_ref()
            .expect("ensure_screen_image_pipeline not called");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("screen_image_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Phase 12: if the item carries per-pixel depth data, upload a R32Float depth texture
        // and create a second bind group for the depth-composite pipeline.
        let (depth_texture_opt, depth_bind_group_opt) = if let Some(depth_values) = &item.depth {
            let dc_bgl = self
                .screen_image_dc_bgl
                .as_ref()
                .expect("ensure_screen_image_dc_pipeline not called before upload_screen_image");

            let dtex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("screen_image_depth_tex"),
                size: wgpu::Extent3d {
                    width: item.width.max(1),
                    height: item.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
                    wgpu::TexelCopyTextureInfo {
                        texture: &dtex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&safe_depth),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(item.width * 4),
                        rows_per_image: Some(item.height),
                    },
                    wgpu::Extent3d {
                        width: item.width,
                        height: item.height,
                        depth_or_array_layers: 1,
                    },
                );
            }

            let dview = dtex.create_view(&wgpu::TextureViewDescriptor::default());

            let dc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("screen_image_dc_bg"),
                layout: dc_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&dview),
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

    /// Upload one [`OverlayImageItem`] to the GPU and return its render data (Phase 7).
    ///
    /// Reuses the `screen_image_pipeline` and its bind group layout; the shaders and
    /// uniform layout are identical. No depth path: `OverlayImageItem` has no depth field.
    ///
    /// Caller must have called [`ensure_screen_image_pipeline`] first.
    pub(crate) fn upload_overlay_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::OverlayImageItem,
        viewport_w: f32,
        viewport_h: f32,
    ) -> ScreenImageGpuData {
        use crate::ImageAnchor;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay_image_tex"),
            size: wgpu::Extent3d {
                width: item.width.max(1),
                height: item.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        if !item.pixels.is_empty() && item.width > 0 && item.height > 0 {
            let raw: Vec<u8> = item.pixels.iter().flat_map(|p| p.iter().copied()).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &raw,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(item.width * 4),
                    rows_per_image: Some(item.height),
                },
                wgpu::Extent3d {
                    width: item.width,
                    height: item.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

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

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_image_uniform"),
            size: std::mem::size_of::<ScreenImageUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .screen_image_bgl
            .as_ref()
            .expect("ensure_screen_image_pipeline not called before upload_overlay_image");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_image_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
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
    // Phase 10 : Volume Surface Slice representation
    // -------------------------------------------------------------------------

    /// Lazily create the volume surface slice render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when
    /// `frame.scene.volume_surface_slices` is non-empty.
    pub(crate) fn ensure_volume_surface_slice_pipeline(&mut self, device: &wgpu::Device) {
        if self.volume_surface_slice_pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume_surface_slice_bgl"),
            entries: &[
                // binding 0: VolumeSurfaceSliceUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: texture_3d<f32> (R32Float, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // binding 2: vol_sampler (non-filtering nearest)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 3: lut_tex (colourmap texture_2d)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volume_surface_slice_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/volume_surface_slice.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volume_surface_slice_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &bgl],
            push_constant_ranges: &[],
        });

        let sample_count = self.sample_count;
        let make = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("volume_surface_slice_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        self.volume_surface_slice_bgl = Some(bgl);
        self.volume_surface_slice_pipeline = Some(DualPipeline {
            ldr: make(self.target_format),
            hdr: make(wgpu::TextureFormat::Rgba16Float),
        });
    }

    /// Upload one [`VolumeSurfaceSliceItem`] and return per-frame GPU data.
    ///
    /// Creates a uniform buffer and a bind group pointing at the existing uploaded
    /// volume texture and colourmap LUT. The mesh vertex/index buffers are referenced
    /// by `MeshId` and looked up from the mesh store at draw time.
    pub(crate) fn upload_volume_surface_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::VolumeSurfaceSliceItem,
    ) -> Option<crate::resources::VolumeSurfaceSliceGpuData> {
        if item.volume_id.0 >= self.volume_textures.len() {
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
            opacity: item.opacity,
            _pad: [0.0; 3],
        };

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_surface_slice_uniform"),
            size: std::mem::size_of::<VolumeSurfaceSliceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let vol_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume_surface_slice_vol_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let lut_view_idx: Option<usize> = self.builtin_colourmap_ids.and_then(|ids| {
            let preset_id = item
                .colour_lut
                .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
            if preset_id.0 < self.colourmap_views.len() {
                Some(preset_id.0)
            } else {
                None
            }
        });

        let bgl = self
            .volume_surface_slice_bgl
            .as_ref()
            .expect("ensure_volume_surface_slice_pipeline not called");

        let vol_view = &self.volume_textures[item.volume_id.0].1;
        let lut_view = lut_view_idx
            .map(|i| &self.colourmap_views[i])
            .unwrap_or(&self.fallback_lut_view);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume_surface_slice_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&vol_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
            ],
        });

        Some(crate::resources::VolumeSurfaceSliceGpuData {
            bind_group,
            _uniform_buf: uniform_buf,
            mesh_id: item.mesh_id,
        })
    }

    /// Lazily create the screen-rect outline mask pipeline.
    ///
    /// Renders an NDC-space quad into the R8Unorm outline mask. Uses a single
    /// bind group (group 0) with one uniform binding (NdcRectUniform, 16 bytes).
    /// No camera bind group needed. No-op if already created.
    pub(crate) fn ensure_screen_rect_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.screen_rect_outline_mask_pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("screen_rect_outline_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("screen_rect_outline_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/outline_mask_ndc.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("screen_rect_outline_mask_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screen_rect_outline_mask_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        self.screen_rect_outline_bgl = Some(bgl);
        self.screen_rect_outline_mask_pipeline = Some(pipeline);
    }
}
