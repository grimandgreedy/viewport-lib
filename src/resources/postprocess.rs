use super::*;

impl ViewportGpuResources {
    /// Create or recreate the offscreen outline color + depth/stencil textures and
    /// the fullscreen composite pipeline used to blit the outline onto the main pass.
    /// No-op if the size hasn't changed and resources already exist.
    pub(crate) fn ensure_outline_target(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);

        if self.outline_target_size == [w, h] && self.outline_color_texture.is_some() {
            return;
        }
        self.outline_target_size = [w, h];

        // Offscreen RGBA color texture (transparent clear).
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("outline_color_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth+stencil texture for the stencil outline passes.
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("outline_depth_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Sampler (linear, clamp-to-edge).
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("outline_composite_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bind group layout: texture + sampler.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("outline_composite_bgl"),
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

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outline_composite_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Fullscreen composite pipeline (alpha blending).
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outline_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/outline_composite.wgsl").into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outline_composite_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline_single = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("outline_composite_pipeline_single"),
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
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let pipeline_msaa = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("outline_composite_pipeline_msaa"),
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
                count: self.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        self.outline_color_texture = Some(color_tex);
        self.outline_color_view = Some(color_view);
        self.outline_depth_texture = Some(depth_tex);
        self.outline_depth_view = Some(depth_view);
        // HDR-format variant for compositing onto the Rgba16Float HDR texture.
        // Created here in case outline resources are initialized after the HDR
        // target already exists.
        if self.hdr_texture.is_some() && self.outline_composite_pipeline_hdr.is_none() {
            let hdr_pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("outline_composite_pipeline_hdr"),
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
                            format: wgpu::TextureFormat::Rgba16Float,
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
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });
            self.outline_composite_pipeline_hdr = Some(hdr_pipeline);
        }

        self.outline_composite_pipeline_single = Some(pipeline_single);
        self.outline_composite_pipeline_msaa = Some(pipeline_msaa);
        self.outline_composite_bgl = Some(bgl);
        self.outline_composite_bind_group = Some(bg);
        self.outline_composite_sampler = Some(sampler);
    }

    /// Create or recreate HDR textures, post-process pipelines, and HDR scene pipelines.
    /// No-op if the size hasn't changed and resources already exist.
    pub(crate) fn ensure_hdr_target(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
        w: u32,
        h: u32,
    ) {
        let w = w.max(1);
        let h = h.max(1);

        // Early return if size matches and resources exist.
        if self.hdr_size == [w, h] && self.hdr_texture.is_some() && self.tone_map_pipeline.is_some()
        {
            return;
        }

        self.hdr_size = [w, h];

        // ------------------------------------------------------------------
        // HDR color texture (Rgba16Float)
        // ------------------------------------------------------------------
        let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // HDR depth+stencil texture (single-sample)
        // ------------------------------------------------------------------
        let hdr_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_depth_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hdr_depth_view = hdr_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let hdr_depth_only_view = hdr_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        // ------------------------------------------------------------------
        // Bloom textures
        // ------------------------------------------------------------------
        let bloom_threshold_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom_threshold_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_threshold_view =
            bloom_threshold_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let hw = (w / 2).max(1);
        let hh = (h / 2).max(1);

        let bloom_ping_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom_ping_texture"),
            size: wgpu::Extent3d {
                width: hw,
                height: hh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_ping_view = bloom_ping_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let bloom_pong_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom_pong_texture"),
            size: wgpu::Extent3d {
                width: hw,
                height: hh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_pong_view = bloom_pong_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // SSAO textures
        // ------------------------------------------------------------------
        let ssao_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_view = ssao_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let ssao_blur_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao_blur_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_blur_view = ssao_blur_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // Contact shadow texture (R8Unorm, full-res)
        // ------------------------------------------------------------------
        let cs_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("contact_shadow_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cs_view = cs_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // Shared samplers (only create once)
        // ------------------------------------------------------------------
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pp_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pp_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ------------------------------------------------------------------
        // Fallback normal map / AO map pixel upload (only on first call)
        // ------------------------------------------------------------------
        if !self.fallback_textures_uploaded {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.fallback_normal_map,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[128u8, 128u8, 255u8, 255u8],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.fallback_ao_map,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255u8, 255u8, 255u8, 255u8],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            // Also upload the fallback albedo texture (1×1 white sRGB).
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.fallback_texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255u8, 255u8, 255u8, 255u8],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.fallback_textures_uploaded = true;
        }

        // ------------------------------------------------------------------
        // Placeholder textures (only on first call)
        // ------------------------------------------------------------------
        if self.bloom_placeholder_view.is_none() {
            let bloom_placeholder_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("bloom_placeholder"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &bloom_placeholder_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8; 8],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.bloom_placeholder_view =
                Some(bloom_placeholder_tex.create_view(&wgpu::TextureViewDescriptor::default()));

            let ao_placeholder_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ao_placeholder"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &ao_placeholder_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255u8],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.ao_placeholder_view =
                Some(ao_placeholder_tex.create_view(&wgpu::TextureViewDescriptor::default()));

            // CS placeholder: 1×1 R8Unorm, fully lit (255 = 1.0).
            let cs_placeholder_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("cs_placeholder"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &cs_placeholder_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255u8],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.cs_placeholder_view =
                Some(cs_placeholder_tex.create_view(&wgpu::TextureViewDescriptor::default()));
        }

        // ------------------------------------------------------------------
        // SSAO noise texture (4×4, Rgba8Unorm, deterministic random directions)
        // ------------------------------------------------------------------
        if self.ssao_noise_view.is_none() {
            let noise_data: Vec<u8> = (0..16)
                .flat_map(|i| {
                    let angle = (i as f32 / 16.0) * std::f32::consts::TAU;
                    let x = ((angle.cos() * 0.5 + 0.5) * 255.0) as u8;
                    let y = ((angle.sin() * 0.5 + 0.5) * 255.0) as u8;
                    [x, y, 128u8, 255u8]
                })
                .collect();

            let noise_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ssao_noise"),
                size: wgpu::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &noise_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &noise_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * 4),
                    rows_per_image: Some(4),
                },
                wgpu::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
            );
            self.ssao_noise_view =
                Some(noise_tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.ssao_noise_texture = Some(noise_tex);
        }

        // ------------------------------------------------------------------
        // SSAO hemisphere kernel buffer (64 × vec4<f32>)
        // ------------------------------------------------------------------
        if self.ssao_kernel_buf.is_none() {
            let kernel_data: Vec<[f32; 4]> = (0..64)
                .map(|i| {
                    let t = i as f32 / 64.0;
                    let phi = t * std::f32::consts::TAU * 2.4;
                    let theta = (t * 1.0_f32).acos().min(std::f32::consts::FRAC_PI_2 * 0.99);
                    let scale = (i as f32 / 64.0).powi(2) * 0.9 + 0.1;
                    let x = theta.sin() * phi.cos() * scale;
                    let y = theta.sin() * phi.sin() * scale;
                    let z = theta.cos() * scale;
                    [x, y, z.abs(), 0.0]
                })
                .collect();

            let kernel_bytes: &[u8] = bytemuck::cast_slice(&kernel_data);
            let kernel_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ssao_kernel_buf"),
                size: kernel_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&kernel_buf, 0, kernel_bytes);
            self.ssao_kernel_buf = Some(kernel_buf);
        }

        // ------------------------------------------------------------------
        // Uniform buffers
        // ------------------------------------------------------------------
        let tone_map_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tone_map_uniform_buf"),
            size: std::mem::size_of::<ToneMapUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bloom_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom_uniform_buf"),
            size: std::mem::size_of::<BloomUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Constant H-blur and V-blur uniform buffers (horizontal flag baked in).
        let bloom_h_uniform_buf = {
            let data = BloomUniform {
                threshold: 0.0,
                intensity: 0.0,
                horizontal: 1,
                _pad: 0,
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom_h_uniform_buf"),
                size: std::mem::size_of::<BloomUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[data]));
            buf
        };

        let bloom_v_uniform_buf = {
            let data = BloomUniform {
                threshold: 0.0,
                intensity: 0.0,
                horizontal: 0,
                _pad: 0,
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom_v_uniform_buf"),
                size: std::mem::size_of::<BloomUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[data]));
            buf
        };

        let ssao_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao_uniform_buf"),
            size: std::mem::size_of::<SsaoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cs_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_shadow_uniform_buf"),
            size: std::mem::size_of::<ContactShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ------------------------------------------------------------------
        // Bind group layouts
        // ------------------------------------------------------------------

        // Tone map BGL: hdr_tex, sampler, uniform, bloom_tex, ao_tex.
        let tone_map_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tone_map_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 5: contact shadow texture (R8Unorm, 1.0=lit 0.0=shadowed)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Bloom/blur shared BGL: input_tex, sampler, uniform.
        let bloom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // SSAO BGL: depth, sampler(non-filter), noise, noise_sampler, kernel, uniform.
        let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // SSAO blur BGL: ssao_tex, sampler.
        let ssao_blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_blur_bgl"),
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

        // Contact shadow BGL: depth_tex, depth_sampler(non-filter), uniform.
        let cs_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contact_shadow_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ------------------------------------------------------------------
        // Fullscreen pipeline helper closure
        // ------------------------------------------------------------------
        let make_fullscreen_pipeline =
            |label: &str,
             shader: wgpu::ShaderModule,
             entry_vs: &str,
             entry_fs: &str,
             bgl: &wgpu::BindGroupLayout,
             target_format: wgpu::TextureFormat| {
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{label}_layout")),
                    bind_group_layouts: &[bgl],
                    push_constant_ranges: &[],
                });
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(entry_vs),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(entry_fs),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: target_format,
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
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            };

        // ------------------------------------------------------------------
        // Post-process pipelines
        // ------------------------------------------------------------------
        let tone_map_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tone_map_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tone_map.wgsl").into()),
        });
        let tone_map_pipeline = make_fullscreen_pipeline(
            "tone_map_pipeline",
            tone_map_shader,
            "vs_main",
            "fs_main",
            &tone_map_bgl,
            output_format,
        );

        let bloom_threshold_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_threshold_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/bloom_threshold.wgsl").into(),
            ),
        });
        let bloom_threshold_pipeline = make_fullscreen_pipeline(
            "bloom_threshold_pipeline",
            bloom_threshold_shader,
            "vs_main",
            "fs_main",
            &bloom_bgl,
            wgpu::TextureFormat::Rgba16Float,
        );

        let bloom_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bloom_blur.wgsl").into()),
        });
        let bloom_blur_pipeline = make_fullscreen_pipeline(
            "bloom_blur_pipeline",
            bloom_blur_shader,
            "vs_main",
            "fs_main",
            &bloom_bgl,
            wgpu::TextureFormat::Rgba16Float,
        );

        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao.wgsl").into()),
        });
        let ssao_pipeline = make_fullscreen_pipeline(
            "ssao_pipeline",
            ssao_shader,
            "vs_main",
            "fs_main",
            &ssao_bgl,
            wgpu::TextureFormat::R8Unorm,
        );

        let ssao_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao_blur.wgsl").into()),
        });
        let ssao_blur_pipeline = make_fullscreen_pipeline(
            "ssao_blur_pipeline",
            ssao_blur_shader,
            "vs_main",
            "fs_main",
            &ssao_blur_bgl,
            wgpu::TextureFormat::R8Unorm,
        );

        let cs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/contact_shadow.wgsl").into()),
        });
        let cs_pipeline = make_fullscreen_pipeline(
            "contact_shadow_pipeline",
            cs_shader,
            "vs_main",
            "fs_main",
            &cs_bgl,
            wgpu::TextureFormat::R8Unorm,
        );

        // ------------------------------------------------------------------
        // Bind groups
        // ------------------------------------------------------------------
        let bloom_placeholder_view = self.bloom_placeholder_view.as_ref().unwrap();
        let ao_placeholder_view = self.ao_placeholder_view.as_ref().unwrap();
        let cs_placeholder_view = self.cs_placeholder_view.as_ref().unwrap();

        // Contact shadow bind group — reads depth + uniform.
        let cs_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact_shadow_bg"),
            layout: &cs_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cs_uniform_buf.as_entire_binding(),
                },
            ],
        });

        // Tone map bind group uses placeholder textures initially (no bloom/SSAO/CS yet).
        let tone_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_map_bg"),
            layout: &tone_map_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tone_map_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(bloom_placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao_placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(cs_placeholder_view),
                },
            ],
        });

        let bloom_threshold_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_threshold_bg"),
            layout: &bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_uniform_buf.as_entire_binding(),
                },
            ],
        });

        let bloom_blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_blur_h_bg"),
            layout: &bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_threshold_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_h_uniform_buf.as_entire_binding(),
                },
            ],
        });

        let bloom_blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_blur_v_bg"),
            layout: &bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_ping_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_v_uniform_buf.as_entire_binding(),
                },
            ],
        });

        // H-blur bind group reading from pong (for iteration passes 2+).
        let bloom_blur_h_pong_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_blur_h_pong_bg"),
            layout: &bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_pong_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_h_uniform_buf.as_entire_binding(),
                },
            ],
        });

        let ssao_noise_view = self.ssao_noise_view.as_ref().unwrap();
        let ssao_kernel_buf = self.ssao_kernel_buf.as_ref().unwrap();

        let ssao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_bg"),
            layout: &ssao_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ssao_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ssao_kernel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ssao_uniform_buf.as_entire_binding(),
                },
            ],
        });

        let ssao_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_blur_bg"),
            layout: &ssao_blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });

        // ------------------------------------------------------------------
        // HDR scene pipelines (same shaders as LDR but Rgba16Float target, no MSAA)
        // ------------------------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader_hdr"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh.wgsl").into()),
        });

        let hdr_depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let hdr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdr_mesh_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &self.object_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let hdr_solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hdr_solid_pipeline"),
            layout: Some(&hdr_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(hdr_depth_stencil.clone()),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let _hdr_solid_two_sided_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_solid_two_sided_pipeline"),
                layout: Some(&hdr_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None, // No culling: visible from both sides.
                    ..Default::default()
                },
                depth_stencil: Some(hdr_depth_stencil.clone()),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

        let hdr_transparent_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_transparent_pipeline"),
                layout: Some(&hdr_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

        let hdr_wireframe_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_wireframe_pipeline"),
                layout: Some(&hdr_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(hdr_depth_stencil),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

        // HDR overlay pipeline for cap fill (overlay.wgsl targeting Rgba16Float).
        let overlay_shader_hdr = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay_shader_hdr"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/overlay.wgsl").into()),
        });
        let hdr_overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdr_overlay_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &self.overlay_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let hdr_overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hdr_overlay_pipeline"),
            layout: Some(&hdr_overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader_hdr,
                entry_point: Some("vs_main"),
                buffers: &[OverlayVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader_hdr,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        // HDR instanced pipelines (only if instance BGL exists).
        let (hdr_solid_instanced, hdr_transparent_instanced) = if let Some(ref instance_bgl) =
            self.instance_bind_group_layout
        {
            let instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh_instanced_shader_hdr"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/mesh_instanced.wgsl").into(),
                ),
            });
            let instanced_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hdr_instanced_pipeline_layout"),
                bind_group_layouts: &[&self.camera_bind_group_layout, instance_bgl],
                push_constant_ranges: &[],
            });
            let solid = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_solid_instanced_pipeline"),
                layout: Some(&instanced_layout),
                vertex: wgpu::VertexState {
                    module: &instanced_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &instanced_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });
            let trans = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_transparent_instanced_pipeline"),
                layout: Some(&instanced_layout),
                vertex: wgpu::VertexState {
                    module: &instanced_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &instanced_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });
            (Some(solid), Some(trans))
        } else {
            (None, None)
        };

        // ------------------------------------------------------------------
        // Store everything
        // ------------------------------------------------------------------
        self.hdr_texture = Some(hdr_tex);
        self.hdr_view = Some(hdr_view);
        self.hdr_depth_texture = Some(hdr_depth_tex);
        self.hdr_depth_view = Some(hdr_depth_view);
        self.hdr_depth_only_view = Some(hdr_depth_only_view);

        self.bloom_threshold_texture = Some(bloom_threshold_tex);
        self.bloom_threshold_view = Some(bloom_threshold_view);
        self.bloom_ping_texture = Some(bloom_ping_tex);
        self.bloom_ping_view = Some(bloom_ping_view);
        self.bloom_pong_texture = Some(bloom_pong_tex);
        self.bloom_pong_view = Some(bloom_pong_view);

        self.ssao_texture = Some(ssao_tex);
        self.ssao_view = Some(ssao_view);
        self.ssao_blur_texture = Some(ssao_blur_tex);
        self.ssao_blur_view = Some(ssao_blur_view);

        self.pp_linear_sampler = Some(linear_sampler);
        self.pp_nearest_sampler = Some(nearest_sampler);

        self.tone_map_bgl = Some(tone_map_bgl);
        self.tone_map_pipeline = Some(tone_map_pipeline);
        self.tone_map_bind_group = Some(tone_map_bind_group);
        self.tone_map_uniform_buf = Some(tone_map_uniform_buf);

        self.bloom_threshold_pipeline = Some(bloom_threshold_pipeline);
        self.bloom_blur_pipeline = Some(bloom_blur_pipeline);
        self.bloom_threshold_bg = Some(bloom_threshold_bg);
        self.bloom_blur_h_bg = Some(bloom_blur_h_bg);
        self.bloom_blur_v_bg = Some(bloom_blur_v_bg);
        self.bloom_blur_h_pong_bg = Some(bloom_blur_h_pong_bg);
        self.bloom_uniform_buf = Some(bloom_uniform_buf);
        self.bloom_h_uniform_buf = Some(bloom_h_uniform_buf);
        self.bloom_v_uniform_buf = Some(bloom_v_uniform_buf);

        self.ssao_pipeline = Some(ssao_pipeline);
        self.ssao_blur_pipeline = Some(ssao_blur_pipeline);
        self.ssao_bg = Some(ssao_bg);
        self.ssao_blur_bg = Some(ssao_blur_bg);
        self.ssao_uniform_buf = Some(ssao_uniform_buf);

        self.contact_shadow_texture = Some(cs_tex);
        self.contact_shadow_view = Some(cs_view);
        self.contact_shadow_pipeline = Some(cs_pipeline);
        self.contact_shadow_bgl = Some(cs_bgl);
        self.contact_shadow_bg = Some(cs_bg);
        self.contact_shadow_uniform_buf = Some(cs_uniform_buf);

        self.hdr_solid_pipeline = Some(hdr_solid_pipeline);
        self.hdr_transparent_pipeline = Some(hdr_transparent_pipeline);
        self.hdr_wireframe_pipeline = Some(hdr_wireframe_pipeline);
        self.hdr_solid_instanced_pipeline = hdr_solid_instanced;
        self.hdr_transparent_instanced_pipeline = hdr_transparent_instanced;
        self.hdr_overlay_pipeline = Some(hdr_overlay_pipeline);

        // ------------------------------------------------------------------
        // HDR-format outline composite pipeline (Rgba16Float target).
        // The LDR pipelines are created in init_outline_resources() for the
        // surface format; this variant is for compositing outlines onto the
        // HDR texture.
        // ------------------------------------------------------------------
        if let Some(ref bgl) = self.outline_composite_bgl {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("outline_composite_shader_hdr"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/outline_composite.wgsl").into(),
                ),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("outline_composite_layout_hdr"),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            self.outline_composite_pipeline_hdr =
                Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("outline_composite_pipeline_hdr"),
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
                            format: wgpu::TextureFormat::Rgba16Float,
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
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                }));
        }

        // ------------------------------------------------------------------
        // FXAA intermediate texture and pipeline.
        // The tone-map pass writes to fxaa_texture instead of the output when
        // FXAA is enabled. The FXAA pass then reads from fxaa_texture and
        // writes the final anti-aliased result to the surface texture.
        // ------------------------------------------------------------------
        let fxaa_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fxaa_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: output_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fxaa_view = fxaa_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let fxaa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fxaa_bgl"),
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

        let fxaa_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fxaa_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let fxaa_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa_bg"),
            layout: &fxaa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fxaa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&fxaa_sampler),
                },
            ],
        });

        let fxaa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fxaa_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fxaa.wgsl").into()),
        });

        let fxaa_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fxaa_pipeline_layout"),
            bind_group_layouts: &[&fxaa_bgl],
            push_constant_ranges: &[],
        });

        let fxaa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fxaa_pipeline"),
            layout: Some(&fxaa_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fxaa_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fxaa_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.fxaa_texture = Some(fxaa_tex);
        self.fxaa_view = Some(fxaa_view);
        self.fxaa_bgl = Some(fxaa_bgl);
        self.fxaa_bind_group = Some(fxaa_bind_group);
        self.fxaa_pipeline = Some(fxaa_pipeline);
    }

    /// Rebuild the tone-map bind group with device, swapping in the active bloom/AO textures.
    pub(crate) fn rebuild_tone_map_bind_group_with_device(
        &mut self,
        device: &wgpu::Device,
        use_bloom: bool,
        use_ssao: bool,
        use_contact_shadows: bool,
    ) {
        let bgl = match &self.tone_map_bgl {
            Some(b) => b,
            None => return,
        };
        let hdr_view = match &self.hdr_view {
            Some(v) => v,
            None => return,
        };
        let sampler = match &self.pp_linear_sampler {
            Some(s) => s,
            None => return,
        };
        let uniform_buf = match &self.tone_map_uniform_buf {
            Some(b) => b,
            None => return,
        };

        let bloom_view: &wgpu::TextureView = if use_bloom && self.bloom_pong_view.is_some() {
            self.bloom_pong_view.as_ref().unwrap()
        } else {
            match &self.bloom_placeholder_view {
                Some(v) => v,
                None => return,
            }
        };

        let ao_view: &wgpu::TextureView = if use_ssao && self.ssao_blur_view.is_some() {
            self.ssao_blur_view.as_ref().unwrap()
        } else {
            match &self.ao_placeholder_view {
                Some(v) => v,
                None => return,
            }
        };

        let cs_view: &wgpu::TextureView =
            if use_contact_shadows && self.contact_shadow_view.is_some() {
                self.contact_shadow_view.as_ref().unwrap()
            } else {
                match &self.cs_placeholder_view {
                    Some(v) => v,
                    None => return,
                }
            };

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_map_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(cs_view),
                },
            ],
        });
        self.tone_map_bind_group = Some(bg);
    }
}
