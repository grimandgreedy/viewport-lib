use super::*;

impl ViewportGpuResources {
    /// Create or recreate the offscreen outline colour + depth/stencil textures and
    /// the fullscreen composite pipeline used to blit the outline onto the main pass.
    /// No-op if the size hasn't changed and resources already exist.
    #[allow(dead_code)]
    pub(crate) fn ensure_outline_target(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);

        if self.outline_target_size == [w, h] && self.outline_colour_texture.is_some() {
            return;
        }
        self.outline_target_size = [w, h];

        // Offscreen RGBA colour texture (transparent clear).
        let colour_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("outline_colour_texture"),
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
        let colour_view = colour_tex.create_view(&wgpu::TextureViewDescriptor::default());

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
                    resource: wgpu::BindingResource::TextureView(&colour_view),
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

        self.outline_colour_texture = Some(colour_tex);
        self.outline_colour_view = Some(colour_view);
        self.outline_depth_texture = Some(depth_tex);
        self.outline_depth_view = Some(depth_view);
        // HDR-format variant for compositing onto the Rgba16Float HDR texture.
        // Created here in case outline resources are initialized after the HDR
        // target already exists.
        if self.hdr_texture.is_some() && self.outline_composite_pipeline_hdr.is_none() {
            let hdr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

    /// Rebuild the tone-map bind group with device, swapping in the active bloom/AO textures.
    #[allow(dead_code)]
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(
                        self.hdr_depth_only_view.as_ref().expect("hdr depth view"),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(
                        self.lic_placeholder_view
                            .as_ref()
                            .or(self.cs_placeholder_view.as_ref())
                            .expect("ensure_hdr_shared not called"),
                    ),
                },
            ],
        });
        self.tone_map_bind_group = Some(bg);
    }

    // -----------------------------------------------------------------------
    // Phase 3: Per-viewport HDR state : shared infrastructure
    // -----------------------------------------------------------------------

    /// Create all shared HDR/post-process infrastructure (BGLs, pipelines,
    /// samplers, placeholder textures, SSAO noise/kernel) on `self`.
    /// No-op after the first call. Must be called before `create_hdr_viewport_state`.
    pub(crate) fn ensure_hdr_shared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
    ) {
        // Guard: if all three sentinel fields exist, everything is created.
        if self.tone_map_pipeline.is_some()
            && self.bloom_bgl.is_some()
            && self.fxaa_sampler.is_some()
        {
            return;
        }

        // --- Fallback textures (one-time uploads) ---
        if !self.fallback_textures_uploaded {
            let upload = |tex: &wgpu::Texture, data: &[u8]| {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
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
            };
            upload(&self.fallback_normal_map, &[128u8, 128u8, 255u8, 255u8]);
            upload(&self.fallback_ao_map, &[255u8, 255u8, 255u8, 255u8]);
            upload(
                &self.fallback_texture.texture,
                &[255u8, 255u8, 255u8, 255u8],
            );
            self.fallback_textures_uploaded = true;
        }

        // --- Placeholder textures (one-time) ---
        if self.bloom_placeholder_view.is_none() {
            let make_placeholder = |device: &wgpu::Device,
                                    queue: &wgpu::Queue,
                                    label: &str,
                                    format: wgpu::TextureFormat,
                                    data: &[u8],
                                    bytes_per_row: u32|
             -> (wgpu::Texture, wgpu::TextureView) {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(1),
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                (tex, view)
            };

            let (_bt, bv) = make_placeholder(
                device,
                queue,
                "bloom_placeholder",
                wgpu::TextureFormat::Rgba16Float,
                &[0u8; 8],
                8,
            );
            self.bloom_placeholder_view = Some(bv);

            let (_at, av) = make_placeholder(
                device,
                queue,
                "ao_placeholder",
                wgpu::TextureFormat::R8Unorm,
                &[255u8],
                1,
            );
            self.ao_placeholder_view = Some(av);

            let (_ct, cv) = make_placeholder(
                device,
                queue,
                "cs_placeholder",
                wgpu::TextureFormat::R8Unorm,
                &[255u8],
                1,
            );
            self.cs_placeholder_view = Some(cv);

            // LIC placeholder: 1x1 R8Unorm, 128 = 0.5 -> lic_factor = 1.0 (no modulation).
            let (_lt, lv) = make_placeholder(
                device,
                queue,
                "lic_placeholder",
                wgpu::TextureFormat::R8Unorm,
                &[128u8],
                1,
            );
            self.lic_placeholder_view = Some(lv);
        }

        // --- SSAO noise (one-time) ---
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

        // --- SSAO kernel (one-time) ---
        if self.ssao_kernel_buf.is_none() {
            let kernel_data: Vec<[f32; 4]> = (0..64)
                .map(|i| {
                    let t = i as f32 / 64.0;
                    let phi = t * std::f32::consts::TAU * 2.4;
                    let theta = (t * 1.0_f32).acos().min(std::f32::consts::FRAC_PI_2 * 0.99);
                    let scale = (i as f32 / 64.0).powi(2) * 0.9 + 0.1;
                    [
                        theta.sin() * phi.cos() * scale,
                        theta.sin() * phi.sin() * scale,
                        theta.cos().abs() * scale,
                        0.0,
                    ]
                })
                .collect();
            let kernel_bytes: &[u8] = bytemuck::cast_slice(&kernel_data);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ssao_kernel_buf"),
                size: kernel_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, kernel_bytes);
            self.ssao_kernel_buf = Some(buf);
        }

        // --- Shared samplers ---
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
        let fxaa_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fxaa_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let oit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("oit_composite_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let outline_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("outline_composite_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Bind group layouts ---
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 7: LIC intensity texture (R8Unorm). Placeholder when LIC is disabled.
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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

        let oit_composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oit_composite_bgl"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let outline_composite_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // --- Fullscreen pipeline helper ---
        let make_fs_pipeline = |label: &str,
                                shader: wgpu::ShaderModule,
                                vs: &str,
                                fs: &str,
                                bgl: &wgpu::BindGroupLayout,
                                fmt: wgpu::TextureFormat,
                                depth: Option<wgpu::DepthStencilState>|
         -> wgpu::RenderPipeline {
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
                    entry_point: Some(vs),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
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
                depth_stencil: depth,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        // Tone map pipeline
        let tone_map_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tone_map_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tone_map.wgsl").into()),
        });
        let tone_map_pipeline = make_fs_pipeline(
            "tone_map_pipeline",
            tone_map_shader,
            "vs_main",
            "fs_main",
            &tone_map_bgl,
            output_format,
            None,
        );

        // Bloom pipelines
        let bloom_threshold_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_threshold_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/bloom_threshold.wgsl").into(),
            ),
        });
        let bloom_threshold_pipeline = make_fs_pipeline(
            "bloom_threshold_pipeline",
            bloom_threshold_shader,
            "vs_main",
            "fs_main",
            &bloom_bgl,
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let bloom_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bloom_blur.wgsl").into()),
        });
        let bloom_blur_pipeline = make_fs_pipeline(
            "bloom_blur_pipeline",
            bloom_blur_shader,
            "vs_main",
            "fs_main",
            &bloom_bgl,
            wgpu::TextureFormat::Rgba16Float,
            None,
        );

        // SSAO pipelines
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao.wgsl").into()),
        });
        let ssao_pipeline = make_fs_pipeline(
            "ssao_pipeline",
            ssao_shader,
            "vs_main",
            "fs_main",
            &ssao_bgl,
            wgpu::TextureFormat::R8Unorm,
            None,
        );
        let ssao_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao_blur.wgsl").into()),
        });
        let ssao_blur_pipeline = make_fs_pipeline(
            "ssao_blur_pipeline",
            ssao_blur_shader,
            "vs_main",
            "fs_main",
            &ssao_blur_bgl,
            wgpu::TextureFormat::R8Unorm,
            None,
        );

        // Contact shadow pipeline
        let cs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/contact_shadow.wgsl").into()),
        });
        let cs_pipeline = make_fs_pipeline(
            "contact_shadow_pipeline",
            cs_shader,
            "vs_main",
            "fs_main",
            &cs_bgl,
            wgpu::TextureFormat::R8Unorm,
            None,
        );

        // FXAA pipeline
        let fxaa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fxaa_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fxaa.wgsl").into()),
        });
        let fxaa_pipeline = make_fs_pipeline(
            "fxaa_pipeline",
            fxaa_shader,
            "vs_main",
            "fs_main",
            &fxaa_bgl,
            output_format,
            None,
        );

        // OIT composite pipeline
        let oit_comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oit_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/oit_composite.wgsl").into()),
        });
        let premul_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let oit_comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oit_composite_pipeline_layout"),
            bind_group_layouts: &[&oit_composite_bgl],
            push_constant_ranges: &[],
        });
        let oit_composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("oit_composite_pipeline"),
                layout: Some(&oit_comp_layout),
                vertex: wgpu::VertexState {
                    module: &oit_comp_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &oit_comp_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

        // OIT mesh pipelines
        let accum_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let reveal_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let oit_depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let oit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_oit_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh_oit.wgsl").into()),
        });
        let oit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oit_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &self.object_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let oit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_pipeline"),
            layout: Some(&oit_layout),
            vertex: wgpu::VertexState {
                module: &oit_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &oit_shader,
                entry_point: Some("fs_oit_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(accum_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: Some(reveal_blend),
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(oit_depth_stencil.clone()),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let skinned_oit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned_oit_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &self.object_bind_group_layout,
                &self.skinning.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let skinned_oit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_skinned_shader_oit"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mesh_skinned.wgsl").into(),
            ),
        });
        let skinned_oit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned_oit_pipeline"),
            layout: Some(&skinned_oit_layout),
            vertex: wgpu::VertexState {
                module: &skinned_oit_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &oit_shader,
                entry_point: Some("fs_oit_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(accum_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: Some(reveal_blend),
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(oit_depth_stencil.clone()),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        self.skinned_oit_pipeline = Some(skinned_oit_pipeline);

        let oit_instanced_pipeline = if let Some(ref instance_bgl) = self.instance_bind_group_layout
        {
            let instanced_oit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh_instanced_oit_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/mesh_instanced_oit.wgsl").into(),
                ),
            });
            let instanced_oit_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("oit_instanced_pipeline_layout"),
                    bind_group_layouts: &[&self.camera_bind_group_layout, instance_bgl],
                    push_constant_ranges: &[],
                });
            Some(
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("oit_instanced_pipeline"),
                    layout: Some(&instanced_oit_layout),
                    vertex: wgpu::VertexState {
                        module: &instanced_oit_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[Vertex::buffer_layout()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &instanced_oit_shader,
                        entry_point: Some("fs_oit_main"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: Some(accum_blend),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::R8Unorm,
                                blend: Some(reveal_blend),
                                write_mask: wgpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(oit_depth_stencil),
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        ..Default::default()
                    },
                    multiview: None,
                    cache: None,
                }),
            )
        } else {
            None
        };

        // HDR scene pipelines
        let hdr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
        let make_hdr_mesh = |label: &str,
                             cull: Option<wgpu::Face>,
                             blend: Option<wgpu::BlendState>,
                             topo: wgpu::PrimitiveTopology,
                             depth_write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&hdr_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &hdr_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &hdr_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: topo,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: depth_write,
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
            })
        };
        let hdr_solid_pipeline = make_hdr_mesh(
            "hdr_solid_pipeline",
            Some(wgpu::Face::Back),
            None,
            wgpu::PrimitiveTopology::TriangleList,
            true,
        );
        let hdr_solid_two_sided_pipeline = make_hdr_mesh(
            "hdr_solid_two_sided_pipeline",
            None,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            true,
        );
        let hdr_transparent_pipeline = make_hdr_mesh(
            "hdr_transparent_pipeline",
            None,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::PrimitiveTopology::TriangleList,
            false,
        );
        let hdr_wireframe_pipeline = make_hdr_mesh(
            "hdr_wireframe_pipeline",
            None,
            None,
            wgpu::PrimitiveTopology::LineList,
            true,
        );

        let hdr_overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
                module: &hdr_overlay_shader,
                entry_point: Some("vs_main"),
                buffers: &[OverlayVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &hdr_overlay_shader,
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

        // Outline composite pipelines
        let outline_comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outline_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/outline_composite.wgsl").into(),
            ),
        });
        let outline_comp_blend = wgpu::BlendState {
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
        };
        let outline_comp_ds = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let outline_comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outline_composite_layout"),
            bind_group_layouts: &[&outline_composite_bgl],
            push_constant_ranges: &[],
        });
        let make_outline_pipeline = |label: &str, fmt: wgpu::TextureFormat, sample_count: u32| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&outline_comp_layout),
                vertex: wgpu::VertexState {
                    module: &outline_comp_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &outline_comp_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(outline_comp_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(outline_comp_ds.clone()),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        };
        let outline_composite_pipeline_single =
            make_outline_pipeline("outline_composite_pipeline_single", self.target_format, 1);
        let outline_composite_pipeline_msaa = make_outline_pipeline(
            "outline_composite_pipeline_msaa",
            self.target_format,
            self.sample_count,
        );
        let outline_composite_pipeline_hdr = make_outline_pipeline(
            "outline_composite_pipeline_hdr",
            wgpu::TextureFormat::Rgba16Float,
            1,
        );

        // Store everything
        self.pp_linear_sampler = Some(linear_sampler);
        self.pp_nearest_sampler = Some(nearest_sampler);
        self.fxaa_sampler = Some(fxaa_sampler);
        self.oit_composite_sampler = Some(oit_sampler);
        self.outline_composite_sampler = Some(outline_sampler);

        self.tone_map_bgl = Some(tone_map_bgl);
        self.bloom_bgl = Some(bloom_bgl);
        self.ssao_bgl = Some(ssao_bgl);
        self.ssao_blur_bgl = Some(ssao_blur_bgl);
        self.contact_shadow_bgl = Some(cs_bgl);
        self.fxaa_bgl = Some(fxaa_bgl);
        self.oit_composite_bgl = Some(oit_composite_bgl);
        self.outline_composite_bgl = Some(outline_composite_bgl);

        self.tone_map_pipeline = Some(tone_map_pipeline);
        self.bloom_threshold_pipeline = Some(bloom_threshold_pipeline);
        self.bloom_blur_pipeline = Some(bloom_blur_pipeline);
        self.ssao_pipeline = Some(ssao_pipeline);
        self.ssao_blur_pipeline = Some(ssao_blur_pipeline);
        self.contact_shadow_pipeline = Some(cs_pipeline);
        self.fxaa_pipeline = Some(fxaa_pipeline);

        // --- SSAA resolve pipeline ---
        let ssaa_resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssaa_resolve_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
        let ssaa_resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssaa_resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssaa_resolve.wgsl").into()),
        });
        let ssaa_resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssaa_resolve_layout"),
            bind_group_layouts: &[&ssaa_resolve_bgl],
            push_constant_ranges: &[],
        });
        let ssaa_resolve_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ssaa_resolve_pipeline"),
                layout: Some(&ssaa_resolve_layout),
                vertex: wgpu::VertexState {
                    module: &ssaa_resolve_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssaa_resolve_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
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
        self.ssaa_resolve_bgl = Some(ssaa_resolve_bgl);
        self.ssaa_resolve_pipeline = Some(ssaa_resolve_pipeline);

        // DoF pipeline
        let dof_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_bgl"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let dof_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dof.wgsl").into()),
        });
        let dof_pipeline = make_fs_pipeline(
            "dof_pipeline",
            dof_shader,
            "vs_main",
            "fs_main",
            &dof_bgl,
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        self.dof_bgl = Some(dof_bgl);
        self.dof_pipeline = Some(dof_pipeline);

        self.oit_pipeline = Some(oit_pipeline);
        if let Some(p) = oit_instanced_pipeline {
            self.oit_instanced_pipeline = Some(p);
        }
        self.oit_composite_pipeline = Some(oit_composite_pipeline);
        // HDR skinned variant: same shape as hdr_solid_pipeline but with the
        // skinned vertex stage from `mesh_skinned.wgsl` and a 3-group layout
        // (camera, object, skin sidecar). Fragment stage is shared with the
        // standard HDR mesh shader.
        let hdr_skinned_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh_skinned_shader_hdr"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/mesh_skinned.wgsl").into(),
                ),
            });
        let hdr_skinned_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdr_skinned_solid_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &self.object_bind_group_layout,
                &self.skinning.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let hdr_skinned_solid_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_skinned_solid_pipeline"),
                layout: Some(&hdr_skinned_layout),
                vertex: wgpu::VertexState {
                    module: &hdr_skinned_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &hdr_shader,
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
        let hdr_skinned_solid_two_sided_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_skinned_solid_two_sided_pipeline"),
                layout: Some(&hdr_skinned_layout),
                vertex: wgpu::VertexState {
                    module: &hdr_skinned_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &hdr_shader,
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
                    cull_mode: None,
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
        let hdr_skinned_transparent_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_skinned_transparent_pipeline"),
                layout: Some(&hdr_skinned_layout),
                vertex: wgpu::VertexState {
                    module: &hdr_skinned_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &hdr_shader,
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

        let hdr_skinned_wireframe_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr_skinned_wireframe_pipeline"),
                layout: Some(&hdr_skinned_layout),
                vertex: wgpu::VertexState {
                    module: &hdr_skinned_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &hdr_shader,
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
                depth_stencil: Some(hdr_depth_stencil.clone()),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

        self.hdr_solid_pipeline = Some(hdr_solid_pipeline);
        self.hdr_skinned_solid_pipeline = Some(hdr_skinned_solid_pipeline);
        self.hdr_skinned_solid_two_sided_pipeline = Some(hdr_skinned_solid_two_sided_pipeline);
        self.hdr_skinned_transparent_pipeline = Some(hdr_skinned_transparent_pipeline);
        self.hdr_skinned_wireframe_pipeline = Some(hdr_skinned_wireframe_pipeline);
        self.hdr_solid_two_sided_pipeline = Some(hdr_solid_two_sided_pipeline);
        self.hdr_transparent_pipeline = Some(hdr_transparent_pipeline);
        self.hdr_wireframe_pipeline = Some(hdr_wireframe_pipeline);
        self.hdr_overlay_pipeline = Some(hdr_overlay_pipeline);
        self.outline_composite_pipeline_single = Some(outline_composite_pipeline_single);
        self.outline_composite_pipeline_msaa = Some(outline_composite_pipeline_msaa);
        self.outline_composite_pipeline_hdr = Some(outline_composite_pipeline_hdr);

        let _ = hdr_depth_stencil; // used in make_hdr_mesh closure above

        // --- Surface LIC shared resources ---
        if self.lic_noise_sampler.is_none() {
            // Bilinear sampler used for lic_vector_texture in the advect pass.
            let samp = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("lic_linear_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            self.lic_noise_sampler = Some(samp);
        }

        // LIC surface BGL (group 1): object uniform only.
        // Flow vectors are passed as vertex buffer 1 (not a storage binding).
        if self.lic_surface_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lic_surface_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            self.lic_surface_bgl = Some(bgl);
        }

        // LIC advect BGL (fullscreen): params uniform, vector tex, noise tex, sampler x2.
        if self.lic_advect_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lic_advect_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
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
                ],
            });
            self.lic_advect_bgl = Some(bgl);
        }

        // LIC surface pipeline: renders mesh into Rgba8Unorm lic_vector_texture.
        // Group 0 = camera_bind_group_layout (already on self), group 1 = lic_surface_bgl.
        if self.lic_surface_pipeline.is_none() {
            if let Some(surface_bgl) = self.lic_surface_bgl.as_ref() {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lic_surface_shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("../shaders/lic_surface.wgsl").into(),
                    ),
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("lic_surface_layout"),
                    bind_group_layouts: &[&self.camera_bind_group_layout, surface_bgl],
                    push_constant_ranges: &[],
                });
                // Vertex buffer 0: full Vertex stride, position at location 0.
                let lic_vertex_layout = wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                };
                // Vertex buffer 1: tightly-packed [f32;3] flow vectors at location 1.
                let lic_flow_layout = wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                };
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("lic_surface_pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[lic_vertex_layout, lic_flow_layout],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
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
                });
                self.lic_surface_pipeline = Some(pipeline);
            }
        }

        // LIC advect pipeline: fullscreen render into R8Unorm lic_output_texture.
        if self.lic_advect_pipeline.is_none() {
            if let Some(advect_bgl) = &self.lic_advect_bgl {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lic_advect_shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("../shaders/lic_advect.wgsl").into(),
                    ),
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("lic_advect_layout"),
                    bind_group_layouts: &[advect_bgl],
                    push_constant_ranges: &[],
                });
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("lic_advect_pipeline"),
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
                            format: wgpu::TextureFormat::R8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
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
                self.lic_advect_pipeline = Some(pipeline);
            }
        }

        // --- Depth blit pipeline (lazily created once) ---
        // Copies a scene-resolution depth texture to a native-resolution depth-only target.
        // Used when render_scale < 1.0 so post-tone-map passes can use a native-res depth buf.
        if self.depth_blit_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("depth_blit_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("depth_blit_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/depth_blit.wgsl").into(),
                ),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("depth_blit_layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth_blit_pipeline"),
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
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
            self.depth_blit_bgl = Some(bgl);
            self.depth_blit_pipeline = Some(pipeline);
        }

        // --- Decal shared resources (D1) ---
        if self.decal_depth_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("decal_depth_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            self.decal_depth_bgl = Some(bgl);
        }
        if self.decal_sampler.is_none() {
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("decal_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            self.decal_sampler = Some(sampler);
        }
    }

    /// Create a fresh [`ViewportHdrState`] for the given viewport dimensions.
    ///
    /// `w, h` are the native output dimensions. `scene_w, scene_h` are the effective
    /// render target dimensions after applying render scale (equal to `w, h` when
    /// render_scale = 1.0). Scene-side textures (HDR colour, depth, bloom, SSAO, etc.)
    /// are allocated at `scene_w x scene_h`; output-side textures (FXAA) remain at
    /// `w x h`. The tone map pass upscales from scene to output resolution.
    ///
    /// [`ensure_hdr_shared`](Self::ensure_hdr_shared) must have been called first so that
    /// BGLs, samplers, and placeholder textures are available on `self`.
    pub(crate) fn create_hdr_viewport_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
        w: u32,
        h: u32,
        scene_w: u32,
        scene_h: u32,
        ssaa_factor: u32,
    ) -> ViewportHdrState {
        let w = w.max(1);
        let h = h.max(1);
        let scene_w = scene_w.max(1);
        let scene_h = scene_h.max(1);
        // Half-resolution for bloom ping/pong -- based on scene size.
        let hw = (scene_w / 2).max(1);
        let hh = (scene_h / 2).max(1);
        let ssaa_factor = ssaa_factor.max(1);

        let make_tex = |label: &str,
                        fmt: wgpu::TextureFormat,
                        tw: u32,
                        th: u32,
                        extra_usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: tw,
                    height: th,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | extra_usage,
                view_formats: &[],
            })
        };

        // HDR scene colour and depth -- at scene resolution (render_scale * output).
        let hdr_tex = make_tex(
            "hdr_texture",
            wgpu::TextureFormat::Rgba16Float,
            scene_w,
            scene_h,
            wgpu::TextureUsages::empty(),
        );
        let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let hdr_depth_tex = make_tex(
            "hdr_depth_texture",
            wgpu::TextureFormat::Depth24PlusStencil8,
            scene_w,
            scene_h,
            wgpu::TextureUsages::empty(),
        );
        let hdr_depth_view = hdr_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let hdr_depth_only_view = hdr_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        // Bloom -- at scene resolution (hw/hh are scene_w/2, scene_h/2).
        let bloom_threshold_tex = make_tex(
            "bloom_threshold_texture",
            wgpu::TextureFormat::Rgba16Float,
            scene_w,
            scene_h,
            wgpu::TextureUsages::empty(),
        );
        let bloom_threshold_view =
            bloom_threshold_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bloom_ping_tex = make_tex(
            "bloom_ping_texture",
            wgpu::TextureFormat::Rgba16Float,
            hw,
            hh,
            wgpu::TextureUsages::empty(),
        );
        let bloom_ping_view = bloom_ping_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bloom_pong_tex = make_tex(
            "bloom_pong_texture",
            wgpu::TextureFormat::Rgba16Float,
            hw,
            hh,
            wgpu::TextureUsages::empty(),
        );
        let bloom_pong_view = bloom_pong_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // SSAO -- at scene resolution.
        let ssao_tex = make_tex(
            "ssao_texture",
            wgpu::TextureFormat::R8Unorm,
            scene_w,
            scene_h,
            wgpu::TextureUsages::empty(),
        );
        let ssao_view = ssao_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let ssao_blur_tex = make_tex(
            "ssao_blur_texture",
            wgpu::TextureFormat::R8Unorm,
            scene_w,
            scene_h,
            wgpu::TextureUsages::empty(),
        );
        let ssao_blur_view = ssao_blur_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth of field -- at scene resolution.
        let dof_tex = make_tex(
            "dof_texture",
            wgpu::TextureFormat::Rgba16Float,
            scene_w,
            scene_h,
            wgpu::TextureUsages::empty(),
        );
        let dof_view = dof_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Contact shadow -- at scene resolution.
        let cs_tex = make_tex(
            "contact_shadow_texture",
            wgpu::TextureFormat::R8Unorm,
            scene_w,
            scene_h,
            wgpu::TextureUsages::empty(),
        );
        let cs_view = cs_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // FXAA -- at scene resolution so the whole post-process chain runs at
        // the scaled size when render_scale < 1.0.
        let fxaa_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fxaa_texture"),
            size: wgpu::Extent3d {
                width: scene_w,
                height: scene_h,
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

        // Outline offscreen : mask (R8), colour (target_format), and depth -- at scene resolution.
        let outline_mask_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("outline_mask_texture"),
            size: wgpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let outline_mask_view =
            outline_mask_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let outline_colour_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("outline_colour_texture"),
            size: wgpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let outline_colour_view =
            outline_colour_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let outline_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("outline_depth_texture"),
            size: wgpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let outline_depth_view =
            outline_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Uniform buffers
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
        let bloom_h_uniform_buf = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom_h_uniform_buf"),
                size: std::mem::size_of::<BloomUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &buf,
                0,
                bytemuck::cast_slice(&[BloomUniform {
                    threshold: 0.0,
                    intensity: 0.0,
                    horizontal: 1,
                    _pad: 0,
                }]),
            );
            buf
        };
        let bloom_v_uniform_buf = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom_v_uniform_buf"),
                size: std::mem::size_of::<BloomUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &buf,
                0,
                bytemuck::cast_slice(&[BloomUniform {
                    threshold: 0.0,
                    intensity: 0.0,
                    horizontal: 0,
                    _pad: 0,
                }]),
            );
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
        let dof_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_uniform_buf"),
            size: std::mem::size_of::<DofUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shared references needed for bind groups
        let linear_sampler = self
            .pp_linear_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let nearest_sampler = self
            .pp_nearest_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let fxaa_sampler = self
            .fxaa_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let oit_sampler = self
            .oit_composite_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let outline_sampler = self
            .outline_composite_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let bloom_placeholder_view = self
            .bloom_placeholder_view
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ao_placeholder_view = self
            .ao_placeholder_view
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let cs_placeholder_view = self
            .cs_placeholder_view
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ssao_noise_view = self
            .ssao_noise_view
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ssao_kernel_buf = self
            .ssao_kernel_buf
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let tone_map_bgl = self
            .tone_map_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let bloom_bgl = self
            .bloom_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ssao_bgl = self
            .ssao_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ssao_blur_bgl = self
            .ssao_blur_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let cs_bgl = self
            .contact_shadow_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let fxaa_bgl = self
            .fxaa_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let oit_composite_bgl = self
            .oit_composite_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let outline_composite_bgl = self
            .outline_composite_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");

        // Bind groups
        let tone_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_map_bg"),
            layout: tone_map_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(
                        self.lic_placeholder_view
                            .as_ref()
                            .expect("ensure_hdr_shared not called"),
                    ),
                },
            ],
        });
        let bloom_threshold_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_threshold_bg"),
            layout: bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let bloom_blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_blur_h_bg"),
            layout: bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_threshold_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_h_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let bloom_blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_blur_v_bg"),
            layout: bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_ping_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_v_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let bloom_blur_h_pong_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_blur_h_pong_bg"),
            layout: bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_pong_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_h_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let ssao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_bg"),
            layout: ssao_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ssao_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
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
            layout: ssao_blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
            ],
        });
        let dof_bgl = self.dof_bgl.as_ref().expect("ensure_hdr_shared not called");
        let dof_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_bg"),
            layout: dof_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dof_uniform_buf.as_entire_binding(),
                },
            ],
        });
        // dof_bind_group: same layout as dof_bg but reads dof_view (for tone map input).
        // This is rebuilt in rebuild_tone_map_bind_group when dof is active.
        let dof_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_bind_group_placeholder"),
            layout: dof_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dof_uniform_buf.as_entire_binding(),
                },
            ],
        });

        let contact_shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact_shadow_bg"),
            layout: cs_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cs_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let fxaa_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa_bg"),
            layout: fxaa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fxaa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(fxaa_sampler),
                },
            ],
        });
        let outline_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outline_composite_bg"),
            layout: outline_composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&outline_colour_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(outline_sampler),
                },
            ],
        });

        // Edge-detection bind group : reads the R8 mask, writes outline ring.
        let outline_edge_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outline_edge_uniform_buf"),
            size: std::mem::size_of::<OutlineEdgeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let outline_edge_bgl = &self.outline_edge_bgl;
        let outline_edge_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outline_edge_bg"),
            layout: outline_edge_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&outline_mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(outline_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: outline_edge_uniform_buf.as_entire_binding(),
                },
            ],
        });

        // OIT composite bind group placeholder (created lazily via ensure_viewport_oit)
        // We create a dummy one using placeholders so the bind group is always valid.
        // It will be rebuilt on first ensure_viewport_oit call.
        let oit_composite_bg_placeholder = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_composite_bg_placeholder"),
            layout: oit_composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(bloom_placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(bloom_placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(oit_sampler),
                },
            ],
        });

        let _ = oit_composite_bg_placeholder; // will not use the placeholder - OIT is Option<>

        // --- SSAA targets (allocated when ssaa_factor > 1) ---
        let (
            ssaa_colour_texture,
            ssaa_colour_view,
            ssaa_depth_texture,
            ssaa_depth_view,
            ssaa_resolve_bind_group,
            ssaa_uniform_buf,
        ) = if ssaa_factor > 1 {
            let sw = scene_w * ssaa_factor;
            let sh = scene_h * ssaa_factor;
            let ssaa_colour_tex = make_tex(
                "ssaa_colour_texture",
                wgpu::TextureFormat::Rgba16Float,
                sw,
                sh,
                wgpu::TextureUsages::empty(),
            );
            let ssaa_colour_view =
                ssaa_colour_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let ssaa_depth_tex = make_tex(
                "ssaa_depth_texture",
                wgpu::TextureFormat::Depth24PlusStencil8,
                sw,
                sh,
                wgpu::TextureUsages::empty(),
            );
            let ssaa_depth_view =
                ssaa_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

            // Build the resolve bind group if the pipeline is available.
            let (ssaa_resolve_bg, ssaa_ubuf) = if let (Some(bgl), Some(nearest)) =
                (&self.ssaa_resolve_bgl, &self.pp_nearest_sampler)
            {
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct SsaaUniformData {
                    factor: u32,
                    _pad: [u32; 3],
                }
                let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ssaa_uniform_buf"),
                    size: std::mem::size_of::<SsaaUniformData>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(
                    &ubuf,
                    0,
                    bytemuck::cast_slice(&[SsaaUniformData {
                        factor: ssaa_factor,
                        _pad: [0; 3],
                    }]),
                );
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssaa_resolve_bg"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&ssaa_colour_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(nearest),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ubuf.as_entire_binding(),
                        },
                    ],
                });
                (Some(bg), Some(ubuf))
            } else {
                (None, None)
            };

            (
                Some(ssaa_colour_tex),
                Some(ssaa_colour_view),
                Some(ssaa_depth_tex),
                Some(ssaa_depth_view),
                ssaa_resolve_bg,
                ssaa_ubuf,
            )
        } else {
            (None, None, None, None, None, None)
        };

        // --- Surface LIC per-viewport textures and bind group -- at scene resolution ---
        let lic_vector_tex = make_tex(
            "lic_vector",
            wgpu::TextureFormat::Rgba8Unorm,
            scene_w,
            scene_h,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let lic_vector_view = lic_vector_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let lic_output_tex = make_tex(
            "lic_output",
            wgpu::TextureFormat::R8Unorm,
            scene_w,
            scene_h,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let lic_output_view = lic_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Per-pixel white noise at scene resolution.
        let lic_noise_data: Vec<u8> = (0u32..scene_w * scene_h)
            .map(|i| {
                // xorshift32 mix of pixel index -- uniform [0,255] distribution.
                let mut v = i.wrapping_add(1).wrapping_mul(2246822519);
                v ^= v >> 13;
                v ^= v << 17;
                v ^= v >> 5;
                v as u8
            })
            .collect();
        let lic_noise_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lic_noise"),
            size: wgpu::Extent3d {
                width: scene_w,
                height: scene_h,
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
                texture: &lic_noise_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &lic_noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(scene_w),
                rows_per_image: Some(scene_h),
            },
            wgpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
        );
        let lic_noise_view = lic_noise_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let lic_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lic_advect_uniform"),
            size: std::mem::size_of::<crate::resources::types::LicAdvectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lic_advect_bgl = self
            .lic_advect_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let lic_advect_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lic_advect_bg"),
            layout: lic_advect_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lic_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&lic_vector_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&lic_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(
                        self.lic_noise_sampler
                            .as_ref()
                            .expect("ensure_hdr_shared not called"),
                    ),
                },
            ],
        });

        // Output-resolution depth for post-tone-map passes.
        // When render scale = 1.0 (scene == output), reuse hdr_depth as a second view.
        // When render scale < 1.0, allocate a separate native-res texture and create a
        // bind group so the depth blit pass can copy hdr_depth into it each frame.
        let (output_depth_texture, output_depth_view, depth_blit_bind_group) =
            if scene_w != w || scene_h != h {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("output_depth_texture"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                let bg = self.depth_blit_bgl.as_ref().map(|bgl| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("depth_blit_bg"),
                        layout: bgl,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&hdr_depth_only_view),
                        }],
                    })
                });
                (Some(tex), view, bg)
            } else {
                let view = hdr_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
                (None, view, None)
            };

        // HDR upscale target: when scene_size != output_size, tone-map and FXAA
        // run at scene resolution and write to this texture. An upscale-blit pass
        // then copies the result to output_view at native resolution.
        let (upscale_texture, upscale_view, upscale_bind_group) =
            if scene_w != w || scene_h != h {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("hdr_upscale_texture"),
                    size: wgpu::Extent3d {
                        width: scene_w,
                        height: scene_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: output_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                let bgl = self.dyn_res_upscale_bgl.as_ref().unwrap();
                let sampler = self.dyn_res_linear_sampler.as_ref().unwrap();
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hdr_upscale_bg"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                });
                (Some(tex), Some(view), Some(bg))
            } else {
                (None, None, None)
            };

        let decal_depth_bg = self.create_decal_depth_bg(device, &hdr_depth_only_view);

        ViewportHdrState {
            hdr_texture: hdr_tex,
            hdr_view,
            hdr_depth_texture: hdr_depth_tex,
            hdr_depth_view,
            hdr_depth_only_view,
            bloom_threshold_texture: bloom_threshold_tex,
            bloom_threshold_view,
            bloom_ping_texture: bloom_ping_tex,
            bloom_ping_view,
            bloom_pong_texture: bloom_pong_tex,
            bloom_pong_view,
            ssao_texture: ssao_tex,
            ssao_view,
            ssao_blur_texture: ssao_blur_tex,
            ssao_blur_view,
            dof_texture: dof_tex,
            dof_view,
            dof_bind_group,
            dof_uniform_buf,
            contact_shadow_texture: cs_tex,
            contact_shadow_view: cs_view,
            fxaa_texture: fxaa_tex,
            fxaa_view,
            ssaa_colour_texture,
            ssaa_colour_view,
            ssaa_depth_texture,
            ssaa_depth_view,
            ssaa_resolve_bind_group,
            ssaa_uniform_buf,
            ssaa_factor,
            oit_accum_texture: None,
            oit_accum_view: None,
            oit_reveal_texture: None,
            oit_reveal_view: None,
            oit_composite_bind_group: None,
            oit_size: [0, 0],
            outline_mask_texture: outline_mask_tex,
            outline_mask_view,
            outline_colour_texture: outline_colour_tex,
            outline_colour_view,
            outline_depth_texture: outline_depth_tex,
            outline_depth_view,
            outline_edge_bind_group,
            outline_edge_uniform_buf,
            outline_composite_bind_group,
            tone_map_bind_group,
            bloom_threshold_bg,
            bloom_blur_h_bg,
            bloom_blur_v_bg,
            bloom_blur_h_pong_bg,
            ssao_bg,
            ssao_blur_bg,
            dof_bg,
            contact_shadow_bg,
            fxaa_bind_group,
            tone_map_uniform_buf,
            bloom_uniform_buf,
            bloom_h_uniform_buf,
            bloom_v_uniform_buf,
            ssao_uniform_buf,
            contact_shadow_uniform_buf: cs_uniform_buf,
            lic_vector_texture: lic_vector_tex,
            lic_vector_view,
            lic_output_texture: lic_output_tex,
            lic_output_view,
            lic_noise_texture: lic_noise_tex,
            lic_noise_view,
            lic_advect_bind_group,
            lic_uniform_buf,
            output_size: [w, h],
            scene_size: [scene_w, scene_h],
            output_depth_texture,
            output_depth_view,
            depth_blit_bind_group,
            upscale_texture,
            upscale_view,
            upscale_bind_group,
            decal_depth_bg,
        }
    }

    /// Rebuild the tone-map bind group for a per-viewport HDR state, swapping in
    /// the active bloom/AO/contact-shadow texture views.
    pub(crate) fn rebuild_tone_map_bind_group(
        &self,
        device: &wgpu::Device,
        hdr: &mut ViewportHdrState,
        use_bloom: bool,
        use_ssao: bool,
        use_contact_shadows: bool,
        use_lic: bool,
        use_dof: bool,
    ) {
        let bgl = match &self.tone_map_bgl {
            Some(b) => b,
            None => return,
        };
        let sampler = match &self.pp_linear_sampler {
            Some(s) => s,
            None => return,
        };
        let bloom_placeholder = match &self.bloom_placeholder_view {
            Some(v) => v,
            None => return,
        };
        let ao_placeholder = match &self.ao_placeholder_view {
            Some(v) => v,
            None => return,
        };
        let cs_placeholder = match &self.cs_placeholder_view {
            Some(v) => v,
            None => return,
        };

        let bloom_view = if use_bloom {
            &hdr.bloom_pong_view
        } else {
            bloom_placeholder
        };
        let ao_view = if use_ssao {
            &hdr.ssao_blur_view
        } else {
            ao_placeholder
        };
        let cs_view = if use_contact_shadows {
            &hdr.contact_shadow_view
        } else {
            cs_placeholder
        };

        let tone_map_hdr_input: &wgpu::TextureView = if use_dof {
            &hdr.dof_view
        } else {
            &hdr.hdr_view
        };
        hdr.tone_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_map_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(tone_map_hdr_input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: hdr.tone_map_uniform_buf.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&hdr.hdr_depth_only_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(if use_lic {
                        &hdr.lic_output_view
                    } else {
                        self.lic_placeholder_view.as_ref().unwrap_or(cs_placeholder)
                    }),
                },
            ],
        });
    }

    /// Ensure OIT (order-independent transparency) render targets exist for the
    /// given per-viewport HDR state, creating or resizing them as needed.
    pub(crate) fn ensure_viewport_oit(
        &self,
        device: &wgpu::Device,
        hdr: &mut ViewportHdrState,
        w: u32,
        h: u32,
    ) {
        let w = w.max(1);
        let h = h.max(1);
        if hdr.oit_size == [w, h] && hdr.oit_accum_texture.is_some() {
            return;
        }
        hdr.oit_size = [w, h];

        let accum_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("oit_accum_texture"),
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
        let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let reveal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("oit_reveal_texture"),
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
        let reveal_view = reveal_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = self
            .oit_composite_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let bgl = self
            .oit_composite_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oit_composite_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&reveal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        hdr.oit_accum_texture = Some(accum_tex);
        hdr.oit_accum_view = Some(accum_view);
        hdr.oit_reveal_texture = Some(reveal_tex);
        hdr.oit_reveal_view = Some(reveal_view);
        hdr.oit_composite_bind_group = Some(composite_bg);
    }
}
