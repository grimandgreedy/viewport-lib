use super::*;

impl ViewportGpuResources {
    /// Create all GPU resources for the viewport.
    ///
    /// Call once at application startup. `target_format` must match the swap-chain surface
    /// format. Use `sample_count = 1` unless the caller is providing MSAA resolve targets.
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        use wgpu;

        // ------------------------------------------------------------------
        // Shader module
        // ------------------------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh.wgsl").into()),
        });

        // ------------------------------------------------------------------
        // Bind group layouts
        // ------------------------------------------------------------------
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: clip planes uniform (section view clipping).
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: shadow atlas uniform (CSM matrices, splits, PCSS params).
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
                // Binding 6: clip volume uniform (box/sphere/plane extended clip region).
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Object bind group layout (group 1 for non-instanced pipelines).
        // binding 0: per-object uniform (model matrix, material, selection state)
        // binding 1: albedo texture (filterable)
        // binding 2: shared filtering sampler
        // binding 3: normal map texture (filterable)
        // binding 4: AO map texture (filterable)
        //
        // Textures are co-located in group 1 (rather than a separate group 2) so that
        // the total bind group count stays at 2, compatible with iced's wgpu device
        // which hardcodes max_bind_groups = 2 in its DeviceDescriptor.
        let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object_bgl"),
            entries: &[
                // binding 0: per-object uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: albedo texture (filterable)
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
                // binding 2: shared filtering sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: normal map texture (filterable)
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
                // binding 4: AO map texture (filterable)
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
                // binding 5: LUT (colormap) texture (256×1 Rgba8Unorm, FRAGMENT, filterable)
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
                // binding 6: scalar attribute storage buffer (VERTEX | FRAGMENT, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Texture-only bind group layout — kept for the instanced pipeline (group 1 bindings
        // 1-4 are added alongside the storage buffer binding 0 in init_instanced_pipeline).
        // Also used as the standalone layout when creating material bind groups keyed by
        // texture combination for the instanced path.
        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bgl"),
            entries: &[
                // binding 0: albedo texture (filterable)
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
                // binding 1: shared filtering sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: normal map texture (filterable)
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
                // binding 3: AO map texture (filterable)
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
            ],
        });

        // ------------------------------------------------------------------
        // Pipeline layout (shared between solid and transparent pipelines)
        // Groups: 0=camera, 1=object+texture (merged to stay within iced's max_bind_groups=2)
        // ------------------------------------------------------------------
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[&camera_bgl, &object_bgl],
            push_constant_ranges: &[],
        });

        // ------------------------------------------------------------------
        // Depth stencil state (shared between solid and wireframe pipelines)
        // ------------------------------------------------------------------
        let depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // ------------------------------------------------------------------
        // Solid render pipeline (TriangleList)
        // ------------------------------------------------------------------
        let solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid_pipeline"),
            layout: Some(&pipeline_layout),
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
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ------------------------------------------------------------------
        // Solid two-sided render pipeline (TriangleList, no blending, no culling)
        // Identical to solid_pipeline but with cull_mode = None.
        // Used for analytical surfaces (plots, isosurfaces) viewed from both sides.
        // ------------------------------------------------------------------
        let solid_two_sided_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("solid_two_sided_pipeline"),
                layout: Some(&pipeline_layout),
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
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None, // No culling: surface visible from both sides.
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(depth_stencil.clone()),
                // The outline/stencil passes render into a dedicated offscreen target
                // allocated at sample_count = 1, even when the main scene uses MSAA.
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        // ------------------------------------------------------------------
        // Transparent render pipeline (TriangleList, alpha blending)
        // Identical to solid_pipeline but with alpha blending enabled.
        // Used for objects with material.opacity < 1.0.
        // ------------------------------------------------------------------
        let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("transparent_pipeline"),
            layout: Some(&pipeline_layout),
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
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for transparent objects (viewed from all angles).
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false, // Transparent objects don't write depth.
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            // The outline ring is also rendered into the same single-sample
            // offscreen outline target.
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ------------------------------------------------------------------
        // Wireframe render pipeline (LineList, no back-face culling)
        // ------------------------------------------------------------------
        let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wireframe_pipeline"),
            layout: Some(&pipeline_layout),
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
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for wireframe lines
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ------------------------------------------------------------------
        // Camera uniform buffer and bind group
        // ------------------------------------------------------------------
        let camera_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniform_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_uniform_buf"),
            size: std::mem::size_of::<LightUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Clip planes uniform buffer (binding 4 of camera bind group).
        // Initialized to count=0 (no active clip planes).
        let clip_planes_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clip_planes_uniform_buf"),
            size: std::mem::size_of::<ClipPlanesUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Clip volume uniform buffer (binding 6 of camera bind group, 128 bytes).
        // Initialized to volume_type=0 (None — no clip volume).
        let clip_volume_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clip_volume_uniform_buf"),
            size: std::mem::size_of::<ClipVolumeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ------------------------------------------------------------------
        // Shadow map texture, sampler, and bind group
        // ------------------------------------------------------------------
        let shadow_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_atlas"),
            size: wgpu::Extent3d {
                width: SHADOW_ATLAS_SIZE,
                height: SHADOW_ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view =
            shadow_map_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Shadow atlas uniform buffer (binding 5).
        let shadow_info_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_info_buf"),
            size: std::mem::size_of::<ShadowAtlasUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: light_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: clip_planes_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: clip_volume_uniform_buf.as_entire_binding(),
                },
            ],
        });

        // ------------------------------------------------------------------
        // Shadow pass pipeline (depth-only, renders from light's POV)
        // ------------------------------------------------------------------
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
        });

        // Shadow pass uses a simple bind group layout: just the light uniform.
        let shadow_camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    // Dynamic offset lets the cascade loop select per-cascade matrix slot
                    // without calling write_buffer inside the render pass (which would be
                    // a no-op per-cascade since wgpu batches all writes before execution).
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[&shadow_camera_bgl, &object_bgl],
                push_constant_ranges: &[],
            });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None, // Depth-only pass.
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front), // Front-face culling reduces shadow acne.
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                // Zero slope-scale to eliminate contact-shadow gap; shader-side shadow_bias
                // handles acne prevention. constant=2 keeps depth testing stable.
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1, // Shadow map is always single-sample.
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Shadow pass uniform buffer — 4 cascade slots × 256 bytes (wgpu dynamic-offset alignment).
        // Each slot holds one 4×4 matrix (64 bytes); the remaining 192 bytes per slot are padding.
        const SHADOW_SLOT_STRIDE: u64 = 256;
        let shadow_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniform_buf"),
            size: 4 * SHADOW_SLOT_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_bind_group"),
            layout: &shadow_camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                // Bind only the first 64-byte matrix slot; dynamic offset selects cascade.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shadow_uniform_buf,
                    offset: 0,
                    size: Some(
                        wgpu::BufferSize::new(std::mem::size_of::<[[f32; 4]; 4]>() as u64).unwrap(),
                    ),
                }),
            }],
        });

        // ------------------------------------------------------------------
        // Gizmo shader module
        // ------------------------------------------------------------------
        let gizmo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gizmo_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gizmo.wgsl").into()),
        });

        // ------------------------------------------------------------------
        // Gizmo bind group layout (group 1: model matrix uniform)
        // ------------------------------------------------------------------
        let gizmo_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gizmo_bgl"),
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

        // ------------------------------------------------------------------
        // Gizmo pipeline layout
        // ------------------------------------------------------------------
        let gizmo_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gizmo_pipeline_layout"),
                bind_group_layouts: &[&camera_bgl, &gizmo_bgl],
                push_constant_ranges: &[],
            });

        // ------------------------------------------------------------------
        // Gizmo render pipeline
        // depth_compare: Always — gizmo always renders on top of scene (Pitfall 8).
        // depth_write_enabled: false — do not corrupt depth buffer.
        // ------------------------------------------------------------------
        let gizmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo_pipeline"),
            layout: Some(&gizmo_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &gizmo_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gizmo_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling: gizmo geometry is viewed from all angles.
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always, // Always on top.
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ------------------------------------------------------------------
        // Gizmo vertex/index buffers (initial mesh: no hover highlight)
        // ------------------------------------------------------------------
        let (gizmo_verts, gizmo_indices) = crate::interaction::gizmo::build_gizmo_mesh(
            crate::interaction::gizmo::GizmoMode::Translate,
            crate::interaction::gizmo::GizmoAxis::None,
            glam::Quat::IDENTITY,
        );

        let gizmo_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo_vertex_buf"),
            size: (std::mem::size_of::<Vertex>() * gizmo_verts.len().max(1)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        gizmo_vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&gizmo_verts));
        gizmo_vertex_buffer.unmap();

        let gizmo_index_count = gizmo_indices.len() as u32;
        let gizmo_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo_index_buf"),
            size: (std::mem::size_of::<u32>() * gizmo_indices.len().max(1)) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        gizmo_index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&gizmo_indices));
        gizmo_index_buffer.unmap();

        // ------------------------------------------------------------------
        // Gizmo uniform buffer (model matrix — identity until first update)
        // ------------------------------------------------------------------
        let gizmo_uniform = crate::interaction::gizmo::GizmoUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
        };
        let gizmo_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo_uniform_buf"),
            size: std::mem::size_of::<crate::interaction::gizmo::GizmoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        gizmo_uniform_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&[gizmo_uniform]));
        gizmo_uniform_buf.unmap();

        let gizmo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gizmo_bind_group"),
            layout: &gizmo_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gizmo_uniform_buf.as_entire_binding(),
            }],
        });

        // ------------------------------------------------------------------
        // Overlay shader module
        // ------------------------------------------------------------------
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/overlay.wgsl").into()),
        });

        // ------------------------------------------------------------------
        // Overlay bind group layout (group 1: model + color uniform)
        // ------------------------------------------------------------------
        let overlay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // ------------------------------------------------------------------
        // Overlay pipeline layout (group 0: camera, group 1: overlay uniform)
        // ------------------------------------------------------------------
        let overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("overlay_pipeline_layout"),
                bind_group_layouts: &[&camera_bgl, &overlay_bgl],
                push_constant_ranges: &[],
            });

        // ------------------------------------------------------------------
        // Overlay render pipeline
        // TriangleList topology with alpha blending for semi-transparent quads.
        // depth_write_enabled: false — do not corrupt depth buffer with overlays.
        // depth_compare: Less — overlays respect depth (hidden by geometry in front).
        // cull_mode: None — quads viewed from both sides.
        // ------------------------------------------------------------------
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_pipeline"),
            layout: Some(&overlay_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_main"),
                buffers: &[OverlayVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // BC quads are visible from both sides.
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false, // Do not write to depth buffer.
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ------------------------------------------------------------------
        // Overlay line pipeline (LineList)
        // Uses the same overlay shader + bind group layout as the triangle overlay.
        // No alpha blending needed for line overlays.
        // depth_write_enabled: false — overlay lines don't corrupt depth buffer.
        // ------------------------------------------------------------------
        let overlay_line_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay_line_pipeline"),
                layout: Some(&overlay_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &overlay_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[OverlayVertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        // ------------------------------------------------------------------
        // Full-screen analytical grid pipeline
        //
        // No vertex buffer. A hardcoded triangle in the vertex shader covers
        // the entire screen. The fragment shader ray-marches to the grid plane,
        // computes analytical anti-aliased lines with fwidth(), and writes
        // clip-space depth via @builtin(frag_depth) for correct occlusion.
        // Horizon fade eliminates clipping artefacts at shallow viewing angles.
        // ------------------------------------------------------------------
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/grid.wgsl").into()),
        });
        let grid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grid_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let grid_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("grid_pipeline_layout"),
                bind_group_layouts: &[&grid_bgl],
                push_constant_ranges: &[],
            });
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid_pipeline"),
            layout: Some(&grid_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // no vertex buffer — positions hardcoded in shader
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        // Default-zero uniform — overwritten every frame in prepare().
        let grid_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_uniform_buf"),
            size: std::mem::size_of::<GridUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_bind_group"),
            layout: &grid_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_uniform_buf.as_entire_binding(),
            }],
        });

        // ------------------------------------------------------------------
        // Axes indicator pipeline (screen-space, no camera, no depth)
        // ------------------------------------------------------------------
        let axes_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("axes_overlay_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/axes_overlay.wgsl").into()),
        });

        let axes_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("axes_pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let axes_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("axes_pipeline"),
            layout: Some(&axes_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &axes_shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::widgets::axes_indicator::AxesVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &axes_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Pre-allocate vertex buffer (resized in prepare if needed).
        let axes_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("axes_vertex_buf"),
            size: (std::mem::size_of::<crate::widgets::axes_indicator::AxesVertex>() * 2048) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ------------------------------------------------------------------
        // Shared material sampler (linear + repeat — reused for all material textures)
        // ------------------------------------------------------------------
        let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ------------------------------------------------------------------
        // Fallback normal map: 1×1 [128, 128, 255, 255] — flat tangent-space normal
        // ------------------------------------------------------------------
        let fallback_normal_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fallback_normal_map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fallback_normal_map_view =
            fallback_normal_map.create_view(&wgpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // Fallback AO map: 1×1 [255, 255, 255, 255] — no occlusion
        // ------------------------------------------------------------------
        let fallback_ao_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fallback_ao_map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fallback_ao_map_view =
            fallback_ao_map.create_view(&wgpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // Fallback texture: 1×1 white RGBA (used when no albedo texture is assigned)
        // ------------------------------------------------------------------
        let fallback_texture = {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fallback_texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Texture pixels are uploaded lazily on first prepare() via queue.write_texture.
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("fallback_texture_sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fallback_texture_bg"),
                layout: &texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&fallback_normal_map_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&fallback_ao_map_view),
                    },
                ],
            });
            GpuTexture {
                texture: tex,
                view,
                sampler,
                bind_group,
            }
        };

        // ------------------------------------------------------------------
        // Colormap / LUT fallback resources
        // ------------------------------------------------------------------
        let fallback_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fallback_lut_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Content of fallback_lut_view is never sampled by the shader when has_attribute=0.
        // Data is intentionally left uninitialised here; it will be a zeroed 1-pixel texture
        // after the GPU zeros it on allocation (implementation-defined but harmless).
        let fallback_lut_view =
            fallback_lut_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let fallback_scalar_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fallback_scalar_buf"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = fallback_scalar_buf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(&[0u8; 4]);
        }
        fallback_scalar_buf.unmap();

        // ------------------------------------------------------------------
        // Hardcoded unit cube mesh (test scene object)
        // Created here — after fallback textures — so the combined bind group
        // can reference the fallback texture views at creation time.
        // ------------------------------------------------------------------
        let (cube_verts, cube_indices) = build_unit_cube();
        let cube_mesh = Self::create_mesh(
            device,
            &object_bgl,
            &fallback_texture.view,
            &fallback_normal_map_view,
            &fallback_ao_map_view,
            &fallback_texture.sampler,
            &fallback_lut_view,
            &fallback_scalar_buf,
            &cube_verts,
            &cube_indices,
        );

        // ------------------------------------------------------------------
        // Outline & x-ray pipelines
        // ------------------------------------------------------------------

        // Bind group layout for OutlineUniform (group 1).
        let outline_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("outline_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outline_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/outline.wgsl").into()),
        });

        let outline_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("outline_pipeline_layout"),
                bind_group_layouts: &[&camera_bgl, &outline_bgl],
                push_constant_ranges: &[],
            });

        // Pass 1: render selected objects writing stencil=1 (same mesh shader, same layout).
        // StencilState: compare=Always, pass_op=Replace -> writes ref(=1) on depth pass.
        let stencil_write_face = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Replace,
        };
        let stencil_write_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stencil_write_pipeline"),
                layout: Some(&pipeline_layout),
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
                        format: target_format,
                        blend: None,
                        // Don't write color — stencil write pass is depth+stencil only.
                        write_mask: wgpu::ColorWrites::empty(),
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
                    stencil: wgpu::StencilState {
                        front: stencil_write_face,
                        back: stencil_write_face,
                        read_mask: 0xFF,
                        write_mask: 0xFF,
                    },
                    bias: wgpu::DepthBiasState::default(),
                }),
                // Outline passes render into a dedicated single-sample target.
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        // Pass 2: draw expanded silhouette ring where stencil != 1 (outline ring).
        // depth_compare=Always so ring always appears on top of occluding geometry.
        let outline_ring_face = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::NotEqual,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Keep,
        };
        let outline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("outline_pipeline"),
            layout: Some(&outline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &outline_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &outline_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // No culling: ring is drawn from both sides.
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState {
                    front: outline_ring_face,
                    back: outline_ring_face,
                    read_mask: 0xFF,
                    write_mask: 0x00, // Don't modify stencil in pass 2.
                },
                bias: wgpu::DepthBiasState::default(),
            }),
            // Outline passes render into a dedicated single-sample target.
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // X-ray pipeline: render selected objects through all geometry as a semi-transparent tint.
        let xray_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("xray_pipeline"),
            layout: Some(&outline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &outline_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &outline_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
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
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            target_format,
            sample_count,
            solid_pipeline,
            solid_two_sided_pipeline,
            transparent_pipeline,
            wireframe_pipeline,
            camera_uniform_buf,
            light_uniform_buf,
            camera_bind_group,
            camera_bind_group_layout: camera_bgl,
            object_bind_group_layout: object_bgl,
            mesh_store: {
                let mut store = crate::resources::mesh_store::MeshStore::new();
                store.insert(cube_mesh);
                store
            },
            shadow_map_texture,
            shadow_map_view,
            shadow_sampler,
            shadow_pipeline,
            shadow_uniform_buf,
            shadow_bind_group,
            shadow_info_buf,
            shadow_atlas_size: SHADOW_ATLAS_SIZE,
            gizmo_pipeline,
            gizmo_vertex_buffer,
            gizmo_index_buffer,
            gizmo_index_count,
            gizmo_uniform_buf,
            gizmo_bind_group,
            overlay_pipeline,
            overlay_line_pipeline,
            grid_pipeline,
            grid_uniform_buf,
            grid_bind_group,
            overlay_bind_group_layout: overlay_bgl,
            constraint_line_buffers: Vec::new(),
            cap_buffers: Vec::new(),
            axes_pipeline,
            axes_vertex_buffer,
            axes_vertex_count: 0,
            texture_bind_group_layout: texture_bgl,
            fallback_texture,
            fallback_normal_map,
            fallback_normal_map_view,
            fallback_ao_map,
            fallback_ao_map_view,
            material_sampler,
            material_bind_groups: std::collections::HashMap::new(),
            textures: Vec::new(),
            fallback_textures_uploaded: false,
            fxaa_texture: None,
            fxaa_view: None,
            fxaa_pipeline: None,
            fxaa_bgl: None,
            fxaa_bind_group: None,
            clip_planes_uniform_buf,
            clip_volume_uniform_buf,
            outline_bind_group_layout: outline_bgl,
            stencil_write_pipeline,
            outline_pipeline,
            xray_pipeline,
            outline_object_buffers: Vec::new(),
            xray_object_buffers: Vec::new(),
            outline_color_texture: None,
            outline_color_view: None,
            outline_depth_texture: None,
            outline_depth_view: None,
            outline_target_size: [0, 0],
            outline_composite_pipeline_single: None,
            outline_composite_pipeline_msaa: None,
            outline_composite_bgl: None,
            outline_composite_bind_group: None,
            outline_composite_sampler: None,
            instance_bind_group_layout: None,
            instance_storage_buf: None,
            instance_storage_capacity: 0,
            instance_bind_groups: std::collections::HashMap::new(),
            solid_instanced_pipeline: None,
            transparent_instanced_pipeline: None,
            shadow_instanced_pipeline: None,
            shadow_instanced_cascade_bufs: [None, None, None, None],
            shadow_instanced_cascade_bgs: [None, None, None, None],
            // Post-processing (all None until ensure_hdr_target is called).
            hdr_texture: None,
            hdr_view: None,
            hdr_depth_texture: None,
            hdr_depth_view: None,
            hdr_depth_only_view: None,
            hdr_size: [0, 0],
            tone_map_pipeline: None,
            tone_map_bgl: None,
            tone_map_bind_group: None,
            tone_map_uniform_buf: None,
            bloom_threshold_texture: None,
            bloom_threshold_view: None,
            bloom_ping_texture: None,
            bloom_ping_view: None,
            bloom_pong_texture: None,
            bloom_pong_view: None,
            bloom_threshold_pipeline: None,
            bloom_blur_pipeline: None,
            bloom_threshold_bg: None,
            bloom_blur_h_bg: None,
            bloom_blur_v_bg: None,
            bloom_blur_h_pong_bg: None,
            bloom_uniform_buf: None,
            bloom_h_uniform_buf: None,
            bloom_v_uniform_buf: None,
            ssao_texture: None,
            ssao_view: None,
            ssao_blur_texture: None,
            ssao_blur_view: None,
            ssao_noise_texture: None,
            ssao_noise_view: None,
            ssao_kernel_buf: None,
            ssao_pipeline: None,
            ssao_blur_pipeline: None,
            ssao_bg: None,
            ssao_blur_bg: None,
            ssao_uniform_buf: None,
            contact_shadow_texture: None,
            contact_shadow_view: None,
            contact_shadow_pipeline: None,
            contact_shadow_bgl: None,
            contact_shadow_bg: None,
            contact_shadow_uniform_buf: None,
            bloom_placeholder_view: None,
            ao_placeholder_view: None,
            cs_placeholder_view: None,
            pp_linear_sampler: None,
            pp_nearest_sampler: None,
            hdr_solid_pipeline: None,
            hdr_solid_two_sided_pipeline: None,
            hdr_transparent_pipeline: None,
            hdr_wireframe_pipeline: None,
            hdr_solid_instanced_pipeline: None,
            hdr_transparent_instanced_pipeline: None,
            hdr_overlay_pipeline: None,
            colormap_textures: Vec::new(),
            colormap_views: Vec::new(),
            colormaps_cpu: Vec::new(),
            fallback_lut_texture,
            fallback_lut_view,
            fallback_scalar_buf,
            builtin_colormap_ids: None,
            colormaps_initialized: false,
            point_cloud_pipeline: None,
            glyph_pipeline: None,
            point_cloud_bgl: None,
            glyph_bgl: None,
            glyph_instance_bgl: None,
            glyph_arrow_mesh: None,
            glyph_sphere_mesh: None,
            glyph_cube_mesh: None,
            volume_textures: Vec::new(),
            volume_pipeline: None,
            volume_bgl: None,
            volume_cube_vb: None,
            volume_cube_ib: None,
            volume_default_opacity_lut: None,
            volume_default_opacity_lut_view: None,
            polyline_pipeline: None,
            polyline_bgl: None,
            streamtube_pipeline: None,
            streamtube_bgl: None,
            streamtube_instance_bgl: None,
            streamtube_cylinder_mesh: None,
            compute_filter_pipeline: None,
            compute_filter_bgl: None,
            oit_accum_texture: None,
            oit_accum_view: None,
            oit_reveal_texture: None,
            oit_reveal_view: None,
            oit_pipeline: None,
            oit_instanced_pipeline: None,
            oit_composite_pipeline: None,
            oit_composite_bgl: None,
            oit_composite_bind_group: None,
            oit_composite_sampler: None,
            oit_size: [0, 0],
            pick_pipeline: None,
            pick_bind_group_layout_1: None,
            pick_camera_bgl: None,
        }
    }
}
