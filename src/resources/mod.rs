mod extra_impls;
mod types;
/// Built-in colormap LUT data.
pub mod colormap_data;
/// Slotted GPU mesh storage with free-list removal.
pub mod mesh_store;

pub use self::types::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColormap, CameraUniform,
    ClipVolumeUniform, ColormapId, GpuMesh, GpuTexture, LightUniform, LightsUniform,
    MeshData, OverlayVertex, PointCloudGpuData, PolylineGpuData, SingleLightUniform, Vertex,
    ViewportGpuResources, VolumeGpuData, VolumeId,
};
pub use self::extra_impls::{ComputeFilterResult, lerp_attributes};
pub(crate) use self::types::{
    BloomUniform, ClipPlanesUniform, ContactShadowUniform, GlyphBaseMesh, GlyphGpuData,
    InstanceData, ObjectUniform, OutlineObjectBuffers, OutlineUniform, OverlayUniform,
    PickInstance, ShadowAtlasUniform, SHADOW_ATLAS_SIZE, SsaoUniform, StreamtubeGpuData,
    ToneMapUniform,
};
use self::extra_impls::{
    build_glyph_arrow, build_glyph_sphere, build_streamtube_cylinder, build_unit_cube,
    generate_edge_indices,
};

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
        // Overlay line pipeline (LineList — for domain wireframe)
        // Uses the same overlay shader + bind group layout as the triangle overlay.
        // No alpha blending needed; domain wireframe is fully opaque white.
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
        // Domain wireframe uniform buffer and bind group
        // Initial state: identity model matrix, white color, full alpha.
        // ------------------------------------------------------------------
        let domain_uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
        };
        let domain_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("domain_uniform_buf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        domain_uniform_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&[domain_uniform_data]));
        domain_uniform_buf.unmap();

        let domain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("domain_bind_group"),
            layout: &overlay_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: domain_uniform_buf.as_entire_binding(),
            }],
        });

        // ------------------------------------------------------------------
        // Grid uniform buffer and bind group
        // ------------------------------------------------------------------
        let grid_uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color: [0.65, 0.65, 0.65, 0.5],
        };
        let grid_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_uniform_buf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        grid_uniform_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&[grid_uniform_data]));
        grid_uniform_buf.unmap();

        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_bind_group"),
            layout: &overlay_bgl,
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
        // StencilState: compare=Always, pass_op=Replace → writes ref(=1) on depth pass.
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
            overlay_bind_group_layout: overlay_bgl,
            domain_vertex_buffer: None,
            domain_index_buffer: None,
            domain_index_count: 0,
            domain_uniform_buf,
            domain_bind_group,
            grid_vertex_buffer: None,
            grid_index_buffer: None,
            grid_index_count: 0,
            grid_uniform_buf,
            grid_bind_group,
            bc_quad_buffers: Vec::new(),
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

    /// Ensure the instanced pipelines and bind group layout are created.
    /// Called lazily when the instanced draw path is first needed.
    pub(crate) fn ensure_instanced_pipelines(&mut self, device: &wgpu::Device) {
        if self.instance_bind_group_layout.is_some() {
            return; // Already initialized.
        }

        // Instanced bind group layout (group 1 for instanced pipelines).
        // binding 0: instance storage buffer
        // binding 1-4: albedo texture, sampler, normal map, AO map
        // Co-located in group 1 to stay within iced's max_bind_groups = 2.
        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_bgl"),
            entries: &[
                // binding 0: instance storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: albedo texture
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
                // binding 2: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: normal map texture
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
                // binding 4: AO map texture
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
            ],
        });

        // Instanced mesh shader.
        let instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_instanced_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh_instanced.wgsl").into()),
        });

        let instanced_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instanced_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &instance_bgl],
            push_constant_ranges: &[],
        });

        // Solid instanced pipeline.
        let solid_instanced = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid_instanced_pipeline"),
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
                    format: self.target_format,
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
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        // Transparent instanced pipeline.
        let transparent_instanced =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("transparent_instanced_pipeline"),
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

        // Shadow instanced pipeline.
        let shadow_instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_instanced_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_instanced.wgsl").into()),
        });

        // Shadow instanced uses the shadow bind group layout (group 0) + instance_bgl (group 1).
        // Re-derive the shadow BGL from the existing shadow_bind_group.
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bgl_for_instanced"),
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

        let shadow_instanced_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_instanced_pipeline_layout"),
                bind_group_layouts: &[&shadow_bgl, &instance_bgl],
                push_constant_ranges: &[],
            });

        let shadow_instanced = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_instanced_pipeline"),
            layout: Some(&shadow_instanced_layout),
            vertex: wgpu::VertexState {
                module: &shadow_instanced_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Allocate 4 per-cascade uniform buffers (64 bytes each = one mat4x4) and
        // create bind groups for shadow_instanced_pipeline group 0.
        // Each cascade has its own small buffer so we can write_buffer(buf, 0, ...) without
        // dynamic offsets (shadow_instanced.wgsl group 0 binds a single uniform, not an array).
        let cascade_bufs: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("shadow_instanced_cascade_buf_{i}")),
                size: 64, // sizeof(mat4x4<f32>)
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let cascade_bgs: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shadow_instanced_cascade_bg_{i}")),
                layout: &shadow_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cascade_bufs[i].as_entire_binding(),
                }],
            })
        });
        self.shadow_instanced_cascade_bufs = cascade_bufs.map(Some);
        self.shadow_instanced_cascade_bgs = cascade_bgs.map(Some);

        self.instance_bind_group_layout = Some(instance_bgl);
        self.solid_instanced_pipeline = Some(solid_instanced);
        self.transparent_instanced_pipeline = Some(transparent_instanced);
        self.shadow_instanced_pipeline = Some(shadow_instanced);
    }

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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/outline_composite.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bloom_threshold.wgsl").into()),
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

    /// Upload instance data to the storage buffer, resizing if needed.
    /// Returns the bind group for the instance storage buffer.
    pub(crate) fn upload_instance_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[InstanceData],
    ) {
        if data.is_empty() {
            return;
        }

        let _bgl = self
            .instance_bind_group_layout
            .as_ref()
            .expect("ensure_instanced_pipelines must be called first");

        // Clamp to the device's max_storage_buffer_binding_size so bind group
        // creation never panics regardless of scene size.
        let max_instances = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<InstanceData>();
        let data = &data[..data.len().min(max_instances)];

        let needed = data.len();
        if needed > self.instance_storage_capacity {
            // Grow with 2x strategy, capped at the device limit.
            let new_cap = (needed * 2).max(64).min(max_instances);
            let buf_size = (new_cap * std::mem::size_of::<InstanceData>()) as u64;
            self.instance_storage_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_storage_buf"),
                size: buf_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_storage_capacity = new_cap;

            // Invalidate all per-texture-key bind groups; they reference the old buffer.
            self.instance_bind_groups.clear();
        }

        queue.write_buffer(
            self.instance_storage_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(data),
        );
    }

    /// Get or create a combined instance+texture bind group for the instanced pipeline.
    ///
    /// The bind group combines the shared instance storage buffer (binding 0) with the
    /// texture views for the given material key (bindings 1-4). Results are cached by key.
    ///
    /// `u64::MAX` in any key component means "use fallback texture for that slot".
    pub(crate) fn get_instance_bind_group(
        &mut self,
        device: &wgpu::Device,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
    ) -> Option<&wgpu::BindGroup> {
        let key = (
            albedo_id.unwrap_or(u64::MAX),
            normal_map_id.unwrap_or(u64::MAX),
            ao_map_id.unwrap_or(u64::MAX),
        );

        if !self.instance_bind_groups.contains_key(&key) {
            let bgl = self.instance_bind_group_layout.as_ref()?;
            let buf = self.instance_storage_buf.as_ref()?;

            let albedo_view = match albedo_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("instance_tex_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                ],
            });
            self.instance_bind_groups.insert(key, bg);
        }

        self.instance_bind_groups.get(&key)
    }

    /// Re-upload the gizmo mesh with updated hover highlight colors.
    ///
    /// Called each frame when the hovered axis changes to brighten the appropriate axis color.
    /// The gizmo mesh is small (~300 vertices), so re-uploading every frame is acceptable.
    pub fn update_gizmo_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mode: crate::interaction::gizmo::GizmoMode,
        hovered: crate::interaction::gizmo::GizmoAxis,
        space_orientation: glam::Quat,
    ) {
        let (verts, indices) = crate::interaction::gizmo::build_gizmo_mesh(mode, hovered, space_orientation);

        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        // Recreate buffers if the new mesh is larger than the current allocation.
        if vert_bytes.len() as u64 > self.gizmo_vertex_buffer.size() {
            self.gizmo_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gizmo_vertex_buf"),
                size: vert_bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if idx_bytes.len() as u64 > self.gizmo_index_buffer.size() {
            self.gizmo_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gizmo_index_buf"),
                size: idx_bytes.len() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(&self.gizmo_vertex_buffer, 0, vert_bytes);
        queue.write_buffer(&self.gizmo_index_buffer, 0, idx_bytes);
        self.gizmo_index_count = indices.len() as u32;
    }

    /// Update the gizmo model matrix uniform (translation to gizmo center + scale for screen size).
    pub fn update_gizmo_uniform(&self, queue: &wgpu::Queue, model: glam::Mat4) {
        let uniform = crate::interaction::gizmo::GizmoUniform {
            model: model.to_cols_array_2d(),
        };
        queue.write_buffer(&self.gizmo_uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Upload domain wireframe mesh (8 corners, 12 edges as LineList — 24 indices).
    ///
    /// Creates or replaces the domain vertex/index buffers. Call when domain extents change.
    /// The wireframe is drawn with the existing `wireframe_pipeline` (LineList topology).
    pub fn upload_domain_wireframe(&mut self, device: &wgpu::Device, nx: f32, ny: f32, nz: f32) {
        use bytemuck::cast_slice;
        use wgpu;

        // 8 corners of the box from (0,0,0) to (nx,ny,nz)
        let corners: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0],
            [nx, 0.0, 0.0],
            [nx, ny, 0.0],
            [0.0, ny, 0.0],
            [0.0, 0.0, nz],
            [nx, 0.0, nz],
            [nx, ny, nz],
            [0.0, ny, nz],
        ];
        // 12 edges as LineList pairs (24 indices total)
        let edge_indices: [u32; 24] = [
            0, 1, 1, 2, 2, 3, 3, 0, // bottom face
            4, 5, 5, 6, 6, 7, 7, 4, // top face
            0, 4, 1, 5, 2, 6, 3, 7, // vertical edges
        ];

        let vertices: Vec<OverlayVertex> = corners
            .iter()
            .map(|p| OverlayVertex { position: *p })
            .collect();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("domain_wireframe_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("domain_wireframe_ibuf"),
            size: (std::mem::size_of::<u32>() * edge_indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&edge_indices));
        index_buffer.unmap();

        self.domain_vertex_buffer = Some(vertex_buffer);
        self.domain_index_buffer = Some(index_buffer);
        self.domain_index_count = edge_indices.len() as u32;
    }

    /// Upload an infinite-style ground-plane grid extending well beyond the domain.
    ///
    /// For 3D: XZ plane at y=0. For 2D: XY plane at z=0.
    /// Grid spacing matches the domain (~20 divisions), but lines extend 5x the
    /// domain extent in every direction so the grid fills the visible space.
    pub fn upload_grid(&mut self, device: &wgpu::Device, nx: f32, ny: f32, nz: f32, is_2d: bool) {
        use bytemuck::cast_slice;
        use wgpu;

        let mut vertices = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Grid spacing based on domain, but lines extend far beyond.
        let divisions = 20usize;
        let extend = 5.0; // how many domain-widths to extend in each direction

        if is_2d {
            // XY grid at z=0
            let spacing_x = nx / divisions as f32;
            let spacing_y = ny / divisions as f32;
            let x_min = -(nx * extend);
            let x_max = nx * (1.0 + extend);
            let y_min = -(ny * extend);
            let y_max = ny * (1.0 + extend);

            // Lines parallel to Y axis (stepping along X)
            let ix_start = (x_min / spacing_x).floor() as i32;
            let ix_end = (x_max / spacing_x).ceil() as i32;
            for i in ix_start..=ix_end {
                let x = i as f32 * spacing_x;
                let idx = vertices.len() as u32;
                vertices.push(OverlayVertex {
                    position: [x, y_min, 0.0],
                });
                vertices.push(OverlayVertex {
                    position: [x, y_max, 0.0],
                });
                indices.push(idx);
                indices.push(idx + 1);
            }
            // Lines parallel to X axis (stepping along Y)
            let iy_start = (y_min / spacing_y).floor() as i32;
            let iy_end = (y_max / spacing_y).ceil() as i32;
            for i in iy_start..=iy_end {
                let y = i as f32 * spacing_y;
                let idx = vertices.len() as u32;
                vertices.push(OverlayVertex {
                    position: [x_min, y, 0.0],
                });
                vertices.push(OverlayVertex {
                    position: [x_max, y, 0.0],
                });
                indices.push(idx);
                indices.push(idx + 1);
            }
        } else {
            // XZ grid at y=0
            let spacing_x = nx / divisions as f32;
            let spacing_z = nz / divisions as f32;
            let x_min = -(nx * extend);
            let x_max = nx * (1.0 + extend);
            let z_min = -(nz * extend);
            let z_max = nz * (1.0 + extend);

            // Lines parallel to Z axis (stepping along X)
            let ix_start = (x_min / spacing_x).floor() as i32;
            let ix_end = (x_max / spacing_x).ceil() as i32;
            for i in ix_start..=ix_end {
                let x = i as f32 * spacing_x;
                let idx = vertices.len() as u32;
                vertices.push(OverlayVertex {
                    position: [x, 0.0, z_min],
                });
                vertices.push(OverlayVertex {
                    position: [x, 0.0, z_max],
                });
                indices.push(idx);
                indices.push(idx + 1);
            }
            // Lines parallel to X axis (stepping along Z)
            let iz_start = (z_min / spacing_z).floor() as i32;
            let iz_end = (z_max / spacing_z).ceil() as i32;
            for i in iz_start..=iz_end {
                let z = i as f32 * spacing_z;
                let idx = vertices.len() as u32;
                vertices.push(OverlayVertex {
                    position: [x_min, 0.0, z],
                });
                vertices.push(OverlayVertex {
                    position: [x_max, 0.0, z],
                });
                indices.push(idx);
                indices.push(idx + 1);
            }
        }

        if vertices.is_empty() {
            self.grid_vertex_buffer = None;
            self.grid_index_buffer = None;
            self.grid_index_count = 0;
            return;
        }

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&indices));
        index_buffer.unmap();

        self.grid_vertex_buffer = Some(vertex_buffer);
        self.grid_index_buffer = Some(index_buffer);
        self.grid_index_count = indices.len() as u32;
    }

    /// Create a quad mesh (2 triangles, 4 vertices) for a BC overlay on a given domain face.
    ///
    /// Create an overlay quad from pre-computed corner positions and color.
    ///
    /// Corners should be in CCW winding order when viewed from outside.
    /// Returns (vertex_buffer, index_buffer, uniform_buffer, bind_group).
    pub fn create_overlay_quad(
        &self,
        device: &wgpu::Device,
        corners: &[[f32; 3]; 4],
        color: [f32; 4],
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
        use bytemuck::cast_slice;
        use wgpu;

        let quad_verts = corners;

        // 2 triangles: [0,1,2] and [0,2,3]
        let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        let vertices: Vec<OverlayVertex> = quad_verts
            .iter()
            .map(|p| OverlayVertex { position: *p })
            .collect();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc_quad_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc_quad_ibuf"),
            size: (std::mem::size_of::<u32>() * quad_indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&quad_indices));
        index_buffer.unmap();

        // Uniform buffer: identity model matrix + given color.
        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc_quad_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[uniform_data]));
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bc_quad_bg"),
            layout: &self.overlay_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        (vertex_buffer, index_buffer, uniform_buffer, bind_group)
    }

    /// Create a line-list overlay for an active transform constraint.
    pub fn create_constraint_overlay(
        &self,
        device: &wgpu::Device,
        overlay: &crate::interaction::snap::ConstraintOverlay,
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    ) {
        use bytemuck::cast_slice;

        let (vertices, color): (Vec<OverlayVertex>, [f32; 4]) = match overlay {
            crate::interaction::snap::ConstraintOverlay::AxisLine {
                origin,
                direction,
                color,
            } => (
                vec![
                    OverlayVertex {
                        position: (*origin - *direction).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin + *direction).to_array(),
                    },
                ],
                *color,
            ),
            crate::interaction::snap::ConstraintOverlay::Plane {
                origin,
                axis_a,
                axis_b,
                color,
            } => (
                vec![
                    OverlayVertex {
                        position: (*origin - *axis_a).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin + *axis_a).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin - *axis_b).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin + *axis_b).to_array(),
                    },
                ],
                *color,
            ),
        };
        let indices: Vec<u32> = (0..vertices.len() as u32).collect();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("constraint_overlay_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("constraint_overlay_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&indices));
        index_buffer.unmap();

        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("constraint_overlay_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[uniform_data]));
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constraint_overlay_bg"),
            layout: &self.overlay_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        (
            vertex_buffer,
            index_buffer,
            indices.len() as u32,
            uniform_buffer,
            bind_group,
        )
    }

    /// Upload cap geometry (cross-section fill) as transient overlay buffers.
    ///
    /// Uses the overlay pipeline (position-only vertices + flat color uniform).
    pub(crate) fn upload_cap_geometry(
        &self,
        device: &wgpu::Device,
        cap: &crate::geometry::cap_geometry::CapMesh,
        color: [f32; 4],
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    ) {
        use bytemuck::cast_slice;

        let vertices: Vec<OverlayVertex> = cap
            .positions
            .iter()
            .map(|p| OverlayVertex { position: *p })
            .collect();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cap_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cap_ibuf"),
            size: (std::mem::size_of::<u32>() * cap.indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&cap.indices));
        index_buffer.unmap();

        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cap_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[uniform_data]));
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cap_bg"),
            layout: &self.overlay_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let idx_count = cap.indices.len() as u32;
        (
            vertex_buffer,
            index_buffer,
            idx_count,
            uniform_buffer,
            bind_group,
        )
    }

    /// Create a GpuMesh from vertex/index slices and register it into the resource list.
    ///
    /// Returns the index of the new mesh in `self.meshes`.
    pub fn upload_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> usize {
        let mesh = Self::create_mesh(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            vertices,
            indices,
        );
        self.mesh_store.insert(mesh).index()
    }

    /// Upload a `MeshData` (from the geometry primitives module) directly.
    ///
    /// Converts positions/normals/indices to the GPU `Vertex` layout (white color)
    /// and creates a normal visualization line buffer (light blue #a0c4ff, length 0.1).
    /// Returns the mesh index.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::EmptyMesh`](crate::error::ViewportError::EmptyMesh) if positions or indices are empty,
    /// [`ViewportError::MeshLengthMismatch`](crate::error::ViewportError::MeshLengthMismatch) if positions and normals differ in length,
    /// or [`ViewportError::InvalidVertexIndex`](crate::error::ViewportError::InvalidVertexIndex) if an index references a nonexistent vertex.
    pub fn upload_mesh_data(
        &mut self,
        device: &wgpu::Device,
        data: &MeshData,
    ) -> crate::error::ViewportResult<usize> {
        Self::validate_mesh_data(data)?;

        // Compute tangents: use provided, auto-compute from UVs, or zero-fill.
        let computed_tangents: Option<Vec<[f32; 4]>> = if data.tangents.is_none() {
            data.uvs.as_ref().map(|uvs| {
                Self::compute_tangents(&data.positions, &data.normals, uvs, &data.indices)
            })
        } else {
            None
        };
        let tangent_slice = data.tangents.as_deref().or(computed_tangents.as_deref());

        let vertices: Vec<Vertex> = data
            .positions
            .iter()
            .zip(data.normals.iter())
            .enumerate()
            .map(|(i, (p, n))| {
                let uv = data
                    .uvs
                    .as_ref()
                    .and_then(|uvs| uvs.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                let tangent = tangent_slice
                    .and_then(|ts| ts.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                Vertex {
                    position: *p,
                    normal: *n,
                    color: [1.0, 1.0, 1.0, 1.0],
                    uv,
                    tangent,
                }
            })
            .collect();

        let normal_line_verts = Self::build_normal_lines(data);

        let mut mesh = Self::create_mesh_with_normals(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            &vertices,
            &data.indices,
            Some(&normal_line_verts),
        );
        mesh.cpu_positions = Some(data.positions.clone());
        mesh.cpu_indices = Some(data.indices.clone());
        let (attr_bufs, attr_ranges) =
            Self::upload_attributes(device, &data.attributes, &data.positions, &data.indices);
        mesh.attribute_buffers = attr_bufs;
        mesh.attribute_ranges = attr_ranges;
        let id = self.mesh_store.insert(mesh);
        tracing::debug!(
            mesh_index = id.index(),
            vertices = data.positions.len(),
            indices = data.indices.len(),
            "mesh uploaded"
        );
        Ok(id.index())
    }

    /// Replace the mesh at `mesh_index` with new geometry data.
    ///
    /// Used when primitive parameters change (re-tessellation).
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::MeshIndexOutOfBounds`](crate::error::ViewportError::MeshIndexOutOfBounds) if `mesh_index` is out of range,
    /// or any mesh validation error from the new data.
    pub fn replace_mesh_data(
        &mut self,
        device: &wgpu::Device,
        mesh_index: usize,
        data: &MeshData,
    ) -> crate::error::ViewportResult<()> {
        let mesh_id = crate::resources::mesh_store::MeshId(mesh_index);
        if !self.mesh_store.contains(mesh_id) {
            return Err(crate::error::ViewportError::MeshIndexOutOfBounds {
                index: mesh_index,
                count: self.mesh_store.len(),
            });
        }
        Self::validate_mesh_data(data)?;

        // Compute tangents: use provided, auto-compute from UVs, or zero-fill.
        let computed_tangents: Option<Vec<[f32; 4]>> = if data.tangents.is_none() {
            data.uvs.as_ref().map(|uvs| {
                Self::compute_tangents(&data.positions, &data.normals, uvs, &data.indices)
            })
        } else {
            None
        };
        let tangent_slice = data.tangents.as_deref().or(computed_tangents.as_deref());

        let vertices: Vec<Vertex> = data
            .positions
            .iter()
            .zip(data.normals.iter())
            .enumerate()
            .map(|(i, (p, n))| {
                let uv = data
                    .uvs
                    .as_ref()
                    .and_then(|uvs| uvs.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                let tangent = tangent_slice
                    .and_then(|ts| ts.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                Vertex {
                    position: *p,
                    normal: *n,
                    color: [1.0, 1.0, 1.0, 1.0],
                    uv,
                    tangent,
                }
            })
            .collect();
        let normal_line_verts = Self::build_normal_lines(data);
        let mut new_mesh = Self::create_mesh_with_normals(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            &vertices,
            &data.indices,
            Some(&normal_line_verts),
        );
        new_mesh.cpu_positions = Some(data.positions.clone());
        new_mesh.cpu_indices = Some(data.indices.clone());
        let (attr_bufs, attr_ranges) =
            Self::upload_attributes(device, &data.attributes, &data.positions, &data.indices);
        new_mesh.attribute_buffers = attr_bufs;
        new_mesh.attribute_ranges = attr_ranges;
        // replace() cannot fail here — we checked contains() above.
        let _ = self.mesh_store.replace(mesh_id, new_mesh);
        tracing::debug!(
            mesh_index,
            vertices = data.positions.len(),
            indices = data.indices.len(),
            "mesh replaced"
        );
        Ok(())
    }

    /// Get a reference to the mesh at the given index, or `None` if the slot is empty/invalid.
    pub fn mesh(&self, index: usize) -> Option<&GpuMesh> {
        self.mesh_store.get(crate::resources::mesh_store::MeshId(index))
    }

    /// Total number of mesh slots (including empty/removed slots).
    pub fn mesh_slot_count(&self) -> usize {
        self.mesh_store.slot_count()
    }

    /// Remove a mesh, dropping its GPU buffers and freeing its slot for reuse.
    ///
    /// Returns `true` if a mesh was removed, `false` if the slot was already empty.
    pub fn remove_mesh(&mut self, index: usize) -> bool {
        self.mesh_store.remove(crate::resources::mesh_store::MeshId(index))
    }

    /// Upload an RGBA texture to the GPU and return its texture ID.
    ///
    /// The ID can be stored in `Material::texture_id` to apply the texture to objects.
    /// `rgba_data` must be exactly `width * height * 4` bytes in RGBA8 format.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData) if the data length is incorrect.
    pub fn upload_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> crate::error::ViewportResult<u64> {
        let expected = (width * height * 4) as usize;
        if rgba_data.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba_data.len(),
            });
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("user_texture"),
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("user_texture_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_texture_bg"),
            layout: &self.texture_bind_group_layout,
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
                    resource: wgpu::BindingResource::TextureView(&self.fallback_normal_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_ao_map_view),
                },
            ],
        });

        let id = self.textures.len() as u64;
        self.textures.push(GpuTexture {
            texture,
            view,
            sampler,
            bind_group,
        });
        tracing::debug!(texture_id = id, width, height, "texture uploaded");
        Ok(id)
    }

    /// Upload an RGBA texture as a normal map and return its texture ID.
    ///
    /// Uses Rgba8Unorm format (not sRGB) so values are linear — required for correct
    /// normal map decoding. `rgba_data` must be `width * height * 4` bytes.
    pub fn upload_normal_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> crate::error::ViewportResult<u64> {
        let expected = (width * height * 4) as usize;
        if rgba_data.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba_data.len(),
            });
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normal_map_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm, // Linear, not sRGB
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("normal_map_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal_map_bg"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_ao_map_view),
                },
            ],
        });

        let id = self.textures.len() as u64;
        self.textures.push(GpuTexture {
            texture,
            view,
            sampler,
            bind_group,
        });
        tracing::debug!(texture_id = id, width, height, "normal map uploaded");
        Ok(id)
    }

    /// Get or create a cached material bind group for (albedo, normal_map, ao_map) texture combo.
    ///
    /// `u64::MAX` sentinel means "use fallback texture for that slot".
    /// The bind group is cached in `material_bind_groups` keyed by the 3-tuple.
    #[allow(dead_code)]
    pub(crate) fn get_material_bind_group(
        &mut self,
        device: &wgpu::Device,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
    ) -> &wgpu::BindGroup {
        let key = (
            albedo_id.unwrap_or(u64::MAX),
            normal_map_id.unwrap_or(u64::MAX),
            ao_map_id.unwrap_or(u64::MAX),
        );

        if !self.material_bind_groups.contains_key(&key) {
            let albedo_view = match albedo_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_bg"),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                ],
            });
            self.material_bind_groups.insert(key, bg);
        }

        self.material_bind_groups.get(&key).unwrap()
    }

    /// Rebuild `mesh.object_bind_group` so it includes the texture views, LUT, and scalar
    /// buffer for the given material + attribute key. Called from `prepare()` when
    /// `mesh.last_tex_key` differs from the current frame's material/attribute state.
    ///
    /// The bind group layout is `object_bgl`:
    ///   binding 0 → object uniform buffer
    ///   binding 1 → albedo texture view
    ///   binding 2 → material sampler (also used for LUT sampling)
    ///   binding 3 → normal map view
    ///   binding 4 → AO map view
    ///   binding 5 → LUT (colormap) texture view
    ///   binding 6 → scalar attribute storage buffer
    pub(crate) fn update_mesh_texture_bind_group(
        &mut self,
        device: &wgpu::Device,
        mesh_index: usize,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
        lut_id: Option<ColormapId>,
        active_attr: Option<&str>,
    ) {
        let attr_hash = active_attr
            .map(|name| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                name.hash(&mut h);
                h.finish()
            })
            .unwrap_or(u64::MAX);

        let key = (
            albedo_id.unwrap_or(u64::MAX),
            normal_map_id.unwrap_or(u64::MAX),
            ao_map_id.unwrap_or(u64::MAX),
            lut_id.map(|id| id.0 as u64).unwrap_or(u64::MAX),
            attr_hash,
        );

        {
            let Some(mesh) = self.mesh_store.get(crate::resources::mesh_store::MeshId(mesh_index)) else {
                return;
            };
            if mesh.last_tex_key == key {
                return; // Already up to date.
            }
        }

        let albedo_view = match albedo_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_texture.view,
        };
        let normal_view = match normal_map_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_normal_map_view,
        };
        let ao_view = match ao_map_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_ao_map_view,
        };
        let lut_view = match lut_id {
            Some(id) if id.0 < self.colormap_views.len() => &self.colormap_views[id.0],
            _ => &self.fallback_lut_view,
        };

        // Re-borrow mutably to update.
        let Some(mesh) = self
            .mesh_store
            .get_mut(crate::resources::mesh_store::MeshId(mesh_index))
        else {
            return;
        };

        // Resolve the scalar storage buffer: from the mesh's attribute map, or fallback.
        // SAFETY: fallback_scalar_buf is a separate field from mesh_store; borrow is valid.
        let scalar_buf: &wgpu::Buffer = match active_attr {
            Some(name) => mesh
                .attribute_buffers
                .get(name)
                .unwrap_or(&self.fallback_scalar_buf),
            None => &self.fallback_scalar_buf,
        };

        mesh.object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object_bind_group"),
            layout: &self.object_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh.object_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scalar_buf.as_entire_binding(),
                },
            ],
        });
        mesh.last_tex_key = key;
    }

    // -----------------------------------------------------------------------
    // Colormap / LUT API
    // -----------------------------------------------------------------------

    /// Upload a 256-sample RGBA colormap to the GPU and return its `ColormapId`.
    ///
    /// The returned ID can be stored in `SceneRenderItem::colormap_id`.
    /// Use `BuiltinColormap` variants + [`Self::builtin_colormap_id`] for the built-in presets.
    pub fn upload_colormap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba_data: &[[u8; 4]; 256],
    ) -> ColormapId {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lut_texture"),
            size: wgpu::Extent3d {
                width: 256,
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
        // Upload pixel data directly.
        // Flatten [[u8;4]; 256] → &[u8] (256 * 4 = 1024 bytes).
        let flat: Vec<u8> = rgba_data.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &flat,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = ColormapId(self.colormap_textures.len());
        self.colormap_textures.push(texture);
        self.colormap_views.push(view);
        self.colormaps_cpu.push(*rgba_data);
        id
    }

    /// Return the CPU-side colormap data for `id`, or `None` if the id is invalid.
    ///
    /// Use this to draw an egui scalar bar gradient strip via `egui::Painter::image`.
    pub fn get_colormap_rgba(&self, id: ColormapId) -> Option<&[[u8; 4]; 256]> {
        self.colormaps_cpu.get(id.0)
    }

    /// Return the `ColormapId` for a built-in preset.
    ///
    /// Call [`Self::ensure_colormaps_initialized`] first (done automatically by
    /// `ViewportRenderer::prepare`).  Panics if colormaps have not been initialized yet.
    pub fn builtin_colormap_id(&self, preset: BuiltinColormap) -> ColormapId {
        self.builtin_colormap_ids
            .expect("call ensure_colormaps_initialized before using built-in colormaps")
            [preset as usize]
    }

    /// Ensure built-in colormaps are uploaded to the GPU.
    ///
    /// Called automatically by `ViewportRenderer::prepare()` on the first frame.
    /// Safe to call multiple times — no-op after first invocation.
    pub fn ensure_colormaps_initialized(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.colormaps_initialized {
            return;
        }
        let viridis = self.upload_colormap(device, queue, &crate::resources::colormap_data::viridis_rgba());
        let plasma = self.upload_colormap(device, queue, &crate::resources::colormap_data::plasma_rgba());
        let greyscale =
            self.upload_colormap(device, queue, &crate::resources::colormap_data::greyscale_rgba());
        let coolwarm = self.upload_colormap(device, queue, &crate::resources::colormap_data::coolwarm_rgba());
        let rainbow = self.upload_colormap(device, queue, &crate::resources::colormap_data::rainbow_rgba());
        self.builtin_colormap_ids = Some([viridis, plasma, greyscale, coolwarm, rainbow]);
        self.colormaps_initialized = true;
    }

    // -----------------------------------------------------------------------
    // SciVis Phase B — pipeline creation and per-frame upload helpers
    // -----------------------------------------------------------------------

    /// Lazily create the point cloud render pipeline (PointList topology).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.point_clouds` is non-empty.
    pub(crate) fn ensure_point_cloud_pipeline(&mut self, device: &wgpu::Device) {
        if self.point_cloud_pipeline.is_some() {
            return;
        }

        // ---- bind group layout for group 1 ----
        // binding 0: PointCloudUniform (uniform buffer)
        // binding 1: LUT texture (2D, filterable float)
        // binding 2: LUT sampler (filtering)
        // binding 3: scalar storage buffer (read-only f32 array)
        // binding 4: color storage buffer  (read-only vec4 array)
        let pc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_bgl"),
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
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_cloud_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/point_cloud.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point_cloud_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &pc_bgl],
            push_constant_ranges: &[],
        });

        // Point cloud vertex layout: position only (12 bytes = vec3<f32>), per instance.
        // Six vertices are drawn per instance (billboard quad = 2 triangles).
        // vertex_index (0-5) selects the quad corner; instance_index is the point index.
        let pc_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_cloud_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[pc_vertex_layout],
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
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.point_cloud_bgl = Some(pc_bgl);
        self.point_cloud_pipeline = Some(pipeline);
    }

    /// Upload one [`PointCloudItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.point_clouds`.
    pub(crate) fn upload_point_cloud(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::PointCloudItem,
    ) -> PointCloudGpuData {
        let point_count = item.positions.len() as u32;

        // ---- vertex buffer: positions (12 bytes per point) ----
        let pos_bytes: Vec<u8> = item
            .positions
            .iter()
            .flat_map(|p| bytemuck::bytes_of(p).iter().copied())
            .collect();
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_vertex_buf"),
            size: pos_bytes.len().max(12) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &pos_bytes);

        // ---- scalar storage buffer ----
        let (scalar_buf, has_scalars, scalar_min, scalar_max) = if !item.scalars.is_empty() {
            let min = item
                .scalar_range
                .map(|r| r.0)
                .unwrap_or_else(|| item.scalars.iter().cloned().fold(f32::INFINITY, f32::min));
            let max = item.scalar_range.map(|r| r.1).unwrap_or_else(|| {
                item.scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            });
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_scalar_buf"),
                size: (std::mem::size_of::<f32>() * item.scalars.len()).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.scalars));
            (buf, 1u32, min, max)
        } else {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_scalar_buf_fallback"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32, 0.0f32, 1.0f32)
        };

        // ---- color storage buffer ----
        let (color_buf, has_colors) = if !item.colors.is_empty() && has_scalars == 0 {
            let bytes: &[u8] = bytemuck::cast_slice(&item.colors);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_color_buf"),
                size: bytes.len().max(16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytes);
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_color_buf_fallback"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        // ---- uniform buffer: PointCloudUniform (112 bytes) ----
        // Layout: model(64) + default_color(16) + point_size(4) + has_scalars(4) +
        //         scalar_min(4) + scalar_max(4) + has_colors(4) + pad(12)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PointCloudUniform {
            model: [[f32; 4]; 4],
            default_color: [f32; 4],
            point_size: f32,
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            has_colors: u32,
            _pad: [u32; 3],
        }
        let uniform_data = PointCloudUniform {
            model: item.model,
            default_color: item.default_color,
            point_size: item.point_size,
            has_scalars,
            scalar_min,
            scalar_max,
            has_colors,
            _pad: [0; 3],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_uniform_buf"),
            size: std::mem::size_of::<PointCloudUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // ---- select LUT view ----
        let lut_view = self
            .builtin_colormap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colormap_id
                    .unwrap_or(ids[crate::resources::BuiltinColormap::Viridis as usize]);
                self.colormap_views.get(preset_id.0)
            })
            .unwrap_or(&self.fallback_lut_view);

        // ---- LUT sampler: reuse the material sampler (linear, clamp) ----
        let lut_sampler = &self.material_sampler;

        // ---- bind group ----
        let bgl = self
            .point_cloud_bgl
            .as_ref()
            .expect("ensure_point_cloud_pipeline not called");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pc_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scalar_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: color_buf.as_entire_binding(),
                },
            ],
        });

        PointCloudGpuData {
            vertex_buffer,
            point_count,
            bind_group,
            _uniform_buf: uniform_buf,
            _scalar_buf: scalar_buf,
            _color_buf: color_buf,
        }
    }

    // -----------------------------------------------------------------------
    // SciVis Phase M8 — polyline pipeline creation and per-frame upload helpers
    // -----------------------------------------------------------------------

    /// Lazily create the polyline render pipeline (LineStrip topology).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.polylines` is non-empty.
    pub(crate) fn ensure_polyline_pipeline(&mut self, device: &wgpu::Device) {
        if self.polyline_pipeline.is_some() {
            return;
        }

        // ---- bind group layout for group 1 ----
        // binding 0: PolylineUniform (uniform buffer)
        // binding 1: LUT texture (2D, filterable float)
        // binding 2: LUT sampler (filtering)
        let pl_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("polyline_bgl"),
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
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polyline_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/polyline.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("polyline_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &pl_bgl],
            push_constant_ranges: &[],
        });

        // Polyline vertex layout: [position: vec3f, scalar: f32] = 16 bytes per vertex.
        let pl_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("polyline_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[pl_vertex_layout],
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
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
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
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.polyline_bgl = Some(pl_bgl);
        self.polyline_pipeline = Some(pipeline);
    }

    /// Upload one [`PolylineItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.polylines`.
    pub(crate) fn upload_polyline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::PolylineItem,
    ) -> PolylineGpuData {
        let vertex_count = item.positions.len() as u32;

        // ---- interleaved vertex buffer: [x, y, z, scalar] (16 bytes per vertex) ----
        let mut raw_verts: Vec<f32> = Vec::with_capacity(item.positions.len() * 4);
        for (i, pos) in item.positions.iter().enumerate() {
            raw_verts.push(pos[0]);
            raw_verts.push(pos[1]);
            raw_verts.push(pos[2]);
            raw_verts.push(item.scalars.get(i).copied().unwrap_or(0.0));
        }
        let vert_bytes: &[u8] = bytemuck::cast_slice(&raw_verts);
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polyline_vertex_buf"),
            size: vert_bytes.len().max(16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, vert_bytes);

        // ---- compute per-strip vertex ranges from strip_lengths ----
        let mut strip_ranges: Vec<std::ops::Range<u32>> = Vec::new();
        let mut offset: u32 = 0;
        for &len in &item.strip_lengths {
            if len >= 2 {
                strip_ranges.push(offset..offset + len);
            }
            offset += len;
        }
        // Fallback: if no strip_lengths provided but positions exist, treat as one strip.
        if item.strip_lengths.is_empty() && vertex_count >= 2 {
            strip_ranges.push(0..vertex_count);
        }

        // ---- scalar range ----
        let (has_scalar, scalar_min, scalar_max) = if !item.scalars.is_empty() {
            let (min, max) = item.scalar_range.unwrap_or_else(|| {
                let mn = item.scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = item
                    .scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                (mn, mx)
            });
            (1u32, min, max)
        } else {
            (0u32, 0.0f32, 1.0f32)
        };

        // ---- uniform buffer: PolylineUniform (48 bytes) ----
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PolylineUniform {
            default_color: [f32; 4],
            line_width: f32,
            scalar_min: f32,
            scalar_max: f32,
            has_scalar: u32,
            _pad: [f32; 4],
        }
        let uniform_data = PolylineUniform {
            default_color: item.default_color,
            line_width: item.line_width,
            scalar_min,
            scalar_max,
            has_scalar,
            _pad: [0.0; 4],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polyline_uniform_buf"),
            size: std::mem::size_of::<PolylineUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // ---- select LUT view ----
        let lut_view = self
            .builtin_colormap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colormap_id
                    .unwrap_or(ids[crate::resources::BuiltinColormap::Viridis as usize]);
                self.colormap_views.get(preset_id.0)
            })
            .unwrap_or(&self.fallback_lut_view);

        let lut_sampler = &self.material_sampler;

        // ---- bind group ----
        let bgl = self
            .polyline_bgl
            .as_ref()
            .expect("ensure_polyline_pipeline not called");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("polyline_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        PolylineGpuData {
            vertex_buffer,
            vertex_count,
            strip_ranges,
            bind_group,
            _uniform_buf: uniform_buf,
        }
    }

    /// Lazily create the glyph render pipeline (instanced TriangleList).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.glyphs` is non-empty.
    pub(crate) fn ensure_glyph_pipeline(&mut self, device: &wgpu::Device) {
        if self.glyph_pipeline.is_some() {
            return;
        }

        // ---- bind group layout for group 1: glyph uniform + LUT ----
        let glyph_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glyph_bgl"),
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
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ---- bind group layout for group 2: instance storage buffer ----
        let glyph_instance_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("glyph_instance_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glyph_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/glyph.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glyph_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &glyph_bgl,
                &glyph_instance_bgl,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glyph_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                // Glyph base mesh uses the full Vertex layout.
                buffers: &[Vertex::buffer_layout()],
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
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.glyph_bgl = Some(glyph_bgl);
        self.glyph_instance_bgl = Some(glyph_instance_bgl);
        self.glyph_pipeline = Some(pipeline);
    }

    /// Upload one [`GlyphItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.glyphs`.
    /// The glyph base mesh is cached in `glyph_arrow_mesh` / `glyph_sphere_mesh` / `glyph_cube_mesh`.
    pub(crate) fn upload_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::GlyphItem,
    ) -> GlyphGpuData {
        let instance_count = item.positions.len() as u32;

        // ---- ensure base mesh is cached ----
        self.ensure_glyph_mesh(device, item.glyph_type);

        // Obtain raw pointers to the cached buffers.
        // Safety: `ViewportGpuResources` owns these and they live at least as long as
        // this method's returned `GlyphGpuData`.  We extend the lifetime to `'static`
        // here so we can store it in the struct; the caller (ViewportRenderer) always
        // drops the Vec<GlyphGpuData> at the start of the next prepare(), before any
        // resize/invalidation of the mesh caches.
        let (mesh_vbuf, mesh_ibuf, mesh_idx_count) = {
            let mesh = match item.glyph_type {
                crate::renderer::GlyphType::Arrow => self.glyph_arrow_mesh.as_ref(),
                crate::renderer::GlyphType::Sphere => self.glyph_sphere_mesh.as_ref(),
                crate::renderer::GlyphType::Cube => self.glyph_cube_mesh.as_ref(),
            }
            .expect("glyph mesh should have been created by ensure_glyph_mesh");

            let vbuf: &'static wgpu::Buffer = unsafe { &*(&mesh.vertex_buffer as *const _) };
            let ibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.index_buffer as *const _) };
            (vbuf, ibuf, mesh.index_count)
        };

        // ---- compute scalar range and magnitude range ----
        let mags: Vec<f32> = item
            .vectors
            .iter()
            .map(|v| {
                let gv = glam::Vec3::from(*v);
                gv.length()
            })
            .collect();

        let (scalar_min, scalar_max) = if !item.scalars.is_empty() {
            item.scalar_range.unwrap_or_else(|| {
                let min = item.scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = item
                    .scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
        } else {
            // Color by magnitude.
            item.scalar_range.unwrap_or_else(|| {
                let min = mags.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = mags.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
        };

        let (mag_clamp_min, mag_clamp_max, has_mag_clamp) = item
            .magnitude_clamp
            .map(|(mn, mx)| (mn, mx, 1u32))
            .unwrap_or((0.0, 1.0, 0u32));

        // ---- build instance storage buffer: 32 bytes per instance ----
        // GlyphInstance { position: vec3, _pad: f32, direction: vec3, scalar: f32 }
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GlyphInstance {
            position: [f32; 3],
            _pad0: f32,
            direction: [f32; 3],
            scalar: f32,
        }

        let instances: Vec<GlyphInstance> = (0..item.positions.len())
            .map(|i| GlyphInstance {
                position: item.positions[i],
                _pad0: 0.0,
                direction: item.vectors.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                scalar: item
                    .scalars
                    .get(i)
                    .copied()
                    .unwrap_or(mags.get(i).copied().unwrap_or(0.0)),
            })
            .collect();

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_instance_buf"),
            size: (std::mem::size_of::<GlyphInstance>() * instances.len()).max(32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

        // ---- glyph uniform buffer ----
        // Layout: global_scale(4) + scale_by_magnitude(4) + has_scalars(4) + scalar_min(4) +
        //         scalar_max(4) + mag_clamp_min(4) + mag_clamp_max(4) + has_mag_clamp(4) +
        //         3 * vec4 padding(48) = 80 bytes
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GlyphUniform {
            global_scale: f32,
            scale_by_magnitude: u32,
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            mag_clamp_min: f32,
            mag_clamp_max: f32,
            has_mag_clamp: u32,
            _pad: [[f32; 4]; 3],
        }
        let uniform_data = GlyphUniform {
            global_scale: item.scale,
            scale_by_magnitude: if item.scale_by_magnitude { 1 } else { 0 },
            has_scalars: if !item.scalars.is_empty() { 1 } else { 0 },
            scalar_min,
            scalar_max,
            mag_clamp_min,
            mag_clamp_max,
            has_mag_clamp,
            _pad: [[0.0; 4]; 3],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_uniform_buf"),
            size: std::mem::size_of::<GlyphUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // ---- select LUT view ----
        let lut_view = self
            .builtin_colormap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colormap_id
                    .unwrap_or(ids[crate::resources::BuiltinColormap::Viridis as usize]);
                self.colormap_views.get(preset_id.0)
            })
            .unwrap_or(&self.fallback_lut_view);

        let lut_sampler = &self.material_sampler;

        // ---- group 1 bind group: uniform + LUT ----
        let bgl1 = self
            .glyph_bgl
            .as_ref()
            .expect("ensure_glyph_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glyph_uniform_bg"),
            layout: bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        // ---- group 2 bind group: instance storage buffer ----
        let bgl2 = self
            .glyph_instance_bgl
            .as_ref()
            .expect("ensure_glyph_pipeline not called");
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glyph_instance_bg"),
            layout: bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        GlyphGpuData {
            mesh_vertex_buffer: mesh_vbuf,
            mesh_index_buffer: mesh_ibuf,
            mesh_index_count: mesh_idx_count,
            instance_count,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Ensure a glyph base mesh is cached for the given [`GlyphType`].
    /// Creates and uploads the mesh on first call for that type.
    fn ensure_glyph_mesh(&mut self, device: &wgpu::Device, glyph_type: crate::renderer::GlyphType) {
        use crate::renderer::GlyphType;

        let already_cached = match glyph_type {
            GlyphType::Arrow => self.glyph_arrow_mesh.is_some(),
            GlyphType::Sphere => self.glyph_sphere_mesh.is_some(),
            GlyphType::Cube => self.glyph_cube_mesh.is_some(),
        };
        if already_cached {
            return;
        }

        let (verts, indices) = match glyph_type {
            GlyphType::Arrow => build_glyph_arrow(),
            GlyphType::Sphere => build_glyph_sphere(),
            GlyphType::Cube => build_unit_cube(),
        };

        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_mesh_vbuf"),
            size: (std::mem::size_of::<Vertex>() * verts.len()).max(64) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vbuf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&verts));
        vbuf.unmap();

        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_mesh_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()).max(12) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        ibuf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&indices));
        ibuf.unmap();

        let mesh = GlyphBaseMesh {
            vertex_buffer: vbuf,
            index_buffer: ibuf,
            index_count: indices.len() as u32,
        };

        match glyph_type {
            GlyphType::Arrow => self.glyph_arrow_mesh = Some(mesh),
            GlyphType::Sphere => self.glyph_sphere_mesh = Some(mesh),
            GlyphType::Cube => self.glyph_cube_mesh = Some(mesh),
        }
    }

    // -----------------------------------------------------------------------
    // SciVis Phase M — streamtube pipeline + upload helpers
    // -----------------------------------------------------------------------

    /// Lazily create the streamtube render pipeline (instanced cylinder TriangleList).
    ///
    /// No-op if already created.  Called from `prepare()` when `frame.streamtube_items`
    /// is non-empty.
    pub(crate) fn ensure_streamtube_pipeline(&mut self, device: &wgpu::Device) {
        if self.streamtube_pipeline.is_some() {
            return;
        }

        // ---- bind group layout 1: tube uniform (color + radius) ----
        let streamtube_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("streamtube_bgl"),
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

        // ---- bind group layout 2: per-instance storage buffer ----
        let streamtube_instance_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("streamtube_instance_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("streamtube_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/streamtube.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("streamtube_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &streamtube_bgl,
                &streamtube_instance_bgl,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("streamtube_pipeline"),
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
                    format: self.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.streamtube_bgl = Some(streamtube_bgl);
        self.streamtube_instance_bgl = Some(streamtube_instance_bgl);
        self.streamtube_pipeline = Some(pipeline);
    }

    /// Upload one [`StreamtubeItem`] to the GPU and return draw data.
    ///
    /// Converts each consecutive point pair in each strip into one cylinder instance.
    /// The cylinder base mesh is cached in `streamtube_cylinder_mesh` and created on
    /// first call.  Returns a [`StreamtubeGpuData`] with `instance_count = 0` when
    /// the item has no renderable segments.
    pub(crate) fn upload_streamtube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::StreamtubeItem,
    ) -> StreamtubeGpuData {
        // ---- ensure cylinder mesh is cached ----
        if self.streamtube_cylinder_mesh.is_none() {
            let (verts, indices) = build_streamtube_cylinder();

            let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("streamtube_cylinder_vbuf"),
                size: (std::mem::size_of::<Vertex>() * verts.len()).max(64) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            vbuf.slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&verts));
            vbuf.unmap();

            let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("streamtube_cylinder_ibuf"),
                size: (std::mem::size_of::<u32>() * indices.len()).max(12) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            ibuf.slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&indices));
            ibuf.unmap();

            self.streamtube_cylinder_mesh = Some(GlyphBaseMesh {
                vertex_buffer: vbuf,
                index_buffer: ibuf,
                index_count: indices.len() as u32,
            });
        }

        // ---- obtain raw mesh pointers (same lifetime extension as glyph path) ----
        let (mesh_vbuf, mesh_ibuf, mesh_idx_count) = {
            let mesh = self
                .streamtube_cylinder_mesh
                .as_ref()
                .expect("streamtube cylinder mesh created above");
            let vbuf: &'static wgpu::Buffer = unsafe { &*(&mesh.vertex_buffer as *const _) };
            let ibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.index_buffer as *const _) };
            (vbuf, ibuf, mesh.index_count)
        };

        // ---- build per-instance data: one cylinder per consecutive point pair ----
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StreamtubeInstance {
            position: [f32; 3],  // segment midpoint
            half_len: f32,       // half segment length
            direction: [f32; 3], // normalized direction
            _pad: f32,
        }

        let mut instances: Vec<StreamtubeInstance> = Vec::new();

        let positions = &item.positions;
        let mut strip_start = 0usize;
        for &len in &item.strip_lengths {
            let len = len as usize;
            let strip_end = (strip_start + len).min(positions.len());
            // Each consecutive pair of points in the strip → one cylinder.
            for i in strip_start..strip_end.saturating_sub(1) {
                let a = glam::Vec3::from(positions[i]);
                let b = glam::Vec3::from(positions[i + 1]);
                let seg = b - a;
                let seg_len = seg.length();
                if seg_len < f32::EPSILON {
                    continue; // degenerate segment — skip
                }
                instances.push(StreamtubeInstance {
                    position: ((a + b) * 0.5).to_array(),
                    half_len: seg_len * 0.5,
                    direction: (seg / seg_len).to_array(),
                    _pad: 0.0,
                });
            }
            strip_start += len;
        }

        let instance_count = instances.len() as u32;

        // Allocate at least 32 bytes to avoid zero-size buffer (wgpu requirement).
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streamtube_instance_buf"),
            size: (std::mem::size_of::<StreamtubeInstance>() * instances.len().max(1)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !instances.is_empty() {
            queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));
        }

        // ---- tube uniform: color + radius (32 bytes) ----
        // WGSL struct layout: color (vec4, 16B) + radius (f32, 4B) + implicit 12B gap
        // (vec3 align=16, so _pad starts at offset 32) + vec3 (12B) = 48B total.
        // Rust must match: 16 + 4 + 28 = 48.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StreamtubeUniform {
            color: [f32; 4],
            radius: f32,
            _pad: [f32; 7],
        }
        let uniform_data = StreamtubeUniform {
            color: item.color,
            radius: item.radius.max(f32::EPSILON),
            _pad: [0.0; 7],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streamtube_uniform_buf"),
            size: std::mem::size_of::<StreamtubeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // ---- group 1 bind group: tube uniform ----
        let bgl1 = self
            .streamtube_bgl
            .as_ref()
            .expect("ensure_streamtube_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("streamtube_uniform_bg"),
            layout: bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ---- group 2 bind group: instance storage ----
        let bgl2 = self
            .streamtube_instance_bgl
            .as_ref()
            .expect("ensure_streamtube_pipeline not called");
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("streamtube_instance_bg"),
            layout: bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        StreamtubeGpuData {
            mesh_vertex_buffer: mesh_vbuf,
            mesh_index_buffer: mesh_ibuf,
            mesh_index_count: mesh_idx_count,
            instance_count,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    // -----------------------------------------------------------------------
    // Scalar attribute helpers
    // -----------------------------------------------------------------------

    /// Upload per-vertex and per-cell scalar attributes to GPU storage buffers.
    ///
    /// Returns `(attribute_buffers, attribute_ranges)` — maps from attribute name to GPU buffer
    /// and to the (min, max) scalar range computed at upload time.
    fn upload_attributes(
        device: &wgpu::Device,
        attributes: &std::collections::HashMap<String, AttributeData>,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> (
        std::collections::HashMap<String, wgpu::Buffer>,
        std::collections::HashMap<String, (f32, f32)>,
    ) {
        let mut bufs = std::collections::HashMap::new();
        let mut ranges = std::collections::HashMap::new();
        for (name, attr_data) in attributes {
            let scalars: Vec<f32> = match attr_data {
                AttributeData::Vertex(v) => v.clone(),
                AttributeData::Cell(c) => Self::expand_cell_to_vertex(c, positions, indices),
            };
            if scalars.is_empty() {
                continue;
            }
            let min = scalars.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("attr_{name}")),
                size: (std::mem::size_of::<f32>() * scalars.len()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            {
                let mut view = buf.slice(..).get_mapped_range_mut();
                view.copy_from_slice(bytemuck::cast_slice(&scalars));
            }
            buf.unmap();
            bufs.insert(name.clone(), buf);
            ranges.insert(name.clone(), (min, max));
        }
        (bufs, ranges)
    }

    /// Expand per-cell (per-triangle) scalar values to per-vertex by averaging contributions.
    fn expand_cell_to_vertex(
        cell_values: &[f32],
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Vec<f32> {
        let n = positions.len();
        let mut sum = vec![0.0f32; n];
        let mut count = vec![0u32; n];
        for (tri_idx, chunk) in indices.chunks(3).enumerate() {
            let v = cell_values.get(tri_idx).copied().unwrap_or(0.0);
            for &vi in chunk {
                let vi = vi as usize;
                if vi < n {
                    sum[vi] += v;
                    count[vi] += 1;
                }
            }
        }
        (0..n)
            .map(|i| {
                if count[i] > 0 {
                    sum[i] / count[i] as f32
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compute per-vertex tangents using Gram-Schmidt orthogonalization with handedness.
    ///
    /// Returns a `Vec<[f32; 4]>` of length `positions.len()` where each element is
    /// `[tx, ty, tz, w]` with `w = ±1.0` encoding bitangent handedness.
    ///
    /// Requires triangulated indices (every 3 indices = one triangle).
    /// If any triangle is degenerate (zero-area or zero UV area), its contribution is skipped.
    fn compute_tangents(
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        uvs: &[[f32; 2]],
        indices: &[u32],
    ) -> Vec<[f32; 4]> {
        let n = positions.len();
        let mut tan1 = vec![[0.0f32; 3]; n];
        let mut tan2 = vec![[0.0f32; 3]; n];

        let tri_count = indices.len() / 3;
        for t in 0..tri_count {
            let i0 = indices[t * 3] as usize;
            let i1 = indices[t * 3 + 1] as usize;
            let i2 = indices[t * 3 + 2] as usize;

            let p0 = positions[i0];
            let p1 = positions[i1];
            let p2 = positions[i2];
            let uv0 = uvs[i0];
            let uv1 = uvs[i1];
            let uv2 = uvs[i2];

            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let du1 = uv1[0] - uv0[0];
            let dv1 = uv1[1] - uv0[1];
            let du2 = uv2[0] - uv0[0];
            let dv2 = uv2[1] - uv0[1];

            let det = du1 * dv2 - du2 * dv1;
            if det.abs() < 1e-10 {
                continue;
            }
            let r = 1.0 / det;

            let sdir = [
                (dv2 * e1[0] - dv1 * e2[0]) * r,
                (dv2 * e1[1] - dv1 * e2[1]) * r,
                (dv2 * e1[2] - dv1 * e2[2]) * r,
            ];
            let tdir = [
                (du1 * e2[0] - du2 * e1[0]) * r,
                (du1 * e2[1] - du2 * e1[1]) * r,
                (du1 * e2[2] - du2 * e1[2]) * r,
            ];

            for &vi in &[i0, i1, i2] {
                for k in 0..3 {
                    tan1[vi][k] += sdir[k];
                }
                for k in 0..3 {
                    tan2[vi][k] += tdir[k];
                }
            }
        }

        (0..n)
            .map(|i| {
                let n_v = normals[i];
                let t = tan1[i];
                // Gram-Schmidt orthogonalize
                let dot = n_v[0] * t[0] + n_v[1] * t[1] + n_v[2] * t[2];
                let tx = t[0] - n_v[0] * dot;
                let ty = t[1] - n_v[1] * dot;
                let tz = t[2] - n_v[2] * dot;
                let len = (tx * tx + ty * ty + tz * tz).sqrt();
                let (tx, ty, tz) = if len > 1e-7 {
                    (tx / len, ty / len, tz / len)
                } else {
                    (1.0, 0.0, 0.0)
                };
                // Handedness: cross(n, t) · tan2
                let cx = n_v[1] * tz - n_v[2] * ty;
                let cy = n_v[2] * tx - n_v[0] * tz;
                let cz = n_v[0] * ty - n_v[1] * tx;
                let w = if cx * tan2[i][0] + cy * tan2[i][1] + cz * tan2[i][2] < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                [tx, ty, tz, w]
            })
            .collect()
    }

    /// Validate mesh data before upload.
    fn validate_mesh_data(data: &MeshData) -> crate::error::ViewportResult<()> {
        if data.positions.is_empty() || data.indices.is_empty() {
            return Err(crate::error::ViewportError::EmptyMesh {
                positions: data.positions.len(),
                indices: data.indices.len(),
            });
        }
        if data.positions.len() != data.normals.len() {
            return Err(crate::error::ViewportError::MeshLengthMismatch {
                positions: data.positions.len(),
                normals: data.normals.len(),
            });
        }
        let vertex_count = data.positions.len();
        for &idx in &data.indices {
            if (idx as usize) >= vertex_count {
                return Err(crate::error::ViewportError::InvalidVertexIndex {
                    vertex_index: idx,
                    vertex_count,
                });
            }
        }
        Ok(())
    }

    /// Build per-vertex normal visualization lines from mesh data.
    fn build_normal_lines(data: &MeshData) -> Vec<Vertex> {
        let normal_color = [0.627_f32, 0.769, 1.0, 1.0];
        let normal_length = 0.1_f32;
        let mut normal_line_verts: Vec<Vertex> = Vec::with_capacity(data.positions.len() * 2);
        for (p, n) in data.positions.iter().zip(data.normals.iter()) {
            let tip = [
                p[0] + n[0] * normal_length,
                p[1] + n[1] * normal_length,
                p[2] + n[2] * normal_length,
            ];
            normal_line_verts.push(Vertex {
                position: *p,
                normal: *n,
                color: normal_color,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
            normal_line_verts.push(Vertex {
                position: tip,
                normal: *n,
                color: normal_color,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
        normal_line_verts
    }

    fn create_mesh(
        device: &wgpu::Device,
        object_bgl: &wgpu::BindGroupLayout,
        fallback_albedo_view: &wgpu::TextureView,
        fallback_normal_view: &wgpu::TextureView,
        fallback_ao_view: &wgpu::TextureView,
        fallback_sampler: &wgpu::Sampler,
        fallback_lut_view: &wgpu::TextureView,
        fallback_scalar_buf: &wgpu::Buffer,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> GpuMesh {
        Self::create_mesh_with_normals(
            device,
            object_bgl,
            fallback_albedo_view,
            fallback_normal_view,
            fallback_ao_view,
            fallback_sampler,
            fallback_lut_view,
            fallback_scalar_buf,
            vertices,
            indices,
            None,
        )
    }

    fn create_mesh_with_normals(
        device: &wgpu::Device,
        object_bgl: &wgpu::BindGroupLayout,
        fallback_albedo_view: &wgpu::TextureView,
        fallback_normal_view: &wgpu::TextureView,
        fallback_ao_view: &wgpu::TextureView,
        fallback_sampler: &wgpu::Sampler,
        fallback_lut_view: &wgpu::TextureView,
        fallback_scalar_buf: &wgpu::Buffer,
        vertices: &[Vertex],
        indices: &[u32],
        normal_line_verts: Option<&[Vertex]>,
    ) -> GpuMesh {
        use bytemuck::cast_slice;
        use wgpu;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex_buf"),
            size: (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index_buf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(indices));
        index_buffer.unmap();

        // Generate deduplicated edge indices for wireframe rendering.
        let edge_indices = generate_edge_indices(indices);
        // Edge buffer needs at least 1 element to avoid zero-size buffer errors.
        let edge_buf_size = (std::mem::size_of::<u32>() * edge_indices.len().max(2)) as u64;
        let edge_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_index_buf"),
            size: edge_buf_size,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut mapped = edge_index_buffer.slice(..).get_mapped_range_mut();
            let edge_bytes = cast_slice::<u32, u8>(&edge_indices);
            mapped[..edge_bytes.len()].copy_from_slice(edge_bytes);
        }
        edge_index_buffer.unmap();

        // Default object uniform: identity model matrix, not selected, not wireframe.
        let identity = glam::Mat4::IDENTITY.to_cols_array_2d();
        let object_uniform = ObjectUniform {
            model: identity,
            color: [1.0, 1.0, 1.0, 1.0],
            selected: 0,
            wireframe: 0,
            ambient: 0.15,
            diffuse: 0.75,
            specular: 0.4,
            shininess: 32.0,
            has_texture: 0,
            use_pbr: 0,
            metallic: 0.0,
            roughness: 0.5,
            has_normal_map: 0,
            has_ao_map: 0,
            has_attribute: 0,
            scalar_min: 0.0,
            scalar_max: 1.0,
            _pad_scalar: 0,
            nan_color: [0.0, 0.0, 0.0, 0.0],
            use_nan_color: 0,
            _pad_nan: [0; 3],
        };
        let object_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_uniform_buf"),
            size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        object_uniform_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[object_uniform]));
        object_uniform_buf.unmap();

        // Combined bind group: per-object uniform (binding 0) + fallback textures/bufs (1-6).
        // Texture views and scalar buffer are updated (bind group rebuilt) when material or
        // active attribute changes in prepare().
        let object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object_bind_group"),
            layout: object_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: object_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(fallback_albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(fallback_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(fallback_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(fallback_ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(fallback_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: fallback_scalar_buf.as_entire_binding(),
                },
            ],
        });

        // Normal-line override uniform: same model matrix as object but selected=0, wireframe=0.
        let normal_override_uniform = ObjectUniform {
            model: identity,
            color: [1.0, 1.0, 1.0, 1.0],
            selected: 0,
            wireframe: 0,
            ambient: 0.15,
            diffuse: 0.75,
            specular: 0.4,
            shininess: 32.0,
            has_texture: 0,
            use_pbr: 0,
            metallic: 0.0,
            roughness: 0.5,
            has_normal_map: 0,
            has_ao_map: 0,
            has_attribute: 0,
            scalar_min: 0.0,
            scalar_max: 1.0,
            _pad_scalar: 0,
            nan_color: [0.0, 0.0, 0.0, 0.0],
            use_nan_color: 0,
            _pad_nan: [0; 3],
        };
        let normal_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("normal_uniform_buf"),
            size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        normal_uniform_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[normal_override_uniform]));
        normal_uniform_buf.unmap();

        // Normal-line bind group: also needs fallback textures/bufs in bindings 1-6 since
        // it uses the same pipeline layout as solid/wireframe.
        let normal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal_bind_group"),
            layout: object_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normal_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(fallback_albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(fallback_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(fallback_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(fallback_ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(fallback_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: fallback_scalar_buf.as_entire_binding(),
                },
            ],
        });

        // Build normal line buffer if normal vertex data was provided.
        let (normal_line_buffer, normal_line_count) = if let Some(nl_verts) = normal_line_verts {
            if nl_verts.is_empty() {
                (None, 0)
            } else {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("normal_line_buf"),
                    size: (std::mem::size_of::<Vertex>() * nl_verts.len()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buf.slice(..)
                    .get_mapped_range_mut()
                    .copy_from_slice(cast_slice(nl_verts));
                buf.unmap();
                let count = nl_verts.len() as u32;
                (Some(buf), count)
            }
        } else {
            (None, 0)
        };

        // Compute local-space AABB from vertex positions.
        let aabb = crate::scene::aabb::Aabb::from_positions(
            &vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
        );

        GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            edge_index_buffer,
            edge_index_count: edge_indices.len() as u32,
            normal_line_buffer,
            normal_line_count,
            object_uniform_buf,
            object_bind_group,
            last_tex_key: (u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX), // fallback textures/bufs currently bound
            normal_uniform_buf,
            normal_bind_group,
            aabb,
            cpu_positions: None,
            cpu_indices: None,
            attribute_buffers: std::collections::HashMap::new(),
            attribute_ranges: std::collections::HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // SciVis Phase D: Volume rendering
    // -----------------------------------------------------------------------

    /// Upload a 3D scalar field to the GPU as an `R32Float` 3D texture.
    ///
    /// `data` must be a flat array of `dims[0] * dims[1] * dims[2]` scalars in
    /// x-fastest order (index = x + y*nx + z*nx*ny).
    ///
    /// Returns a [`VolumeId`](crate::resources::VolumeId) that can be stored in [`VolumeItem::volume_id`](crate::renderer::VolumeItem::volume_id).
    pub fn upload_volume(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[f32],
        dims: [u32; 3],
    ) -> VolumeId {
        let expected = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        assert_eq!(
            data.len(),
            expected,
            "volume data length {} does not match dims {:?} (expected {})",
            data.len(),
            dims,
            expected
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume_3d_texture"),
            size: wgpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let bytes: &[u8] = bytemuck::cast_slice(data);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims[0] * 4), // R32Float = 4 bytes per texel
                rows_per_image: Some(dims[1]),
            },
            wgpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = VolumeId(self.volume_textures.len());
        self.volume_textures.push((texture, view));
        id
    }

    /// Create the volume render pipeline and bind group layout (lazy init).
    pub(crate) fn ensure_volume_pipeline(&mut self, device: &wgpu::Device) {
        if self.volume_pipeline.is_some() {
            return;
        }

        // Bind group layout for group 1 (volume-specific).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume_bgl"),
            entries: &[
                // binding 0: VolumeUniform
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
                // binding 1: 3D texture (R32Float — not filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: nearest sampler for 3D volume texture (non-filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 3: color LUT texture (2D, Rgba8Unorm — filterable)
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
                // binding 4: opacity LUT texture (2D, R8Unorm — filterable)
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
                // binding 5: linear sampler for LUT textures (filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader_src = include_str!("../shaders/volume.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volume_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volume_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12, // 3 x f32
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
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
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Need both front and back faces for entry/exit
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false, // Read-only depth — volume respects scene geometry
                depth_compare: wgpu::CompareFunction::Less,
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

        self.volume_pipeline = Some(pipeline);
        self.volume_bgl = Some(bgl);
    }

    /// Ensure the unit cube vertex + index buffers for volume bounding box proxy exist.
    fn ensure_volume_cube(&mut self, device: &wgpu::Device) {
        if self.volume_cube_vb.is_some() {
            return;
        }

        // Unit cube [0,1]^3 — 8 vertices, 36 indices (12 triangles).
        #[rustfmt::skip]
        let vertices: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
            [0.0, 0.0, 1.0], // 4
            [1.0, 0.0, 1.0], // 5
            [1.0, 1.0, 1.0], // 6
            [0.0, 1.0, 1.0], // 7
        ];

        #[rustfmt::skip]
        let indices: [u32; 36] = [
            // Front (z=0)
            0, 2, 1,  0, 3, 2,
            // Back (z=1)
            4, 5, 6,  4, 6, 7,
            // Left (x=0)
            0, 4, 7,  0, 7, 3,
            // Right (x=1)
            1, 2, 6,  1, 6, 5,
            // Bottom (y=0)
            0, 1, 5,  0, 5, 4,
            // Top (y=1)
            3, 7, 6,  3, 6, 2,
        ];

        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_cube_vb"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = vbuf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&vertices));
        }
        vbuf.unmap();

        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_cube_ib"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = ibuf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&indices));
        }
        ibuf.unmap();

        self.volume_cube_vb = Some(vbuf);
        self.volume_cube_ib = Some(ibuf);
    }

    /// Ensure the default linear ramp opacity LUT exists.
    fn ensure_default_opacity_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.volume_default_opacity_lut.is_some() {
            return;
        }

        // 256-texel linear ramp: opacity goes from 0 to 255.
        let mut data = [0u8; 256];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as u8;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume_default_opacity_lut"),
            size: wgpu::Extent3d {
                width: 256,
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
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.volume_default_opacity_lut = Some(texture);
        self.volume_default_opacity_lut_view = Some(view);
    }

    /// Prepare per-frame GPU data for a single volume item.
    pub(crate) fn upload_volume_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::VolumeItem,
        clip_planes: &[crate::renderer::ClipPlane],
    ) -> VolumeGpuData {
        self.ensure_volume_cube(device);
        self.ensure_default_opacity_lut(device, queue);

        let vol_id = item.volume_id.0;
        assert!(
            vol_id < self.volume_textures.len(),
            "invalid VolumeId: {} (only {} volumes uploaded)",
            vol_id,
            self.volume_textures.len()
        );

        let dims = {
            let tex = &self.volume_textures[vol_id].0;
            let size = tex.size();
            [size.width, size.height, size.depth_or_array_layers]
        };

        // Build model matrix: translate(bbox_min) * scale(bbox_max - bbox_min) * item.model
        let item_model = glam::Mat4::from_cols_array_2d(&item.model);
        let bbox_min = glam::Vec3::from(item.bbox_min);
        let bbox_max = glam::Vec3::from(item.bbox_max);
        let extent = bbox_max - bbox_min;
        let bbox_model = glam::Mat4::from_translation(bbox_min) * glam::Mat4::from_scale(extent);
        let model = item_model * bbox_model;
        let inv_model = model.inverse();

        // Compute step size in model/unit-cube space: the shader marches rays in the unit cube
        // [0,1]^3, so step_size must be expressed in that space (not world space).
        // One voxel in unit cube space = 1 / max_dim, regardless of world-space extent.
        // The world-space extent is already encoded in the model matrix.
        let max_dim = dims[0].max(dims[1]).max(dims[2]) as f32;
        let step_size = item.step_scale / max_dim.max(1.0);

        // Build clip planes (only enabled ones, up to 6).
        let mut clip_plane_data = [[0.0f32; 4]; 6];
        let mut num_clip = 0u32;
        for cp in clip_planes.iter().filter(|c| c.enabled).take(6) {
            clip_plane_data[num_clip as usize] =
                [cp.normal[0], cp.normal[1], cp.normal[2], cp.distance];
            num_clip += 1;
        }

        // VolumeUniform (304 bytes, 16-byte aligned):
        //   model (64), inv_model (64), bbox_min+step_size (16), bbox_max+opacity_scale (16),
        //   scalar_min(4) + scalar_max(4) + threshold_min(4) + threshold_max(4) = 16,
        //   enable_shading(4) + num_clip_planes(4) + use_nan_color(4) + _pad0(4) = 16,
        //   nan_color (16),
        //   clip_planes (6 * 16 = 96)
        //   Total: 64+64+16+16+16+16+16+96 = 304 bytes
        let mut uniform_data = [0u8; 304];
        {
            let mut offset = 0usize;
            let model_arr = model.to_cols_array();
            let model_bytes: &[u8] = bytemuck::bytes_of(&model_arr);
            uniform_data[offset..offset + 64].copy_from_slice(model_bytes);
            offset += 64;
            let inv_model_arr = inv_model.to_cols_array();
            let inv_model_bytes: &[u8] = bytemuck::bytes_of(&inv_model_arr);
            uniform_data[offset..offset + 64].copy_from_slice(inv_model_bytes);
            offset += 64;
            // bbox_min + step_size
            uniform_data[offset..offset + 12].copy_from_slice(bytemuck::bytes_of(&item.bbox_min));
            offset += 12;
            uniform_data[offset..offset + 4].copy_from_slice(bytemuck::bytes_of(&step_size));
            offset += 4;
            // bbox_max + opacity_scale
            uniform_data[offset..offset + 12].copy_from_slice(bytemuck::bytes_of(&item.bbox_max));
            offset += 12;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.opacity_scale));
            offset += 4;
            // scalar_min, scalar_max, threshold_min, threshold_max
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.scalar_range.0));
            offset += 4;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.scalar_range.1));
            offset += 4;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.threshold_min));
            offset += 4;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.threshold_max));
            offset += 4;
            // enable_shading, num_clip_planes, use_nan_color, _pad0
            let shading_u32: u32 = if item.enable_shading { 1 } else { 0 };
            uniform_data[offset..offset + 4].copy_from_slice(bytemuck::bytes_of(&shading_u32));
            offset += 4;
            uniform_data[offset..offset + 4].copy_from_slice(bytemuck::bytes_of(&num_clip));
            offset += 4;
            let use_nan_color_u32: u32 = if item.nan_color.is_some() { 1 } else { 0 };
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&use_nan_color_u32));
            offset += 4;
            offset += 4; // _pad0 (already zero from [0u8; 304])
            // nan_color (vec4<f32>)
            let nan_color = item.nan_color.unwrap_or([0.0f32; 4]);
            uniform_data[offset..offset + 16].copy_from_slice(bytemuck::bytes_of(&nan_color));
            offset += 16;
            // clip_planes (6 * vec4<f32>)
            for cp in &clip_plane_data {
                uniform_data[offset..offset + 16].copy_from_slice(bytemuck::bytes_of(cp));
                offset += 16;
            }
            debug_assert_eq!(offset, 304);
        }

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_uniform_buf"),
            size: 304,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = uniform_buf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(&uniform_data);
        }
        uniform_buf.unmap();

        // Look up texture views.
        let volume_view = &self.volume_textures[vol_id].1;

        // Color LUT: use specified colormap or fall back to viridis.
        let color_lut_view = if let Some(cmap_id) = item.color_lut {
            self.colormap_views
                .get(cmap_id.0)
                .unwrap_or(&self.fallback_lut_view)
        } else {
            // Default to viridis (first builtin).
            if let Some(ids) = &self.builtin_colormap_ids {
                self.colormap_views
                    .get(ids[0].0)
                    .unwrap_or(&self.fallback_lut_view)
            } else {
                &self.fallback_lut_view
            }
        };

        // Opacity LUT: use specified or default linear ramp.
        let opacity_lut_view = if let Some(cmap_id) = item.opacity_lut {
            self.colormap_views
                .get(cmap_id.0)
                .unwrap_or(self.volume_default_opacity_lut_view.as_ref().unwrap())
        } else {
            self.volume_default_opacity_lut_view.as_ref().unwrap()
        };

        // Nearest sampler for 3D volume texture (R32Float is not filterable).
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Linear sampler for LUT textures (Rgba8Unorm / R8Unorm are filterable).
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume_lut_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bgl = self
            .volume_bgl
            .as_ref()
            .expect("ensure_volume_pipeline not called");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(color_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(opacity_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });

        // Clone the cached unit cube buffers.
        let vbuf = self.volume_cube_vb.as_ref().unwrap();
        let ibuf = self.volume_cube_ib.as_ref().unwrap();

        // For the vertex/index buffers we need owned copies per VolumeGpuData, but
        // since the cube is shared we use the same buffers. However wgpu::Buffer
        // cannot be cloned, so we share them by creating lightweight copies via device.
        // Actually, the draw call just needs a reference, but VolumeGpuData owns them.
        // Since all volumes share the same unit cube, let's just create one pair.
        // We'll store references to the cached buffers. But VolumeGpuData needs owned
        // buffers or static refs. Let's use the same trick as GlyphGpuData.

        // Safety: volume_cube_vb/ib live as long as ViewportGpuResources.
        let vbuf_ref: &'static wgpu::Buffer = unsafe { &*(vbuf as *const wgpu::Buffer) };
        let ibuf_ref: &'static wgpu::Buffer = unsafe { &*(ibuf as *const wgpu::Buffer) };

        // We need to return owned buffers in VolumeGpuData. Since we use static refs
        // for the cube, let's change VolumeGpuData to use static refs like GlyphGpuData.
        // But we already defined it with owned buffers. Let's create a small per-volume
        // vertex buffer since it's cheap (96 bytes) and avoids unsafe.
        // Actually, 8 verts * 12 bytes = 96 bytes per volume is fine.
        let _ = vbuf_ref;
        let _ = ibuf_ref;

        // Just create per-volume copies from the cached data to keep it simple.
        #[rustfmt::skip]
        let vertices: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        #[rustfmt::skip]
        let indices: [u32; 36] = [
            0,2,1, 0,3,2, 4,5,6, 4,6,7,
            0,4,7, 0,7,3, 1,2,6, 1,6,5,
            0,1,5, 0,5,4, 3,7,6, 3,6,2,
        ];

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_cube_vb_frame"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = vertex_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&vertices));
        }
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_cube_ib_frame"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = index_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&indices));
        }
        index_buffer.unmap();

        VolumeGpuData {
            bind_group,
            vertex_buffer,
            index_buffer,
            _dims: dims,
            _uniform_buf: uniform_buf,
        }
    }
}
