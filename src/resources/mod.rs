/// Built-in colormap LUT data.
pub mod colormap_data;
mod extra_impls;
/// Slotted GPU mesh storage with free-list removal.
pub mod mesh_store;
mod meshes;
mod overlays;
mod scivis;
mod textures;
mod types;
mod volumes;

pub use self::extra_impls::{ComputeFilterResult, lerp_attributes};
use self::extra_impls::{
    build_glyph_arrow, build_glyph_sphere, build_streamtube_cylinder, build_unit_cube,
    generate_edge_indices,
};
pub use self::types::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColormap, CameraUniform, ClipVolumeUniform,
    ColormapId, GpuMesh, GpuTexture, LightUniform, LightsUniform, MeshData, OverlayVertex,
    PointCloudGpuData, PolylineGpuData, SingleLightUniform, Vertex, ViewportGpuResources,
    VolumeGpuData, VolumeId,
};
pub(crate) use self::types::{
    BloomUniform, ClipPlanesUniform, ContactShadowUniform, GlyphBaseMesh, GlyphGpuData,
    InstanceData, ObjectUniform, OutlineObjectBuffers, OutlineUniform, OverlayUniform,
    PickInstance, SHADOW_ATLAS_SIZE, ShadowAtlasUniform, SsaoUniform, StreamtubeGpuData,
    ToneMapUniform,
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
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/shadow_instanced.wgsl").into(),
            ),
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
}
