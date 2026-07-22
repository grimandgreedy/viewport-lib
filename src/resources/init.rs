use super::*;

impl DeviceResources {
    /// Create all GPU resources for the viewport.
    ///
    /// Call once at application startup. `target_format` must match the swap-chain surface
    /// format. Use `sample_count = 1` unless the caller is providing MSAA resolve targets.
    pub fn new(
        device: &crate::gpu::Device,
        target_format: crate::gpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        Self::new_with_cache(device, target_format, sample_count, None)
    }

    /// Like [`new`](Self::new), but seeds a `wgpu::PipelineCache` from previously
    /// saved data so shader compilation can be skipped on later launches.
    ///
    /// `pipeline_cache_data` should come from a prior
    /// [`ViewportRenderer::pipeline_cache_data`](crate::renderer::ViewportRenderer::pipeline_cache_data)
    /// call, or `None` on first run. The cache is only created when the device
    /// enables `Features::PIPELINE_CACHE`; otherwise this behaves exactly like
    /// `new` and the data is ignored.
    pub fn new_with_cache(
        device: &crate::gpu::Device,
        target_format: crate::gpu::TextureFormat,
        sample_count: u32,
        pipeline_cache_data: Option<&[u8]>,
    ) -> Self {
        // A pipeline cache records compiled pipelines so a later run (seeded with
        // saved data) skips recompilation. Only available when the device enables
        // `Features::PIPELINE_CACHE`; `fallback: true` discards stale/invalid data
        // rather than failing. Safety: the data, if any, came from a prior
        // `PipelineCache::get_data` call as the API requires.
        let pipeline_cache = if device
            .features()
            .contains(crate::gpu::Features::PIPELINE_CACHE)
        {
            Some(unsafe {
                device.create_pipeline_cache(&crate::gpu::PipelineCacheDescriptor {
                    label: Some("viewport_pipeline_cache"),
                    data: pipeline_cache_data,
                    fallback: true,
                })
            })
        } else {
            None
        };

        // Cold-start instrumentation. Pipeline compilation and large depth-texture
        // allocation can dominate construction on some backends (notably Adreno
        // Vulkan, where pipeline creation compiles shaders synchronously). These
        // marks attribute the per-phase cost. Filter with
        // `RUST_LOG=viewport_lib::init=info`.
        let init_start = std::time::Instant::now();
        let mut init_ckpt = init_start;
        let mut mark = |section: &str| {
            let now = std::time::Instant::now();
            tracing::info!(
                target: "viewport_lib::init",
                section,
                ms = now.duration_since(init_ckpt).as_secs_f32() * 1000.0,
                "gpu resources init phase"
            );
            init_ckpt = now;
        };

        // ------------------------------------------------------------------
        // Shader module
        // ------------------------------------------------------------------
        // iced_wgpu and other WebGL2-targeting backends cap max_bind_groups at
        // 2. viewport-lib's mesh shader family uses 3 groups (camera, object,
        // deform). On limited devices we compile the _noop variants which omit
        // the deform group, and build 2-group pipeline layouts instead.
        let deform_enabled = device.limits().max_bind_groups >= 3;

        let mesh_src = if deform_enabled {
            include_str!(concat!(env!("OUT_DIR"), "/mesh.wgsl"))
        } else {
            include_str!(concat!(env!("OUT_DIR"), "/mesh_noop.wgsl"))
        };
        // Lit modules compile without the pixel-inspector debug block
        // (debug_vis_shaders starts false); DebugVis rebuilds them with it.
        let shader = crate::resources::builders::wgsl_module(
            device,
            "mesh_shader",
            crate::resources::builders::strip_mesh_non_pbr(
                crate::resources::builders::strip_mesh_discards(
                    crate::resources::builders::strip_debug_vis(mesh_src, false),
                ),
            ),
        );

        // ------------------------------------------------------------------
        // Bind group layouts
        // ------------------------------------------------------------------
        let camera_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(
                        crate::gpu::SamplerBindingType::Comparison,
                    ),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: clip planes uniform (section view clipping).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: shadow atlas uniform (CSM matrices, splits, PCSS params).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: clip volume uniform (box/sphere/plane extended clip region).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: IBL irradiance equirect texture.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 8: IBL prefiltered specular equirect texture.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 9: BRDF integration LUT texture.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 10: IBL sampler (linear, clamp-to-edge).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 11: Skybox/environment equirect texture (full-res for skybox).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 12: per-fragment debug storage buffer (written in debug_vis.wgsl).
                // Sized to viewport_width * viewport_height * 16 bytes when debug is active;
                // a 16-byte sentinel buffer is used otherwise.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 13: read-only storage buffer of `SingleLightUniform`
                // entries. Indexed against the `count` field of the lights
                // header uniform (binding 3). Capacity = `MAX_SCENE_LIGHTS`.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 14: clustered-shading grid uniform (dimensions,
                // near/far, screen size, fallback flag).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 15: cluster cell storage (offset + count per cell),
                // read-only in the fragment stage.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 16: global cluster light index list, read-only in
                // the fragment stage.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 17: point-light shadow depth array.
                // iOS Metal does not support cube array textures, so on that
                // target the shader uses texture_depth_2d_array and we bind a
                // D2Array view instead.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: if cfg!(target_os = "ios") {
                            crate::gpu::TextureViewDimension::D2Array
                        } else {
                            crate::gpu::TextureViewDimension::CubeArray
                        },
                        multisampled: false,
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
        let object_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("object_bgl"),
            entries: &[
                // binding 0: per-object uniform buffer
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: albedo texture (filterable)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: shared filtering sampler
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: normal map texture (filterable)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: AO map texture (filterable)
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
                // binding 5: LUT (colourmap) texture (256x1 Rgba8Unorm, FRAGMENT, filterable)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 6: scalar attribute storage buffer (VERTEX | FRAGMENT, read-only)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7: matcap texture (FRAGMENT, filterable 2D)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 8: per-face colour storage buffer (VERTEX | FRAGMENT, read-only)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 9: warp vector storage buffer (VERTEX, read-only)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 10: LUT clamp sampler (FRAGMENT, filtering)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 11: metallic-roughness ORM texture (FRAGMENT, filterable)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 12: emissive texture (FRAGMENT, filterable)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 13: position override storage buffer (VERTEX, read-only).
                // Fallback sentinel (1x vec3<f32>) is bound when no override is set.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 14: normal override storage buffer (VERTEX, read-only).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Texture-only bind group layout : kept for the instanced pipeline (group 1 bindings
        // 1-4 are added alongside the storage buffer binding 0 in init_instanced_pipeline).
        // Also used as the standalone layout when creating material bind groups keyed by
        // texture combination for the instanced path.
        let texture_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("texture_bgl"),
            entries: &[
                // binding 0: albedo texture (filterable)
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
                // binding 1: shared filtering sampler
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: normal map texture (filterable)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: AO map texture (filterable)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
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

        mark("shaders_and_bind_group_layouts");

        // ------------------------------------------------------------------
        // Per-vertex deformation sidecar. Constructed early because every
        // mesh-family pipeline layout binds its group(2) BGL.
        // ------------------------------------------------------------------
        let deform = if deform_enabled {
            crate::resources::mesh_sidecar::deform::DeformationState::new(device)
        } else {
            crate::resources::mesh_sidecar::deform::DeformationState::new_disabled(device)
        };
        let deform_bgl = deform.enabled.then_some(&deform.bind_group_layout);

        // ------------------------------------------------------------------
        // Pipeline layout (shared between solid and transparent pipelines)
        // Groups: 0=camera, 1=object+texture, and optionally 2=deform sidecar
        // ------------------------------------------------------------------
        let pipeline_layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            "mesh_pipeline_layout",
            &camera_bgl,
            &object_bgl,
            deform_bgl,
        );

        // ------------------------------------------------------------------
        // LDR mesh.wgsl pipelines: solid + two-sided + transparent + wireframe.
        // Built through the shared factory so `register_deformer` can rebuild
        // them with a freshly composed shader module.
        // ------------------------------------------------------------------
        let ldr = crate::resources::mesh::mesh_pipelines::build_ldr_mesh_pipelines(
            device,
            &pipeline_layout,
            &shader,
            target_format,
            sample_count,
            pipeline_cache.as_ref(),
        );
        let solid_pipeline = ldr.solid;
        let solid_two_sided_pipeline = ldr.solid_two_sided;
        let transparent_pipeline = ldr.transparent;
        let wireframe_pipeline = ldr.wireframe;

        mark("mesh_pipelines");

        // ------------------------------------------------------------------
        // Camera uniform buffer and bind group
        // ------------------------------------------------------------------
        let camera_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("camera_uniform_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("light_uniform_buf"),
            size: std::mem::size_of::<LightUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_storage_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("light_storage_buf"),
            size: (std::mem::size_of::<crate::resources::SingleLightUniform>()
                * crate::resources::MAX_SCENE_LIGHTS) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Clip planes uniform buffer (binding 4 of camera bind group).
        // Initialized to count=0 (no active clip planes).
        let clip_planes_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("clip_planes_uniform_buf"),
            size: std::mem::size_of::<ClipPlanesUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Clip volume uniform buffer (binding 6 of camera bind group).
        // Holds up to CLIP_VOLUME_MAX box/sphere entries; initialized to count=0 (no volumes).
        let clip_volume_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("clip_volume_uniform_buf"),
            size: std::mem::size_of::<ClipVolumesUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ------------------------------------------------------------------
        // Shadow map texture, sampler, and bind group
        // ------------------------------------------------------------------
        let shadow_map_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("shadow_atlas"),
            size: crate::gpu::Extent3d {
                width: SHADOW_ATLAS_SIZE,
                height: SHADOW_ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth32Float,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view =
            shadow_map_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        let shadow_sampler = crate::resources::builders::comparison_sampler(
            device,
            "shadow_sampler",
            crate::gpu::CompareFunction::LessEqual,
        );

        // ------------------------------------------------------------------
        // Point-light cubemap shadow texture array.
        //
        // Stored as a 2D texture array of `MAX_POINT_SHADOW_LIGHTS * 6`
        // layers and viewed two ways:
        //  - Per-face 2D-array views (one per face) for shadow render passes.
        //  - A `CubeArray` view bound to the lit pass for sampling.
        // ------------------------------------------------------------------
        let point_shadow_face_size = crate::renderer::POINT_SHADOW_FACE_SIZE;
        let point_shadow_max_lights = crate::renderer::MAX_POINT_SHADOW_LIGHTS;
        let point_shadow_layers = point_shadow_max_lights * 6;
        let point_shadow_cube_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("point_shadow_cube_array"),
            size: crate::gpu::Extent3d {
                width: point_shadow_face_size,
                height: point_shadow_face_size,
                depth_or_array_layers: point_shadow_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth32Float,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let point_shadow_cube_view =
            point_shadow_cube_texture.create_view(&crate::gpu::TextureViewDescriptor {
                label: Some("point_shadow_cube_view"),
                // iOS Metal does not support CubeArray views. Use D2Array instead;
                // the shader is patched at build time to match.
                dimension: Some(if cfg!(target_os = "ios") {
                    crate::gpu::TextureViewDimension::D2Array
                } else {
                    crate::gpu::TextureViewDimension::CubeArray
                }),
                aspect: crate::gpu::TextureAspect::DepthOnly,
                base_array_layer: 0,
                array_layer_count: Some(point_shadow_layers),
                base_mip_level: 0,
                mip_level_count: Some(1),
                format: Some(crate::gpu::TextureFormat::Depth32Float),
                usage: None,
            });
        let point_shadow_face_views: Vec<crate::gpu::TextureView> = (0..point_shadow_layers)
            .map(|layer| {
                point_shadow_cube_texture.create_view(&crate::gpu::TextureViewDescriptor {
                    label: Some("point_shadow_face_view"),
                    dimension: Some(crate::gpu::TextureViewDimension::D2),
                    aspect: crate::gpu::TextureAspect::DepthOnly,
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    format: Some(crate::gpu::TextureFormat::Depth32Float),
                    usage: None,
                })
            })
            .collect();

        // Includes the 4096^2 directional atlas and the point-shadow cube array
        // (POINT_SHADOW_FACE_SIZE^2 * MAX_POINT_SHADOW_LIGHTS * 6 layers), both
        // allocated unconditionally here. Watch this number on mobile.
        mark("buffers_and_shadow_textures");

        // Non-comparison sampler (no compare field) for plain float depth reads.
        let shadow_atlas_depth_sampler =
            crate::resources::builders::clamp_nearest_sampler(device, "shadow_atlas_depth_sampler");

        // Shadow atlas uniform buffer (binding 5).
        let shadow_info_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("shadow_info_buf"),
            size: std::mem::size_of::<ShadowAtlasUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ------------------------------------------------------------------
        // IBL fallback textures: 1x1 black (Rgba16Float) placeholder for all IBL slots,
        // and a linear/repeat sampler. Never sampled : the `ibl_enabled` uniform guard
        // prevents IBL calculations when no environment map is uploaded.
        // ------------------------------------------------------------------
        let ibl_fallback_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("ibl_fallback_black"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba16Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ibl_fallback_view =
            ibl_fallback_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        // BRDF integration LUT placeholder: a 1x1 black fallback that's swapped for the real
        // 128x128 LUT on the first call to `upload_environment_map`. The LUT is scene-independent
        // (function of roughness x N.V only); idempotent caching inside `upload_environment_map`
        // means subsequent uploads skip its ~16.7M Hammersley samples.
        let ibl_fallback_brdf_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("ibl_fallback_brdf"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba16Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ibl_fallback_brdf_view =
            ibl_fallback_brdf_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        let ibl_sampler = crate::resources::builders::env_sampler(device, "ibl_sampler");

        // 16-byte sentinel bound at group 0 binding 12 when the debug fragment buffer is inactive.
        let debug_frag_sentinel_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("debug_frag_sentinel_buf"),
            size: 16,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let clustered = crate::resources::gpu::clustered::ClusteredResources::new(device);

        mark("clustered");

        let camera_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(&shadow_map_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&shadow_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: light_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: clip_planes_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_info_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: clip_volume_uniform_buf.as_entire_binding(),
                },
                // IBL textures (bindings 7-11) : fallback until environment is uploaded.
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: crate::gpu::BindingResource::TextureView(&ibl_fallback_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: crate::gpu::BindingResource::TextureView(&ibl_fallback_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 9,
                    resource: crate::gpu::BindingResource::TextureView(&ibl_fallback_brdf_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 10,
                    resource: crate::gpu::BindingResource::Sampler(&ibl_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 11,
                    resource: crate::gpu::BindingResource::TextureView(&ibl_fallback_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 12,
                    resource: debug_frag_sentinel_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 13,
                    resource: light_storage_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 14,
                    resource: clustered.grid_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 15,
                    resource: clustered.cluster_grid_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 16,
                    resource: clustered.light_index_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 17,
                    resource: crate::gpu::BindingResource::TextureView(&point_shadow_cube_view),
                },
            ],
        });

        // ------------------------------------------------------------------
        // Shadow pass pipeline (depth-only, renders from light's POV)
        // ------------------------------------------------------------------
        let shadow_src = if deform_enabled {
            include_str!(concat!(env!("OUT_DIR"), "/shadow.wgsl"))
        } else {
            include_str!(concat!(env!("OUT_DIR"), "/shadow_noop.wgsl"))
        };
        let shadow_shader =
            crate::resources::builders::wgsl_module(device, "shadow_shader", shadow_src);

        // Shadow pass uses a simple bind group layout: just the light uniform.
        let shadow_camera_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("shadow_camera_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        // Dynamic offset lets the cascade loop select per-cascade matrix slot
                        // without calling write_buffer inside the render pass (which would be
                        // a no-op per-cascade since wgpu batches all writes before execution).
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shadow_pl_bgls: Vec<&crate::gpu::BindGroupLayout> = if let Some(d) = deform_bgl {
            vec![&shadow_camera_bgl, &object_bgl, d]
        } else {
            vec![&shadow_camera_bgl, &object_bgl]
        };
        let shadow_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "shadow_pipeline_layout",
            &shadow_pl_bgls,
        );

        // Depth-only pass through the shared factory so register_deformer
        // can rebuild it from composed source. Two variants:
        // - cull-front (default) for closed solids: back faces are the
        //   casters, so a solid's own front face is never compared against
        //   itself in the shadow map.
        // - cull-none for two-sided materials (`BackfacePolicy::Identical`):
        //   both sides of cloth and planar surfaces rasterise; a larger
        //   caster-side bias keeps the receiver from self-shadowing.
        let shadow_pipeline = crate::resources::mesh::mesh_pipelines::build_shadow_pipeline(
            device,
            &shadow_pipeline_layout,
            &shadow_shader,
            Some(crate::gpu::Face::Front),
            pipeline_cache.as_ref(),
        );
        let shadow_pipeline_two_sided =
            crate::resources::mesh::mesh_pipelines::build_shadow_pipeline(
                device,
                &shadow_pipeline_layout,
                &shadow_shader,
                None,
                pipeline_cache.as_ref(),
            );

        // Shadow pass uniform buffer : 4 cascade slots x 256 bytes (wgpu dynamic-offset alignment).
        // Each slot holds one 4x4 matrix (64 bytes); the remaining 192 bytes per slot are padding.
        const SHADOW_SLOT_STRIDE: u64 = 256;
        let shadow_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("shadow_uniform_buf"),
            size: 4 * SHADOW_SLOT_STRIDE,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("shadow_bind_group"),
            layout: &shadow_camera_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                // Bind only the first 64-byte matrix slot; dynamic offset selects cascade.
                resource: crate::gpu::BindingResource::Buffer(crate::gpu::BufferBinding {
                    buffer: &shadow_uniform_buf,
                    offset: 0,
                    size: Some(
                        crate::gpu::BufferSize::new(std::mem::size_of::<[[f32; 4]; 4]>() as u64)
                            .unwrap(),
                    ),
                }),
            }],
        });

        // ------------------------------------------------------------------
        // Point-light cubemap shadow pipeline.
        //
        // One render pass per (light slot, face) writes linear distance to
        // the light into a per-face depth attachment. The bind group layout
        // mirrors the cascade pipeline (same object + deform bind groups)
        // and carries a per-face uniform with view_proj + light_pos + range.
        // ------------------------------------------------------------------
        let shadow_point_src = if deform_enabled {
            include_str!(concat!(env!("OUT_DIR"), "/shadow_point.wgsl"))
        } else {
            include_str!(concat!(env!("OUT_DIR"), "/shadow_point_noop.wgsl"))
        };
        let shadow_point_shader = crate::resources::builders::wgsl_module(
            device,
            "shadow_point_shader",
            shadow_point_src,
        );
        let shadow_point_face_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("shadow_point_face_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let shadow_point_pl_bgls: Vec<&crate::gpu::BindGroupLayout> = if let Some(d) = deform_bgl {
            vec![&shadow_point_face_bgl, &object_bgl, d]
        } else {
            vec![&shadow_point_face_bgl, &object_bgl]
        };
        let shadow_point_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "shadow_point_pipeline_layout",
            &shadow_point_pl_bgls,
        );
        let shadow_point_pipeline =
            crate::resources::mesh::mesh_pipelines::build_shadow_point_pipeline(
                device,
                &shadow_point_pipeline_layout,
                &shadow_point_shader,
                pipeline_cache.as_ref(),
            );

        // Per-face uniform buffer. Stride 256 satisfies wgpu's dynamic-offset
        // alignment requirement. Total slots = MAX_POINT_SHADOW_LIGHTS * 6.
        const SHADOW_POINT_FACE_STRIDE: u64 = 256;
        let shadow_point_face_count = (point_shadow_max_lights * 6) as u64;
        let shadow_point_face_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("shadow_point_face_buf"),
            size: shadow_point_face_count * SHADOW_POINT_FACE_STRIDE,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_point_face_bind_group =
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("shadow_point_face_bind_group"),
                layout: &shadow_point_face_bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::Buffer(crate::gpu::BufferBinding {
                        buffer: &shadow_point_face_buf,
                        offset: 0,
                        // Bind one PointFace slot worth (96 bytes rounded up
                        // to 16-byte alignment is fine here).
                        size: Some(crate::gpu::BufferSize::new(96).unwrap()),
                    }),
                }],
            });

        mark("shadow_pipelines");

        // ------------------------------------------------------------------
        // Gizmo shader module
        // ------------------------------------------------------------------
        let gizmo_shader = crate::resources::builders::wgsl_module(
            device,
            "gizmo_shader",
            crate::resources::builders::wgsl_source!("gizmo"),
        );

        // ------------------------------------------------------------------
        // Gizmo bind group layout (group 1: model matrix uniform)
        // ------------------------------------------------------------------
        let gizmo_bgl = crate::resources::builders::uniform_bgl(
            device,
            "gizmo_bgl",
            crate::gpu::ShaderStages::VERTEX,
        );

        // ------------------------------------------------------------------
        // Gizmo pipeline layout
        // ------------------------------------------------------------------
        let gizmo_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "gizmo_pipeline_layout",
            &[&camera_bgl, &gizmo_bgl],
        );

        // ------------------------------------------------------------------
        // Gizmo render pipeline
        // depth_compare: Always : gizmo always renders on top of scene (Pitfall 8).
        // depth_write_enabled: false : do not corrupt depth buffer.
        // ------------------------------------------------------------------
        let gizmo_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "gizmo_pipeline",
                layout: &gizmo_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &gizmo_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &gizmo_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None, // No culling: gizmo geometry is viewed from all angles.
                    unclipped_depth: false,
                    polygon_mode: crate::gpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Always, // Always on top.
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );

        // ------------------------------------------------------------------
        // Gizmo vertex/index buffers (initial mesh: no hover highlight)
        // ------------------------------------------------------------------
        let (gizmo_verts, gizmo_indices) =
            crate::interaction::manipulation::gizmo::build_gizmo_mesh(
                crate::interaction::manipulation::gizmo::GizmoMode::Translate,
                crate::interaction::manipulation::gizmo::GizmoAxis::None,
                glam::Quat::IDENTITY,
            );

        let gizmo_vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("gizmo_vertex_buf"),
            size: (std::mem::size_of::<Vertex>() * gizmo_verts.len().max(1)) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            gizmo_vertex_buffer.slice(..),
            bytemuck::cast_slice(&gizmo_verts),
        );
        gizmo_vertex_buffer.unmap();

        let gizmo_index_count = gizmo_indices.len() as u32;
        let gizmo_index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("gizmo_index_buf"),
            size: (std::mem::size_of::<u32>() * gizmo_indices.len().max(1)) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            gizmo_index_buffer.slice(..),
            bytemuck::cast_slice(&gizmo_indices),
        );
        gizmo_index_buffer.unmap();

        // ------------------------------------------------------------------
        // Gizmo uniform buffer (model matrix : identity until first update)
        // ------------------------------------------------------------------
        let gizmo_uniform = crate::interaction::manipulation::gizmo::GizmoUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
        };
        let gizmo_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("gizmo_uniform_buf"),
            size: std::mem::size_of::<crate::interaction::manipulation::gizmo::GizmoUniform>()
                as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            gizmo_uniform_buf.slice(..),
            bytemuck::cast_slice(&[gizmo_uniform]),
        );
        gizmo_uniform_buf.unmap();

        let gizmo_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("gizmo_bind_group"),
            layout: &gizmo_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: gizmo_uniform_buf.as_entire_binding(),
            }],
        });

        // ------------------------------------------------------------------
        // Overlay shader module
        // ------------------------------------------------------------------
        let overlay_shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_shader",
            crate::resources::builders::wgsl_source!("overlay"),
        );

        // ------------------------------------------------------------------
        // Overlay bind group layout (group 1: model + colour uniform)
        // ------------------------------------------------------------------
        let overlay_bgl = crate::resources::builders::uniform_bgl(
            device,
            "overlay_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
        );

        // ------------------------------------------------------------------
        // Overlay pipeline layout (group 0: camera, group 1: overlay uniform)
        // ------------------------------------------------------------------
        let overlay_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "overlay_pipeline_layout",
            &[&camera_bgl, &overlay_bgl],
        );

        // ------------------------------------------------------------------
        // Overlay render pipeline
        // TriangleList topology with alpha blending for semi-transparent quads.
        // depth_write_enabled: false : do not corrupt depth buffer with overlays.
        // depth_compare: Less : overlays respect depth (hidden by geometry in front).
        // cull_mode: None : quads viewed from both sides.
        // ------------------------------------------------------------------
        let overlay_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_pipeline",
                layout: &overlay_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &overlay_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[OverlayVertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None, // BC quads are visible from both sides.
                    unclipped_depth: false,
                    polygon_mode: crate::gpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false, // Do not write to depth buffer.
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );

        // ------------------------------------------------------------------
        // Overlay line pipeline (LineList)
        // Uses the same overlay shader + bind group layout as the triangle overlay.
        // No alpha blending needed for line overlays.
        // depth_write_enabled: false : overlay lines don't corrupt depth buffer.
        // ------------------------------------------------------------------
        let overlay_line_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_line_pipeline",
                layout: &overlay_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &overlay_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[OverlayVertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: crate::gpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );

        // ------------------------------------------------------------------
        // Full-screen analytical grid pipeline
        //
        // No vertex buffer. A hardcoded triangle in the vertex shader covers
        // the entire screen. The fragment shader ray-marches to the grid plane,
        // computes analytical anti-aliased lines with fwidth(), and writes
        // clip-space depth via @builtin(frag_depth) for correct occlusion.
        // Horizon fade eliminates clipping artefacts at shallow viewing angles.
        // ------------------------------------------------------------------
        let grid_shader = crate::resources::builders::wgsl_module(
            device,
            "grid_shader",
            crate::resources::builders::wgsl_source!("grid"),
        );
        let grid_bgl = crate::resources::builders::uniform_bgl(
            device,
            "grid_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
        );
        let grid_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "grid_pipeline_layout",
            &[&grid_bgl],
        );
        let grid_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "grid_pipeline",
                layout: &grid_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &grid_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[], // no vertex buffer : positions hardcoded in shader
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &grid_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::gpu::DepthStencilState {
                    format: crate::gpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: crate::resources::builders::dwrite(true),
                    depth_compare: crate::resources::builders::dcompare(
                        crate::gpu::CompareFunction::LessEqual,
                    ),
                    stencil: crate::gpu::StencilState::default(),
                    bias: crate::gpu::DepthBiasState {
                        // Push grid depth slightly behind coplanar geometry to prevent
                        // z-fighting when object faces coincide with the grid plane.
                        // 4 x the minimum representable Depth24 unit ~ 2.4e-7 : invisible
                        // at any distance but reliably loses the depth test to geometry.
                        constant: 4,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                }),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );
        // Default-zero uniform : overwritten every frame in prepare().
        let grid_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("grid_uniform_buf"),
            size: std::mem::size_of::<GridUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("grid_bind_group"),
            layout: &grid_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: grid_uniform_buf.as_entire_binding(),
            }],
        });

        // ------------------------------------------------------------------
        // Ground plane pipeline
        //
        // Full-screen ray-march approach (same as grid).  The fragment shader
        // intersects the camera ray with a horizontal plane at a configurable
        // Z height, then renders one of four modes: None (skipped), ShadowOnly,
        // Tile, SolidColour.  Uses @builtin(frag_depth) for depth occlusion.
        // ------------------------------------------------------------------
        let ground_plane_shader = crate::resources::builders::wgsl_module(
            device,
            "ground_plane_shader",
            crate::resources::builders::wgsl_source!("ground_plane"),
        );
        let ground_plane_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("ground_plane_bgl"),
                entries: &[
                    // binding 0: GroundPlaneUniform
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::VERTEX
                            | crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: shadow atlas (depth texture)
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Depth,
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2: shadow comparison sampler
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::Comparison,
                        ),
                        count: None,
                    },
                    // binding 3: shadow atlas info (cascade matrices, splits, atlas rects)
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let ground_plane_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "ground_plane_pipeline_layout",
            &[&ground_plane_bgl],
        );
        let ground_plane_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "ground_plane_pipeline",
                layout: &ground_plane_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &ground_plane_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &ground_plane_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    true,
                    crate::gpu::CompareFunction::LessEqual,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );
        let ground_plane_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("ground_plane_uniform_buf"),
            size: std::mem::size_of::<GroundPlaneUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ground_plane_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("ground_plane_bind_group"),
            layout: &ground_plane_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: ground_plane_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(&shadow_map_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&shadow_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: shadow_info_buf.as_entire_binding(),
                },
            ],
        });

        // ------------------------------------------------------------------
        // Shadow atlas viewer pipeline (corner overlay, no vertex buffers)
        // ------------------------------------------------------------------
        let atlas_blit_shader = crate::resources::builders::wgsl_module(
            device,
            "shadow_atlas_blit",
            crate::resources::builders::wgsl_source!("shadow_atlas_blit"),
        );
        let atlas_blit_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("atlas_blit_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::VERTEX
                            | crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Depth,
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::NonFiltering,
                        ),
                        count: None,
                    },
                ],
            });
        let atlas_blit_layout = crate::resources::builders::pipeline_layout(
            device,
            "atlas_blit_layout",
            &[&atlas_blit_bgl],
        );
        let shadow_atlas_viewer_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("shadow_atlas_viewer_buf"),
            size: std::mem::size_of::<AtlasBlitUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_atlas_viewer_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("shadow_atlas_viewer_bg"),
            layout: &atlas_blit_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_atlas_viewer_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(&shadow_map_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&shadow_atlas_depth_sampler),
                },
            ],
        });
        let shadow_atlas_viewer_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "shadow_atlas_viewer_pipeline",
                layout: &atlas_blit_layout,
                vertex: crate::gpu::VertexState {
                    module: &atlas_blit_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &atlas_blit_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );

        // ------------------------------------------------------------------
        // Axes indicator pipeline (screen-space, no camera, no depth)
        // ------------------------------------------------------------------
        let axes_shader = crate::resources::builders::wgsl_module(
            device,
            "axes_overlay_shader",
            crate::resources::builders::wgsl_source!("axes_overlay"),
        );

        let axes_pipeline_layout =
            crate::resources::builders::pipeline_layout(device, "axes_pipeline_layout", &[]);

        let axes_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "axes_pipeline",
                layout: &axes_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &axes_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        crate::interaction::widgets::axes_indicator::AxesVertex::buffer_layout(),
                    ],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &axes_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: crate::gpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );

        // Pre-allocate vertex buffer (resized in prepare if needed).
        let axes_vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("axes_vertex_buf"),
            size: (std::mem::size_of::<crate::interaction::widgets::axes_indicator::AxesVertex>()
                * 2048) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        mark("ui_pipelines");

        // ------------------------------------------------------------------
        // Shared material sampler (linear + repeat : reused for all material textures)
        // ------------------------------------------------------------------
        let material_sampler = crate::resources::builders::repeat_linear_sampler(
            device,
            "material_sampler",
            crate::gpu::FilterMode::Nearest,
        );

        // Clamp-to-edge sampler for colourmap LUT lookups (prevents wrap artifact at scalar extremes).
        let lut_sampler = crate::resources::builders::clamp_linear_sampler(device, "lut_sampler");

        // ------------------------------------------------------------------
        // Fallback normal map: 1x1 [128, 128, 255, 255] : flat tangent-space normal
        // ------------------------------------------------------------------
        let fallback_normal_map = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("fallback_normal_map"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fallback_normal_map_view =
            fallback_normal_map.create_view(&crate::gpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // Fallback AO map: 1x1 [255, 255, 255, 255] : no occlusion
        // ------------------------------------------------------------------
        let fallback_ao_map = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("fallback_ao_map"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fallback_ao_map_view =
            fallback_ao_map.create_view(&crate::gpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // Fallback metallic-roughness texture: 1x1 Rgba8Unorm.
        // Content is uninitialized: shader only samples when has_metallic_roughness_tex != 0.
        // ------------------------------------------------------------------
        let fallback_metallic_roughness_texture =
            device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("fallback_metallic_roughness_texture"),
                size: crate::gpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: crate::gpu::TextureFormat::Rgba8Unorm,
                usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                    | crate::gpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
        let fallback_metallic_roughness_texture_view = fallback_metallic_roughness_texture
            .create_view(&crate::gpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // Fallback emissive texture: 1x1 Rgba8Unorm.
        // Content is uninitialized: shader only samples when has_emissive_tex != 0.
        // ------------------------------------------------------------------
        let fallback_emissive_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("fallback_emissive_texture"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fallback_emissive_texture_view =
            fallback_emissive_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        // ------------------------------------------------------------------
        // Fallback texture: 1x1 white RGBA (used when no albedo texture is assigned)
        // ------------------------------------------------------------------
        let fallback_texture = {
            let tex = device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("fallback_texture"),
                size: crate::gpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: crate::gpu::TextureFormat::Rgba8UnormSrgb,
                usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                    | crate::gpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Texture pixels are uploaded lazily on first prepare() via queue.write_texture.
            let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
            // Same config as material_sampler (repeat, linear), so share it.
            let sampler = material_sampler.clone();
            let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("fallback_texture_bg"),
                layout: &texture_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(&view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(&sampler),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::TextureView(
                            &fallback_normal_map_view,
                        ),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: crate::gpu::BindingResource::TextureView(&fallback_ao_map_view),
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
        // Colourmap / LUT fallback resources
        // ------------------------------------------------------------------
        let fallback_lut_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("fallback_lut_texture"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Content of fallback_lut_view is never sampled by the shader when has_attribute=0.
        // Data is intentionally left uninitialised here; it will be a zeroed 1-pixel texture
        // after the GPU zeros it on allocation (implementation-defined but harmless).
        let fallback_lut_view =
            fallback_lut_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

        let fallback_scalar_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("fallback_scalar_buf"),
            size: 4,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(fallback_scalar_buf.slice(..), &[0u8; 4]);
        fallback_scalar_buf.unmap();

        let fallback_face_colour_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("fallback_face_colour_buf"),
            size: 16, // one vec4<f32>
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(fallback_face_colour_buf.slice(..), &[0u8; 16]);
        fallback_face_colour_buf.unmap();

        let fallback_warp_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("fallback_warp_buf"),
            size: 12, // one vec3<f32>
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(fallback_warp_buf.slice(..), &[0u8; 12]);
        fallback_warp_buf.unmap();

        let fallback_position_override_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("fallback_position_override_buf"),
            size: 12, // one vec3<f32>
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            fallback_position_override_buf.slice(..),
            &[0u8; 12],
        );
        fallback_position_override_buf.unmap();

        let fallback_normal_override_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("fallback_normal_override_buf"),
            size: 12,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            fallback_normal_override_buf.slice(..),
            &[0u8; 12],
        );
        fallback_normal_override_buf.unmap();

        // ------------------------------------------------------------------
        // Hardcoded unit cube mesh (test scene object)
        // Created here : after fallback textures : so the combined bind group
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
            &lut_sampler,
            &fallback_lut_view,
            &fallback_scalar_buf,
            &fallback_texture.view,
            &fallback_face_colour_buf,
            &fallback_warp_buf,
            &fallback_position_override_buf,
            &fallback_normal_override_buf,
            &fallback_metallic_roughness_texture_view,
            &fallback_emissive_texture_view,
            &cube_verts,
            &cube_indices,
        );

        // ------------------------------------------------------------------
        // Outline & x-ray pipelines
        // ------------------------------------------------------------------

        // Bind group layout for OutlineUniform (group 1).
        let outline_bgl = crate::resources::builders::uniform_bgl(
            device,
            "outline_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
        );

        let xray_shader = crate::resources::builders::wgsl_module(
            device,
            "xray_shader",
            crate::resources::builders::wgsl_source!("xray"),
        );

        let outline_pl_bgls: Vec<&crate::gpu::BindGroupLayout> = if let Some(d) = deform_bgl {
            vec![&camera_bgl, &outline_bgl, d]
        } else {
            vec![&camera_bgl, &outline_bgl]
        };
        let outline_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "outline_pipeline_layout",
            &outline_pl_bgls,
        );

        // Mask-write pipeline: renders selected objects as r=1.0 to an R8 mask
        // texture with depth testing, replacing the old stencil-based approach.
        let outline_mask_src = if deform_enabled {
            include_str!(concat!(env!("OUT_DIR"), "/outline_mask.wgsl"))
        } else {
            include_str!(concat!(env!("OUT_DIR"), "/outline_mask_noop.wgsl"))
        };
        let outline_mask_shader = crate::resources::builders::wgsl_module(
            device,
            "outline_mask_shader",
            outline_mask_src,
        );
        let outline_masks = crate::resources::mesh::mesh_pipelines::build_outline_mask_pipelines(
            device,
            &outline_pipeline_layout,
            &outline_mask_shader,
            crate::gpu::TextureFormat::R8Unorm,
            pipeline_cache.as_ref(),
        );
        let outline_mask_pipeline = outline_masks.mask;
        let outline_mask_two_sided_pipeline = outline_masks.mask_two_sided;

        // Billboard disc pipeline for the Gaussian splat outline mask pass.
        // Reuses the same pipeline layout as the mesh mask pipelines (camera_bgl + outline_bgl).
        // Positions are instance-stepped vec3; each instance expands to a 6-vertex quad.
        let splat_outline_mask_shader = crate::resources::builders::wgsl_module(
            device,
            "splat_outline_mask_shader",
            crate::resources::builders::wgsl_source!("splat_outline_mask"),
        );
        let splat_outline_pos_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }];
        let splat_outline_pos_layout = crate::gpu::VertexBufferLayout {
            array_stride: 12, // vec3<f32>
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &splat_outline_pos_attrs,
        };
        let splat_outline_size_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: crate::gpu::VertexFormat::Float32,
        }];
        let splat_outline_size_layout = crate::gpu::VertexBufferLayout {
            array_stride: 4, // f32
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &splat_outline_size_attrs,
        };
        let splat_outline_mask_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "splat_outline_mask_pipeline",
                layout: &outline_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &splat_outline_mask_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[splat_outline_pos_layout, splat_outline_size_layout],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &splat_outline_mask_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::R8Unorm,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );

        // Edge-detection pipeline: fullscreen pass that reads the R8 mask and
        // outputs an anti-aliased outline ring to the outline colour texture.
        let outline_edge_shader = crate::resources::builders::wgsl_module(
            device,
            "outline_edge_shader",
            crate::resources::builders::wgsl_source!("outline_edge"),
        );
        let outline_edge_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("outline_edge_bgl"),
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
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let outline_edge_layout = crate::resources::builders::pipeline_layout(
            device,
            "outline_edge_layout",
            &[&outline_edge_bgl],
        );
        let outline_edge_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "outline_edge_pipeline",
                layout: &outline_edge_layout,
                vertex: crate::gpu::VertexState {
                    module: &outline_edge_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &outline_edge_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: crate::gpu::MultisampleState::default(),
                cache: pipeline_cache.as_ref(),
            },
        );

        // X-ray pipeline: render selected objects through all geometry as a semi-transparent tint.
        let xray_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "xray_pipeline",
                layout: &outline_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &xray_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &xray_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: pipeline_cache.as_ref(),
            },
        );

        // Skybox pipeline: fullscreen triangle that samples the equirect environment map.
        let skybox_shader = crate::resources::builders::wgsl_module(
            device,
            "skybox_shader",
            crate::resources::builders::wgsl_source!("skybox"),
        );
        let skybox_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "skybox_pipeline_layout",
            &[&camera_bgl],
        );
        let skybox_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "skybox_pipeline",
                layout: &skybox_pipeline_layout,
                vertex: crate::gpu::VertexState {
                    module: &skybox_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &skybox_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                // Drawn after opaques: only sky pixels (depth == 1.0) pass.
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Equal,
                )),
                multisample: crate::gpu::MultisampleState::default(),
                cache: pipeline_cache.as_ref(),
            },
        );

        mark("misc_pipelines");

        // `deform` is constructed earlier (before the mesh pipeline layout).

        let mut resources = Self {
            target_format,
            sample_count,
            debug_vis_shaders: false,
            pipeline_cache,
            solid_pipeline,
            deform,
            shade_hooks: Vec::new(),
            material_plugins: std::collections::HashMap::new(),
            solid_two_sided_pipeline,
            transparent_pipeline,
            wireframe_pipeline,
            camera_uniform_buf,
            light_uniform_buf,
            light_storage_buf,
            clustered,
            camera_bind_group,
            camera_bind_group_layout: camera_bgl,
            object_bind_group_layout: object_bgl,
            mesh_store: {
                let mut store = crate::resources::mesh::mesh_store::MeshStore::new();
                store.insert(cube_mesh);
                store
            },
            lod_groups: crate::resources::mesh::lod::LodGroupStore::new(),
            shadow_map_texture,
            shadow_map_view,
            shadow_sampler,
            point_shadow_cube_texture,
            point_shadow_cube_view,
            point_shadow_face_views,
            shadow_point_pipeline,
            shadow_point_face_bind_group_layout: shadow_point_face_bgl,
            shadow_point_face_buf,
            shadow_point_face_bind_group,
            shadow_pipeline,
            shadow_pipeline_two_sided,
            shadow_camera_bind_group_layout: shadow_camera_bgl,
            shadow_uniform_buf,
            shadow_bind_group,
            shadow_info_buf,
            shadow_atlas_size: SHADOW_ATLAS_SIZE,
            shadow_atlas_depth_sampler,
            shadow_atlas_viewer_pipeline,
            shadow_atlas_viewer_bg,
            shadow_atlas_viewer_buf,
            debug_frag_sentinel_buf,
            gizmo_pipeline,
            gizmo_vertex_buffer,
            gizmo_index_buffer,
            gizmo_index_count,
            gizmo_uniform_buf,
            gizmo_bind_group,
            gizmo_bind_group_layout: gizmo_bgl,
            overlay_pipeline,
            overlay_line_pipeline,
            grid_pipeline,
            grid_uniform_buf,
            grid_bind_group,
            grid_bind_group_layout: grid_bgl,
            ground_plane_pipeline,
            _ground_plane_bgl: ground_plane_bgl,
            ground_plane_uniform_buf,
            ground_plane_bind_group,
            overlay_bind_group_layout: overlay_bgl,
            constraint_line_buffers: Vec::new(),
            axes_pipeline,
            axes_vertex_buffer,
            axes_vertex_count: 0,
            texture_bind_group_layout: texture_bgl,
            fallback_texture,
            fallback_normal_map,
            fallback_normal_map_view,
            fallback_ao_map,
            fallback_ao_map_view,
            fallback_metallic_roughness_texture,
            fallback_metallic_roughness_texture_view,
            fallback_emissive_texture,
            fallback_emissive_texture_view,
            material_sampler,
            lut_sampler,
            content: crate::resources::types::ContentResources {
                material_bind_groups: std::collections::HashMap::new(),
                textures: crate::resources::material::texture_store::TextureStore::new(),
                polyline_store: crate::resources::PolylineStore::new(),
                streamtube_store: crate::resources::StreamtubeStore::new(),
                tube_store: crate::resources::TubeStore::new(),
                ribbon_store: crate::resources::RibbonStore::new(),
                point_cloud_store: crate::resources::PointCloudStore::new(),
                glyph_set_store: crate::resources::GlyphSetStore::new(),
                tensor_glyph_set_store: crate::resources::TensorGlyphSetStore::new(),
                sprite_set_store: crate::resources::SpriteSetStore::new(),
                sprite_instance_set_store: crate::resources::SpriteInstanceSetStore::new(),
                gaussian_splat_store: crate::resources::types::GaussianSplatStore::new(),
                volume_textures: crate::resources::handle::SlotStore::default(),
                projected_tet_store: crate::resources::handle::SlotStore::default(),
                glyph_atlas: crate::resources::overlay::font::GlyphAtlas::new(device),
                overlay_textures: crate::resources::handle::Registry::default(),
                matcap_textures: Vec::new(),
                matcap_views: Vec::new(),
                matcap_sampler: None,
                fallback_matcap_view: None,
                matcaps_initialized: false,
                builtin_matcap_ids: None,
                colourmap_textures: Vec::new(),
                colourmap_views: Vec::new(),
                colourmaps_cpu: Vec::new(),
                fallback_lut_texture,
                fallback_lut_view,
                fallback_scalar_buf,
                fallback_face_colour_buf,
                fallback_warp_buf,
                fallback_position_override_buf,
                fallback_normal_override_buf,
                builtin_colourmap_ids: None,
                colourmaps_initialized: false,
            },
            jobs: std::sync::Mutex::new(crate::resources::upload_jobs::JobRunner::new()),
            job_results: crate::resources::upload_jobs::JobResults::default(),
            fallback_textures_uploaded: false,
            post: crate::resources::postprocess::PostProcessResources::default(),
            clip_planes_uniform_buf,
            clip_volume_uniform_buf,
            outline: crate::resources::types::OutlineResources {
                bind_group_layout: outline_bgl,
                mask_pipeline: outline_mask_pipeline,
                mask_two_sided_pipeline: outline_mask_two_sided_pipeline,
                edge_pipeline: outline_edge_pipeline,
                edge_bgl: outline_edge_bgl,
                xray_pipeline,
                splat_mask_pipeline: splat_outline_mask_pipeline,
                colour_texture: None,
                colour_view: None,
                depth_texture: None,
                depth_view: None,
                target_size: [0, 0],
                composite_pipeline_single: None,
                composite_pipeline_msaa: None,
                composite_pipeline_hdr: None,
                composite_bgl: None,
                composite_bind_group: None,
                composite_sampler: None,
            },
            instancing: crate::resources::mesh::instancing::InstancingResources::default(),
            cull: crate::resources::mesh::instancing::CullResources::default(),
            lic: crate::resources::postprocess::LicResources::default(),
            hdr_solid_pipeline: None,
            hdr_solid_two_sided_pipeline: None,
            hdr_transparent_pipeline: None,
            hdr_wireframe_pipeline: None,
            hdr_overlay_pipeline: None,
            gaussian_splat:
                crate::resources::scivis::gaussian_splat::GaussianSplatResources::default(),
            sprite: crate::resources::scivis::sprite::SpriteResources::default(),
            point_cloud_pipeline: None,
            point_cloud_bgl: None,
            glyph: crate::resources::scivis::glyph::GlyphResources::default(),
            tensor_glyph: crate::resources::scivis::glyph::TensorGlyphResources::default(),
            volume: crate::resources::volume::volumes::VolumeResources::default(),
            polyline: crate::resources::scivis::polyline::PolylineResources::default(),
            streamtube: crate::resources::scivis::tube::StreamtubeResources::default(),
            ribbon: crate::resources::scivis::tube::RibbonResources::default(),
            image_slice: crate::resources::types::ImageSliceResources::default(),
            compute_filter_pipeline: None,
            compute_filter_bgl: None,
            oit: crate::resources::postprocess::OitResources::default(),
            pt: crate::resources::types::ProjectedTetResources::default(),
            // Scatter-volume (participating media) pipeline (lazily created).
            scatter: crate::resources::volume::scatter_volume::ScatterResources::default(),
            // IBL / environment map resources.
            ibl_irradiance_view: None,
            ibl_prefiltered_view: None,
            ibl_brdf_lut_view: None,
            ibl_sampler,
            ibl_skybox_view: None,
            ibl_fallback_texture,
            ibl_fallback_view,
            ibl_fallback_brdf_texture,
            ibl_fallback_brdf_view,
            ibl_irradiance_texture: None,
            ibl_prefiltered_texture: None,
            ibl_brdf_lut_texture: None,
            ibl_skybox_texture: None,
            skybox_pipeline,
            pick: crate::resources::types::PickResources::default(),
            implicit: crate::resources::types::ImplicitResources::default(),
            mc: crate::resources::volume::gpu_marching_cubes::McResources::default(),

            particle: crate::resources::gpu::gpu_particles::ParticleResources::default(),
            screen_image: crate::resources::types::ScreenImageResources::default(),
            sub_highlight: crate::resources::types::SubHighlightResources::default(),
            overlay_text: crate::resources::overlay::overlay_text::OverlayTextResources::default(),
            overlay_shape: crate::resources::overlay::overlay_shape::OverlayShapeResources::default(
            ),
            backdrop_blur: crate::resources::overlay::overlay_shape::BackdropBlurResources::default(
            ),
            frame_upload_bytes: 0,
            frame_pipelines_built: 0,
            resource_free_epoch: 0,
            occlusion_culling_enabled: false,
            decal: crate::resources::decal::DecalResources::default(),
        };
        // Decal pipelines are built here rather than on the first frame that
        // submits a decal: decals tend to appear mid-session (impact marks,
        // scorches), and a lazy build would stall that frame by the compile
        // cost (~8 ms measured on a desktop GPU).
        resources.ensure_decal_shared(device);
        resources.ensure_decal_pipeline(device);
        mark("decal_pipelines");
        // Pipelines built during construction are load-time cost, not a frame
        // hitch; keep them out of the first frame's stats.
        resources.frame_pipelines_built = 0;
        // GPU skinning is opt-in: hosts call
        // `viewport_lib::plugins::skinning::SkinningPlugin::install(&mut resources, &device)`
        // before uploading any skin data. The renderer otherwise carries no
        // skinning state.
        tracing::info!(
            target: "viewport_lib::init",
            ms = init_start.elapsed().as_secs_f32() * 1000.0,
            "gpu resources init total"
        );
        resources
    }
}
