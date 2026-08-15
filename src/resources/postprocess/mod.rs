//! Shared post-processing pipelines and their per-viewport render targets.
//!
//! Holds [`PostProcessResources`] (FXAA/SSAA, bloom, SSAO, tone-map, DoF,
//! contact shadows, depth blit, and dynamic-resolution upscale) and the
//! `impl DeviceResources` methods that build and drive those passes. The
//! post-effect uniforms live in `uniforms`, the Surface LIC pipelines in
//! `lic`, and the order-independent transparency pipelines in `oit`.

use super::*;

pub(crate) mod lic;
pub(crate) mod oit;
pub(crate) mod uniforms;

pub(crate) use self::lic::LicResources;
pub(crate) use self::oit::OitResources;

/// Shared post-processing pipelines, layouts, samplers, and static textures:
/// FXAA / SSAA resolve, bloom, SSAO, tone-map, DoF, contact shadows, the
/// disabled-pass placeholders, the shared PP samplers, depth blit, and dynamic
/// resolution upscale. All device-shared and lazily built. The viewport-sized
/// intermediate textures and per-frame uniforms live on `ViewportHdrState`.
#[derive(Default)]
pub(crate) struct PostProcessResources {
    // The viewport-sized FXAA texture, view, and bind group now live on
    // `ViewportHdrState`; these slots are still populated but no longer read.
    #[allow(dead_code)]
    pub(crate) fxaa_texture: Option<crate::gpu::Texture>,
    #[allow(dead_code)]
    pub(crate) fxaa_view: Option<crate::gpu::TextureView>,
    pub(crate) fxaa_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) fxaa_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) ssaa_resolve_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) ssaa_resolve_bgl: Option<crate::gpu::BindGroupLayout>,
    #[allow(dead_code)]
    pub(crate) fxaa_bind_group: Option<crate::gpu::BindGroup>,
    pub(crate) fxaa_sampler: Option<crate::gpu::Sampler>,
    pub(crate) bloom_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) ssao_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) ssao_blur_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) tone_map_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) tone_map_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) bloom_threshold_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) bloom_blur_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) ssao_noise_texture: Option<crate::gpu::Texture>,
    pub(crate) ssao_noise_view: Option<crate::gpu::TextureView>,
    pub(crate) ssao_kernel_buf: Option<crate::gpu::Buffer>,
    pub(crate) ssao_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) ssao_blur_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) dof_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) dof_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) contact_shadow_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) contact_shadow_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) bloom_placeholder_view: Option<crate::gpu::TextureView>,
    pub(crate) ao_placeholder_view: Option<crate::gpu::TextureView>,
    pub(crate) cs_placeholder_view: Option<crate::gpu::TextureView>,
    /// 1x1 depth placeholder at 1.0 (uncovered) bound in place of the
    /// foreground depth when the foreground pass did not run.
    pub(crate) foreground_placeholder_view: Option<crate::gpu::TextureView>,
    /// Writes near depth into the output depth buffer where the foreground
    /// depth records coverage, so post-tone-map passes (grid, ground plane)
    /// are occluded by foreground geometry.
    pub(crate) foreground_stamp_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) foreground_stamp_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) pp_linear_sampler: Option<crate::gpu::Sampler>,
    pub(crate) pp_nearest_sampler: Option<crate::gpu::Sampler>,
    pub(crate) depth_blit_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) depth_blit_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) dyn_res_upscale_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) dyn_res_upscale_ds_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) dyn_res_upscale_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) dyn_res_linear_sampler: Option<crate::gpu::Sampler>,
}

impl DeviceResources {
    /// Create or recreate the offscreen outline colour + depth/stencil textures and
    /// the fullscreen composite pipeline used to blit the outline onto the main pass.
    /// No-op if the size hasn't changed and resources already exist.
    #[allow(dead_code)]
    pub(crate) fn ensure_outline_target(&mut self, device: &crate::gpu::Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);

        if self.outline.target_size == [w, h] && self.outline.colour_texture.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        self.outline.target_size = [w, h];

        // Offscreen RGBA colour texture (transparent clear).
        let colour_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("outline_colour_texture"),
            size: crate::gpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: self.target_format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let colour_view = colour_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        // Depth+stencil texture for the stencil outline passes.
        let depth_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("outline_depth_texture"),
            size: crate::gpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth24PlusStencil8,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        // Sampler (linear, clamp-to-edge).
        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "outline_composite_sampler");

        // Bind group layout: texture + sampler.
        let bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "outline_composite_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("outline_composite_bg"),
            layout: &bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&colour_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Fullscreen composite pipeline (alpha blending).
        let shader = crate::resources::builders::wgsl_module(
            device,
            "outline_composite_shader",
            crate::resources::builders::wgsl_source!("outline_composite"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "outline_composite_layout",
            &[&bgl],
        );
        let pipeline_single = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "outline_composite_pipeline_single",
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
                        blend: Some(crate::gpu::BlendState {
                            color: crate::gpu::BlendComponent {
                                src_factor: crate::gpu::BlendFactor::SrcAlpha,
                                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                                operation: crate::gpu::BlendOperation::Add,
                            },
                            alpha: crate::gpu::BlendComponent {
                                src_factor: crate::gpu::BlendFactor::One,
                                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                                operation: crate::gpu::BlendOperation::Add,
                            },
                        }),
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
                multisample: crate::gpu::MultisampleState::default(),
                cache: None,
            },
        );

        let pipeline_msaa = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "outline_composite_pipeline_msaa",
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
                        blend: Some(crate::gpu::BlendState {
                            color: crate::gpu::BlendComponent {
                                src_factor: crate::gpu::BlendFactor::SrcAlpha,
                                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                                operation: crate::gpu::BlendOperation::Add,
                            },
                            alpha: crate::gpu::BlendComponent {
                                src_factor: crate::gpu::BlendFactor::One,
                                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                                operation: crate::gpu::BlendOperation::Add,
                            },
                        }),
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
                    count: self.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
            },
        );

        self.outline.colour_texture = Some(colour_tex);
        self.outline.colour_view = Some(colour_view);
        self.outline.depth_texture = Some(depth_tex);
        self.outline.depth_view = Some(depth_view);
        self.outline.composite_pipeline_single = Some(pipeline_single);
        self.outline.composite_pipeline_msaa = Some(pipeline_msaa);
        self.outline.composite_bgl = Some(bgl);
        self.outline.composite_bind_group = Some(bg);
        self.outline.composite_sampler = Some(sampler);
    }

    // -----------------------------------------------------------------------
    // Per-viewport HDR state : shared infrastructure
    // -----------------------------------------------------------------------

    /// Create all shared HDR/post-process infrastructure (BGLs, pipelines,
    /// samplers, placeholder textures, SSAO noise/kernel) on `self`.
    /// No-op after the first call. Must be called before `create_hdr_viewport_state`.
    pub(crate) fn ensure_hdr_shared(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_format: crate::gpu::TextureFormat,
    ) {
        // Guard: if all three sentinel fields exist, everything is created.
        if self.post.tone_map_pipeline.is_some()
            && self.post.bloom_bgl.is_some()
            && self.post.fxaa_sampler.is_some()
        {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // --- Fallback textures (one-time uploads) ---
        if !self.fallback_textures_uploaded {
            let upload = |tex: &crate::gpu::Texture, data: &[u8]| {
                queue.write_texture(
                    crate::gpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: crate::gpu::Origin3d::ZERO,
                        aspect: crate::gpu::TextureAspect::All,
                    },
                    data,
                    crate::gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4),
                        rows_per_image: Some(1),
                    },
                    crate::gpu::Extent3d {
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
        if self.post.bloom_placeholder_view.is_none() {
            let make_placeholder = |device: &crate::gpu::Device,
                                    queue: &crate::gpu::Queue,
                                    label: &str,
                                    format: crate::gpu::TextureFormat,
                                    data: &[u8],
                                    bytes_per_row: u32|
             -> (crate::gpu::Texture, crate::gpu::TextureView) {
                let tex = device.create_texture(&crate::gpu::TextureDescriptor {
                    label: Some(label),
                    size: crate::gpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: crate::gpu::TextureDimension::D2,
                    format,
                    usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                        | crate::gpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    crate::gpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: 0,
                        origin: crate::gpu::Origin3d::ZERO,
                        aspect: crate::gpu::TextureAspect::All,
                    },
                    data,
                    crate::gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(1),
                    },
                    crate::gpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
                (tex, view)
            };

            let (_bt, bv) = make_placeholder(
                device,
                queue,
                "bloom_placeholder",
                crate::gpu::TextureFormat::Rgba16Float,
                &[0u8; 8],
                8,
            );
            self.post.bloom_placeholder_view = Some(bv);

            let (_at, av) = make_placeholder(
                device,
                queue,
                "ao_placeholder",
                crate::gpu::TextureFormat::R8Unorm,
                &[255u8],
                1,
            );
            self.post.ao_placeholder_view = Some(av);

            let (_ct, cv) = make_placeholder(
                device,
                queue,
                "cs_placeholder",
                crate::gpu::TextureFormat::R8Unorm,
                &[255u8],
                1,
            );
            self.post.cs_placeholder_view = Some(cv);

            // LIC placeholder: 1x1 R8Unorm, 128 = 0.5 -> lic_factor = 1.0 (no modulation).
            let (_lt, lv) = make_placeholder(
                device,
                queue,
                "lic_placeholder",
                crate::gpu::TextureFormat::R8Unorm,
                &[128u8],
                1,
            );
            self.lic.placeholder_view = Some(lv);

            // Foreground depth placeholder: 1x1 depth at 1.0 = no coverage.
            // Depth16Unorm is the only depth format write_texture accepts,
            // and it binds to texture_depth_2d like any other depth format.
            let (_ft, fv) = make_placeholder(
                device,
                queue,
                "foreground_depth_placeholder",
                crate::gpu::TextureFormat::Depth16Unorm,
                &[0xFFu8, 0xFF],
                2,
            );
            self.post.foreground_placeholder_view = Some(fv);
        }

        // --- SSAO noise (one-time) ---
        if self.post.ssao_noise_view.is_none() {
            let noise_data: Vec<u8> = (0..16)
                .flat_map(|i| {
                    let angle = (i as f32 / 16.0) * std::f32::consts::TAU;
                    let x = ((angle.cos() * 0.5 + 0.5) * 255.0) as u8;
                    let y = ((angle.sin() * 0.5 + 0.5) * 255.0) as u8;
                    [x, y, 128u8, 255u8]
                })
                .collect();
            let noise_tex = device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("ssao_noise"),
                size: crate::gpu::Extent3d {
                    width: 4,
                    height: 4,
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
            queue.write_texture(
                crate::gpu::TexelCopyTextureInfo {
                    texture: &noise_tex,
                    mip_level: 0,
                    origin: crate::gpu::Origin3d::ZERO,
                    aspect: crate::gpu::TextureAspect::All,
                },
                &noise_data,
                crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * 4),
                    rows_per_image: Some(4),
                },
                crate::gpu::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
            );
            self.post.ssao_noise_view =
                Some(noise_tex.create_view(&crate::gpu::TextureViewDescriptor::default()));
            self.post.ssao_noise_texture = Some(noise_tex);
        }

        // --- SSAO kernel (one-time) ---
        if self.post.ssao_kernel_buf.is_none() {
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
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("ssao_kernel_buf"),
                size: kernel_bytes.len() as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, kernel_bytes);
            self.post.ssao_kernel_buf = Some(buf);
        }

        // --- Shared samplers ---
        let linear_sampler =
            crate::resources::builders::clamp_linear_sampler(device, "pp_linear_sampler");
        let nearest_sampler =
            crate::resources::builders::clamp_nearest_sampler(device, "pp_nearest_sampler");
        let fxaa_sampler = crate::resources::builders::clamp_linear_sampler(device, "fxaa_sampler");
        let oit_sampler =
            crate::resources::builders::clamp_linear_sampler(device, "oit_composite_sampler");
        let outline_sampler =
            crate::resources::builders::clamp_linear_sampler(device, "outline_composite_sampler");

        // --- Bind group layouts ---
        let tone_map_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("tone_map_bgl"),
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
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Depth,
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 7: LIC intensity texture (R8Unorm). Placeholder when LIC is disabled.
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
                    // binding 8: foreground depth (coverage mask). Placeholder
                    // when the foreground pass did not run.
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Depth,
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let bloom_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("bloom_bgl"),
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
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
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

        let ssao_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(
                        crate::gpu::SamplerBindingType::NonFiltering,
                    ),
                    count: None,
                },
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
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        });

        let ssao_blur_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "ssao_blur_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let cs_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("contact_shadow_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(
                        crate::gpu::SamplerBindingType::NonFiltering,
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

        let fxaa_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "fxaa_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let oit_composite_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("oit_composite_bgl"),
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
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        let outline_composite_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "outline_composite_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        // Fullscreen pass helper: build the single-bgl layout, then the pipeline.
        let make_fs_pipeline = |label: &str,
                                shader: crate::gpu::ShaderModule,
                                bgl: &crate::gpu::BindGroupLayout,
                                fmt: crate::gpu::TextureFormat|
         -> crate::gpu::RenderPipeline {
            let layout = crate::resources::builders::pipeline_layout(
                device,
                format!("{label}_layout").as_str(),
                &[bgl],
            );
            crate::resources::builders::build_fullscreen_pipeline(
                device, label, &layout, &shader, fmt, None,
            )
        };

        // Tone map pipeline
        let tone_map_shader = crate::resources::builders::wgsl_module(
            device,
            "tone_map_shader",
            crate::resources::builders::wgsl_source!("tone_map"),
        );
        let tone_map_pipeline = make_fs_pipeline(
            "tone_map_pipeline",
            tone_map_shader,
            &tone_map_bgl,
            output_format,
        );

        // Bloom pipelines
        let bloom_threshold_shader = crate::resources::builders::wgsl_module(
            device,
            "bloom_threshold_shader",
            crate::resources::builders::wgsl_source!("bloom_threshold"),
        );
        let bloom_threshold_pipeline = make_fs_pipeline(
            "bloom_threshold_pipeline",
            bloom_threshold_shader,
            &bloom_bgl,
            crate::gpu::TextureFormat::Rgba16Float,
        );
        let bloom_blur_shader = crate::resources::builders::wgsl_module(
            device,
            "bloom_blur_shader",
            crate::resources::builders::wgsl_source!("bloom_blur"),
        );
        let bloom_blur_pipeline = make_fs_pipeline(
            "bloom_blur_pipeline",
            bloom_blur_shader,
            &bloom_bgl,
            crate::gpu::TextureFormat::Rgba16Float,
        );

        // SSAO pipelines
        let ssao_shader = crate::resources::builders::wgsl_module(
            device,
            "ssao_shader",
            crate::resources::builders::wgsl_source!("ssao"),
        );
        let ssao_pipeline = make_fs_pipeline(
            "ssao_pipeline",
            ssao_shader,
            &ssao_bgl,
            crate::gpu::TextureFormat::R8Unorm,
        );
        let ssao_blur_shader = crate::resources::builders::wgsl_module(
            device,
            "ssao_blur_shader",
            crate::resources::builders::wgsl_source!("ssao_blur"),
        );
        let ssao_blur_pipeline = make_fs_pipeline(
            "ssao_blur_pipeline",
            ssao_blur_shader,
            &ssao_blur_bgl,
            crate::gpu::TextureFormat::R8Unorm,
        );

        // Contact shadow pipeline
        let cs_shader = crate::resources::builders::wgsl_module(
            device,
            "contact_shadow_shader",
            crate::resources::builders::wgsl_source!("contact_shadow"),
        );
        let cs_pipeline = make_fs_pipeline(
            "contact_shadow_pipeline",
            cs_shader,
            &cs_bgl,
            crate::gpu::TextureFormat::R8Unorm,
        );

        // FXAA pipeline
        let fxaa_shader = crate::resources::builders::wgsl_module(
            device,
            "fxaa_shader",
            crate::resources::builders::wgsl_source!("fxaa"),
        );
        let fxaa_pipeline =
            make_fs_pipeline("fxaa_pipeline", fxaa_shader, &fxaa_bgl, output_format);

        // OIT composite pipeline
        let oit_comp_shader = crate::resources::builders::wgsl_module(
            device,
            "oit_composite_shader",
            crate::resources::builders::wgsl_source!("oit_composite"),
        );
        let premul_blend = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                operation: crate::gpu::BlendOperation::Add,
            },
        };
        let oit_comp_layout = crate::resources::builders::pipeline_layout(
            device,
            "oit_composite_pipeline_layout",
            &[&oit_composite_bgl],
        );
        let oit_composite_pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "oit_composite_pipeline",
            &oit_comp_layout,
            &oit_comp_shader,
            crate::gpu::TextureFormat::Rgba16Float,
            Some(premul_blend),
        );

        // OIT mesh pipeline. Two color targets (`Rgba16Float` accumulation +
        // `R8Unorm` reveal) and depth-test-only.
        let oit_mesh_source = {
            let base = if self.deform.enabled {
                include_str!(concat!(env!("OUT_DIR"), "/mesh_oit.wgsl"))
            } else {
                include_str!(concat!(env!("OUT_DIR"), "/mesh_oit_noop.wgsl"))
            };
            crate::resources::mesh_sidecar::registry::compose_shader(
                base,
                &self.deform.registrations,
            )
        };
        let oit_shader = crate::resources::builders::wgsl_module(
            device,
            "mesh_oit_shader",
            crate::resources::builders::builtin_hook_env(
                crate::resources::builders::strip_debug_vis(
                    oit_mesh_source,
                    self.debug_vis_shaders,
                ),
            ),
        );
        let oit_layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            "oit_pipeline_layout",
            &self.camera_bind_group_layout,
            &self.object_bind_group_layout,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        let oit_pipeline = crate::resources::mesh::mesh_pipelines::build_oit_pipeline(
            device,
            &oit_layout,
            &oit_shader,
        );

        // oit_instanced_pipeline is created lazily by ensure_oit_instanced_pipeline()
        // once instance_bind_group_layout becomes available. Splitting it out avoids the
        // empty-scene-on-frame-1 trap where this ensure_hdr_shared early-returns before
        // the BGL exists and never re-runs.

        // HDR scene pipelines. Compose the shader with currently registered
        // deformers so any host or in-crate registration (skinning, wind,
        // etc.) is picked up the first time HDR is enabled. Without this
        // the HDR mesh pipeline would use the identity hook bodies even
        // though the LDR pipeline was rebuilt at registration time.
        let hdr_mesh_source = {
            let base = if self.deform.enabled {
                include_str!(concat!(env!("OUT_DIR"), "/mesh.wgsl"))
            } else {
                include_str!(concat!(env!("OUT_DIR"), "/mesh_noop.wgsl"))
            };
            crate::resources::mesh_sidecar::registry::compose_shader(
                base,
                &self.deform.registrations,
            )
        };
        // Final composed HDR source, materialized so the discard-free twin can be
        // stripped from the exact same source the normal module compiles.
        let hdr_final_src = crate::resources::builders::builtin_hook_env(
            crate::resources::builders::strip_debug_vis(hdr_mesh_source, self.debug_vis_shaders),
        )
        .into_owned();
        let hdr_shader = crate::resources::builders::wgsl_module(
            device,
            "mesh_shader_hdr",
            hdr_final_src.clone(),
        );
        // Early-Z twin: identical shading with every `discard;` removed, valid
        // only for draws that would not have discarded (see the per-object gate
        // in hdr_path.rs).
        let hdr_shader_nodiscard = crate::resources::builders::wgsl_module(
            device,
            "mesh_shader_hdr_nodiscard",
            crate::resources::builders::strip_discards(&hdr_final_src),
        );
        let hdr_depth_stencil = crate::resources::builders::scene_depth_stencil(
            true,
            crate::gpu::CompareFunction::Less,
        );
        let hdr_pipeline_layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            "hdr_mesh_pipeline_layout",
            &self.camera_bind_group_layout,
            &self.object_bind_group_layout,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        let hdr = crate::resources::mesh::mesh_pipelines::build_hdr_mesh_pipelines(
            device,
            &hdr_pipeline_layout,
            &hdr_shader,
        );
        let hdr_solid_pipeline = hdr.solid;
        let hdr_solid_two_sided_pipeline = hdr.solid_two_sided;
        let hdr_transparent_pipeline = hdr.transparent;
        let hdr_wireframe_pipeline = hdr.wireframe;

        // Discard-free solid twins for the early-Z fast path. Only .solid and
        // .solid_two_sided are used; the transparent/wireframe twins are
        // discarded (transparency and wireframe do not benefit from early-Z).
        let hdr_nd = crate::resources::mesh::mesh_pipelines::build_hdr_mesh_pipelines(
            device,
            &hdr_pipeline_layout,
            &hdr_shader_nodiscard,
        );
        let hdr_solid_nodiscard_pipeline = hdr_nd.solid;
        let hdr_solid_two_sided_nodiscard_pipeline = hdr_nd.solid_two_sided;

        let hdr_overlay_shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_shader_hdr",
            crate::resources::builders::wgsl_source!("overlay"),
        );
        let hdr_overlay_layout = crate::resources::builders::pipeline_layout(
            device,
            "hdr_overlay_pipeline_layout",
            &[
                &self.camera_bind_group_layout,
                &self.overlay_bind_group_layout,
            ],
        );
        let hdr_overlay_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "hdr_overlay_pipeline",
                layout: &hdr_overlay_layout,
                vertex: crate::gpu::VertexState {
                    module: &hdr_overlay_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[OverlayVertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &hdr_overlay_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::Rgba16Float,
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
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        // Outline composite pipelines
        let outline_comp_shader = crate::resources::builders::wgsl_module(
            device,
            "outline_composite_shader",
            crate::resources::builders::wgsl_source!("outline_composite"),
        );
        let outline_comp_blend = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::SrcAlpha,
                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                operation: crate::gpu::BlendOperation::Add,
            },
        };
        let outline_comp_ds = crate::resources::builders::scene_depth_stencil(
            false,
            crate::gpu::CompareFunction::Always,
        );
        let outline_comp_layout = crate::resources::builders::pipeline_layout(
            device,
            "outline_composite_layout",
            &[&outline_composite_bgl],
        );
        let make_outline_pipeline =
            |label: &str, fmt: crate::gpu::TextureFormat, sample_count: u32| {
                crate::resources::builders::render_pipeline(
                    device,
                    crate::resources::builders::RenderPipelineDesc {
                        label,
                        layout: &outline_comp_layout,
                        vertex: crate::gpu::VertexState {
                            module: &outline_comp_shader,
                            entry_point: Some("vs_main"),
                            buffers: &[],
                            compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        },
                        fragment: Some(crate::gpu::FragmentState {
                            module: &outline_comp_shader,
                            entry_point: Some("fs_main"),
                            targets: &[Some(crate::gpu::ColorTargetState {
                                format: fmt,
                                blend: Some(outline_comp_blend),
                                write_mask: crate::gpu::ColorWrites::ALL,
                            })],
                            compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        }),
                        primitive: crate::gpu::PrimitiveState {
                            topology: crate::gpu::PrimitiveTopology::TriangleList,
                            cull_mode: None,
                            ..Default::default()
                        },
                        depth_stencil: Some(outline_comp_ds.clone()),
                        multisample: crate::gpu::MultisampleState {
                            count: sample_count,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        cache: None,
                    },
                )
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
            crate::gpu::TextureFormat::Rgba16Float,
            1,
        );

        // Store everything
        self.post.pp_linear_sampler = Some(linear_sampler);
        self.post.pp_nearest_sampler = Some(nearest_sampler);
        self.post.fxaa_sampler = Some(fxaa_sampler);
        self.oit.composite_sampler = Some(oit_sampler);
        self.outline.composite_sampler = Some(outline_sampler);

        self.post.tone_map_bgl = Some(tone_map_bgl);
        self.post.bloom_bgl = Some(bloom_bgl);
        self.post.ssao_bgl = Some(ssao_bgl);
        self.post.ssao_blur_bgl = Some(ssao_blur_bgl);
        self.post.contact_shadow_bgl = Some(cs_bgl);
        self.post.fxaa_bgl = Some(fxaa_bgl);
        self.oit.composite_bgl = Some(oit_composite_bgl);
        self.outline.composite_bgl = Some(outline_composite_bgl);

        self.post.tone_map_pipeline = Some(tone_map_pipeline);
        self.post.bloom_threshold_pipeline = Some(bloom_threshold_pipeline);
        self.post.bloom_blur_pipeline = Some(bloom_blur_pipeline);
        self.post.ssao_pipeline = Some(ssao_pipeline);
        self.post.ssao_blur_pipeline = Some(ssao_blur_pipeline);
        self.post.contact_shadow_pipeline = Some(cs_pipeline);
        self.post.fxaa_pipeline = Some(fxaa_pipeline);

        // --- SSAA resolve pipeline ---
        let ssaa_resolve_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("ssaa_resolve_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::NonFiltering,
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
        let ssaa_resolve_shader = crate::resources::builders::wgsl_module(
            device,
            "ssaa_resolve_shader",
            crate::resources::builders::wgsl_source!("ssaa_resolve"),
        );
        let ssaa_resolve_layout = crate::resources::builders::pipeline_layout(
            device,
            "ssaa_resolve_layout",
            &[&ssaa_resolve_bgl],
        );
        let ssaa_resolve_pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "ssaa_resolve_pipeline",
            &ssaa_resolve_layout,
            &ssaa_resolve_shader,
            crate::gpu::TextureFormat::Rgba16Float,
            None,
        );
        self.post.ssaa_resolve_bgl = Some(ssaa_resolve_bgl);
        self.post.ssaa_resolve_pipeline = Some(ssaa_resolve_pipeline);

        // DoF pipeline
        let dof_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("dof_bgl"),
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
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
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
                // binding 4: foreground depth (coverage mask). Placeholder
                // when the foreground pass did not run.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let dof_shader = crate::resources::builders::wgsl_module(
            device,
            "dof_shader",
            crate::resources::builders::wgsl_source!("dof"),
        );
        let dof_pipeline = make_fs_pipeline(
            "dof_pipeline",
            dof_shader,
            &dof_bgl,
            crate::gpu::TextureFormat::Rgba16Float,
        );
        self.post.dof_bgl = Some(dof_bgl);
        self.post.dof_pipeline = Some(dof_pipeline);

        self.oit.pipeline = Some(oit_pipeline);
        self.oit.composite_pipeline = Some(oit_composite_pipeline);
        self.hdr_solid_pipeline = Some(hdr_solid_pipeline);
        self.hdr_solid_two_sided_pipeline = Some(hdr_solid_two_sided_pipeline);
        self.hdr_solid_nodiscard_pipeline = Some(hdr_solid_nodiscard_pipeline);
        self.hdr_solid_two_sided_nodiscard_pipeline = Some(hdr_solid_two_sided_nodiscard_pipeline);
        self.hdr_transparent_pipeline = Some(hdr_transparent_pipeline);
        self.hdr_wireframe_pipeline = Some(hdr_wireframe_pipeline);
        self.hdr_overlay_pipeline = Some(hdr_overlay_pipeline);
        self.outline.composite_pipeline_single = Some(outline_composite_pipeline_single);
        self.outline.composite_pipeline_msaa = Some(outline_composite_pipeline_msaa);
        self.outline.composite_pipeline_hdr = Some(outline_composite_pipeline_hdr);

        let _ = hdr_depth_stencil; // used in make_hdr_mesh closure above

        // --- Surface LIC shared resources ---
        if self.lic.noise_sampler.is_none() {
            // Bilinear sampler used for lic_vector_texture in the advect pass.
            let samp =
                crate::resources::builders::clamp_linear_sampler(device, "lic_linear_sampler");
            self.lic.noise_sampler = Some(samp);
        }

        // LIC surface BGL (group 1): object uniform only.
        // Flow vectors are passed as vertex buffer 1 (not a storage binding).
        if self.lic.surface_bgl.is_none() {
            let bgl = crate::resources::builders::uniform_bgl(
                device,
                "lic_surface_bgl",
                crate::gpu::ShaderStages::VERTEX_FRAGMENT,
            );
            self.lic.surface_bgl = Some(bgl);
        }

        // LIC advect BGL (fullscreen): params uniform, vector tex, noise tex, sampler x2.
        if self.lic.advect_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("lic_advect_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
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
                            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
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
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });
            self.lic.advect_bgl = Some(bgl);
        }

        // LIC surface pipeline: renders mesh into Rgba8Unorm lic_vector_texture.
        // Group 0 = camera_bind_group_layout (already on self), group 1 = lic_surface_bgl.
        if self.lic.surface_pipeline.is_none() {
            if let Some(surface_bgl) = self.lic.surface_bgl.as_ref() {
                let shader = crate::resources::builders::wgsl_module(
                    device,
                    "lic_surface_shader",
                    crate::resources::builders::wgsl_source!("lic_surface"),
                );
                let layout = crate::resources::builders::pipeline_layout(
                    device,
                    "lic_surface_layout",
                    &[&self.camera_bind_group_layout, surface_bgl],
                );
                // Vertex buffer 0: full Vertex stride, position at location 0.
                let lic_vertex_layout = crate::gpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as crate::gpu::BufferAddress,
                    step_mode: crate::gpu::VertexStepMode::Vertex,
                    attributes: &[crate::gpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: crate::gpu::VertexFormat::Float32x3,
                    }],
                };
                // Vertex buffer 1: tightly-packed [f32;3] flow vectors at location 1.
                let lic_flow_layout = crate::gpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: crate::gpu::VertexStepMode::Vertex,
                    attributes: &[crate::gpu::VertexAttribute {
                        offset: 0,
                        shader_location: 1,
                        format: crate::gpu::VertexFormat::Float32x3,
                    }],
                };
                let pipeline = crate::resources::builders::render_pipeline(
                    device,
                    crate::resources::builders::RenderPipelineDesc {
                        label: "lic_surface_pipeline",
                        layout: &layout,
                        vertex: crate::gpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs_main"),
                            buffers: &[lic_vertex_layout, lic_flow_layout],
                            compilation_options: Default::default(),
                        },
                        fragment: Some(crate::gpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs_main"),
                            targets: &[Some(crate::gpu::ColorTargetState {
                                format: crate::gpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: crate::gpu::ColorWrites::ALL,
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: crate::gpu::PrimitiveState {
                            topology: crate::gpu::PrimitiveTopology::TriangleList,
                            cull_mode: None,
                            ..Default::default()
                        },
                        depth_stencil: None,
                        multisample: crate::gpu::MultisampleState::default(),
                        cache: None,
                    },
                );
                self.lic.surface_pipeline = Some(pipeline);
            }
        }

        // LIC advect pipeline: fullscreen render into R8Unorm lic_output_texture.
        if self.lic.advect_pipeline.is_none() {
            if let Some(advect_bgl) = &self.lic.advect_bgl {
                let shader = crate::resources::builders::wgsl_module(
                    device,
                    "lic_advect_shader",
                    crate::resources::builders::wgsl_source!("lic_advect"),
                );
                let layout = crate::resources::builders::pipeline_layout(
                    device,
                    "lic_advect_layout",
                    &[advect_bgl],
                );
                let pipeline = crate::resources::builders::build_fullscreen_pipeline(
                    device,
                    "lic_advect_pipeline",
                    &layout,
                    &shader,
                    crate::gpu::TextureFormat::R8Unorm,
                    None,
                );
                self.lic.advect_pipeline = Some(pipeline);
            }
        }

        // --- Depth blit pipeline (lazily created once) ---
        // Copies a scene-resolution depth texture to a native-resolution depth-only target.
        // Used when render_scale < 1.0 so post-tone-map passes can use a native-res depth buf.
        if self.post.depth_blit_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("depth_blit_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            let shader = crate::resources::builders::wgsl_module(
                device,
                "depth_blit_shader",
                crate::resources::builders::wgsl_source!("depth_blit"),
            );
            let layout =
                crate::resources::builders::pipeline_layout(device, "depth_blit_layout", &[&bgl]);
            let pipeline = crate::resources::builders::render_pipeline(
                device,
                crate::resources::builders::RenderPipelineDesc {
                    label: "depth_blit_pipeline",
                    layout: &layout,
                    vertex: crate::gpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(crate::gpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[],
                        compilation_options: Default::default(),
                    }),
                    primitive: crate::gpu::PrimitiveState {
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                        true,
                        crate::gpu::CompareFunction::Always,
                    )),
                    multisample: crate::gpu::MultisampleState::default(),
                    cache: None,
                },
            );
            self.post.depth_blit_bgl = Some(bgl);
            self.post.depth_blit_pipeline = Some(pipeline);
        }

        // --- Foreground depth stamp pipeline (lazily created once) ---
        // Writes near depth into the output depth buffer where the foreground
        // pass drew, so post-tone-map passes are occluded by foreground items.
        if self.post.foreground_stamp_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("foreground_stamp_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            let shader = crate::resources::builders::wgsl_module(
                device,
                "foreground_stamp_shader",
                crate::resources::builders::wgsl_source!("foreground_depth_stamp"),
            );
            let layout = crate::resources::builders::pipeline_layout(
                device,
                "foreground_stamp_layout",
                &[&bgl],
            );
            let pipeline = crate::resources::builders::render_pipeline(
                device,
                crate::resources::builders::RenderPipelineDesc {
                    label: "foreground_stamp_pipeline",
                    layout: &layout,
                    vertex: crate::gpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(crate::gpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[],
                        compilation_options: Default::default(),
                    }),
                    primitive: crate::gpu::PrimitiveState {
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                        true,
                        crate::gpu::CompareFunction::Always,
                    )),
                    multisample: crate::gpu::MultisampleState::default(),
                    cache: None,
                },
            );
            self.post.foreground_stamp_bgl = Some(bgl);
            self.post.foreground_stamp_pipeline = Some(pipeline);
        }

        // --- Decal shared resources (D1) ---
        self.ensure_decal_shared(device);
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
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_format: crate::gpu::TextureFormat,
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
                        fmt: crate::gpu::TextureFormat,
                        tw: u32,
                        th: u32,
                        extra_usage: crate::gpu::TextureUsages| {
            device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some(label),
                size: crate::gpu::Extent3d {
                    width: tw,
                    height: th,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: fmt,
                usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                    | crate::gpu::TextureUsages::TEXTURE_BINDING
                    | extra_usage,
                view_formats: &[],
            })
        };

        // HDR scene colour and depth -- at scene resolution (render_scale * output).
        // COPY_SRC enables the refractive sprite pass to copy the resolved
        // scene colour into its sample texture before drawing distortion.
        let hdr_tex = make_tex(
            "hdr_texture",
            crate::gpu::TextureFormat::Rgba16Float,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::COPY_SRC,
        );
        let hdr_view = hdr_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let hdr_depth_tex = make_tex(
            "hdr_depth_texture",
            crate::gpu::TextureFormat::Depth24PlusStencil8,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::empty(),
        );
        let hdr_depth_view =
            hdr_depth_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let hdr_depth_only_view = hdr_depth_tex.create_view(&crate::gpu::TextureViewDescriptor {
            aspect: crate::gpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let hdr_stencil_only_view = hdr_depth_tex.create_view(&crate::gpu::TextureViewDescriptor {
            aspect: crate::gpu::TextureAspect::StencilOnly,
            ..Default::default()
        });

        // Bloom -- at scene resolution (hw/hh are scene_w/2, scene_h/2).
        let bloom_threshold_tex = make_tex(
            "bloom_threshold_texture",
            crate::gpu::TextureFormat::Rgba16Float,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::empty(),
        );
        let bloom_threshold_view =
            bloom_threshold_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let bloom_ping_tex = make_tex(
            "bloom_ping_texture",
            crate::gpu::TextureFormat::Rgba16Float,
            hw,
            hh,
            crate::gpu::TextureUsages::empty(),
        );
        let bloom_ping_view =
            bloom_ping_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let bloom_pong_tex = make_tex(
            "bloom_pong_texture",
            crate::gpu::TextureFormat::Rgba16Float,
            hw,
            hh,
            crate::gpu::TextureUsages::empty(),
        );
        let bloom_pong_view =
            bloom_pong_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        // SSAO -- at scene resolution.
        let ssao_tex = make_tex(
            "ssao_texture",
            crate::gpu::TextureFormat::R8Unorm,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::empty(),
        );
        let ssao_view = ssao_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let ssao_blur_tex = make_tex(
            "ssao_blur_texture",
            crate::gpu::TextureFormat::R8Unorm,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::empty(),
        );
        let ssao_blur_view =
            ssao_blur_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        // Depth of field -- at scene resolution.
        let dof_tex = make_tex(
            "dof_texture",
            crate::gpu::TextureFormat::Rgba16Float,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::empty(),
        );
        let dof_view = dof_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        // Contact shadow -- at scene resolution.
        let cs_tex = make_tex(
            "contact_shadow_texture",
            crate::gpu::TextureFormat::R8Unorm,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::empty(),
        );
        let cs_view = cs_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        // FXAA -- at scene resolution so the whole post-process chain runs at
        // the scaled size when render_scale < 1.0.
        let fxaa_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("fxaa_texture"),
            size: crate::gpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: output_format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fxaa_view = fxaa_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        // Outline offscreen : mask (R8), colour (target_format), and depth -- at scene resolution.
        let outline_mask_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("outline_mask_texture"),
            size: crate::gpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R8Unorm,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let outline_mask_view =
            outline_mask_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let outline_colour_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("outline_colour_texture"),
            size: crate::gpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: self.target_format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let outline_colour_view =
            outline_colour_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let outline_depth_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("outline_depth_texture"),
            size: crate::gpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth24PlusStencil8,
            // TEXTURE_BINDING so the HiZ occlusion prev-depth copy can sample the
            // LDR scene depth (the LDR path renders into this target).
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let outline_depth_view =
            outline_depth_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let outline_depth_only_view =
            outline_depth_tex.create_view(&crate::gpu::TextureViewDescriptor {
                aspect: crate::gpu::TextureAspect::DepthOnly,
                ..Default::default()
            });

        // Uniform buffers
        let tone_map_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tone_map_uniform_buf"),
            size: std::mem::size_of::<ToneMapUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bloom_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("bloom_uniform_buf"),
            size: std::mem::size_of::<BloomUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bloom_h_uniform_buf = {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("bloom_h_uniform_buf"),
                size: std::mem::size_of::<BloomUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &buf,
                0,
                bytemuck::cast_slice(&[BloomUniform {
                    threshold: 0.0,
                    intensity: 0.0,
                    horizontal: 1,
                    max_brightness: 0.0,
                }]),
            );
            buf
        };
        let bloom_v_uniform_buf = {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("bloom_v_uniform_buf"),
                size: std::mem::size_of::<BloomUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &buf,
                0,
                bytemuck::cast_slice(&[BloomUniform {
                    threshold: 0.0,
                    intensity: 0.0,
                    horizontal: 0,
                    max_brightness: 0.0,
                }]),
            );
            buf
        };
        let ssao_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("ssao_uniform_buf"),
            size: std::mem::size_of::<SsaoUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cs_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("contact_shadow_uniform_buf"),
            size: std::mem::size_of::<ContactShadowUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dof_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("dof_uniform_buf"),
            size: std::mem::size_of::<DofUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shared references needed for bind groups
        let linear_sampler = self
            .post
            .pp_linear_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let nearest_sampler = self
            .post
            .pp_nearest_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let fxaa_sampler = self
            .post
            .fxaa_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let oit_sampler = self
            .oit
            .composite_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let outline_sampler = self
            .outline
            .composite_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let bloom_placeholder_view = self
            .post
            .bloom_placeholder_view
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ao_placeholder_view = self
            .post
            .ao_placeholder_view
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let cs_placeholder_view = self
            .post
            .cs_placeholder_view
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ssao_noise_view = self
            .post
            .ssao_noise_view
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ssao_kernel_buf = self
            .post
            .ssao_kernel_buf
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let tone_map_bgl = self
            .post
            .tone_map_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let bloom_bgl = self
            .post
            .bloom_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ssao_bgl = self
            .post
            .ssao_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let ssao_blur_bgl = self
            .post
            .ssao_blur_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let cs_bgl = self
            .post
            .contact_shadow_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let fxaa_bgl = self
            .post
            .fxaa_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let oit_composite_bgl = self
            .oit
            .composite_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let outline_composite_bgl = self
            .outline
            .composite_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");

        // Bind groups
        let tone_map_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tone_map_bg"),
            layout: tone_map_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: tone_map_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(bloom_placeholder_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(ao_placeholder_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::TextureView(cs_placeholder_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: crate::gpu::BindingResource::TextureView(
                        self.lic
                            .placeholder_view
                            .as_ref()
                            .expect("ensure_hdr_shared not called"),
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: crate::gpu::BindingResource::TextureView(
                        self.post
                            .foreground_placeholder_view
                            .as_ref()
                            .expect("ensure_hdr_shared not called"),
                    ),
                },
            ],
        });
        let bloom_threshold_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("bloom_threshold_bg"),
            layout: bloom_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let bloom_blur_h_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("bloom_blur_h_bg"),
            layout: bloom_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&bloom_threshold_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_h_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let bloom_blur_v_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("bloom_blur_v_bg"),
            layout: bloom_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&bloom_ping_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_v_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let bloom_blur_h_pong_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("bloom_blur_h_pong_bg"),
            layout: bloom_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&bloom_pong_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: bloom_h_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let ssao_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("ssao_bg"),
            layout: ssao_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(nearest_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(ssao_noise_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: ssao_kernel_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: ssao_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let ssao_blur_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("ssao_blur_bg"),
            layout: ssao_blur_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&ssao_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
            ],
        });
        let dof_bgl = self
            .post
            .dof_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let dof_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("dof_bg"),
            layout: dof_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: dof_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(
                        self.post
                            .foreground_placeholder_view
                            .as_ref()
                            .expect("ensure_hdr_shared not called"),
                    ),
                },
            ],
        });
        // dof_bind_group: same layout as dof_bg but reads dof_view (for tone map input).
        // This is rebuilt in rebuild_tone_map_bind_group when dof is active.
        let dof_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("dof_bind_group_placeholder"),
            layout: dof_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: dof_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(
                        self.post
                            .foreground_placeholder_view
                            .as_ref()
                            .expect("ensure_hdr_shared not called"),
                    ),
                },
            ],
        });

        let contact_shadow_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("contact_shadow_bg"),
            layout: cs_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(nearest_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: cs_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let fxaa_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("fxaa_bg"),
            layout: fxaa_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&fxaa_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(fxaa_sampler),
                },
            ],
        });
        let outline_composite_bind_group =
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("outline_composite_bg"),
                layout: outline_composite_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(&outline_colour_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(outline_sampler),
                    },
                ],
            });

        // Edge-detection bind group : reads the R8 mask, writes outline ring.
        let outline_edge_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("outline_edge_uniform_buf"),
            size: std::mem::size_of::<OutlineEdgeUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let outline_edge_bgl = &self.outline.edge_bgl;
        let outline_edge_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("outline_edge_bg"),
            layout: outline_edge_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&outline_mask_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(outline_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: outline_edge_uniform_buf.as_entire_binding(),
                },
            ],
        });

        // OIT composite bind group placeholder (created lazily via ensure_viewport_oit)
        // We create a dummy one using placeholders so the bind group is always valid.
        // It will be rebuilt on first ensure_viewport_oit call.
        let oit_composite_bg_placeholder =
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("oit_composite_bg_placeholder"),
                layout: oit_composite_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(bloom_placeholder_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureView(bloom_placeholder_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(oit_sampler),
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
            ssaa_depth_only_view,
            ssaa_resolve_bind_group,
            ssaa_uniform_buf,
        ) = if ssaa_factor > 1 {
            let sw = scene_w * ssaa_factor;
            let sh = scene_h * ssaa_factor;
            let ssaa_colour_tex = make_tex(
                "ssaa_colour_texture",
                crate::gpu::TextureFormat::Rgba16Float,
                sw,
                sh,
                crate::gpu::TextureUsages::empty(),
            );
            let ssaa_colour_view =
                ssaa_colour_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
            let ssaa_depth_tex = make_tex(
                "ssaa_depth_texture",
                crate::gpu::TextureFormat::Depth24PlusStencil8,
                sw,
                sh,
                crate::gpu::TextureUsages::empty(),
            );
            let ssaa_depth_view =
                ssaa_depth_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
            let ssaa_depth_only_view = Some(ssaa_depth_tex.create_view(
                &crate::gpu::TextureViewDescriptor {
                    aspect: crate::gpu::TextureAspect::DepthOnly,
                    ..Default::default()
                },
            ));

            // Build the resolve bind group if the pipeline is available.
            let (ssaa_resolve_bg, ssaa_ubuf) = if let (Some(bgl), Some(nearest)) =
                (&self.post.ssaa_resolve_bgl, &self.post.pp_nearest_sampler)
            {
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct SsaaUniformData {
                    factor: u32,
                    _pad: [u32; 3],
                }
                let ubuf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("ssaa_uniform_buf"),
                    size: std::mem::size_of::<SsaaUniformData>() as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
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
                let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("ssaa_resolve_bg"),
                    layout: bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: crate::gpu::BindingResource::TextureView(&ssaa_colour_view),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::Sampler(nearest),
                        },
                        crate::gpu::BindGroupEntry {
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
                ssaa_depth_only_view,
                ssaa_resolve_bg,
                ssaa_ubuf,
            )
        } else {
            (None, None, None, None, None, None, None)
        };

        // --- Surface LIC per-viewport textures and bind group -- at scene resolution ---
        let lic_vector_tex = make_tex(
            "lic_vector",
            crate::gpu::TextureFormat::Rgba8Unorm,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let lic_vector_view =
            lic_vector_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        let lic_output_tex = make_tex(
            "lic_output",
            crate::gpu::TextureFormat::R8Unorm,
            scene_w,
            scene_h,
            crate::gpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let lic_output_view =
            lic_output_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

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
        let lic_noise_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("lic_noise"),
            size: crate::gpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &lic_noise_tex,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            &lic_noise_data,
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(scene_w),
                rows_per_image: Some(scene_h),
            },
            crate::gpu::Extent3d {
                width: scene_w,
                height: scene_h,
                depth_or_array_layers: 1,
            },
        );
        let lic_noise_view =
            lic_noise_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        let lic_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("lic_advect_uniform"),
            size: std::mem::size_of::<crate::resources::types::LicAdvectUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lic_advect_bgl = self
            .lic
            .advect_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let lic_advect_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("lic_advect_bg"),
            layout: lic_advect_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: lic_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(&lic_vector_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(&lic_noise_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::Sampler(
                        self.lic
                            .noise_sampler
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
        let (output_depth_texture, output_depth_view, depth_blit_bind_group) = if scene_w != w
            || scene_h != h
        {
            let tex = device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("output_depth_texture"),
                size: crate::gpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: crate::gpu::TextureFormat::Depth24PlusStencil8,
                usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                    | crate::gpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
            let bg = self.post.depth_blit_bgl.as_ref().map(|bgl| {
                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("depth_blit_bg"),
                    layout: bgl,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(&hdr_depth_only_view),
                    }],
                })
            });
            (Some(tex), view, bg)
        } else {
            let view = hdr_depth_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
            (None, view, None)
        };

        // HDR upscale target: when scene_size != output_size, tone-map and FXAA
        // run at scene resolution and write to this texture. An upscale-blit pass
        // then copies the result to output_view at native resolution.
        let (upscale_texture, upscale_view, upscale_bind_group) = if scene_w != w || scene_h != h {
            let tex = device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("hdr_upscale_texture"),
                size: crate::gpu::Extent3d {
                    width: scene_w,
                    height: scene_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: output_format,
                usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                    | crate::gpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
            let bgl = self.post.dyn_res_upscale_bgl.as_ref().unwrap();
            let sampler = self.post.dyn_res_linear_sampler.as_ref().unwrap();
            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("hdr_upscale_bg"),
                layout: bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(&view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            (Some(tex), Some(view), Some(bg))
        } else {
            (None, None, None)
        };

        let decal_depth_bg =
            self.create_decal_depth_bg(device, &hdr_depth_only_view, &hdr_stencil_only_view);

        ViewportHdrState {
            hdr_texture: hdr_tex,
            hdr_view,
            hdr_depth_texture: hdr_depth_tex,
            hdr_depth_view,
            hdr_depth_only_view,
            hdr_stencil_only_view,
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
            ssaa_depth_only_view,
            ssaa_resolve_bind_group,
            ssaa_uniform_buf,
            ssaa_factor,
            oit_accum_texture: None,
            oit_accum_view: None,
            oit_reveal_texture: None,
            oit_reveal_view: None,
            oit_composite_bind_group: None,
            oit_size: [0, 0],
            foreground_depth_texture: None,
            foreground_depth_view: None,
            foreground_depth_only_view: None,
            foreground_depth_size: [0, 0],
            outline_mask_texture: outline_mask_tex,
            outline_mask_view,
            outline_colour_texture: outline_colour_tex,
            outline_colour_view,
            outline_depth_texture: outline_depth_tex,
            outline_depth_view,
            outline_depth_only_view,
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
        device: &crate::gpu::Device,
        hdr: &mut ViewportHdrState,
        use_bloom: bool,
        use_ssao: bool,
        use_contact_shadows: bool,
        use_lic: bool,
        use_dof: bool,
        use_foreground: bool,
    ) {
        let bgl = match &self.post.tone_map_bgl {
            Some(b) => b,
            None => return,
        };
        let sampler = match &self.post.pp_linear_sampler {
            Some(s) => s,
            None => return,
        };
        let bloom_placeholder = match &self.post.bloom_placeholder_view {
            Some(v) => v,
            None => return,
        };
        let ao_placeholder = match &self.post.ao_placeholder_view {
            Some(v) => v,
            None => return,
        };
        let cs_placeholder = match &self.post.cs_placeholder_view {
            Some(v) => v,
            None => return,
        };
        let foreground_placeholder = match &self.post.foreground_placeholder_view {
            Some(v) => v,
            None => return,
        };
        let foreground_view = if use_foreground {
            hdr.foreground_depth_only_view
                .as_ref()
                .unwrap_or(foreground_placeholder)
        } else {
            foreground_placeholder
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

        let tone_map_hdr_input: &crate::gpu::TextureView = if use_dof {
            &hdr.dof_view
        } else {
            &hdr.hdr_view
        };
        hdr.tone_map_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tone_map_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(tone_map_hdr_input),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: hdr.tone_map_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(bloom_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(ao_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::TextureView(cs_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: crate::gpu::BindingResource::TextureView(&hdr.hdr_depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: crate::gpu::BindingResource::TextureView(if use_lic {
                        &hdr.lic_output_view
                    } else {
                        self.lic.placeholder_view.as_ref().unwrap_or(cs_placeholder)
                    }),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: crate::gpu::BindingResource::TextureView(foreground_view),
                },
            ],
        });

        // The DOF gather pass also reads the foreground coverage mask; rebuild
        // its bind group so the mask view matches this frame.
        if use_dof {
            if let Some(dof_bgl) = &self.post.dof_bgl {
                hdr.dof_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("dof_bg"),
                    layout: dof_bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: crate::gpu::BindingResource::TextureView(&hdr.hdr_view),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::Sampler(sampler),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 2,
                            resource: crate::gpu::BindingResource::TextureView(
                                &hdr.hdr_depth_only_view,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 3,
                            resource: hdr.dof_uniform_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 4,
                            resource: crate::gpu::BindingResource::TextureView(foreground_view),
                        },
                    ],
                });
            }
        }
    }

    /// Ensure OIT (order-independent transparency) render targets exist for the
    /// given per-viewport HDR state, creating or resizing them as needed.
    pub(crate) fn ensure_viewport_oit(
        &self,
        device: &crate::gpu::Device,
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

        let accum_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("oit_accum_texture"),
            size: crate::gpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba16Float,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let accum_view = accum_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let reveal_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("oit_reveal_texture"),
            size: crate::gpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R8Unorm,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let reveal_view = reveal_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        let sampler = self
            .oit
            .composite_sampler
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let bgl = self
            .oit
            .composite_bgl
            .as_ref()
            .expect("ensure_hdr_shared not called");
        let composite_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("oit_composite_bind_group"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&accum_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(&reveal_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        hdr.oit_accum_texture = Some(accum_tex);
        hdr.oit_accum_view = Some(accum_view);
        hdr.oit_reveal_texture = Some(reveal_tex);
        hdr.oit_reveal_view = Some(reveal_view);
        hdr.oit_composite_bind_group = Some(composite_bg);
    }

    /// Ensure the foreground pass depth target exists for the given
    /// per-viewport HDR state, creating or resizing it as needed. `w`/`h` are
    /// the scene target dimensions including any SSAA factor.
    pub(crate) fn ensure_viewport_foreground_depth(
        &self,
        device: &crate::gpu::Device,
        hdr: &mut ViewportHdrState,
        w: u32,
        h: u32,
    ) {
        let w = w.max(1);
        let h = h.max(1);
        if hdr.foreground_depth_size == [w, h] && hdr.foreground_depth_texture.is_some() {
            return;
        }
        hdr.foreground_depth_size = [w, h];

        let tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("foreground_depth_texture"),
            size: crate::gpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth24PlusStencil8,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let depth_only_view = tex.create_view(&crate::gpu::TextureViewDescriptor {
            label: Some("foreground_depth_only_view"),
            aspect: crate::gpu::TextureAspect::DepthOnly,
            ..Default::default()
        });

        hdr.foreground_depth_texture = Some(tex);
        hdr.foreground_depth_view = Some(view);
        hdr.foreground_depth_only_view = Some(depth_only_view);
    }
}
