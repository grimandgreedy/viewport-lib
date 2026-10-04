//! Shared post-processing pipelines and their per-viewport render targets.
//!
//! Holds [`PostProcessResources`] (FXAA/SSAA, bloom, SSAO, tone-map, DoF,
//! contact shadows, depth blit, and dynamic-resolution upscale) and the
//! `impl DeviceResources` methods that build and drive those passes. The
//! post-effect uniforms live in `uniforms` and the order-independent
//! transparency pipelines in `oit`.

use super::*;

pub(crate) mod composite;
pub(crate) mod oit;
pub(crate) mod producer;
pub(crate) mod targets;
pub(crate) mod uniforms;

pub(crate) use self::targets::TargetGroups;
use self::targets::{TargetSize, ViewportTargetAllocator};

pub(crate) use self::oit::OitResources;

/// Shared post-processing pipelines, layouts, samplers, and static textures:
/// FXAA / SSAA resolve, bloom, SSAO, tone-map, DoF, contact shadows, the
/// disabled-pass placeholders, the shared PP samplers, depth blit, and dynamic
/// resolution upscale. All device-shared and lazily built. The viewport-sized
/// intermediate textures and per-frame uniforms live on `ViewportHdrState`.
#[derive(Default)]
pub(crate) struct PostProcessResources {
    /// The post-composite stage chain's shared state.
    pub(crate) fxaa: producer::FxaaStage,
    pub(crate) ssaa_resolve_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) ssaa_resolve_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) tone_map_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) tone_map_bgl: Option<crate::gpu::BindGroupLayout>,
    /// The composite-input producers' shared state (pipelines, layouts,
    /// static resources). Per-viewport state lives on `ViewportHdrState`.
    pub(crate) ssao: producer::SsaoProducer,
    pub(crate) bloom: producer::BloomProducer,
    pub(crate) contact_shadow: producer::ContactShadowProducer,
    pub(crate) dof: producer::DofProducer,
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
    /// Depth half of the SSAA resolve: a min reduction over each block.
    pub(crate) ssaa_depth_resolve_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) ssaa_depth_resolve_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) dyn_res_upscale_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) dyn_res_upscale_ds_pipeline: Option<crate::gpu::RenderPipeline>,
    /// The two above with `PREMULTIPLIED_BLEND` instead of no blend, so a
    /// viewport rendered over a transparent background composites into its
    /// destination rather than replacing it. Built only when something asks to
    /// composite rather than blit.
    pub(crate) blit_composite_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) blit_composite_ds_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) dyn_res_upscale_bgl: Option<crate::gpu::BindGroupLayout>,
    pub(crate) dyn_res_linear_sampler: Option<crate::gpu::Sampler>,
}

use crate::resources::pipeline_slot::LazyFamily;

/// A full-screen pass over one bind group, built on first use. `get()`
/// returns `None` while a worker has it, and the pass is skipped that frame.
pub(crate) type LazyFullscreen = LazyFamily<FullscreenRecipe, 1>;

/// What a full-screen pass pipeline build reads. `target` is `None` for a
/// pass that writes depth only.
pub(crate) struct FullscreenRecipe {
    device: crate::gpu::Device,
    label: &'static str,
    layout: crate::gpu::PipelineLayout,
    shader: crate::resources::pipeline_slot::LazyModule,
    target: Option<(crate::gpu::TextureFormat, Option<crate::gpu::BlendState>)>,
    depth_stencil: Option<crate::gpu::DepthStencilState>,
    sample_count: u32,
}

fn build_fullscreen(r: &FullscreenRecipe, _i: usize) -> crate::gpu::RenderPipeline {
    let targets = r.target.map(|(format, blend)| {
        Some(crate::gpu::ColorTargetState {
            format,
            blend,
            write_mask: crate::gpu::ColorWrites::ALL,
        })
    });
    crate::resources::builders::render_pipeline(
        &r.device,
        crate::resources::builders::RenderPipelineDesc {
            label: r.label,
            layout: &r.layout,
            vertex_module: r.shader.get(),
            vertex_entry: "vs_main",
            vertex_buffers: &[],
            fragment: Some(crate::gpu::FragmentState {
                module: r.shader.get(),
                entry_point: Some("fs_main"),
                targets: targets.as_slice(),
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: r.depth_stencil.clone(),
            multisample: crate::gpu::MultisampleState {
                count: r.sample_count,
                ..Default::default()
            },
            cache: None,
        },
    )
}

/// What the outline composite builds read: one shader and layout, three
/// targets (LDR single-sample, LDR multisampled, HDR).
pub(crate) struct OutlineCompositeRecipe {
    device: crate::gpu::Device,
    layout: crate::gpu::PipelineLayout,
    shader: crate::resources::pipeline_slot::LazyModule,
    target_format: crate::gpu::TextureFormat,
    sample_count: u32,
}

pub(crate) const OUTLINE_COMPOSITE_SINGLE: usize = 0;
pub(crate) const OUTLINE_COMPOSITE_MSAA: usize = 1;
pub(crate) const OUTLINE_COMPOSITE_HDR: usize = 2;

fn build_outline_composite(r: &OutlineCompositeRecipe, i: usize) -> crate::gpu::RenderPipeline {
    let (label, format, sample_count) = match i {
        OUTLINE_COMPOSITE_SINGLE => ("outline_composite_pipeline_single", r.target_format, 1),
        OUTLINE_COMPOSITE_MSAA => (
            "outline_composite_pipeline_msaa",
            r.target_format,
            r.sample_count,
        ),
        _ => (
            "outline_composite_pipeline_hdr",
            crate::gpu::TextureFormat::Rgba16Float,
            1,
        ),
    };
    let blend = crate::gpu::BlendState {
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
    crate::resources::builders::render_pipeline(
        &r.device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout: &r.layout,
            vertex_module: r.shader.get(),
            vertex_entry: "vs_main",
            vertex_buffers: &[],
            fragment: Some(crate::gpu::FragmentState {
                module: r.shader.get(),
                entry_point: Some("fs_main"),
                targets: &[Some(crate::gpu::ColorTargetState {
                    format,
                    blend: Some(blend),
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
            cache: None,
        },
    )
}

impl DeviceResources {
    /// The composite-input producers, in encode order. Exposure runs last:
    /// its metering reads the sharp scene HDR and its result feeds the
    /// composite's exposure slot.
    pub(crate) fn post_producers(&self) -> [&dyn producer::PostProducer; 5] {
        [
            &self.post.ssao,
            &self.post.contact_shadow,
            &self.post.bloom,
            &self.post.dof,
            &self.exposure,
        ]
    }

    /// The post-composite stages, in chain order.
    pub(crate) fn post_stages(&self) -> [&dyn producer::PostStage; 1] {
        [&self.post.fxaa]
    }
}

impl DeviceResources {
    // -----------------------------------------------------------------------
    // Per-viewport HDR state : shared infrastructure
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Per-viewport HDR state : shared infrastructure
    // -----------------------------------------------------------------------

    /// Create the shared post-process infrastructure that per-viewport HDR state
    /// is built against: samplers, bind group layouts, placeholder textures, the
    /// SSAO noise texture and its kernel buffer. Builds no pipelines and compiles
    /// no shaders, so a frame that never takes the HDR path pays none of that.
    /// No-op after the first call. Must be called before `create_hdr_viewport_state`.
    pub(crate) fn ensure_hdr_infra(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        // Guard: if the sampler and one layout exist, everything here is created.
        if self.post.bloom.bgl.is_some() && self.post.fxaa.sampler.is_some() {
            return;
        }
        // --- Fallback textures (one-time uploads) ---
        if !self.material.uploaded {
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
            upload(&self.material.normal_map, &[128u8, 128u8, 255u8, 255u8]);
            upload(&self.material.ao_map, &[255u8, 255u8, 255u8, 255u8]);
            upload(
                self.material
                    .texture
                    .texture
                    .as_ref()
                    .expect("fallback albedo texture is owned"),
                &[255u8, 255u8, 255u8, 255u8],
            );
            self.material.uploaded = true;
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
        if self.post.ssao.noise_view.is_none() {
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
            self.post.ssao.noise_view =
                Some(noise_tex.create_view(&crate::gpu::TextureViewDescriptor::default()));
            self.post.ssao.noise_texture = Some(noise_tex);
        }

        // --- SSAO kernel (one-time) ---
        if self.post.ssao.kernel_buf.is_none() {
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
            self.post.ssao.kernel_buf = Some(buf);
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
        // The tone-map layout is driven by the composite binding table; see
        // `composite.rs` for the slot set and each binding's role.
        let tone_map_bgl = composite::create_tone_map_bgl(device);

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

        self.ensure_outline_composite_bgl(device);

        // --- SSAA resolve bind group layout ---
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
        // --- DoF bind group layout ---
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

        // Store everything
        self.post.pp_linear_sampler = Some(linear_sampler);
        self.post.pp_nearest_sampler = Some(nearest_sampler);
        self.post.fxaa.sampler = Some(fxaa_sampler);
        self.oit.composite_sampler = Some(oit_sampler);
        self.outline.composite_sampler = Some(outline_sampler);

        self.post.tone_map_bgl = Some(tone_map_bgl);
        self.post.bloom.bgl = Some(bloom_bgl);
        self.post.ssao.bgl = Some(ssao_bgl);
        self.post.ssao.blur_bgl = Some(ssao_blur_bgl);
        self.post.contact_shadow.bgl = Some(cs_bgl);
        self.post.fxaa.bgl = Some(fxaa_bgl);
        self.oit.composite_bgl = Some(oit_composite_bgl);
        self.post.ssaa_resolve_bgl = Some(ssaa_resolve_bgl);
        self.post.dof.bgl = Some(dof_bgl);

        // --- Depth blit bind group layout ---
        // The blit copies a scene-resolution depth texture to a native-resolution
        // depth-only target, for when render_scale < 1.0.
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
            self.post.depth_blit_bgl = Some(bgl);
        }

        // --- SSAA depth resolve bind group layout ---
        if self.post.ssaa_depth_resolve_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("ssaa_depth_resolve_bgl"),
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
                    crate::resources::builders::uniform_entry(
                        1,
                        crate::gpu::ShaderStages::FRAGMENT,
                    ),
                ],
            });
            self.post.ssaa_depth_resolve_bgl = Some(bgl);
        }

        // --- Foreground depth stamp bind group layout ---
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
            self.post.foreground_stamp_bgl = Some(bgl);
        }
    }

    /// The outline composite's bind group layout, shared by the three composite
    /// pipelines and by each viewport's composite bind group.
    fn ensure_outline_composite_bgl(&mut self, device: &crate::gpu::Device) {
        if self.outline.composite_bgl.is_none() {
            self.outline.composite_bgl = Some(crate::resources::builders::texture_sampler_bgl(
                device,
                "outline_composite_bgl",
                crate::gpu::ShaderStages::FRAGMENT,
            ));
        }
    }

    /// Compose the three fullscreen pipelines that blit the offscreen outline
    /// texture onto the main target: LDR single-sample, LDR multisampled, and
    /// the HDR variant. Composed on the first frame that has a selection
    /// outline to draw, in either path; each is built when a pass binds it.
    pub(crate) fn ensure_outline_composite_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.outline.composite.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        self.ensure_outline_composite_bgl(device);
        let bgl = self.outline.composite_bgl.clone().expect("just ensured");
        let shader = self.shared_module(
            device,
            "outline_composite_shader",
            crate::resources::builders::wgsl_source!("outline_composite"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "outline_composite_layout",
            &[&bgl],
        );
        self.outline.composite = Some(LazyFamily::new(
            OutlineCompositeRecipe {
                device: device.clone(),
                layout,
                shader,
                target_format: self.target_format,
                sample_count: self.sample_count,
            },
            std::sync::Arc::clone(&self.pipeline_compiler),
            build_outline_composite,
        ));
    }

    /// Build every shared pipeline the HDR path can bind, whatever a frame asks
    /// for. A frame builds only the groups it uses, through the per-group
    /// `ensure_*` functions below; this is for a caller that wants the whole
    /// set at once. Each group is a no-op once built.
    #[cfg(test)]
    pub(crate) fn ensure_hdr_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_format: crate::gpu::TextureFormat,
    ) {
        self.ensure_hdr_infra(device, queue);
        self.ensure_tone_map_pipeline(device, output_format);
        self.exposure
            .ensure_pipelines(device, &self.pipeline_compiler);
        self.ensure_bloom_pipelines(device);
        self.ensure_ssao_pipelines(device);
        self.ensure_contact_shadow_pipeline(device);
        self.ensure_fxaa_pipeline(device, output_format);
        self.ensure_dof_pipeline(device);
        self.ensure_oit_composite_pipeline(device);
        self.ensure_oit_mesh_pipelines(device);
        self.ensure_hdr_mesh_pipelines(device);
        self.ensure_outline_composite_pipelines(device);
        self.ensure_ssaa_resolve_pipelines(device);
        self.ensure_depth_blit_pipeline(device);
        self.ensure_foreground_stamp_pipeline(device);
    }

    /// A full-screen pass over a single bind group, built on first use: the
    /// layout now, the pipeline when a frame binds it.
    fn fullscreen_pass_pipeline(
        &self,
        device: &crate::gpu::Device,
        label: &'static str,
        shader: crate::resources::pipeline_slot::LazyModule,
        bgl: &crate::gpu::BindGroupLayout,
        format: crate::gpu::TextureFormat,
    ) -> LazyFullscreen {
        self.lazy_fullscreen(device, label, shader, bgl, Some((format, None)), None, 1)
    }

    /// A full-screen pass over one bind group with the colour target, depth
    /// state and sample count spelled out. `target` is `None` for a pass
    /// that writes depth only.
    #[allow(clippy::too_many_arguments)]
    fn lazy_fullscreen(
        &self,
        device: &crate::gpu::Device,
        label: &'static str,
        shader: crate::resources::pipeline_slot::LazyModule,
        bgl: &crate::gpu::BindGroupLayout,
        target: Option<(crate::gpu::TextureFormat, Option<crate::gpu::BlendState>)>,
        depth_stencil: Option<crate::gpu::DepthStencilState>,
        sample_count: u32,
    ) -> LazyFullscreen {
        let layout = crate::resources::builders::pipeline_layout(
            device,
            format!("{label}_layout").as_str(),
            &[bgl],
        );
        LazyFamily::new(
            FullscreenRecipe {
                device: device.clone(),
                label,
                layout,
                shader,
                target,
                depth_stencil,
                sample_count,
            },
            std::sync::Arc::clone(&self.pipeline_compiler),
            build_fullscreen,
        )
    }

    /// The tone-map composite, which every HDR frame ends with. Needs
    /// `ensure_hdr_infra`.
    pub(crate) fn ensure_tone_map_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        output_format: crate::gpu::TextureFormat,
    ) {
        if self.post.tone_map_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .post
            .tone_map_bgl
            .clone()
            .expect("ensure_hdr_infra not called");
        let shader = self.shared_module(
            device,
            "tone_map_shader",
            crate::resources::builders::wgsl_source!("tone_map"),
        );
        // Built now whatever the policy: without it an HDR frame has no image.
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "tone_map_pipeline_layout",
            &[&bgl],
        );
        self.post.tone_map_pipeline = Some(crate::resources::builders::build_fullscreen_pipeline(
            device,
            "tone_map_pipeline",
            &layout,
            shader.get(),
            output_format,
            None,
        ));
    }

    /// Bloom threshold and blur. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_bloom_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.post.bloom.threshold_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .post
            .bloom
            .bgl
            .clone()
            .expect("ensure_hdr_infra not called");
        let threshold_shader = self.shared_module(
            device,
            "bloom_threshold_shader",
            crate::resources::builders::wgsl_source!("bloom_threshold"),
        );
        let blur_shader = self.shared_module(
            device,
            "bloom_blur_shader",
            crate::resources::builders::wgsl_source!("bloom_blur"),
        );
        self.post.bloom.threshold_pipeline = Some(self.fullscreen_pass_pipeline(
            device,
            "bloom_threshold_pipeline",
            threshold_shader,
            &bgl,
            crate::gpu::TextureFormat::Rgba16Float,
        ));
        self.post.bloom.blur_pipeline = Some(self.fullscreen_pass_pipeline(
            device,
            "bloom_blur_pipeline",
            blur_shader,
            &bgl,
            crate::gpu::TextureFormat::Rgba16Float,
        ));
    }

    /// SSAO occlusion and blur. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_ssao_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.post.ssao.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let missing = "ensure_hdr_infra not called";
        let bgl = self.post.ssao.bgl.clone().expect(missing);
        let blur_bgl = self.post.ssao.blur_bgl.clone().expect(missing);
        let shader = self.shared_module(
            device,
            "ssao_shader",
            crate::resources::builders::wgsl_source!("ssao"),
        );
        let blur_shader = self.shared_module(
            device,
            "ssao_blur_shader",
            crate::resources::builders::wgsl_source!("ssao_blur"),
        );
        self.post.ssao.pipeline = Some(self.fullscreen_pass_pipeline(
            device,
            "ssao_pipeline",
            shader,
            &bgl,
            crate::gpu::TextureFormat::R8Unorm,
        ));
        self.post.ssao.blur_pipeline = Some(self.fullscreen_pass_pipeline(
            device,
            "ssao_blur_pipeline",
            blur_shader,
            &blur_bgl,
            crate::gpu::TextureFormat::R8Unorm,
        ));
    }

    /// Contact shadows. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_contact_shadow_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.post.contact_shadow.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .post
            .contact_shadow
            .bgl
            .clone()
            .expect("ensure_hdr_infra not called");
        let shader = self.shared_module(
            device,
            "contact_shadow_shader",
            crate::resources::builders::wgsl_source!("contact_shadow"),
        );
        self.post.contact_shadow.pipeline = Some(self.fullscreen_pass_pipeline(
            device,
            "contact_shadow_pipeline",
            shader,
            &bgl,
            crate::gpu::TextureFormat::R8Unorm,
        ));
    }

    /// FXAA. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_fxaa_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        output_format: crate::gpu::TextureFormat,
    ) {
        if self.post.fxaa.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .post
            .fxaa
            .bgl
            .clone()
            .expect("ensure_hdr_infra not called");
        let shader = self.shared_module(
            device,
            "fxaa_shader",
            crate::resources::builders::wgsl_source!("fxaa"),
        );
        self.post.fxaa.pipeline = Some(self.fullscreen_pass_pipeline(
            device,
            "fxaa_pipeline",
            shader,
            &bgl,
            output_format,
        ));
    }

    /// Depth of field. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_dof_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.post.dof.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .post
            .dof
            .bgl
            .clone()
            .expect("ensure_hdr_infra not called");
        let shader = self.shared_module(
            device,
            "dof_shader",
            crate::resources::builders::wgsl_source!("dof"),
        );
        self.post.dof.pipeline = Some(self.fullscreen_pass_pipeline(
            device,
            "dof_pipeline",
            shader,
            &bgl,
            crate::gpu::TextureFormat::Rgba16Float,
        ));
    }

    /// The OIT resolve, which composites the accumulation and reveal targets
    /// over the opaque image. Every frame that runs the OIT pass binds it,
    /// whoever drew into the pass. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_oit_composite_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.oit.composite_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .oit
            .composite_bgl
            .clone()
            .expect("ensure_hdr_infra not called");
        let shader = self.shared_module(
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
        self.oit.composite_pipeline = Some(self.lazy_fullscreen(
            device,
            "oit_composite_pipeline",
            shader,
            &bgl,
            Some((crate::gpu::TextureFormat::Rgba16Float, Some(premul_blend))),
            None,
            1,
        ));
    }

    /// The per-object OIT mesh pipelines: two colour targets (`Rgba16Float`
    /// accumulation and `R8Unorm` reveal), depth-test only. Composed with the
    /// registered deformers. The instanced twin is built separately by
    /// `ensure_oit_instanced_pipeline`, once the instance layout exists.
    pub(crate) fn ensure_oit_mesh_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.oit.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let source = {
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
        let shader = self.shared_module(
            device,
            "mesh_oit_shader",
            crate::resources::builders::builtin_hook_env(
                crate::resources::builders::strip_debug_vis(source, self.debug_vis_shaders),
            )
            .as_ref(),
        );
        let layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            "oit_pipeline_layout",
            &self.binds.camera_bgl,
            &self.binds.object_bgl,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        self.oit.pipeline = Some(crate::resources::pipeline_slot::LazyFamily::new(
            oit::OitContext {
                device: device.clone(),
                layout,
                shader,
            },
            std::sync::Arc::clone(&self.pipeline_compiler),
            oit::build_per_object,
        ));
    }

    /// The SSAA colour resolve and its depth half, which downsamples the
    /// supersampled depth into the scene-resolution buffer, taking the nearest
    /// sample of each block. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_ssaa_resolve_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.post.ssaa_resolve_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let missing = "ensure_hdr_infra not called";
        let resolve_bgl = self.post.ssaa_resolve_bgl.clone().expect(missing);
        let shader = self.shared_module(
            device,
            "ssaa_resolve_shader",
            crate::resources::builders::wgsl_source!("ssaa_resolve"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "ssaa_resolve_layout",
            &[&resolve_bgl],
        );
        self.post.ssaa_resolve_pipeline =
            Some(crate::resources::builders::build_fullscreen_pipeline(
                device,
                "ssaa_resolve_pipeline",
                &layout,
                shader.get(),
                crate::gpu::TextureFormat::Rgba16Float,
                None,
            ));

        let depth_bgl = self.post.ssaa_depth_resolve_bgl.clone().expect(missing);
        self.post.ssaa_depth_resolve_pipeline = Some(Self::depth_only_fullscreen_pipeline(
            device,
            "ssaa_depth_resolve",
            crate::resources::builders::wgsl_source!("ssaa_depth_resolve"),
            &depth_bgl,
        ));
    }

    /// A full-screen pass that writes depth and no colour, over one bind group.
    /// `name` prefixes the shader, layout and pipeline labels.
    fn depth_only_fullscreen_pipeline(
        device: &crate::gpu::Device,
        name: &str,
        source: &'static str,
        bgl: &crate::gpu::BindGroupLayout,
    ) -> crate::gpu::RenderPipeline {
        let shader = crate::resources::builders::wgsl_module(
            device,
            format!("{name}_shader").as_str(),
            source,
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            format!("{name}_layout").as_str(),
            &[bgl],
        );
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: format!("{name}_pipeline").as_str(),
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[],
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
        )
    }

    /// The depth upscale for the render-scale path, which takes a single
    /// sub-sample. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_depth_blit_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.post.depth_blit_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .post
            .depth_blit_bgl
            .clone()
            .expect("ensure_hdr_infra not called");
        self.post.depth_blit_pipeline = Some(Self::depth_only_fullscreen_pipeline(
            device,
            "depth_blit",
            crate::resources::builders::wgsl_source!("depth_blit"),
            &bgl,
        ));
    }

    /// The foreground depth stamp: writes near depth into the output depth
    /// buffer where the foreground pass drew, so post-tone-map passes are
    /// occluded by foreground items. Needs `ensure_hdr_infra`.
    pub(crate) fn ensure_foreground_stamp_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.post.foreground_stamp_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .post
            .foreground_stamp_bgl
            .clone()
            .expect("ensure_hdr_infra not called");
        self.post.foreground_stamp_pipeline = Some(Self::depth_only_fullscreen_pipeline(
            device,
            "foreground_stamp",
            crate::resources::builders::wgsl_source!("foreground_depth_stamp"),
            &bgl,
        ));
    }

    /// Create a fresh [`ViewportHdrState`] for the given viewport dimensions.
    ///
    /// `w, h` are the native output dimensions. `scene_w, scene_h` are the effective
    /// render target dimensions after applying render scale (equal to `w, h` when
    /// render_scale = 1.0). Scene-side textures (HDR colour, depth, bloom, SSAO, etc.)
    /// are allocated at `scene_w x scene_h`; output-side textures (FXAA) remain at
    /// `w x h`. The tone map pass upscales from scene to output resolution.
    ///
    /// Only the targets of the groups in `groups` are allocated at these sizes.
    /// Every other target is a one-texel stand-in, so a viewport holds memory
    /// for what its frames use and nothing else, and the bind groups built here
    /// are valid either way.
    ///
    /// `reuse` is the state this one replaces when a group is being promoted
    /// and no size has changed. Its live groups, uniform buffers and lazily
    /// allocated targets carry over by handle, so anything already rendered
    /// into them this frame (the outline mask is drawn at prepare time) and
    /// anything that persists across frames (the exposure state) survives.
    ///
    /// [`ensure_hdr_infra`](Self::ensure_hdr_infra) must have been called first so that
    /// BGLs, samplers, and placeholder textures are available on `self`.
    #[allow(clippy::too_many_arguments)]
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
        groups: TargetGroups,
        reuse: Option<&ViewportHdrState>,
    ) -> ViewportHdrState {
        let w = w.max(1);
        let h = h.max(1);
        let scene_w = scene_w.max(1);
        let scene_h = scene_h.max(1);
        let ssaa_factor = ssaa_factor.max(1);

        // All viewport-sized targets go through the allocator, which owns the
        // resolution classes and the base usage; see `targets.rs`.
        let alloc = ViewportTargetAllocator::new(device, w, h, scene_w, scene_h, ssaa_factor);
        // The old state, for a group that was already live in it.
        let kept = |group: TargetGroups| reuse.filter(|old| old.groups.contains(group));

        // HDR scene colour and depth -- at scene resolution (render_scale * output).
        // COPY_SRC enables the refractive sprite pass to copy the resolved
        // scene colour into its sample texture before drawing distortion.
        let scene_size = groups.size(TargetGroups::SCENE, TargetSize::Scene);
        let (hdr_tex, hdr_view) = match kept(TargetGroups::SCENE) {
            Some(old) => (old.hdr_texture.clone(), old.hdr_view.clone()),
            None => alloc.colour(
                "hdr_texture",
                crate::gpu::TextureFormat::Rgba16Float,
                scene_size,
                crate::gpu::TextureUsages::COPY_SRC,
            ),
        };
        let (hdr_depth_tex, hdr_depth_view, hdr_depth_only_view) = match kept(TargetGroups::SCENE) {
            Some(old) => (
                old.hdr_depth_texture.clone(),
                old.hdr_depth_view.clone(),
                old.hdr_depth_only_view.clone(),
            ),
            None => {
                let depth = alloc.depth("hdr_depth_texture", scene_size);
                (depth.texture, depth.view, depth.depth_only_view)
            }
        };
        let hdr_stencil_only_view = hdr_depth_tex.create_view(&crate::gpu::TextureViewDescriptor {
            aspect: crate::gpu::TextureAspect::StencilOnly,
            ..Default::default()
        });

        // Bloom -- threshold at scene resolution, ping/pong at half.
        let (
            (bloom_threshold_tex, bloom_threshold_view),
            (bloom_ping_tex, bloom_ping_view),
            (bloom_pong_tex, bloom_pong_view),
        ) = match kept(TargetGroups::BLOOM) {
            Some(old) => (
                (
                    old.bloom.threshold_texture.clone(),
                    old.bloom.threshold_view.clone(),
                ),
                (old.bloom.ping_texture.clone(), old.bloom.ping_view.clone()),
                (old.bloom.pong_texture.clone(), old.bloom.pong_view.clone()),
            ),
            None => {
                let half = groups.size(TargetGroups::BLOOM, TargetSize::HalfScene);
                (
                    alloc.colour(
                        "bloom_threshold_texture",
                        crate::gpu::TextureFormat::Rgba16Float,
                        groups.size(TargetGroups::BLOOM, TargetSize::Scene),
                        crate::gpu::TextureUsages::empty(),
                    ),
                    alloc.colour(
                        "bloom_ping_texture",
                        crate::gpu::TextureFormat::Rgba16Float,
                        half,
                        crate::gpu::TextureUsages::empty(),
                    ),
                    alloc.colour(
                        "bloom_pong_texture",
                        crate::gpu::TextureFormat::Rgba16Float,
                        half,
                        crate::gpu::TextureUsages::empty(),
                    ),
                )
            }
        };

        // SSAO -- at scene resolution.
        let ((ssao_tex, ssao_view), (ssao_blur_tex, ssao_blur_view)) =
            match kept(TargetGroups::SSAO) {
                Some(old) => (
                    (old.ssao.texture.clone(), old.ssao.view.clone()),
                    (old.ssao.blur_texture.clone(), old.ssao.blur_view.clone()),
                ),
                None => {
                    let size = groups.size(TargetGroups::SSAO, TargetSize::Scene);
                    (
                        alloc.colour(
                            "ssao_texture",
                            crate::gpu::TextureFormat::R8Unorm,
                            size,
                            crate::gpu::TextureUsages::empty(),
                        ),
                        alloc.colour(
                            "ssao_blur_texture",
                            crate::gpu::TextureFormat::R8Unorm,
                            size,
                            crate::gpu::TextureUsages::empty(),
                        ),
                    )
                }
            };

        // Depth of field -- at scene resolution.
        let (dof_tex, dof_view) = match kept(TargetGroups::DOF) {
            Some(old) => (old.dof.texture.clone(), old.dof.view.clone()),
            None => alloc.colour(
                "dof_texture",
                crate::gpu::TextureFormat::Rgba16Float,
                groups.size(TargetGroups::DOF, TargetSize::Scene),
                crate::gpu::TextureUsages::empty(),
            ),
        };

        // Contact shadow -- at scene resolution.
        let (cs_tex, cs_view) = match kept(TargetGroups::CONTACT_SHADOW) {
            Some(old) => (
                old.contact_shadow.texture.clone(),
                old.contact_shadow.view.clone(),
            ),
            None => alloc.colour(
                "contact_shadow_texture",
                crate::gpu::TextureFormat::R8Unorm,
                groups.size(TargetGroups::CONTACT_SHADOW, TargetSize::Scene),
                crate::gpu::TextureUsages::empty(),
            ),
        };

        // FXAA -- at scene resolution so the whole post-process chain runs at
        // the scaled size when render_scale < 1.0.
        let (fxaa_tex, fxaa_view) = match kept(TargetGroups::FXAA) {
            Some(old) => (old.fxaa.texture.clone(), old.fxaa.view.clone()),
            None => alloc.colour(
                "fxaa_texture",
                output_format,
                groups.size(TargetGroups::FXAA, TargetSize::Scene),
                crate::gpu::TextureUsages::empty(),
            ),
        };

        // Outline offscreen : mask (R8), colour (target_format), and depth -- at scene resolution.
        let ((outline_mask_tex, outline_mask_view), (outline_colour_tex, outline_colour_view)) =
            match kept(TargetGroups::OUTLINE) {
                Some(old) => (
                    (
                        old.outline_mask_texture.clone(),
                        old.outline_mask_view.clone(),
                    ),
                    (
                        old.outline_colour_texture.clone(),
                        old.outline_colour_view.clone(),
                    ),
                ),
                None => {
                    let size = groups.size(TargetGroups::OUTLINE, TargetSize::Scene);
                    (
                        alloc.colour(
                            "outline_mask_texture",
                            crate::gpu::TextureFormat::R8Unorm,
                            size,
                            crate::gpu::TextureUsages::empty(),
                        ),
                        alloc.colour(
                            "outline_colour_texture",
                            self.target_format,
                            size,
                            crate::gpu::TextureUsages::empty(),
                        ),
                    )
                }
            };
        // The LDR path renders against this depth target and the outline mask
        // pass tests against it. It is sampleable so the HiZ occlusion
        // prev-depth copy can read the LDR scene depth.
        let (outline_depth_tex, outline_depth_view, outline_depth_only_view) =
            match kept(TargetGroups::LDR_DEPTH) {
                Some(old) => (
                    old.outline_depth_texture.clone(),
                    old.outline_depth_view.clone(),
                    old.outline_depth_only_view.clone(),
                ),
                None => {
                    let depth = alloc.depth(
                        "outline_depth_texture",
                        groups.size(TargetGroups::LDR_DEPTH, TargetSize::Scene),
                    );
                    (depth.texture, depth.view, depth.depth_only_view)
                }
            };

        // Uniform buffers. None depends on a target's size, so a promotion keeps
        // the old state's: the outline pass has already written its edge
        // uniform this frame, and the exposure state is the adaptation history.
        let uniform = |old: Option<&crate::gpu::Buffer>, label: &str, size: usize| match old {
            Some(buf) => buf.clone(),
            None => device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        let tone_map_uniform_buf = uniform(
            reuse.map(|o| &o.tone_map_uniform_buf),
            "tone_map_uniform_buf",
            std::mem::size_of::<ToneMapUniform>(),
        );
        // Auto-exposure per-viewport buffers (bind group built after `hdr_view`).
        let (exposure_histogram_buf, exposure_state_buf, exposure_params_buf) = match reuse {
            Some(old) => (
                old.exposure_histogram_buf.clone(),
                old.exposure_state_buf.clone(),
                old.exposure_params_buf.clone(),
            ),
            None => {
                crate::resources::gpu::exposure::ExposureResources::create_viewport_buffers(device)
            }
        };
        let bloom_uniform_buf = uniform(
            reuse.map(|o| &o.bloom.uniform_buf),
            "bloom_uniform_buf",
            std::mem::size_of::<BloomUniform>(),
        );
        let bloom_h_uniform_buf = if let Some(old) = reuse {
            old.bloom.h_uniform_buf.clone()
        } else {
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
        let bloom_v_uniform_buf = if let Some(old) = reuse {
            old.bloom.v_uniform_buf.clone()
        } else {
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
        let ssao_uniform_buf = uniform(
            reuse.map(|o| &o.ssao.uniform_buf),
            "ssao_uniform_buf",
            std::mem::size_of::<SsaoUniform>(),
        );
        let cs_uniform_buf = uniform(
            reuse.map(|o| &o.contact_shadow.uniform_buf),
            "contact_shadow_uniform_buf",
            std::mem::size_of::<ContactShadowUniform>(),
        );
        let dof_uniform_buf = uniform(
            reuse.map(|o| &o.dof.uniform_buf),
            "dof_uniform_buf",
            std::mem::size_of::<DofUniform>(),
        );

        // Shared references needed for bind groups
        let linear_sampler = self
            .post
            .pp_linear_sampler
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let nearest_sampler = self
            .post
            .pp_nearest_sampler
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let fxaa_sampler = self
            .post
            .fxaa
            .sampler
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let oit_sampler = self
            .oit
            .composite_sampler
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let outline_sampler = self
            .outline
            .composite_sampler
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let bloom_placeholder_view = self
            .post
            .bloom_placeholder_view
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let ao_placeholder_view = self
            .post
            .ao_placeholder_view
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let cs_placeholder_view = self
            .post
            .cs_placeholder_view
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let ssao_noise_view = self
            .post
            .ssao
            .noise_view
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let ssao_kernel_buf = self
            .post
            .ssao
            .kernel_buf
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let tone_map_bgl = self
            .post
            .tone_map_bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let bloom_bgl = self
            .post
            .bloom
            .bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let ssao_bgl = self
            .post
            .ssao
            .bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let ssao_blur_bgl = self
            .post
            .ssao
            .blur_bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let cs_bgl = self
            .post
            .contact_shadow
            .bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let fxaa_bgl = self
            .post
            .fxaa
            .bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let oit_composite_bgl = self
            .oit
            .composite_bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
        let outline_composite_bgl = self
            .outline
            .composite_bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");

        // Bind groups
        let tone_map_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tone_map_bg"),
            layout: tone_map_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::HDR_COLOUR,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::SAMPLER,
                    resource: crate::gpu::BindingResource::Sampler(linear_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::PARAMS,
                    resource: tone_map_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::BLOOM,
                    resource: crate::gpu::BindingResource::TextureView(bloom_placeholder_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::AO,
                    resource: crate::gpu::BindingResource::TextureView(ao_placeholder_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::CONTACT_SHADOW,
                    resource: crate::gpu::BindingResource::TextureView(cs_placeholder_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::SCENE_DEPTH,
                    resource: crate::gpu::BindingResource::TextureView(&hdr_depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::FOREGROUND_DEPTH,
                    resource: crate::gpu::BindingResource::TextureView(
                        self.post
                            .foreground_placeholder_view
                            .as_ref()
                            .expect("ensure_hdr_infra not called"),
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::EXPOSURE,
                    resource: exposure_state_buf.as_entire_binding(),
                },
                // Neutral stand-in; the shader gates on `grade_enabled`.
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::GRADE_LUT,
                    resource: crate::gpu::BindingResource::TextureView(ao_placeholder_view),
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
            .dof
            .bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
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
                            .expect("ensure_hdr_infra not called"),
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
        let outline_edge_uniform_buf = uniform(
            reuse.map(|o| &o.outline_edge_uniform_buf),
            "outline_edge_uniform_buf",
            std::mem::size_of::<OutlineEdgeUniform>(),
        );
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
            ssaa_depth_blit_bind_group,
            ssaa_uniform_buf,
        ) = if ssaa_factor > 1 && groups.contains(TargetGroups::SCENE) {
            let (ssaa_colour_tex, ssaa_colour_view) = alloc.colour(
                "ssaa_colour_texture",
                crate::gpu::TextureFormat::Rgba16Float,
                TargetSize::SsaaScene,
                crate::gpu::TextureUsages::empty(),
            );
            let ssaa_depth = alloc.depth("ssaa_depth_texture", TargetSize::SsaaScene);
            let (ssaa_depth_tex, ssaa_depth_view) = (ssaa_depth.texture, ssaa_depth.view);

            let ssaa_depth_only_view = Some(ssaa_depth.depth_only_view);

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

            // Depth half of the resolve. The colour resolve alone leaves
            // `hdr_depth` untouched for the whole frame, and every pass after
            // it attaches that buffer and depth-tests against it. Shares the
            // factor uniform with the colour resolve: the reduction needs the
            // block size.
            let ssaa_depth_blit_bg = match (
                self.post.ssaa_depth_resolve_bgl.as_ref(),
                ssaa_depth_only_view.as_ref(),
                ssaa_ubuf.as_ref(),
            ) {
                (Some(bgl), Some(depth_view), Some(ubuf)) => {
                    Some(device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("ssaa_depth_resolve_bg"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::gpu::BindingResource::TextureView(depth_view),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: ubuf.as_entire_binding(),
                            },
                        ],
                    }))
                }
                _ => None,
            };

            (
                Some(ssaa_colour_tex),
                Some(ssaa_colour_view),
                Some(ssaa_depth_tex),
                Some(ssaa_depth_view),
                ssaa_depth_only_view,
                ssaa_resolve_bg,
                ssaa_depth_blit_bg,
                ssaa_ubuf,
            )
        } else {
            (None, None, None, None, None, None, None, None)
        };

        // Output-resolution depth for post-tone-map passes.
        // When render scale = 1.0 (scene == output), reuse hdr_depth as a second view.
        // When render scale < 1.0, allocate a separate native-res texture and create a
        // bind group so the depth blit pass can copy hdr_depth into it each frame.
        // Both render-scale targets serve the HDR path only.
        let scaled = (scene_w != w || scene_h != h) && groups.contains(TargetGroups::SCENE);
        let (output_depth_texture, output_depth_view, depth_blit_bind_group) = if scaled {
            let output_depth = alloc.depth("output_depth_texture", TargetSize::Output);
            let (tex, view) = (output_depth.texture, output_depth.view);
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
        let (upscale_texture, upscale_view, upscale_bind_group) = if scaled {
            let (tex, view) = alloc.colour(
                "hdr_upscale_texture",
                output_format,
                TargetSize::Scene,
                crate::gpu::TextureUsages::empty(),
            );
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

        // Auto-exposure compute bind group. Metering reads the sharp scene HDR
        // (`hdr_view`), never the DOF-blurred copy. The buffers are allocated
        // earlier so the tone-map bind group can bind `exposure_state_buf`.
        let exposure_bind_group = self.exposure.create_bind_group(
            device,
            &hdr_view,
            &hdr_depth_only_view,
            &exposure_params_buf,
            &exposure_histogram_buf,
            &exposure_state_buf,
        );

        ViewportHdrState {
            hdr_texture: hdr_tex,
            hdr_view,
            hdr_depth_texture: hdr_depth_tex,
            hdr_depth_view,
            hdr_depth_only_view,
            hdr_stencil_only_view,
            bloom: producer::BloomViewport {
                threshold_texture: bloom_threshold_tex,
                threshold_view: bloom_threshold_view,
                ping_texture: bloom_ping_tex,
                ping_view: bloom_ping_view,
                pong_texture: bloom_pong_tex,
                pong_view: bloom_pong_view,
                threshold_bg: bloom_threshold_bg,
                blur_h_bg: bloom_blur_h_bg,
                blur_v_bg: bloom_blur_v_bg,
                blur_h_pong_bg: bloom_blur_h_pong_bg,
                uniform_buf: bloom_uniform_buf,
                h_uniform_buf: bloom_h_uniform_buf,
                v_uniform_buf: bloom_v_uniform_buf,
            },
            ssao: producer::SsaoViewport {
                texture: ssao_tex,
                view: ssao_view,
                blur_texture: ssao_blur_tex,
                blur_view: ssao_blur_view,
                bg: ssao_bg,
                blur_bg: ssao_blur_bg,
                uniform_buf: ssao_uniform_buf,
            },
            contact_shadow: producer::ContactShadowViewport {
                texture: cs_tex,
                view: cs_view,
                bg: contact_shadow_bg,
                uniform_buf: cs_uniform_buf,
            },
            dof: producer::DofViewport {
                texture: dof_tex,
                view: dof_view,
                bg: dof_bg,
                uniform_buf: dof_uniform_buf,
            },
            fxaa: producer::FxaaViewport {
                texture: fxaa_tex,
                view: fxaa_view,
                bind_group: fxaa_bind_group,
            },
            ssaa_colour_texture,
            ssaa_colour_view,
            ssaa_depth_texture,
            ssaa_depth_view,
            ssaa_depth_only_view,
            ssaa_resolve_bind_group,
            ssaa_depth_blit_bind_group,
            ssaa_uniform_buf,
            ssaa_factor,
            // The OIT and foreground targets are allocated on first use by
            // their own `ensure_*`; a promotion keeps what is already there.
            oit_accum_texture: reuse.and_then(|o| o.oit_accum_texture.clone()),
            oit_accum_view: reuse.and_then(|o| o.oit_accum_view.clone()),
            oit_reveal_texture: reuse.and_then(|o| o.oit_reveal_texture.clone()),
            oit_reveal_view: reuse.and_then(|o| o.oit_reveal_view.clone()),
            oit_composite_bind_group: reuse.and_then(|o| o.oit_composite_bind_group.clone()),
            oit_size: reuse.map_or([0, 0], |o| o.oit_size),
            foreground_depth_texture: reuse.and_then(|o| o.foreground_depth_texture.clone()),
            foreground_depth_view: reuse.and_then(|o| o.foreground_depth_view.clone()),
            foreground_depth_only_view: reuse.and_then(|o| o.foreground_depth_only_view.clone()),
            foreground_depth_size: reuse.map_or([0, 0], |o| o.foreground_depth_size),
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
            tone_map_uniform_buf,
            exposure_state_buf,
            exposure_histogram_buf,
            exposure_params_buf,
            exposure_bind_group,
            output_size: [w, h],
            scene_size: [scene_w, scene_h],
            groups,
            output_depth_texture,
            output_depth_view,
            depth_blit_bind_group,
            upscale_texture,
            upscale_view,
            upscale_bind_group,
        }
    }

    /// Rebuild the tone-map bind group for a per-viewport HDR state, binding
    /// each enabled input's live texture view and each disabled input's
    /// neutral placeholder, per `inputs`.
    ///
    /// `slot_overrides` carries this frame's external producer
    /// contributions: an entry replaces the slot's built-in (or placeholder)
    /// view, and the last entry for a slot wins. The caller is responsible
    /// for forcing the matching enable lanes on in `inputs` and the uniform.
    pub(crate) fn rebuild_tone_map_bind_group(
        &self,
        device: &crate::gpu::Device,
        hdr: &mut ViewportHdrState,
        inputs: composite::CompositeInputs,
        slot_overrides: &[(crate::plugin_api::PostEffectSlot, crate::gpu::TextureView)],
    ) {
        let overridden = |slot: crate::plugin_api::PostEffectSlot| {
            slot_overrides
                .iter()
                .rev()
                .find(|(s, _)| *s == slot)
                .map(|(_, v)| v)
        };
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
        let foreground_view = if inputs.foreground {
            hdr.foreground_depth_only_view
                .as_ref()
                .unwrap_or(foreground_placeholder)
        } else {
            foreground_placeholder
        };

        let bloom_view =
            overridden(crate::plugin_api::PostEffectSlot::Bloom).unwrap_or(if inputs.bloom {
                &hdr.bloom.pong_view
            } else {
                bloom_placeholder
            });
        let ao_view = overridden(crate::plugin_api::PostEffectSlot::AmbientOcclusion).unwrap_or(
            if inputs.ssao {
                &hdr.ssao.blur_view
            } else {
                ao_placeholder
            },
        );
        let cs_view = overridden(crate::plugin_api::PostEffectSlot::ContactShadow).unwrap_or(
            if inputs.contact_shadows {
                &hdr.contact_shadow.view
            } else {
                cs_placeholder
            },
        );

        let tone_map_hdr_input: &crate::gpu::TextureView = if inputs.dof {
            &hdr.dof.view
        } else {
            &hdr.hdr_view
        };
        hdr.tone_map_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tone_map_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::HDR_COLOUR,
                    resource: crate::gpu::BindingResource::TextureView(tone_map_hdr_input),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::SAMPLER,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::PARAMS,
                    resource: hdr.tone_map_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::BLOOM,
                    resource: crate::gpu::BindingResource::TextureView(bloom_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::AO,
                    resource: crate::gpu::BindingResource::TextureView(ao_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::CONTACT_SHADOW,
                    resource: crate::gpu::BindingResource::TextureView(cs_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::SCENE_DEPTH,
                    resource: crate::gpu::BindingResource::TextureView(&hdr.hdr_depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::FOREGROUND_DEPTH,
                    resource: crate::gpu::BindingResource::TextureView(foreground_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::EXPOSURE,
                    resource: hdr.exposure_state_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: composite::slot::GRADE_LUT,
                    resource: crate::gpu::BindingResource::TextureView(
                        inputs
                            .grade_lut
                            .and_then(|id| self.content.textures.get(id))
                            .map(|t| &t.view)
                            .unwrap_or(ao_placeholder),
                    ),
                },
            ],
        });

        // The DOF gather pass also reads the foreground coverage mask; rebuild
        // its bind group so the mask view matches this frame.
        if inputs.dof {
            if let Some(dof_bgl) = &self.post.dof.bgl {
                hdr.dof.bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
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
                            resource: hdr.dof.uniform_buf.as_entire_binding(),
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
            .expect("ensure_hdr_infra not called");
        let bgl = self
            .oit
            .composite_bgl
            .as_ref()
            .expect("ensure_hdr_infra not called");
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
