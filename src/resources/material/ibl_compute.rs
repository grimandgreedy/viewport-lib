//! GPU compute path for IBL precomputation.
//!
//! Three compute shaders replace the CPU reference port for the heavy parts of
//! `upload_environment_map`:
//!
//! - `ibl_irradiance.wgsl`: cosine-weighted hemisphere convolution
//! - `ibl_prefilter.wgsl` : GGX importance-sampled roughness convolution (per mip)
//! - `ibl_brdf_lut.wgsl`  : Hsplit-sum BRDF integration
//!
//! Selected at runtime via [`compute_supported`]. When unavailable, callers must
//! fall back to the CPU path in `environment.rs`.

use crate::gpu::util::DeviceExt;

const PREFILTER_PARAMS_SIZE: u64 = 16; // 4 floats (1 used, 3 pad) = 16 bytes

/// Return whether the device supports the storage-texture features the compute
/// path requires.
///
/// Currently gated on [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`],
/// which is the WebGPU mechanism that exposes Rgba16Float storage-write access
/// on adapters that allow it. Consumers that want the fast IBL path should
/// include this feature in their `request_device` call.
pub(crate) fn compute_supported(device: &crate::gpu::Device) -> bool {
    device
        .features()
        .contains(crate::gpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
}

/// Upload the source HDR equirect as a sampleable Rgba16Float texture
/// (the input to the irradiance and prefilter compute dispatches).
fn upload_source(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> crate::gpu::Texture {
    use rayon::prelude::*;
    let half: Vec<u16> = pixels
        .par_iter()
        .map(|&f| half::f16::from_f32(f).to_bits())
        .collect();
    let tex = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some("ibl_skybox"),
        size: crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::Rgba16Float,
        usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        crate::gpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: crate::gpu::Origin3d::ZERO,
            aspect: crate::gpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&half),
        crate::gpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    tex
}

/// Sampler shared by all compute dispatches that read the source equirect.
fn make_sampler(device: &crate::gpu::Device) -> crate::gpu::Sampler {
    crate::resources::builders::env_sampler(device, "ibl_compute_sampler")
}

/// BGL shared by irradiance + prefilter (params buffer is optional via
/// distinct BGLs in practice; we use two BGLs).
struct ConvolutionBgls {
    irradiance_bgl: crate::gpu::BindGroupLayout,
    prefilter_bgl: crate::gpu::BindGroupLayout,
    brdf_bgl: crate::gpu::BindGroupLayout,
}

fn make_bgls(device: &crate::gpu::Device) -> ConvolutionBgls {
    let irradiance_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("ibl_irradiance_bgl"),
        entries: &[
            crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::COMPUTE,
                ty: crate::gpu::BindingType::Texture {
                    sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    view_dimension: crate::gpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            crate::gpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: crate::gpu::ShaderStages::COMPUTE,
                ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                count: None,
            },
            crate::gpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: crate::gpu::ShaderStages::COMPUTE,
                ty: crate::gpu::BindingType::StorageTexture {
                    access: crate::gpu::StorageTextureAccess::WriteOnly,
                    format: crate::gpu::TextureFormat::Rgba16Float,
                    view_dimension: crate::gpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let prefilter_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("ibl_prefilter_bgl"),
        entries: &[
            crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::COMPUTE,
                ty: crate::gpu::BindingType::Texture {
                    sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    view_dimension: crate::gpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            crate::gpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: crate::gpu::ShaderStages::COMPUTE,
                ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                count: None,
            },
            crate::gpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: crate::gpu::ShaderStages::COMPUTE,
                ty: crate::gpu::BindingType::StorageTexture {
                    access: crate::gpu::StorageTextureAccess::WriteOnly,
                    format: crate::gpu::TextureFormat::Rgba16Float,
                    view_dimension: crate::gpu::TextureViewDimension::D2,
                },
                count: None,
            },
            crate::gpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: crate::gpu::ShaderStages::COMPUTE,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let brdf_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("ibl_brdf_bgl"),
        entries: &[crate::gpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: crate::gpu::ShaderStages::COMPUTE,
            ty: crate::gpu::BindingType::StorageTexture {
                access: crate::gpu::StorageTextureAccess::WriteOnly,
                format: crate::gpu::TextureFormat::Rgba16Float,
                view_dimension: crate::gpu::TextureViewDimension::D2,
            },
            count: None,
        }],
    });

    ConvolutionBgls {
        irradiance_bgl,
        prefilter_bgl,
        brdf_bgl,
    }
}

/// Build the irradiance compute pipeline.
fn build_irradiance_pipeline(
    device: &crate::gpu::Device,
    bgl: &crate::gpu::BindGroupLayout,
) -> crate::gpu::ComputePipeline {
    let shader = crate::resources::builders::wgsl_module(
        device,
        "ibl_irradiance_shader",
        crate::resources::builders::wgsl_source!("ibl_irradiance"),
    );
    let layout =
        crate::resources::builders::pipeline_layout(device, "ibl_irradiance_layout", &[bgl]);
    crate::resources::builders::compute_pipeline(
        device,
        "ibl_irradiance_pipeline",
        &layout,
        &shader,
        "cs_main",
    )
}

fn build_prefilter_pipeline(
    device: &crate::gpu::Device,
    bgl: &crate::gpu::BindGroupLayout,
) -> crate::gpu::ComputePipeline {
    let shader = crate::resources::builders::wgsl_module(
        device,
        "ibl_prefilter_shader",
        crate::resources::builders::wgsl_source!("ibl_prefilter"),
    );
    let layout =
        crate::resources::builders::pipeline_layout(device, "ibl_prefilter_layout", &[bgl]);
    crate::resources::builders::compute_pipeline(
        device,
        "ibl_prefilter_pipeline",
        &layout,
        &shader,
        "cs_main",
    )
}

fn build_brdf_pipeline(
    device: &crate::gpu::Device,
    bgl: &crate::gpu::BindGroupLayout,
) -> crate::gpu::ComputePipeline {
    let shader = crate::resources::builders::wgsl_module(
        device,
        "ibl_brdf_lut_shader",
        crate::resources::builders::wgsl_source!("ibl_brdf_lut"),
    );
    let layout = crate::resources::builders::pipeline_layout(device, "ibl_brdf_lut_layout", &[bgl]);
    crate::resources::builders::compute_pipeline(
        device,
        "ibl_brdf_lut_pipeline",
        &layout,
        &shader,
        "cs_main",
    )
}

/// Result of a GPU IBL precomputation.
pub(crate) struct IblComputeResult {
    pub skybox_texture: crate::gpu::Texture,
    pub skybox_view: crate::gpu::TextureView,
    pub irradiance_texture: crate::gpu::Texture,
    pub irradiance_view: crate::gpu::TextureView,
    pub prefilter_texture: crate::gpu::Texture,
    pub prefilter_view: crate::gpu::TextureView,
    /// Generated only if requested (skipped when a cached LUT already exists).
    pub brdf_texture: Option<crate::gpu::Texture>,
    pub brdf_view: Option<crate::gpu::TextureView>,
    /// Submission index for the queue submit that ran the compute passes.
    /// Callers gate downstream work on this submission completing.
    pub submission: crate::gpu::SubmissionIndex,
}

/// Run the full IBL precomputation on the GPU.
///
/// If `compute_brdf` is `false`, the BRDF LUT step is skipped (caller is reusing
/// a previously-cached LUT).
pub(crate) fn compute_ibl(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pixels: &[f32],
    width: u32,
    height: u32,
    compute_brdf: bool,
) -> IblComputeResult {
    let irr_w = 64u32;
    let irr_h = 32u32;
    let prefilter_w = 128u32;
    let prefilter_h = 64u32;
    let prefilter_mips = 5u32;
    let brdf_size = 128u32;

    // ----- Source skybox -----
    let skybox_texture = upload_source(device, queue, pixels, width, height);
    let skybox_view = skybox_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

    // ----- Destination textures (Rgba16Float, STORAGE + TEXTURE binding) -----
    let irradiance_texture = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some("ibl_irradiance"),
        size: crate::gpu::Extent3d {
            width: irr_w,
            height: irr_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::Rgba16Float,
        usage: crate::gpu::TextureUsages::TEXTURE_BINDING
            | crate::gpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let irradiance_view =
        irradiance_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

    let prefilter_texture = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some("ibl_prefiltered"),
        size: crate::gpu::Extent3d {
            width: prefilter_w,
            height: prefilter_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: prefilter_mips,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::Rgba16Float,
        usage: crate::gpu::TextureUsages::TEXTURE_BINDING
            | crate::gpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let prefilter_view =
        prefilter_texture.create_view(&crate::gpu::TextureViewDescriptor::default());

    let sampler = make_sampler(device);
    let bgls = make_bgls(device);
    let irradiance_pipeline = build_irradiance_pipeline(device, &bgls.irradiance_bgl);
    let prefilter_pipeline = build_prefilter_pipeline(device, &bgls.prefilter_bgl);

    let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("ibl_compute_encoder"),
    });

    // ----- Irradiance dispatch -----
    {
        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("ibl_irradiance_bg"),
            layout: &bgls.irradiance_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&skybox_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(&sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(&irradiance_view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
            label: Some("ibl_irradiance_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&irradiance_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(irr_w.div_ceil(8), irr_h.div_ceil(8), 1);
    }

    // ----- Prefilter dispatches (one per mip) -----
    for mip in 0..prefilter_mips {
        let mip_w = (prefilter_w >> mip).max(1);
        let mip_h = (prefilter_h >> mip).max(1);
        let roughness = mip as f32 / (prefilter_mips - 1).max(1) as f32;

        let params = [roughness, 0.0f32, 0.0, 0.0];
        let params_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("ibl_prefilter_params"),
            contents: bytemuck::cast_slice(&params),
            usage: crate::gpu::BufferUsages::UNIFORM,
        });
        debug_assert_eq!(std::mem::size_of_val(&params) as u64, PREFILTER_PARAMS_SIZE);

        let mip_view = prefilter_texture.create_view(&crate::gpu::TextureViewDescriptor {
            label: Some("ibl_prefilter_mip_view"),
            base_mip_level: mip,
            mip_level_count: Some(1),
            ..Default::default()
        });

        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("ibl_prefilter_bg"),
            layout: &bgls.prefilter_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&skybox_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(&sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(&mip_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
            label: Some("ibl_prefilter_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&prefilter_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(mip_w.div_ceil(8), mip_h.div_ceil(8), 1);
    }

    // ----- BRDF LUT (one-time) -----
    let (brdf_texture, brdf_view) = if compute_brdf {
        let brdf_pipeline = build_brdf_pipeline(device, &bgls.brdf_bgl);
        let brdf_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("ibl_brdf_lut"),
            size: crate::gpu::Extent3d {
                width: brdf_size,
                height: brdf_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba16Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                | crate::gpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let brdf_v = brdf_tex.create_view(&crate::gpu::TextureViewDescriptor::default());

        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("ibl_brdf_bg"),
            layout: &bgls.brdf_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: crate::gpu::BindingResource::TextureView(&brdf_v),
            }],
        });

        let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
            label: Some("ibl_brdf_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&brdf_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(brdf_size.div_ceil(8), brdf_size.div_ceil(8), 1);
        drop(pass);

        (Some(brdf_tex), Some(brdf_v))
    } else {
        (None, None)
    };

    let submission = queue.submit(std::iter::once(encoder.finish()));

    // Callers gate on the returned submission index instead of blocking
    // here. The synchronous `upload_environment_map` wrapper drains the
    // upload-job runner until the matching job reports Ready.

    IblComputeResult {
        skybox_texture,
        skybox_view,
        irradiance_texture,
        irradiance_view,
        prefilter_texture,
        prefilter_view,
        brdf_texture,
        brdf_view,
        submission,
    }
}
