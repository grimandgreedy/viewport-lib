//! Scatter-volume pipeline state and per-frame upload.
//!
//! The scatter pass renders each visible `ScatterVolume` as a separate
//! instanced draw whose vertex shader projects the volume's world bounding box
//! to a screen-space rectangle. Only pixels inside that rectangle execute the
//! ray-march; volumes that do not touch a pixel cost nothing on that pixel.
//!
//! Pipeline layout:
//!
//!   group 0: shared camera (matches mesh / projected_tet bindings)
//!   group 1: per-volume `GpuScatterVolume` uniform with dynamic offset
//!   group 2: per-volume colourmap LUT + 3D density texture + samplers
//!   group 3: shared per-frame uniform (time / blue noise / frame index) +
//!            opaque depth texture + depth sampler
//!
//! The temporal blend is no longer inside the scatter shader -- a separate
//! single-attachment temporal-resolve pass (when `ScatterSettings::temporal`
//! is on) reads `raw_current` and the previous frame's history slot, blends,
//! and writes the new history slot. The composite pass then samples either
//! the history slot (when temporal is on) or `raw_current` (when off) and
//! composites onto the HDR target with premultiplied alpha-over.

use crate::scene::scatter_volume::{
    ColourSource, GpuRefractionVolume, GpuScatterVolume, MAX_SCATTER_VOLUMES, ScatterVolume,
};

/// Scatter-volume (participating media) pipelines, layouts, and per-frame
/// upload buffers. All device-shared and lazily built by the `ensure_scatter_*`
/// methods; the uploaded density textures are keyed elsewhere.
#[derive(Default)]
pub(crate) struct ScatterGpu {
    /// Render pipeline for the scatter-volume pass. None until first item submitted.
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 1 layout (per-volume uniform with dynamic offset).
    pub(crate) per_volume_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Group 2 layout (per-volume LUT + density texture + samplers).
    pub(crate) per_volume_tex_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Group 3 layout (per-frame uniform + opaque depth + samplers).
    pub(crate) frame_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Per-volume uniform buffer holding the packed `GpuScatterVolume` array,
    /// stride-padded for dynamic offsetting.
    pub(crate) per_volume_buffer: Option<crate::gpu::Buffer>,
    /// Bind group for the per-volume uniform (group 1).
    pub(crate) per_volume_bg: Option<crate::gpu::BindGroup>,
    /// Stride between dynamic-offset uniform slots, in bytes.
    pub(crate) per_volume_stride: u32,
    /// Capacity of `per_volume_buffer` in slots.
    pub(crate) per_volume_capacity: u32,
    /// Per-frame uniform buffer (group 3 binding 0).
    pub(crate) frame_uniform_buffer: Option<crate::gpu::Buffer>,
    /// Cache of group 2 bind groups, keyed by `(lut_id, density_id)`.
    pub(crate) per_volume_tex_cache:
        Vec<((usize, crate::resources::VolumeId), crate::gpu::BindGroup)>,
    /// Linear sampler used to read opaque depth in the scatter pass.
    pub(crate) depth_sampler: Option<crate::gpu::Sampler>,
    /// Linear-clamp sampler used to read the colourmap LUT in the scatter pass.
    pub(crate) colourmap_sampler: Option<crate::gpu::Sampler>,
    /// 1x1x1 R32Float fallback view bound at the per-volume 3D density slot.
    pub(crate) density_fallback_view: Option<crate::gpu::TextureView>,
    /// Composite pipeline that blends a scatter intermediate onto the HDR target.
    pub(crate) composite_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout for the composite pass (one sampled RGBA16F + sampler).
    pub(crate) composite_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Bilinear-clamp sampler used by the composite pass.
    pub(crate) composite_sampler: Option<crate::gpu::Sampler>,
    /// Temporal-resolve pipeline: mixes (raw_current, history_prev) into history_new.
    pub(crate) temporal_resolve_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout for the temporal-resolve pass.
    pub(crate) temporal_resolve_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Per-frame uniform buffer for the temporal-resolve pass.
    pub(crate) temporal_resolve_uniform_buffer: Option<crate::gpu::Buffer>,
    /// Refraction pass: per-volume distortion using a noise-driven gradient.
    pub(crate) refraction_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout for the refraction pass's per-volume uniform.
    pub(crate) refraction_per_volume_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Bind group layout for the refraction pass's source-scene + depth bindings.
    pub(crate) refraction_source_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Dynamic-offset uniform buffer holding every refractive volume's params.
    pub(crate) refraction_per_volume_buffer: Option<crate::gpu::Buffer>,
    /// Stride between refractive-volume slots.
    pub(crate) refraction_per_volume_stride: u32,
    /// Capacity (slot count) the refraction per-volume buffer is sized for.
    pub(crate) refraction_per_volume_capacity: u32,
    /// Dynamic-offset bind group for the refraction per-volume uniform buffer.
    pub(crate) refraction_per_volume_bg: Option<crate::gpu::BindGroup>,
    /// Blit pipeline that copies the HDR target into the refraction source texture.
    pub(crate) refraction_blit_pipeline: Option<crate::gpu::RenderPipeline>,
}

/// Per-frame uniform layout shared across every per-volume draw.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub(crate) struct ScatterFrameUniformRaw {
    /// x = elapsed seconds since renderer start. yzw reserved.
    pub time_pack: [f32; 4],
    /// x = global step count, y = blue noise enabled (0/1),
    /// z = frame index low 32, w = reserved.
    pub count_pack: [u32; 4],
}

/// Uniform layout for the temporal-resolve pass.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub(crate) struct ScatterTemporalUniformRaw {
    pub prev_view_proj: [[f32; 4]; 4],
    /// x = blend factor (0..1), y = history valid (0/1),
    /// z = reserved, w = reserved.
    pub temporal_pack: [f32; 4],
}

impl ScatterGpu {
    // ---------------------------------------------------------------------
    // Bind group layouts
    // ---------------------------------------------------------------------

    fn ensure_per_volume_bgl(&mut self, device: &crate::gpu::Device) {
        if self.per_volume_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("scatter_per_volume_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::VERTEX_FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    // GpuScatterVolume = 144 bytes; the actual slot stride is
                    // padded to `min_uniform_buffer_offset_alignment`. The
                    // bound range is exactly the struct size.
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<GpuScatterVolume>() as u64,
                    ),
                },
                count: None,
            }],
        });
        self.per_volume_bgl = Some(bgl);
    }

    fn ensure_per_volume_tex_bgl(&mut self, device: &crate::gpu::Device) {
        if self.per_volume_tex_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("scatter_per_volume_tex_bgl"),
            entries: &[
                // 0: colourmap LUT (256x1 RGBA, used when FLAG_USE_RAMP).
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
                // 1: LUT sampler.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 2: 3D density texture (used when FLAG_USE_DENSITY_TEXTURE).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
                        view_dimension: crate::gpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: 3D density sampler.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(
                        crate::gpu::SamplerBindingType::NonFiltering,
                    ),
                    count: None,
                },
            ],
        });
        self.per_volume_tex_bgl = Some(bgl);
    }

    fn ensure_frame_bgl(&mut self, device: &crate::gpu::Device) {
        if self.frame_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("scatter_frame_bgl"),
            entries: &[
                // 0: per-frame uniform (time, blue noise, frame index).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            ScatterFrameUniformRaw,
                        >()
                            as u64),
                    },
                    count: None,
                },
                // 1: opaque depth texture.
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
                // 2: depth sampler (NonFiltering for the textureLoad path).
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
        self.frame_bgl = Some(bgl);
    }

    fn ensure_temporal_resolve_bgl(&mut self, device: &crate::gpu::Device) {
        if self.temporal_resolve_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("scatter_temporal_resolve_bgl"),
            entries: &[
                // 0: temporal uniform (prev_view_proj + temporal_pack).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            ScatterTemporalUniformRaw,
                        >()
                            as u64),
                    },
                    count: None,
                },
                // 1: raw_current texture (this frame's scatter output).
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
                // 2: history_prev texture.
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
                // 3: bilinear sampler (reuses scatter composite sampler).
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 4: opaque depth texture (for reprojection).
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
                // 5: depth sampler.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(
                        crate::gpu::SamplerBindingType::NonFiltering,
                    ),
                    count: None,
                },
            ],
        });
        self.temporal_resolve_bgl = Some(bgl);
    }

    fn ensure_density_fallback(&mut self, device: &crate::gpu::Device, queue: &crate::gpu::Queue) {
        if self.density_fallback_view.is_some() {
            return;
        }
        let tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("scatter_density_fallback"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D3,
            format: crate::gpu::TextureFormat::R32Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let data: [f32; 1] = [1.0];
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
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
        self.density_fallback_view =
            Some(tex.create_view(&crate::gpu::TextureViewDescriptor::default()));
    }

    fn ensure_depth_sampler(&mut self, device: &crate::gpu::Device) {
        if self.depth_sampler.is_some() {
            return;
        }
        self.depth_sampler = Some(crate::resources::builders::clamp_nearest_sampler(
            device,
            "scatter_depth_sampler",
        ));
    }

    fn ensure_colourmap_sampler(&mut self, device: &crate::gpu::Device) {
        if self.colourmap_sampler.is_some() {
            return;
        }
        self.colourmap_sampler = Some(crate::resources::builders::clamp_linear_sampler(
            device,
            "scatter_colourmap_sampler",
        ));
    }

    // ---------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------

    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        camera_bgl: &crate::gpu::BindGroupLayout,
        colour_format: crate::gpu::TextureFormat,
    ) {
        if self.pipeline.is_some() {
            return;
        }
        self.ensure_per_volume_bgl(device);
        self.ensure_per_volume_tex_bgl(device);
        self.ensure_frame_bgl(device);

        let per_vol = self.per_volume_bgl.as_ref().unwrap();
        let per_tex = self.per_volume_tex_bgl.as_ref().unwrap();
        let frame_bgl = self.frame_bgl.as_ref().unwrap();

        let shader = crate::resources::builders::wgsl_module(
            device,
            "scatter_volume_shader",
            crate::resources::builders::wgsl_source!("scatter_volume"),
        );

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "scatter_volume_pipeline_layout",
            &[camera_bgl, per_vol, per_tex, frame_bgl],
        );

        // Premultiplied alpha-over: per-volume draws composite into the
        // (cleared) raw_current target in back-to-front order.
        let blend = crate::gpu::BlendState {
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

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "scatter_volume_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[],
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: colour_format,
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
                depth_stencil: None,
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        self.pipeline = Some(pipeline);
    }

    pub(crate) fn ensure_composite_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        colour_format: crate::gpu::TextureFormat,
    ) {
        if self.composite_pipeline.is_some() {
            return;
        }
        let bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "scatter_composite_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );
        let sampler =
            crate::resources::builders::clamp_linear_sampler(device, "scatter_composite_sampler");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "scatter_composite_shader",
            crate::resources::builders::wgsl_source!("scatter_composite"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "scatter_composite_pipeline_layout",
            &[&bgl],
        );
        let blend = crate::gpu::BlendState {
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
        let pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "scatter_composite_pipeline",
            &layout,
            &shader,
            colour_format,
            Some(blend),
        );
        self.composite_pipeline = Some(pipeline);
        self.composite_bgl = Some(bgl);
        self.composite_sampler = Some(sampler);
    }

    pub(crate) fn ensure_temporal_resolve_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.temporal_resolve_pipeline.is_some() {
            return;
        }
        self.ensure_temporal_resolve_bgl(device);
        let bgl = self.temporal_resolve_bgl.as_ref().unwrap();
        let shader = crate::resources::builders::wgsl_module(
            device,
            "scatter_temporal_resolve_shader",
            crate::resources::builders::wgsl_source!("scatter_temporal_resolve"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "scatter_temporal_resolve_pipeline_layout",
            &[bgl],
        );
        // History textures are RGBA16F. Blend is None: this pass owns the new
        // history fully and overwrites it.
        let pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "scatter_temporal_resolve_pipeline",
            &layout,
            &shader,
            crate::gpu::TextureFormat::Rgba16Float,
            None,
        );
        self.temporal_resolve_pipeline = Some(pipeline);
    }

    // ---------------------------------------------------------------------
    // Per-frame uniform / bind group construction
    // ---------------------------------------------------------------------

    /// Pack visible volumes into the per-volume dynamic-offset uniform buffer.
    /// Volumes are written in submission order (caller is responsible for
    /// back-to-front sort). Returns the number of slots written.
    pub(crate) fn write_per_volume_buffer(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        volumes: &[(ScatterVolume, f32, u32)],
    ) -> u32 {
        // Stride = aligned per-volume uniform slot size. Recomputed once.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let struct_size = std::mem::size_of::<GpuScatterVolume>() as u64;
        let stride = ((struct_size + align - 1) / align * align).max(struct_size) as u32;
        let capacity = volumes.len().min(MAX_SCATTER_VOLUMES).max(1) as u32;
        let buffer_size = (stride as u64) * (capacity as u64);

        let need_realloc = self.per_volume_buffer.is_none()
            || self.per_volume_stride != stride
            || self.per_volume_capacity < capacity;
        if need_realloc {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("scatter_per_volume_uniform"),
                size: buffer_size,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.per_volume_buffer = Some(buf);
            self.per_volume_stride = stride;
            self.per_volume_capacity = capacity;
            self.per_volume_bg = None;
        }

        // Build the dynamic-offset bind group lazily.
        if self.per_volume_bg.is_none() {
            self.ensure_per_volume_bgl(device);
            let bgl = self.per_volume_bgl.as_ref().unwrap();
            let buf = self.per_volume_buffer.as_ref().unwrap();
            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("scatter_per_volume_bg"),
                layout: bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::Buffer(crate::gpu::BufferBinding {
                        buffer: buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(struct_size),
                    }),
                }],
            });
            self.per_volume_bg = Some(bg);
        }

        // Pack and upload.
        let mut bytes = vec![0u8; buffer_size as usize];
        let mut n: u32 = 0;
        for (volume, mult, flags) in volumes.iter() {
            if n as usize >= MAX_SCATTER_VOLUMES {
                break;
            }
            if let Some(packed) = GpuScatterVolume::pack(volume, *mult, *flags) {
                let offset = (n as usize) * (stride as usize);
                let src = bytemuck::bytes_of(&packed);
                bytes[offset..offset + src.len()].copy_from_slice(src);
                n += 1;
            }
        }
        if let Some(buf) = self.per_volume_buffer.as_ref() {
            queue.write_buffer(
                buf,
                0,
                &bytes[..(n as usize * stride as usize).max(stride as usize)],
            );
        }
        n
    }

    /// Allocate the per-frame uniform buffer. Its size is fixed, so this runs
    /// once and the frame write below only fills it.
    pub(crate) fn ensure_frame_uniform_buffer(&mut self, device: &crate::gpu::Device) {
        self.ensure_frame_bgl(device);
        self.ensure_depth_sampler(device);
        if self.frame_uniform_buffer.is_none() {
            self.frame_uniform_buffer = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("scatter_frame_uniform"),
                size: std::mem::size_of::<ScatterFrameUniformRaw>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    /// Write this frame's time / blue noise / frame index into the uniform.
    pub(crate) fn write_frame_uniform(
        &self,
        queue: &crate::gpu::Queue,
        time_seconds: f32,
        global_steps: u32,
        blue_noise_jitter: bool,
        frame_index: u64,
    ) {
        let raw = ScatterFrameUniformRaw {
            time_pack: [time_seconds, 0.0, 0.0, 0.0],
            count_pack: [
                global_steps.clamp(1, 128),
                if blue_noise_jitter { 1 } else { 0 },
                frame_index as u32,
                0,
            ],
        };
        if let Some(buf) = self.frame_uniform_buffer.as_ref() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&raw));
        }
    }

    /// Build the group 3 bind group over `depth_view`. Built per frame rather
    /// than cached: the scene depth attachment can be reallocated at the same
    /// size, which a size-derived cache key cannot see.
    pub(crate) fn make_frame_bg(
        &self,
        device: &crate::gpu::Device,
        depth_view: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        let bgl = self.frame_bgl.as_ref().unwrap();
        let buf = self.frame_uniform_buffer.as_ref().unwrap();
        let sampler = self.depth_sampler.as_ref().unwrap();
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("scatter_frame_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(depth_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Look up or build a group 2 bind group for the `(lut_id, density)` pair.
    /// Pass `usize::MAX` for `lut_id` or [`VolumeId::INVALID`] for `density` to
    /// bind the fallback. The density half of the cache key carries the volume
    /// handle's generation, so a freed-then-reused slot at the same index does
    /// not hit a bind group built against the previous occupant.
    pub(crate) fn ensure_per_volume_tex_bg(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        res: &crate::resources::DeviceResources,
        lut_id: usize,
        density: crate::resources::VolumeId,
    ) -> crate::gpu::BindGroup {
        self.ensure_per_volume_tex_bgl(device);
        self.ensure_colourmap_sampler(device);
        self.ensure_density_fallback(device, queue);

        let key = (lut_id, density);
        if let Some((_, bg)) = self.per_volume_tex_cache.iter().find(|(k, _)| *k == key) {
            return bg.clone();
        }
        let bgl = self.per_volume_tex_bgl.as_ref().unwrap();
        let lut_sampler = self.colourmap_sampler.as_ref().unwrap();
        let density_sampler = self.depth_sampler.as_ref().unwrap();
        let lut_view: &crate::gpu::TextureView = if lut_id == usize::MAX {
            &res.content.fallback_lut_view
        } else {
            res.content
                .colourmap_views
                .get(lut_id)
                .unwrap_or(&res.content.fallback_lut_view)
        };
        let density_fallback = self.density_fallback_view.as_ref().unwrap();
        let density_view: &crate::gpu::TextureView =
            if density == crate::resources::VolumeId::INVALID {
                density_fallback
            } else {
                res.content
                    .volume_textures
                    .get(density)
                    .map(|(_, v)| v)
                    .unwrap_or(density_fallback)
            };
        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("scatter_per_volume_tex_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(lut_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(density_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::Sampler(density_sampler),
                },
            ],
        });
        self.per_volume_tex_cache.push((key, bg.clone()));
        bg
    }

    /// Resolve a volume's `(lut_id, density)` pair. `usize::MAX` / [`VolumeId::INVALID`]
    /// indicate the fallback should be bound.
    pub(crate) fn volume_tex_ids(volume: &ScatterVolume) -> (usize, crate::resources::VolumeId) {
        let lut_id = match volume.colour {
            ColourSource::Ramp(id) => id.0,
            _ => usize::MAX,
        };
        let density = volume
            .density_texture
            .unwrap_or(crate::resources::VolumeId::INVALID);
        (lut_id, density)
    }

    /// Clear the per-volume texture bind group cache. Call when the
    /// underlying texture vectors may have been mutated (uploads added).
    pub(crate) fn clear_per_volume_tex_cache(&mut self) {
        self.per_volume_tex_cache.clear();
    }

    // ---------------------------------------------------------------------
    // Composite + temporal-resolve helpers
    // ---------------------------------------------------------------------

    pub(crate) fn make_composite_bg(
        &self,
        device: &crate::gpu::Device,
        source_view: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        let bgl = self.composite_bgl.as_ref().unwrap();
        let sampler = self.composite_sampler.as_ref().unwrap();
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("scatter_composite_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(source_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Build a temporal-resolve bind group sampling `(raw_view, history_view)`
    /// alongside the bound depth and uniform.
    pub(crate) fn make_temporal_resolve_bg(
        &self,
        device: &crate::gpu::Device,
        raw_view: &crate::gpu::TextureView,
        history_view: &crate::gpu::TextureView,
        depth_view: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        let bgl = self.temporal_resolve_bgl.as_ref().unwrap();
        let buf = self.temporal_resolve_uniform_buffer.as_ref().unwrap();
        let bilinear = self.composite_sampler.as_ref().unwrap();
        let depth_sampler = self.depth_sampler.as_ref().unwrap();
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("scatter_temporal_resolve_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(raw_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(history_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::Sampler(bilinear),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(depth_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::Sampler(depth_sampler),
                },
            ],
        })
    }

    // ---------------------------------------------------------------------
    // Refraction pass
    // ---------------------------------------------------------------------

    fn ensure_refraction_per_volume_bgl(&mut self, device: &crate::gpu::Device) {
        if self.refraction_per_volume_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("scatter_refraction_per_volume_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::VERTEX_FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                        GpuRefractionVolume,
                    >() as u64),
                },
                count: None,
            }],
        });
        self.refraction_per_volume_bgl = Some(bgl);
    }

    fn ensure_refraction_source_bgl(&mut self, device: &crate::gpu::Device) {
        if self.refraction_source_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("scatter_refraction_source_bgl"),
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
            ],
        });
        self.refraction_source_bgl = Some(bgl);
    }

    pub(crate) fn ensure_refraction_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        camera_bgl: &crate::gpu::BindGroupLayout,
        colour_format: crate::gpu::TextureFormat,
    ) {
        if self.refraction_pipeline.is_some() {
            return;
        }
        self.ensure_refraction_per_volume_bgl(device);
        self.ensure_refraction_source_bgl(device);
        self.ensure_composite_pipeline(device, colour_format);

        let per_vol = self.refraction_per_volume_bgl.as_ref().unwrap();
        let source_bgl = self.refraction_source_bgl.as_ref().unwrap();

        let shader = crate::resources::builders::wgsl_module(
            device,
            "scatter_refraction_shader",
            crate::resources::builders::wgsl_source!("scatter_refraction"),
        );

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "scatter_refraction_pipeline_layout",
            &[camera_bgl, per_vol, source_bgl],
        );

        // Replace blend: the distorted sample overwrites the HDR pixel before
        // the scatter pass composites on top.
        let pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "scatter_refraction_pipeline",
            &layout,
            &shader,
            colour_format,
            None,
        );

        self.refraction_pipeline = Some(pipeline);
    }

    /// Build a render pipeline that samples a source colour texture (via the
    /// composite BGL / shader) and writes it to a render target with replace
    /// blend. Used to copy the HDR scene into the refraction source texture
    /// before the per-volume distortion runs.
    pub(crate) fn ensure_refraction_blit_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        colour_format: crate::gpu::TextureFormat,
    ) {
        if self.refraction_blit_pipeline.is_some() {
            return;
        }
        self.ensure_composite_pipeline(device, colour_format);
        let bgl = self.composite_bgl.as_ref().unwrap();
        let shader = crate::resources::builders::wgsl_module(
            device,
            "scatter_refraction_blit_shader",
            crate::resources::builders::wgsl_source!("scatter_composite"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "scatter_refraction_blit_pipeline_layout",
            &[bgl],
        );
        let pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "scatter_refraction_blit_pipeline",
            &layout,
            &shader,
            colour_format,
            None,
        );
        self.refraction_blit_pipeline = Some(pipeline);
    }

    /// Size the refraction per-volume buffer and its bind group for `count`
    /// volumes. Separate from the write below so the write, which needs the
    /// frame's animation clock, can run from a shared borrow at encode time.
    pub(crate) fn ensure_refraction_per_volume_buffer(
        &mut self,
        device: &crate::gpu::Device,
        count: usize,
    ) {
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let struct_size = std::mem::size_of::<GpuRefractionVolume>() as u64;
        let stride = ((struct_size + align - 1) / align * align).max(struct_size) as u32;
        let capacity = count.min(MAX_SCATTER_VOLUMES).max(1) as u32;
        let buffer_size = (stride as u64) * (capacity as u64);

        let need_realloc = self.refraction_per_volume_buffer.is_none()
            || self.refraction_per_volume_stride != stride
            || self.refraction_per_volume_capacity < capacity;
        if need_realloc {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("scatter_refraction_per_volume_uniform"),
                size: buffer_size,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.refraction_per_volume_buffer = Some(buf);
            self.refraction_per_volume_stride = stride;
            self.refraction_per_volume_capacity = capacity;
            self.refraction_per_volume_bg = None;
        }

        if self.refraction_per_volume_bg.is_none() {
            self.ensure_refraction_per_volume_bgl(device);
            let bgl = self.refraction_per_volume_bgl.as_ref().unwrap();
            let buf = self.refraction_per_volume_buffer.as_ref().unwrap();
            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("scatter_refraction_per_volume_bg"),
                layout: bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::Buffer(crate::gpu::BufferBinding {
                        buffer: buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(struct_size),
                    }),
                }],
            });
            self.refraction_per_volume_bg = Some(bg);
        }
    }

    /// Pack visible refractive volumes into the dynamic-offset uniform buffer
    /// at the frame's animation clock. Returns the number of slots written.
    pub(crate) fn write_refraction_per_volume_buffer(
        &self,
        queue: &crate::gpu::Queue,
        volumes: &[(ScatterVolume, f32)],
        time_seconds: f32,
    ) -> u32 {
        let stride = self.refraction_per_volume_stride;
        if stride == 0 {
            return 0;
        }
        let capacity = self.refraction_per_volume_capacity as usize;
        let mut bytes = vec![0u8; stride as usize * capacity.max(1)];
        let mut n: u32 = 0;
        for (volume, _) in volumes.iter() {
            if n as usize >= capacity.min(MAX_SCATTER_VOLUMES) {
                break;
            }
            if let Some(packed) = GpuRefractionVolume::pack(volume, time_seconds) {
                let offset = (n as usize) * (stride as usize);
                let src = bytemuck::bytes_of(&packed);
                bytes[offset..offset + src.len()].copy_from_slice(src);
                n += 1;
            }
        }
        if let Some(buf) = self.refraction_per_volume_buffer.as_ref() {
            queue.write_buffer(
                buf,
                0,
                &bytes[..(n as usize * stride as usize).max(stride as usize)],
            );
        }
        n
    }

    /// Build the bind group sampling the refraction source texture + depth.
    pub(crate) fn make_refraction_source_bg(
        &self,
        device: &crate::gpu::Device,
        source_view: &crate::gpu::TextureView,
        depth_view: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        let bgl = self.refraction_source_bgl.as_ref().unwrap();
        let sampler = self.composite_sampler.as_ref().unwrap();
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("scatter_refraction_source_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(source_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(depth_view),
                },
            ],
        })
    }

    /// Allocate the temporal-resolve uniform buffer. Fixed size, so this runs
    /// once and the frame write below only fills it.
    pub(crate) fn ensure_temporal_uniform_buffer(&mut self, device: &crate::gpu::Device) {
        if self.temporal_resolve_uniform_buffer.is_none() {
            self.temporal_resolve_uniform_buffer =
                Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("scatter_temporal_resolve_uniform"),
                    size: std::mem::size_of::<ScatterTemporalUniformRaw>() as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
    }

    /// Write the temporal-resolve uniform.
    pub(crate) fn write_temporal_uniform(
        &self,
        queue: &crate::gpu::Queue,
        prev_view_proj: [[f32; 4]; 4],
        blend: f32,
        history_valid: bool,
    ) {
        let raw = ScatterTemporalUniformRaw {
            prev_view_proj,
            temporal_pack: [
                blend.clamp(0.0, 0.99),
                if history_valid { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        if let Some(buf) = self.temporal_resolve_uniform_buffer.as_ref() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&raw));
        }
    }
}

/// Per-viewport scatter intermediates: the accumulation target the per-volume
/// draws write into, the two history slots the temporal blend ping-pongs
/// between, and the scene-colour copy the refraction pass samples.
///
/// Everything held here is allocated by the plugin, and every cached bind
/// group reads a view in this struct. Bind groups over the lib's own scene
/// attachments are deliberately absent: those are built per frame in `encode`,
/// because the HDR attachments can be reallocated at the same size and a
/// cached bind group would then point at a dead view.
pub(crate) struct ScatterViewportState {
    // Textures keep the GPU allocation alive; views are sampled or rendered
    // into.
    /// Per-volume scatter draws accumulate into this target each frame.
    /// Cleared at the start of the scatter pass.
    #[allow(dead_code)]
    pub raw_current_texture: crate::gpu::Texture,
    pub raw_current_view: crate::gpu::TextureView,
    /// History ping-pong. The temporal-resolve pass reads one slot
    /// (history_prev) and writes the other (history_new). `parity` selects.
    #[allow(dead_code)]
    pub history_a_texture: crate::gpu::Texture,
    pub history_a_view: crate::gpu::TextureView,
    #[allow(dead_code)]
    pub history_b_texture: crate::gpu::Texture,
    pub history_b_view: crate::gpu::TextureView,
    /// Composite bind group reading the raw-current texture.
    /// Used when temporal accumulation is disabled.
    pub composite_bg_raw: crate::gpu::BindGroup,
    /// Composite bind groups reading either history slot, used as the source
    /// after the temporal-resolve pass has written history_new.
    pub composite_bg_history_a: crate::gpu::BindGroup,
    pub composite_bg_history_b: crate::gpu::BindGroup,
    /// Current allocated intermediate size, [width, height].
    pub size: [u32; 2],
    /// Whether `size` reflects the downsampled (half-res) allocation.
    pub downsampled: bool,
    /// Index of the history slot the next frame writes to (0 = A, 1 = B).
    /// The other slot is read as the previous-frame history.
    pub parity: u32,
    /// True when the history slot opposite `parity` holds a usable
    /// previous-frame composite result.
    pub history_valid: bool,
    /// Previous frame's view-projection (row-major mat4).
    pub prev_view_proj: [[f32; 4]; 4],
    /// Scene colour copy sampled by the refraction pass. Allocated on demand
    /// when at least one volume has refraction enabled. Matches the HDR
    /// target's size and format.
    #[allow(dead_code)]
    pub refraction_source_texture: Option<crate::gpu::Texture>,
    /// View paired with `refraction_source_texture`. Bound as the source
    /// during the refraction pass and as the render target during the
    /// preceding blit-copy of the HDR scene.
    pub refraction_source_view: Option<crate::gpu::TextureView>,
    /// Allocated size of the refraction source, matched to the HDR target.
    pub refraction_source_size: [u32; 2],
}

impl ScatterViewportState {
    /// Allocate the accumulation and history targets at `size`, along with the
    /// composite bind groups that read them.
    pub(crate) fn new(
        device: &crate::gpu::Device,
        gpu: &ScatterGpu,
        size: [u32; 2],
        downsampled: bool,
    ) -> Self {
        let make_tex = |label: &str| {
            device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some(label),
                size: crate::gpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: crate::gpu::TextureFormat::Rgba16Float,
                usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                    | crate::gpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let raw_current_texture = make_tex("scatter_raw_current");
        let history_a_texture = make_tex("scatter_history_a");
        let history_b_texture = make_tex("scatter_history_b");
        let raw_current_view =
            raw_current_texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        let history_a_view =
            history_a_texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        let history_b_view =
            history_b_texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        Self {
            composite_bg_raw: gpu.make_composite_bg(device, &raw_current_view),
            composite_bg_history_a: gpu.make_composite_bg(device, &history_a_view),
            composite_bg_history_b: gpu.make_composite_bg(device, &history_b_view),
            raw_current_texture,
            raw_current_view,
            history_a_texture,
            history_a_view,
            history_b_texture,
            history_b_view,
            size,
            downsampled,
            parity: 0,
            history_valid: false,
            prev_view_proj: [[0.0; 4]; 4],
            refraction_source_texture: None,
            refraction_source_view: None,
            refraction_source_size: [0, 0],
        }
    }

    /// Allocate (or resize) the scene-colour copy the refraction pass reads.
    /// Sized to the HDR target rather than to the scatter intermediates, which
    /// may be half-resolution.
    pub(crate) fn ensure_refraction_source(&mut self, device: &crate::gpu::Device, size: [u32; 2]) {
        if self.refraction_source_view.is_some() && self.refraction_source_size == size {
            return;
        }
        let tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("scatter_refraction_source"),
            size: crate::gpu::Extent3d {
                width: size[0].max(1),
                height: size[1].max(1),
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
        self.refraction_source_view =
            Some(tex.create_view(&crate::gpu::TextureViewDescriptor::default()));
        self.refraction_source_texture = Some(tex);
        self.refraction_source_size = size;
    }
}
