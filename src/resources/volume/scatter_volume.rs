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
    ColourSource, GpuRefractionVolume, GpuScatterVolume, ScatterVolume,
};

/// Hard cap on the number of volumes per frame. Same as the original cap; the
/// per-volume draw flow handles up to this many active volumes.
pub const MAX_SCATTER_VOLUMES: usize = 16;

/// Scatter-volume (participating media) pipelines, layouts, and per-frame
/// upload buffers. All device-shared and lazily built by the `ensure_scatter_*`
/// methods; the uploaded density textures are keyed elsewhere.
#[derive(Default)]
pub(crate) struct ScatterResources {
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
    pub(crate) per_volume_tex_cache: Vec<((usize, usize), crate::gpu::BindGroup)>,
    /// Bind group for group 3, rebuilt when opaque depth view changes.
    pub(crate) frame_bg: Option<crate::gpu::BindGroup>,
    /// Linear sampler used to read opaque depth in the scatter pass.
    pub(crate) depth_sampler: Option<crate::gpu::Sampler>,
    /// Linear-clamp sampler used to read the colourmap LUT in the scatter pass.
    pub(crate) colourmap_sampler: Option<crate::gpu::Sampler>,
    /// 1x1x1 R32Float fallback view bound at the per-volume 3D density slot.
    pub(crate) density_fallback_view: Option<crate::gpu::TextureView>,
    /// Token combining (depth view, frame uniform buffer) for `frame_bg` reuse.
    pub(crate) bound_depth: u64,
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

impl crate::resources::DeviceResources {
    // ---------------------------------------------------------------------
    // Bind group layouts
    // ---------------------------------------------------------------------

    fn ensure_scatter_per_volume_bgl(&mut self, device: &crate::gpu::Device) {
        if self.scatter.per_volume_bgl.is_some() {
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
        self.scatter.per_volume_bgl = Some(bgl);
    }

    fn ensure_scatter_per_volume_tex_bgl(&mut self, device: &crate::gpu::Device) {
        if self.scatter.per_volume_tex_bgl.is_some() {
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
        self.scatter.per_volume_tex_bgl = Some(bgl);
    }

    fn ensure_scatter_frame_bgl(&mut self, device: &crate::gpu::Device) {
        if self.scatter.frame_bgl.is_some() {
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
        self.scatter.frame_bgl = Some(bgl);
    }

    fn ensure_scatter_temporal_resolve_bgl(&mut self, device: &crate::gpu::Device) {
        if self.scatter.temporal_resolve_bgl.is_some() {
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
        self.scatter.temporal_resolve_bgl = Some(bgl);
    }

    fn ensure_scatter_density_fallback(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        if self.scatter.density_fallback_view.is_some() {
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
        self.scatter.density_fallback_view =
            Some(tex.create_view(&crate::gpu::TextureViewDescriptor::default()));
    }

    fn ensure_scatter_depth_sampler(&mut self, device: &crate::gpu::Device) {
        if self.scatter.depth_sampler.is_some() {
            return;
        }
        self.scatter.depth_sampler = Some(crate::resources::builders::clamp_nearest_sampler(
            device,
            "scatter_depth_sampler",
        ));
    }

    fn ensure_scatter_colourmap_sampler(&mut self, device: &crate::gpu::Device) {
        if self.scatter.colourmap_sampler.is_some() {
            return;
        }
        self.scatter.colourmap_sampler = Some(crate::resources::builders::clamp_linear_sampler(
            device,
            "scatter_colourmap_sampler",
        ));
    }

    // ---------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------

    pub(crate) fn ensure_scatter_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        colour_format: crate::gpu::TextureFormat,
    ) {
        if self.scatter.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        self.ensure_scatter_per_volume_bgl(device);
        self.ensure_scatter_per_volume_tex_bgl(device);
        self.ensure_scatter_frame_bgl(device);

        let per_vol = self.scatter.per_volume_bgl.as_ref().unwrap();
        let per_tex = self.scatter.per_volume_tex_bgl.as_ref().unwrap();
        let frame_bgl = self.scatter.frame_bgl.as_ref().unwrap();

        let shader = crate::resources::builders::wgsl_module(
            device,
            "scatter_volume_shader",
            crate::resources::builders::wgsl_source!("scatter_volume"),
        );

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "scatter_volume_pipeline_layout",
            &[&self.camera_bind_group_layout, per_vol, per_tex, frame_bgl],
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

        self.scatter.pipeline = Some(pipeline);
    }

    pub(crate) fn ensure_scatter_composite_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        colour_format: crate::gpu::TextureFormat,
    ) {
        if self.scatter.composite_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
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
        self.scatter.composite_pipeline = Some(pipeline);
        self.scatter.composite_bgl = Some(bgl);
        self.scatter.composite_sampler = Some(sampler);
    }

    pub(crate) fn ensure_scatter_temporal_resolve_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.scatter.temporal_resolve_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        self.ensure_scatter_temporal_resolve_bgl(device);
        let bgl = self.scatter.temporal_resolve_bgl.as_ref().unwrap();
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
        self.scatter.temporal_resolve_pipeline = Some(pipeline);
    }

    // ---------------------------------------------------------------------
    // Per-frame uniform / bind group construction
    // ---------------------------------------------------------------------

    /// Pack visible volumes into the per-volume dynamic-offset uniform buffer.
    /// Volumes are written in submission order (caller is responsible for
    /// back-to-front sort). Returns the number of slots written.
    pub(crate) fn write_scatter_per_volume_buffer(
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

        let need_realloc = self.scatter.per_volume_buffer.is_none()
            || self.scatter.per_volume_stride != stride
            || self.scatter.per_volume_capacity < capacity;
        if need_realloc {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("scatter_per_volume_uniform"),
                size: buffer_size,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter.per_volume_buffer = Some(buf);
            self.scatter.per_volume_stride = stride;
            self.scatter.per_volume_capacity = capacity;
            self.scatter.per_volume_bg = None;
        }

        // Build the dynamic-offset bind group lazily.
        if self.scatter.per_volume_bg.is_none() {
            self.ensure_scatter_per_volume_bgl(device);
            let bgl = self.scatter.per_volume_bgl.as_ref().unwrap();
            let buf = self.scatter.per_volume_buffer.as_ref().unwrap();
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
            self.scatter.per_volume_bg = Some(bg);
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
        if let Some(buf) = self.scatter.per_volume_buffer.as_ref() {
            queue.write_buffer(
                buf,
                0,
                &bytes[..(n as usize * stride as usize).max(stride as usize)],
            );
        }
        n
    }

    /// Write the per-frame uniform (time / blue noise / frame index) and
    /// build / rebuild the frame bind group for the given opaque depth view.
    pub(crate) fn write_scatter_frame_uniform(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        depth_view: &crate::gpu::TextureView,
        depth_view_token: u64,
        time_seconds: f32,
        global_steps: u32,
        blue_noise_jitter: bool,
        frame_index: u64,
    ) {
        self.ensure_scatter_frame_bgl(device);
        self.ensure_scatter_depth_sampler(device);
        if self.scatter.frame_uniform_buffer.is_none() {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("scatter_frame_uniform"),
                size: std::mem::size_of::<ScatterFrameUniformRaw>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter.frame_uniform_buffer = Some(buf);
            self.scatter.frame_bg = None;
        }
        let raw = ScatterFrameUniformRaw {
            time_pack: [time_seconds, 0.0, 0.0, 0.0],
            count_pack: [
                global_steps.clamp(1, 128),
                if blue_noise_jitter { 1 } else { 0 },
                frame_index as u32,
                0,
            ],
        };
        if let Some(buf) = self.scatter.frame_uniform_buffer.as_ref() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&raw));
        }
        if self.scatter.frame_bg.is_none() || self.scatter.bound_depth != depth_view_token {
            let bgl = self.scatter.frame_bgl.as_ref().unwrap();
            let buf = self.scatter.frame_uniform_buffer.as_ref().unwrap();
            let sampler = self.scatter.depth_sampler.as_ref().unwrap();
            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
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
            });
            self.scatter.frame_bg = Some(bg);
            self.scatter.bound_depth = depth_view_token;
        }
    }

    /// Look up or build a group 2 bind group for the (lut_id, density_id)
    /// pair. Pass `u32::MAX` for either id to bind the fallback.
    pub(crate) fn ensure_scatter_per_volume_tex_bg(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        lut_id: usize,
        density_id: usize,
    ) -> crate::gpu::BindGroup {
        self.ensure_scatter_per_volume_tex_bgl(device);
        self.ensure_scatter_colourmap_sampler(device);
        self.ensure_scatter_density_fallback(device, queue);

        let key = (lut_id, density_id);
        if let Some((_, bg)) = self
            .scatter
            .per_volume_tex_cache
            .iter()
            .find(|(k, _)| *k == key)
        {
            return bg.clone();
        }
        let bgl = self.scatter.per_volume_tex_bgl.as_ref().unwrap();
        let lut_sampler = self.scatter.colourmap_sampler.as_ref().unwrap();
        let density_sampler = self.scatter.depth_sampler.as_ref().unwrap();
        let lut_view: &crate::gpu::TextureView = if lut_id == usize::MAX {
            &self.content.fallback_lut_view
        } else {
            self.content
                .colourmap_views
                .get(lut_id)
                .unwrap_or(&self.content.fallback_lut_view)
        };
        let density_fallback = self.scatter.density_fallback_view.as_ref().unwrap();
        let density_view: &crate::gpu::TextureView = if density_id == usize::MAX {
            density_fallback
        } else {
            self.content
                .volume_textures
                .get(density_id)
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
        self.scatter.per_volume_tex_cache.push((key, bg.clone()));
        bg
    }

    /// Resolve a volume's `(lut_id, density_id)` pair. `usize::MAX` indicates
    /// the fallback should be bound.
    pub(crate) fn scatter_volume_tex_ids(volume: &ScatterVolume) -> (usize, usize) {
        let lut_id = match volume.colour {
            ColourSource::Ramp(id) => id.0,
            _ => usize::MAX,
        };
        let density_id = volume.density_texture.map(|id| id.0).unwrap_or(usize::MAX);
        (lut_id, density_id)
    }

    /// Clear the per-volume texture bind group cache. Call when the
    /// underlying texture vectors may have been mutated (uploads added).
    pub(crate) fn clear_scatter_per_volume_tex_cache(&mut self) {
        self.scatter.per_volume_tex_cache.clear();
    }

    /// Stride between dynamic-offset slots, in bytes.
    pub(crate) fn scatter_per_volume_stride(&self) -> u32 {
        self.scatter.per_volume_stride
    }

    // ---------------------------------------------------------------------
    // Composite + temporal-resolve helpers
    // ---------------------------------------------------------------------

    pub(crate) fn make_scatter_composite_bg(
        &self,
        device: &crate::gpu::Device,
        source_view: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        let bgl = self.scatter.composite_bgl.as_ref().unwrap();
        let sampler = self.scatter.composite_sampler.as_ref().unwrap();
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
    pub(crate) fn make_scatter_temporal_resolve_bg(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        raw_view: &crate::gpu::TextureView,
        history_view: &crate::gpu::TextureView,
        depth_view: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        // The composite sampler is built by the composite pipeline; the
        // depth sampler is built when the per-frame uniform is written.
        // Either may not exist yet on the first frame, so ensure both here.
        self.ensure_scatter_temporal_resolve_bgl(device);
        self.ensure_scatter_depth_sampler(device);
        self.ensure_scatter_composite_pipeline(device, crate::gpu::TextureFormat::Rgba16Float);
        if self.scatter.temporal_resolve_uniform_buffer.is_none() {
            self.write_scatter_temporal_uniform(device, queue, [[0.0; 4]; 4], 0.0, false);
        }
        let bgl = self.scatter.temporal_resolve_bgl.as_ref().unwrap();
        let buf = self
            .scatter
            .temporal_resolve_uniform_buffer
            .as_ref()
            .unwrap();
        let bilinear = self.scatter.composite_sampler.as_ref().unwrap();
        let depth_sampler = self.scatter.depth_sampler.as_ref().unwrap();
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

    fn ensure_scatter_refraction_per_volume_bgl(&mut self, device: &crate::gpu::Device) {
        if self.scatter.refraction_per_volume_bgl.is_some() {
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
        self.scatter.refraction_per_volume_bgl = Some(bgl);
    }

    fn ensure_scatter_refraction_source_bgl(&mut self, device: &crate::gpu::Device) {
        if self.scatter.refraction_source_bgl.is_some() {
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
        self.scatter.refraction_source_bgl = Some(bgl);
    }

    pub(crate) fn ensure_scatter_refraction_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        colour_format: crate::gpu::TextureFormat,
    ) {
        if self.scatter.refraction_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        self.ensure_scatter_refraction_per_volume_bgl(device);
        self.ensure_scatter_refraction_source_bgl(device);
        self.ensure_scatter_composite_pipeline(device, colour_format);

        let per_vol = self.scatter.refraction_per_volume_bgl.as_ref().unwrap();
        let source_bgl = self.scatter.refraction_source_bgl.as_ref().unwrap();

        let shader = crate::resources::builders::wgsl_module(
            device,
            "scatter_refraction_shader",
            crate::resources::builders::wgsl_source!("scatter_refraction"),
        );

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "scatter_refraction_pipeline_layout",
            &[&self.camera_bind_group_layout, per_vol, source_bgl],
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

        self.scatter.refraction_pipeline = Some(pipeline);
    }

    /// Build a render pipeline that samples a source colour texture (via the
    /// composite BGL / shader) and writes it to a render target with replace
    /// blend. Used to copy the HDR scene into the refraction source texture
    /// before the per-volume distortion runs.
    pub(crate) fn ensure_scatter_refraction_blit_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        colour_format: crate::gpu::TextureFormat,
    ) {
        if self.scatter.refraction_blit_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        self.ensure_scatter_composite_pipeline(device, colour_format);
        let bgl = self.scatter.composite_bgl.as_ref().unwrap();
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
        self.scatter.refraction_blit_pipeline = Some(pipeline);
    }

    /// Pack visible refractive volumes into the dynamic-offset uniform buffer.
    /// Returns the number of slots written.
    pub(crate) fn write_scatter_refraction_per_volume_buffer(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        volumes: &[(ScatterVolume, f32)],
        time_seconds: f32,
    ) -> u32 {
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let struct_size = std::mem::size_of::<GpuRefractionVolume>() as u64;
        let stride = ((struct_size + align - 1) / align * align).max(struct_size) as u32;
        let capacity = volumes.len().min(MAX_SCATTER_VOLUMES).max(1) as u32;
        let buffer_size = (stride as u64) * (capacity as u64);

        let need_realloc = self.scatter.refraction_per_volume_buffer.is_none()
            || self.scatter.refraction_per_volume_stride != stride
            || self.scatter.refraction_per_volume_capacity < capacity;
        if need_realloc {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("scatter_refraction_per_volume_uniform"),
                size: buffer_size,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter.refraction_per_volume_buffer = Some(buf);
            self.scatter.refraction_per_volume_stride = stride;
            self.scatter.refraction_per_volume_capacity = capacity;
            self.scatter.refraction_per_volume_bg = None;
        }

        if self.scatter.refraction_per_volume_bg.is_none() {
            self.ensure_scatter_refraction_per_volume_bgl(device);
            let bgl = self.scatter.refraction_per_volume_bgl.as_ref().unwrap();
            let buf = self.scatter.refraction_per_volume_buffer.as_ref().unwrap();
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
            self.scatter.refraction_per_volume_bg = Some(bg);
        }

        let mut bytes = vec![0u8; buffer_size as usize];
        let mut n: u32 = 0;
        for (volume, _) in volumes.iter() {
            if n as usize >= MAX_SCATTER_VOLUMES {
                break;
            }
            if let Some(packed) = GpuRefractionVolume::pack(volume, time_seconds) {
                let offset = (n as usize) * (stride as usize);
                let src = bytemuck::bytes_of(&packed);
                bytes[offset..offset + src.len()].copy_from_slice(src);
                n += 1;
            }
        }
        if let Some(buf) = self.scatter.refraction_per_volume_buffer.as_ref() {
            queue.write_buffer(
                buf,
                0,
                &bytes[..(n as usize * stride as usize).max(stride as usize)],
            );
        }
        n
    }

    /// Stride between dynamic-offset slots in the refraction per-volume buffer.
    pub(crate) fn scatter_refraction_per_volume_stride(&self) -> u32 {
        self.scatter.refraction_per_volume_stride
    }

    /// Build the bind group sampling the refraction source texture + depth.
    pub(crate) fn make_scatter_refraction_source_bg(
        &mut self,
        device: &crate::gpu::Device,
        source_view: &crate::gpu::TextureView,
        depth_view: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        self.ensure_scatter_refraction_source_bgl(device);
        self.ensure_scatter_composite_pipeline(device, crate::gpu::TextureFormat::Rgba16Float);
        let bgl = self.scatter.refraction_source_bgl.as_ref().unwrap();
        let sampler = self.scatter.composite_sampler.as_ref().unwrap();
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

    /// Write the temporal-resolve uniform.
    pub(crate) fn write_scatter_temporal_uniform(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        prev_view_proj: [[f32; 4]; 4],
        blend: f32,
        history_valid: bool,
    ) {
        if self.scatter.temporal_resolve_uniform_buffer.is_none() {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("scatter_temporal_resolve_uniform"),
                size: std::mem::size_of::<ScatterTemporalUniformRaw>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter.temporal_resolve_uniform_buffer = Some(buf);
        }
        let raw = ScatterTemporalUniformRaw {
            prev_view_proj,
            temporal_pack: [
                blend.clamp(0.0, 0.99),
                if history_valid { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        if let Some(buf) = self.scatter.temporal_resolve_uniform_buffer.as_ref() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&raw));
        }
    }
}
