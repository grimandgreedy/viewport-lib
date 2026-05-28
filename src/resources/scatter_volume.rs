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

use crate::scene::scatter_volume::{ColourSource, GpuScatterVolume, ScatterVolume};

/// Hard cap on the number of volumes per frame. Same as the original cap; the
/// per-volume draw flow handles up to this many active volumes.
pub const MAX_SCATTER_VOLUMES: usize = 16;

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

impl crate::resources::ViewportGpuResources {
    // ---------------------------------------------------------------------
    // Bind group layouts
    // ---------------------------------------------------------------------

    fn ensure_scatter_per_volume_bgl(&mut self, device: &wgpu::Device) {
        if self.scatter_per_volume_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_per_volume_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
        self.scatter_per_volume_bgl = Some(bgl);
    }

    fn ensure_scatter_per_volume_tex_bgl(&mut self, device: &wgpu::Device) {
        if self.scatter_per_volume_tex_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_per_volume_tex_bgl"),
            entries: &[
                // 0: colourmap LUT (256x1 RGBA, used when FLAG_USE_RAMP).
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
                // 1: LUT sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 2: 3D density texture (used when FLAG_USE_DENSITY_TEXTURE).
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: 3D density sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        self.scatter_per_volume_tex_bgl = Some(bgl);
    }

    fn ensure_scatter_frame_bgl(&mut self, device: &wgpu::Device) {
        if self.scatter_frame_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_frame_bgl"),
            entries: &[
                // 0: per-frame uniform (time, blue noise, frame index).
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<ScatterFrameUniformRaw>() as u64,
                        ),
                    },
                    count: None,
                },
                // 1: opaque depth texture.
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
                // 2: depth sampler (NonFiltering for the textureLoad path).
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        self.scatter_frame_bgl = Some(bgl);
    }

    fn ensure_scatter_temporal_resolve_bgl(&mut self, device: &wgpu::Device) {
        if self.scatter_temporal_resolve_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_temporal_resolve_bgl"),
            entries: &[
                // 0: temporal uniform (prev_view_proj + temporal_pack).
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<ScatterTemporalUniformRaw>() as u64,
                        ),
                    },
                    count: None,
                },
                // 1: raw_current texture (this frame's scatter output).
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
                // 2: history_prev texture.
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
                // 3: bilinear sampler (reuses scatter composite sampler).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 4: opaque depth texture (for reprojection).
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: depth sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        self.scatter_temporal_resolve_bgl = Some(bgl);
    }

    fn ensure_scatter_density_fallback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.scatter_density_fallback_view.is_some() {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scatter_density_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let data: [f32; 1] = [1.0];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
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
        self.scatter_density_fallback_view =
            Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    fn ensure_scatter_depth_sampler(&mut self, device: &wgpu::Device) {
        if self.scatter_depth_sampler.is_some() {
            return;
        }
        self.scatter_depth_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scatter_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));
    }

    fn ensure_scatter_colourmap_sampler(&mut self, device: &wgpu::Device) {
        if self.scatter_colourmap_sampler.is_some() {
            return;
        }
        self.scatter_colourmap_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scatter_colourmap_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));
    }

    // ---------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------

    pub(crate) fn ensure_scatter_pipeline(
        &mut self,
        device: &wgpu::Device,
        colour_format: wgpu::TextureFormat,
    ) {
        if self.scatter_pipeline.is_some() {
            return;
        }
        self.ensure_scatter_per_volume_bgl(device);
        self.ensure_scatter_per_volume_tex_bgl(device);
        self.ensure_scatter_frame_bgl(device);

        let per_vol = self.scatter_per_volume_bgl.as_ref().unwrap();
        let per_tex = self.scatter_per_volume_tex_bgl.as_ref().unwrap();
        let frame_bgl = self.scatter_frame_bgl.as_ref().unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter_volume_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/scatter_volume.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter_volume_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, per_vol, per_tex, frame_bgl],
            push_constant_ranges: &[],
        });

        // Premultiplied alpha-over: per-volume draws composite into the
        // (cleared) raw_current target in back-to-front order.
        let blend = wgpu::BlendState {
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scatter_volume_pipeline"),
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
                    format: colour_format,
                    blend: Some(blend),
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
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.scatter_pipeline = Some(pipeline);
    }

    pub(crate) fn ensure_scatter_composite_pipeline(
        &mut self,
        device: &wgpu::Device,
        colour_format: wgpu::TextureFormat,
    ) {
        if self.scatter_composite_pipeline.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_composite_bgl"),
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scatter_composite_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/scatter_composite.wgsl")).into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter_composite_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let blend = wgpu::BlendState {
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
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scatter_composite_pipeline"),
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
                    format: colour_format,
                    blend: Some(blend),
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
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        self.scatter_composite_pipeline = Some(pipeline);
        self.scatter_composite_bgl = Some(bgl);
        self.scatter_composite_sampler = Some(sampler);
    }

    pub(crate) fn ensure_scatter_temporal_resolve_pipeline(
        &mut self,
        device: &wgpu::Device,
    ) {
        if self.scatter_temporal_resolve_pipeline.is_some() {
            return;
        }
        self.ensure_scatter_temporal_resolve_bgl(device);
        let bgl = self.scatter_temporal_resolve_bgl.as_ref().unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter_temporal_resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/scatter_temporal_resolve.wgsl")).into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter_temporal_resolve_pipeline_layout"),
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scatter_temporal_resolve_pipeline"),
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
                    // History textures are RGBA16F.
                    format: wgpu::TextureFormat::Rgba16Float,
                    // Replace blend -- this pass owns the new history fully.
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
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        self.scatter_temporal_resolve_pipeline = Some(pipeline);
    }

    // ---------------------------------------------------------------------
    // Per-frame uniform / bind group construction
    // ---------------------------------------------------------------------

    /// Pack visible volumes into the per-volume dynamic-offset uniform buffer.
    /// Volumes are written in submission order (caller is responsible for
    /// back-to-front sort). Returns the number of slots written.
    pub(crate) fn write_scatter_per_volume_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volumes: &[(ScatterVolume, f32, u32)],
    ) -> u32 {
        // Stride = aligned per-volume uniform slot size. Recomputed once.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let struct_size = std::mem::size_of::<GpuScatterVolume>() as u64;
        let stride = ((struct_size + align - 1) / align * align).max(struct_size) as u32;
        let capacity = volumes.len().min(MAX_SCATTER_VOLUMES).max(1) as u32;
        let buffer_size = (stride as u64) * (capacity as u64);

        let need_realloc = self.scatter_per_volume_buffer.is_none()
            || self.scatter_per_volume_stride != stride
            || self.scatter_per_volume_capacity < capacity;
        if need_realloc {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scatter_per_volume_uniform"),
                size: buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter_per_volume_buffer = Some(buf);
            self.scatter_per_volume_stride = stride;
            self.scatter_per_volume_capacity = capacity;
            self.scatter_per_volume_bg = None;
        }

        // Build the dynamic-offset bind group lazily.
        if self.scatter_per_volume_bg.is_none() {
            self.ensure_scatter_per_volume_bgl(device);
            let bgl = self.scatter_per_volume_bgl.as_ref().unwrap();
            let buf = self.scatter_per_volume_buffer.as_ref().unwrap();
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_per_volume_bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(struct_size),
                    }),
                }],
            });
            self.scatter_per_volume_bg = Some(bg);
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
        if let Some(buf) = self.scatter_per_volume_buffer.as_ref() {
            queue.write_buffer(buf, 0, &bytes[..(n as usize * stride as usize).max(stride as usize)]);
        }
        n
    }

    /// Write the per-frame uniform (time / blue noise / frame index) and
    /// build / rebuild the frame bind group for the given opaque depth view.
    pub(crate) fn write_scatter_frame_uniform(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth_view: &wgpu::TextureView,
        depth_view_token: u64,
        time_seconds: f32,
        global_steps: u32,
        blue_noise_jitter: bool,
        frame_index: u64,
    ) {
        self.ensure_scatter_frame_bgl(device);
        self.ensure_scatter_depth_sampler(device);
        if self.scatter_frame_uniform_buffer.is_none() {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scatter_frame_uniform"),
                size: std::mem::size_of::<ScatterFrameUniformRaw>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter_frame_uniform_buffer = Some(buf);
            self.scatter_frame_bg = None;
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
        if let Some(buf) = self.scatter_frame_uniform_buffer.as_ref() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&raw));
        }
        if self.scatter_frame_bg.is_none() || self.scatter_bound_depth != depth_view_token {
            let bgl = self.scatter_frame_bgl.as_ref().unwrap();
            let buf = self.scatter_frame_uniform_buffer.as_ref().unwrap();
            let sampler = self.scatter_depth_sampler.as_ref().unwrap();
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_frame_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.scatter_frame_bg = Some(bg);
            self.scatter_bound_depth = depth_view_token;
        }
    }

    /// Look up or build a group 2 bind group for the (lut_id, density_id)
    /// pair. Pass `u32::MAX` for either id to bind the fallback.
    pub(crate) fn ensure_scatter_per_volume_tex_bg(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lut_id: usize,
        density_id: usize,
    ) -> wgpu::BindGroup {
        self.ensure_scatter_per_volume_tex_bgl(device);
        self.ensure_scatter_colourmap_sampler(device);
        self.ensure_scatter_density_fallback(device, queue);

        let key = (lut_id, density_id);
        if let Some((_, bg)) = self.scatter_per_volume_tex_cache.iter().find(|(k, _)| *k == key) {
            return bg.clone();
        }
        let bgl = self.scatter_per_volume_tex_bgl.as_ref().unwrap();
        let lut_sampler = self.scatter_colourmap_sampler.as_ref().unwrap();
        let density_sampler = self.scatter_depth_sampler.as_ref().unwrap();
        let lut_view: &wgpu::TextureView = if lut_id == usize::MAX {
            &self.fallback_lut_view
        } else {
            self.colourmap_views
                .get(lut_id)
                .unwrap_or(&self.fallback_lut_view)
        };
        let density_fallback = self.scatter_density_fallback_view.as_ref().unwrap();
        let density_view: &wgpu::TextureView = if density_id == usize::MAX {
            density_fallback
        } else {
            self.volume_textures
                .get(density_id)
                .map(|(_, v)| v)
                .unwrap_or(density_fallback)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter_per_volume_tex_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(density_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(density_sampler),
                },
            ],
        });
        self.scatter_per_volume_tex_cache.push((key, bg.clone()));
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
        self.scatter_per_volume_tex_cache.clear();
    }

    /// Stride between dynamic-offset slots, in bytes.
    pub(crate) fn scatter_per_volume_stride(&self) -> u32 {
        self.scatter_per_volume_stride
    }

    // ---------------------------------------------------------------------
    // Composite + temporal-resolve helpers
    // ---------------------------------------------------------------------

    pub(crate) fn make_scatter_composite_bg(
        &self,
        device: &wgpu::Device,
        source_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let bgl = self.scatter_composite_bgl.as_ref().unwrap();
        let sampler = self.scatter_composite_sampler.as_ref().unwrap();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter_composite_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Build a temporal-resolve bind group sampling `(raw_view, history_view)`
    /// alongside the bound depth and uniform.
    pub(crate) fn make_scatter_temporal_resolve_bg(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raw_view: &wgpu::TextureView,
        history_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        // The composite sampler is built by the composite pipeline; the
        // depth sampler is built when the per-frame uniform is written.
        // Either may not exist yet on the first frame, so ensure both here.
        self.ensure_scatter_temporal_resolve_bgl(device);
        self.ensure_scatter_depth_sampler(device);
        self.ensure_scatter_composite_pipeline(device, wgpu::TextureFormat::Rgba16Float);
        if self.scatter_temporal_resolve_uniform_buffer.is_none() {
            self.write_scatter_temporal_uniform(device, queue, [[0.0; 4]; 4], 0.0, false);
        }
        let bgl = self.scatter_temporal_resolve_bgl.as_ref().unwrap();
        let buf = self.scatter_temporal_resolve_uniform_buffer.as_ref().unwrap();
        let bilinear = self.scatter_composite_sampler.as_ref().unwrap();
        let depth_sampler = self.scatter_depth_sampler.as_ref().unwrap();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter_temporal_resolve_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(raw_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(bilinear),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(depth_sampler),
                },
            ],
        })
    }

    /// Write the temporal-resolve uniform.
    pub(crate) fn write_scatter_temporal_uniform(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prev_view_proj: [[f32; 4]; 4],
        blend: f32,
        history_valid: bool,
    ) {
        if self.scatter_temporal_resolve_uniform_buffer.is_none() {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scatter_temporal_resolve_uniform"),
                size: std::mem::size_of::<ScatterTemporalUniformRaw>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter_temporal_resolve_uniform_buffer = Some(buf);
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
        if let Some(buf) = self.scatter_temporal_resolve_uniform_buffer.as_ref() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&raw));
        }
    }
}
