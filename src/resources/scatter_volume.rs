//! Scatter-volume pipeline state and per-frame upload.
//!
//! Owns the lazily-created bind group layout, render pipeline, and uniform
//! buffer for the participating-media pass. The renderer calls
//! [`ViewportGpuResources::ensure_scatter_pipeline`] before recording the
//! pass and [`ViewportGpuResources::write_scatter_volumes`] each frame to
//! upload the packed volume array.

use crate::scene::scatter_volume::{ColourSource, GpuScatterVolume, ScatterVolume};

/// Hard cap on the number of volumes processed by the shader per fragment.
/// Must match `MAX_SCATTER_VOLUMES` in `src/shaders/scatter_volume.wgsl`.
pub const MAX_SCATTER_VOLUMES: usize = 16;

/// CPU twin of the WGSL `ScatterUniforms` struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ScatterUniformsRaw {
    pub volumes: [GpuScatterVolume; MAX_SCATTER_VOLUMES],
    // x = count, y = global step count, z = blue_noise_jitter (0/1),
    // w = frame index (low 32 bits).
    pub count_pack: [u32; 4],
    // x = elapsed seconds, yzw reserved.
    pub time_pack: [f32; 4],
    /// Previous frame's view-projection. Used by the shader to reproject the
    /// current pixel into the history buffer for temporal accumulation.
    pub prev_view_proj: [[f32; 4]; 4],
    /// Temporal-blend parameters.
    /// x = blend factor (history weight, 0..1),
    /// y = history_valid (0 / 1),
    /// z = temporal_enabled (0 / 1),
    /// w = reserved.
    pub temporal_pack: [f32; 4],
}

impl Default for ScatterUniformsRaw {
    fn default() -> Self {
        Self {
            volumes: [GpuScatterVolume {
                shape_kind: 0,
                flags: 0,
                remap_kind: 0,
                emission_kind: 0,
                p0: [0.0; 4],
                p1: [0.0; 4],
                colour_density: [0.0; 4],
                params: [0.0; 4],
                remap_data: [0.0; 4],
                remap_data2: [0.0; 4],
                noise_pack: [0.0; 4],
                noise_vel: [0.0; 4],
            }; MAX_SCATTER_VOLUMES],
            count_pack: [0; 4],
            time_pack: [0.0; 4],
            prev_view_proj: [[0.0; 4]; 4],
            temporal_pack: [0.0; 4],
        }
    }
}

impl crate::resources::ViewportGpuResources {
    /// Lazily build the scatter-volume bind group layout.
    pub(crate) fn ensure_scatter_bind_group_layout(&mut self, device: &wgpu::Device) {
        if self.scatter_bind_group_layout.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_volume_bgl"),
            entries: &[
                // binding 0: uniform buffer (volumes array + count)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: opaque depth texture
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
                // binding 2: depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 3: colourmap LUT (256x1 RGBA texture) for Ramp colour sources
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
                // binding 4: linear-clamp sampler for the colourmap LUT
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 5: 3D density texture (R32Float, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 6: non-filtering sampler for the 3D density texture
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 7: previous-frame scatter history texture (RGBA16F).
                // Bound either to the inactive ping-pong target (history valid)
                // or to a 1x1 fallback (history invalid / temporal disabled).
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 8: linear-clamp sampler for the history texture.
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        self.scatter_bind_group_layout = Some(bgl);
    }

    /// Ensure the 1x1x1 R32Float fallback texture (bound when no volume
    /// supplies its own 3D density texture) exists.
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
        // Initialise to 1.0 so volumes that mistakenly enable
        // `USE_DENSITY_TEXTURE` without supplying one render as the base
        // density rather than as fully transparent.
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

    /// Ensure the colourmap-LUT sampler used by the scatter pass exists.
    fn ensure_scatter_colourmap_sampler(&mut self, device: &wgpu::Device) {
        if self.scatter_colourmap_sampler.is_some() {
            return;
        }
        self.scatter_colourmap_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scatter_volume_colourmap_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));
    }

    /// Lazily build the scatter-volume render pipeline.
    pub(crate) fn ensure_scatter_pipeline(
        &mut self,
        device: &wgpu::Device,
        colour_format: wgpu::TextureFormat,
    ) {
        if self.scatter_pipeline.is_some() {
            return;
        }
        self.ensure_scatter_bind_group_layout(device);
        let bgl = self
            .scatter_bind_group_layout
            .as_ref()
            .expect("scatter bgl exists");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter_volume_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/scatter_volume.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter_volume_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, bgl],
            push_constant_ranges: &[],
        });

        // Premultiplied alpha-over blend: out = src.rgb + dst.rgb * (1 - src.a)
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
            // Fullscreen triangle does its own depth handling via the sampled
            // opaque depth texture. No depth-stencil attachment is bound at
            // the pass level, and the pipeline declares no depth state.
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

    /// Ensure the per-frame uniform buffer exists.
    fn ensure_scatter_uniform_buffer(&mut self, device: &wgpu::Device) {
        if self.scatter_uniform_buffer.is_some() {
            return;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter_volume_uniform"),
            size: std::mem::size_of::<ScatterUniformsRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.scatter_uniform_buffer = Some(buf);
    }

    /// Ensure the linear depth sampler used by the scatter pass exists.
    fn ensure_scatter_depth_sampler(&mut self, device: &wgpu::Device) {
        if self.scatter_depth_sampler.is_some() {
            return;
        }
        self.scatter_depth_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scatter_volume_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));
    }

    /// Ensure the 1x1 RGBA16F fallback view bound to the history slot when
    /// temporal accumulation is disabled or the history is not yet valid.
    fn ensure_scatter_history_fallback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.scatter_history_fallback_view.is_some() {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scatter_history_fallback"),
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
        // Initialise to zero so the history blend reads as "no contribution".
        let zero: [u16; 4] = [0, 0, 0, 0];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&zero),
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
        self.scatter_history_fallback_view =
            Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    /// Ensure the linear sampler used to read the scatter history texture.
    fn ensure_scatter_history_sampler(&mut self, device: &wgpu::Device) {
        if self.scatter_history_sampler.is_some() {
            return;
        }
        self.scatter_history_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scatter_history_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));
    }

    /// Pack visible volumes into the uniform buffer and (re)build the bind
    /// group if the depth view has changed. Volumes whose effective density
    /// after `density_multiplier` is non-positive are skipped.
    ///
    /// `history_view` is the inactive ping-pong target read by the temporal
    /// blend. Pass `None` to bind the 1x1 fallback (e.g. when temporal is
    /// disabled or the history is not yet valid).
    ///
    /// Returns the number of volumes packed (0 if the scatter pass should
    /// be skipped this frame).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_scatter_volumes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volumes: &[(ScatterVolume, f32, u32)],
        depth_view: &wgpu::TextureView,
        depth_view_token: u64,
        time_seconds: f32,
        global_steps: u32,
        blue_noise_jitter: bool,
        frame_index: u64,
        history_view: Option<&wgpu::TextureView>,
        history_token: u64,
        prev_view_proj: [[f32; 4]; 4],
        temporal_enabled: bool,
        history_valid: bool,
        temporal_blend: f32,
    ) -> u32 {
        self.ensure_scatter_bind_group_layout(device);
        self.ensure_scatter_uniform_buffer(device);
        self.ensure_scatter_depth_sampler(device);
        self.ensure_scatter_colourmap_sampler(device);
        self.ensure_scatter_density_fallback(device, queue);
        self.ensure_scatter_history_fallback(device, queue);
        self.ensure_scatter_history_sampler(device);

        let mut raw = ScatterUniformsRaw::default();
        let mut n: u32 = 0;
        for (volume, mult, flags) in volumes.iter() {
            if n as usize >= MAX_SCATTER_VOLUMES {
                break;
            }
            if let Some(packed) = GpuScatterVolume::pack(volume, *mult, *flags) {
                raw.volumes[n as usize] = packed;
                n += 1;
            }
        }
        raw.count_pack[0] = n;
        raw.count_pack[1] = global_steps.clamp(1, 128);
        raw.count_pack[2] = if blue_noise_jitter { 1 } else { 0 };
        raw.count_pack[3] = frame_index as u32;
        raw.time_pack[0] = time_seconds;
        raw.prev_view_proj = prev_view_proj;
        raw.temporal_pack = [
            temporal_blend.clamp(0.0, 0.99),
            if history_valid { 1.0 } else { 0.0 },
            if temporal_enabled { 1.0 } else { 0.0 },
            0.0,
        ];

        if let Some(buf) = self.scatter_uniform_buffer.as_ref() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&raw));
        }

        // Resolve the LUT to bind: the first Ramp volume's colourmap wins
        // for the whole pass. Other Ramp volumes share it; consumers wanting
        // multiple ramps per frame must defer to a follow-up phase that
        // builds a per-frame LUT atlas. Fallback view is bound when no
        // Ramp volume is present so the binding is always valid.
        let mut lut_id: u64 = u64::MAX;
        for (volume, _, _) in volumes.iter() {
            if let ColourSource::Ramp(id) = volume.colour {
                lut_id = id.0 as u64;
                break;
            }
        }

        // Resolve the 3D density texture to bind, applying the same
        // "first-wins" policy. Fallback 1x1x1 view binds otherwise.
        let mut density_id: u64 = u64::MAX;
        for (volume, _, _) in volumes.iter() {
            if let Some(id) = volume.density_texture {
                density_id = id.0 as u64;
                break;
            }
        }

        // Rebuild bind group when any of (depth view, bound LUT, bound
        // density texture, history view) changes.
        let token = depth_view_token
            ^ (lut_id.wrapping_mul(0x9E3779B97F4A7C15))
            ^ (density_id.wrapping_mul(0xBF58476D1CE4E5B9))
            ^ (history_token.wrapping_mul(0x94D049BB133111EB));
        if self.scatter_bind_group.is_none() || self.scatter_bound_depth != token {
            let bgl = self.scatter_bind_group_layout.as_ref().unwrap();
            let buf = self.scatter_uniform_buffer.as_ref().unwrap();
            let depth_sampler = self.scatter_depth_sampler.as_ref().unwrap();
            let lut_sampler = self.scatter_colourmap_sampler.as_ref().unwrap();
            let lut_view: &wgpu::TextureView = if lut_id == u64::MAX {
                &self.fallback_lut_view
            } else {
                self.colourmap_views
                    .get(lut_id as usize)
                    .unwrap_or(&self.fallback_lut_view)
            };
            let density_fallback = self.scatter_density_fallback_view.as_ref().unwrap();
            let density_view: &wgpu::TextureView = if density_id == u64::MAX {
                density_fallback
            } else {
                self.volume_textures
                    .get(density_id as usize)
                    .map(|(_, v)| v)
                    .unwrap_or(density_fallback)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_volume_bind_group"),
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
                        resource: wgpu::BindingResource::Sampler(depth_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(lut_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(density_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(depth_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            history_view.unwrap_or_else(|| {
                                self.scatter_history_fallback_view.as_ref().unwrap()
                            }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(
                            self.scatter_history_sampler.as_ref().unwrap(),
                        ),
                    },
                ],
            });
            self.scatter_bind_group = Some(bg);
            self.scatter_bound_depth = token;
        }

        n
    }

    /// Lazily build the composite pipeline that samples a half-/full-res
    /// scatter intermediate and writes premultiplied alpha-over into the HDR
    /// target.
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

    /// Build a composite bind group sampling the given intermediate view.
    /// Caller stores the result on the per-viewport HDR slot.
    pub(crate) fn make_scatter_composite_bg(
        &self,
        device: &wgpu::Device,
        source_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let bgl = self.scatter_composite_bgl.as_ref().unwrap();
        let sampler = self.scatter_composite_sampler.as_ref().unwrap();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter_composite_bind_group"),
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
}
