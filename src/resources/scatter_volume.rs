//! Scatter-volume pipeline state and per-frame upload.
//!
//! Owns the lazily-created bind group layout, render pipeline, and uniform
//! buffer for the participating-media pass. The renderer calls
//! [`ViewportGpuResources::ensure_scatter_pipeline`] before recording the
//! pass and [`ViewportGpuResources::write_scatter_volumes`] each frame to
//! upload the packed volume array.

use crate::scene::scatter_volume::{GpuScatterVolume, ScatterVolume};

/// Hard cap on the number of volumes processed by the shader per fragment.
/// Must match `MAX_SCATTER_VOLUMES` in `src/shaders/scatter_volume.wgsl`.
pub const MAX_SCATTER_VOLUMES: usize = 16;

/// CPU twin of the WGSL `ScatterUniforms` struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ScatterUniformsRaw {
    pub volumes: [GpuScatterVolume; MAX_SCATTER_VOLUMES],
    pub count: u32,
    pub _pad: [u32; 3],
}

impl Default for ScatterUniformsRaw {
    fn default() -> Self {
        Self {
            volumes: [GpuScatterVolume {
                shape_kind: 0,
                flags: 0,
                _pad0: [0; 2],
                p0: [0.0; 4],
                p1: [0.0; 4],
                colour_density: [0.0; 4],
                params: [0.0; 4],
            }; MAX_SCATTER_VOLUMES],
            count: 0,
            _pad: [0; 3],
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
            ],
        });
        self.scatter_bind_group_layout = Some(bgl);
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

    /// Pack visible volumes into the uniform buffer and (re)build the bind
    /// group if the depth view has changed. Volumes whose effective density
    /// after `density_multiplier` is non-positive are skipped.
    ///
    /// Returns the number of volumes packed (0 if the scatter pass should
    /// be skipped this frame).
    pub(crate) fn write_scatter_volumes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volumes: &[(ScatterVolume, f32, u32)],
        depth_view: &wgpu::TextureView,
        depth_view_token: u64,
    ) -> u32 {
        self.ensure_scatter_bind_group_layout(device);
        self.ensure_scatter_uniform_buffer(device);
        self.ensure_scatter_depth_sampler(device);

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
        raw.count = n;

        if let Some(buf) = self.scatter_uniform_buffer.as_ref() {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&raw));
        }

        // Rebuild bind group when the bound depth view changes (resize, new
        // viewport) or on first use.
        if self.scatter_bind_group.is_none() || self.scatter_bound_depth != depth_view_token {
            let bgl = self.scatter_bind_group_layout.as_ref().unwrap();
            let buf = self.scatter_uniform_buffer.as_ref().unwrap();
            let sampler = self.scatter_depth_sampler.as_ref().unwrap();
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
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.scatter_bind_group = Some(bg);
            self.scatter_bound_depth = depth_view_token;
        }

        n
    }
}
