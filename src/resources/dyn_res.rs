//! Dynamic resolution render target for the LDR render path.
//!
//! When `render_scale < 1.0`, the scene is rendered into a scaled intermediate
//! texture and then upscaled to the surface via bilinear filtering.

use super::ViewportGpuResources;

/// Per-viewport intermediate render target for dynamic resolution rendering.
///
/// Owned by the viewport slot; created or recreated whenever the render scale
/// or surface size changes.
pub(crate) struct DynResTarget {
    /// Scaled colour texture (render_scale × surface_size).
    pub _colour_texture: wgpu::Texture,
    /// View of `colour_texture`.
    pub colour_view: wgpu::TextureView,
    /// Depth texture matching the scaled resolution.
    pub _depth_texture: wgpu::Texture,
    /// View of `depth_texture`.
    pub depth_view: wgpu::TextureView,
    /// Bind group for the upscale pass: colour_texture + linear sampler.
    pub upscale_bind_group: wgpu::BindGroup,
    /// Dimensions of the intermediate target `[w, h]`.
    pub scaled_size: [u32; 2],
    /// Native surface dimensions this target was created for `[w, h]`.
    pub surface_size: [u32; 2],
}

impl ViewportGpuResources {
    /// Ensure the shared upscale pipeline and sampler exist, creating them on
    /// first call. Idempotent.
    pub(crate) fn ensure_dyn_res_pipeline(&mut self, device: &wgpu::Device) {
        if self.dyn_res_upscale_pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dyn_res_upscale_bgl"),
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
            label: Some("dyn_res_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dyn_res_upscale_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/dyn_res_upscale.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dyn_res_upscale_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dyn_res_upscale_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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

        self.dyn_res_upscale_bgl = Some(bgl);
        self.dyn_res_upscale_pipeline = Some(pipeline);
        self.dyn_res_linear_sampler = Some(sampler);
    }

    /// Ensure the depth-stencil compatible upscale pipeline exists for use inside
    /// eframe's paint render pass, which always has a `Depth24PlusStencil8` attachment.
    ///
    /// Identical to [`ensure_dyn_res_pipeline`](Self::ensure_dyn_res_pipeline) except
    /// `depth_stencil` is set to read-only `Depth24PlusStencil8` so the pipeline is
    /// compatible with any render pass that carries that depth attachment.
    /// [`ensure_dyn_res_pipeline`](Self::ensure_dyn_res_pipeline) must be called first.
    pub(crate) fn ensure_dyn_res_ds_pipeline(&mut self, device: &wgpu::Device) {
        if self.dyn_res_upscale_ds_pipeline.is_some() {
            return;
        }

        let bgl = self
            .dyn_res_upscale_bgl
            .as_ref()
            .expect("ensure_dyn_res_pipeline must be called before ensure_dyn_res_ds_pipeline");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dyn_res_upscale_ds_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/dyn_res_upscale.wgsl")).into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dyn_res_upscale_ds_layout"),
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dyn_res_upscale_ds_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
        self.dyn_res_upscale_ds_pipeline = Some(pipeline);
    }

    /// Create a [`DynResTarget`] at `scaled_size`, bound for upscaling to
    /// `surface_size`. The shared pipeline must already exist (call
    /// [`ensure_dyn_res_pipeline`](Self::ensure_dyn_res_pipeline) first).
    pub(crate) fn create_dyn_res_target(
        &self,
        device: &wgpu::Device,
        scaled_size: [u32; 2],
        surface_size: [u32; 2],
    ) -> DynResTarget {
        let [sw, sh] = scaled_size;

        let colour_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dyn_res_colour"),
            size: wgpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let colour_view = colour_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dyn_res_depth"),
            size: wgpu::Extent3d {
                width: sw,
                height: sh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bgl = self.dyn_res_upscale_bgl.as_ref().unwrap();
        let sampler = self.dyn_res_linear_sampler.as_ref().unwrap();
        let upscale_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dyn_res_upscale_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&colour_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        DynResTarget {
            _colour_texture: colour_texture,
            colour_view,
            _depth_texture: depth_texture,
            depth_view,
            upscale_bind_group,
            scaled_size,
            surface_size,
        }
    }

    /// Create a [`HdrCallbackTarget`] at `size` for use with the eframe HDR callback path.
    ///
    /// The shared pipeline and sampler must already exist — call
    /// [`ensure_dyn_res_pipeline`](Self::ensure_dyn_res_pipeline) first.
    pub(crate) fn create_hdr_callback_target(
        &self,
        device: &wgpu::Device,
        size: [u32; 2],
    ) -> HdrCallbackTarget {
        let [w, h] = size;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_callback_target"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let blit_bind_group = {
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bgl = self.dyn_res_upscale_bgl.as_ref().unwrap();
            let sampler = self.dyn_res_linear_sampler.as_ref().unwrap();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hdr_callback_blit_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };

        HdrCallbackTarget {
            texture,
            blit_bind_group,
            size,
        }
    }
}

/// Per-viewport intermediate render target for the HDR eframe callback path.
///
/// Allocated when [`prepare_hdr_callback`](crate::ViewportRenderer::prepare_hdr_callback)
/// is first called for a viewport and recreated when the viewport size changes.
/// The full HDR pipeline (OIT, EDL, tone-map) renders into `texture`; `blit_bind_group`
/// is then used by
/// [`paint_hdr_blit`](crate::ViewportRenderer::paint_hdr_blit) to copy the result
/// into the egui render pass.
pub(crate) struct HdrCallbackTarget {
    /// Intermediate LDR colour texture (same format as `target_format`).
    ///
    /// Stored so we can create a fresh `TextureView` each frame inside
    /// `prepare_hdr_callback`, avoiding a simultaneous mutable + immutable borrow.
    pub texture: wgpu::Texture,
    /// Bind group for the blit pass: `texture` view + linear sampler.
    pub blit_bind_group: wgpu::BindGroup,
    /// Dimensions `[w, h]`.
    pub size: [u32; 2],
}
