//! Lazy pipeline creation for SDF overlay shapes.

use crate::renderer::OverlayTextureId;
use crate::resources::types::{
    OverlayShapeTexVertex, OverlayShapeTextureEntry, OverlayShapeVertex,
};

/// SDF overlay shape pipelines (solid + textured fill) and their shared sampler.
/// Lazily built on the first frame with non-empty shapes.
#[derive(Default)]
pub(crate) struct OverlayShapeResources {
    /// Render pipeline for screen-space SDF shapes (rounded rects, circles, etc.).
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    /// Render pipeline for SDF shapes with texture fill.
    pub(crate) tex_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the texture pipeline (group 0: texture + sampler).
    pub(crate) tex_bgl: Option<wgpu::BindGroupLayout>,
    /// Clamp-to-edge linear sampler shared across all texture shape bind groups.
    pub(crate) tex_sampler: Option<wgpu::Sampler>,
}

/// Fullscreen separable Gaussian blur pipeline used to produce the blurred
/// scene texture for `backdrop_blur` overlay shapes.
#[derive(Default)]
pub(crate) struct BackdropBlurResources {
    /// Fullscreen separable Gaussian blur pipeline.
    pub(crate) pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout (group 0: source texture + sampler + uniforms).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Linear clamp sampler shared by blur passes.
    pub(crate) sampler: Option<wgpu::Sampler>,
}

impl crate::resources::DeviceResources {
    /// Lazily create the SDF overlay shape render pipeline.
    ///
    /// No-op if already created. Called from `prepare_viewport_internal()` when
    /// `frame.overlays.shapes` is non-empty.
    pub(crate) fn ensure_overlay_shape_pipeline(&mut self, device: &wgpu::Device) {
        if self.overlay_shape.pipeline.is_some() {
            return;
        }

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_shape_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay_shape.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/overlay_shape.wgsl")).into(),
            ),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_shape_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[OverlayShapeVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
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
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        self.overlay_shape.pipeline = Some(pipeline);
    }

    /// Lazily create the SDF overlay shape render pipeline with texture fill.
    ///
    /// No-op if already created. Called from `prepare_viewport_internal()` when
    /// any shape in `OverlayFrame.shapes` carries an `OverlayTextureId`.
    pub(crate) fn ensure_overlay_shape_tex_pipeline(&mut self, device: &wgpu::Device) {
        if self.overlay_shape.tex_pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_shape_tex_bgl"),
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
            label: Some("overlay_shape_tex_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_shape_tex_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay_shape_tex.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/overlay_shape_tex.wgsl")).into(),
            ),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_shape_tex_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[OverlayShapeTexVertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
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
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        self.overlay_shape.tex_bgl = Some(bgl);
        self.overlay_shape.tex_sampler = Some(sampler);
        self.overlay_shape.tex_pipeline = Some(pipeline);
    }

    /// Upload RGBA8 pixel data as a persistent texture for overlay shape fills.
    ///
    /// Returns an `OverlayTextureId` that can be stored in
    /// `OverlayShapeItem::texture`. The texture persists for the lifetime of
    /// this `DeviceResources`.
    ///
    /// `rgba_data` must contain exactly `width * height * 4` bytes in
    /// row-major, top-to-bottom order. The data is treated as sRGB-encoded
    /// (standard 8-bit image data).
    pub fn upload_overlay_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> OverlayTextureId {
        assert_eq!(
            rgba_data.len(),
            (width * height * 4) as usize,
            "upload_overlay_texture: rgba_data length does not match width * height * 4"
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay_shape_tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = self.content.overlay_textures.len() as u64;
        self.content
            .overlay_textures
            .push(OverlayShapeTextureEntry {
                _texture: texture,
                view,
            });
        OverlayTextureId(id)
    }

    /// Start an asynchronous overlay texture upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Texture
    /// creation and `queue.write_texture` run on a worker thread on cloned
    /// `Device` and `Queue` handles; the apply step inserts the prepared
    /// `OverlayShapeTextureEntry` into the store. Take the resulting
    /// [`OverlayTextureId`] via
    /// [`upload_result_overlay_texture`](Self::upload_result_overlay_texture).
    ///
    /// Ownership of `rgba_data` transfers into the worker.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// before submission when `rgba_data.len() != width * height * 4`.
    pub fn begin_upload_overlay_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: Vec<u8>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        let expected = (width * height * 4) as usize;
        if rgba_data.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba_data.len(),
            });
        }

        let slot = crate::resources::ResultSlot::<OverlayTextureId>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let queue_for_worker = queue.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let texture = device_for_worker.create_texture(&wgpu::TextureDescriptor {
                    label: Some("overlay_shape_tex"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue_for_worker.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rgba_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut crate::resources::DeviceResources| {
                        let id = resources.content.overlay_textures.len() as u64;
                        resources
                            .content
                            .overlay_textures
                            .push(OverlayShapeTextureEntry {
                                _texture: texture,
                                view,
                            });
                        slot_for_apply.set(OverlayTextureId(id));
                    }),
                ))
            })
        };

        self.job_results
            .overlay_texture
            .lock()
            .expect("overlay texture result map poisoned")
            .insert(id, slot);
        Ok(id)
    }

    /// Take the [`OverlayTextureId`] produced by a completed
    /// [`begin_upload_overlay_texture`](Self::begin_upload_overlay_texture) job.
    pub fn upload_result_overlay_texture(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<OverlayTextureId> {
        let mut map = self
            .job_results
            .overlay_texture
            .lock()
            .expect("overlay texture result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(tid) => {
                map.remove(&id);
                Ok(tid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Lazily create the separable Gaussian blur pipeline used to blur the
    /// scene texture for `backdrop_blur` overlay shapes.
    ///
    /// No-op if already created.
    pub(crate) fn ensure_backdrop_blur_pipeline(&mut self, device: &wgpu::Device) {
        if self.backdrop_blur.pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop_blur_bgl"),
            entries: &[
                // binding 0: source texture
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
                // binding 1: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: blur parameters uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("backdrop_blur_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("backdrop_blur_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backdrop_blur.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/backdrop_blur.wgsl")).into(),
            ),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("backdrop_blur_pipeline"),
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

        self.backdrop_blur.bgl = Some(bgl);
        self.backdrop_blur.sampler = Some(sampler);
        self.backdrop_blur.pipeline = Some(pipeline);
    }
}
