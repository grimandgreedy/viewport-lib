//! Screen-space decal pipeline (D1).

use crate::resources::{DualPipeline, ViewportGpuResources};
use wgpu::util::DeviceExt as _;

// ---------------------------------------------------------------------------
// GPU-internal types
// ---------------------------------------------------------------------------

/// Flat uniform buffer matching the WGSL `DecalUniform` struct (80 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DecalUniformRaw {
    pub inv_transform: [[f32; 4]; 4],
    pub blend_mode: u32,
    pub alpha: f32,
    pub _pad: [f32; 2],
}

/// Per-draw GPU data for one [`DecalItem`](crate::renderer::DecalItem).
pub(crate) struct DecalGpuItem {
    pub blend_mode: crate::renderer::DecalBlendMode,
    pub _uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Pipeline init and upload (impl ViewportGpuResources)
// ---------------------------------------------------------------------------

impl ViewportGpuResources {
    /// Lazily create the decal render pipelines and item bind group layout.
    ///
    /// No-op if already created.  Requires `decal_depth_bgl` to exist (created
    /// by `ensure_hdr_shared`).
    pub(crate) fn ensure_decal_pipeline(&mut self, device: &wgpu::Device) {
        if self.decal_replace_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("decal_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/decal.wgsl").into()),
        });

        // Group 2: per-item (uniform buffer + albedo texture + sampler).
        let item_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decal_item_bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let depth_bgl = self
            .decal_depth_bgl
            .as_ref()
            .expect("decal_depth_bgl must exist before ensure_decal_pipeline");

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decal_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, depth_bgl, &item_bgl],
            push_constant_ranges: &[],
        });

        let make = |fmt: wgpu::TextureFormat, blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("decal_pipeline"),
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
                        format: fmt,
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
                // No depth attachment: decals read depth as a texture, they do not
                // write to the depth buffer.
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let replace_blend = wgpu::BlendState::ALPHA_BLENDING;
        // Multiply: result.rgb = dst.rgb * src.rgb; result.a = dst.a.
        let multiply_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        self.decal_item_bgl = Some(item_bgl);
        self.decal_replace_pipeline = Some(DualPipeline {
            ldr: make(self.target_format, replace_blend),
            hdr: make(wgpu::TextureFormat::Rgba16Float, replace_blend),
        });
        self.decal_multiply_pipeline = Some(DualPipeline {
            ldr: make(self.target_format, multiply_blend),
            hdr: make(wgpu::TextureFormat::Rgba16Float, multiply_blend),
        });
    }

    /// Create the per-viewport depth bind group used by the decal pass.
    ///
    /// Must be called after `ensure_hdr_shared` (which creates `decal_depth_bgl`).
    /// Rebuilt when the viewport is resized.
    pub(crate) fn create_decal_depth_bg(
        &self,
        device: &wgpu::Device,
        depth_only_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let bgl = self
            .decal_depth_bgl
            .as_ref()
            .expect("decal_depth_bgl not created");
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decal_depth_bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_only_view),
            }],
        })
    }

    /// Upload one [`DecalItem`](crate::renderer::DecalItem) to GPU and return the per-draw data.
    ///
    /// Panics if called before `ensure_decal_pipeline`.
    pub(crate) fn upload_decal_item(
        &self,
        device: &wgpu::Device,
        item: &crate::renderer::DecalItem,
    ) -> DecalGpuItem {
        let model = glam::Mat4::from_cols_array_2d(&item.transform);
        let inv_transform = model.inverse().to_cols_array_2d();

        let blend_mode_u32 = match item.blend_mode {
            crate::renderer::DecalBlendMode::Replace => 0u32,
            crate::renderer::DecalBlendMode::Multiply => 1u32,
        };

        let raw = DecalUniformRaw {
            inv_transform,
            blend_mode: blend_mode_u32,
            alpha: item.alpha,
            _pad: [0.0; 2],
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("decal_uniform_buf"),
            contents: bytemuck::bytes_of(&raw),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let tex_view: &wgpu::TextureView = self
            .textures
            .get(item.texture_id as usize)
            .map(|t| &t.view)
            .unwrap_or(&self.fallback_texture.view);

        let bgl = self
            .decal_item_bgl
            .as_ref()
            .expect("ensure_decal_pipeline not called");

        let sampler = self
            .decal_sampler
            .as_ref()
            .expect("decal_sampler not created");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decal_item_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        DecalGpuItem {
            blend_mode: item.blend_mode,
            _uniform_buf: uniform_buf,
            bind_group,
        }
    }
}
