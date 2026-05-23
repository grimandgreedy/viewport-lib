//! Screen-space decal pipeline (D1 + D2).

use crate::resources::{DualPipeline, ViewportGpuResources};
use wgpu::util::DeviceExt as _;

// ---------------------------------------------------------------------------
// GPU-internal types
// ---------------------------------------------------------------------------

/// Flat uniform buffer matching the WGSL `DecalUniform` struct (112 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DecalUniformRaw {
    pub inv_transform: [[f32; 4]; 4],   // 64 bytes
    pub blend_mode: u32,                 //  4
    pub alpha: f32,                      //  4
    pub normal_blend_strength: f32,      //  4
    pub has_normal: u32,                 //  4
    // D3
    pub roughness: f32,                  //  4
    pub metallic: f32,                   //  4
    pub has_roughness_tex: u32,          //  4
    pub has_metallic_tex: u32,           //  4
    // D4 -- vec2 pairs, 8-byte aligned
    pub uv_offset: [f32; 2],            //  8
    pub uv_scale: [f32; 2],             //  8
    // total: 112 bytes
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

        let tex2d_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        // Group 2: per-item uniforms + textures.
        //  0: DecalUniform buffer
        //  1: albedo texture
        //  2: sampler (shared by all texture slots)
        //  3: normal map   (D2; fallback_texture when absent)
        //  4: roughness map (D3; fallback_texture when absent)
        //  5: metallic map  (D3; fallback_texture when absent)
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
                tex2d_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex2d_entry(3), // D2: normal map
                tex2d_entry(4), // D3: roughness map
                tex2d_entry(5), // D3: metallic map
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

        let has_normal        = item.normal_texture_id.is_some()    as u32;
        let has_roughness_tex = item.roughness_texture_id.is_some() as u32;
        let has_metallic_tex  = item.metallic_texture_id.is_some()  as u32;

        let raw = DecalUniformRaw {
            inv_transform,
            blend_mode: blend_mode_u32,
            alpha: item.alpha,
            normal_blend_strength: if has_normal != 0 { item.normal_blend_strength } else { 0.0 },
            has_normal,
            roughness: item.roughness,
            metallic: item.metallic,
            has_roughness_tex,
            has_metallic_tex,
            uv_offset: item.uv_offset,
            uv_scale: item.uv_scale,
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("decal_uniform_buf"),
            contents: bytemuck::bytes_of(&raw),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let resolve_tex = |id: Option<u64>| -> &wgpu::TextureView {
            id.and_then(|i| self.textures.get(i as usize))
                .map(|t| &t.view)
                .unwrap_or(&self.fallback_texture.view)
        };

        let tex_view      = self.textures.get(item.texture_id as usize)
                                .map(|t| &t.view)
                                .unwrap_or(&self.fallback_texture.view);
        let normal_view   = resolve_tex(item.normal_texture_id);
        let roughness_view = resolve_tex(item.roughness_texture_id);
        let metallic_view  = resolve_tex(item.metallic_texture_id);

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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(metallic_view),
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
