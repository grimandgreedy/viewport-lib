//! Factory functions for the mesh-family pipelines that share a single
//! shader source. The same factories run at startup (from `init.rs` and
//! `postprocess.rs`) and from the deformer registry's pipeline rebuild
//! path, so a registered deformer can swap in a freshly composed
//! `ShaderModule` without duplicating pipeline descriptors.

use crate::resources::types::Vertex;

/// The four LDR `mesh.wgsl` pipelines that draw into the swapchain.
pub(crate) struct LdrMeshPipelines {
    pub solid: wgpu::RenderPipeline,
    pub solid_two_sided: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
    pub wireframe: wgpu::RenderPipeline,
}

pub(crate) fn build_ldr_mesh_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
    sample_count: u32,
) -> LdrMeshPipelines {
    let depth_stencil = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    };

    let make = |label: &str,
                cull: Option<wgpu::Face>,
                blend: Option<wgpu::BlendState>,
                topo: wgpu::PrimitiveTopology,
                depth_write: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: topo,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: cull,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: depth_write,
                ..depth_stencil.clone()
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    };

    LdrMeshPipelines {
        solid: make(
            "solid_pipeline",
            Some(wgpu::Face::Back),
            None,
            wgpu::PrimitiveTopology::TriangleList,
            true,
        ),
        solid_two_sided: make(
            "solid_two_sided_pipeline",
            None,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            true,
        ),
        transparent: make(
            "transparent_pipeline",
            None,
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::Zero,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            wgpu::PrimitiveTopology::TriangleList,
            false,
        ),
        wireframe: make(
            "wireframe_pipeline",
            None,
            None,
            wgpu::PrimitiveTopology::LineList,
            true,
        ),
    }
}

/// The four HDR `mesh.wgsl` pipelines that draw into the Rgba16Float
/// intermediate.
pub(crate) struct HdrMeshPipelines {
    pub solid: wgpu::RenderPipeline,
    pub solid_two_sided: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
    pub wireframe: wgpu::RenderPipeline,
}

pub(crate) fn build_hdr_mesh_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> HdrMeshPipelines {
    let make = |label: &str,
                cull: Option<wgpu::Face>,
                blend: Option<wgpu::BlendState>,
                topo: wgpu::PrimitiveTopology,
                depth_write: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: topo,
                cull_mode: cull,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: depth_write,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    };

    HdrMeshPipelines {
        solid: make(
            "hdr_solid_pipeline",
            Some(wgpu::Face::Back),
            None,
            wgpu::PrimitiveTopology::TriangleList,
            true,
        ),
        solid_two_sided: make(
            "hdr_solid_two_sided_pipeline",
            None,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            true,
        ),
        transparent: make(
            "hdr_transparent_pipeline",
            None,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::PrimitiveTopology::TriangleList,
            false,
        ),
        wireframe: make(
            "hdr_wireframe_pipeline",
            None,
            None,
            wgpu::PrimitiveTopology::LineList,
            true,
        ),
    }
}

/// `mesh_oit.wgsl`: weighted-blended OIT pipeline. Draws into the
/// `Rgba16Float` accumulation target and the `R8Unorm` reveal target.
pub(crate) fn build_oit_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let accum_blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    };
    let reveal_blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::OneMinusSrc,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::OneMinusSrc,
            operation: wgpu::BlendOperation::Add,
        },
    };
    let depth_stencil = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::LessEqual,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("oit_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_oit_main"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(accum_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: Some(reveal_blend),
                    write_mask: wgpu::ColorWrites::RED,
                }),
            ],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(depth_stencil),
        multisample: wgpu::MultisampleState {
            count: 1,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    })
}

/// `shadow.wgsl`: depth-only shadow pass pipeline.
pub(crate) fn build_shadow_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Front),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

/// `outline_mask.wgsl`: two pipelines that rasterise the selection
/// silhouette into the R8 mask texture.
pub(crate) struct OutlineMaskPipelines {
    pub mask: wgpu::RenderPipeline,
    pub mask_two_sided: wgpu::RenderPipeline,
}

pub(crate) fn build_outline_mask_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    mask_format: wgpu::TextureFormat,
) -> OutlineMaskPipelines {
    let make = |label: &str, cull: Option<wgpu::Face>| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: mask_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: cull,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    };
    OutlineMaskPipelines {
        mask: make("outline_mask_pipeline", Some(wgpu::Face::Back)),
        mask_two_sided: make("outline_mask_two_sided_pipeline", None),
    }
}

/// Build the shared mesh pipeline layout used by both LDR and HDR mesh
/// pipelines. Camera + object + deform BGLs at slots 0, 1, 2.
pub(crate) fn mesh_pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    camera_bgl: &wgpu::BindGroupLayout,
    object_bgl: &wgpu::BindGroupLayout,
    deform_bgl: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[camera_bgl, object_bgl, deform_bgl],
        push_constant_ranges: &[],
    })
}
