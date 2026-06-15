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

/// Depth bias values used by every CSM shadow caster pipeline (the per-item
/// pipeline here and the two instanced pipelines in `instancing.rs`). Pulled
/// into one place so the three pipelines stay aligned: a difference in bias
/// between them shows up as a visible step where the active draw path changes.
///
/// `constant` is a fixed depth offset in the cascade's NDC, tiny in
/// Depth32Float (about 5e-7 NDC per unit), enough to close the coplanar
/// leak class without detaching the shadow at contact points.
///
/// `slope_scale` is held at zero on purpose. Receiver-side normal bias in
/// `sample_shadow_csm` already scales `texel_world * 1.5` at grazing
/// angles, which is where slope-scaled caster bias would otherwise earn
/// its keep. Stacking a slope-scaled caster bias on top of that visibly
/// detaches shadows from grazing-angle casters (tall walls lit obliquely
/// were the test case).
pub(crate) const CSM_SHADOW_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: 2,
    slope_scale: 0.0,
    clamp: 0.0,
};

/// `shadow.wgsl`: depth-only shadow pass pipeline.
///
/// `cull_mode` selects which faces are rasterised into the shadow atlas:
/// - `Some(Face::Front)` for closed solids (`BackfacePolicy::Cull`). Back
///   faces become the casters, so a solid's own front face is never compared
///   against itself in the shadow map.
/// - `None` for two-sided surfaces (`BackfacePolicy::Identical` and friends,
///   typically single-quad planes, cloth, foliage). Both sides rasterise so
///   the surface can still cast a shadow regardless of which side faces the
///   light. The receiver-side normal bias is what keeps the self-shadow
///   class quiet on this path.
pub(crate) fn build_shadow_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    cull_mode: Option<wgpu::Face>,
) -> wgpu::RenderPipeline {
    let label = match cull_mode {
        Some(wgpu::Face::Front) => "shadow_pipeline",
        Some(wgpu::Face::Back) => "shadow_pipeline_cull_back",
        None => "shadow_pipeline_two_sided",
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
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
            cull_mode,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: CSM_SHADOW_BIAS,
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

/// `shadow_point.wgsl`: depth + fragment pipeline for point-light cubemap
/// shadow faces. Writes linear distance-to-light to `frag_depth`.
///
/// Uses back-face culling (front faces visible to the light are rendered) so
/// the cubemap stores the near-side distance of each occluder. With linear
/// distance and front-face culling, the stored depth is the object's far
/// side, which bakes in an implicit "bias = object thickness" and causes
/// peter-panning on objects sitting flush against receivers. A small slope-
/// scale bias offsets shadow acne on the lit side without re-introducing the
/// gap.
pub(crate) fn build_shadow_point_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow_point_pipeline"),
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
            targets: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            // Y-flipped projection reverses triangle winding in screen space:
            // a triangle that's CCW in world space rasterises as CW after the
            // flip. Treating CW as the front face keeps back-face culling
            // (cull_mode: Back) working in the original sense: the surfaces
            // facing the light are kept, the surfaces facing away are culled.
            front_face: wgpu::FrontFace::Cw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
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

/// LDR `mesh_instanced.wgsl` pipelines: solid + transparent, both
/// drawing through `vs_main` with the instance storage buffer at
/// group 1.
pub(crate) struct LdrInstancedMeshPipelines {
    pub solid: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
}

pub(crate) fn build_ldr_instanced_mesh_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
    sample_count: u32,
) -> LdrInstancedMeshPipelines {
    let make = |label: &str,
                cull: Option<wgpu::Face>,
                blend: Option<wgpu::BlendState>,
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
                topology: wgpu::PrimitiveTopology::TriangleList,
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
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    };
    LdrInstancedMeshPipelines {
        solid: make("solid_instanced_pipeline", Some(wgpu::Face::Back), None, true),
        transparent: make(
            "transparent_instanced_pipeline",
            None,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
        ),
    }
}

/// HDR `mesh_instanced.wgsl` pipelines, all `vs_main`. Includes the
/// additive and premultiplied variants the particle system draws into.
pub(crate) struct HdrInstancedMeshPipelines {
    pub solid: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
    pub additive: wgpu::RenderPipeline,
    pub premultiplied: wgpu::RenderPipeline,
}

pub(crate) fn build_hdr_instanced_mesh_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> HdrInstancedMeshPipelines {
    let additive_blend = wgpu::BlendState {
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
    let premultiplied_blend = wgpu::BlendState {
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
    let make = |label: &str,
                cull: Option<wgpu::Face>,
                blend: Option<wgpu::BlendState>,
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
                topology: wgpu::PrimitiveTopology::TriangleList,
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
    HdrInstancedMeshPipelines {
        solid: make(
            "hdr_solid_instanced_pipeline",
            Some(wgpu::Face::Back),
            None,
            true,
        ),
        transparent: make(
            "hdr_transparent_instanced_pipeline",
            None,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
        ),
        additive: make(
            "hdr_instanced_additive_pipeline",
            None,
            Some(additive_blend),
            false,
        ),
        premultiplied: make(
            "hdr_instanced_premultiplied_pipeline",
            None,
            Some(premultiplied_blend),
            false,
        ),
    }
}

/// GPU-cull HDR solid pipeline: same as the HDR solid instanced pipeline
/// but using `vs_main_cull` so the compute pass can write
/// visibility indices.
pub(crate) fn build_hdr_instanced_cull_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hdr_solid_instanced_cull_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main_cull"),
            buffers: &[Vertex::buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
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
            ..Default::default()
        },
        multiview: None,
        cache: None,
    })
}

/// OIT instanced pipeline shared between the non-cull (`vs_main`) and
/// the cull (`vs_main_cull`) variants. Two color targets, depth-test
/// only.
pub(crate) fn build_oit_instanced_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    vs_entry: &str,
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
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs_entry),
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
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
}

/// Build a pipeline layout for instanced mesh-family pipelines.
/// Group 0=camera, 1=instance/cull, 2=deform.
pub(crate) fn instanced_pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    camera_bgl: &wgpu::BindGroupLayout,
    instance_bgl: &wgpu::BindGroupLayout,
    deform_bgl: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[camera_bgl, instance_bgl, deform_bgl],
        push_constant_ranges: &[],
    })
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
