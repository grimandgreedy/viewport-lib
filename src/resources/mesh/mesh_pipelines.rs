//! Factory functions for the mesh-family pipelines that share a single
//! shader source. The same factories run at startup (from `init.rs` and
//! `postprocess.rs`) and from the deformer registry's pipeline rebuild
//! path, so a registered deformer can swap in a freshly composed
//! `ShaderModule` without duplicating pipeline descriptors.

use crate::resources::types::Vertex;

/// The four LDR `mesh.wgsl` pipelines that draw into the swapchain.
pub(crate) struct LdrMeshPipelines {
    pub solid: crate::gpu::RenderPipeline,
    pub solid_two_sided: crate::gpu::RenderPipeline,
    pub transparent: crate::gpu::RenderPipeline,
    pub wireframe: crate::gpu::RenderPipeline,
}

pub(crate) fn build_ldr_mesh_pipelines(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    target_format: crate::gpu::TextureFormat,
    sample_count: u32,
    cache: Option<&crate::gpu::PipelineCache>,
) -> LdrMeshPipelines {
    let depth_stencil =
        crate::resources::builders::scene_depth_stencil(true, crate::gpu::CompareFunction::Less);

    let make = |label: &str,
                cull: Option<crate::gpu::Face>,
                blend: Option<crate::gpu::BlendState>,
                topo: crate::gpu::PrimitiveTopology,
                depth_write: bool| {
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label,
                layout,
                vertex: crate::gpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: topo,
                    strip_index_format: None,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: cull,
                    unclipped_depth: false,
                    polygon_mode: crate::gpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(crate::gpu::DepthStencilState {
                    depth_write_enabled: crate::resources::builders::dwrite(depth_write),
                    ..depth_stencil.clone()
                }),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache,
            },
        )
    };

    LdrMeshPipelines {
        solid: make(
            "solid_pipeline",
            Some(crate::gpu::Face::Back),
            None,
            crate::gpu::PrimitiveTopology::TriangleList,
            true,
        ),
        solid_two_sided: make(
            "solid_two_sided_pipeline",
            None,
            None,
            crate::gpu::PrimitiveTopology::TriangleList,
            true,
        ),
        transparent: make(
            "transparent_pipeline",
            None,
            Some(crate::gpu::BlendState {
                color: crate::gpu::BlendComponent {
                    src_factor: crate::gpu::BlendFactor::SrcAlpha,
                    dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                    operation: crate::gpu::BlendOperation::Add,
                },
                alpha: crate::gpu::BlendComponent {
                    src_factor: crate::gpu::BlendFactor::One,
                    dst_factor: crate::gpu::BlendFactor::Zero,
                    operation: crate::gpu::BlendOperation::Add,
                },
            }),
            crate::gpu::PrimitiveTopology::TriangleList,
            false,
        ),
        wireframe: make(
            "wireframe_pipeline",
            None,
            None,
            crate::gpu::PrimitiveTopology::LineList,
            true,
        ),
    }
}

/// The four HDR `mesh.wgsl` pipelines that draw into the Rgba16Float
/// intermediate.
pub(crate) struct HdrMeshPipelines {
    pub solid: crate::gpu::RenderPipeline,
    pub solid_two_sided: crate::gpu::RenderPipeline,
    pub transparent: crate::gpu::RenderPipeline,
    pub wireframe: crate::gpu::RenderPipeline,
}

pub(crate) fn build_hdr_mesh_pipelines(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
) -> HdrMeshPipelines {
    let make = |label: &str,
                cull: Option<crate::gpu::Face>,
                blend: Option<crate::gpu::BlendState>,
                topo: crate::gpu::PrimitiveTopology,
                depth_write: bool| {
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label,
                layout,
                vertex: crate::gpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: topo,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    depth_write,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        )
    };

    HdrMeshPipelines {
        solid: make(
            "hdr_solid_pipeline",
            Some(crate::gpu::Face::Back),
            None,
            crate::gpu::PrimitiveTopology::TriangleList,
            true,
        ),
        solid_two_sided: make(
            "hdr_solid_two_sided_pipeline",
            None,
            None,
            crate::gpu::PrimitiveTopology::TriangleList,
            true,
        ),
        transparent: make(
            "hdr_transparent_pipeline",
            None,
            Some(crate::gpu::BlendState::ALPHA_BLENDING),
            crate::gpu::PrimitiveTopology::TriangleList,
            false,
        ),
        wireframe: make(
            "hdr_wireframe_pipeline",
            None,
            None,
            crate::gpu::PrimitiveTopology::LineList,
            true,
        ),
    }
}

/// `mesh_oit.wgsl`: weighted-blended OIT pipeline. Draws into the
/// `Rgba16Float` accumulation target and the `R8Unorm` reveal target.
pub(crate) fn build_oit_pipeline(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
) -> crate::gpu::RenderPipeline {
    let accum_blend = crate::gpu::BlendState {
        color: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::One,
            dst_factor: crate::gpu::BlendFactor::One,
            operation: crate::gpu::BlendOperation::Add,
        },
        alpha: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::One,
            dst_factor: crate::gpu::BlendFactor::One,
            operation: crate::gpu::BlendOperation::Add,
        },
    };
    let reveal_blend = crate::gpu::BlendState {
        color: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::Zero,
            dst_factor: crate::gpu::BlendFactor::OneMinusSrc,
            operation: crate::gpu::BlendOperation::Add,
        },
        alpha: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::Zero,
            dst_factor: crate::gpu::BlendFactor::OneMinusSrc,
            operation: crate::gpu::BlendOperation::Add,
        },
    };
    let depth_stencil = crate::resources::builders::scene_depth_stencil(
        false,
        crate::gpu::CompareFunction::LessEqual,
    );
    crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label: "oit_pipeline",
            layout,
            vertex: crate::gpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some("fs_oit_main"),
                targets: &[
                    Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        blend: Some(accum_blend),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    }),
                    Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::R8Unorm,
                        blend: Some(reveal_blend),
                        write_mask: crate::gpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            multisample: crate::gpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            cache: None,
        },
    )
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
pub(crate) const CSM_SHADOW_BIAS: crate::gpu::DepthBiasState = crate::gpu::DepthBiasState {
    constant: 2,
    slope_scale: 0.0,
    clamp: 0.0,
};

/// Depth bias for the cull-none variant used by two-sided materials
/// (`BackfacePolicy::Identical` and friends). On the cull-none path the
/// receiver and caster are the same surface (e.g. a plane rasterised into
/// the shadow map at its own depth, then sampled by its own fragment), so
/// the caster-side bias has to outpace the receiver-side normal bias in
/// `sample_shadow_csm` or the receiver self-shadows uniformly. With the
/// default cull-front caster bias this surface reads as a broad dark patch.
///
/// `constant: 1000` in Depth32Float is ~1e-4 NDC, comfortably above the
/// perpendicular-receiver bias floor. `slope_scale: 8.0` pushes each caster
/// polygon's recorded depth away from the light in proportion to its slope in
/// light space. A wavy two-sided surface (a cloth sheet, a scalar-field graph)
/// has steep polygons that vary a lot in depth within one shadow texel, so
/// without a strong slope term the surface reads its own quantized depth as an
/// occluder and self-shadows in blocky, triangle-aligned patches (shadow acne)
/// that stay visible even zoomed in. Scaling the caster depth by the slope
/// clears that on the steep folds while leaving flat parts untouched.
///
/// This lives on the caster side of the two-sided (cull-none) pipeline only, so
/// one-sided receivers (ground planes, solids) are not touched at all: their
/// cast shadows stay pinned to their casters with no peter-panning. The cost is
/// confined to two-sided surfaces resting on something else, whose contact
/// shadow can lift by roughly `slope_scale` shadow texels; drop the factor if a
/// draped two-sided surface ever needs a tighter contact.
pub(crate) const CSM_SHADOW_BIAS_TWO_SIDED: crate::gpu::DepthBiasState =
    crate::gpu::DepthBiasState {
        constant: 1000,
        slope_scale: 8.0,
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
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    cull_mode: Option<crate::gpu::Face>,
    cache: Option<&crate::gpu::PipelineCache>,
) -> crate::gpu::RenderPipeline {
    let label = match cull_mode {
        Some(crate::gpu::Face::Front) => "shadow_pipeline",
        Some(crate::gpu::Face::Back) => "shadow_pipeline_cull_back",
        None => "shadow_pipeline_two_sided",
    };
    crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout,
            vertex: crate::gpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: crate::gpu::FrontFace::Ccw,
                cull_mode,
                unclipped_depth: false,
                polygon_mode: crate::gpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(crate::gpu::DepthStencilState {
                format: crate::gpu::TextureFormat::Depth32Float,
                depth_write_enabled: crate::resources::builders::dwrite(true),
                depth_compare: crate::resources::builders::dcompare(
                    crate::gpu::CompareFunction::Less,
                ),
                stencil: crate::gpu::StencilState::default(),
                bias: if cull_mode.is_none() {
                    CSM_SHADOW_BIAS_TWO_SIDED
                } else {
                    CSM_SHADOW_BIAS
                },
            }),
            multisample: crate::gpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache,
        },
    )
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
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    cache: Option<&crate::gpu::PipelineCache>,
) -> crate::gpu::RenderPipeline {
    crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label: "shadow_point_pipeline",
            layout,
            vertex: crate::gpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Y-flipped projection reverses triangle winding in screen space:
                // a triangle that's CCW in world space rasterises as CW after the
                // flip. Treating CW as the front face keeps back-face culling
                // (cull_mode: Back) working in the original sense: the surfaces
                // facing the light are kept, the surfaces facing away are culled.
                front_face: crate::gpu::FrontFace::Cw,
                cull_mode: Some(crate::gpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: crate::gpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(crate::resources::builders::depth_stencil(
                crate::gpu::TextureFormat::Depth32Float,
                true,
                crate::gpu::CompareFunction::Less,
            )),
            multisample: crate::gpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache,
        },
    )
}

/// `outline_mask.wgsl`: two pipelines that rasterise the selection
/// silhouette into the R8 mask texture.
pub(crate) struct OutlineMaskPipelines {
    pub mask: crate::gpu::RenderPipeline,
    pub mask_two_sided: crate::gpu::RenderPipeline,
}

pub(crate) fn build_outline_mask_pipelines(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    mask_format: crate::gpu::TextureFormat,
    cache: Option<&crate::gpu::PipelineCache>,
) -> OutlineMaskPipelines {
    let make = |label: &str, cull: Option<crate::gpu::Face>| {
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label,
                layout,
                vertex: crate::gpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: mask_format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    true,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache,
            },
        )
    };
    OutlineMaskPipelines {
        mask: make("outline_mask_pipeline", Some(crate::gpu::Face::Back)),
        mask_two_sided: make("outline_mask_two_sided_pipeline", None),
    }
}

/// LDR `mesh_instanced.wgsl` pipelines: solid + transparent, both
/// drawing through `vs_main` with the instance storage buffer at
/// group 1.
pub(crate) struct LdrInstancedMeshPipelines {
    pub solid: crate::gpu::RenderPipeline,
    /// Same as `solid` but with `cull_mode: None` for two-sided (`Identical`
    /// backface policy) meshes.
    pub solid_two_sided: crate::gpu::RenderPipeline,
    pub transparent: crate::gpu::RenderPipeline,
}

pub(crate) fn build_ldr_instanced_mesh_pipelines(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    target_format: crate::gpu::TextureFormat,
    sample_count: u32,
) -> LdrInstancedMeshPipelines {
    let make = |label: &str,
                cull: Option<crate::gpu::Face>,
                blend: Option<crate::gpu::BlendState>,
                depth_write: bool| {
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label,
                layout,
                vertex: crate::gpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: target_format,
                        blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    depth_write,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    };
    LdrInstancedMeshPipelines {
        solid: make(
            "solid_instanced_pipeline",
            Some(crate::gpu::Face::Back),
            None,
            true,
        ),
        solid_two_sided: make("solid_two_sided_instanced_pipeline", None, None, true),
        transparent: make(
            "transparent_instanced_pipeline",
            None,
            Some(crate::gpu::BlendState::ALPHA_BLENDING),
            false,
        ),
    }
}

/// HDR `mesh_instanced.wgsl` pipelines, all `vs_main`. Includes the
/// additive and premultiplied variants the particle system draws into.
pub(crate) struct HdrInstancedMeshPipelines {
    pub solid: crate::gpu::RenderPipeline,
    /// Same as `solid` but with `cull_mode: None` for two-sided (`Identical`
    /// backface policy) meshes.
    pub solid_two_sided: crate::gpu::RenderPipeline,
    pub transparent: crate::gpu::RenderPipeline,
    pub additive: crate::gpu::RenderPipeline,
    pub premultiplied: crate::gpu::RenderPipeline,
}

pub(crate) fn build_hdr_instanced_mesh_pipelines(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
) -> HdrInstancedMeshPipelines {
    let additive_blend = crate::gpu::BlendState {
        color: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::One,
            dst_factor: crate::gpu::BlendFactor::One,
            operation: crate::gpu::BlendOperation::Add,
        },
        alpha: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::One,
            dst_factor: crate::gpu::BlendFactor::One,
            operation: crate::gpu::BlendOperation::Add,
        },
    };
    let premultiplied_blend = crate::gpu::BlendState {
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
    let make = |label: &str,
                cull: Option<crate::gpu::Face>,
                blend: Option<crate::gpu::BlendState>,
                depth_write: bool| {
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label,
                layout,
                vertex: crate::gpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        blend,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    depth_write,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        )
    };
    HdrInstancedMeshPipelines {
        solid: make(
            "hdr_solid_instanced_pipeline",
            Some(crate::gpu::Face::Back),
            None,
            true,
        ),
        solid_two_sided: make("hdr_solid_two_sided_instanced_pipeline", None, None, true),
        transparent: make(
            "hdr_transparent_instanced_pipeline",
            None,
            Some(crate::gpu::BlendState::ALPHA_BLENDING),
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

/// Opaque solid + two-sided pipeline pair from an already-built module.
/// Matches the `solid` / `solid_two_sided` entries of
/// `build_ldr_instanced_mesh_pipelines` / `build_hdr_instanced_mesh_pipelines`
/// (no blend, depth write on, `Less`); used for the discard-free twins of the
/// lit instanced pipelines.
pub(crate) fn build_instanced_solid_pipelines(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    format: crate::gpu::TextureFormat,
    sample_count: u32,
    label_solid: &str,
    label_two_sided: &str,
) -> (crate::gpu::RenderPipeline, crate::gpu::RenderPipeline) {
    let make = |label: &str, cull: Option<crate::gpu::Face>| {
        crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label,
                layout,
                vertex: crate::gpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    true,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        )
    };
    (
        make(label_solid, Some(crate::gpu::Face::Back)),
        make(label_two_sided, None),
    )
}

/// GPU-cull HDR solid pipeline: same as the HDR solid instanced pipeline
/// but using `vs_main_cull` so the compute pass can write
/// visibility indices.
pub(crate) fn build_hdr_instanced_cull_pipeline(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
) -> crate::gpu::RenderPipeline {
    build_hdr_instanced_cull_pipeline_with(
        device,
        layout,
        shader,
        "hdr_solid_instanced_cull_pipeline",
        Some(crate::gpu::Face::Back),
    )
}

/// Two-sided (`cull_mode: None`) variant of the GPU-cull HDR solid pipeline,
/// used for instanced batches whose material has the `Identical` backface policy.
pub(crate) fn build_hdr_instanced_cull_two_sided_pipeline(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
) -> crate::gpu::RenderPipeline {
    build_hdr_instanced_cull_pipeline_with(
        device,
        layout,
        shader,
        "hdr_solid_instanced_cull_two_sided_pipeline",
        None,
    )
}

pub(crate) fn build_hdr_instanced_cull_pipeline_with(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    label: &str,
    cull_mode: Option<crate::gpu::Face>,
) -> crate::gpu::RenderPipeline {
    crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout,
            vertex: crate::gpu::VertexState {
                module: shader,
                entry_point: Some("vs_main_cull"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(crate::gpu::ColorTargetState {
                    format: crate::gpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: crate::gpu::ColorWrites::ALL,
                })],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                true,
                crate::gpu::CompareFunction::Less,
            )),
            multisample: crate::gpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            cache: None,
        },
    )
}

/// OIT instanced pipeline shared between the non-cull (`vs_main`) and
/// the cull (`vs_main_cull`) variants. Two color targets, depth-test
/// only.
pub(crate) fn build_oit_instanced_pipeline(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    label: &str,
    vs_entry: &str,
) -> crate::gpu::RenderPipeline {
    let accum_blend = crate::gpu::BlendState {
        color: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::One,
            dst_factor: crate::gpu::BlendFactor::One,
            operation: crate::gpu::BlendOperation::Add,
        },
        alpha: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::One,
            dst_factor: crate::gpu::BlendFactor::One,
            operation: crate::gpu::BlendOperation::Add,
        },
    };
    let reveal_blend = crate::gpu::BlendState {
        color: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::Zero,
            dst_factor: crate::gpu::BlendFactor::OneMinusSrc,
            operation: crate::gpu::BlendOperation::Add,
        },
        alpha: crate::gpu::BlendComponent {
            src_factor: crate::gpu::BlendFactor::Zero,
            dst_factor: crate::gpu::BlendFactor::OneMinusSrc,
            operation: crate::gpu::BlendOperation::Add,
        },
    };
    crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout,
            vertex: crate::gpu::VertexState {
                module: shader,
                entry_point: Some(vs_entry),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some("fs_oit_main"),
                targets: &[
                    Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        blend: Some(accum_blend),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    }),
                    Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::R8Unorm,
                        blend: Some(reveal_blend),
                        write_mask: crate::gpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                false,
                crate::gpu::CompareFunction::LessEqual,
            )),
            multisample: crate::gpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            cache: None,
        },
    )
}

/// Build a pipeline layout for instanced mesh-family pipelines.
/// Groups: 0=camera, 1=instance/cull, and optionally 2=deform.
/// Pass `None` for `deform_bgl` on devices with max_bind_groups < 3.
pub(crate) fn instanced_pipeline_layout(
    device: &crate::gpu::Device,
    label: &str,
    camera_bgl: &crate::gpu::BindGroupLayout,
    instance_bgl: &crate::gpu::BindGroupLayout,
    deform_bgl: Option<&crate::gpu::BindGroupLayout>,
) -> crate::gpu::PipelineLayout {
    let layouts: Vec<&crate::gpu::BindGroupLayout> = if let Some(d) = deform_bgl {
        vec![camera_bgl, instance_bgl, d]
    } else {
        vec![camera_bgl, instance_bgl]
    };
    crate::resources::builders::pipeline_layout(device, label, &layouts)
}

/// Build the shared mesh pipeline layout used by both LDR and HDR mesh
/// pipelines. Groups: 0=camera, 1=object+texture, and optionally 2=deform.
/// Pass `None` for `deform_bgl` on devices with max_bind_groups < 3.
pub(crate) fn mesh_pipeline_layout(
    device: &crate::gpu::Device,
    label: &str,
    camera_bgl: &crate::gpu::BindGroupLayout,
    object_bgl: &crate::gpu::BindGroupLayout,
    deform_bgl: Option<&crate::gpu::BindGroupLayout>,
) -> crate::gpu::PipelineLayout {
    let layouts: Vec<&crate::gpu::BindGroupLayout> = if let Some(d) = deform_bgl {
        vec![camera_bgl, object_bgl, d]
    } else {
        vec![camera_bgl, object_bgl]
    };
    crate::resources::builders::pipeline_layout(device, label, &layouts)
}
