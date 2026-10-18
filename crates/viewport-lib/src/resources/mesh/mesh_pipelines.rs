//! Factory functions for the mesh-family pipelines that share a single
//! shader source. The same factories run at startup (from `init.rs` and
//! `postprocess.rs`) and from the deformer registry's pipeline rebuild
//! path, so a registered deformer can swap in a freshly composed
//! `ShaderModule` without duplicating pipeline descriptors.
use crate::resources::VertexBufferLayoutExt;

use crate::resources::types::Vertex;

/// The record words the `mesh.wgsl` and `mesh_oit.wgsl` vertex stages use as
/// addresses, at the [`InstanceData`](crate::resources::mesh::instancing::InstanceData)
/// offsets: the sidecar record base, the colour, warp and lightmap streams,
/// the mesh's first vertex in its slab chunk and the deform table. Locations
/// 5 to 10.
const RECORD_WORD_ATTRIBUTES: [crate::gpu::VertexAttribute; 6] = [
    word_attribute(5, 108),
    word_attribute(6, 116),
    word_attribute(7, 144),
    word_attribute(8, 148),
    word_attribute(9, 152),
    word_attribute(10, 156),
];

/// The same words as the GPU cull packs them for each survivor, six words a
/// slot.
const CULLED_WORD_ATTRIBUTES: [crate::gpu::VertexAttribute; 6] = [
    word_attribute(5, 0),
    word_attribute(6, 4),
    word_attribute(7, 8),
    word_attribute(8, 12),
    word_attribute(9, 16),
    word_attribute(10, 20),
];

/// Bytes the cull writes per survivor into the words region of the
/// visibility buffer.
pub(crate) const CULLED_WORDS_STRIDE: u64 = 24;

const fn word_attribute(location: u32, offset: u64) -> crate::gpu::VertexAttribute {
    crate::gpu::VertexAttribute {
        format: crate::gpu::VertexFormat::Uint32,
        offset,
        shader_location: location,
    }
}

/// Vertex buffers of a `mesh.wgsl` or `mesh_oit.wgsl` pipeline: the mesh
/// vertices in slot 0 and the record words, stepped per instance, in slot 1.
/// A direct draw binds the instance buffer itself as slot 1; a GPU-culled
/// draw (`culled`) binds the words region of its visibility buffer.
pub(crate) fn mesh_vertex_buffers(culled: bool) -> [crate::gpu::VertexBufferLayout<'static>; 2] {
    [
        Vertex::buffer_layout(),
        if culled {
            crate::gpu::VertexBufferLayout {
                array_stride: CULLED_WORDS_STRIDE,
                step_mode: crate::gpu::VertexStepMode::Instance,
                attributes: &CULLED_WORD_ATTRIBUTES,
            }
        } else {
            crate::gpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<crate::resources::mesh::instancing::InstanceData>(
                ) as u64,
                step_mode: crate::gpu::VertexStepMode::Instance,
                attributes: &RECORD_WORD_ATTRIBUTES,
            }
        },
    ]
}

/// One LDR `mesh.wgsl` pipeline drawing into the surface format. `constants`
/// sets the module's `override` constants (empty for the defaults).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ldr_mesh_pipeline_with(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    target_format: crate::gpu::TextureFormat,
    sample_count: u32,
    cache: Option<&crate::gpu::PipelineCache>,
    label: &str,
    cull: Option<crate::gpu::Face>,
    blend: Option<crate::gpu::BlendState>,
    topo: crate::gpu::PrimitiveTopology,
    depth_write: bool,
    constants: &[(&str, f64)],
) -> crate::gpu::RenderPipeline {
    let depth_stencil =
        crate::resources::builders::scene_depth_stencil(true, crate::gpu::CompareFunction::Less);
    crate::resources::builders::render_pipeline_with_constants(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout,
            vertex_module: shader,
            vertex_entry: "vs_main",
            vertex_buffers: &mesh_vertex_buffers(false),
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
                ..depth_stencil
            }),
            multisample: crate::gpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache,
        },
        constants,
    )
}

/// One HDR `mesh.wgsl` pipeline, drawing into the Rgba16Float intermediate.
/// `constants` sets the module's `override` constants (empty for the
/// defaults).
#[allow(clippy::too_many_arguments)]
pub(crate) fn hdr_mesh_pipeline(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    label: &str,
    cull: Option<crate::gpu::Face>,
    blend: Option<crate::gpu::BlendState>,
    topo: crate::gpu::PrimitiveTopology,
    depth_write: bool,
    constants: &[(&str, f64)],
) -> crate::gpu::RenderPipeline {
    hdr_mesh_pipeline_entry(
        device,
        layout,
        shader,
        label,
        cull,
        blend,
        topo,
        depth_write,
        constants,
        false,
    )
}

/// [`hdr_mesh_pipeline`] for a GPU-culled draw (`culled`): the `vs_main_cull`
/// entry and the cull's packed words in slot 1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hdr_mesh_pipeline_entry(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    label: &str,
    cull: Option<crate::gpu::Face>,
    blend: Option<crate::gpu::BlendState>,
    topo: crate::gpu::PrimitiveTopology,
    depth_write: bool,
    constants: &[(&str, f64)],
    culled: bool,
) -> crate::gpu::RenderPipeline {
    crate::resources::builders::render_pipeline_with_constants(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout,
            vertex_module: shader,
            vertex_entry: if culled { "vs_main_cull" } else { "vs_main" },
            vertex_buffers: &mesh_vertex_buffers(culled),
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
        constants,
    )
}

/// One OIT accumulate pipeline from `mesh_oit.wgsl`, with a label and values
/// for the module's `override` constants.
pub(crate) fn build_oit_pipeline_with(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    two_sided: bool,
    label: &str,
    constants: &[(&str, f64)],
) -> crate::gpu::RenderPipeline {
    build_oit_pipeline_entry(device, layout, shader, two_sided, label, constants, false)
}

/// [`build_oit_pipeline_with`] for a GPU-culled draw (`culled`): the
/// `vs_main_cull` entry and the cull's packed words in slot 1.
pub(crate) fn build_oit_pipeline_entry(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    two_sided: bool,
    label: &str,
    constants: &[(&str, f64)],
    culled: bool,
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
    crate::resources::builders::render_pipeline_with_constants(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout,
            vertex_module: shader,
            vertex_entry: if culled { "vs_main_cull" } else { "vs_main" },
            vertex_buffers: &mesh_vertex_buffers(culled),
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
                cull_mode: (!two_sided).then_some(crate::gpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            multisample: crate::gpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            cache: None,
        },
        constants,
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
/// perpendicular-receiver bias floor. `slope_scale: 2.0` pushes each caster
/// polygon's recorded depth away from the light in proportion to its slope in
/// light space. A wavy two-sided surface (a cloth sheet, a scalar-field graph)
/// has steep polygons that vary a lot in depth within one shadow texel, so
/// without a slope term the surface reads its own quantized depth as an
/// occluder and self-shadows in blocky, triangle-aligned patches (shadow acne)
/// that stay visible even zoomed in. Two texel-depths per unit slope clears
/// that on coarse, steep folds under a low sun; one does not.
///
/// The slope term is also a leak: every receiver behind a steep two-sided
/// caster polygon reads it as that much further away. At 8.0 the ground
/// under a draped sheet showed lit holes where the sheet's steepest polygons
/// were pushed past it, and a closed solid routed here by a styled backface
/// policy showed its exterior's depth pushed past its own interior wall.
/// Closed meshes no longer come through this pipeline (`GpuMesh::closed`
/// routes them cull-front), and 2.0 keeps the open-surface leak to a texel
/// or two.
///
/// This lives on the caster side of the two-sided (cull-none) pipeline only, so
/// one-sided receivers (ground planes, solids) are not touched at all: their
/// cast shadows stay pinned to their casters with no peter-panning. The cost is
/// confined to two-sided surfaces resting on something else, whose contact
/// shadow can lift by roughly `slope_scale` shadow texels.
pub const CSM_SHADOW_BIAS_TWO_SIDED: crate::gpu::DepthBiasState = crate::gpu::DepthBiasState {
    constant: 1000,
    slope_scale: 2.0,
    clamp: 0.0,
};

/// `shadow.wgsl`: depth-only shadow pass pipeline.
///
/// `cull_mode` selects which faces are rasterised into the shadow atlas:
/// - `Some(Face::Front)` for closed solids (`BackfacePolicy::Cull`). Back
///   faces become the casters, so a solid's own front face is never compared
///   against itself in the shadow map.
/// - `None` for two-sided surfaces (`BackfacePolicy::Identical` and friends
///   on an open mesh: single-quad planes, cloth, foliage). Both sides
///   rasterise so the surface can still cast a shadow regardless of which
///   side faces the light. A closed mesh takes the cull-front pipeline
///   whatever its policy (`GpuMesh::closed`): its back faces cast, so it
///   never compares against itself, and the cull-none slope bias cannot leak
///   through its wall.
///
/// `cutout` selects `vs_cutout`/`fs_cutout` instead of the plain `vs_main`:
/// the fragment stage samples the caster's albedo alpha and discards below
/// its cutoff, punching holes in the shadow instead of casting a solid
/// silhouette for an `AlphaMode::Mask` material. `culled` selects the
/// `_cull` entry points, which read each visible slot's record index from
/// group-1 binding 15 and draw from the cascade cull's indirect args.
pub(crate) fn build_shadow_pipeline(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    cull_mode: Option<crate::gpu::Face>,
    cutout: bool,
    culled: bool,
    cache: Option<&crate::gpu::PipelineCache>,
) -> crate::gpu::RenderPipeline {
    let label = match (cull_mode, cutout) {
        (Some(crate::gpu::Face::Front), false) => "shadow_pipeline",
        (Some(crate::gpu::Face::Back), false) => "shadow_pipeline_cull_back",
        (None, false) => "shadow_pipeline_two_sided",
        (Some(crate::gpu::Face::Front), true) => "shadow_cutout_pipeline",
        (Some(crate::gpu::Face::Back), true) => "shadow_cutout_pipeline_cull_back",
        (None, true) => "shadow_cutout_pipeline_two_sided",
    };
    let label = if culled {
        format!("{label}_cull")
    } else {
        label.to_string()
    };
    crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label: &label,
            layout,
            vertex_module: shader,
            vertex_entry: match (cutout, culled) {
                (false, false) => "vs_main",
                (false, true) => "vs_main_cull",
                (true, false) => "vs_cutout",
                (true, true) => "vs_cutout_cull",
            },
            vertex_buffers: &[Vertex::buffer_layout()],
            fragment: cutout.then_some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some("fs_cutout"),
                targets: &[],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            }),
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
/// shadow faces. Writes linear distance-to-light to `frag_depth`. `cutout`
/// selects `vs_cutout`/`fs_cutout`, which discard below an
/// `AlphaMode::Mask` caster's cutoff.
///
/// Uses back-face culling (front faces visible to the light are rendered) so
/// the cubemap stores the near-side distance of each occluder. With linear
/// distance and front-face culling, the stored depth is the object's far
/// side, which bakes in an implicit "bias = object thickness" and causes
/// peter-panning on objects sitting flush against receivers. No pipeline
/// depth bias: the fragment stage writes `frag_depth`, and the receiver side
/// (`sample_point_shadow` in the mesh shaders) carries the normal-offset and
/// constant terms that keep the lit side free of acne.
pub(crate) fn build_shadow_point_pipeline(
    device: &crate::gpu::Device,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
    label: &str,
    cutout: bool,
    cache: Option<&crate::gpu::PipelineCache>,
) -> crate::gpu::RenderPipeline {
    crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout,
            vertex_module: shader,
            vertex_entry: if cutout { "vs_cutout" } else { "vs_main" },
            vertex_buffers: &[Vertex::buffer_layout()],
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
                entry_point: Some(if cutout { "fs_cutout" } else { "fs_main" }),
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
                vertex_module: shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
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

/// Additive blend, for `MeshInstanceItem` batches that opt into
/// [`SpriteBlend::Additive`](crate::SpriteBlend::Additive).
pub(crate) const ADDITIVE_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
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

/// Premultiplied-alpha blend, for `MeshInstanceItem` batches with
/// [`SpriteBlend::Premultiplied`](crate::SpriteBlend::Premultiplied).
pub(crate) const PREMULTIPLIED_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
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

/// Build the shared mesh pipeline layout used by both LDR and HDR mesh
/// pipelines. Groups: 0=camera, 1=object+texture.
pub(crate) fn mesh_pipeline_layout(
    device: &crate::gpu::Device,
    label: &str,
    camera_bgl: &crate::gpu::BindGroupLayout,
    object_bgl: &crate::gpu::BindGroupLayout,
) -> crate::gpu::PipelineLayout {
    crate::resources::builders::pipeline_layout(device, label, &[camera_bgl, object_bgl])
}
