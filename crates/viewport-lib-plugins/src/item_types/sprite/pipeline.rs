//! GPU state for the sprite item type: the blend/depth/lit render pipeline
//! variants in each pass, the refractive pipeline and its scene-colour layout, the
//! soft-particle depth layout and its fallback, the weighted-blended OIT
//! pipelines, the pick pipeline, and the selection-outline mask pipeline. Each
//! pipeline is built the first time a draw needs it.
//!
//! The group-1 and group-3 bind group layouts live in `store`, beside the
//! upload that builds bind groups against them; this module borrows them to
//! build pipelines over them.

use viewport_lib::plugin_api::builders::DualPipelineDesc;
use viewport_lib::plugin_api::shared_wgsl;
use viewport_lib::resources::DeviceResources;

/// How a sprite colour pipeline uses the depth buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SpriteDepth {
    /// Tests and writes depth: the opaque scene pass.
    Write,
    /// Tests depth without writing it.
    Test,
    /// Neither tests nor writes: a painter batch drawn over everything.
    Off,
}

impl SpriteDepth {
    const ALL: [Self; 3] = [Self::Write, Self::Test, Self::Off];
}

/// Sprite pipeline variant axes: depth use, blend mode, unlit vs
/// `apply_scene_lighting`-lit shading, and the effect family. Kept separate from the mesh family's
/// `PipelineKey` -- sprite's axes don't map onto `two_sided` / `cutout` /
/// `no_discard_eligible`, and `blend` is five-valued, not boolean, so
/// reusing that type would just be confusing field names for an unrelated
/// set of axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SpriteKey {
    pub depth: SpriteDepth,
    pub blend: viewport_lib::renderer::SpriteBlend,
    pub lit: bool,
    /// `None` for the plain shaders; `Some(0)` for the built-in effects and
    /// `Some(1 + i)` for sprite material `i`. Each family is its own set of
    /// pipelines over the other three axes, so a frame with no effects never
    /// builds or binds one.
    pub fx: Option<u32>,
}

/// Every blend a sprite pipeline is built for, in slot order.
const BLENDS: [viewport_lib::renderer::SpriteBlend; 5] = [
    viewport_lib::renderer::SpriteBlend::AlphaBlend,
    viewport_lib::renderer::SpriteBlend::Additive,
    viewport_lib::renderer::SpriteBlend::Premultiplied,
    viewport_lib::renderer::SpriteBlend::Multiply,
    viewport_lib::renderer::SpriteBlend::Screen,
];

/// Number of [`SpriteKey`] slots.
const KEY_COUNT: usize = 3 * BLENDS.len() * 2;

impl SpriteKey {
    /// The key one batch's draw data selects.
    pub fn of(sprite: &super::store::SpriteGpuData) -> Self {
        let depth = match sprite.painter {
            Some(p) if !p.depth_test => SpriteDepth::Off,
            _ if sprite.depth_write => SpriteDepth::Write,
            _ => SpriteDepth::Test,
        };
        Self {
            depth,
            blend: sprite.blend,
            lit: sprite.lit,
            fx: sprite.fx.as_ref().map(|fx| fx.family),
        }
    }

    fn blend_index(self) -> usize {
        BLENDS
            .iter()
            .position(|b| *b == self.blend)
            .expect("every blend has a slot")
    }

    /// Every combination of the depth, blend and lit axes, in the plain family.
    pub fn all() -> impl Iterator<Item = SpriteKey> {
        BLENDS.into_iter().flat_map(|blend| {
            [false, true].into_iter().flat_map(move |lit| {
                SpriteDepth::ALL.into_iter().map(move |depth| SpriteKey {
                    depth,
                    blend,
                    lit,
                    fx: None,
                })
            })
        })
    }

    /// This key in effect family `fx`.
    pub fn in_family(self, fx: Option<u32>) -> Self {
        Self { fx, ..self }
    }

    /// Dense index in `0..KEY_COUNT` within the key's family, stable across
    /// calls.
    fn slot(self) -> usize {
        self.depth as usize + 3 * self.blend_index() + 3 * BLENDS.len() * (self.lit as usize)
    }

    /// The plain-family key at `slot`; the inverse of [`slot`](Self::slot).
    fn from_slot(slot: usize) -> Self {
        Self {
            depth: SpriteDepth::ALL[slot % 3],
            blend: BLENDS[(slot / 3) % BLENDS.len()],
            lit: slot >= 3 * BLENDS.len(),
            fx: None,
        }
    }
}

/// The per-sprite position, one vec3 per sprite with instance stepping, as
/// every sprite pipeline binds it.
const POSITION_ATTRS: [viewport_lib::gpu::VertexAttribute; 1] =
    [viewport_lib::gpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: viewport_lib::gpu::VertexFormat::Float32x3,
    }];

fn position_layout() -> viewport_lib::gpu::VertexBufferLayout<'static> {
    viewport_lib::gpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: viewport_lib::gpu::VertexStepMode::Instance,
        attributes: &POSITION_ATTRS,
    }
}

/// The packed effect word, one `u32` per sprite with instance stepping, as the
/// effect pipelines bind it at slot 1.
const FX_ATTRS: [viewport_lib::gpu::VertexAttribute; 1] = [viewport_lib::gpu::VertexAttribute {
    offset: 0,
    shader_location: 1,
    format: viewport_lib::gpu::VertexFormat::Uint32,
}];

fn fx_layout() -> viewport_lib::gpu::VertexBufferLayout<'static> {
    viewport_lib::gpu::VertexBufferLayout {
        array_stride: super::effects::FX_STRIDE as u64,
        step_mode: viewport_lib::gpu::VertexStepMode::Instance,
        attributes: &FX_ATTRS,
    }
}

/// What a sprite colour pipeline build reads: the unlit and lit shader and
/// layout pairs, and the LDR target.
pub(super) struct SpriteRecipe {
    device: viewport_lib::gpu::Device,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    lit_layout: viewport_lib::gpu::PipelineLayout,
    lit_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
    depth_format: viewport_lib::gpu::TextureFormat,
    /// An effect family: the shaders read the effect stream at slot 1.
    fx: bool,
}

/// The pass a sprite colour pipeline draws into. The HDR scene pass has the
/// indirect fraction as a second target; the read-only-depth pass has the
/// HDR colour alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SpritePass {
    Ldr,
    Scene,
    DepthRead,
}

impl SpritePass {
    const ALL: [Self; 3] = [Self::Ldr, Self::Scene, Self::DepthRead];
}

/// The sprite colour pipelines: each [`SpriteKey`] for each [`SpritePass`],
/// built the first time a draw selects it.
pub(super) type SpritePipelines =
    viewport_lib::plugin_api::LazyPipelines<SpriteRecipe, { KEY_COUNT * 3 }>;

/// Member index of `key` in `pass`.
pub(super) fn sprite_index(key: SpriteKey, pass: SpritePass) -> usize {
    key.slot() * 3 + pass as usize
}

fn build_sprite(r: &SpriteRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let key = SpriteKey::from_slot(i / 3);
    let pass = SpritePass::ALL[i % 3];
    let plain = [position_layout()];
    let with_fx = [position_layout(), fx_layout()];
    let vertex_buffers: &[viewport_lib::gpu::VertexBufferLayout<'_>] =
        if r.fx { &with_fx } else { &plain };
    let blend = match key.blend {
        viewport_lib::renderer::SpriteBlend::AlphaBlend => {
            viewport_lib::gpu::BlendState::ALPHA_BLENDING
        }
        viewport_lib::renderer::SpriteBlend::Additive => {
            viewport_lib::plugin_api::builders::ADDITIVE_BLEND
        }
        viewport_lib::renderer::SpriteBlend::Premultiplied => {
            viewport_lib::plugin_api::builders::PREMULTIPLIED_BLEND
        }
        // The fragment shapes its output for these two (see `sprite_blend_output`),
        // so each is one fixed-function state.
        viewport_lib::renderer::SpriteBlend::Multiply => MULTIPLY_BLEND,
        viewport_lib::renderer::SpriteBlend::Screen => SCREEN_BLEND,
    };
    let (layout, shader, label) = match (key.lit, r.fx) {
        (true, false) => (
            &r.lit_layout,
            r.lit_shader.get(),
            "sprite_lit_pipeline_variant",
        ),
        (false, false) => (&r.layout, r.shader.get(), "sprite_pipeline_variant"),
        (true, true) => (
            &r.lit_layout,
            r.lit_shader.get(),
            "sprite_lit_fx_pipeline_variant",
        ),
        (false, true) => (&r.layout, r.shader.get(), "sprite_fx_pipeline_variant"),
    };
    viewport_lib::plugin_api::builders::build_dual_pipeline_variant(
        &r.device,
        &DualPipelineDesc {
            label,
            layout,
            shader,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            vertex_buffers,
            blend: Some(blend),
            topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            depth_write: key.depth == SpriteDepth::Write,
            depth_compare: if key.depth == SpriteDepth::Off {
                viewport_lib::gpu::CompareFunction::Always
            } else {
                viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT
            },
            sample_count: r.sample_count,
            // The depth-read pass's one target is the HDR colour.
            ldr_format: if pass == SpritePass::DepthRead {
                viewport_lib::resources::HDR_COLOR_FORMAT
            } else {
                r.ldr_format
            },
            depth_format: r.depth_format,
            indirect_fraction_entry: Some("fs_scene"),
        },
        pass == SpritePass::Scene,
    )
}

/// The alpha of the multiply and screen blends: the sprite's coverage over
/// the destination's, as alpha blending writes it. Where nothing else drew,
/// the HDR path composites the background under the scene by this alpha.
const COVERAGE: viewport_lib::gpu::BlendComponent = viewport_lib::gpu::BlendComponent {
    src_factor: viewport_lib::gpu::BlendFactor::One,
    dst_factor: viewport_lib::gpu::BlendFactor::OneMinusSrcAlpha,
    operation: viewport_lib::gpu::BlendOperation::Add,
};

/// `dst * src`: the fragment has already mixed its colour towards white by
/// `1 - alpha`, so a transparent texel leaves the destination unchanged.
const MULTIPLY_BLEND: viewport_lib::gpu::BlendState = viewport_lib::gpu::BlendState {
    color: viewport_lib::gpu::BlendComponent {
        src_factor: viewport_lib::gpu::BlendFactor::Zero,
        dst_factor: viewport_lib::gpu::BlendFactor::Src,
        operation: viewport_lib::gpu::BlendOperation::Add,
    },
    alpha: COVERAGE,
};

/// `src + dst * (1 - src)`: the fragment has already premultiplied its colour
/// by alpha.
const SCREEN_BLEND: viewport_lib::gpu::BlendState = viewport_lib::gpu::BlendState {
    color: viewport_lib::gpu::BlendComponent {
        src_factor: viewport_lib::gpu::BlendFactor::One,
        dst_factor: viewport_lib::gpu::BlendFactor::OneMinusSrc,
        operation: viewport_lib::gpu::BlendOperation::Add,
    },
    alpha: COVERAGE,
};

/// Members of [`SpritePassPipelines`].
pub(super) const REFRACTION: usize = 0;
pub(super) const PICK: usize = 1;
pub(super) const OUTLINE_MASK: usize = 2;
pub(super) const SURFACE_MASK: usize = 3;
pub(super) const OIT: usize = 4;
pub(super) const OIT_PREMULTIPLIED: usize = 5;
pub(super) const OIT_LIT: usize = 6;
pub(super) const OIT_LIT_PREMULTIPLIED: usize = 7;
/// Pick for painter batches: drawn in the painter sequence with `LessEqual`
/// and no depth write, so of two overlapping draws the later (the one drawn on
/// top) wins.
pub(super) const PICK_PAINTER: usize = 8;
/// Pick for painter batches that ignore scene depth.
pub(super) const PICK_PAINTER_NO_TEST: usize = 9;

/// The OIT member a sprite with these fields draws through.
pub(super) fn oit_index(lit: bool, blend: viewport_lib::renderer::SpriteBlend) -> usize {
    let premultiplied = blend == viewport_lib::renderer::SpriteBlend::Premultiplied;
    match (lit, premultiplied) {
        (true, true) => OIT_LIT_PREMULTIPLIED,
        (true, false) => OIT_LIT,
        (false, true) => OIT_PREMULTIPLIED,
        (false, false) => OIT,
    }
}

/// What a build of the pipelines outside the colour variants reads.
pub(super) struct SpritePassRecipe {
    device: viewport_lib::gpu::Device,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    refraction_layout: viewport_lib::gpu::PipelineLayout,
    refraction_shader: viewport_lib::plugin_api::LazyModule,
    oit_layout: viewport_lib::gpu::PipelineLayout,
    oit_shader: viewport_lib::plugin_api::LazyModule,
    oit_lit_layout: viewport_lib::gpu::PipelineLayout,
    oit_lit_shader: viewport_lib::plugin_api::LazyModule,
    mask_layout: viewport_lib::gpu::PipelineLayout,
    mask_shader: viewport_lib::plugin_api::LazyModule,
    pick_layout: viewport_lib::gpu::PipelineLayout,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
    depth_format: viewport_lib::gpu::TextureFormat,
}

/// The refractive, three pick, outline mask, surface mask and four OIT
/// pipelines, each built the first time a draw needs it.
pub(super) type SpritePassPipelines = viewport_lib::plugin_api::LazyPipelines<SpritePassRecipe, 10>;

fn build_pass(r: &SpritePassRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let vertex_buffers = [position_layout()];
    match i {
        // Group 2 is the scene-colour copy. HDR only: the LDR path has no
        // scene colour to copy.
        REFRACTION => viewport_lib::plugin_api::builders::render_pipeline(
            &r.device,
            viewport_lib::plugin_api::builders::RenderPipelineDesc {
                label: "sprite_refraction_pipeline",
                layout: &r.refraction_layout,
                vertex_module: r.refraction_shader.get(),
                vertex_entry: "vs_main",
                vertex_buffers: &vertex_buffers,
                fragment: Some(viewport_lib::gpu::FragmentState {
                    module: r.refraction_shader.get(),
                    entry_point: Some("fs_main"),
                    targets: &[Some(viewport_lib::gpu::ColorTargetState {
                        format: viewport_lib::gpu::TextureFormat::Rgba16Float,
                        blend: Some(viewport_lib::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: viewport_lib::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: viewport_lib::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(viewport_lib::plugin_api::builders::scene_depth_stencil(
                    r.depth_format,
                    false,
                    viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
                )),
                multisample: viewport_lib::gpu::MultisampleState {
                    count: r.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        ),
        PICK | PICK_PAINTER | PICK_PAINTER_NO_TEST => pick_pipeline(
            &r.device,
            &r.pick_layout,
            r.pick_shader.get(),
            i,
            r.depth_format,
        ),
        OUTLINE_MASK => outline_mask_pipeline(
            &r.device,
            &r.mask_layout,
            r.mask_shader.get(),
            r.depth_format,
        ),
        // The unlit colour shader again, stamping the scene stencil. It
        // discards where the colour pass did, so a cut-out sprite stamps only
        // what it drew.
        SURFACE_MASK => viewport_lib::plugin_api::builders::build_surface_mask_pipeline(
            &r.device,
            "sprite_surface_mask_pipeline",
            &r.layout,
            r.shader.get(),
            r.depth_format,
            &vertex_buffers,
            None,
        ),
        // Weighted-blended OIT, HDR only: the accum/reveal targets do not
        // exist on the LDR path. Straight and premultiplied alpha share the
        // blend state and differ only in the fragment entry point.
        _ => {
            let (layout, shader, entry, label) = match i {
                OIT => (
                    &r.oit_layout,
                    r.oit_shader.get(),
                    "fs_oit",
                    "sprite_oit_pipeline",
                ),
                OIT_PREMULTIPLIED => (
                    &r.oit_layout,
                    r.oit_shader.get(),
                    "fs_oit_premultiplied",
                    "sprite_oit_pipeline_premultiplied",
                ),
                OIT_LIT => (
                    &r.oit_lit_layout,
                    r.oit_lit_shader.get(),
                    "fs_oit",
                    "sprite_lit_oit_pipeline",
                ),
                _ => (
                    &r.oit_lit_layout,
                    r.oit_lit_shader.get(),
                    "fs_oit_premultiplied",
                    "sprite_lit_oit_pipeline_premultiplied",
                ),
            };
            viewport_lib::plugin_api::builders::render_pipeline(
                &r.device,
                viewport_lib::plugin_api::builders::RenderPipelineDesc {
                    label,
                    layout,
                    vertex_module: shader,
                    vertex_entry: "vs_main",
                    vertex_buffers: &vertex_buffers,
                    fragment: Some(viewport_lib::gpu::FragmentState {
                        module: shader,
                        entry_point: Some(entry),
                        targets: &[
                            Some(viewport_lib::gpu::ColorTargetState {
                                format: viewport_lib::gpu::TextureFormat::Rgba16Float,
                                blend: Some(viewport_lib::plugin_api::target_desc::OIT_ACCUM_BLEND),
                                write_mask: viewport_lib::gpu::ColorWrites::ALL,
                            }),
                            Some(viewport_lib::gpu::ColorTargetState {
                                format: viewport_lib::gpu::TextureFormat::R8Unorm,
                                blend: Some(
                                    viewport_lib::plugin_api::target_desc::OIT_REVEAL_BLEND,
                                ),
                                write_mask: viewport_lib::gpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: viewport_lib::gpu::PipelineCompilationOptions::default(
                        ),
                    }),
                    primitive: viewport_lib::gpu::PrimitiveState {
                        topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(viewport_lib::plugin_api::builders::scene_depth_stencil(
                        r.depth_format,
                        false,
                        viewport_lib::plugin_api::depth::DEPTH_COMPARE,
                    )),
                    multisample: viewport_lib::gpu::MultisampleState {
                        count: r.sample_count,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        }
    }
}

/// A pick pipeline: [`PICK`], [`PICK_PAINTER`] or [`PICK_PAINTER_NO_TEST`]
/// by `member`.
fn pick_pipeline(
    device: &viewport_lib::gpu::Device,
    layout: &viewport_lib::gpu::PipelineLayout,
    module: &viewport_lib::gpu::ShaderModule,
    member: usize,
    depth_format: viewport_lib::gpu::TextureFormat,
) -> viewport_lib::gpu::RenderPipeline {
    let vertex_buffers = [position_layout()];
    let id_target = Some(viewport_lib::gpu::ColorTargetState {
        format: viewport_lib::gpu::TextureFormat::R32Uint,
        blend: None,
        write_mask: viewport_lib::gpu::ColorWrites::ALL,
    });
    viewport_lib::plugin_api::builders::render_pipeline(
        device,
        viewport_lib::plugin_api::builders::RenderPipelineDesc {
            label: match member {
                PICK => "sprite_pick_pipeline",
                PICK_PAINTER => "sprite_pick_painter_pipeline",
                _ => "sprite_pick_painter_no_test_pipeline",
            },
            layout,
            vertex_module: module,
            vertex_entry: "vs_main",
            vertex_buffers: &vertex_buffers,
            fragment: Some(viewport_lib::gpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                targets: &[id_target.clone(), id_target.clone(), id_target],
                compilation_options: viewport_lib::gpu::PipelineCompilationOptions::default(),
            }),
            primitive: viewport_lib::gpu::PrimitiveState {
                topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Painter picks test depth without writing it, so the
            // sequence's later draw wins wherever two overlap.
            depth_stencil: Some(viewport_lib::plugin_api::builders::scene_depth_stencil(
                depth_format,
                member == PICK,
                match member {
                    PICK => viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
                    PICK_PAINTER => viewport_lib::plugin_api::depth::DEPTH_COMPARE,
                    _ => viewport_lib::gpu::CompareFunction::Always,
                },
            )),
            multisample: viewport_lib::gpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            cache: None,
        },
    )
}

/// The selection-outline mask pipeline over `layout` and `module`.
fn outline_mask_pipeline(
    device: &viewport_lib::gpu::Device,
    layout: &viewport_lib::gpu::PipelineLayout,
    module: &viewport_lib::gpu::ShaderModule,
    depth_format: viewport_lib::gpu::TextureFormat,
) -> viewport_lib::gpu::RenderPipeline {
    viewport_lib::plugin_api::builders::build_outline_mask_pipeline(
        device,
        "sprite_outline_mask_pipeline",
        layout,
        module,
        viewport_lib::gpu::TextureFormat::R8Unorm,
        depth_format,
        &[position_layout()],
        None,
        false,
        viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
    )
}

/// Members of [`HookPassPipelines`]: the three picks share the indices of
/// [`SpritePassPipelines`], the outline mask and the surface mask follow them.
pub(super) const HOOK_OUTLINE_MASK: usize = 3;
pub(super) const HOOK_SURFACE_MASK: usize = 4;

/// The hook member a pass member draws through: `PICK` and its two painter
/// forms keep their pick, the outline mask moves to [`HOOK_OUTLINE_MASK`].
pub(super) fn hook_member(member: usize) -> usize {
    match member {
        PICK => 0,
        PICK_PAINTER => 1,
        PICK_PAINTER_NO_TEST => 2,
        _ => HOOK_OUTLINE_MASK,
    }
}

/// What a material with a vertex hook builds its pick, outline-mask and
/// surface-mask pipelines from: its composed shaders and the layouts with its
/// group 3.
pub(super) struct HookPassRecipe {
    device: viewport_lib::gpu::Device,
    pick_layout: viewport_lib::gpu::PipelineLayout,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    outline_layout: viewport_lib::gpu::PipelineLayout,
    outline_shader: viewport_lib::plugin_api::LazyModule,
    surface_layout: viewport_lib::gpu::PipelineLayout,
    surface_shader: viewport_lib::plugin_api::LazyModule,
    depth_format: viewport_lib::gpu::TextureFormat,
}

/// A vertex-hook material's three pick pipelines, its outline mask and its
/// surface mask, each built the first time a draw needs it.
pub(super) type HookPassPipelines = viewport_lib::plugin_api::LazyPipelines<HookPassRecipe, 5>;

fn build_hook_pass(r: &HookPassRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    match i {
        HOOK_SURFACE_MASK => viewport_lib::plugin_api::builders::build_surface_mask_pipeline(
            &r.device,
            "sprite_hook_surface_mask_pipeline",
            &r.surface_layout,
            r.surface_shader.get(),
            r.depth_format,
            &[position_layout()],
            None,
        ),
        HOOK_OUTLINE_MASK => outline_mask_pipeline(
            &r.device,
            &r.outline_layout,
            r.outline_shader.get(),
            r.depth_format,
        ),
        0 => pick_pipeline(
            &r.device,
            &r.pick_layout,
            r.pick_shader.get(),
            PICK,
            r.depth_format,
        ),
        1 => pick_pipeline(
            &r.device,
            &r.pick_layout,
            r.pick_shader.get(),
            PICK_PAINTER,
            r.depth_format,
        ),
        _ => pick_pipeline(
            &r.device,
            &r.pick_layout,
            r.pick_shader.get(),
            PICK_PAINTER_NO_TEST,
            r.depth_format,
        ),
    }
}

/// The sprite modules the plain pipelines build from, by shader name.
pub(super) const PLAIN_MODULES: [&str; 7] = [
    "sprite",
    "sprite_refraction",
    "sprite_lit",
    "sprite_oit",
    "sprite_lit_oit",
    "sprite_outline_mask",
    "sprite_pick",
];

/// The assembled source of plain module `name`, one of [`PLAIN_MODULES`].
pub(super) fn plain_source(name: &str) -> String {
    use crate::item_types::shader::{lit_shader, scene_shader, wgsl_source};
    let painter = viewport_lib::plugin_api::shared_wgsl::SHARED_PAINTER_MASK_WGSL;
    let lit_extra = [
        shared_wgsl::SHARED_CLIP_VOLUME_WGSL,
        shared_wgsl::SHARED_CSM_WGSL,
        shared_wgsl::SHARED_SHADOW_ATLAS_WGSL,
        painter,
        super::SPRITE_COMMON,
    ];
    let extra = [painter, super::SPRITE_COMMON];
    match name {
        "sprite" => scene_shader(&extra, wgsl_source!("sprite")),
        "sprite_refraction" => scene_shader(&extra, wgsl_source!("sprite_refraction")),
        "sprite_lit" => lit_shader(&lit_extra, wgsl_source!("sprite_lit")),
        "sprite_oit" => scene_shader(&extra, wgsl_source!("sprite_oit")),
        "sprite_lit_oit" => lit_shader(&lit_extra, wgsl_source!("sprite_lit_oit")),
        "sprite_outline_mask" => scene_shader(&extra, wgsl_source!("sprite_outline_mask")),
        "sprite_pick" => scene_shader(&extra, wgsl_source!("sprite_pick")),
        _ => unreachable!("no plain sprite module named {name}"),
    }
}

/// Every pipeline and layout the sprite draw hooks need, made on the first
/// prepare that sees an item. The pipelines themselves are built the first
/// time a draw needs them.
pub(super) struct SpriteGpu {
    pub(super) pipelines: SpritePipelines,
    /// The effect families, by family index: the built-in effects first,
    /// then one per registered sprite material, each built the first frame
    /// one of its batches is prepared.
    pub(super) fx_families: Vec<Option<SpritePipelines>>,
    /// What an effect family's build reads beyond its shaders and layout.
    fx_base: FxFamilyBase,
    pub(super) passes: SpritePassPipelines,
    /// The pick, outline-mask and surface-mask pipelines of each registered
    /// material with a vertex hook, by material index.
    pub(super) hook_passes: Vec<Option<HookPassPipelines>>,
    /// An empty group, bound at group 2 of a hook outline mask, whose layout
    /// keeps the material's group 3 in its place.
    pub(super) empty_bg: viewport_lib::gpu::BindGroup,
    pub(super) refraction_bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) refraction_sampler: viewport_lib::gpu::Sampler,
    pub(super) pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) soft_fallback_bg: viewport_lib::gpu::BindGroup,
    /// 1x1 Depth32Float texture backing the soft fallback bind group. Held so
    /// the bind group keeps a valid texture; not read after construction.
    #[allow(dead_code)]
    pub(super) soft_fallback_tex: viewport_lib::gpu::Texture,
    pub(super) lit_fallback_bg: viewport_lib::gpu::BindGroup,
}

/// The pieces every effect family shares.
struct FxFamilyBase {
    group0: viewport_lib::gpu::BindGroupLayout,
    bgl: viewport_lib::gpu::BindGroupLayout,
    soft_bgl: viewport_lib::gpu::BindGroupLayout,
    fx_bgl: viewport_lib::gpu::BindGroupLayout,
    pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
    empty_bgl: viewport_lib::gpu::BindGroupLayout,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
    depth_format: viewport_lib::gpu::TextureFormat,
    /// The registered sprite deformers, composed into every module.
    deformers: Vec<super::deform::SpriteDeformerEntry>,
}

/// Build the pick and outline-mask pipelines of `material`, which has a
/// vertex hook.
fn hook_passes(
    device: &viewport_lib::gpu::Device,
    resources: &DeviceResources,
    base: &FxFamilyBase,
    material: &super::effects::SpriteMaterialEntry,
) -> HookPassPipelines {
    let (pick, outline) = super::effects::hook_base_sources();
    let (unlit, _) = super::effects::base_sources();
    let base_deformers = &base.deformers;
    // Checked when the material was registered, so it cannot fail here.
    let compose = |base: &str| {
        super::effects::compose_hook(base, &material.source())
            .and_then(|s| super::deform::sprite_module(s, base_deformers))
            .expect("sprite hook shader composes")
    };
    let name = material.name.as_str();
    let pick_shader = resources.lazy_module(device, &format!("{name}_pick_shader"), compose(&pick));
    let outline_shader =
        resources.lazy_module(device, &format!("{name}_outline_shader"), compose(&outline));
    let surface_shader =
        resources.lazy_module(device, &format!("{name}_surface_shader"), compose(&unlit));
    resources.lazy_pipelines(
        HookPassRecipe {
            device: device.clone(),
            pick_layout: viewport_lib::plugin_api::builders::pipeline_layout(
                device,
                "sprite_hook_pick_pipeline_layout",
                &[&base.group0, &base.bgl, &base.pick_id_bgl, &base.fx_bgl],
            ),
            pick_shader,
            outline_layout: viewport_lib::plugin_api::builders::pipeline_layout(
                device,
                "sprite_hook_outline_pipeline_layout",
                &[&base.group0, &base.bgl, &base.empty_bgl, &base.fx_bgl],
            ),
            outline_shader,
            surface_layout: viewport_lib::plugin_api::builders::pipeline_layout(
                device,
                "sprite_hook_surface_pipeline_layout",
                &[&base.group0, &base.bgl, &base.soft_bgl, &base.fx_bgl],
            ),
            surface_shader,
            depth_format: base.depth_format,
        },
        build_hook_pass,
    )
}

/// Build the colour pipelines of one effect family: the built-in effects, or
/// the built-in effects with `material`.
fn fx_family(
    device: &viewport_lib::gpu::Device,
    resources: &DeviceResources,
    base: &FxFamilyBase,
    material: Option<&super::effects::SpriteMaterialEntry>,
) -> SpritePipelines {
    let (unlit, lit) = super::effects::base_sources();
    let source = material.map(|m| m.source());
    // Composition of the built-in family is checked by the unit tests and a
    // material's at registration, so it cannot fail here.
    let compose = |src: &str, lit: bool| {
        super::effects::compose(src, lit, source.as_ref())
            .and_then(|s| super::deform::sprite_module(s, &base.deformers))
            .expect("sprite effect shader composes")
    };
    let name = material.map_or("sprite_fx", |m| m.name.as_str());
    let shader = resources.lazy_module(device, &format!("{name}_shader"), compose(&unlit, false));
    let lit_shader =
        resources.lazy_module(device, &format!("{name}_lit_shader"), compose(&lit, true));
    let layout = viewport_lib::plugin_api::builders::pipeline_layout(
        device,
        "sprite_fx_pipeline_layout",
        &[&base.group0, &base.bgl, &base.soft_bgl, &base.fx_bgl],
    );
    resources.lazy_pipelines(
        SpriteRecipe {
            device: device.clone(),
            layout: layout.clone(),
            shader,
            lit_layout: layout,
            lit_shader,
            sample_count: base.sample_count,
            ldr_format: base.ldr_format,
            depth_format: base.depth_format,
            fx: true,
        },
        build_sprite,
    )
}

impl SpriteGpu {
    /// The colour pipelines of effect family `fx`, or the plain ones for
    /// `None`. `None` too for a family not built yet.
    pub(super) fn colour_pipelines(&self, fx: Option<u32>) -> Option<&SpritePipelines> {
        match fx {
            None => Some(&self.pipelines),
            Some(f) => self.fx_families.get(f as usize)?.as_ref(),
        }
    }

    /// Build the family of every registered material that has none yet, and
    /// the pick and outline-mask pipelines of each with a vertex hook.
    pub(super) fn ensure_material_families(
        &mut self,
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        materials: &[super::effects::SpriteMaterialEntry],
    ) {
        while self.fx_families.len() < materials.len() + 1 {
            self.fx_families.push(None);
        }
        while self.hook_passes.len() < materials.len() {
            self.hook_passes.push(None);
        }
        for (i, m) in materials.iter().enumerate() {
            if self.fx_families[i + 1].is_none() {
                self.fx_families[i + 1] =
                    Some(fx_family(device, resources, &self.fx_base, Some(m)));
            }
            if m.displaces && self.hook_passes[i].is_none() {
                self.hook_passes[i] = Some(hook_passes(device, resources, &self.fx_base, m));
            }
        }
    }

    /// The hook pick and outline-mask pipelines `sprite` draws those passes
    /// through, when its material has a vertex hook.
    pub(super) fn hook_passes_of<'a>(
        &'a self,
        sprite: &'a super::store::SpriteGpuData,
    ) -> Option<(&'a HookPassPipelines, &'a viewport_lib::gpu::BindGroup)> {
        let fx = sprite.fx.as_ref()?;
        let material = fx.family.checked_sub(1)? as usize;
        let passes = self.hook_passes.get(material)?.as_ref()?;
        Some((passes, &fx.bind_group))
    }

    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::SpriteLayouts,
        deformers: &[super::deform::SpriteDeformerEntry],
    ) -> Self {
        // Every module with the registered deformers composed in, checked
        // when each registered.
        let deformed = |source: String| {
            super::deform::sprite_module(source, deformers).expect("sprite deformers compose")
        };
        let bgl = &layouts.bgl;
        let lit_bgl = &layouts.lit_bgl;
        let group0 = resources.group0_layout();

        // Group 2: scene depth + sampler for the soft-particle fade, then the
        // painter masks. The shader skips the depth sample unless
        // soft_particle_distance > 0, so a pass that cannot expose live depth
        // binds the 1x1 fallback below instead.
        // This is the lib's shared depth-read layout rather than one of our
        // own: the entries are identical, and matching it means the ready-made
        // `DepthReadContext::scene_depth_bind_group` binds here directly.
        let soft_bgl = resources.depth_read_bind_group_layout();

        let fallback_tex = device.create_texture(&viewport_lib::gpu::TextureDescriptor {
            label: Some("sprite_soft_fallback_tex"),
            size: viewport_lib::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: viewport_lib::gpu::TextureDimension::D2,
            format: viewport_lib::gpu::TextureFormat::Depth32Float,
            usage: viewport_lib::gpu::TextureUsages::TEXTURE_BINDING
                | viewport_lib::gpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let fallback_view = fallback_tex.create_view(&viewport_lib::gpu::TextureViewDescriptor {
            aspect: viewport_lib::gpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        // The painter masks the layout carries after the depth: an empty 1x1
        // mask and a uniform naming no slot, so a pass with no masks to test
        // keeps every fragment.
        let mask_fallback = device.create_texture(&viewport_lib::gpu::TextureDescriptor {
            label: Some("sprite_mask_fallback_tex"),
            size: viewport_lib::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: viewport_lib::gpu::TextureDimension::D2,
            format: viewport_lib::resources::PAINTER_MASK_FORMAT,
            usage: viewport_lib::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mask_fallback_view =
            mask_fallback.create_view(&viewport_lib::gpu::TextureViewDescriptor::default());
        let no_masks = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("sprite_no_masks_buf"),
            size: 64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        // Then the light layers: a 1x1 array (two array layers, as GL needs
        // to view it as an array) and a uniform naming no range, so nothing
        // is lit.
        let light_fallback = device.create_texture(&viewport_lib::gpu::TextureDescriptor {
            label: Some("sprite_light_fallback_tex"),
            size: viewport_lib::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: viewport_lib::gpu::TextureDimension::D2,
            format: viewport_lib::gpu::TextureFormat::Rgba8Unorm,
            usage: viewport_lib::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let light_fallback_view =
            light_fallback.create_view(&viewport_lib::gpu::TextureViewDescriptor {
                dimension: Some(viewport_lib::gpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
        let no_lights = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("sprite_no_lights_buf"),
            size: viewport_lib::resources::PAINTER_LIGHTS_UNIFORM_BYTES,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let fallback_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("sprite_soft_fallback_bg"),
            layout: soft_bgl,
            entries: &[
                viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: viewport_lib::gpu::BindingResource::TextureView(&fallback_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_lib::gpu::BindingResource::Sampler(
                        resources.depth_read_sampler(),
                    ),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 2,
                    resource: viewport_lib::gpu::BindingResource::TextureView(&mask_fallback_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 3,
                    resource: no_masks.as_entire_binding(),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 4,
                    resource: viewport_lib::gpu::BindingResource::TextureView(&light_fallback_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 5,
                    resource: no_lights.as_entire_binding(),
                },
            ],
        });

        let shader =
            resources.lazy_module(device, "sprite_shader", deformed(plain_source("sprite")));
        let layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_pipeline_layout",
            &[group0, bgl, soft_bgl],
        );

        // Refractive sprites: group 2 is the scene-colour copy and its sampler.
        let refraction_bgl = viewport_lib::plugin_api::builders::texture_sampler_bgl(
            device,
            "sprite_refraction_bgl",
            viewport_lib::gpu::ShaderStages::FRAGMENT,
        );
        let refraction_sampler = viewport_lib::plugin_api::builders::clamp_linear_sampler(
            device,
            "sprite_refraction_sampler",
        );
        let refraction_shader = resources.lazy_module(
            device,
            "sprite_refraction_shader",
            deformed(plain_source("sprite_refraction")),
        );
        let refraction_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_refraction_pipeline_layout",
            &[group0, bgl, &refraction_bgl],
        );

        // Lit sprites. Group 0 already carries the lights; group 2 is the
        // soft-particle depth, as for the unlit path, and group 3 the optional
        // normal map.
        let lit_shader = resources.lazy_module(
            device,
            "sprite_lit_shader",
            deformed(plain_source("sprite_lit")),
        );
        let lit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_lit_pipeline_layout",
            &[group0, bgl, soft_bgl, lit_bgl],
        );

        // The fallback bind group reuses the crate-wide `fallback_normal_map`,
        // already populated with `(128, 128, 255, 255)` for tangent-space `(0, 0, 1)`.
        let lit_fallback_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("sprite_lit_fallback_bg"),
            layout: lit_bgl,
            entries: &[
                viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: viewport_lib::gpu::BindingResource::TextureView(
                        resources.fallback_texture_view(viewport_lib::TextureSlot::Normal),
                    ),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_lib::gpu::BindingResource::Sampler(
                        resources.material_sampler(),
                    ),
                },
            ],
        });

        // OIT. Only ever selected for `AlphaBlend`/`Premultiplied` sprites
        // with `depth_write: false` and no active soft-particle fade (see
        // `SpriteGpuData::oit_eligible`); everything else keeps drawing through
        // the colour variants. Each OIT shader exposes `fs_oit` and
        // `fs_oit_premultiplied` from one module.
        let oit_shader = resources.lazy_module(
            device,
            "sprite_oit_shader",
            deformed(plain_source("sprite_oit")),
        );
        let oit_lit_shader = resources.lazy_module(
            device,
            "sprite_lit_oit_shader",
            deformed(plain_source("sprite_lit_oit")),
        );
        let oit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_oit_pipeline_layout",
            &[group0, bgl],
        );
        let oit_lit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_lit_oit_pipeline_layout",
            &[group0, bgl, lit_bgl],
        );

        let mask_shader = resources.lazy_module(
            device,
            "sprite_outline_mask_shader",
            deformed(plain_source("sprite_outline_mask")),
        );
        let mask_layout = viewport_lib::plugin_api::builders::standard_scene_layout(
            device,
            "sprite_outline_mask_pipeline_layout",
            group0,
            bgl,
        );

        let pick_id_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("sprite_pick_id_bgl"),
                entries: &[viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Buffer {
                        ty: viewport_lib::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pick_shader = resources.lazy_module(
            device,
            "sprite_pick_shader",
            deformed(plain_source("sprite_pick")),
        );
        let pick_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_pick_pipeline_layout",
            &[group0, bgl, &pick_id_bgl],
        );

        let sample_count = resources.sample_count();

        // The (depth x blend x lit) keys in each pass. Sprites are billboards
        // drawn with a `Less` depth test (or none, for a painter batch that
        // asks for none) and no culling; each is built the first time a draw
        // selects it.
        let pipelines: SpritePipelines = resources.lazy_pipelines(
            SpriteRecipe {
                device: device.clone(),
                layout: layout.clone(),
                shader: shader.clone(),
                lit_layout,
                lit_shader,
                sample_count,
                ldr_format: resources.target_format(),
                depth_format: resources.scene_depth_format(),
                fx: false,
            },
            build_sprite,
        );
        let empty_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("sprite_empty_bgl"),
                entries: &[],
            });
        let empty_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("sprite_empty_bg"),
            layout: &empty_bgl,
            entries: &[],
        });
        let fx_base = FxFamilyBase {
            group0: group0.clone(),
            bgl: bgl.clone(),
            soft_bgl: soft_bgl.clone(),
            fx_bgl: layouts.fx_bgl.clone(),
            pick_id_bgl: pick_id_bgl.clone(),
            empty_bgl,
            sample_count,
            ldr_format: resources.target_format(),
            depth_format: resources.scene_depth_format(),
            deformers: deformers.to_vec(),
        };
        let builtin = fx_family(device, resources, &fx_base, None);
        let passes: SpritePassPipelines = resources.lazy_pipelines(
            SpritePassRecipe {
                device: device.clone(),
                layout,
                shader,
                refraction_layout,
                refraction_shader,
                oit_layout,
                oit_shader,
                oit_lit_layout,
                oit_lit_shader,
                mask_layout,
                mask_shader,
                pick_layout,
                pick_shader,
                sample_count,
                depth_format: resources.scene_depth_format(),
            },
            build_pass,
        );

        Self {
            pipelines,
            fx_families: vec![Some(builtin)],
            fx_base,
            passes,
            hook_passes: Vec::new(),
            empty_bg,
            refraction_bgl,
            refraction_sampler,
            pick_id_bgl,
            soft_fallback_bg: fallback_bg,
            soft_fallback_tex: fallback_tex,
            lit_fallback_bg,
        }
    }

    /// Whether the pipeline `sprite` draws with can draw this frame, in
    /// either format. The outline, surface mask and pick passes wait for it,
    /// so a batch is never outlined, stamped or picked before it is drawn.
    pub(super) fn drawn(&self, sprite: &super::store::SpriteGpuData) -> bool {
        let key = SpriteKey::of(sprite);
        let colour = self.colour_pipelines(key.fx).is_some_and(|family| {
            SpritePass::ALL
                .into_iter()
                .any(|pass| family.available(sprite_index(key, pass)))
        });
        if sprite.refraction_strength > 0.0 {
            self.passes.available(REFRACTION)
        } else if sprite.oit_eligible {
            self.passes.available(oit_index(sprite.lit, sprite.blend)) || colour
        } else {
            colour
        }
    }
}

impl SpriteGpu {
    /// Group-2 bind group carrying one batch's object pick id, for the pick pass.
    /// `alpha_threshold` discards pick fragments whose alpha falls below it;
    /// `None` picks the whole quad.
    pub(super) fn pick_bind_group(
        &self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        pick_id: viewport_lib::renderer::PickId,
        alpha_threshold: Option<f32>,
    ) -> viewport_lib::gpu::BindGroup {
        // The id, the threshold (negative for none) and padding to the
        // uniform's 16-byte size, matching the layout the pick shader declares.
        let threshold = alpha_threshold.map_or(-1.0, |t| t.max(0.0));
        let id_data = [pick_id.0 as u32, threshold.to_bits(), 0u32, 0u32];
        let buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("sprite_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("sprite_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[viewport_lib::gpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A draw reads a variant by `SpriteKey::slot()` and the build decodes the
    /// key back with `SpriteKey::from_slot()`. The two have to agree, or every
    /// variant is built for the wrong key.
    ///
    /// The visible symptom was a validation failure rather than a wrong-looking
    /// sprite: an unlit Additive batch resolved to an AlphaBlend *lit*
    /// pipeline, and the unlit draw path does not bind the group-3 normal map
    /// that pipeline's layout requires.
    #[test]
    fn every_key_resolves_to_the_variant_built_for_it() {
        for key in SpriteKey::all() {
            assert_eq!(
                SpriteKey::from_slot(key.slot()),
                key,
                "slot {} builds the variant for a different key",
                key.slot()
            );
        }
    }

    /// `slot()` has to be a bijection onto `0..KEY_COUNT`, or two keys
    /// collide and a third slot is never written.
    #[test]
    fn slots_are_dense_and_unique() {
        let mut seen = [false; KEY_COUNT];
        let mut count = 0;
        for key in SpriteKey::all() {
            let slot = key.slot();
            assert!(!seen[slot], "slot {slot} claimed twice");
            seen[slot] = true;
            count += 1;
        }
        assert_eq!(count, KEY_COUNT, "all() must yield exactly KEY_COUNT keys");
        assert!(seen.iter().all(|s| *s), "every slot must be claimed");
    }
}
