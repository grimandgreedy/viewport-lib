//! Render-target descriptors published for plugin pipeline construction.
//!
//! Each descriptor captures the colour / depth formats, blend state, and
//! sample count the lib's corresponding pass uses. A pipeline built against
//! one of these descriptors is compatible with that pass.

/// Targets used by the main HDR opaque scene pass.
///
/// Plugins use this to build opaque (and instanced) pipelines that draw into
/// the HDR scene buffer alongside built-in mesh items.
///
/// The HDR scene pass has two colour attachments: the scene colour at
/// `@location(0)` and the indirect fraction at `@location(1)`. A pipeline
/// drawn into it declares both, in that order; [`colour_targets`](Self::colour_targets)
/// returns the list for either pass.
#[derive(Clone, Copy, Debug)]
pub struct OpaqueTargetDesc {
    /// Colour target format. `Rgba16Float` from
    /// [`opaque_target_desc`](crate::resources::DeviceResources::opaque_target_desc)
    /// (the HDR scene buffer); the renderer's configured output format from
    /// [`ldr_opaque_target_desc`](crate::resources::DeviceResources::ldr_opaque_target_desc)
    /// (the LDR scene pass).
    pub color_format: crate::gpu::TextureFormat,
    /// Format of the second colour target, the indirect fraction
    /// ([`INDIRECT_FRACTION_FORMAT`](crate::resources::INDIRECT_FRACTION_FORMAT)),
    /// on the HDR scene pass. `None` on the LDR scene pass, which has one
    /// colour attachment.
    pub indirect_fraction_format: Option<crate::gpu::TextureFormat>,
    /// Depth-stencil format: the renderer's
    /// [`scene_depth_format`](crate::resources::DeviceResources::scene_depth_format).
    pub depth_format: crate::gpu::TextureFormat,
    /// MSAA sample count. Matches the renderer's configured `sample_count`.
    pub sample_count: u32,
}

impl OpaqueTargetDesc {
    /// The colour targets of this pass, in attachment order: the scene colour
    /// with `blend`, then on the HDR pass the indirect fraction (see
    /// [`indirect_fraction_target`]). `writes_indirect_fraction` says whether
    /// the fragment entry writes `@location(1)`.
    pub fn colour_targets(
        &self,
        blend: Option<crate::gpu::BlendState>,
        writes_indirect_fraction: bool,
    ) -> SceneColourTargets {
        let colour = crate::gpu::ColorTargetState {
            format: self.color_format,
            blend,
            write_mask: crate::gpu::ColorWrites::ALL,
        };
        match self.indirect_fraction_format {
            Some(_) => SceneColourTargets::hdr(colour, writes_indirect_fraction),
            None => SceneColourTargets::single(colour),
        }
    }
}

/// The colour target list of a scene-pass pipeline: one target on the LDR
/// pass, two on the HDR pass. Pass [`as_slice`](Self::as_slice) as the
/// fragment state's `targets`.
#[derive(Clone, Debug)]
pub struct SceneColourTargets {
    targets: [Option<crate::gpu::ColorTargetState>; 2],
    len: usize,
}

impl SceneColourTargets {
    /// The HDR scene pass's two targets: `colour`, then the indirect
    /// fraction, written when `writes_indirect_fraction`.
    pub fn hdr(colour: crate::gpu::ColorTargetState, writes_indirect_fraction: bool) -> Self {
        let fraction = indirect_fraction_target(colour.blend, writes_indirect_fraction);
        Self {
            targets: [Some(colour), Some(fraction)],
            len: 2,
        }
    }

    /// A single colour target, for a pass with one attachment.
    pub fn single(colour: crate::gpu::ColorTargetState) -> Self {
        Self {
            targets: [Some(colour), None],
            len: 1,
        }
    }

    /// The targets, for `FragmentState::targets`.
    pub fn as_slice(&self) -> &[Option<crate::gpu::ColorTargetState>] {
        &self.targets[..self.len]
    }
}

/// The HDR scene pass's `@location(1)` target, the indirect fraction: the
/// indirect share of the pixel's light in red, which ambient occlusion
/// scales, and the direct share in green, which contact shadows scale.
///
/// A fragment entry that writes it outputs `vec4(indirect, direct, 0, alpha)`
/// with the colour target's alpha (see `scene_fragment` in
/// [`SHARED_SCENE_FRAGMENT_WGSL`](crate::plugin_api::shared_wgsl::SHARED_SCENE_FRAGMENT_WGSL)),
/// and the target takes the colour target's `blend`, so an alpha-blended draw
/// mixes the fraction the way it mixes the colour. A draw that does not write
/// it gets an empty write mask and leaves the pixel's value as it was: 1 where
/// nothing else drew, so both terms darken it in full. Only an opaque
/// (`blend: None`) or [`ALPHA_BLENDING`](crate::gpu::BlendState::ALPHA_BLENDING)
/// draw writes it; under any other blend (additive, premultiplied) the
/// fraction would not mix as a weight, so the mask is empty whatever
/// `writes` says.
pub fn indirect_fraction_target(
    blend: Option<crate::gpu::BlendState>,
    writes: bool,
) -> crate::gpu::ColorTargetState {
    let writes = writes
        && match blend {
            None => true,
            Some(b) => b == crate::gpu::BlendState::ALPHA_BLENDING,
        };
    crate::gpu::ColorTargetState {
        format: crate::resources::INDIRECT_FRACTION_FORMAT,
        blend: if writes { blend } else { None },
        write_mask: if writes {
            crate::gpu::ColorWrites::ALL
        } else {
            crate::gpu::ColorWrites::empty()
        },
    }
}

/// Targets used by the foreground pass.
///
/// The foreground pass loads the scene colour after the SSAA resolve and
/// draws against its own cleared depth target, so foreground plugin items
/// composite over the finished world. Single-sampled. Like the scene pass it
/// has the indirect fraction as a second colour target; the composite ignores
/// it where the foreground covers, so a foreground pipeline need not write it.
#[derive(Clone, Copy, Debug)]
pub struct ForegroundTargetDesc {
    /// Colour target format. `Rgba16Float` (the HDR scene buffer).
    pub color_format: crate::gpu::TextureFormat,
    /// Format of the second colour target, the indirect fraction
    /// ([`INDIRECT_FRACTION_FORMAT`](crate::resources::INDIRECT_FRACTION_FORMAT)).
    pub indirect_fraction_format: crate::gpu::TextureFormat,
    /// Depth-stencil format of the pass's own cleared depth target.
    /// The renderer's
    /// [`scene_depth_format`](crate::resources::DeviceResources::scene_depth_format).
    pub depth_format: crate::gpu::TextureFormat,
    /// Sample count. The foreground pass runs after the SSAA resolve, so 1.
    pub sample_count: u32,
}

impl ForegroundTargetDesc {
    /// The pass's two colour targets: the scene colour with `blend`, then the
    /// indirect fraction (see [`indirect_fraction_target`]).
    pub fn colour_targets(
        &self,
        blend: Option<crate::gpu::BlendState>,
        writes_indirect_fraction: bool,
    ) -> SceneColourTargets {
        SceneColourTargets::hdr(
            crate::gpu::ColorTargetState {
                format: self.color_format,
                blend,
                write_mask: crate::gpu::ColorWrites::ALL,
            },
            writes_indirect_fraction,
        )
    }
}

/// Targets used by the order-independent transparency (OIT) pass.
///
/// Weighted-blended OIT writes to two MRT attachments: an additive accum
/// target and a multiplicative reveal target. A plugin transparent pipeline
/// must declare both targets in the same order and use the published blend
/// states, or the composite will reject the output.
#[derive(Clone, Copy, Debug)]
pub struct OitTargetDesc {
    /// `@location(0)` accumulation target. `Rgba16Float` with additive blend.
    pub accum_format: crate::gpu::TextureFormat,
    /// `@location(1)` reveal target. `R8Unorm` with `(0, OneMinusSrc)` blend.
    pub reveal_format: crate::gpu::TextureFormat,
    /// Shared depth attachment (read-only `LessEqual` test, no depth write).
    pub depth_format: crate::gpu::TextureFormat,
    /// Blend state for the accum target.
    pub accum_blend: crate::gpu::BlendState,
    /// Blend state for the reveal target.
    pub reveal_blend: crate::gpu::BlendState,
    /// MSAA sample count. OIT runs at the scene sample count.
    pub sample_count: u32,
}

/// Targets used by the read-only-depth plugin pass.
///
/// The pass runs after the opaque scene (and the built-in sprite passes) and
/// before OIT, with the scene depth attachment bound read-only so the plugin's
/// fragment shader can sample it. A pipeline built against this descriptor
/// draws one colour target over the finished opaque scene colour and depth-tests
/// against the opaque depth without writing it.
#[derive(Clone, Copy, Debug)]
pub struct DepthReadTargetDesc {
    /// Colour target format. `Rgba16Float`: the read-only-depth pass exists
    /// on the HDR path only.
    pub color_format: crate::gpu::TextureFormat,
    /// Depth-stencil attachment, bound read-only (`depth_ops: None`). The
    /// pipeline tests against it but must not write. The renderer's
    /// [`scene_depth_format`](crate::resources::DeviceResources::scene_depth_format).
    pub depth_format: crate::gpu::TextureFormat,
    /// MSAA sample count. The pass runs at the scene sample count (the SSAA
    /// target on the HDR path), matching the opaque depth it samples.
    pub sample_count: u32,
}

/// Targets used by the outline mask pass (stage 1 of the selection outline).
///
/// Selected items rasterise a single value (1.0) into an `R8Unorm` mask. The
/// composite pass reads the mask and writes the outline colour.
#[derive(Clone, Copy, Debug)]
pub struct MaskTargetDesc {
    /// `R8Unorm` mask target.
    pub color_format: crate::gpu::TextureFormat,
    /// Depth attachment. The mask pass uses the scene depth buffer for
    /// correct occlusion, so the format matches the scene depth.
    pub depth_format: crate::gpu::TextureFormat,
    /// MSAA sample count. Mask runs at sample count 1 (the mask is a 1x
    /// auxiliary target).
    pub sample_count: u32,
}

/// Targets used by the pick-id pass.
///
/// The pass has three colour attachments plus a depth-stencil attachment. A
/// pipeline drawn into it must declare all three colour targets in order, even
/// if it only writes the object id: a fragment shader whose output struct omits
/// a location still leaves that attachment at its clear value. Use
/// [`SHARED_PICK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_WGSL), which
/// writes all three.
///
/// The renderer reads back the object-id pixel (and, for sub-object picking, the
/// primitive-id pixel) under the cursor to resolve the hit.
#[derive(Clone, Copy, Debug)]
pub struct PickTargetDesc {
    /// `@location(0)`: `R32Uint` object id (the item's `PickId`).
    pub object_id_format: crate::gpu::TextureFormat,
    /// `@location(1)`: `R32Uint` sub-object primitive index. Write 0 unless the
    /// pipeline resolves sub-objects.
    pub primitive_id_format: crate::gpu::TextureFormat,
    /// `@location(2)`: `R32Float` depth channel, written as the fragment's
    /// framebuffer-space `z` so the renderer can reconstruct world position.
    pub depth_channel_format: crate::gpu::TextureFormat,
    /// Depth-stencil attachment used for occlusion within the pass.
    pub depth_stencil_format: crate::gpu::TextureFormat,
    /// MSAA sample count. Pick runs at sample count 1.
    pub sample_count: u32,
}

/// Targets used by the shadow pass.
///
/// Depth-only output into the shadow atlas. The pass loops cascades by
/// `set_viewport` into one texture; pipelines declare no colour target.
#[derive(Clone, Copy, Debug)]
pub struct ShadowTargetDesc {
    /// `Depth32Float` shadow atlas format.
    pub depth_format: crate::gpu::TextureFormat,
    /// MSAA sample count. Shadow runs at sample count 1.
    pub sample_count: u32,
}

/// OIT accum-target blend state: additive `(One, One)` on color and alpha.
pub const OIT_ACCUM_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
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

/// OIT reveal-target blend state: `(Zero, OneMinusSrc)`.
pub const OIT_REVEAL_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
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
