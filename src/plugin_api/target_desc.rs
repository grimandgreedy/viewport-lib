//! Render-target descriptors published for plugin pipeline construction.
//!
//! Each descriptor captures the colour / depth formats, blend state, and
//! sample count the lib's corresponding pass uses. A pipeline built against
//! one of these descriptors is compatible with that pass.

/// Targets used by the main HDR opaque scene pass.
///
/// Plugins use this to build opaque (and instanced) pipelines that draw into
/// the HDR scene buffer alongside built-in mesh items.
#[derive(Clone, Copy, Debug)]
pub struct OpaqueTargetDesc {
    /// Colour target format. `Rgba16Float` on the HDR path; matches the
    /// swap-chain format on the LDR path.
    pub color_format: wgpu::TextureFormat,
    /// Depth-stencil format. `Depth24PlusStencil8` everywhere today.
    pub depth_format: wgpu::TextureFormat,
    /// MSAA sample count. Matches the renderer's configured `sample_count`.
    pub sample_count: u32,
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
    pub accum_format: wgpu::TextureFormat,
    /// `@location(1)` reveal target. `R8Unorm` with `(0, OneMinusSrc)` blend.
    pub reveal_format: wgpu::TextureFormat,
    /// Shared depth attachment (read-only `LessEqual` test, no depth write).
    pub depth_format: wgpu::TextureFormat,
    /// Blend state for the accum target.
    pub accum_blend: wgpu::BlendState,
    /// Blend state for the reveal target.
    pub reveal_blend: wgpu::BlendState,
    /// MSAA sample count. OIT runs at the scene sample count.
    pub sample_count: u32,
}

/// Targets used by the outline mask pass (stage 1 of the selection outline).
///
/// Selected items rasterise a single value (1.0) into an `R8Unorm` mask. The
/// composite pass reads the mask and writes the outline colour.
#[derive(Clone, Copy, Debug)]
pub struct MaskTargetDesc {
    /// `R8Unorm` mask target.
    pub color_format: wgpu::TextureFormat,
    /// Depth attachment. The mask pass uses the scene depth buffer for
    /// correct occlusion, so the format matches the scene depth.
    pub depth_format: wgpu::TextureFormat,
    /// MSAA sample count. Mask runs at sample count 1 (the mask is a 1x
    /// auxiliary target).
    pub sample_count: u32,
}

/// Targets used by the pick-id pass.
///
/// Items rasterise their assigned `PickId` value into an `R32Uint` target;
/// the renderer reads back the pixel under the cursor to resolve picks.
#[derive(Clone, Copy, Debug)]
pub struct PickTargetDesc {
    /// `R32Uint` pick target.
    pub color_format: wgpu::TextureFormat,
    /// Depth attachment.
    pub depth_format: wgpu::TextureFormat,
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
    pub depth_format: wgpu::TextureFormat,
    /// MSAA sample count. Shadow runs at sample count 1.
    pub sample_count: u32,
}

/// OIT accum-target blend state: additive `(One, One)` on color and alpha.
pub const OIT_ACCUM_BLEND: wgpu::BlendState = wgpu::BlendState {
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

/// OIT reveal-target blend state: `(Zero, OneMinusSrc)`.
pub const OIT_REVEAL_BLEND: wgpu::BlendState = wgpu::BlendState {
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
