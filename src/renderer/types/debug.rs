// Debug visualization types for the shadow and lighting pipeline.

/// Selects which intermediate quantity to display in a debug channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DebugQuantity {
    /// Constant black (0.0).
    #[default]
    Zero,
    /// Constant white (1.0).
    One,
    /// Cascade index as a hue (0=red, 1=green, 2=blue, 3=yellow).
    CascadeIndex,
    /// Combined shadow factor [0=fully shadowed, 1=fully lit].
    ShadowFactor,
    /// Contact shadow contribution in isolation.
    ContactShadowFactor,
    /// N dot L (clamped 0..1).
    NdotL,
    /// Normal bias magnitude in world units.
    NormalBiasMagnitude,
    /// Atlas UV X coordinate.
    AtlasUvX,
    /// Atlas UV Y coordinate.
    AtlasUvY,
    /// Tile UV X before atlas remapping.
    TileUvX,
    /// Tile UV Y before atlas remapping.
    TileUvY,
    /// Biased depth used in the shadow comparison.
    BiasedDepth,
    /// Unbiased surface depth from the shadow projection.
    SurfaceDepth,
    /// World-space normal X.
    WorldNormalX,
    /// World-space normal Y.
    WorldNormalY,
    /// World-space normal Z.
    WorldNormalZ,
    /// PBR roughness value.
    Roughness,
    /// PBR metallic value.
    Metallic,
    /// Ambient occlusion factor.
    AoFactor,
    /// Direct light luminance (all lights, no ambient).
    DirectLightLuminance,
    /// Hemisphere or IBL ambient luminance.
    AmbientLuminance,
    /// IBL diffuse luminance only.
    IblDiffuseLuminance,
    /// IBL specular luminance only.
    IblSpecularLuminance,
    /// Emissive luminance only.
    EmissiveLuminance,
}

impl DebugQuantity {
    /// Returns the packed u32 index for this quantity (used in the shader).
    pub fn index(self) -> u32 {
        self as u32
    }

    /// Returns all defined variants in enum order.
    pub fn all_variants() -> &'static [DebugQuantity] {
        &[
            DebugQuantity::Zero,
            DebugQuantity::One,
            DebugQuantity::CascadeIndex,
            DebugQuantity::ShadowFactor,
            DebugQuantity::ContactShadowFactor,
            DebugQuantity::NdotL,
            DebugQuantity::NormalBiasMagnitude,
            DebugQuantity::AtlasUvX,
            DebugQuantity::AtlasUvY,
            DebugQuantity::TileUvX,
            DebugQuantity::TileUvY,
            DebugQuantity::BiasedDepth,
            DebugQuantity::SurfaceDepth,
            DebugQuantity::WorldNormalX,
            DebugQuantity::WorldNormalY,
            DebugQuantity::WorldNormalZ,
            DebugQuantity::Roughness,
            DebugQuantity::Metallic,
            DebugQuantity::AoFactor,
            DebugQuantity::DirectLightLuminance,
            DebugQuantity::AmbientLuminance,
            DebugQuantity::IblDiffuseLuminance,
            DebugQuantity::IblSpecularLuminance,
            DebugQuantity::EmissiveLuminance,
        ]
    }
}

/// How the debug channel value is composited over the normal render.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DebugOutputMode {
    /// Fragment color is entirely replaced by the debug output.
    #[default]
    Replace,
    /// Debug output is blended over the normal color at 50% opacity.
    TintOverlay,
    /// Left half shows normal render, right half shows debug quantity.
    /// Split position is set by `DebugVis::split_x`.
    SplitScreen,
}

/// Debug visualization configuration for the shadow and lighting pipeline.
///
/// Attach to [`LightingSettings::debug_vis`] to activate per-fragment channel output.
/// When `active` is false all overhead is zero: `debug_vis_mode` is written as 0
/// and the shader exits the debug path immediately.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct DebugVis {
    /// Whether debug visualization is active.
    pub active: bool,
    /// How the debug value is composited over the normal render.
    pub mode: DebugOutputMode,
    /// Quantity shown in the red channel.
    pub channel_r: DebugQuantity,
    /// Quantity shown in the green channel.
    pub channel_g: DebugQuantity,
    /// Quantity shown in the blue channel.
    pub channel_b: DebugQuantity,
    /// Multiplier applied to the raw quantity value before display.
    /// Use values > 1.0 to bring small quantities (e.g. bias in world units) into visible range.
    pub scale: f32,
    /// Split position for `SplitScreen` mode, in normalized viewport coordinates (0..1).
    /// 0.5 = center. Ignored unless mode is `SplitScreen`.
    pub split_x: f32,
}

impl DebugVis {
    /// Create a new active debug visualization configuration.
    pub fn new(
        mode: DebugOutputMode,
        channel_r: DebugQuantity,
        channel_g: DebugQuantity,
        channel_b: DebugQuantity,
        scale: f32,
    ) -> Self {
        Self {
            active: true,
            mode,
            channel_r,
            channel_g,
            channel_b,
            scale,
            split_x: 0.5,
        }
    }

    /// Pack the channel selectors, output mode, and active flag into a single u32.
    ///
    /// Bit layout:
    ///   bits 0..5   : channel R quantity index (5 bits, up to 32 quantities)
    ///   bits 5..10  : channel G quantity index
    ///   bits 10..15 : channel B quantity index
    ///   bits 15..17 : output mode (0=Replace, 1=TintOverlay)
    ///   bit 31      : active flag (0=off, non-zero=on)
    ///
    /// Returns 0 when not active, so the shader exits immediately.
    pub fn pack_mode(&self) -> u32 {
        if !self.active {
            return 0;
        }
        let r = self.channel_r.index() & 0x1f;
        let g = self.channel_g.index() & 0x1f;
        let b = self.channel_b.index() & 0x1f;
        let m = match self.mode {
            DebugOutputMode::Replace => 0u32,
            DebugOutputMode::TintOverlay => 1u32,
            DebugOutputMode::SplitScreen => 2u32,
        };
        (1u32 << 31) | (m << 15) | (b << 10) | (g << 5) | r
    }
}

/// Viewport corner where the shadow atlas viewer overlay is anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum AtlasViewerCorner {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner (default).
    #[default]
    BottomRight,
}
