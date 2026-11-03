//! The tier a device lands on.

/// How much of the renderer a device can run.
///
/// Ordered from least to most capable, so `tier >= Tier::OverlayOnly` reads
/// as "can draw overlays".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Tier {
    /// Below the overlay floor: no renderer can be built.
    Unsupported,
    /// Below the full tier, without the bind groups, sampled textures or
    /// uniform blocks the compatibility tier needs (a device with two bind groups,
    /// for example). The renderer builds only the overlay family and draws it
    /// on the LDR path. Scene items are skipped, and the calls that add scene
    /// content return `ViewportError::TierUnsupported`.
    OverlayOnly,
    /// Below the full tier on storage buffers or compute, with the bind groups,
    /// textures and uniform blocks the compatibility group 0 needs (WebGL2 and most
    /// GLES 3.x devices). Draws what the overlay-only tier draws, plus meshes
    /// through a basic-lit path (base colour, vertex colour, an albedo
    /// texture, colourmapped per-vertex scalars, alpha mask and blending) and
    /// the item types whose `requires()` accept this tier, lit by a capped
    /// light list without shadows, image-based lighting or post effects.
    Compatibility,
    /// Everything the renderer has, with the per-feature fallbacks that apply
    /// when an optional feature is missing. The default.
    #[default]
    Full,
}

impl Tier {
    /// Bind groups every tier needs. The mesh and overlay paths bind two, which
    /// is what iced's shared device grants.
    pub const BIND_GROUPS: u32 = 2;
    /// Uniform binding size every tier needs: WebGL2's 16 KiB.
    pub const UNIFORM_BINDING_SIZE: u64 = 16 << 10;
    /// Storage buffers per stage the full tier needs: wgpu's default, and the
    /// most the base lit mesh path binds in one stage.
    pub const FULL_STORAGE_BUFFERS_PER_STAGE: u32 = 8;
    /// Bind groups the compatibility tier needs: group 0, an item type's own group,
    /// and one more for its instances or picking.
    pub const COMPATIBILITY_BIND_GROUPS: u32 = 3;
    /// Sampled textures per stage the compatibility tier needs: WebGL2's 16. Its
    /// group 0 binds 6 in the fragment stage.
    pub const COMPATIBILITY_SAMPLED_TEXTURES_PER_STAGE: u32 = 16;
    /// Uniform buffers per stage the compatibility tier needs: WebGL2's 11. Its
    /// group 0 binds 6 in the fragment stage.
    pub const COMPATIBILITY_UNIFORM_BUFFERS_PER_STAGE: u32 = 11;
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Tier::Unsupported => "unsupported",
            Tier::OverlayOnly => "overlay-only",
            Tier::Compatibility => "compatibility",
            Tier::Full => "full",
        })
    }
}
