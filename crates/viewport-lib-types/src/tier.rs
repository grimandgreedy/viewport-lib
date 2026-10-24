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
    /// Storage buffers or compute are missing (WebGL2, GLES 3.0, many GLES 3.1
    /// devices). The renderer builds only the overlay family and draws it on
    /// the LDR path. Scene items are skipped, and the calls that add scene
    /// content return `ViewportError::TierUnsupported`.
    OverlayOnly,
    /// Reserved for lit meshes without storage buffers or compute. No device
    /// reports it yet.
    Forward,
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
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Tier::Unsupported => "unsupported",
            Tier::OverlayOnly => "overlay-only",
            Tier::Forward => "forward",
            Tier::Full => "full",
        })
    }
}
