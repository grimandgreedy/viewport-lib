//! GPU pick identifier for a renderable surface.

/// Typed GPU pick identifier for a renderable surface.
///
/// The renderer includes a surface in the GPU pick pass only when its `pick_id`
/// is not [`PickId::NONE`]. Helper geometry, transient previews, and any surface
/// that should be invisible to clicks should use `PickId::NONE` (the default).
///
/// Application code owns pick identity: IDs do not need to come from the scene
/// graph. Any nonzero `u64` that the application can map back to a domain object
/// is a valid `PickId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PickId(pub u64);

impl PickId {
    /// Sentinel value meaning "not pickable".
    ///
    /// Surfaces with `pick_id == PickId::NONE` are excluded from the GPU pick
    /// pass. This is the default value for a render item's `pick_id`.
    pub const NONE: Self = Self(0);
}
