use super::*;

mod helpers;
use helpers::*;
mod gpu;
mod point;
mod rect;

// ---------------------------------------------------------------------------
// PickRectResult
// ---------------------------------------------------------------------------

/// Result of a [`ViewportRenderer::pick_rect`] call.
#[derive(Clone, Debug, Default)]
pub struct PickRectResult {
    /// IDs of whole items that have geometry inside the pick rect.
    ///
    /// Populated when [`crate::interaction::pick_mask::PickMask::OBJECT`] is set.
    pub objects: Vec<u64>,
    /// Sub-elements inside the pick rect as `(item_id, sub_object)` pairs.
    ///
    /// Populated when any sub-element bit is set in the mask. All entries
    /// belong to the same geometric dimension when the mask is
    /// dimension-homogeneous (the common case).
    pub elements: Vec<(u64, crate::interaction::sub_object::SubObjectRef)>,
}

impl PickRectResult {
    /// Returns `true` when no objects or elements were found.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.elements.is_empty()
    }
}
