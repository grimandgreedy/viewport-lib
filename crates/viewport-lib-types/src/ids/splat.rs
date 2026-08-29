//! Handle for an uploaded Gaussian splat set.

crate::slot_handle! {
    /// Handle to an uploaded Gaussian splat set.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. A handle whose splat set was removed (its slot freed and
    /// reused by a later upload) resolves to no set on lookup, so it cannot
    /// alias whatever now occupies the slot.
    pub struct GaussianSplatId;
}
