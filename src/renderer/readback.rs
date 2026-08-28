//! Snapshots the renderer reads back from the GPU and hands to the host.
//!
//! These are outputs of a frame, not configuration: values produced on the GPU
//! (metered exposure, cluster occupancy) that a debug panel or auto-exposure
//! logic reads after `prepare`. Keeping them here keeps the per-frame config
//! types (post-process, lighting) free of output-only structs.

/// A snapshot of a viewport's exposure state, read back from the GPU via
/// [`ViewportRenderer::exposure_state`](crate::ViewportRenderer::exposure_state).
///
/// EVs are EV100. For Manual / PhysicalCamera, `current_ev == target_ev` and
/// `adapting` is always `false`; the fields are only interesting under
/// [`ExposureMode::Automatic`](crate::ExposureMode::Automatic).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureReadback {
    /// The linear exposure multiplier currently applied before tone mapping.
    pub exposure: f32,
    /// The adapted EV100 in effect this frame.
    pub current_ev: f32,
    /// The metered target EV100 (what adaptation is easing toward).
    pub target_ev: f32,
    /// Whether adaptation is still easing toward the target (Automatic only).
    pub adapting: bool,
}
