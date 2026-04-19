//! Per-frame viewport context for the new input pipeline.

/// Per-frame viewport context passed to [`super::controller::OrbitCameraController::begin_frame`]
/// (and the lower-level [`super::viewport_input::ViewportInput::begin_frame`]).
///
/// Makes viewport ownership explicit so the library can apply consistent
/// hover / focus policies without each host app re-implementing them.
#[derive(Debug, Clone, Copy)]
pub struct ViewportContext {
    /// Whether the pointer is currently hovering over the viewport.
    pub hovered: bool,
    /// Whether the viewport currently has keyboard focus.
    pub focused: bool,
    /// Viewport size in logical pixels `[width, height]`.
    pub viewport_size: [f32; 2],
}

impl Default for ViewportContext {
    fn default() -> Self {
        Self {
            hovered: false,
            focused: false,
            viewport_size: [1.0, 1.0],
        }
    }
}
