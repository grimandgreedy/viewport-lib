//! Per-frame viewport context for the new input pipeline.

/// Per-frame viewport context passed to [`super::controller::OrbitCameraController::begin_frame`]
/// (and [`super::viewport_input::ViewportInput::begin_frame`]).
///
/// Carries the hover, focus, and size state needed to interpret viewport input.
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
