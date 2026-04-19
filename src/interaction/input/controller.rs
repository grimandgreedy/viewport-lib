//! High-level orbit/pan/zoom camera controller.
//!
//! [`OrbitCameraController`] is the ergonomic entry point for standard viewport
//! camera navigation. It wraps [`super::viewport_input::ViewportInput`] and
//! applies resolved orbit / pan / zoom actions directly to a [`crate::Camera`].

use crate::Camera;

use super::action_frame::ActionFrame;
use super::context::ViewportContext;
use super::event::ViewportEvent;
use super::preset::{BindingPreset, viewport_all_bindings, viewport_primitives_bindings};
use super::viewport_input::ViewportInput;

/// High-level orbit / pan / zoom camera controller.
///
/// Wraps the lower-level [`ViewportInput`] resolver and applies semantic
/// camera actions to a [`Camera`] in a single `apply_to_camera` call.
///
/// # Integration pattern (winit / single window)
///
/// ```text
/// // --- AppState construction ---
/// controller.begin_frame(ViewportContext { hovered: true, focused: true, viewport_size });
///
/// // --- window_event ---
/// controller.push_event(translated_event);
///
/// // --- RedrawRequested ---
/// controller.apply_to_camera(&mut state.camera);
/// controller.begin_frame(ViewportContext { hovered: true, focused: true, viewport_size });
/// // ... render ...
/// ```
///
/// # Integration pattern (eframe / egui)
///
/// ```text
/// // --- update() ---
/// controller.begin_frame(ViewportContext {
///     hovered: response.hovered(),
///     focused: response.has_focus(),
///     viewport_size: [rect.width(), rect.height()],
/// });
/// // push events from ui.input(|i| { ... })
/// controller.apply_to_camera(&mut self.camera);
/// ```
pub struct OrbitCameraController {
    input: ViewportInput,
    /// Sensitivity for drag-based orbit (radians per pixel).
    pub orbit_sensitivity: f32,
    /// Sensitivity for scroll-based zoom (scale factor per pixel).
    pub zoom_sensitivity: f32,
    /// Current viewport size (cached from the last `begin_frame`).
    viewport_size: [f32; 2],
}

impl OrbitCameraController {
    /// Default drag orbit sensitivity: 0.005 radians per pixel.
    pub const DEFAULT_ORBIT_SENSITIVITY: f32 = 0.005;
    /// Default scroll zoom sensitivity: 0.001 scale per pixel.
    pub const DEFAULT_ZOOM_SENSITIVITY: f32 = 0.001;

    /// Create a controller from the given binding preset.
    pub fn new(preset: BindingPreset) -> Self {
        let bindings = match preset {
            BindingPreset::ViewportPrimitives => viewport_primitives_bindings(),
            BindingPreset::ViewportAll => viewport_all_bindings(),
        };
        Self {
            input: ViewportInput::new(bindings),
            orbit_sensitivity: Self::DEFAULT_ORBIT_SENSITIVITY,
            zoom_sensitivity: Self::DEFAULT_ZOOM_SENSITIVITY,
            viewport_size: [1.0, 1.0],
        }
    }

    /// Create a controller with the [`BindingPreset::ViewportPrimitives`] preset.
    ///
    /// This is the canonical control scheme matching `examples/winit_primitives`.
    pub fn viewport_primitives() -> Self {
        Self::new(BindingPreset::ViewportPrimitives)
    }

    /// Create a controller with the [`BindingPreset::ViewportAll`] preset.
    ///
    /// Includes all camera navigation bindings plus keyboard shortcuts for
    /// normal mode, fly mode, and manipulation mode. Use this to replace
    /// [`crate::InputSystem`] entirely.
    pub fn viewport_all() -> Self {
        Self::new(BindingPreset::ViewportAll)
    }

    /// Begin a new frame.
    ///
    /// Resets per-frame accumulators and records viewport context (hover/focus
    /// state and size). Call this at the **end** of each rendered frame — after
    /// `apply_to_camera` — so the accumulator is ready for the next batch of
    /// events.
    ///
    /// Also call once immediately after construction to prime the accumulator.
    pub fn begin_frame(&mut self, ctx: ViewportContext) {
        self.viewport_size = ctx.viewport_size;
        self.input.begin_frame(ctx);
    }

    /// Push a single viewport-scoped event into the accumulator.
    ///
    /// Call this from the host's event handler whenever a relevant native event
    /// arrives, after translating it to a [`ViewportEvent`].
    pub fn push_event(&mut self, event: ViewportEvent) {
        self.input.push_event(event);
    }

    /// Resolve accumulated events into an [`ActionFrame`] without applying any
    /// camera navigation.
    ///
    /// Use this when the caller needs to inspect actions but camera movement
    /// should be suppressed — for example during gizmo manipulation or fly mode
    /// where the camera is driven by other logic.
    pub fn resolve(&self) -> ActionFrame {
        self.input.resolve()
    }

    /// Resolve accumulated events, apply camera navigation, and return the
    /// [`ActionFrame`] for this frame.
    ///
    /// Call this in the render / update step, **before** `begin_frame` for the
    /// next frame.
    pub fn apply_to_camera(&self, camera: &mut Camera) -> ActionFrame {
        let frame = self.input.resolve();
        let nav = &frame.navigation;

        let h = self.viewport_size[1];

        if nav.orbit != glam::Vec2::ZERO {
            camera.orbit(
                nav.orbit.x * self.orbit_sensitivity,
                nav.orbit.y * self.orbit_sensitivity,
            );
        }

        if nav.pan != glam::Vec2::ZERO {
            camera.pan_pixels(nav.pan, h);
        }

        if nav.zoom != 0.0 {
            camera.zoom_by_factor(1.0 - nav.zoom * self.zoom_sensitivity);
        }

        frame
    }
}
