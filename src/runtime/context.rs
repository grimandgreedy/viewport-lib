//! Per-frame and per-step context types for runtime plugins.

use crate::camera::camera::Camera;
use crate::interaction::input::ActionFrame;
use crate::scene::scene::Scene;

use super::output::{RuntimeOutput, TransformWriteback};

/// Per-frame inputs to [`super::ViewportRuntime::step`].
pub struct RuntimeFrameContext<'a> {
    /// Wall-clock delta time in seconds since the last frame.
    pub dt: f32,
    /// Current camera state.
    pub camera: &'a Camera,
    /// Viewport dimensions in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Resolved input actions for this frame.
    pub input: &'a ActionFrame,
    /// Pick result under the cursor for this frame. Supply from CPU or GPU picking.
    /// None if no picking was done or nothing was hit.
    pub pick_hit: Option<crate::interaction::picking::PickHit>,
    /// True on the frame the primary pointer button was clicked (pressed and released without drag).
    pub clicked: bool,
    /// True on the frame a primary drag began.
    pub drag_started: bool,
    /// True while a primary drag is ongoing.
    pub dragging: bool,
    /// Pointer movement in viewport pixels since last frame.
    pub pointer_delta: glam::Vec2,
    /// Cursor position in viewport-local pixels. None when outside the viewport.
    pub cursor_viewport: Option<glam::Vec2>,
    /// True when the shift modifier is held (for shift-click multi-select).
    pub shift_held: bool,
}

/// Context passed to each plugin during its execution step.
///
/// Provides read-only scene access and write access to the transform writeback
/// buffer and output accumulator. To write transforms, call
/// [`TransformWriteback::set`] on `self.writeback`.
pub struct RuntimeStepContext<'a> {
    /// The numeric priority of the plugin executing in this context.
    pub priority: i32,
    /// Delta time for this step. For plugins in the simulate range with a fixed
    /// timestep this is the fixed step size, not the wall dt.
    pub dt: f32,
    /// Read-only scene access. To write transforms, use `writeback`.
    pub scene: &'a Scene,
    /// Accumulate transform writes here. The runtime flushes them to the scene
    /// after the writeback phase.
    pub writeback: &'a mut TransformWriteback,
    /// Accumulate selection changes and contact events here.
    pub output: &'a mut RuntimeOutput,
    /// Forwarded from RuntimeFrameContext for plugins that need the current pick result.
    pub pick_hit: Option<crate::interaction::picking::PickHit>,
}

impl<'a> RuntimeStepContext<'a> {
    /// Construct a [`SimulationStepContext`] from this context.
    ///
    /// Useful for physics plugins that need to track per-step index alongside
    /// the standard context fields.
    pub fn as_simulation(&mut self, step_index: u64) -> SimulationStepContext<'_> {
        SimulationStepContext {
            dt: self.dt,
            step_index,
            scene: self.scene,
            writeback: self.writeback,
        }
    }
}

/// Narrower context for simulation plugins that need a per-step index.
///
/// Constructed via [`RuntimeStepContext::as_simulation`] inside a simulate-range
/// plugin to get the current simulation step count alongside scene and writeback access.
pub struct SimulationStepContext<'a> {
    /// Fixed step size in seconds.
    pub dt: f32,
    /// Monotonically increasing step counter (wraps on overflow).
    pub step_index: u64,
    /// Read-only scene access.
    pub scene: &'a Scene,
    /// Write transforms to this buffer.
    pub writeback: &'a mut TransformWriteback,
}
