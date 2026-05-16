//! Per-frame and per-step context types for runtime plugins.

use crate::camera::camera::Camera;
use crate::interaction::input::ActionFrame;
use crate::scene::scene::Scene;

use super::output::{RuntimeOutput, TransformWriteback};
use super::plugin::RuntimePhase;

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
}

/// Context passed to each plugin during its execution phase.
///
/// Provides read-only scene access and write access to the transform writeback
/// buffer and output accumulator. To write transforms, call
/// [`TransformWriteback::set`] on `self.writeback`.
pub struct RuntimeStepContext<'a> {
    /// The phase this plugin is executing in.
    pub phase: RuntimePhase,
    /// Delta time for this step. For `Simulate` with a fixed timestep this is
    /// the fixed step size, not the wall dt.
    pub dt: f32,
    /// Read-only scene access. To write transforms, use `writeback`.
    pub scene: &'a Scene,
    /// Accumulate transform writes here. The runtime flushes them to the scene
    /// after the `Writeback` phase.
    pub writeback: &'a mut TransformWriteback,
    /// Accumulate selection changes and contact events here.
    pub output: &'a mut RuntimeOutput,
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
/// Constructed via [`RuntimeStepContext::as_simulation`] inside a `Simulate`-phase
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
