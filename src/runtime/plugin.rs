//! Runtime plugin trait and phase ordering.

use super::context::RuntimeStepContext;

/// Execution phase for a runtime plugin.
///
/// Plugins run in the order defined by this enum's variants. Within a single
/// phase, plugins execute in registration order. The `Simulate` phase may run
/// more than once per frame when a fixed timestep is configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RuntimePhase {
    /// Runs first. Update time-dependent state before any queries.
    Prepare,
    /// Ray-cast and object picking.
    Pick,
    /// Selection state updates driven by pick results.
    Select,
    /// Transform manipulation from gizmo drag or keyboard input.
    Manipulate,
    /// Procedural or keyframe animation.
    Animate,
    /// Physics or simulation. With a fixed timestep this runs once per
    /// accumulated step, so it may execute multiple times per rendered frame.
    Simulate,
    /// Flush accumulated transform ops to the scene. Runs once per frame after
    /// all other phases, including after all `Simulate` iterations.
    Writeback,
}

/// A plugin that executes during one phase of the scene step.
///
/// Register plugins with [`super::ViewportRuntime::with_plugin`]. The runtime
/// calls [`step`](RuntimePlugin::step) once per phase execution.
///
/// # Example
///
/// ```rust,ignore
/// use viewport_lib::runtime::{RuntimePhase, RuntimePlugin, RuntimeStepContext};
///
/// struct GravityPlugin {
///     gravity: glam::Vec3,
/// }
///
/// impl RuntimePlugin for GravityPlugin {
///     fn phase(&self) -> RuntimePhase {
///         RuntimePhase::Simulate
///     }
///
///     fn step(&mut self, ctx: &mut RuntimeStepContext) {
///         // Apply gravity to tracked bodies via ctx.writeback.set(id, new_transform).
///     }
/// }
/// ```
pub trait RuntimePlugin: Send + 'static {
    /// The phase this plugin runs in.
    fn phase(&self) -> RuntimePhase;

    /// Called once per phase execution.
    ///
    /// For `Simulate` with a fixed timestep, `ctx.dt` is the fixed step size
    /// and this is called once per step. For all other phases, `ctx.dt` is the
    /// wall-clock frame delta and this is called once per frame.
    fn step(&mut self, ctx: &mut RuntimeStepContext);
}
