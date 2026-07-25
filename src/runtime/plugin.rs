//! Runtime plugin trait and phase ordering.

use super::context::RuntimeStepContext;
use crate::interaction::select::selection::NodeId;
use std::sync::{Arc, Mutex};

/// Named priority band constants for runtime plugins.
///
/// Each constant is the base priority for a logical execution phase. Plugins
/// can use values between bands (e.g. `phase::ANIMATE + 50`) to run at a
/// specific point within a range.
pub mod phase {
    /// First phase each frame. Update time-dependent state before any queries.
    pub const PREPARE: i32 = 100;
    /// Ray-cast and object picking.
    pub const PICK: i32 = 200;
    /// Selection state updates driven by pick results.
    pub const SELECT: i32 = 300;
    /// Transform manipulation from gizmo drag or keyboard input.
    pub const MANIPULATE: i32 = 400;
    /// Procedural or keyframe animation.
    pub const ANIMATE: i32 = 500;
    /// Physics or simulation. With a fixed timestep this runs once per
    /// accumulated step.
    pub const SIMULATE: i32 = 600;
    /// Runs after all Simulate iterations, before Writeback.
    pub const POST_SIM: i32 = 700;
    /// Flush accumulated transform ops to the scene.
    pub const WRITEBACK: i32 = 800;
}

/// Named execution phase for a runtime plugin.
///
/// This enum is a convenience alias for the numeric priority bands in the
/// [`phase`] module. Use [`to_priority`](RuntimePhase::to_priority) to convert
/// to an `i32` priority value.
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
    /// Runs after all Simulate iterations, before Writeback.
    PostSimulate,
    /// Flush accumulated transform ops to the scene. Runs once per frame after
    /// all other phases, including after all `Simulate` iterations.
    Writeback,
}

impl RuntimePhase {
    /// Convert this phase to its numeric priority value.
    pub fn to_priority(&self) -> i32 {
        match self {
            RuntimePhase::Prepare => phase::PREPARE,
            RuntimePhase::Pick => phase::PICK,
            RuntimePhase::Select => phase::SELECT,
            RuntimePhase::Manipulate => phase::MANIPULATE,
            RuntimePhase::Animate => phase::ANIMATE,
            RuntimePhase::Simulate => phase::SIMULATE,
            RuntimePhase::PostSimulate => phase::POST_SIM,
            RuntimePhase::Writeback => phase::WRITEBACK,
        }
    }
}

/// An event emitted by the runtime when the scene node set changes.
#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    /// A node was added to the scene since the last frame.
    NodeAdded(NodeId),
    /// A node was removed from the scene since the last frame.
    NodeRemoved(NodeId),
}

/// A plugin that executes during the scene step.
///
/// Register plugins with [`super::ViewportRuntime::with_plugin`]. Each frame
/// the runtime calls `submit`, then `step` (once or more for simulate phases),
/// then `collect`.
///
/// Registration model: runtime plugins are multi-instance. They carry no
/// external identity, so registering two of the same type runs both, in
/// priority order; the runtime does not deduplicate. This is intentional (two
/// emitters, two solvers with different config are valid) and is frozen
/// behavior. Item-type plugins differ: they are singleton-by-type because
/// their `type_name` is referenced externally. See
/// [`ItemTypePlugin`](crate::plugin_api::ItemTypePlugin).
///
/// # Example
///
/// ```rust,ignore
/// use viewport_lib::runtime::{RuntimePlugin, RuntimeStepContext};
/// use viewport_lib::runtime::plugin::phase;
///
/// struct GravityPlugin {
///     gravity: glam::Vec3,
/// }
///
/// impl RuntimePlugin for GravityPlugin {
///     fn priority(&self) -> i32 {
///         phase::SIMULATE
///     }
///
///     fn step(&mut self, ctx: &mut RuntimeStepContext) {
///         // Apply gravity to tracked bodies via ctx.writeback.set(id, new_transform).
///     }
/// }
/// ```
pub trait RuntimePlugin: Send + 'static {
    /// Numeric execution priority. Lower values run first. Use the constants
    /// in [`phase`] as base values and offset within a band if needed.
    fn priority(&self) -> i32;

    /// A stable name for this plugin, used to key per-plugin timing in
    /// [`RuntimeStats`](crate::RuntimeStats). Defaults to the concrete type
    /// name; override to disambiguate when several plugins share a type or to
    /// give a friendlier label.
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Called once per frame before the step loop, in priority order.
    ///
    /// Use this to kick off background work (e.g. async physics) that will
    /// be read in `collect` on the next frame.
    fn submit(&mut self, _ctx: &RuntimeStepContext<'_>) {}

    /// Called once per frame after the step loop, in priority order.
    ///
    /// Use this to read results from background work started in `submit`.
    fn collect(&mut self, _ctx: &mut RuntimeStepContext<'_>) {}

    /// Called for each lifecycle event before the step loop.
    fn on_event(&mut self, _event: &RuntimeEvent, _ctx: &mut RuntimeStepContext<'_>) {}

    /// Called once per phase execution.
    ///
    /// For plugins in the `[SIMULATE, POST_SIM)` range with a fixed timestep,
    /// `ctx.dt` is the fixed step size and this is called once per step. For
    /// all other priorities, `ctx.dt` is the wall-clock frame delta and this
    /// is called once per frame.
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>);
}

/// Lets a plugin behind an `Arc<Mutex<P>>` register as a runtime plugin.
///
/// A feature that steps on the CPU and also encodes GPU work can implement both
/// `RuntimePlugin` and `GpuPlugin` on one struct, wrap it in `Arc<Mutex<..>>`,
/// and register the same object into both lists by cloning the `Arc`. The host
/// keeps a clone for its own mutation.
///
/// `type_name` forwards to the inner plugin so `RuntimeStats` keys stay keyed by
/// the real plugin name rather than the wrapper. The host must not hold the lock
/// across `runtime.step()` / `pre_prepare()`: the runtime and GPU dispatch run
/// in sequence from the frame loop, so the mutex is uncontended in that usage.
impl<P: RuntimePlugin> RuntimePlugin for Arc<Mutex<P>> {
    fn priority(&self) -> i32 {
        self.lock().unwrap().priority()
    }

    fn type_name(&self) -> &'static str {
        self.lock().unwrap().type_name()
    }

    fn submit(&mut self, ctx: &RuntimeStepContext<'_>) {
        self.lock().unwrap().submit(ctx);
    }

    fn collect(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        self.lock().unwrap().collect(ctx);
    }

    fn on_event(&mut self, event: &RuntimeEvent, ctx: &mut RuntimeStepContext<'_>) {
        self.lock().unwrap().on_event(event, ctx);
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        self.lock().unwrap().step(ctx);
    }
}
