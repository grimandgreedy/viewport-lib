//! A runtime plugin that records its hooks and emits one event and one
//! command.

use crate::fixtures::CallLog;
use viewport_lib::runtime::{CameraCommand, RuntimeEvent, RuntimePlugin, RuntimeStepContext};

/// The event [`LoggingRuntimePlugin`] emits, so a test can assert typed
/// delivery through the runtime's event bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeEvent {
    /// The label of the plugin that emitted it.
    pub label: &'static str,
}

/// Records `submit` / `step` / `collect` / `on_event`, each tagged with the
/// plugin's label, and emits a [`ProbeEvent`] plus one
/// [`CameraCommand`](viewport_lib::runtime::CameraCommand) from `step`.
///
/// Register two of these at different priorities to assert the runtime's
/// ordering: every plugin's `submit` runs (in priority order) before the step
/// loop, and every `collect` runs after it.
pub struct LoggingRuntimePlugin {
    log: CallLog,
    label: &'static str,
    priority: i32,
}

impl LoggingRuntimePlugin {
    /// A plugin logging under `label` at `priority`. Use the
    /// [`phase`](viewport_lib::runtime::plugin::phase) constants for the
    /// priority so the fixture sits in a real band.
    pub fn new(log: CallLog, label: &'static str, priority: i32) -> Self {
        Self {
            log,
            label,
            priority,
        }
    }

    fn record(&self, hook: &str) {
        self.log.record(format!("{}:{hook}", self.label));
    }
}

impl RuntimePlugin for LoggingRuntimePlugin {
    fn priority(&self) -> i32 {
        self.priority
    }

    fn type_name(&self) -> &'static str {
        self.label
    }

    fn submit(&mut self, _ctx: &RuntimeStepContext<'_>) {
        self.record("submit");
    }

    fn collect(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
        self.record("collect");
    }

    fn on_event(&mut self, event: &RuntimeEvent, _ctx: &mut RuntimeStepContext<'_>) {
        let kind = match event {
            RuntimeEvent::NodeAdded(_) => "node_added",
            RuntimeEvent::NodeRemoved(_) => "node_removed",
        };
        self.record(&format!("on_event:{kind}"));
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        self.record("step");
        ctx.output.events.emit(ProbeEvent { label: self.label });
        ctx.output
            .events
            .emit(CameraCommand::OffsetCenter(glam::Vec3::Z));
    }
}
