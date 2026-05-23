//! Scene runtime: per-frame orchestration layer between [`Scene`] and [`ViewportRenderer`].
//!
//! [`ViewportRuntime`] runs registered plugins in a defined priority order each frame,
//! drives a fixed timestep accumulator for physics, and flushes accumulated transform
//! writes back to the scene. It does not own the scene, selection, or GPU resources.
//!
//! # Usage
//!
//! ```rust,ignore
//! use viewport_lib::runtime::{
//!     FixedTimestep, RuntimeFrameContext, RuntimePhase, RuntimePlugin,
//!     RuntimeStepContext, SceneRuntimeMode, ViewportRuntime,
//! };
//!
//! // Build the runtime once at startup.
//! let mut runtime = ViewportRuntime::new()
//!     .with_mode(SceneRuntimeMode::Simulation)
//!     .with_fixed_timestep(FixedTimestep::new(60.0))
//!     .with_plugin(MyPhysicsPlugin::new());
//!
//! // Each frame:
//! let frame_ctx = RuntimeFrameContext {
//!     dt: wall_dt,
//!     camera: &camera,
//!     viewport_size: glam::Vec2::new(width, height),
//!     input: &action_frame,
//! };
//! let output = runtime.step(&mut scene, &mut selection, &frame_ctx);
//!
//! // Handle contact events in game logic:
//! for event in &output.contact_events { /* ... */ }
//!
//! // Render with interpolated transforms between fixed steps:
//! let alpha = runtime.alpha();
//! if let Some(t) = runtime.snapshots().interpolated(node_id, alpha) {
//!     // use t instead of the node's scene transform
//! }
//! ```
//!
//! Existing `prepare` / `paint_to` call sites need no changes. `ViewportRuntime`
//! is purely additive and does not affect [`ViewportRenderer`](crate::ViewportRenderer).

pub mod camera_follow;
pub mod context;
/// Debug draw accumulator for runtime plugins.
pub mod debug_draw;
pub mod events;
/// Async job handoff types for runtime plugins.
pub mod jobs;
pub mod mode;
pub mod output;
pub mod plugin;
/// Built-in animation, constraint, and physics plugins.
pub mod plugins;
pub mod resources;
pub mod snapshot;
/// Built-in interaction systems: SelectionSystem and ManipulationSystem.
pub mod systems;
pub mod timestep;

pub use camera_follow::CameraFollow;
pub use context::{RuntimeFrameContext, RuntimeStepContext, SimulationStepContext};
pub use debug_draw::{DebugDraw, DebugLayer, DebugPrim};
pub use events::RuntimeEventBus;
pub use jobs::{JobPoll, JobSender, JobSlot};
pub use mode::SceneRuntimeMode;
pub use output::{CameraCommand, ContactEvent, NodeTransformOp, RuntimeOutput, SelectionOp, SkinnedMeshUpdate, TransformWriteback};
pub use plugin::{phase, RuntimeEvent, RuntimePhase, RuntimePlugin};
pub use plugins::{
    AnimationPlugin, AnimationTrack, Constraint, ConstraintPlugin, Joint, JointMatrices, Keyframe,
    PhysicsBody, PhysicsLitePlugin, Pose, Skeleton, SkeletonPlugin, apply_skin,
};
pub use resources::RuntimeResources;
pub use snapshot::{TransformSnapshot, TransformSnapshotTable};
pub use systems::{ManipulationSystem, SelectionSystem};
pub use timestep::{FixedStepIter, FixedTimestep};

use crate::interaction::selection::Selection;
use crate::scene::scene::Scene;

// ---- free function ----------------------------------------------------------

/// Run all plugins whose priority is in `[min, max)`, in the order they appear
/// in the slice.
fn run_range(
    plugins: &mut Vec<Box<dyn RuntimePlugin>>,
    min: i32,
    max: i32,
    dt: f32,
    frame: &RuntimeFrameContext,
    scene: &Scene,
    writeback: &mut TransformWriteback,
    output: &mut RuntimeOutput,
    resources: &mut RuntimeResources,
) {
    for plugin in plugins.iter_mut() {
        let p = plugin.priority();
        if p >= min && p < max {
            let mut ctx = RuntimeStepContext {
                priority: p,
                dt,
                scene,
                writeback,
                output,
                pick_hit: frame.pick_hit,
                resources,
            };
            plugin.step(&mut ctx);
        }
    }
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::camera::Camera;
    use crate::interaction::input::ActionFrame;
    use crate::interaction::selection::{NodeId, Selection};
    use crate::scene::material::Material;
    use crate::scene::scene::Scene;
    use std::sync::{Arc, Mutex};

    fn make_frame<'a>(camera: &'a Camera, input: &'a ActionFrame, dt: f32) -> RuntimeFrameContext<'a> {
        RuntimeFrameContext {
            dt,
            camera,
            viewport_size: glam::Vec2::new(800.0, 600.0),
            input,
            pick_hit: None,
            clicked: false,
            drag_started: false,
            dragging: false,
            pointer_delta: glam::Vec2::ZERO,
            cursor_viewport: None,
            shift_held: false,
        }
    }

    // Records (priority, id) each time step() is called.
    struct OrderTracker {
        priority: i32,
        id: u32,
        log: Arc<Mutex<Vec<(i32, u32)>>>,
    }

    impl RuntimePlugin for OrderTracker {
        fn priority(&self) -> i32 {
            self.priority
        }
        fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
            self.log.lock().unwrap().push((self.priority, self.id));
        }
    }

    // Counts total step() calls.
    struct CallCounter {
        priority: i32,
        count: Arc<Mutex<u32>>,
    }

    impl RuntimePlugin for CallCounter {
        fn priority(&self) -> i32 {
            self.priority
        }
        fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
            *self.count.lock().unwrap() += 1;
        }
    }

    // Writes a fixed transform for one node each step.
    struct WritebackPlugin {
        node_id: NodeId,
        transform: glam::Affine3A,
    }

    impl RuntimePlugin for WritebackPlugin {
        fn priority(&self) -> i32 {
            phase::SIMULATE
        }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            ctx.writeback.set(self.node_id, self.transform);
        }
    }

    // Records the dt value passed to each Simulate step.
    struct DtRecorder {
        dts: Arc<Mutex<Vec<f32>>>,
    }

    impl RuntimePlugin for DtRecorder {
        fn priority(&self) -> i32 {
            phase::SIMULATE
        }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            self.dts.lock().unwrap().push(ctx.dt);
        }
    }

    #[test]
    fn test_phase_execution_order() {
        // Register plugins in a scrambled order; verify they execute in priority order.
        let log = Arc::new(Mutex::new(Vec::<(i32, u32)>::new()));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(OrderTracker { priority: phase::WRITEBACK, id: 3, log: log.clone() })
            .with_plugin(OrderTracker { priority: phase::SIMULATE,  id: 2, log: log.clone() })
            .with_plugin(OrderTracker { priority: phase::ANIMATE,   id: 1, log: log.clone() })
            .with_plugin(OrderTracker { priority: phase::PREPARE,   id: 0, log: log.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let calls = log.lock().unwrap();
        let priorities: Vec<i32> = calls.iter().map(|(p, _)| *p).collect();
        // Verify priorities are in ascending order.
        for w in priorities.windows(2) {
            assert!(w[0] <= w[1], "expected ascending order, got {:?}", priorities);
        }
        assert_eq!(priorities.len(), 4);
    }

    #[test]
    fn test_registration_order_within_phase() {
        // Two plugins at the same priority must execute in registration order.
        let log = Arc::new(Mutex::new(Vec::<(i32, u32)>::new()));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(OrderTracker { priority: phase::SIMULATE, id: 0, log: log.clone() })
            .with_plugin(OrderTracker { priority: phase::SIMULATE, id: 1, log: log.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let calls = log.lock().unwrap();
        let ids: Vec<u32> = calls.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn test_simulate_runs_once_without_fixed_timestep() {
        let count = Arc::new(Mutex::new(0u32));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(CallCounter { priority: phase::SIMULATE, count: count.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        assert_eq!(*count.lock().unwrap(), 1);
    }

    #[test]
    fn test_simulate_runs_n_times_with_fixed_timestep() {
        let count = Arc::new(Mutex::new(0u32));
        let hz = 60.0_f32;
        let step_dt = 1.0 / hz;
        let mut runtime = ViewportRuntime::new()
            .with_fixed_timestep(FixedTimestep::new(hz))
            .with_plugin(CallCounter { priority: phase::SIMULATE, count: count.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();

        // A frame spanning exactly 3 steps plus a tiny remainder yields 3 steps.
        let dt = step_dt * 3.0 + 0.0001;
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, dt));

        assert_eq!(*count.lock().unwrap(), 3);
    }

    #[test]
    fn test_simulate_dt_equals_step_dt_with_fixed_timestep() {
        let dts = Arc::new(Mutex::new(Vec::<f32>::new()));
        let hz = 60.0_f32;
        let step_dt = 1.0 / hz;
        let mut runtime = ViewportRuntime::new()
            .with_fixed_timestep(FixedTimestep::new(hz))
            .with_plugin(DtRecorder { dts: dts.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();

        // Frame spanning 2 steps.
        let dt = step_dt * 2.0 + 0.0001;
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, dt));

        let recorded = dts.lock().unwrap();
        assert_eq!(recorded.len(), 2);
        for &d in recorded.iter() {
            assert!((d - step_dt).abs() < 1e-6, "expected {step_dt}, got {d}");
        }
    }

    #[test]
    fn test_writeback_flushes_to_scene() {
        let target = glam::Affine3A::from_translation(glam::Vec3::new(3.0, 4.0, 5.0));
        let mut scene = Scene::new();
        let node_id = scene.add(None, glam::Mat4::IDENTITY, Material::default());

        let mut runtime = ViewportRuntime::new()
            .with_plugin(WritebackPlugin { node_id, transform: target });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let node = scene.node(node_id).expect("node not found");
        let pos = node.world_transform().col(3).truncate();
        assert!((pos - glam::Vec3::new(3.0, 4.0, 5.0)).length() < 1e-5, "pos was {pos:?}");
    }

    #[test]
    fn test_writeback_ops_in_output() {
        let target = glam::Affine3A::from_translation(glam::Vec3::new(1.0, 0.0, 0.0));
        let mut scene = Scene::new();
        let node_id = scene.add(None, glam::Mat4::IDENTITY, Material::default());

        let mut runtime = ViewportRuntime::new()
            .with_plugin(WritebackPlugin { node_id, transform: target });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut sel = Selection::new();
        let output = runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        assert_eq!(output.node_transform_ops.len(), 1);
        assert_eq!(output.node_transform_ops[0].id, node_id);
    }

    #[test]
    fn test_step_index_increments_each_simulate() {
        let mut runtime = ViewportRuntime::new();
        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();

        assert_eq!(runtime.step_index(), 0);
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));
        assert_eq!(runtime.step_index(), 1);
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));
        assert_eq!(runtime.step_index(), 2);
    }

    #[test]
    fn test_step_index_increments_n_times_with_fixed_timestep() {
        let hz = 60.0_f32;
        let step_dt = 1.0 / hz;
        let mut runtime = ViewportRuntime::new()
            .with_fixed_timestep(FixedTimestep::new(hz));

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();

        // 3 fixed steps in one frame.
        let dt = step_dt * 3.0 + 0.0001;
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, dt));
        assert_eq!(runtime.step_index(), 3);
    }

    #[test]
    fn test_snapshot_updated_after_writeback() {
        let target = glam::Affine3A::from_translation(glam::Vec3::new(7.0, 0.0, 0.0));
        let mut scene = Scene::new();
        let node_id = scene.add(None, glam::Mat4::IDENTITY, Material::default());

        let mut runtime = ViewportRuntime::new()
            .with_plugin(WritebackPlugin { node_id, transform: target });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let snap = runtime.snapshots().get(node_id).expect("snapshot missing");
        assert!((snap.curr.translation.x - 7.0).abs() < 1e-5);
    }

    // ---- new tests ----------------------------------------------------------

    #[test]
    fn test_post_sim_runs_after_simulate() {
        let log = Arc::new(Mutex::new(Vec::<(i32, u32)>::new()));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(OrderTracker { priority: phase::POST_SIM,  id: 1, log: log.clone() })
            .with_plugin(OrderTracker { priority: phase::SIMULATE,  id: 0, log: log.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let calls = log.lock().unwrap();
        let ids: Vec<u32> = calls.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![0, 1], "simulate must run before post_sim");
    }

    #[test]
    fn test_plugin_between_bands() {
        let log = Arc::new(Mutex::new(Vec::<(i32, u32)>::new()));
        let mid = phase::ANIMATE + 50;
        let mut runtime = ViewportRuntime::new()
            .with_plugin(OrderTracker { priority: phase::SIMULATE, id: 2, log: log.clone() })
            .with_plugin(OrderTracker { priority: mid,             id: 1, log: log.clone() })
            .with_plugin(OrderTracker { priority: phase::ANIMATE,  id: 0, log: log.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let calls = log.lock().unwrap();
        let ids: Vec<u32> = calls.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, vec![0, 1, 2], "plugins must execute in priority order");
    }

    // Plugin that records RuntimeEvent::NodeAdded / NodeRemoved calls.
    struct EventRecorder {
        added: Arc<Mutex<Vec<NodeId>>>,
        removed: Arc<Mutex<Vec<NodeId>>>,
    }

    impl RuntimePlugin for EventRecorder {
        fn priority(&self) -> i32 {
            phase::PREPARE
        }
        fn on_event(&mut self, event: &RuntimeEvent, _ctx: &mut RuntimeStepContext<'_>) {
            match event {
                RuntimeEvent::NodeAdded(id)   => self.added.lock().unwrap().push(*id),
                RuntimeEvent::NodeRemoved(id) => self.removed.lock().unwrap().push(*id),
            }
        }
        fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {}
    }

    #[test]
    fn test_lifecycle_events_node_added() {
        let added = Arc::new(Mutex::new(Vec::<NodeId>::new()));
        let removed = Arc::new(Mutex::new(Vec::<NodeId>::new()));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(EventRecorder { added: added.clone(), removed: removed.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();

        // First step: initializes snapshot, no events.
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));
        assert!(added.lock().unwrap().is_empty(), "no events on first step");

        // Add a node, then step.
        let node_id = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let a = added.lock().unwrap();
        assert!(a.contains(&node_id), "expected NodeAdded event for {:?}", node_id);
        assert!(removed.lock().unwrap().is_empty());
    }

    #[test]
    fn test_lifecycle_events_node_removed() {
        let added = Arc::new(Mutex::new(Vec::<NodeId>::new()));
        let removed = Arc::new(Mutex::new(Vec::<NodeId>::new()));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(EventRecorder { added: added.clone(), removed: removed.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();

        let node_id = scene.add(None, glam::Mat4::IDENTITY, Material::default());

        // First step with the node present.
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));
        added.lock().unwrap().clear();

        // Remove the node, then step.
        scene.remove(node_id);
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let r = removed.lock().unwrap();
        assert!(r.contains(&node_id), "expected NodeRemoved event for {:?}", node_id);
    }

    // Plugin that records which of submit, step, collect were called each frame.
    #[derive(Default)]
    struct LifecycleRecorder {
        log: Arc<Mutex<Vec<&'static str>>>,
    }

    impl RuntimePlugin for LifecycleRecorder {
        fn priority(&self) -> i32 {
            phase::PREPARE
        }
        fn submit(&mut self, _ctx: &RuntimeStepContext<'_>) {
            self.log.lock().unwrap().push("submit");
        }
        fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
            self.log.lock().unwrap().push("step");
        }
        fn collect(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
            self.log.lock().unwrap().push("collect");
        }
    }

    #[test]
    fn test_submit_collect_called() {
        let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(LifecycleRecorder { log: log.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();

        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let calls = log.lock().unwrap();
        assert!(calls.contains(&"submit"),  "submit must be called");
        assert!(calls.contains(&"step"),    "step must be called");
        assert!(calls.contains(&"collect"), "collect must be called");
        // submit before step before collect.
        let si = calls.iter().position(|&s| s == "submit").unwrap();
        let st = calls.iter().position(|&s| s == "step").unwrap();
        let co = calls.iter().position(|&s| s == "collect").unwrap();
        assert!(si < st, "submit must come before step");
        assert!(st < co, "step must come before collect");
    }

    // ---- resource tests -----------------------------------------------------

    #[test]
    fn test_resource_insert_get() {
        let mut res = RuntimeResources::new();
        res.insert(42u32);
        assert_eq!(res.get::<u32>(), Some(&42));
        assert!(res.get::<u64>().is_none());
    }

    #[test]
    fn test_resource_insert_overwrites() {
        let mut res = RuntimeResources::new();
        res.insert(1u32);
        res.insert(2u32);
        assert_eq!(res.get::<u32>(), Some(&2));
    }

    #[test]
    fn test_resource_get_mut() {
        let mut res = RuntimeResources::new();
        res.insert(0u32);
        *res.get_mut::<u32>().unwrap() = 99;
        assert_eq!(res.get::<u32>(), Some(&99));
    }

    #[test]
    fn test_resource_remove() {
        let mut res = RuntimeResources::new();
        res.insert(7u32);
        assert!(res.contains::<u32>());
        let val = res.remove::<u32>();
        assert_eq!(val, Some(7));
        assert!(!res.contains::<u32>());
        // remove again returns None
        assert!(res.remove::<u32>().is_none());
    }

    #[test]
    fn test_resource_missing_returns_none() {
        let res = RuntimeResources::new();
        assert!(res.get::<u32>().is_none());
        assert!(!res.contains::<u32>());
    }

    // Plugin that inserts a counter resource in step().
    struct CounterWriter {
        value: u32,
    }

    impl RuntimePlugin for CounterWriter {
        fn priority(&self) -> i32 { phase::ANIMATE }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            ctx.resources.insert(self.value);
        }
    }

    // Plugin that reads the counter resource and records it.
    struct CounterReader {
        recorded: Arc<Mutex<Vec<u32>>>,
    }

    impl RuntimePlugin for CounterReader {
        fn priority(&self) -> i32 { phase::POST_SIM }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            if let Some(&v) = ctx.resources.get::<u32>() {
                self.recorded.lock().unwrap().push(v);
            }
        }
    }

    #[test]
    fn test_resource_shared_across_plugins_same_frame() {
        // CounterWriter runs at ANIMATE, CounterReader at POST_SIM.
        // The value written by CounterWriter must be visible to CounterReader in the same frame.
        let recorded = Arc::new(Mutex::new(Vec::<u32>::new()));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(CounterWriter { value: 42 })
            .with_plugin(CounterReader { recorded: recorded.clone() });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));

        let vals = recorded.lock().unwrap();
        assert_eq!(vals.as_slice(), &[42], "reader must see value written by writer in the same frame");
    }

    #[test]
    fn test_resource_persists_across_frames() {
        // A resource inserted in frame 1 must still be present in frame 2 if not removed.
        let mut runtime = ViewportRuntime::new()
            .with_plugin(CounterWriter { value: 10 });

        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();

        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));
        assert!(runtime.resources().contains::<u32>(), "resource must persist after step");
    }

    #[test]
    fn test_runtime_works_without_resources() {
        // A runtime with no plugins and no resource usage must compile and step correctly.
        let mut runtime = ViewportRuntime::new();
        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        let output = runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0));
        assert!(output.contact_events.is_empty());
        assert!(output.node_transform_ops.is_empty());
    }

    // ---- event bus tests ----------------------------------------------------

    // Two distinct event types to verify typed routing.
    #[derive(Debug, PartialEq)]
    struct GameplayEvent { id: u32 }

    #[derive(Debug, PartialEq)]
    struct DiagnosticsEvent { frame_ms: f32 }

    // Plugin that emits GameplayEvents.
    struct GameplayEmitter { count: u32 }

    impl RuntimePlugin for GameplayEmitter {
        fn priority(&self) -> i32 { phase::SIMULATE }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            for i in 0..self.count {
                ctx.output.events.emit(GameplayEvent { id: i });
            }
        }
    }

    // Plugin that emits DiagnosticsEvents.
    struct DiagnosticsEmitter;

    impl RuntimePlugin for DiagnosticsEmitter {
        fn priority(&self) -> i32 { phase::POST_SIM }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            ctx.output.events.emit(DiagnosticsEvent { frame_ms: ctx.dt * 1000.0 });
        }
    }

    // Plugin that reads GameplayEvents from the bus during its own step.
    struct GameplayReader {
        seen: Arc<Mutex<Vec<u32>>>,
    }

    impl RuntimePlugin for GameplayReader {
        fn priority(&self) -> i32 { phase::WRITEBACK }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            for ev in ctx.output.events.read::<GameplayEvent>() {
                self.seen.lock().unwrap().push(ev.id);
            }
        }
    }

    fn run_one_frame(runtime: &mut ViewportRuntime) -> RuntimeOutput {
        let camera = Camera::default();
        let input = ActionFrame::default();
        let mut scene = Scene::new();
        let mut sel = Selection::new();
        runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input, 1.0 / 60.0))
    }

    #[test]
    fn test_event_bus_emit_and_read() {
        let mut bus = RuntimeEventBus::new();
        bus.emit(GameplayEvent { id: 1 });
        bus.emit(GameplayEvent { id: 2 });
        let ids: Vec<u32> = bus.read::<GameplayEvent>().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_event_bus_typed_isolation() {
        // Reading a different type returns nothing even when other types have events.
        let mut bus = RuntimeEventBus::new();
        bus.emit(GameplayEvent { id: 42 });
        assert!(!bus.has::<DiagnosticsEvent>());
        assert_eq!(bus.count::<DiagnosticsEvent>(), 0);
        let diag: Vec<_> = bus.read::<DiagnosticsEvent>().collect();
        assert!(diag.is_empty());
    }

    #[test]
    fn test_event_bus_drain() {
        let mut bus = RuntimeEventBus::new();
        bus.emit(GameplayEvent { id: 7 });
        bus.emit(GameplayEvent { id: 8 });
        let drained = bus.drain::<GameplayEvent>();
        assert_eq!(drained, vec![GameplayEvent { id: 7 }, GameplayEvent { id: 8 }]);
        // Second drain returns empty.
        assert!(bus.drain::<GameplayEvent>().is_empty());
        assert!(!bus.has::<GameplayEvent>());
    }

    #[test]
    fn test_event_bus_has_and_count() {
        let mut bus = RuntimeEventBus::new();
        assert!(!bus.has::<GameplayEvent>());
        assert_eq!(bus.count::<GameplayEvent>(), 0);
        bus.emit(GameplayEvent { id: 1 });
        bus.emit(GameplayEvent { id: 2 });
        assert!(bus.has::<GameplayEvent>());
        assert_eq!(bus.count::<GameplayEvent>(), 2);
    }

    #[test]
    fn test_event_bus_is_empty() {
        let mut bus = RuntimeEventBus::new();
        assert!(bus.is_empty());
        bus.emit(GameplayEvent { id: 0 });
        assert!(!bus.is_empty());
    }

    #[test]
    fn test_events_emitted_by_plugin_visible_in_output() {
        let mut runtime = ViewportRuntime::new()
            .with_plugin(GameplayEmitter { count: 3 });
        let output = run_one_frame(&mut runtime);
        assert_eq!(output.events.count::<GameplayEvent>(), 3);
        let ids: Vec<u32> = output.events.read::<GameplayEvent>().map(|e| e.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_events_typed_routing_across_plugins() {
        // GameplayEmitter (SIMULATE) emits GameplayEvents; GameplayReader (WRITEBACK)
        // reads them in the same frame via ctx.output.events.
        let seen = Arc::new(Mutex::new(Vec::<u32>::new()));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(GameplayEmitter { count: 2 })
            .with_plugin(GameplayReader { seen: seen.clone() });
        run_one_frame(&mut runtime);
        let ids = seen.lock().unwrap();
        assert_eq!(ids.as_slice(), &[0, 1]);
    }

    #[test]
    fn test_multiple_event_types_do_not_interfere() {
        // Both GameplayEmitter and DiagnosticsEmitter emit; each type is independent.
        let mut runtime = ViewportRuntime::new()
            .with_plugin(GameplayEmitter { count: 2 })
            .with_plugin(DiagnosticsEmitter);
        let output = run_one_frame(&mut runtime);
        assert_eq!(output.events.count::<GameplayEvent>(), 2);
        assert_eq!(output.events.count::<DiagnosticsEvent>(), 1);
    }

    #[test]
    fn test_events_cleared_each_frame() {
        // Events from frame N must not appear in frame N+1.
        let mut runtime = ViewportRuntime::new()
            .with_plugin(GameplayEmitter { count: 1 });
        run_one_frame(&mut runtime);
        let output2 = run_one_frame(&mut runtime);
        // The second frame also emits 1 event, but it must be exactly 1, not 2.
        assert_eq!(output2.events.count::<GameplayEvent>(), 1);
    }

    #[test]
    fn test_existing_output_fields_unaffected() {
        // Backward-compat: contact_events, selection_ops, node_transform_ops,
        // and camera_follow_target continue to work with no changes.
        let mut runtime = ViewportRuntime::new();
        let output = run_one_frame(&mut runtime);
        assert!(output.contact_events.is_empty());
        assert!(output.selection_ops.is_empty());
        assert!(output.node_transform_ops.is_empty());
        assert!(output.camera_follow_target.is_none());
        assert!(output.events.is_empty());
    }

    // ---- job handoff tests --------------------------------------------------

    use crate::runtime::jobs::{JobPoll, JobSlot};

    #[test]
    fn test_job_slot_empty_by_default() {
        let slot: JobSlot<u32> = JobSlot::empty();
        assert!(slot.is_empty());
        assert!(!slot.is_pending());
        assert!(!slot.is_ready());
    }

    #[test]
    fn test_job_slot_pending_after_new() {
        let (slot, _sender) = JobSlot::<u32>::new();
        assert!(slot.is_pending());
        assert!(!slot.is_empty());
        assert!(!slot.is_ready());
    }

    #[test]
    fn test_job_handoff_complete() {
        let (mut slot, sender) = JobSlot::<u32>::new();
        assert!(slot.is_pending());
        sender.complete(42);
        assert!(slot.is_ready());
        match slot.take() {
            JobPoll::Ready(v) => assert_eq!(v, 42),
            _ => panic!("expected Ready, got other variant"),
        }
        // Slot resets to empty after taking the result.
        assert!(slot.is_empty());
    }

    #[test]
    fn test_job_handoff_fail() {
        let (mut slot, sender) = JobSlot::<u32>::new();
        sender.fail("disk error");
        match slot.take() {
            JobPoll::Failed(msg) => assert_eq!(msg, "disk error"),
            _ => panic!("expected Failed"),
        }
        assert!(slot.is_empty());
    }

    #[test]
    fn test_job_handoff_cancel() {
        let (mut slot, sender) = JobSlot::<u32>::new();
        sender.cancel();
        match slot.take() {
            JobPoll::Cancelled => {}
            _ => panic!("expected Cancelled"),
        }
        assert!(slot.is_empty());
    }

    #[test]
    fn test_job_sender_drop_cancels() {
        let (mut slot, sender) = JobSlot::<u32>::new();
        drop(sender);
        match slot.take() {
            JobPoll::Cancelled => {}
            _ => panic!("expected Cancelled after sender drop"),
        }
        assert!(slot.is_empty());
    }

    #[test]
    fn test_job_take_on_empty_returns_empty() {
        let mut slot: JobSlot<u32> = JobSlot::empty();
        match slot.take() {
            JobPoll::Empty => {}
            _ => panic!("expected Empty"),
        }
    }

    #[test]
    fn test_job_take_while_pending_returns_pending() {
        let (mut slot, _sender) = JobSlot::<u32>::new();
        match slot.take() {
            JobPoll::Pending => {}
            _ => panic!("expected Pending"),
        }
        // Still pending after a non-consuming poll.
        assert!(slot.is_pending());
    }

    #[test]
    fn test_job_async_handoff_across_frames() {
        // Plugin holds a JobSlot. Frame 1: start job. Frame 2+: result arrives.
        struct LoaderPlugin {
            slot: JobSlot<u32>,
            result: Arc<Mutex<Option<u32>>>,
        }

        impl RuntimePlugin for LoaderPlugin {
            fn priority(&self) -> i32 { phase::PREPARE }

            fn submit(&mut self, _ctx: &RuntimeStepContext<'_>) {
                if self.slot.is_empty() {
                    let (new_slot, sender) = JobSlot::new();
                    self.slot = new_slot;
                    // Inline completion simulating a fast background thread.
                    sender.complete(99);
                }
            }

            fn collect(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
                if let JobPoll::Ready(v) = self.slot.take() {
                    *self.result.lock().unwrap() = Some(v);
                }
            }

            fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {}
        }

        let result = Arc::new(Mutex::new(None::<u32>));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(LoaderPlugin { slot: JobSlot::empty(), result: result.clone() });

        run_one_frame(&mut runtime);
        assert_eq!(*result.lock().unwrap(), Some(99));
    }

    #[test]
    fn test_job_staged_resource_update_ordering() {
        // A low-priority plugin integrates a job result into resources in collect.
        // A high-priority plugin's collect must NOT see it (runs first due to sort order).
        // A later-running plugin's collect DOES see it.

        struct Integrator {
            slot: JobSlot<u32>,
        }

        impl RuntimePlugin for Integrator {
            fn priority(&self) -> i32 { phase::PREPARE }  // low number = runs early in collect

            fn submit(&mut self, _ctx: &RuntimeStepContext<'_>) {
                if self.slot.is_empty() {
                    let (s, sender) = JobSlot::new();
                    self.slot = s;
                    sender.complete(77);
                }
            }

            fn collect(&mut self, ctx: &mut RuntimeStepContext<'_>) {
                if let JobPoll::Ready(v) = self.slot.take() {
                    ctx.resources.insert(v);
                }
            }

            fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {}
        }

        struct Reader {
            seen: Arc<Mutex<Option<u32>>>,
        }

        impl RuntimePlugin for Reader {
            fn priority(&self) -> i32 { phase::POST_SIM }  // higher number = runs later in collect

            fn collect(&mut self, ctx: &mut RuntimeStepContext<'_>) {
                *self.seen.lock().unwrap() = ctx.resources.get::<u32>().copied();
            }

            fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {}
        }

        let seen = Arc::new(Mutex::new(None::<u32>));
        let mut runtime = ViewportRuntime::new()
            .with_plugin(Integrator { slot: JobSlot::empty() })
            .with_plugin(Reader { seen: seen.clone() });

        run_one_frame(&mut runtime);
        // Integrator (PREPARE) integrates into resources in collect before Reader (POST_SIM) collect.
        assert_eq!(*seen.lock().unwrap(), Some(77));
    }

    #[test]
    fn test_job_slot_in_resources() {
        // A JobSlot stored in RuntimeResources is accessible to other plugins.
        let (slot, sender) = JobSlot::<u32>::new();
        let mut res = RuntimeResources::new();
        res.insert(slot);
        sender.complete(55);
        let found = res.get_mut::<JobSlot<u32>>().unwrap();
        match found.take() {
            JobPoll::Ready(v) => assert_eq!(v, 55),
            _ => panic!("expected Ready"),
        }
    }

    // ---- skeleton / skinning tests ------------------------------------------

    use crate::resources::SkinWeights;
    use crate::runtime::plugins::skeleton_plugin::{
        Joint, JointMatrices, Pose, Skeleton, SkeletonPlugin, apply_skin,
    };

    fn two_joint_skeleton() -> Skeleton {
        Skeleton::new(vec![
            Joint {
                name: "root".into(),
                parent: None,
                inverse_bind: glam::Affine3A::IDENTITY,
            },
            Joint {
                name: "child".into(),
                parent: Some(0),
                inverse_bind: glam::Affine3A::from_translation(-glam::Vec3::Y),
            },
        ])
    }

    fn single_vertex_weights(joint: u8, weight: f32) -> SkinWeights {
        SkinWeights {
            joint_indices: vec![[joint, 0, 0, 0]],
            joint_weights: vec![[weight, 0.0, 0.0, 0.0]],
        }
    }

    #[test]
    fn test_skeleton_joint_count() {
        let sk = two_joint_skeleton();
        assert_eq!(sk.joint_count(), 2);
    }

    #[test]
    fn test_skeleton_find_joint() {
        let sk = two_joint_skeleton();
        assert_eq!(sk.find_joint("root"), Some(0));
        assert_eq!(sk.find_joint("child"), Some(1));
        assert_eq!(sk.find_joint("missing"), None);
    }

    #[test]
    fn test_pose_identity_transforms() {
        let pose = Pose::identity(3);
        assert_eq!(pose.joint_count(), 3);
        for t in &pose.local_transforms {
            assert_eq!(*t, glam::Affine3A::IDENTITY);
        }
    }

    #[test]
    fn test_joint_matrices_single_root_identity() {
        let sk = Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: glam::Affine3A::IDENTITY,
        }]);
        let pose = Pose::identity(1);
        let mats = JointMatrices::compute(&sk, &pose);
        let m = mats.as_slice()[0];
        // identity local * identity inverse_bind = identity
        assert!(m.matrix3.col(0).abs_diff_eq(glam::Vec3A::X, 1e-5));
        assert!(m.translation.abs_diff_eq(glam::Vec3A::ZERO, 1e-5));
    }

    #[test]
    fn test_joint_matrices_parent_child_chain() {
        // Root translated by (1,0,0) in local space.
        // Child at local identity, parent = 0.
        // child.inverse_bind offsets by -Y.
        // Expected child world = root_world * child_local = translate(1,0,0)
        // Final matrix = world * inverse_bind = translate(1,0,0) * translate(0,-1,0) = translate(1,-1,0)
        let sk = two_joint_skeleton();
        let mut pose = Pose::identity(2);
        pose.local_transforms[0] = glam::Affine3A::from_translation(glam::Vec3::X);
        let mats = JointMatrices::compute(&sk, &pose);
        let child = mats.as_slice()[1];
        let expected_translation = glam::Vec3A::new(1.0, -1.0, 0.0);
        assert!(
            child.translation.abs_diff_eq(expected_translation, 1e-5),
            "child translation was {:?}, expected {:?}",
            child.translation,
            expected_translation,
        );
    }

    #[test]
    fn test_apply_skin_identity_pose() {
        let sk = Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: glam::Affine3A::IDENTITY,
        }]);
        let pose = Pose::identity(1);
        let mats = JointMatrices::compute(&sk, &pose);

        let positions = vec![[1.0f32, 2.0, 3.0]];
        let normals = vec![[0.0f32, 1.0, 0.0]];
        let weights = single_vertex_weights(0, 1.0);
        let (out_pos, out_nrm) = apply_skin(&positions, &normals, &weights, &mats);

        assert!((out_pos[0][0] - 1.0).abs() < 1e-5);
        assert!((out_pos[0][1] - 2.0).abs() < 1e-5);
        assert!((out_pos[0][2] - 3.0).abs() < 1e-5);
        assert!((out_nrm[0][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_skin_single_joint_full_weight() {
        // Joint 0 translates by (5,0,0). Vertex at origin must move to (5,0,0).
        let sk = Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: glam::Affine3A::IDENTITY,
        }]);
        let mut pose = Pose::identity(1);
        pose.local_transforms[0] = glam::Affine3A::from_translation(glam::Vec3::new(5.0, 0.0, 0.0));
        let mats = JointMatrices::compute(&sk, &pose);

        let positions = vec![[0.0f32, 0.0, 0.0]];
        let normals = vec![[1.0f32, 0.0, 0.0]];
        let weights = single_vertex_weights(0, 1.0);
        let (out_pos, _) = apply_skin(&positions, &normals, &weights, &mats);
        assert!((out_pos[0][0] - 5.0).abs() < 1e-5);
        assert!(out_pos[0][1].abs() < 1e-5);
    }

    #[test]
    fn test_apply_skin_two_joint_blend() {
        // Two joints: joint 0 at (0,0,0), joint 1 at (10,0,0).
        // Vertex at origin, 50/50 blend -> output at (5,0,0).
        let sk = Skeleton::new(vec![
            Joint { name: "a".into(), parent: None, inverse_bind: glam::Affine3A::IDENTITY },
            Joint { name: "b".into(), parent: None, inverse_bind: glam::Affine3A::IDENTITY },
        ]);
        let mut pose = Pose::identity(2);
        pose.local_transforms[1] = glam::Affine3A::from_translation(glam::Vec3::new(10.0, 0.0, 0.0));
        let mats = JointMatrices::compute(&sk, &pose);

        let positions = vec![[0.0f32, 0.0, 0.0]];
        let normals = vec![[1.0f32, 0.0, 0.0]];
        let weights = SkinWeights {
            joint_indices: vec![[0, 1, 0, 0]],
            joint_weights: vec![[0.5, 0.5, 0.0, 0.0]],
        };
        let (out_pos, _) = apply_skin(&positions, &normals, &weights, &mats);
        assert!((out_pos[0][0] - 5.0).abs() < 1e-5, "expected 5.0, got {}", out_pos[0][0]);
    }

    #[test]
    fn test_apply_skin_normal_renormalized() {
        // Even with a uniform scale, output normals must have unit length.
        let sk = Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: glam::Affine3A::IDENTITY,
        }]);
        let mut pose = Pose::identity(1);
        pose.local_transforms[0] = glam::Affine3A::from_scale(glam::Vec3::splat(2.0));
        let mats = JointMatrices::compute(&sk, &pose);

        let positions = vec![[1.0f32, 0.0, 0.0]];
        let normals = vec![[1.0f32, 0.0, 0.0]];
        let weights = single_vertex_weights(0, 1.0);
        let (_, out_nrm) = apply_skin(&positions, &normals, &weights, &mats);
        let len = (out_nrm[0][0].powi(2) + out_nrm[0][1].powi(2) + out_nrm[0][2].powi(2)).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "normal length was {len}");
    }

    #[test]
    fn test_skeleton_plugin_writes_update_to_output() {
        let sk = Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: glam::Affine3A::IDENTITY,
        }]);
        let positions = vec![[0.0f32, 0.0, 0.0]];
        let normals = vec![[0.0f32, 1.0, 0.0]];
        let weights = single_vertex_weights(0, 1.0);
        let mesh_id = crate::resources::mesh_store::MeshId(0);

        let mut runtime = ViewportRuntime::new()
            .with_plugin(SkeletonPlugin::new(sk, mesh_id, positions, normals, weights));

        // Insert a pose so the plugin fires.
        runtime.resources_mut().insert(Pose::identity(1));

        let output = run_one_frame(&mut runtime);
        assert_eq!(output.skinned_mesh_updates.len(), 1);
        assert_eq!(output.skinned_mesh_updates[0].mesh_id, mesh_id);
    }

    #[test]
    fn test_skinned_mesh_updates_empty_without_pose() {
        let sk = Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: glam::Affine3A::IDENTITY,
        }]);
        let mesh_id = crate::resources::mesh_store::MeshId(0);
        let mut runtime = ViewportRuntime::new()
            .with_plugin(SkeletonPlugin::new(
                sk,
                mesh_id,
                vec![[0.0f32; 3]],
                vec![[0.0f32, 1.0, 0.0]],
                single_vertex_weights(0, 1.0),
            ));
        // No Pose inserted -> no update.
        let output = run_one_frame(&mut runtime);
        assert!(output.skinned_mesh_updates.is_empty());
    }

    #[test]
    fn test_skinned_mesh_updates_cleared_each_frame() {
        let sk = Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: glam::Affine3A::IDENTITY,
        }]);
        let mesh_id = crate::resources::mesh_store::MeshId(0);
        let mut runtime = ViewportRuntime::new()
            .with_plugin(SkeletonPlugin::new(
                sk,
                mesh_id,
                vec![[0.0f32; 3]],
                vec![[0.0f32, 1.0, 0.0]],
                single_vertex_weights(0, 1.0),
            ));
        runtime.resources_mut().insert(Pose::identity(1));

        run_one_frame(&mut runtime);
        let output2 = run_one_frame(&mut runtime);
        // Each frame produces exactly 1 update, not 2.
        assert_eq!(output2.skinned_mesh_updates.len(), 1);
    }
}

/// Per-frame scene orchestration layer.
///
/// Owns plugins, an optional fixed timestep accumulator, and a transform snapshot
/// table. Does not own the scene, selection, GPU resources, or window.
///
/// # Priority execution order
///
/// Each call to [`step`](Self::step) executes plugins in ascending priority order:
///
/// 1. `[i32::MIN, SELECT)` -- Prepare (100), Pick (200): once, at wall dt
/// 2. `[SELECT, MANIPULATE)` -- Select (300): once (built-in SelectionSystem runs first)
/// 3. `[MANIPULATE, SIMULATE)` -- Manipulate (400), Animate (500): once
/// 4. `[SIMULATE, POST_SIM)` -- Simulate (600): once per fixed step or once at wall dt
/// 5. `[POST_SIM, i32::MAX]` -- PostSim (700), Writeback (800): once at wall dt
pub struct ViewportRuntime {
    mode: SceneRuntimeMode,
    plugins: Vec<Box<dyn RuntimePlugin>>,
    fixed_timestep: Option<FixedTimestep>,
    snapshots: TransformSnapshotTable,
    step_index: u64,
    selection_system: Option<SelectionSystem>,
    manipulation_system: Option<ManipulationSystem>,
    camera_follow: Option<CameraFollow>,
    /// Typed resource registry shared across plugins each frame.
    resources: RuntimeResources,
    /// Node IDs present at the end of the previous frame.
    prev_node_ids: std::collections::HashSet<crate::interaction::selection::NodeId>,
    /// False on the very first step call; after that, lifecycle events are emitted.
    scene_initialized: bool,
}

impl Default for ViewportRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewportRuntime {
    /// Create a runtime with default settings and no plugins.
    pub fn new() -> Self {
        Self {
            mode: SceneRuntimeMode::default(),
            plugins: Vec::new(),
            fixed_timestep: None,
            snapshots: TransformSnapshotTable::new(),
            step_index: 0,
            selection_system: None,
            manipulation_system: None,
            camera_follow: None,
            resources: RuntimeResources::new(),
            prev_node_ids: std::collections::HashSet::new(),
            scene_initialized: false,
        }
    }

    /// Enable the built-in click-to-select system.
    ///
    /// The SelectionSystem runs before the `Select` phase each frame. It reads
    /// `RuntimeFrameContext::clicked`, `pick_hit`, and `shift_held` to produce
    /// SelectionOp entries in RuntimeOutput.
    pub fn with_selection_system(mut self) -> Self {
        self.selection_system = Some(SelectionSystem::new());
        self
    }

    /// Enable the built-in manipulation system.
    ///
    /// The ManipulationSystem runs before the `Manipulate` phase each frame. It
    /// drives G/R/S sessions and gizmo drag from RuntimeFrameContext inputs, writing
    /// transform changes via TransformWriteback.
    pub fn with_manipulation_system(mut self) -> Self {
        self.manipulation_system = Some(ManipulationSystem::new());
        self
    }

    /// True while the built-in manipulation system has an active G/R/S session or gizmo drag.
    ///
    /// Use this to suppress orbit camera movement while manipulating objects.
    pub fn is_manipulating(&self) -> bool {
        self.manipulation_system
            .as_ref()
            .map_or(false, |m| m.is_active())
    }

    /// Access the built-in manipulation system, if enabled.
    pub fn manipulation_system(&self) -> Option<&ManipulationSystem> {
        self.manipulation_system.as_ref()
    }

    /// Set the runtime mode.
    pub fn with_mode(mut self, mode: SceneRuntimeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Register a plugin. Plugins run in ascending priority order each frame.
    /// Plugins with equal priority run in registration order (stable sort).
    pub fn with_plugin(mut self, plugin: impl RuntimePlugin) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Enable fixed-timestep accumulation for physics plugins.
    ///
    /// When set, plugins in the `[SIMULATE, POST_SIM)` range run once per
    /// accumulated fixed step rather than once per frame at wall dt. All other
    /// priority ranges always run once per frame.
    pub fn with_fixed_timestep(mut self, ts: FixedTimestep) -> Self {
        self.fixed_timestep = Some(ts);
        self
    }

    /// Replace the fixed timestep at runtime. Resets the accumulator.
    ///
    /// Use this to change the simulation rate after construction without
    /// rebuilding the runtime or its plugins.
    pub fn set_fixed_timestep(&mut self, ts: FixedTimestep) {
        self.fixed_timestep = Some(ts);
    }

    /// Remove the fixed timestep, reverting to one simulate call per frame at wall dt.
    pub fn clear_fixed_timestep(&mut self) {
        self.fixed_timestep = None;
    }

    /// The current runtime mode.
    pub fn mode(&self) -> SceneRuntimeMode {
        self.mode
    }

    /// The transform snapshot table for interpolated rendering.
    ///
    /// Updated each frame during writeback. Pass [`alpha`](Self::alpha) as the
    /// blend factor to [`TransformSnapshotTable::interpolated`].
    pub fn snapshots(&self) -> &TransformSnapshotTable {
        &self.snapshots
    }

    /// Read access to the shared resource registry.
    ///
    /// Use this after `step` to inspect resources without running the frame loop.
    /// During the frame loop, access resources through `RuntimeStepContext::resources`.
    pub fn resources(&self) -> &RuntimeResources {
        &self.resources
    }

    /// Write access to the shared resource registry.
    ///
    /// Use this to pre-populate resources before the first `step`, or to inject
    /// resources from outside the plugin system (e.g. from the application).
    pub fn resources_mut(&mut self) -> &mut RuntimeResources {
        &mut self.resources
    }

    /// Blend factor for rendering interpolation between fixed steps.
    ///
    /// Returns `1.0` if no fixed timestep is configured (render directly from
    /// current scene transforms without interpolation).
    pub fn alpha(&self) -> f32 {
        self.fixed_timestep.as_ref().map_or(1.0, |ts| ts.alpha())
    }

    /// Monotonically increasing simulation step counter.
    ///
    /// Incremented once per simulate execution. With a fixed timestep this
    /// means it may increment multiple times per rendered frame. Wraps on overflow.
    pub fn step_index(&self) -> u64 {
        self.step_index
    }

    /// Set a camera follow binding (builder style).
    ///
    /// Each call to [`step`](Self::step) will compute a suggested camera center
    /// from the followed node and return it in [`RuntimeOutput::camera_follow_target`].
    pub fn with_camera_follow(mut self, follow: CameraFollow) -> Self {
        self.camera_follow = Some(follow);
        self
    }

    /// Update the camera follow binding at runtime.
    pub fn set_camera_follow(&mut self, follow: CameraFollow) {
        self.camera_follow = Some(follow);
    }

    /// Remove the camera follow binding. [`RuntimeOutput::camera_follow_target`]
    /// will be `None` after this call.
    pub fn clear_camera_follow(&mut self) {
        self.camera_follow = None;
    }

    /// The current camera follow binding.
    pub fn camera_follow(&self) -> Option<&CameraFollow> {
        self.camera_follow.as_ref()
    }

    /// Run one frame of the runtime.
    ///
    /// Plugins run in ascending priority order. The simulate range
    /// `[SIMULATE, POST_SIM)` runs once per accumulated fixed step (or once at
    /// `frame.dt` when no fixed timestep is configured). All other ranges run
    /// once per frame. Transform ops are flushed to `scene` and the snapshot
    /// table is updated after the writeback range.
    ///
    /// Selection ops produced by plugins are applied to `selection` before returning.
    /// Contact events and the applied transform ops are returned in [`RuntimeOutput`].
    pub fn step(
        &mut self,
        scene: &mut Scene,
        selection: &mut Selection,
        frame: &RuntimeFrameContext,
    ) -> RuntimeOutput {
        let mut output = RuntimeOutput::default();
        let mut writeback = TransformWriteback::default();

        // --- Lifecycle event detection ---------------------------------------
        // Collect current node IDs from the scene.
        let current_ids: std::collections::HashSet<crate::interaction::selection::NodeId> =
            scene.nodes().map(|n| n.id()).collect();

        if self.scene_initialized {
            // Diff against previous frame and dispatch events.
            let mut events: Vec<RuntimeEvent> = Vec::new();
            for &id in current_ids.difference(&self.prev_node_ids) {
                events.push(RuntimeEvent::NodeAdded(id));
            }
            for &id in self.prev_node_ids.difference(&current_ids) {
                events.push(RuntimeEvent::NodeRemoved(id));
            }

            if !events.is_empty() {
                // Take plugins out to avoid aliasing self while dispatching.
                let mut plugins = std::mem::take(&mut self.plugins);
                for event in &events {
                    for plugin in plugins.iter_mut() {
                        let mut ctx = RuntimeStepContext {
                            priority: plugin.priority(),
                            dt: frame.dt,
                            scene,
                            writeback: &mut writeback,
                            output: &mut output,
                            pick_hit: frame.pick_hit,
                            resources: &mut self.resources,
                        };
                        plugin.on_event(event, &mut ctx);
                    }
                }
                self.plugins = plugins;
            }
        } else {
            self.scene_initialized = true;
        }
        self.prev_node_ids = current_ids;

        // --- Sort plugins by priority (stable: preserves registration order within a band).
        self.plugins.sort_by_key(|p| p.priority());

        // Take plugins out so we can access self freely during the step loop.
        let mut plugins = std::mem::take(&mut self.plugins);

        // --- Submit pass: all plugins in priority order ----------------------
        for plugin in plugins.iter_mut() {
            let ctx = RuntimeStepContext {
                priority: plugin.priority(),
                dt: frame.dt,
                scene,
                writeback: &mut writeback,
                output: &mut output,
                pick_hit: frame.pick_hit,
                resources: &mut self.resources,
            };
            plugin.submit(&ctx);
        }

        // --- Step loop -------------------------------------------------------

        // Prepare + Pick: [MIN, SELECT)
        run_range(
            &mut plugins,
            i32::MIN,
            phase::SELECT,
            frame.dt,
            frame,
            scene,
            &mut writeback,
            &mut output,
            &mut self.resources,
        );

        // Built-in selection system runs before the Select range.
        if let Some(sel_sys) = &self.selection_system {
            sel_sys.step(frame, &mut output);
        }
        // Select: [SELECT, MANIPULATE)
        run_range(
            &mut plugins,
            phase::SELECT,
            phase::MANIPULATE,
            frame.dt,
            frame,
            scene,
            &mut writeback,
            &mut output,
            &mut self.resources,
        );

        // Built-in manipulation system runs before the Manipulate range.
        if self.manipulation_system.is_some() {
            let mut manip_sys = self.manipulation_system.take().unwrap();
            manip_sys.step(frame, scene, selection, &mut writeback, &mut output);
            self.manipulation_system = Some(manip_sys);
        }
        // Manipulate + Animate: [MANIPULATE, SIMULATE)
        run_range(
            &mut plugins,
            phase::MANIPULATE,
            phase::SIMULATE,
            frame.dt,
            frame,
            scene,
            &mut writeback,
            &mut output,
            &mut self.resources,
        );

        // Simulate: [SIMULATE, POST_SIM) -- once per fixed step or once at wall dt.
        if self.fixed_timestep.is_some() {
            let mut ts = self.fixed_timestep.take().unwrap();
            for step_dt in ts.advance(frame.dt) {
                run_range(
                    &mut plugins,
                    phase::SIMULATE,
                    phase::POST_SIM,
                    step_dt,
                    frame,
                    scene,
                    &mut writeback,
                    &mut output,
                    &mut self.resources,
                );
                self.step_index = self.step_index.wrapping_add(1);
            }
            self.fixed_timestep = Some(ts);
        } else {
            run_range(
                &mut plugins,
                phase::SIMULATE,
                phase::POST_SIM,
                frame.dt,
                frame,
                scene,
                &mut writeback,
                &mut output,
                &mut self.resources,
            );
            self.step_index = self.step_index.wrapping_add(1);
        }

        // PostSim + Writeback: [POST_SIM, MAX]
        run_range(
            &mut plugins,
            phase::POST_SIM,
            i32::MAX,
            frame.dt,
            frame,
            scene,
            &mut writeback,
            &mut output,
            &mut self.resources,
        );

        // --- Collect pass: all plugins in priority order ---------------------
        for plugin in plugins.iter_mut() {
            let mut ctx = RuntimeStepContext {
                priority: plugin.priority(),
                dt: frame.dt,
                scene,
                writeback: &mut writeback,
                output: &mut output,
                pick_hit: frame.pick_hit,
                resources: &mut self.resources,
            };
            plugin.collect(&mut ctx);
        }

        // Restore plugins.
        self.plugins = plugins;

        // --- Flush writeback -------------------------------------------------
        let ops = writeback.into_ops();
        for op in &ops {
            scene.set_local_transform(op.id, glam::Mat4::from(op.transform));
            self.snapshots.update(op.id, op.transform);
        }
        if !ops.is_empty() {
            scene.update_transforms();
        }
        output.node_transform_ops = ops;

        // Apply selection ops produced by plugins.
        for op in &output.selection_ops {
            op.apply_to(selection);
        }

        // Compute camera follow target.
        if let Some(CameraFollow::Node { id, offset, .. }) = &self.camera_follow {
            let id = *id;
            let offset = *offset;
            let alpha = self.alpha();
            let pos = self
                .snapshots
                .interpolated(id, alpha)
                .map(|t| glam::Vec3::from(t.translation))
                .or_else(|| {
                    scene
                        .node(id)
                        .map(|n| n.world_transform().col(3).truncate())
                });
            if let Some(pos) = pos {
                output.camera_follow_target = Some(pos + offset);
            }
        }

        output
    }
}
