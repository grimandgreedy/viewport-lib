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
pub mod mode;
pub mod output;
pub mod plugin;
/// Built-in animation, constraint, and physics plugins.
pub mod plugins;
pub mod snapshot;
/// Built-in interaction systems: SelectionSystem and ManipulationSystem.
pub mod systems;
pub mod timestep;

pub use camera_follow::CameraFollow;
pub use context::{RuntimeFrameContext, RuntimeStepContext, SimulationStepContext};
pub use mode::SceneRuntimeMode;
pub use output::{ContactEvent, NodeTransformOp, RuntimeOutput, SelectionOp, TransformWriteback};
pub use plugin::{phase, RuntimeEvent, RuntimePhase, RuntimePlugin};
pub use plugins::{
    AnimationPlugin, AnimationTrack, Constraint, ConstraintPlugin, Keyframe, PhysicsBody,
    PhysicsLitePlugin,
};
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
