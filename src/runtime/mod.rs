//! Scene runtime: per-frame orchestration layer between [`Scene`] and [`ViewportRenderer`].
//!
//! [`ViewportRuntime`] runs registered plugins in a defined phase order each frame,
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
pub use plugin::{RuntimePhase, RuntimePlugin};
pub use plugins::{
    AnimationPlugin, AnimationTrack, Constraint, ConstraintPlugin, Keyframe, PhysicsBody,
    PhysicsLitePlugin,
};
pub use snapshot::{TransformSnapshot, TransformSnapshotTable};
pub use systems::{ManipulationSystem, SelectionSystem};
pub use timestep::{FixedStepIter, FixedTimestep};

use crate::interaction::selection::Selection;
use crate::scene::scene::Scene;

/// Per-frame scene orchestration layer.
///
/// Owns plugins, an optional fixed timestep accumulator, and a transform snapshot
/// table. Does not own the scene, selection, GPU resources, or window.
///
/// # Phase execution order
///
/// Each call to [`step`](Self::step) executes phases in this order:
///
/// 1. `Prepare` -- once, at wall dt
/// 2. `Pick` -- once, at wall dt
/// 3. `Select` -- once, at wall dt (built-in SelectionSystem runs first if enabled)
/// 4. `Manipulate` -- once, at wall dt (built-in ManipulationSystem runs first if enabled)
/// 5. `Animate` -- once, at wall dt
/// 6. `Simulate` -- once per fixed step (may be 0, 1, or N times); once at wall
///    dt if no fixed timestep is configured
/// 7. `Writeback` -- once; the runtime then flushes transform ops to the scene
///    and updates the snapshot table
pub struct ViewportRuntime {
    mode: SceneRuntimeMode,
    plugins: Vec<Box<dyn RuntimePlugin>>,
    fixed_timestep: Option<FixedTimestep>,
    snapshots: TransformSnapshotTable,
    step_index: u64,
    selection_system: Option<SelectionSystem>,
    manipulation_system: Option<ManipulationSystem>,
    camera_follow: Option<CameraFollow>,
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

    /// Register a plugin. Plugins run in their declared [`RuntimePhase`] order.
    /// Within a phase, plugins run in registration order.
    pub fn with_plugin(mut self, plugin: impl RuntimePlugin) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Enable fixed-timestep accumulation for physics plugins.
    ///
    /// When set, the `Simulate` phase runs once per accumulated fixed step rather
    /// than once per frame at the wall dt. All other phases always run once per frame.
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

    /// Remove the fixed timestep, reverting to one `Simulate` call per frame at wall dt.
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
    /// Incremented once per `Simulate` execution. With a fixed timestep this
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
    /// Phases `Prepare` through `Animate` run once at `frame.dt`. `Simulate`
    /// runs once per accumulated fixed step (or once at `frame.dt` when no fixed
    /// timestep is configured). `Writeback` runs once; transform ops are then
    /// flushed to `scene` and the snapshot table is updated.
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

        // Phases before Select run unconditionally once per frame.
        for phase in [RuntimePhase::Prepare, RuntimePhase::Pick] {
            self.run_phase(phase, frame.dt, frame, scene, &mut writeback, &mut output);
        }

        // Built-in selection system runs before the Select phase.
        if let Some(sel_sys) = &self.selection_system {
            sel_sys.step(frame, &mut output);
        }
        self.run_phase(
            RuntimePhase::Select,
            frame.dt,
            frame,
            scene,
            &mut writeback,
            &mut output,
        );

        // Built-in manipulation system runs before the Manipulate phase.
        if self.manipulation_system.is_some() {
            // Take the system out temporarily to satisfy the borrow checker.
            let mut manip_sys = self.manipulation_system.take().unwrap();
            manip_sys.step(frame, scene, selection, &mut writeback, &mut output);
            self.manipulation_system = Some(manip_sys);
        }
        self.run_phase(
            RuntimePhase::Manipulate,
            frame.dt,
            frame,
            scene,
            &mut writeback,
            &mut output,
        );

        self.run_phase(
            RuntimePhase::Animate,
            frame.dt,
            frame,
            scene,
            &mut writeback,
            &mut output,
        );

        // Simulate: runs once per fixed step, or once at wall dt if no fixed timestep.
        if self.fixed_timestep.is_some() {
            // Take the timestep out so we can borrow self.plugins separately.
            let mut ts = self.fixed_timestep.take().unwrap();
            for step_dt in ts.advance(frame.dt) {
                self.run_phase(
                    RuntimePhase::Simulate,
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
            self.run_phase(
                RuntimePhase::Simulate,
                frame.dt,
                frame,
                scene,
                &mut writeback,
                &mut output,
            );
            self.step_index = self.step_index.wrapping_add(1);
        }

        // Writeback phase.
        self.run_phase(
            RuntimePhase::Writeback,
            frame.dt,
            frame,
            scene,
            &mut writeback,
            &mut output,
        );

        // Flush accumulated transform ops to the scene and snapshot table.
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

        // Compute camera follow target. Use the interpolated snapshot position so
        // the camera tracks the same point that the renderer draws, not the raw
        // post-step scene position. This avoids jitter at low simulation rates
        // when interpolation is enabled.
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

    /// Run all plugins registered for `phase`, in registration order.
    fn run_phase(
        &mut self,
        phase: RuntimePhase,
        dt: f32,
        frame: &RuntimeFrameContext,
        scene: &Scene,
        writeback: &mut TransformWriteback,
        output: &mut RuntimeOutput,
    ) {
        for plugin in &mut self.plugins {
            if plugin.phase() == phase {
                let mut ctx = RuntimeStepContext {
                    phase,
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
}
