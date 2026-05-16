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

pub mod context;
pub mod mode;
pub mod output;
pub mod plugin;
pub mod snapshot;
pub mod timestep;

pub use context::{RuntimeFrameContext, RuntimeStepContext, SimulationStepContext};
pub use mode::SceneRuntimeMode;
pub use output::{ContactEvent, NodeTransformOp, RuntimeOutput, SelectionOp, TransformWriteback};
pub use plugin::{RuntimePhase, RuntimePlugin};
pub use snapshot::{TransformSnapshot, TransformSnapshotTable};
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
/// 3. `Select` -- once, at wall dt
/// 4. `Manipulate` -- once, at wall dt
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
        }
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

        // Phases that always run once per frame.
        for phase in [
            RuntimePhase::Prepare,
            RuntimePhase::Pick,
            RuntimePhase::Select,
            RuntimePhase::Manipulate,
            RuntimePhase::Animate,
        ] {
            self.run_phase(phase, frame.dt, scene, &mut writeback, &mut output);
        }

        // Simulate: runs once per fixed step, or once at wall dt if no fixed timestep.
        if self.fixed_timestep.is_some() {
            // Take the timestep out so we can borrow self.plugins separately.
            let mut ts = self.fixed_timestep.take().unwrap();
            for step_dt in ts.advance(frame.dt) {
                self.run_phase(
                    RuntimePhase::Simulate,
                    step_dt,
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

        output
    }

    /// Run all plugins registered for `phase`, in registration order.
    fn run_phase(
        &mut self,
        phase: RuntimePhase,
        dt: f32,
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
                };
                plugin.step(&mut ctx);
            }
        }
    }
}
