//! Integration tests for the generic runtime event bus.
//!
//! Covers typed emission, same-frame cross-plugin routing, multi-type isolation,
//! per-frame clearing, and the chain pattern where one plugin reads events
//! emitted by an earlier plugin and emits its own in response.

use viewport_lib::runtime::{
    FixedTimestep, RuntimeFrameContext, RuntimePlugin, RuntimeStepContext, ViewportRuntime,
    plugin::phase,
};
use viewport_lib::camera::camera::Camera;
use viewport_lib::interaction::input::ActionFrame;
use viewport_lib::interaction::selection::Selection;
use viewport_lib::scene::scene::Scene;
use std::sync::{Arc, Mutex};

fn make_frame<'a>(camera: &'a Camera, input: &'a ActionFrame) -> RuntimeFrameContext<'a> {
    RuntimeFrameContext {
        dt: 1.0 / 60.0,
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

fn step(runtime: &mut ViewportRuntime) -> viewport_lib::runtime::RuntimeOutput {
    let camera = Camera::default();
    let input = ActionFrame::default();
    let mut scene = Scene::new();
    let mut sel = Selection::new();
    runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input))
}

// --- Event types -------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct TriggerEntered { trigger_id: u32, actor_id: u32 }

#[derive(Debug, PartialEq)]
struct AudioCue { trigger_id: u32 }

#[derive(Debug)]
struct FrameDiagnostics { #[allow(dead_code)] step_dt_ms: f32 }

// --- Plugins -----------------------------------------------------------------

struct TriggerPlugin { fires: bool }

impl RuntimePlugin for TriggerPlugin {
    fn priority(&self) -> i32 { phase::SIMULATE }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        if self.fires {
            ctx.output.events.emit(TriggerEntered { trigger_id: 1, actor_id: 42 });
        }
    }
}

// Reads TriggerEntered events and re-emits an AudioCue for each.
struct AudioPlugin;

impl RuntimePlugin for AudioPlugin {
    fn priority(&self) -> i32 { phase::POST_SIM }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        let ids: Vec<u32> = ctx.output.events
            .read::<TriggerEntered>()
            .map(|e| e.trigger_id)
            .collect();
        for id in ids {
            ctx.output.events.emit(AudioCue { trigger_id: id });
        }
    }
}

struct DiagnosticsPlugin;

impl RuntimePlugin for DiagnosticsPlugin {
    fn priority(&self) -> i32 { phase::WRITEBACK }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        ctx.output.events.emit(FrameDiagnostics { step_dt_ms: ctx.dt * 1000.0 });
    }
}

// Collects events of type T into a shared vec during its step.
struct EventCollector<T: Send + 'static> {
    log: Arc<Mutex<Vec<T>>>,
    priority: i32,
}

impl<T: Send + 'static + Clone> RuntimePlugin for EventCollector<T> {
    fn priority(&self) -> i32 { self.priority }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        let items: Vec<T> = ctx.output.events.read::<T>().cloned().collect();
        self.log.lock().unwrap().extend(items);
    }
}

// --- Tests -------------------------------------------------------------------

#[test]
fn events_visible_in_output_after_step() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(TriggerPlugin { fires: true });

    let output = step(&mut runtime);

    assert_eq!(output.events.count::<TriggerEntered>(), 1);
    let ev = output.events.read::<TriggerEntered>().next().unwrap();
    assert_eq!(ev.trigger_id, 1);
    assert_eq!(ev.actor_id, 42);
}

#[test]
fn event_chain_across_plugins_same_frame() {
    // TriggerPlugin emits TriggerEntered; AudioPlugin reads it and emits AudioCue.
    // Both are visible in the returned output.
    let mut runtime = ViewportRuntime::new()
        .with_plugin(TriggerPlugin { fires: true })
        .with_plugin(AudioPlugin);

    let output = step(&mut runtime);

    assert_eq!(output.events.count::<TriggerEntered>(), 1);
    assert_eq!(output.events.count::<AudioCue>(), 1);
    assert_eq!(output.events.read::<AudioCue>().next().unwrap().trigger_id, 1);
}

#[test]
fn event_types_are_isolated() {
    // DiagnosticsPlugin only emits FrameDiagnostics; TriggerEntered must be absent.
    let mut runtime = ViewportRuntime::new()
        .with_plugin(DiagnosticsPlugin);

    let output = step(&mut runtime);

    assert!(!output.events.has::<TriggerEntered>());
    assert!(output.events.has::<FrameDiagnostics>());
}

#[test]
fn no_trigger_means_no_audio_cue() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(TriggerPlugin { fires: false })
        .with_plugin(AudioPlugin);

    let output = step(&mut runtime);

    assert!(!output.events.has::<TriggerEntered>());
    assert!(!output.events.has::<AudioCue>());
}

#[test]
fn events_cleared_between_frames() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(TriggerPlugin { fires: true });

    step(&mut runtime); // frame 1: emits 1 event
    let output2 = step(&mut runtime); // frame 2: also emits 1 event

    // Must be exactly 1 from this frame, not 2 accumulated across frames.
    assert_eq!(output2.events.count::<TriggerEntered>(), 1);
}

#[test]
fn drain_returns_events_and_clears_slot() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(TriggerPlugin { fires: true });

    let mut output = step(&mut runtime);

    let drained = output.events.drain::<TriggerEntered>();
    assert_eq!(drained.len(), 1);
    assert!(output.events.drain::<TriggerEntered>().is_empty());
}

#[test]
fn multiple_events_same_type_accumulate() {
    struct MultiEmitter;
    impl RuntimePlugin for MultiEmitter {
        fn priority(&self) -> i32 { phase::SIMULATE }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            ctx.output.events.emit(TriggerEntered { trigger_id: 1, actor_id: 10 });
            ctx.output.events.emit(TriggerEntered { trigger_id: 2, actor_id: 20 });
            ctx.output.events.emit(TriggerEntered { trigger_id: 3, actor_id: 30 });
        }
    }

    let mut runtime = ViewportRuntime::new().with_plugin(MultiEmitter);
    let output = step(&mut runtime);

    assert_eq!(output.events.count::<TriggerEntered>(), 3);
    let ids: Vec<u32> = output.events.read::<TriggerEntered>().map(|e| e.trigger_id).collect();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[test]
fn collector_plugin_reads_events_mid_frame() {
    // EventCollector at WRITEBACK must see events emitted at SIMULATE.
    let log: Arc<Mutex<Vec<TriggerEntered>>> = Arc::new(Mutex::new(Vec::new()));
    let mut runtime = ViewportRuntime::new()
        .with_plugin(TriggerPlugin { fires: true })
        .with_plugin(EventCollector::<TriggerEntered> {
            log: log.clone(),
            priority: phase::WRITEBACK,
        });

    step(&mut runtime);

    let collected = log.lock().unwrap();
    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0].trigger_id, 1);
}

#[test]
fn diagnostics_emitted_once_per_wall_frame_regardless_of_fixed_timestep() {
    // DiagnosticsPlugin runs at WRITEBACK (outside the fixed-step range).
    // It must fire exactly once per wall frame even with multiple fixed steps.
    let mut runtime = ViewportRuntime::new()
        .with_fixed_timestep(FixedTimestep::new(60.0))
        .with_plugin(DiagnosticsPlugin);

    let camera = Camera::default();
    let input = ActionFrame::default();
    let mut scene = Scene::new();
    let mut sel = Selection::new();

    // 30 fps wall clock -> 2 fixed steps, but WRITEBACK runs once.
    let output = runtime.step(&mut scene, &mut sel, &RuntimeFrameContext {
        dt: 1.0 / 30.0,
        camera: &camera,
        viewport_size: glam::Vec2::new(800.0, 600.0),
        input: &input,
        pick_hit: None,
        clicked: false,
        drag_started: false,
        dragging: false,
        pointer_delta: glam::Vec2::ZERO,
        cursor_viewport: None,
        shift_held: false,
    });

    assert_eq!(output.events.count::<FrameDiagnostics>(), 1);
}

#[test]
fn existing_output_fields_unaffected_by_event_bus() {
    let mut runtime = ViewportRuntime::new();
    let output = step(&mut runtime);
    assert!(output.contact_events.is_empty());
    assert!(output.selection_ops.is_empty());
    assert!(output.node_transform_ops.is_empty());
    assert!(output.camera_follow_target.is_none());
    assert!(output.events.is_empty());
}
