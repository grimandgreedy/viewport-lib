//! Integration tests for the runtime shared resource registry.
//!
//! Covers the two-plugin coordination pattern: one plugin writes a typed resource
//! during its step, another reads it later in the same frame without any
//! application-level wiring between them.

use viewport_lib::runtime::{
    FixedTimestep, RuntimeFrameContext, RuntimePlugin, RuntimeStepContext, ViewportRuntime,
    plugin::phase,
};
use viewport_lib::camera::camera::Camera;
use viewport_lib::interaction::input::ActionFrame;
use viewport_lib::interaction::selection::Selection;
use viewport_lib::scene::scene::Scene;
use std::sync::{Arc, Mutex};

fn make_frame(camera: &Camera, input: &ActionFrame) -> RuntimeFrameContext {
    let mut frame = RuntimeFrameContext::default();
    frame.dt = 1.0 / 60.0;
    frame.camera = camera.clone();
    frame.viewport_size = glam::Vec2::new(800.0, 600.0);
    frame.input = input.clone();
    frame
}

fn step(runtime: &mut ViewportRuntime) {
    let camera = Camera::default();
    let input = ActionFrame::default();
    let mut scene = Scene::new();
    let mut sel = Selection::new();
    runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input));
}

// --- Shared state type -------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct PhysicsState {
    contact_count: u32,
    total_impulse: f32,
}

// --- Plugins -----------------------------------------------------------------

struct PhysicsWriter {
    contacts: u32,
}

impl RuntimePlugin for PhysicsWriter {
    fn priority(&self) -> i32 { phase::SIMULATE }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        ctx.resources.insert(PhysicsState {
            contact_count: self.contacts,
            total_impulse: self.contacts as f32 * 5.0,
        });
    }
}

struct AudioReader {
    triggered: Arc<Mutex<u32>>,
}

impl RuntimePlugin for AudioReader {
    fn priority(&self) -> i32 { phase::POST_SIM }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        if let Some(state) = ctx.resources.get::<PhysicsState>() {
            if state.contact_count > 0 {
                *self.triggered.lock().unwrap() += 1;
            }
        }
    }
}

// --- Tests -------------------------------------------------------------------

#[test]
fn resource_written_by_simulate_readable_by_post_sim() {
    let triggered = Arc::new(Mutex::new(0u32));
    let mut runtime = ViewportRuntime::new()
        .with_plugin(PhysicsWriter { contacts: 3 })
        .with_plugin(AudioReader { triggered: triggered.clone() });

    step(&mut runtime);

    assert_eq!(*triggered.lock().unwrap(), 1);
}

#[test]
fn resource_accessible_via_runtime_after_step() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(PhysicsWriter { contacts: 2 });

    step(&mut runtime);

    let state = runtime.resources().get::<PhysicsState>().unwrap();
    assert_eq!(state.contact_count, 2);
    assert_eq!(state.total_impulse, 10.0);
}

#[test]
fn resource_persists_across_frames() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(PhysicsWriter { contacts: 1 });

    step(&mut runtime);
    step(&mut runtime);

    assert!(runtime.resources().contains::<PhysicsState>());
}

#[test]
fn resource_not_triggered_when_zero_contacts() {
    let triggered = Arc::new(Mutex::new(0u32));
    let mut runtime = ViewportRuntime::new()
        .with_plugin(PhysicsWriter { contacts: 0 })
        .with_plugin(AudioReader { triggered: triggered.clone() });

    step(&mut runtime);

    assert_eq!(*triggered.lock().unwrap(), 0);
}

#[test]
fn resource_prepopulated_from_outside_visible_to_plugins() {
    // The app inserts a resource before the first step; the plugin should see it.
    struct ResourceReader {
        seen: Arc<Mutex<Option<u32>>>,
    }
    impl RuntimePlugin for ResourceReader {
        fn priority(&self) -> i32 { phase::PREPARE }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            *self.seen.lock().unwrap() = ctx.resources.get::<u32>().copied();
        }
    }

    let seen = Arc::new(Mutex::new(None::<u32>));
    let mut runtime = ViewportRuntime::new()
        .with_plugin(ResourceReader { seen: seen.clone() });
    runtime.resources_mut().insert(42u32);

    step(&mut runtime);

    assert_eq!(*seen.lock().unwrap(), Some(42));
}

#[test]
fn runtime_without_resources_works_unchanged() {
    let mut runtime = ViewportRuntime::new();
    let camera = Camera::default();
    let input = ActionFrame::default();
    let mut scene = Scene::new();
    let mut sel = Selection::new();
    let output = runtime.step(&mut scene, &mut sel, &make_frame(&camera, &input));
    assert!(output.contact_events.is_empty());
    assert!(output.node_transform_ops.is_empty());
}

#[test]
fn fixed_timestep_resource_reflects_last_step() {
    // With a fixed timestep, the writer runs multiple times per wall frame.
    // The resource after step() should reflect the last simulate call.
    struct StepCounter {
        count: u32,
    }
    impl RuntimePlugin for StepCounter {
        fn priority(&self) -> i32 { phase::SIMULATE }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            self.count += 1;
            ctx.resources.insert(self.count);
        }
    }

    let mut runtime = ViewportRuntime::new()
        .with_fixed_timestep(FixedTimestep::new(60.0))
        .with_plugin(StepCounter { count: 0 });

    // 30 fps wall clock -> 2 fixed steps per frame
    let camera = Camera::default();
    let input = ActionFrame::default();
    let mut scene = Scene::new();
    let mut sel = Selection::new();
    let mut fixed_frame = RuntimeFrameContext::default();
    fixed_frame.dt = 1.0 / 30.0;
    fixed_frame.camera = camera.clone();
    fixed_frame.viewport_size = glam::Vec2::new(800.0, 600.0);
    fixed_frame.input = input.clone();
    runtime.step(&mut scene, &mut sel, &fixed_frame);

    // 2 steps ran; the resource holds the count after the last step.
    assert_eq!(runtime.resources().get::<u32>().copied(), Some(2));
}
