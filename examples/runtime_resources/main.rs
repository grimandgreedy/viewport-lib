//! Demonstrates shared runtime resources between two plugins.
//!
//! Run with:
//!   cargo run --example runtime-resources
//!
//! PhysicsState is inserted into the resource registry by SimulatePlugin each frame.
//! AudioPlugin reads it in the same frame to decide whether to trigger a sound.
//! Neither plugin knows about the other; the runtime coordinates them through
//! the shared resource.

use viewport_lib::runtime::{
    FixedTimestep, RuntimeFrameContext, RuntimePlugin, RuntimeStepContext, ViewportRuntime,
    plugin::phase,
};
use viewport_lib::camera::camera::Camera;
use viewport_lib::interaction::input::ActionFrame;
use viewport_lib::interaction::selection::Selection;
use viewport_lib::scene::scene::Scene;

// A resource written by SimulatePlugin and read by AudioPlugin.
#[derive(Debug)]
struct PhysicsState {
    contact_count: u32,
    total_impulse: f32,
}

// Simulates contacts and writes PhysicsState into shared resources.
struct SimulatePlugin {
    frame: u32,
}

impl RuntimePlugin for SimulatePlugin {
    fn priority(&self) -> i32 { phase::SIMULATE }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        self.frame += 1;
        // Simulate some contacts every other frame.
        let contacts = if self.frame % 2 == 0 { 3 } else { 0 };
        let impulse = contacts as f32 * 5.0;
        ctx.resources.insert(PhysicsState {
            contact_count: contacts,
            total_impulse: impulse,
        });
    }
}

// Reads PhysicsState and logs when contacts occur.
struct AudioPlugin {
    triggered: u32,
}

impl RuntimePlugin for AudioPlugin {
    fn priority(&self) -> i32 { phase::POST_SIM }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        if let Some(state) = ctx.resources.get::<PhysicsState>() {
            if state.contact_count > 0 {
                self.triggered += 1;
                println!(
                    "  AudioPlugin: {} contact(s), impulse {:.1} -> play sound (trigger #{})",
                    state.contact_count, state.total_impulse, self.triggered
                );
            }
        }
    }
}

fn main() {
    let mut runtime = ViewportRuntime::new()
        .with_fixed_timestep(FixedTimestep::new(60.0))
        .with_plugin(SimulatePlugin { frame: 0 })
        .with_plugin(AudioPlugin { triggered: 0 });

    let camera = Camera::default();
    let input = ActionFrame::default();
    let mut scene = Scene::new();
    let mut selection = Selection::new();

    let frame_dt = 1.0 / 30.0; // 30 fps wall clock, 60 Hz fixed timestep -> 2 steps/frame

    for frame in 1..=6 {
        println!("Frame {frame}:");
        runtime.step(&mut scene, &mut selection, &RuntimeFrameContext {
            dt: frame_dt,
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

        // Resources are still accessible after step() via runtime.resources().
        if let Some(state) = runtime.resources().get::<PhysicsState>() {
            println!("  runtime.resources: contact_count={}", state.contact_count);
        }
    }
}
