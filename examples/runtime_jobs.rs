//! Runtime jobs example.
//!
//! Demonstrates how a plugin uses [`JobSlot`] and [`JobSender`] to perform
//! background work and integrate the result on a later frame.
//!
//! No window or GPU required. Run with:
//!
//!     cargo run --example runtime_jobs
//!
//! # What this shows
//!
//! - A `LoaderPlugin` at the `PREPARE` phase holds a `JobSlot<Vec<u8>>`.
//! - In `submit`, if the slot is idle, it starts a background thread that
//!   simulates loading and calls `sender.complete(data)`.
//! - In `collect`, it polls the slot. When the result arrives it inserts a
//!   `LoadedAsset` into shared resources.
//! - A second plugin at `POST_SIM` reads the resource each frame to show when
//!   it becomes available.
//! - A third job demonstrates the `fail` path.
//! - A fourth job demonstrates drop-cancellation.

use std::thread;
use std::time::Duration;

use viewport_lib::{
    runtime::{
        JobPoll, JobSlot, RuntimeFrameContext, RuntimePlugin, RuntimeStepContext,
        ViewportRuntime,
        plugin::phase,
    },
    camera::camera::Camera,
    interaction::input::ActionFrame,
    interaction::selection::Selection,
    scene::scene::Scene,
};

// ---------------------------------------------------------------------------
// Shared resource: the result of a background load.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct LoadedAsset {
    data: Vec<u8>,
    loaded_on_frame: u64,
}

// ---------------------------------------------------------------------------
// LoaderPlugin: starts a background job in submit, integrates in collect.
// ---------------------------------------------------------------------------

struct LoaderPlugin {
    slot: JobSlot<Vec<u8>>,
    frame: u64,
    load_delay_ms: u64,
}

impl LoaderPlugin {
    fn new(load_delay_ms: u64) -> Self {
        Self {
            slot: JobSlot::empty(),
            frame: 0,
            load_delay_ms,
        }
    }
}

impl RuntimePlugin for LoaderPlugin {
    fn priority(&self) -> i32 {
        phase::PREPARE
    }

    fn submit(&mut self, _ctx: &RuntimeStepContext<'_>) {
        self.frame += 1;
        if self.slot.is_empty() {
            let (new_slot, sender) = JobSlot::new();
            self.slot = new_slot;
            let delay = self.load_delay_ms;
            let frame = self.frame;
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(delay));
                let data = vec![1u8, 2, 3, 4, 5];
                println!("  [loader] background thread finishing (started at frame {frame})");
                sender.complete(data);
            });
        }
    }

    fn collect(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        match self.slot.take() {
            JobPoll::Ready(data) => {
                let len = data.len();
                println!("  [loader] integrated {len} bytes into resources on frame {}", self.frame);
                ctx.resources.insert(LoadedAsset { data, loaded_on_frame: self.frame });
            }
            JobPoll::Pending => {}
            JobPoll::Failed(msg) => {
                println!("  [loader] job failed: {msg}");
            }
            JobPoll::Cancelled => {
                println!("  [loader] job was cancelled");
            }
            JobPoll::Empty => {}
        }
    }

    fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {}
}

// ---------------------------------------------------------------------------
// ReaderPlugin: reads the LoadedAsset resource each frame.
// ---------------------------------------------------------------------------

struct ReaderPlugin {
    frame: u64,
}

impl RuntimePlugin for ReaderPlugin {
    fn priority(&self) -> i32 {
        phase::POST_SIM
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        self.frame += 1;
        match ctx.resources.get::<LoadedAsset>() {
            Some(asset) => {
                println!("  [reader] frame {}: asset available ({} bytes, loaded on frame {})",
                    self.frame, asset.data.len(), asset.loaded_on_frame);
            }
            None => {
                println!("  [reader] frame {}: asset not yet available", self.frame);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn step_runtime(runtime: &mut ViewportRuntime, camera: &Camera, input: &ActionFrame) {
    let mut scene = Scene::new();
    let mut sel = Selection::new();
    runtime.step(&mut scene, &mut sel, &make_frame(camera, input, 1.0 / 60.0));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let camera = Camera::default();
    let input = ActionFrame::default();

    // --- Demo 1: fast job with a short frame delay ---------------------------
    println!("\n=== Demo 1: fast job (short delay between frames) ===");
    {
        let mut runtime = ViewportRuntime::new()
            .with_plugin(LoaderPlugin::new(0))
            .with_plugin(ReaderPlugin { frame: 0 });

        for i in 0..4 {
            println!("frame {}:", i + 1);
            step_runtime(&mut runtime, &camera, &input);
            thread::sleep(Duration::from_millis(5));
        }
    }

    // --- Demo 2: load that takes more than one frame -------------------------
    println!("\n=== Demo 2: slow job (completes after a few frames) ===");
    {
        let mut runtime = ViewportRuntime::new()
            .with_plugin(LoaderPlugin::new(50))
            .with_plugin(ReaderPlugin { frame: 0 });

        for i in 0..6 {
            println!("frame {}:", i + 1);
            step_runtime(&mut runtime, &camera, &input);
            // Simulate real frame pacing so the background thread has time to finish.
            thread::sleep(Duration::from_millis(20));
        }
    }

    // --- Demo 3: failure path ------------------------------------------------
    println!("\n=== Demo 3: job failure ===");
    {
        struct FailPlugin { slot: JobSlot<u32>, ran: bool }

        impl RuntimePlugin for FailPlugin {
            fn priority(&self) -> i32 { phase::PREPARE }

            fn submit(&mut self, _ctx: &RuntimeStepContext<'_>) {
                if !self.ran {
                    self.ran = true;
                    let (s, sender) = JobSlot::new();
                    self.slot = s;
                    // Signal failure synchronously so collect() sees it in the same frame.
                    sender.fail("file not found");
                }
            }

            fn collect(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
                match self.slot.take() {
                    JobPoll::Failed(msg) => println!("  [fail plugin] handled failure: {msg}"),
                    _ => {}
                }
            }

            fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {}
        }

        let mut runtime = ViewportRuntime::new()
            .with_plugin(FailPlugin { slot: JobSlot::empty(), ran: false });

        println!("frame 1:");
        step_runtime(&mut runtime, &camera, &input);
    }

    // --- Demo 4: drop-cancellation -------------------------------------------
    println!("\n=== Demo 4: sender dropped without signaling (auto-cancel) ===");
    {
        struct DropPlugin { slot: JobSlot<u32>, ran: bool }

        impl RuntimePlugin for DropPlugin {
            fn priority(&self) -> i32 { phase::PREPARE }

            fn submit(&mut self, _ctx: &RuntimeStepContext<'_>) {
                if !self.ran {
                    self.ran = true;
                    let (s, sender) = JobSlot::new();
                    self.slot = s;
                    // Drop the sender immediately without calling complete/fail/cancel.
                    drop(sender);
                }
            }

            fn collect(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
                match self.slot.take() {
                    JobPoll::Cancelled => println!("  [drop plugin] slot cancelled (sender dropped)"),
                    _ => {}
                }
            }

            fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {}
        }

        let mut runtime = ViewportRuntime::new()
            .with_plugin(DropPlugin { slot: JobSlot::empty(), ran: false });

        println!("frame 1:");
        step_runtime(&mut runtime, &camera, &input);
    }

    println!("\nDone.");
}
