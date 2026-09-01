//! Per-plugin cost attribution, without linking any specific plugin.
//!
//! The runtime times each registered plugin's `step` / `pre_prepare` /
//! `post_paint` and exposes it through `ViewportRuntime::last_stats()`, keyed by
//! the plugin's `type_name()`. So this bench reads per-plugin timing generically:
//! a real app swaps the synthetic plugins below for wind / terrain / Hamilton and
//! gets the same per-plugin rows with no change here.
//!
//! It drives the runtime with a fixed `dt` (deterministic), sweeps a per-plugin
//! workload, and writes one CSV row per (plugin, seam, workload). Skips with a
//! message when no GPU adapter is present (the `pre_prepare` seam needs a device).
//!
//! Usage:
//!     cargo run --release --example plugin_bench -- --frames 200 --out plugin_bench.csv

use glam::Vec2;
use viewport_lib::runtime::plugin::phase;
use viewport_lib::runtime::{GpuFrameContext, GpuPlugin};
use viewport_lib::wgpu;
use viewport_lib::{
    Camera, RuntimeFrameContext, RuntimePlugin, RuntimeStepContext, Scene, Selection,
    ViewportRuntime,
};
use viewport_lib_testkit::headless_device;

/// A physics-like simulation plugin: integrates `bodies` particles each step.
/// Cost scales with the body count, like a real rigid-body solver.
struct PhysicsSim {
    pos: Vec<[f32; 3]>,
    vel: Vec<[f32; 3]>,
}
impl PhysicsSim {
    fn new(bodies: usize) -> Self {
        Self {
            pos: vec![[0.0; 3]; bodies],
            vel: vec![[0.1, 0.2, 0.3]; bodies],
        }
    }
}
impl RuntimePlugin for PhysicsSim {
    fn priority(&self) -> i32 {
        phase::SIMULATE
    }
    fn type_name(&self) -> &'static str {
        "physics"
    }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        let dt = ctx.dt;
        for (p, v) in self.pos.iter_mut().zip(self.vel.iter_mut()) {
            v[2] -= 9.81 * dt;
            for k in 0..3 {
                p[k] += v[k] * dt;
            }
            // A little extra arithmetic so the per-body cost is measurable.
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            v[0] *= 1.0 - 0.001 * speed * dt;
        }
    }
}

/// A cheap clock plugin (wind-style): constant tiny cost regardless of workload.
struct Clock {
    t: f32,
}
impl RuntimePlugin for Clock {
    fn priority(&self) -> i32 {
        phase::ANIMATE
    }
    fn type_name(&self) -> &'static str {
        "clock"
    }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        self.t += ctx.dt;
    }
}

/// A GPU-compute-style plugin whose `pre_prepare` CPU encode cost scales with a
/// workload (stand-in for writing per-frame slot params or instance data).
struct GpuCompute {
    scratch: Vec<f32>,
}
impl GpuCompute {
    fn new(work: usize) -> Self {
        Self {
            scratch: vec![0.0; work.max(1)],
        }
    }
}
impl GpuPlugin for GpuCompute {
    fn type_name(&self) -> &'static str {
        "gpu_compute"
    }
    fn pre_prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        for (i, s) in self.scratch.iter_mut().enumerate() {
            *s = (i as f32 + ctx.frame_index as f32).sin();
        }
        Vec::new()
    }
}

fn pct(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let idx = ((p * (v.len() - 1) as f32).round() as usize).min(v.len() - 1);
    v[idx]
}

fn main() {
    let mut frames = 200u32;
    let mut warmup = 30u32;
    let mut out = "plugin_bench.csv".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--frames" => frames = it.next().and_then(|s| s.parse().ok()).unwrap_or(frames),
            "--warmup" => warmup = it.next().and_then(|s| s.parse().ok()).unwrap_or(warmup),
            "--out" => out = it.next().unwrap_or(out),
            other => eprintln!("ignoring {other}"),
        }
    }

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let dt = 1.0 / 60.0; // fixed -> deterministic sub-stepping
    let camera = Camera::default();
    let workloads = [100usize, 1_000, 10_000, 50_000];

    let mut csv = String::from("plugin,seam,workload,frames,ms_p50,ms_p95\n");

    for &workload in &workloads {
        // Register the plugins. Swap these three lines for real consumer plugins
        // (wind / terrain / Hamilton) and the per-plugin rows below are unchanged.
        let mut runtime = ViewportRuntime::new()
            .with_plugin(PhysicsSim::new(workload))
            .with_plugin(Clock { t: 0.0 })
            .with_gpu_plugin(GpuCompute::new(workload));

        let mut scene = Scene::new();
        let mut sel = Selection::new();

        // Per-plugin, per-seam samples for this workload.
        let mut step_samples: std::collections::HashMap<&'static str, Vec<f32>> =
            std::collections::HashMap::new();
        let mut pre_samples: std::collections::HashMap<&'static str, Vec<f32>> =
            std::collections::HashMap::new();

        for f in 0..(warmup + frames) {
            let mut fc = RuntimeFrameContext::default();
            fc.dt = dt;
            fc.camera = camera.clone();
            fc.viewport_size = Vec2::new(1280.0, 720.0);
            runtime.step(&mut scene, &mut sel, &fc);
            // Read per-plugin step timing (link-free: keyed by type_name).
            let step_now: Vec<(&'static str, f32)> = runtime
                .last_stats()
                .step_ms
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();

            let gctx = GpuFrameContext::new(&camera, Vec2::new(1280.0, 720.0), dt, f as u64);
            let bufs = runtime.pre_prepare(&device, &queue, &gctx);
            let pre_now: Vec<(&'static str, f32)> = runtime
                .last_stats()
                .pre_prepare_ms
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            if !bufs.is_empty() {
                queue.submit(bufs);
                let _ = device.poll(wgpu::PollType::Poll);
            }

            if f < warmup {
                continue;
            }
            for (k, v) in step_now {
                step_samples.entry(k).or_default().push(v);
            }
            for (k, v) in pre_now {
                pre_samples.entry(k).or_default().push(v);
            }
        }

        for (plugin, mut s) in step_samples {
            csv.push_str(&format!(
                "{},step,{},{},{:.5},{:.5}\n",
                plugin,
                workload,
                frames,
                pct(&mut s, 0.50),
                pct(&mut s, 0.95)
            ));
        }
        for (plugin, mut s) in pre_samples {
            csv.push_str(&format!(
                "{},pre_prepare,{},{},{:.5},{:.5}\n",
                plugin,
                workload,
                frames,
                pct(&mut s, 0.50),
                pct(&mut s, 0.95)
            ));
        }
        println!("workload {workload}: done");
    }

    std::fs::write(&out, csv).expect("write csv");
    println!("done -> {out}");
}
