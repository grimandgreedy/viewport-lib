//! What a full warm-up costs: `warm_pipelines(PipelineSet::all())` on a
//! renderer with every built-in item type installed, as a loading screen
//! would call it.
//!
//! ```bash
//! cargo run --release --example warm_up_cost
//! ```
//!
//! Prints how long the call took to return, how many compiles were still
//! pending when it did, and how long until none were. Under `Blocking` the
//! call does all the work; under `Background` it returns once the modules and
//! layouts exist and the workers do the rest. Pick the policy with
//! `VPL_PIPELINE_COMPILATION=blocking|background`; the default follows the
//! platform. `VPL_DETAIL=1` lists the ten most expensive objects.

use std::time::Instant;

use viewport_lib::resources::build_log;
use viewport_lib::{PipelineSet, ViewportRenderer, wgpu};
use viewport_lib_testkit::{DeviceProfile, device::headless_device_with_info};

fn ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}

fn report(what: &str, took: f32) {
    let mut builds = build_log::drain();
    // An empty f32 sum is -0.0.
    let total: f32 = builds.iter().map(|(_, ms)| ms).sum::<f32>() + 0.0;
    println!(
        "{what:<28} {took:9.2} ms, {:>4} pipelines/modules ({total:.2} ms of compile)",
        builds.len()
    );
    if std::env::var_os("VPL_DETAIL").is_some() {
        builds.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (label, ms) in builds.iter().take(10) {
            println!("    {label:<60} {ms:8.3} ms");
        }
    }
}

fn main() {
    build_log::enable();
    let profile = DeviceProfile::high_performance("warm-up-cost").with_recommended_features();
    let (device, queue, info) = headless_device_with_info(&profile).expect("no GPU adapter");
    println!(
        "adapter: {:?} / {} ({:?})",
        info.backend, info.name, info.device_type
    );

    let t = Instant::now();
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    viewport_lib_plugins::item_types::install(&mut renderer, &device);
    report("new + install", ms(t));
    println!("policy: {:?}", renderer.pipeline_compilation());

    let t = Instant::now();
    renderer.warm_pipelines(&device, &queue, &PipelineSet::all());
    let returned = ms(t);
    let pending = renderer.pipelines_pending();
    report("warm_pipelines returned", returned);
    println!("    pending when it returned: {pending}");

    renderer.wait_for_pipelines(&device);
    // Cumulative from the warm-up call; the builds are the workers'.
    report("until none pending", ms(t));
}
