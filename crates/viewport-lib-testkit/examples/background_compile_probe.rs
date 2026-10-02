//! Does compiling pipelines on a worker thread leave the main thread free to
//! render?
//!
//! One renderer draws a small scene on the main thread and its frame times are
//! taken twice: idle, and while a second renderer on the same device builds
//! five material plugins' full pipeline sets on a worker. Run it with the
//! driver's shader cache cold, or the worker finishes too soon to tell.
//!
//! ```bash
//! cargo run --release --example background_compile_probe
//! ```
use std::time::Instant;
use viewport_lib::{
    Camera, FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, ViewportRenderer, wgpu,
};
use viewport_lib_testkit::{DeviceProfile, device::headless_device_with_info};

#[path = "../../viewport-lib-examples/eframe/examples/plugins/surface_detail_plugin.rs"]
#[allow(dead_code)]
mod surface_detail_plugin;
#[path = "../../viewport-lib-examples/eframe/examples/plugins/toon_plugin.rs"]
#[allow(dead_code)]
mod toon_plugin;

fn main() {
    let profile =
        DeviceProfile::high_performance("background-compile-probe").with_recommended_features();
    let (device, queue, info) = headless_device_with_info(&profile).expect("no adapter");
    println!("adapter: {:?} / {}", info.backend, info.name);
    let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;

    // Main-thread renderer, fully warmed for its own frame.
    let mut main = ViewportRenderer::new(&device, fmt);
    let mesh = viewport_lib::primitives::sphere(1.0, 48, 24);
    let mesh_id = main
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [512.0, 512.0];
    let items: Vec<SceneRenderItem> = (0..4)
        .map(|i| {
            let mut it = SceneRenderItem::default();
            it.mesh_id = mesh_id;
            it.model =
                glam::Mat4::from_translation(glam::Vec3::new(i as f32 * 2.5 - 4.0, 0.0, 0.0))
                    .to_cols_array_2d();
            it
        })
        .collect();
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    for _ in 0..5 {
        let _ = main.render_offscreen(&device, &queue, &frame, 512, 512);
    }

    let frames = |n: usize, main: &mut ViewportRenderer| {
        let mut t: Vec<f32> = (0..n)
            .map(|_| {
                let s = Instant::now();
                let _ = main.render_offscreen(&device, &queue, &frame, 512, 512);
                s.elapsed().as_secs_f32() * 1000.0
            })
            .collect();
        t.sort_by(|a, b| a.total_cmp(b));
        (t[t.len() / 2], t[t.len() * 95 / 100], *t.last().unwrap())
    };
    let (p50, p95, max) = frames(200, &mut main);
    println!("idle:              p50 {p50:.2} ms  p95 {p95:.2} ms  max {max:.2} ms");

    // Worker: a second renderer on the same device builds five plugins' full sets.
    let dev = device.clone();
    let started = Instant::now();
    let worker = std::thread::spawn(move || {
        let mut other = ViewportRenderer::new(&dev, fmt);
        let r = other.resources_mut();
        let ids: Vec<_> = [
            r.register_material_plugin(&dev, &toon_plugin::ToonPlugin),
            r.register_material_plugin(&dev, &toon_plugin::RimPlugin),
            r.register_material_plugin(&dev, &surface_detail_plugin::DetailLayerPlugin),
            r.register_material_plugin(&dev, &surface_detail_plugin::ParallaxPlugin),
            r.register_material_plugin(&dev, &surface_detail_plugin::DissolvePlugin),
        ]
        .into_iter()
        .map(|i| i.unwrap())
        .collect();
        r.warm_material_plugin_pipelines(&dev, &ids);
        started.elapsed().as_secs_f32() * 1000.0
    });
    let mut during: Vec<f32> = Vec::new();
    while !worker.is_finished() {
        let s = Instant::now();
        let _ = main.render_offscreen(&device, &queue, &frame, 512, 512);
        during.push(s.elapsed().as_secs_f32() * 1000.0);
    }
    let worker_ms = worker.join().unwrap();
    during.sort_by(|a, b| a.total_cmp(b));
    println!(
        "while compiling:   p50 {:.2} ms  p95 {:.2} ms  max {:.2} ms  ({} frames, worker took {worker_ms:.0} ms)",
        during[during.len() / 2],
        during[during.len() * 95 / 100],
        during.last().unwrap(),
        during.len()
    );
}
