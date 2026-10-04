//! Does compiling pipelines on the workers leave the render thread free?
//!
//! One renderer under `PipelineCompilation::Background` draws a small scene
//! and its frame times are taken twice: idle, and while the workers build
//! five material plugins' full pipeline sets after a warm-up call. Run it
//! with the driver's shader cache cold, or the workers finish too soon to
//! tell.
//!
//! ```bash
//! cargo run --release --example background_compile_probe
//! ```
use std::time::Instant;
use viewport_lib::{
    Camera, FrameData, PipelineCompilation, RenderCamera, SceneRenderItem, SurfaceSubmission,
    ViewportRenderer, wgpu,
};
use viewport_lib_testkit::{DeviceProfile, device::headless_device_with_info};

#[path = "../../viewport-lib-examples/eframe/examples/plugins/surface_detail_plugin.rs"]
#[allow(dead_code)]
mod surface_detail_plugin;
#[path = "../../viewport-lib-examples/eframe/examples/plugins/toon_plugin.rs"]
#[allow(dead_code)]
mod toon_plugin;

/// Draw `frame` the way a presented frame is drawn, so the compilation policy
/// applies (`render_offscreen` compiles blocking whatever the policy), and wait
/// for the GPU so the time covers the frame.
fn draw(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    frame: &FrameData,
) {
    renderer.render_to_texture(device, queue, view, frame);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}

/// A render target the size of the frame, in the renderer's format.
fn target(device: &wgpu::Device, format: wgpu::TextureFormat, size: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("live_target"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn main() {
    let profile =
        DeviceProfile::high_performance("background-compile-probe").with_recommended_features();
    let (device, queue, info) = headless_device_with_info(&profile).expect("no adapter");
    println!("adapter: {:?} / {}", info.backend, info.name);
    let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;

    let mut renderer = ViewportRenderer::new(&device, fmt);
    renderer.set_pipeline_compilation(PipelineCompilation::Background);
    let mesh = viewport_lib::primitives::sphere(1.0, 48, 24);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();
    let ids: Vec<_> = {
        let r = renderer.resources_mut();
        [
            r.register_material_plugin(&device, &toon_plugin::ToonPlugin),
            r.register_material_plugin(&device, &toon_plugin::RimPlugin),
            r.register_material_plugin(&device, &surface_detail_plugin::DetailLayerPlugin),
            r.register_material_plugin(&device, &surface_detail_plugin::ParallaxPlugin),
            r.register_material_plugin(&device, &surface_detail_plugin::DissolvePlugin),
        ]
        .into_iter()
        .map(|i| i.unwrap())
        .collect()
    };

    // A small built-in scene, fully warmed for its own frame.
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
    let view = target(&device, fmt, 512);
    for _ in 0..5 {
        draw(&mut renderer, &device, &queue, &view, &frame);
    }

    let summary = |mut t: Vec<f32>| {
        t.sort_by(|a, b| a.total_cmp(b));
        (t[t.len() / 2], t[t.len() * 95 / 100], *t.last().unwrap())
    };
    let idle: Vec<f32> = (0..200)
        .map(|_| {
            let s = Instant::now();
            draw(&mut renderer, &device, &queue, &view, &frame);
            s.elapsed().as_secs_f32() * 1000.0
        })
        .collect();
    let (p50, p95, max) = summary(idle);
    println!("idle:              p50 {p50:.2} ms  p95 {p95:.2} ms  max {max:.2} ms");

    // The warm-up returns at once; the workers build the five sets while the
    // same renderer keeps drawing its scene.
    let started = Instant::now();
    renderer
        .resources_mut()
        .warm_material_plugin_pipelines(&device, &ids);
    let handed_over = renderer.pipelines_pending();
    let mut during: Vec<f32> = Vec::new();
    while renderer.pipelines_pending() > 0 {
        let s = Instant::now();
        draw(&mut renderer, &device, &queue, &view, &frame);
        during.push(s.elapsed().as_secs_f32() * 1000.0);
    }
    let workers_ms = started.elapsed().as_secs_f32() * 1000.0;
    let frames = during.len();
    let (p50, p95, max) = summary(during);
    println!(
        "while compiling:   p50 {p50:.2} ms  p95 {p95:.2} ms  max {max:.2} ms  ({frames} frames, {handed_over} pipelines, workers took {workers_ms:.0} ms)"
    );
}
