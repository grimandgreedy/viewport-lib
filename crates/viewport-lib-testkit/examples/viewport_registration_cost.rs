//! What registering and driving extra viewports costs with no scene geometry.
//!
//! A CAD quad view or a split-screen layout registers several viewports through
//! `create_viewport`, and each one carries its own camera state and its own post
//! chain targets. This measures the three stages separately:
//!
//!   1. `create_viewport` itself.
//!   2. The first frame a viewport is drawn, which is where its render targets
//!      are allocated.
//!   3. The steady-state per-viewport frame cost.
//!
//! The scene is empty throughout: no meshes uploaded, no items submitted. Every
//! figure here is what a viewport costs before it draws any 3D content.
//!
//! Run with:
//!   VPL_BUILD_LOG=1 cargo run --release --example viewport-registration-cost   # from crates/viewport-lib-testkit

use std::time::Instant;

use viewport_lib as vpl;
use vpl::wgpu;
use vpl::{Camera, FrameData, RenderCamera, ViewportId, ViewportRenderer};

/// Viewport size, overridable with `VPL_W` / `VPL_H`: the per-viewport targets
/// scale with area, so this is the knob that matters for the memory figure.
fn size() -> (u32, u32) {
    let get = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    (get("VPL_W", 1280), get("VPL_H", 720))
}
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const VIEWPORTS: usize = 4;
const FRAMES: usize = 120;

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * p).round() as usize]
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Sum and print whatever render targets have been allocated since the last drain.
fn report_targets(stage: &str) {
    let targets = vpl::resources::build_log::drain_textures();
    if targets.is_empty() {
        return;
    }
    let total: u64 = targets.iter().map(|(_, b)| b).sum();
    println!(
        "    {stage}: {} render targets, {:.1} MiB",
        targets.len(),
        mib(total)
    );
    let mut sorted = targets.clone();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (label, bytes) in sorted.iter().take(24) {
        println!("      {label:<34} {:6.2} MiB", mib(*bytes));
    }
}

fn frame_for(vp: usize) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        let (w, h) = size();
        rc.aspect = w as f32 / h as f32;
        rc
    };
    let (w, h) = size();
    frame.camera.viewport_size = [w as f32, h as f32];
    frame.camera.viewport_index = vp;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.1, 0.1, 0.12, 1.0].into());
    // `direct` selects PipelineMode::Direct, the LDR passthrough that skips the
    // whole post chain, to see whether its targets are still allocated.
    if std::env::args().nth(1).as_deref() == Some("direct") {
        frame.effects.display.mode = vpl::PipelineMode::Direct;
    }
    frame
}

fn main() {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        #[cfg(feature = "wgpu30")]
        apply_limit_buckets: false,
    }))
    .expect("no wgpu adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("viewport-registration-cost"),
        required_limits: ViewportRenderer::recommended_device_limits(&adapter),
        required_features: ViewportRenderer::recommended_device_features(&adapter),
        ..Default::default()
    }))
    .expect("no wgpu device");

    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    // Viewport 0 exists implicitly: the single-viewport path uses slot 0, so
    // only the extra ones are registered here.
    let _ = vpl::resources::build_log::drain();
    let _ = vpl::resources::build_log::drain_textures();

    println!("adapter: {:?} / {}", info.backend, info.name);
    println!();
    println!("create_viewport");
    let mut ids: Vec<ViewportId> = Vec::new();
    for i in 0..VIEWPORTS {
        let t = Instant::now();
        let id = renderer.create_viewport(&device);
        let ms = t.elapsed().as_secs_f32() * 1000.0;
        println!("  viewport {i}: {ms:8.3} ms");
        ids.push(id);
    }
    report_targets("during create_viewport");
    println!("  (nothing above means create_viewport allocates no render targets)");

    // One colour target per viewport to render into.
    let views: Vec<wgpu::TextureView> = (0..VIEWPORTS)
        .map(|i| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("vp_target"),
                size: wgpu::Extent3d {
                    width: size().0,
                    height: size().1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let _ = i;
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        })
        .collect();
    let _ = vpl::resources::build_log::drain_textures();

    let frames: Vec<FrameData> = (0..VIEWPORTS).map(frame_for).collect();

    println!();
    println!("first frame per viewport (empty scene: this is the post chain allocating)");
    let mut steady: Vec<Vec<f32>> = vec![Vec::new(); VIEWPORTS];
    let mut scene_ms: Vec<f32> = Vec::new();

    for f in 0..FRAMES {
        let t_scene = Instant::now();
        let (scene_fx, _) = frames[0].effects.split();
        let token = renderer
            .owned()
            .prepare_scene(&device, &queue, &frames[0], &scene_fx);
        scene_ms.push(t_scene.elapsed().as_secs_f32() * 1000.0);

        for (i, id) in ids.iter().enumerate() {
            let t = Instant::now();
            renderer
                .owned()
                .prepare_viewport(&device, &queue, &token, *id, &frames[i]);
            let cmd = renderer
                .owned()
                .render_viewport(&device, &queue, &views[i], *id, &frames[i]);
            queue.submit(std::iter::once(cmd));
            let ms = t.elapsed().as_secs_f32() * 1000.0;
            if f == 0 {
                println!("  viewport {i}: {ms:8.3} ms");
                report_targets(&format!("viewport {i} targets"));
            } else {
                steady[i].push(ms);
            }
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
    }

    println!();
    println!("steady state, {} warm frames, empty scene", FRAMES - 1);
    let warm = (FRAMES - 1) / 4;
    let mut total_p50 = 0.0;
    for (i, mut v) in steady.into_iter().enumerate() {
        let v = {
            v.sort_by(f32::total_cmp);
            v[warm..].to_vec()
        };
        let p50 = percentile(&v, 0.5);
        total_p50 += p50;
        println!(
            "  viewport {i}: prepare_viewport + render_viewport  p50 {p50:6.3} ms   p95 {:6.3} ms",
            percentile(&v, 0.95)
        );
    }
    let mut sc: Vec<f32> = scene_ms[warm..].to_vec();
    sc.sort_by(f32::total_cmp);
    println!(
        "  prepare_scene (once, shared)                    p50 {:6.3} ms",
        percentile(&sc, 0.5)
    );
    println!(
        "  all {VIEWPORTS} viewports per frame                       {:6.3} ms",
        total_p50 + percentile(&sc, 0.5)
    );

    println!();
    println!(
        "shadow depth textures after {FRAMES} empty frames across {VIEWPORTS} viewports: {:.1} MiB",
        mib(renderer.shadow_allocation_bytes())
    );
}
