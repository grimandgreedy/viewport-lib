//! Measures the overlay tessellation cache: 2000 vector icons (a stroked star
//! with a drop shadow) that are either static or moving every frame with the
//! same shape, plus a set of static polylines. No scene geometry and no text.
//!
//! Per case it prints the p50 of the overlay prepare time and the number of
//! items tessellated per frame (the cache misses) over the measured frames.
//!
//! Run with:
//!   cargo run --release --example vector_icons   # from crates/viewport-lib-testkit

use std::time::Instant;

use viewport_lib as vpl;
use vpl::wgpu;
use vpl::{
    AnimTrack, Camera, Colour, FillRule, FrameData, OverlayAnimations, OverlayFill, OverlayFrame,
    OverlayPolylineItem, OverlayShape, OverlayShapeItem, OverlayStroke, RenderCamera, ShadowLayer,
    SubPath, ViewportRenderer,
};

const W: u32 = 1280;
const H: u32 = 800;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const WARMUP: usize = 20;
const FRAMES: usize = 120;
const ICONS: u32 = 2000;
const POLYLINES: u32 = 200;

fn percentile(samples: &mut Vec<f32>, p: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[((samples.len() - 1) as f32 * p).round() as usize]
}

fn star() -> OverlayShape {
    let pts: Vec<[f32; 2]> = (0..10)
        .map(|i| {
            let a = i as f32 * std::f32::consts::PI / 5.0 - std::f32::consts::FRAC_PI_2;
            let r = if i % 2 == 0 { 9.0 } else { 4.0 };
            [10.0 + a.cos() * r, 10.0 + a.sin() * r]
        })
        .collect();
    OverlayShape::Vector {
        subpaths: vec![SubPath::polygon(&pts)],
        fill_rule: FillRule::NonZero,
        stroke: Some(OverlayStroke::new(1.0, Colour::srgb(0.1, 0.1, 0.1, 1.0))),
        inverse: false,
    }
}

/// The icons and polylines. `moving` gives every icon a translate track, so
/// it moves each frame with the same shape.
fn overlays(moving: bool, time: f64) -> OverlayFrame {
    let mut frame = OverlayFrame::default();
    frame.time = time;
    let shape = star();
    for i in 0..ICONS {
        let f = i as f32;
        let x = (f * 0.618_034).fract() * (W as f32 - 20.0);
        let y = (f * 0.381_966).fract() * (H as f32 - 20.0);
        let mut icon = OverlayShapeItem::new(shape.clone(), [x, y], [20.0, 20.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.95, 0.75, 0.2, 1.0)))
            .with_shadow(ShadowLayer::new(
                Colour::srgb(0.0, 0.0, 0.0, 0.5),
                2.0,
                [1.0, 1.0],
            ));
        if moving {
            icon =
                icon.with_animations(OverlayAnimations::default().with_translate(AnimTrack::new(
                    0.0,
                    4.0,
                    [x, y],
                    [x + 40.0, y + 25.0],
                )));
        }
        frame.shapes.push(icon);
    }
    for i in 0..POLYLINES {
        let y = 4.0 + i as f32 * (H as f32 / POLYLINES as f32);
        let pts = (0..24)
            .map(|k| [k as f32 * 54.0, y + 3.0 * (k as f32 * 0.7 + i as f32).sin()])
            .collect();
        frame
            .polylines
            .push(OverlayPolylineItem::new(pts).with_thickness(1.5));
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
        label: Some("vector-icons"),
        required_limits: ViewportRenderer::recommended_device_limits(&adapter),
        required_features: ViewportRenderer::recommended_device_features(&adapter)
            | wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    }))
    .expect("no wgpu device");
    let mut renderer = ViewportRenderer::with_config(
        &device,
        &vpl::RendererConfig::new(FORMAT)
            .with_pipeline_compilation(vpl::PipelineCompilation::Blocking),
    );
    let colour = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = colour.create_view(&wgpu::TextureViewDescriptor::default());

    println!("adapter: {:?} / {}", info.backend, info.name);
    println!(
        "{:>8} {:>14} {:>16}",
        "icons", "prepare ovl", "tessellations"
    );
    for moving in [false, true] {
        let mut frame = FrameData::default();
        frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
        frame.camera.viewport_size = [W as f32, H as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some(Colour::linear(0.1, 0.1, 0.12, 1.0));
        frame.effects.display.mode = vpl::PipelineMode::Direct;
        let (mut prepare, mut tess) = (Vec::new(), Vec::new());
        for i in 0..(WARMUP + FRAMES) {
            frame.overlays = overlays(moving, i as f64 / 60.0);
            let t = Instant::now();
            let cmd = renderer.owned().render(&device, &queue, &view, &frame);
            queue.submit(std::iter::once(cmd));
            let _ = t.elapsed();
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            });
            let stats = renderer.last_frame_stats();
            if i >= WARMUP {
                prepare.push(stats.prepare_breakdown.overlay_ms);
                tess.push(stats.overlay_tessellations as f32);
            }
        }
        println!(
            "{:>8} {:>11.3} ms {:>16}",
            if moving { "moving" } else { "static" },
            percentile(&mut prepare, 0.5),
            percentile(&mut tess, 0.5)
        );
    }
}
