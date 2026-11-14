//! Measures the shape pipeline's own cost: a scatter of analytic overlay
//! markers (circles and squares) whose positions and colours change every
//! frame, swept over the marker count, with no scene geometry and no text.
//!
//! Per marker count it prints the p50 of the CPU record time, the overlay
//! prepare and the GPU overlay pass over the measured frames.
//!
//! Run with:
//!   cargo run --release --example marker_scatter   # from crates/viewport-lib-testkit

use std::time::Instant;

use viewport_lib as vpl;
use vpl::wgpu;
use vpl::{
    Camera, Colour, FrameData, OverlayFill, OverlayFrame, OverlayShape, OverlayShapeItem,
    RenderCamera, ViewportRenderer,
};

const W: u32 = 1280;
const H: u32 = 800;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const WARMUP: usize = 20;
const FRAMES: usize = 120;
const COUNTS: &[u32] = &[5_000, 20_000, 50_000];

fn percentile(samples: &mut Vec<f32>, p: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[((samples.len() - 1) as f32 * p).round() as usize]
}

/// `n` markers drifting on a noise field at time `t`.
fn markers(n: u32, t: f32) -> OverlayFrame {
    let mut frame = OverlayFrame::default();
    frame.shapes.reserve(n as usize);
    for i in 0..n {
        let f = i as f32;
        let x = (f * 0.618_034).fract() * W as f32 + 6.0 * (t * 0.7 + f * 0.01).sin();
        let y = (f * 0.381_966).fract() * H as f32 + 6.0 * (t * 0.9 + f * 0.013).cos();
        let shape = if i % 3 == 0 {
            OverlayShape::Rect { corner_radius: 1.0 }
        } else {
            OverlayShape::Circle
        };
        let hue = (f * 0.0007 + t * 0.05).fract();
        frame.shapes.push(
            OverlayShapeItem::new(shape, [x - 3.0, y - 3.0], [6.0, 6.0]).with_fill(
                OverlayFill::Solid(Colour::srgb(
                    0.3 + 0.7 * (hue * 6.283).sin().abs(),
                    0.3 + 0.6 * (hue * 6.283 + 2.1).sin().abs(),
                    0.4 + 0.6 * (hue * 6.283 + 4.2).sin().abs(),
                    0.85,
                )),
            ),
        );
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
        label: Some("marker-scatter"),
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
        "{:>8} {:>12} {:>14} {:>14}",
        "markers", "record p50", "prepare ovl", "gpu ovl"
    );
    for &n in COUNTS {
        let mut frame = FrameData::default();
        frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
        frame.camera.viewport_size = [W as f32, H as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some(Colour::linear(0.1, 0.1, 0.12, 1.0));
        frame.effects.display.mode = vpl::PipelineMode::Direct;
        let (mut record, mut prepare, mut gpu) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..(WARMUP + FRAMES) {
            frame.overlays = markers(n, i as f32 / 60.0);
            let t = Instant::now();
            let cmd = renderer.owned().render(&device, &queue, &view, &frame);
            queue.submit(std::iter::once(cmd));
            let ms = t.elapsed().as_secs_f32() * 1000.0;
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            });
            let stats = renderer.last_frame_stats();
            if i >= WARMUP {
                record.push(ms);
                prepare.push(stats.prepare_breakdown.overlay_ms);
                gpu.push(stats.gpu_breakdown.overlay_ms);
            }
        }
        println!(
            "{n:>8} {:>9.3} ms {:>11.3} ms {:>11.3} ms",
            percentile(&mut record, 0.5),
            percentile(&mut prepare, 0.5),
            percentile(&mut gpu, 0.5)
        );
    }
}
