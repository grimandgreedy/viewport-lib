//! Per-viewport memory and time across a range of viewport sizes.
//!
//! One viewport, resized through a list of common window and tile sizes. For
//! each size it reports what the per-viewport post-chain targets cost to
//! allocate (bytes and wall clock) and what a frame then costs to drive. The
//! scene is empty throughout: no meshes uploaded, no items submitted, every post
//! effect at its default, so these are the figures a viewport carries before it
//! draws any 3D content.
//!
//! Sizes are swept on one renderer after a warm-up frame, so the shared post
//! pipelines are already compiled and the allocation column is target allocation
//! alone rather than shader compilation.
//!
//! Run with:
//!   cargo run --release --example viewport-size-sweep   # from crates/viewport-lib-testkit
//!
//! Pass `direct` to sweep with `PipelineMode::Direct`, the LDR passthrough, to
//! compare what the path that runs no post chain still allocates.

use std::time::Instant;

use viewport_lib as vpl;
use vpl::wgpu;
use vpl::{Camera, FrameData, RenderCamera, ViewportRenderer};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const FRAMES: usize = 150;

/// (label, width, height)
const SIZES: &[(&str, u32, u32)] = &[
    ("640x480", 640, 480),
    ("960x540 (quad tile of 1080p)", 960, 540),
    ("1280x720", 1280, 720),
    ("1600x900", 1600, 900),
    ("1920x1080", 1920, 1080),
    ("1920x1200", 1920, 1200),
    ("2560x1440", 2560, 1440),
    ("3440x1440 (ultrawide)", 3440, 1440),
    ("3840x2160 (4K)", 3840, 2160),
    ("5120x2880 (5K)", 5120, 2880),
];

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * p).round() as usize]
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let direct = std::env::args().nth(1).as_deref() == Some("direct");

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
        label: Some("viewport-size-sweep"),
        required_limits: ViewportRenderer::recommended_device_limits(&adapter),
        required_features: ViewportRenderer::recommended_device_features(&adapter)
            | wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    }))
    .expect("no wgpu device");

    let mut renderer = ViewportRenderer::new(&device, FORMAT);

    // Warm-up frame at a throwaway size. The first frame any renderer draws
    // compiles the shared post-chain pipelines, tens of milliseconds that have
    // nothing to do with viewport size; without this the first row of the sweep
    // carries all of it and reads as though small viewports allocate slowly.
    {
        let (w, h) = (256u32, 256u32);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("warmup_target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut frame = FrameData::default();
        frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
        frame.camera.viewport_size = [w as f32, h as f32];
        if direct {
            frame.effects.display.mode = vpl::PipelineMode::Direct;
        }
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        });
    }
    let _ = vpl::resources::build_log::drain_textures();
    let _ = vpl::resources::build_log::drain();

    println!("adapter: {:?} / {}", info.backend, info.name);
    println!(
        "mode: {}, empty scene, all post effects default (off)",
        if direct {
            "Direct (LDR passthrough)"
        } else {
            "Hdr (default)"
        }
    );
    println!();
    println!(
        "{:<30} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "viewport size", "targets", "hdr+depth", "alloc ms", "cpu p50", "gpu p50", "gpu min"
    );
    println!("{}", "-".repeat(90));

    let mut rows: Vec<(&str, u64, u64)> = Vec::new();
    // Target bytes per effect group, one column per swept size.
    let mut by_group: Vec<[u64; GROUPS.len()]> = Vec::new();

    for (label, w, h) in SIZES {
        let (w, h) = (*w, *h);
        let colour = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sweep_target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
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
        // The output texture above is the consumer's own surface, not something
        // the renderer allocates; drop it from the log before the frame runs.
        let _ = vpl::resources::build_log::drain_textures();

        let mut frame = FrameData::default();
        frame.camera.render_camera = {
            let mut rc = RenderCamera::from_camera(&Camera::default());
            rc.aspect = w as f32 / h as f32;
            rc
        };
        frame.camera.viewport_size = [w as f32, h as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some([0.1, 0.1, 0.12, 1.0].into());
        if direct {
            frame.effects.display.mode = vpl::PipelineMode::Direct;
        }

        let mut cpu: Vec<f32> = Vec::with_capacity(FRAMES);
        let mut gpu: Vec<f32> = Vec::with_capacity(FRAMES);
        let mut alloc_ms = 0.0;
        let mut target_bytes = 0;
        // The HDR colour and depth pair, the only two targets a plain frame on
        // the HDR path needs. The gap to `target_bytes` is the effects.
        let mut minimum_bytes = 0;

        for f in 0..FRAMES {
            let t = Instant::now();
            let cmd = renderer.owned().render(&device, &queue, &view, &frame);
            queue.submit(std::iter::once(cmd));
            let ms = t.elapsed().as_secs_f32() * 1000.0;
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            });
            if f == 0 {
                // The resize dropped the previous size's targets and allocated
                // this size's, which is the whole of this frame's extra cost.
                let targets = vpl::resources::build_log::drain_textures();
                target_bytes = targets.iter().map(|(_, b)| b).sum();
                minimum_bytes = targets
                    .iter()
                    .filter(|(l, _)| l == "hdr_texture" || l == "hdr_depth_texture")
                    .map(|(_, b)| b)
                    .sum();
                alloc_ms = ms;
                let mut groups = [0u64; GROUPS.len()];
                for (l, b) in &targets {
                    groups[group_of(l)] += b;
                }
                by_group.push(groups);
            } else {
                cpu.push(ms);
                if let Some(g) = renderer.last_frame_stats().gpu_frame_ms {
                    gpu.push(g);
                }
            }
        }

        let warm = cpu.len() / 4;
        cpu.sort_by(f32::total_cmp);
        gpu.sort_by(f32::total_cmp);
        let cpu_p50 = percentile(&cpu[warm..], 0.5);
        let gpu_p50 = if gpu.is_empty() {
            0.0
        } else {
            percentile(&gpu[warm.min(gpu.len() - 1)..], 0.5)
        };

        println!(
            "{label:<30} {:>9.1}M {:>9.1}M {:>9.2} {:>9.3} {:>9.3} {:>9.3}",
            mib(target_bytes),
            mib(minimum_bytes),
            alloc_ms,
            cpu_p50,
            gpu_p50,
            gpu.first().copied().unwrap_or(0.0)
        );
        rows.push((label, target_bytes, minimum_bytes));
    }

    println!();
    println!("a four-viewport layout at each size (targets only), as allocated and at the floor:");
    for (label, bytes, minimum) in &rows {
        println!(
            "  {label:<30} {:>9.1} MiB   floor {:>8.1} MiB",
            mib(bytes * 4),
            mib(minimum * 4)
        );
    }

    // The same bytes cut by the effect that owns each target, at three sizes.
    println!();
    println!("targets by effect group (MiB):");
    let picks: Vec<usize> = ["1280x720", "1920x1080", "3840x2160 (4K)"]
        .iter()
        .filter_map(|want| SIZES.iter().position(|(l, _, _)| l == want))
        .collect();
    print!("  {:<16}", "group");
    for &i in &picks {
        print!(" {:>16}", SIZES[i].0);
    }
    println!();
    for (g, name) in GROUPS.iter().enumerate() {
        if picks.iter().all(|&i| by_group[i][g] == 0) {
            continue;
        }
        print!("  {name:<16}");
        for &i in &picks {
            print!(" {:>16.2}", mib(by_group[i][g]));
        }
        println!();
    }

    println!();
    println!(
        "shadow depth textures across the whole sweep: {:.1} MiB (empty scene casts nothing)",
        mib(renderer.shadow_allocation_bytes())
    );
}

/// Effect groups a per-viewport target can belong to. `scene` is the HDR
/// colour and depth pair; `other` catches a label this list does not know.
const GROUPS: [&str; 10] = [
    "scene",
    "bloom",
    "ssao",
    "dof",
    "contact shadow",
    "fxaa",
    "outline",
    "ssaa",
    "render scale",
    "other",
];

fn group_of(label: &str) -> usize {
    let name = if label == "hdr_texture" || label == "hdr_depth_texture" {
        "scene"
    } else if label.starts_with("bloom") {
        "bloom"
    } else if label.starts_with("ssao") {
        "ssao"
    } else if label.starts_with("dof") {
        "dof"
    } else if label.starts_with("contact_shadow") {
        "contact shadow"
    } else if label.starts_with("fxaa") {
        "fxaa"
    } else if label.starts_with("outline") {
        "outline"
    } else if label.starts_with("ssaa") {
        "ssaa"
    } else if label.starts_with("upscale") || label.starts_with("output_depth") {
        "render scale"
    } else {
        "other"
    };
    GROUPS.iter().position(|g| *g == name).unwrap()
}
