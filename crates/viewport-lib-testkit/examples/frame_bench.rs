//! Split-axis GPU frame benchmark.
//!
//! Renders a bounded matrix of cells headlessly with GPU timestamp queries and
//! writes one CSV row per cell. The matrix is not a cross-product: it is a
//! baseline cell plus single-axis sweeps (object count, instancing, per-mesh
//! triangles, camera motion, render path) plus a realism cell, so a regression
//! is attributable to one axis. `scripts/bench_compare.py` diffs the CSV against
//! a committed `benches/baseline.json`, per cell, never averaged.
//!
//! Usage:
//!     cargo run --release --example frame_bench -- --frames 120 --out frame_bench.csv
//!
//! Skips with a message when no GPU adapter is present. GPU-ms columns are empty
//! on backends without TIMESTAMP_QUERY.

use glam::Vec3;
use std::f32::consts::TAU;
use viewport_lib::wgpu;
use viewport_lib::{
    BackfacePolicy, Camera, FrameData, Material, MeshId, SceneRenderItem, ViewportRenderer,
    primitives,
};
use viewport_lib_testkit::{BuildCtx, BuiltScene, frame_for, orbit_camera, rigs, scene_by_name};

#[derive(Clone, Copy, PartialEq)]
enum CameraMode {
    Stationary,
    Orbit,
}

struct Cell {
    id: &'static str,
    family: &'static str,
    mesh_class: &'static str,
    tris: u64,
    count: u32,
    material: &'static str,
    camera: CameraMode,
    hdr: bool,
    centre: Vec3,
    distance: f32,
    build: Box<dyn Fn(&mut BuildCtx<'_>) -> BuiltScene>,
}

fn cube_grid(mesh: MeshId, count: u32, instanced: bool) -> Vec<SceneRenderItem> {
    let side = (count as f64).cbrt().ceil() as i32;
    let mut items = Vec::with_capacity(count as usize);
    'outer: for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let mut it = SceneRenderItem::default();
                it.mesh_id = mesh;
                it.model =
                    glam::Mat4::from_translation(Vec3::new(x as f32, y as f32, z as f32) * 1.5)
                        .to_cols_array_2d();
                it.material = Material::pbr([0.7, 0.6, 0.5], 0.2, 0.5);
                if !instanced {
                    it.material.backface_policy = BackfacePolicy::Tint(0.4);
                }
                items.push(it);
                if items.len() == count as usize {
                    break 'outer;
                }
            }
        }
    }
    items
}

fn cube_cell(id: &'static str, count: u32, instanced: bool, hdr: bool, camera: CameraMode) -> Cell {
    let extent = (count as f64).cbrt().ceil() as f32 * 1.5;
    Cell {
        id,
        family: if instanced { "instanced" } else { "per_object" },
        mesh_class: "convex",
        tris: 12,
        count,
        material: "pbr",
        camera,
        hdr,
        centre: Vec3::ZERO,
        distance: extent * 2.5 + 6.0,
        build: Box::new(move |ctx| {
            let cube = ctx
                .res
                .upload_mesh_data(ctx.device, &primitives::cube(0.6))
                .expect("cube");
            BuiltScene {
                items: cube_grid(cube, count, instanced),
                lighting: rigs::from_above(),
                background: None,
                ..Default::default()
            }
        }),
    }
}

fn mesh_cell(id: &'static str, sub: u32, tris: u64) -> Cell {
    Cell {
        id,
        family: "single",
        mesh_class: "convex",
        tris,
        count: 1,
        material: "pbr",
        camera: CameraMode::Orbit,
        hdr: false,
        centre: Vec3::ZERO,
        distance: 5.0,
        build: Box::new(move |ctx| {
            let m = ctx
                .res
                .upload_mesh_data(ctx.device, &primitives::icosphere(1.5, sub))
                .expect("icosphere");
            let mut it = SceneRenderItem::default();
            it.mesh_id = m;
            it.material = Material::pbr([0.7, 0.6, 0.85], 0.3, 0.4);
            BuiltScene {
                items: vec![it],
                lighting: rigs::from_above(),
                background: None,
                ..Default::default()
            }
        }),
    }
}

fn game_cell() -> Cell {
    let gb = scene_by_name("game_mix").expect("game_mix").build;
    Cell {
        id: "game_mix",
        family: "mixed",
        mesh_class: "mixed",
        tris: 0,
        count: 0,
        material: "mixed",
        camera: CameraMode::Orbit,
        hdr: false,
        centre: Vec3::new(0.0, 0.0, 1.0),
        distance: 32.0,
        build: Box::new(move |ctx| gb(ctx)),
    }
}

fn cells() -> Vec<Cell> {
    vec![
        // baseline: 1000 instanced cubes, orbiting, LDR.
        cube_cell("baseline", 1000, true, false, CameraMode::Orbit),
        // object-count sweep (instancing held at instanced).
        cube_cell("count_100", 100, true, false, CameraMode::Orbit),
        cube_cell("count_10000", 10000, true, false, CameraMode::Orbit),
        // instancing axis: per-object at the baseline count.
        cube_cell("per_object_1000", 1000, false, false, CameraMode::Orbit),
        // per-mesh triangle sweep (single mesh, count = 1).
        mesh_cell("tris_1280", 3, 1280),
        mesh_cell("tris_20480", 5, 20480),
        mesh_cell("tris_81920", 6, 81920),
        // camera-motion axis: stationary vs the orbiting baseline.
        cube_cell(
            "camera_stationary",
            1000,
            true,
            false,
            CameraMode::Stationary,
        ),
        // render-path axis: HDR at the baseline.
        cube_cell("path_hdr", 1000, true, true, CameraMode::Orbit),
        // realism cell.
        game_cell(),
    ]
}

fn camera_for(cell: &Cell, f: u32, frames: u32) -> Camera {
    let yaw = match cell.camera {
        CameraMode::Stationary => 0.6,
        CameraMode::Orbit => (f as f32 / frames.max(1) as f32) * TAU,
    };
    orbit_camera(cell.centre, cell.distance, yaw, 1.0)
}

fn pct(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let idx = ((p * (v.len() - 1) as f32).round() as usize).min(v.len() - 1);
    v[idx]
}

struct Args {
    frames: u32,
    warmup: u32,
    width: u32,
    height: u32,
    out: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        frames: 120,
        warmup: 30,
        width: 1280,
        height: 720,
        out: "frame_bench.csv".to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--frames" => a.frames = it.next().and_then(|s| s.parse().ok()).unwrap_or(a.frames),
            "--warmup" => a.warmup = it.next().and_then(|s| s.parse().ok()).unwrap_or(a.warmup),
            "--width" => a.width = it.next().and_then(|s| s.parse().ok()).unwrap_or(a.width),
            "--height" => a.height = it.next().and_then(|s| s.parse().ok()).unwrap_or(a.height),
            "--out" => a.out = it.next().unwrap_or(a.out),
            other => eprintln!("ignoring unknown arg {other}"),
        }
    }
    a
}

fn init_device() -> Option<(wgpu::Device, wgpu::Queue, String, bool)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let name = adapter.get_info().name;
    let has_ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let mut wanted = wgpu::Features::empty();
    if has_ts {
        wanted |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if adapter
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        wanted |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("frame-bench"),
        required_features: wanted,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue, name, has_ts))
}

fn main() {
    let args = parse_args();
    let Some((device, queue, gpu_name, has_ts)) = init_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    if !has_ts {
        eprintln!("warning: TIMESTAMP_QUERY unsupported; GPU-ms columns will be 0");
    }

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame_bench_target"),
        size: wgpu::Extent3d {
            width: args.width,
            height: args.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let cells = cells();
    println!(
        "frame_bench: {} cells, {}x{}, {} frames each ({} warmup) on {} -> {}",
        cells.len(),
        args.width,
        args.height,
        args.frames,
        args.warmup,
        gpu_name,
        args.out
    );

    let mut csv = String::new();
    csv.push_str(
        "gpu,cell,family,mesh_class,tris,count,material,camera,hdr,frames,\
         gpu_ms_p50,gpu_ms_p95,gpu_ms_p99,scene_ms_p50,shadow_ms_p50,oit_ms_p50,post_ms_p50,\
         cull_ms_p50,prepare_ms_p50,paint_ms_p50,total_ms_p50,total_ms_p95,\
         draw_calls,instanced_batches,triangles,visible,per_object_items\n",
    );

    for (i, cell) in cells.iter().enumerate() {
        let built = {
            let mut ctx = BuildCtx {
                res: renderer.resources_mut(),
                device: &device,
                queue: &queue,
            };
            (cell.build)(&mut ctx)
        };
        renderer.force_dirty();

        let mut gpu = Vec::new();
        let mut scene = Vec::new();
        let mut shadow = Vec::new();
        let mut oit = Vec::new();
        let mut post = Vec::new();
        let mut cull = Vec::new();
        let mut prepare = Vec::new();
        let mut paint = Vec::new();
        let mut total = Vec::new();
        let mut counters = (0u32, 0u32, 0u64, 0u32, 0u32);

        for f in 0..(args.warmup + args.frames) {
            let cam = camera_for(cell, f, args.frames);
            let mut fd: FrameData =
                frame_for(&built, &cam, [args.width as f32, args.height as f32]);
            fd.effects.display.mode = if cell.hdr {
                viewport_lib::PipelineMode::Hdr
            } else {
                viewport_lib::PipelineMode::Direct
            };
            renderer.render_to_texture(&device, &queue, &view, &fd);
            if f % 16 == 0 {
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(5)),
                });
            } else {
                let _ = device.poll(wgpu::PollType::Poll);
            }
            if f < args.warmup {
                continue;
            }
            let s = renderer.last_frame_stats();
            if let Some(g) = s.gpu_frame_ms {
                gpu.push(g);
            }
            scene.push(s.gpu_breakdown.scene_ms);
            shadow.push(s.gpu_breakdown.shadow_ms);
            oit.push(s.gpu_breakdown.oit_ms);
            post.push(s.gpu_breakdown.post_ms);
            cull.push(s.gpu_breakdown.cull_ms);
            prepare.push(s.cpu_prepare_ms);
            paint.push(s.cpu_paint_ms);
            total.push(s.total_frame_ms);
            counters = (
                s.draw_calls,
                s.instanced_batches,
                s.triangles_submitted,
                s.visible_objects,
                s.per_object_items,
            );
        }

        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},\
             {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
             {},{},{},{},{}\n",
            gpu_name,
            cell.id,
            cell.family,
            cell.mesh_class,
            cell.tris,
            cell.count,
            cell.material,
            if cell.camera == CameraMode::Orbit {
                "orbit"
            } else {
                "stationary"
            },
            cell.hdr as u8,
            args.frames,
            pct(&mut gpu, 0.50),
            pct(&mut gpu, 0.95),
            pct(&mut gpu, 0.99),
            pct(&mut scene, 0.50),
            pct(&mut shadow, 0.50),
            pct(&mut oit, 0.50),
            pct(&mut post, 0.50),
            pct(&mut cull, 0.50),
            pct(&mut prepare, 0.50),
            pct(&mut paint, 0.50),
            pct(&mut total, 0.50),
            pct(&mut total, 0.95),
            counters.0,
            counters.1,
            counters.2,
            counters.3,
            counters.4,
        ));
        println!("[{}/{}] {}", i + 1, cells.len(), cell.id);
    }

    std::fs::write(&args.out, csv).expect("write csv");
    println!("done -> {}", args.out);
}
