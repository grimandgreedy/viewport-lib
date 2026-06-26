//! Headless performance benchmark for GPU culling and HiZ occlusion.
//!
//! Renders a fixed, deterministic camera flight over a configured set of runs
//! and writes a timestamped CSV of per-segment GPU/CPU timings and the cull
//! breakdown. Frame timing comes from GPU timestamp queries (uncapped, no
//! vsync), so the numbers reflect real render cost rather than a vsync ceiling.
//!
//! Run:
//!   cargo run --release --example perf-bench
//!   cargo run --release --example perf-bench -- --count 64000 --frames 600
//!
//! Selecting runs: edit the `RUNS` array below. Each entry's first field is an
//! `enabled` flag. Set one to `true` and the rest to `false` to run just that
//! one; all `true` runs the full sweep.

use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use viewport_lib::{
    FrameData, ItemSettings, LightSource, LightingSettings, Material, MeshId, RenderCamera,
    SceneRenderItem, SurfaceSubmission, ViewportRenderer, primitives,
};

// ---------------------------------------------------------------------------
// Run matrix
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum SceneKind {
    /// Uniform box field (dense small instances, weak occlusion).
    Grid,
    /// Box field plus a large solid wall in front (strong occlusion case).
    Occluder,
}

#[derive(Clone, Copy, PartialEq)]
enum CullMode {
    /// GPU-driven culling off (direct instanced draw).
    None,
    /// GPU frustum cull only.
    Frustum,
    /// GPU frustum cull plus HiZ occlusion.
    FrustumHiz,
}

#[derive(Clone, Copy)]
struct Run {
    enabled: bool,
    scene: SceneKind,
    cull: CullMode,
    textured: bool,
    lit: bool,
    shadows: bool,
}

const fn run(
    enabled: bool,
    scene: SceneKind,
    cull: CullMode,
    textured: bool,
    lit: bool,
    shadows: bool,
) -> Run {
    Run {
        enabled,
        scene,
        cull,
        textured,
        lit,
        shadows,
    }
}

use CullMode::{Frustum, FrustumHiz, None as NoCull};
use SceneKind::{Grid, Occluder};

// The full sweep: scene x cull x textured x lit x shadows = 48 runs. Flip the
// `enabled` flag (first column) to pick a subset.
#[rustfmt::skip]
const RUNS: &[Run] = &[
    //   enabled  scene     cull        tex    lit    shadow
    run( true,    Grid,     NoCull,     false, false, false),
    run( true,    Grid,     NoCull,     false, false, true ),
    run( true,    Grid,     NoCull,     false, true,  false),
    run( true,    Grid,     NoCull,     false, true,  true ),
    run( true,    Grid,     NoCull,     true,  false, false),
    run( true,    Grid,     NoCull,     true,  false, true ),
    run( true,    Grid,     NoCull,     true,  true,  false),
    run( true,    Grid,     NoCull,     true,  true,  true ),
    run( true,    Grid,     Frustum,    false, false, false),
    run( true,    Grid,     Frustum,    false, false, true ),
    run( true,    Grid,     Frustum,    false, true,  false),
    run( true,    Grid,     Frustum,    false, true,  true ),
    run( true,    Grid,     Frustum,    true,  false, false),
    run( true,    Grid,     Frustum,    true,  false, true ),
    run( true,    Grid,     Frustum,    true,  true,  false),
    run( true,    Grid,     Frustum,    true,  true,  true ),
    run( true,    Grid,     FrustumHiz, false, false, false),
    run( true,    Grid,     FrustumHiz, false, false, true ),
    run( true,    Grid,     FrustumHiz, false, true,  false),
    run( true,    Grid,     FrustumHiz, false, true,  true ),
    run( true,    Grid,     FrustumHiz, true,  false, false),
    run( true,    Grid,     FrustumHiz, true,  false, true ),
    run( true,    Grid,     FrustumHiz, true,  true,  false),
    run( true,    Grid,     FrustumHiz, true,  true,  true ),
    run( true,    Occluder, NoCull,     false, false, false),
    run( true,    Occluder, NoCull,     false, false, true ),
    run( true,    Occluder, NoCull,     false, true,  false),
    run( true,    Occluder, NoCull,     false, true,  true ),
    run( true,    Occluder, NoCull,     true,  false, false),
    run( true,    Occluder, NoCull,     true,  false, true ),
    run( true,    Occluder, NoCull,     true,  true,  false),
    run( true,    Occluder, NoCull,     true,  true,  true ),
    run( true,    Occluder, Frustum,    false, false, false),
    run( true,    Occluder, Frustum,    false, false, true ),
    run( true,    Occluder, Frustum,    false, true,  false),
    run( true,    Occluder, Frustum,    false, true,  true ),
    run( true,    Occluder, Frustum,    true,  false, false),
    run( true,    Occluder, Frustum,    true,  false, true ),
    run( true,    Occluder, Frustum,    true,  true,  false),
    run( true,    Occluder, Frustum,    true,  true,  true ),
    run( true,    Occluder, FrustumHiz, false, false, false),
    run( true,    Occluder, FrustumHiz, false, false, true ),
    run( true,    Occluder, FrustumHiz, false, true,  false),
    run( true,    Occluder, FrustumHiz, false, true,  true ),
    run( true,    Occluder, FrustumHiz, true,  false, false),
    run( true,    Occluder, FrustumHiz, true,  false, true ),
    run( true,    Occluder, FrustumHiz, true,  true,  false),
    run( true,    Occluder, FrustumHiz, true,  true,  true ),
];

// ---------------------------------------------------------------------------
// CLI parameters (do not change the sweep, only the scene size / path length)
// ---------------------------------------------------------------------------

struct Params {
    count: u32,
    frames: u32,
    render_scale: f32,
    out: Option<String>,
}

fn parse_params() -> Params {
    let mut p = Params {
        count: 125_000,
        frames: 900,
        render_scale: 1.0,
        out: None,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned();
        match args[i].as_str() {
            "--count" => {
                if let Some(v) = next(i).and_then(|s| s.parse().ok()) {
                    p.count = v;
                }
                i += 2;
            }
            "--frames" => {
                if let Some(v) = next(i).and_then(|s| s.parse().ok()) {
                    p.frames = v;
                }
                i += 2;
            }
            "--render-scale" => {
                if let Some(v) = next(i).and_then(|s| s.parse().ok()) {
                    p.render_scale = v;
                }
                i += 2;
            }
            "--out" => {
                p.out = next(i);
                i += 2;
            }
            _ => i += 1,
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Camera path
// ---------------------------------------------------------------------------

/// Path segments, each an equal slice of the frame budget.
const SEGMENTS: &[&str] = &[
    "orbit_far",
    "approach",
    "punch_through",
    "inside_pan",
    "face_away",
];

/// Segment label for a frame.
fn segment_for(frame: u32, total: u32) -> &'static str {
    let idx = ((frame as f32 / total as f32) * SEGMENTS.len() as f32) as usize;
    SEGMENTS[idx.min(SEGMENTS.len() - 1)]
}

/// Eye and target for a frame along the deterministic flight. `e` is the grid
/// half-extent. The path orbits, approaches from -X, flies through, looks
/// around inside, then faces outward.
fn camera_for(frame: u32, total: u32, e: f32) -> (glam::Vec3, glam::Vec3) {
    let g = frame as f32 / total as f32; // 0..1 over the whole path
    let seg = (g * SEGMENTS.len() as f32) as usize;
    let local = (g * SEGMENTS.len() as f32) - seg as f32; // 0..1 within segment
    let s = smoothstep(local);
    let z_mid = 0.25 * e;
    match seg {
        0 => {
            // Orbit far, whole field in view.
            let az = std::f32::consts::PI * (0.25 + 0.6 * s);
            let r = 2.3 * e;
            (
                glam::Vec3::new(r * az.cos(), r * az.sin(), 0.7 * e),
                glam::Vec3::new(0.0, 0.0, z_mid),
            )
        }
        1 => {
            // Approach from -X toward the field (through the wall, occluder scene).
            let x = lerp(-2.3 * e, -1.05 * e, s);
            (
                glam::Vec3::new(x, 0.0, 0.5 * e),
                glam::Vec3::new(0.0, 0.0, z_mid),
            )
        }
        2 => {
            // Punch straight through along +X.
            let x = lerp(-1.05 * e, 1.05 * e, s);
            (
                glam::Vec3::new(x, 0.0, z_mid),
                glam::Vec3::new(x + e, 0.0, z_mid),
            )
        }
        3 => {
            // Pan around from inside the field.
            let a = std::f32::consts::TAU * s;
            (
                glam::Vec3::new(0.0, 0.0, z_mid),
                glam::Vec3::new(e * a.cos(), e * a.sin(), z_mid),
            )
        }
        _ => {
            // Move out +X and face outward (most of the field behind the camera).
            let x = lerp(1.05 * e, 2.3 * e, s);
            (
                glam::Vec3::new(x, 0.0, 0.5 * e),
                glam::Vec3::new(x + e, 0.0, 0.5 * e),
            )
        }
    }
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

struct Meshes {
    box_id: MeshId,
    slab_id: MeshId,
    textures: Vec<u64>,
    /// Grid half-extent in world units.
    extent: f32,
}

fn build_meshes(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    count: u32,
) -> Meshes {
    let box_id = renderer
        .resources_mut()
        .upload_mesh_data(device, &primitives::cube(1.0))
        .expect("box upload");

    let n = (count as f32).cbrt().round().max(1.0) as u32;
    let spacing = 2.5_f32;
    let extent = (n as f32 - 1.0) * spacing * 0.5;

    // A solid wall thin in X, spanning the field in Y and Z, used as the occluder.
    let slab = primitives::cuboid(2.0, 2.6 * extent, 2.6 * extent);
    let slab_id = renderer
        .resources_mut()
        .upload_mesh_data(device, &slab)
        .expect("slab upload");

    // A small texture pool (solid hues): enough to exercise sampling plus a few
    // batches without dominating memory.
    let mut textures = Vec::new();
    for i in 0..8u32 {
        let rgba = solid_texture(i);
        let id = renderer
            .resources_mut()
            .upload_texture(device, queue, 8, 8, &rgba)
            .expect("texture upload");
        textures.push(id);
    }

    Meshes {
        box_id,
        slab_id,
        textures,
        extent,
    }
}

fn solid_texture(i: u32) -> Vec<u8> {
    let (r, g, b) = hsv(i as f32 / 8.0, 0.6, 0.9);
    let px = [
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
        255u8,
    ];
    px.iter().copied().cycle().take(8 * 8 * 4).collect()
}

fn hsv(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// Build the static item list for one run (camera-independent).
fn build_items(m: &Meshes, count: u32, run: &Run) -> Vec<SceneRenderItem> {
    let n = (count as f32).cbrt().round().max(1.0) as u32;
    let spacing = 2.5_f32;
    let half = (n as f32 - 1.0) * 0.5;

    let mut items = Vec::with_capacity((n * n * n) as usize + 1);
    let mut idx = 0u32;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let pos = glam::Vec3::new(
                    (x as f32 - half) * spacing,
                    (y as f32 - half) * spacing,
                    (z as f32 - half) * spacing,
                );
                items.push(make_item(
                    m.box_id,
                    glam::Mat4::from_translation(pos),
                    m,
                    run,
                    idx,
                ));
                idx += 1;
            }
        }
    }

    if run.scene == SceneKind::Occluder {
        // Wall just outside the -X face so the approach view is blocked.
        let model = glam::Mat4::from_translation(glam::Vec3::new(-m.extent - 2.0, 0.0, 0.0));
        items.push(make_item(m.slab_id, model, m, run, idx));
    }

    items
}

fn make_item(mesh: MeshId, model: glam::Mat4, m: &Meshes, run: &Run, idx: u32) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = model.to_cols_array_2d();
    item.material = Material::flat([0.8, 0.8, 0.8]);
    if run.textured {
        item.material.texture_id = Some(m.textures[(idx as usize) % m.textures.len()]);
    }
    let mut settings = ItemSettings::default();
    settings.unlit = !run.lit;
    item.settings = settings;
    item
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Samples {
    total_ms: Vec<f32>,
    prepare_ms: Vec<f32>,
    paint_ms: Vec<f32>,
    gpu_ms: Vec<f32>,
    scene_ms: Vec<f32>,
    shadow_ms: Vec<f32>,
    cull_ms: Vec<f32>,
    post_ms: Vec<f32>,
    // CPU prepare phase split.
    prep_plugin_ms: Vec<f32>,
    prep_lighting_ms: Vec<f32>,
    prep_uniforms_ms: Vec<f32>,
    prep_instancing_ms: Vec<f32>,
    prep_geometry_ms: Vec<f32>,
    prep_shadow_ms: Vec<f32>,
    prep_viewport_ms: Vec<f32>,
    prep_other_ms: Vec<f32>,
    visible: Vec<f32>,
    frustum_vis: Vec<f32>,
    total_considered: Vec<f32>,
    occlusion_culled: Vec<f32>,
    batches_reuploaded: Vec<f32>,
    batches_skipped: Vec<f32>,
    draw_calls: Vec<f32>,
    batches: Vec<f32>,
    triangles: Vec<f32>,
}

fn pct(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p * (v.len() - 1) as f32).round() as usize).min(v.len() - 1);
    v[idx]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let params = parse_params();

    let (device, queue, has_ts, has_indirect) = match init_device() {
        Some(t) => t,
        None => {
            eprintln!("no suitable GPU adapter; aborting");
            return;
        }
    };
    if !has_indirect {
        eprintln!("warning: INDIRECT_FIRST_INSTANCE unsupported; GPU culling will be inactive");
    }
    if !has_ts {
        eprintln!("warning: TIMESTAMP_QUERY unsupported; GPU timings will be missing");
    }

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let meshes = build_meshes(&mut renderer, &device, &queue, params.count);

    let base_w = 1920u32;
    let base_h = 1080u32;
    let w = ((base_w as f32 * params.render_scale) as u32).max(16);
    let h = ((base_h as f32 * params.render_scale) as u32).max(16);

    let out_path = params.out.clone().unwrap_or_else(|| timestamped_filename());
    let mut csv = std::fs::File::create(&out_path).expect("create CSV");
    write_header(&mut csv);

    let enabled: Vec<&Run> = RUNS.iter().filter(|r| r.enabled).collect();
    println!(
        "perf-bench: {} runs, {}x{} ({} boxes), {} frames each -> {}",
        enabled.len(),
        w,
        h,
        params.count,
        params.frames,
        out_path
    );

    let warmup = 30.min(params.frames / 4);
    for (ri, run) in enabled.iter().enumerate() {
        println!(
            "[{}/{}] scene={} cull={} tex={} lit={} shadows={}",
            ri + 1,
            enabled.len(),
            scene_name(run.scene),
            cull_name(run.cull),
            run.textured,
            run.lit,
            run.shadows
        );
        run_one(
            &device,
            &queue,
            &mut renderer,
            &meshes,
            run,
            &params,
            w,
            h,
            warmup,
            &mut csv,
        );
    }

    println!("done -> {}", out_path);
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut ViewportRenderer,
    meshes: &Meshes,
    run: &Run,
    params: &Params,
    w: u32,
    h: u32,
    warmup: u32,
    csv: &mut std::fs::File,
) {
    // Cull configuration.
    match run.cull {
        CullMode::None => {
            renderer.disable_gpu_driven_culling();
            renderer.set_occlusion_culling(false);
        }
        CullMode::Frustum => {
            renderer.enable_gpu_driven_culling();
            renderer.set_occlusion_culling(false);
        }
        CullMode::FrustumHiz => {
            renderer.enable_gpu_driven_culling();
            renderer.set_occlusion_culling(true);
        }
    }

    let items: std::sync::Arc<[SceneRenderItem]> = build_items(meshes, params.count, run).into();

    // One sample bucket per segment.
    let mut buckets: std::collections::HashMap<&'static str, Samples> =
        SEGMENTS.iter().map(|s| (*s, Samples::default())).collect();

    let near = 0.1_f32;
    let far = (meshes.extent * 8.0).max(500.0);
    let fov = 60f32.to_radians();
    let aspect = w as f32 / h as f32;
    let proj = glam::Mat4::perspective_rh(fov, aspect, near, far);

    let mut frame = FrameData::default();
    frame.camera.viewport_size = [w as f32, h as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    // HiZ builds in the HDR scene pass, so keep the HDR pipeline on for every
    // run (so the cull comparison is not contaminated by HDR-vs-LDR).
    frame.effects.post_process.enabled = true;
    frame.scene.surfaces = SurfaceSubmission::Flat(items);
    // One directional light. `shadows_enabled` is the global toggle that gates
    // the directional shadow pass; the per-light `cast_shadows` does not gate
    // the CSM caster, so set both to match the run.
    frame.effects.lighting = {
        let mut light = LightSource::default();
        light.cast_shadows = run.shadows;
        let mut l = LightingSettings::default();
        l.lights = vec![light];
        l.shadows_enabled = run.shadows;
        l
    };

    // Each run reuses the renderer but changes the material set (textured/lit),
    // and the scene generation never bumps, so the instanced batch cache would
    // otherwise serve the first run's batches to every later run. Force a
    // rebuild so this run's textures and unlit flag actually take effect; the
    // rebuild happens on the first (warmup) frame, steady-state frames still hit
    // the cache.
    renderer.force_dirty();

    for f in 0..params.frames {
        let (eye, target) = camera_for(f, params.frames, meshes.extent);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Z);
        let forward = (target - eye).normalize_or_zero();
        let orientation = glam::Quat::from_mat3(&glam::Mat3::from_mat4(view.inverse())).normalize();
        frame.camera.render_camera = RenderCamera {
            view,
            projection: proj,
            eye_position: eye.to_array(),
            forward: forward.to_array(),
            orientation,
            near,
            far,
            distance: (target - eye).length().max(0.001),
            fov,
            aspect,
        };

        let _ = renderer.render_offscreen(device, queue, &frame, w, h);
        let _ = device.poll(wgpu::PollType::Poll);

        if f < warmup {
            continue;
        }
        let st = renderer.last_frame_stats();
        let b = buckets.get_mut(segment_for(f, params.frames)).unwrap();
        b.total_ms.push(st.total_frame_ms);
        b.prepare_ms.push(st.cpu_prepare_ms);
        b.paint_ms.push(st.cpu_paint_ms);
        if let Some(g) = st.gpu_frame_ms {
            b.gpu_ms.push(g);
        }
        b.scene_ms.push(st.gpu_breakdown.scene_ms);
        b.shadow_ms.push(st.gpu_breakdown.shadow_ms);
        b.cull_ms.push(st.gpu_breakdown.cull_ms);
        b.post_ms.push(st.gpu_breakdown.post_ms);
        let pb = &st.prepare_breakdown;
        b.prep_plugin_ms.push(pb.plugin_ms);
        b.prep_lighting_ms.push(pb.lighting_ms);
        b.prep_uniforms_ms.push(pb.uniforms_ms);
        b.prep_instancing_ms.push(pb.instancing_ms);
        b.prep_geometry_ms.push(pb.geometry_ms);
        b.prep_shadow_ms.push(pb.shadow_ms);
        b.prep_viewport_ms.push(pb.viewport_ms);
        b.prep_other_ms.push(pb.other_ms);
        b.batches_reuploaded.push(st.batches_reuploaded as f32);
        b.batches_skipped.push(st.batches_skipped as f32);
        if let Some(v) = st.gpu_visible_instances {
            b.visible.push(v as f32);
        }
        if let Some(v) = st.gpu_frustum_visible {
            b.frustum_vis.push(v as f32);
        }
        if let Some(total) = st.gpu_culled_total {
            b.total_considered.push(total as f32);
        }
        if let (Some(fr), Some(drawn)) = (st.gpu_frustum_visible, st.gpu_visible_instances) {
            b.occlusion_culled.push((fr.saturating_sub(drawn)) as f32);
        }
        b.draw_calls.push(st.draw_calls as f32);
        b.batches.push(st.instanced_batches as f32);
        b.triangles.push(st.triangles_submitted as f32);
    }

    for seg in SEGMENTS {
        let s = buckets.get_mut(*seg).unwrap();
        write_row(csv, run, seg, s);
    }
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

fn init_device() -> Option<(wgpu::Device, wgpu::Queue, bool, bool)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;

    let avail = adapter.features();
    let has_ts = avail.contains(wgpu::Features::TIMESTAMP_QUERY);
    let has_indirect = avail.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
    let mut wanted = wgpu::Features::empty();
    if has_ts {
        wanted |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if has_indirect {
        wanted |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("perf-bench"),
        required_features: wanted,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue, has_ts, has_indirect))
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

fn write_header(f: &mut std::fs::File) {
    let cols = [
        "scene",
        "cull",
        "textured",
        "lit",
        "shadows",
        "segment",
        "frames",
        "gpu_ms_p50",
        "gpu_ms_p95",
        "gpu_ms_p99",
        "scene_ms_p50",
        "scene_ms_p95",
        "cull_ms_p50",
        "shadow_ms_p50",
        "post_ms_p50",
        "prepare_ms_p50",
        "prep_plugin_ms_p50",
        "prep_lighting_ms_p50",
        "prep_uniforms_ms_p50",
        "prep_instancing_ms_p50",
        "prep_geometry_ms_p50",
        "prep_shadow_ms_p50",
        "prep_viewport_ms_p50",
        "prep_other_ms_p50",
        "paint_ms_p50",
        "total_ms_p50",
        "total_ms_p95",
        "visible_p50",
        "frustum_visible_p50",
        "total_considered_p50",
        "occlusion_culled_p50",
        "batches_reuploaded_p50",
        "batches_skipped_p50",
        "draw_calls_p50",
        "batches_p50",
        "triangles_p50",
    ];
    writeln!(f, "{}", cols.join(",")).unwrap();
}

fn write_row(f: &mut std::fs::File, run: &Run, segment: &str, s: &mut Samples) {
    let n = s.total_ms.len();
    let row = format!(
        "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0}",
        scene_name(run.scene),
        cull_name(run.cull),
        run.textured,
        run.lit,
        run.shadows,
        segment,
        n,
        pct(&mut s.gpu_ms, 0.50),
        pct(&mut s.gpu_ms, 0.95),
        pct(&mut s.gpu_ms, 0.99),
        pct(&mut s.scene_ms, 0.50),
        pct(&mut s.scene_ms, 0.95),
        pct(&mut s.cull_ms, 0.50),
        pct(&mut s.shadow_ms, 0.50),
        pct(&mut s.post_ms, 0.50),
        pct(&mut s.prepare_ms, 0.50),
        pct(&mut s.prep_plugin_ms, 0.50),
        pct(&mut s.prep_lighting_ms, 0.50),
        pct(&mut s.prep_uniforms_ms, 0.50),
        pct(&mut s.prep_instancing_ms, 0.50),
        pct(&mut s.prep_geometry_ms, 0.50),
        pct(&mut s.prep_shadow_ms, 0.50),
        pct(&mut s.prep_viewport_ms, 0.50),
        pct(&mut s.prep_other_ms, 0.50),
        pct(&mut s.paint_ms, 0.50),
        pct(&mut s.total_ms, 0.50),
        pct(&mut s.total_ms, 0.95),
        pct(&mut s.visible, 0.50),
        pct(&mut s.frustum_vis, 0.50),
        pct(&mut s.total_considered, 0.50),
        pct(&mut s.occlusion_culled, 0.50),
        pct(&mut s.batches_reuploaded, 0.50),
        pct(&mut s.batches_skipped, 0.50),
        pct(&mut s.draw_calls, 0.50),
        pct(&mut s.batches, 0.50),
        pct(&mut s.triangles, 0.50),
    );
    writeln!(f, "{}", row).unwrap();
}

fn scene_name(s: SceneKind) -> &'static str {
    match s {
        SceneKind::Grid => "grid",
        SceneKind::Occluder => "occluder",
    }
}

fn cull_name(c: CullMode) -> &'static str {
    match c {
        CullMode::None => "none",
        CullMode::Frustum => "frustum",
        CullMode::FrustumHiz => "frustum+hiz",
    }
}

// ---------------------------------------------------------------------------
// Timestamped filename: perf_bench_DDMMYYYY_<unixtime>.csv
// ---------------------------------------------------------------------------

fn timestamped_filename() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("perf_bench_{:02}{:02}{:04}_{}.csv", d, m, y, secs)
}

/// Convert days since 1970-01-01 to (year, month, day). Standard civil-date
/// algorithm, valid for the full range of dates.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}
