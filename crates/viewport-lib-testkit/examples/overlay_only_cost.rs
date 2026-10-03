//! Measures what an overlay-only consumer pays for the 3D scene machinery.
//!
//! Two figures:
//!
//!   1. Startup: how long `ViewportRenderer::new` takes, split by init phase
//!      (the `viewport_lib::init` tracing target already marks them).
//!   2. Per frame: CPU record time and the `prepare` breakdown for a frame that
//!      carries overlays and no scene geometry at all.
//!
//! Run with:
//!   cargo run --release --example overlay-only-cost   # from crates/viewport-lib-testkit

use std::sync::{Arc, Mutex};
use std::time::Instant;

use viewport_lib as vpl;
use vpl::wgpu;
use vpl::{
    Camera, FrameData, LabelItem, OverlayFrame, OverlayShape, OverlayShapeItem, RenderCamera,
    ViewportRenderer,
};

const W: u32 = 1280;
const H: u32 = 720;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const FRAMES: usize = 240;

// ---------------------------------------------------------------------------
// Collect the init-phase marks the library emits on the `viewport_lib::init`
// target instead of printing them as log lines.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Marks(Arc<Mutex<Vec<(String, f32)>>>);

struct MarkLayer(Arc<Mutex<Vec<(String, f32)>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for MarkLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if event.metadata().target() != "viewport_lib::init" {
            return;
        }
        let mut v = Visitor {
            section: None,
            ms: None,
        };
        event.record(&mut v);
        match (v.section, v.ms) {
            (Some(s), Some(ms)) => self.0.lock().unwrap().push((s, ms)),
            // The closing event carries the total and no section name.
            (None, Some(ms)) => self
                .0
                .lock()
                .unwrap()
                .push(("= DeviceResources::new total".to_string(), ms)),
            _ => {}
        }
    }
}

struct Visitor {
    section: Option<String>,
    ms: Option<f32>,
}

impl tracing::field::Visit for Visitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "section" {
            self.section = Some(value.to_string());
        }
    }
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        if field.name() == "ms" {
            self.ms = Some(value as f32);
        }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "section" && self.section.is_none() {
            self.section = Some(format!("{value:?}").trim_matches('"').to_string());
        }
    }
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx]
}

fn main() {
    use tracing_subscriber::layer::{Layer as _, SubscriberExt};
    let marks = Marks::default();
    let collected = marks.0.clone();
    // The filter matters: an unfiltered layer makes `tracing::enabled!` return
    // true for every target, which switches on the library's own debug
    // instrumentation (the shadow pass then polls the device to completion every
    // frame). Only the init target is wanted here.
    let filter =
        tracing_subscriber::filter::FilterFn::new(|meta| meta.target() == "viewport_lib::init");
    let subscriber =
        tracing_subscriber::registry().with(MarkLayer(collected.clone()).with_filter(filter));
    tracing::subscriber::set_global_default(subscriber).expect("set subscriber");

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

    let t_device = Instant::now();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("overlay-only-cost"),
        required_limits: ViewportRenderer::recommended_device_limits(&adapter),
        required_features: ViewportRenderer::recommended_device_features(&adapter)
            | wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    }))
    .expect("no wgpu device");
    let device_ms = t_device.elapsed().as_secs_f32() * 1000.0;

    // --------------------------------------------------------------------
    // Startup
    // --------------------------------------------------------------------
    let t_new = Instant::now();
    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    let new_ms = t_new.elapsed().as_secs_f32() * 1000.0;
    let sections = collected.lock().unwrap().clone();

    // Per-pipeline attribution of what `new` just built (VPL_BUILD_LOG=1).
    let startup_builds = vpl::resources::build_log::drain();
    // And every buffer and texture it allocated for itself.
    let startup_allocs = vpl::resources::build_log::drain_allocations();

    // What a consumer that installs the built-in item types pays on top. The
    // registration itself is cheap; what it adds is pipelines built on the
    // first frame, which the per-frame build log below shows.
    let install_all = std::env::var("VPL_NO_ITEM_TYPES").is_err();
    let mut install_total = 0.0;
    if install_all {
        let t = Instant::now();
        viewport_lib_item_types::install(&mut renderer, &device);
        install_total = t.elapsed().as_secs_f32() * 1000.0;
    }
    // Drained here so a type that builds at registration is not counted
    // against the first frame.
    let install_builds = vpl::resources::build_log::drain();

    println!(
        "adapter: {:?} / {} ({:?})",
        info.backend, info.name, info.device_type
    );
    println!();
    println!("startup");
    println!("  request_device            {device_ms:8.2} ms");
    println!("  ViewportRenderer::new     {new_ms:8.2} ms");
    let mut accounted = 0.0;
    for (name, ms) in &sections {
        println!("    {name:<32} {ms:8.2} ms");
        if !name.starts_with('=') {
            accounted += ms;
        }
    }
    println!(
        "    {:<32} {:8.2} ms",
        "(unmarked remainder)",
        new_ms - accounted
    );
    if install_all {
        println!("  item_types::install       {install_total:8.2} ms  (16 built-in item types)");
    } else {
        println!("  item_types::install          (skipped)");
    }
    println!(
        "  total to usable renderer  {:8.2} ms",
        device_ms + new_ms + install_total
    );
    if !install_builds.is_empty() {
        let total: f32 = install_builds.iter().map(|(_, ms)| ms).sum();
        println!(
            "  {} pipelines/modules built during install(), {total:.2} ms of the {install_total:.2} ms:",
            install_builds.len()
        );
        for (label, ms) in &install_builds {
            println!("    {label:<48} {ms:7.3} ms");
        }
    }
    if !startup_builds.is_empty() {
        let total: f32 = startup_builds.iter().map(|(_, ms)| ms).sum();
        println!(
            "  {} pipelines/modules built during new(), {total:.2} ms of the {new_ms:.2} ms:",
            startup_builds.len()
        );
        let mut sorted = startup_builds.clone();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (label, ms) in sorted.iter().take(10) {
            println!("    {label:<48} {ms:7.3} ms");
        }
        // Everything a frame with no scene geometry never binds.
        let unused = [
            "mesh_shader",
            "solid",
            "transparent",
            "wireframe",
            "shadow",
            "decal",
            "cluster",
            "ground_plane",
            "xray",
            "outline",
            "skybox",
            "exposure",
        ];
        let wasted: f32 = startup_builds
            .iter()
            .filter(|(l, _)| unused.iter().any(|u| l.contains(u)))
            .map(|(_, ms)| ms)
            .sum();
        let n = startup_builds
            .iter()
            .filter(|(l, _)| unused.iter().any(|u| l.contains(u)))
            .count();
        println!(
            "    of which {n} are scene-geometry pipelines an overlay-only frame never binds: \
             {wasted:.2} ms"
        );
    }
    if !startup_allocs.is_empty() {
        let total: u64 = startup_allocs.iter().map(|(_, b)| b).sum();
        println!(
            "  {} buffers/textures allocated during new(), {:.2} MiB:",
            startup_allocs.len(),
            total as f64 / (1024.0 * 1024.0)
        );
        let mut sorted = startup_allocs.clone();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (label, bytes) in sorted.iter().take(12) {
            println!(
                "    {label:<48} {:9.3} MiB",
                *bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }

    // --------------------------------------------------------------------
    // Per frame, overlays only: no meshes uploaded, no scene items submitted.
    // --------------------------------------------------------------------
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

    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = W as f32 / H as f32;
        rc
    };
    frame.camera.viewport_size = [W as f32, H as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.1, 0.1, 0.12, 1.0].into());

    // Modes: pass a name on the command line to turn one cost off and see the
    // delta. `baseline` is what an overlay-only consumer gets by default today.
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "baseline".to_string());
    match mode.as_str() {
        "baseline" => {}
        // Nothing in the frame casts a shadow, so an overlay-only consumer can
        // already turn this off by hand. Measures what that is worth.
        "no-shadows" => frame.effects.lighting.shadows.enabled = false,
        // No lights at all.
        "no-lights" => {
            frame.effects.lighting.lights.clear();
            frame.effects.lighting.shadows.enabled = false;
        }
        // The LDR passthrough: no HDR target, no tone map, no post chain. An
        // overlay-only consumer can select this today.
        "direct" => frame.effects.display.mode = vpl::PipelineMode::Direct,
        "direct-no-shadows" => {
            frame.effects.display.mode = vpl::PipelineMode::Direct;
            frame.effects.lighting.shadows.enabled = false;
        }
        other => panic!("unknown mode {other}"),
    }
    println!();
    println!("mode: {mode}");

    // A modest interface: 64 shapes and 64 labels, the kind of load an
    // overlay-only consumer draws every frame.
    let mut overlays = OverlayFrame::default();
    for i in 0..64 {
        let x = (i % 8) as f32 * 150.0 + 20.0;
        let y = (i / 8) as f32 * 80.0 + 20.0;
        overlays.shapes.push(OverlayShapeItem::new(
            OverlayShape::RoundedRect { radii: [6.0; 4] },
            [x, y],
            [130.0, 60.0],
        ));
        overlays
            .labels
            .push(LabelItem::new(format!("item {i}")).with_screen_anchor([x + 8.0, y + 20.0]));
    }
    frame.overlays = overlays;

    let mut cpu = Vec::with_capacity(FRAMES);
    let mut prep = Vec::with_capacity(FRAMES);
    let mut gpu = Vec::with_capacity(FRAMES);
    let mut breakdown_sum = [0.0f64; 8];

    for i in 0..FRAMES {
        // Vary a little so nothing is trivially cached across the whole run.
        frame.overlays.shapes[0].transform.translate[0] = 20.0 + (i % 4) as f32;
        let t = Instant::now();
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
        cpu.push(t.elapsed().as_secs_f32() * 1000.0);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let stats = renderer.last_frame_stats();
        if i < 6 {
            println!(
                "  frame {i}: {:7.3} ms cpu, {} pipelines built",
                cpu[i], stats.pipelines_built_this_frame
            );
        }
        if i < 6 {
            let allocs = vpl::resources::build_log::drain_allocations();
            let targets = vpl::resources::build_log::drain_textures();
            if !allocs.is_empty() || !targets.is_empty() {
                let a: u64 = allocs.iter().map(|(_, b)| b).sum();
                let t: u64 = targets.iter().map(|(_, b)| b).sum();
                println!(
                    "      frame {i} allocated {} buffers/textures ({:.2} MiB) and {} render \
                     targets ({:.2} MiB)",
                    allocs.len(),
                    a as f64 / (1024.0 * 1024.0),
                    targets.len(),
                    t as f64 / (1024.0 * 1024.0)
                );
            }
            let builds = vpl::resources::build_log::drain();
            if !builds.is_empty() {
                let total: f32 = builds.iter().map(|(_, ms)| ms).sum();
                let overlay: f32 = builds
                    .iter()
                    .filter(|(l, _)| l.contains("overlay"))
                    .map(|(_, ms)| ms)
                    .sum();
                println!(
                    "      frame {i} built {} pipelines/modules, {total:.2} ms \
                     ({overlay:.2} ms of it overlay, {:.2} ms not)",
                    builds.len(),
                    total - overlay
                );
                let mut sorted = builds.clone();
                sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (label, ms) in sorted.iter().take(6) {
                    println!("        {label:<46} {ms:7.3} ms");
                }
            }
        }
        prep.push(stats.cpu_prepare_ms);
        if let Some(g) = stats.gpu_frame_ms {
            gpu.push(g);
        }
        let b = stats.prepare_breakdown;
        for (slot, v) in breakdown_sum.iter_mut().zip([
            b.plugin_ms,
            b.plugin_cull_ms,
            b.lighting_ms,
            b.uniforms_ms,
            b.instancing_ms,
            b.geometry_ms,
            b.shadow_ms,
            b.viewport_ms,
        ]) {
            *slot += v as f64;
        }
        breakdown_sum[7] += 0.0;
    }

    // Drop the warmup frames: pipelines warm and buffers grow on the first few.
    let warm = FRAMES / 4;
    let mut cpu_w: Vec<f32> = cpu[warm..].to_vec();
    let mut prep_w: Vec<f32> = prep[warm..].to_vec();
    cpu_w.sort_by(f32::total_cmp);
    prep_w.sort_by(f32::total_cmp);

    println!();
    println!(
        "per frame, overlays only ({W}x{H}, 64 shapes + 64 labels, {} warm frames)",
        cpu_w.len()
    );
    println!(
        "  cpu record+submit   p50 {:6.3} ms   p95 {:6.3} ms   first frame {:6.3} ms",
        percentile(&cpu_w, 0.5),
        percentile(&cpu_w, 0.95),
        cpu[0]
    );
    println!(
        "  prepare (stats)     p50 {:6.3} ms   p95 {:6.3} ms",
        percentile(&prep_w, 0.5),
        percentile(&prep_w, 0.95)
    );
    if !gpu.is_empty() {
        let mut g = gpu[warm.min(gpu.len())..].to_vec();
        g.sort_by(f32::total_cmp);
        println!(
            "  gpu frame           p50 {:6.3} ms   p95 {:6.3} ms",
            percentile(&g, 0.5),
            percentile(&g, 0.95)
        );
    }

    let n = FRAMES as f64;
    let names = [
        "plugin",
        "plugin_cull",
        "lighting",
        "uniforms",
        "instancing",
        "geometry",
        "shadow",
        "viewport",
    ];
    println!("  prepare breakdown, mean ms per frame:");
    for (name, sum) in names.iter().zip(breakdown_sum.iter()) {
        println!("    {name:<14} {:7.4}", sum / n);
    }

    let s = renderer.last_frame_stats();
    println!(
        "  last frame: total_objects {} visible {} draw_calls {} shadow_draw_calls {} shadow_draw_commands {}",
        s.total_objects,
        s.visible_objects,
        s.draw_calls,
        s.shadow_draw_calls,
        s.shadow_draw_commands
    );

    // Shadow device memory. Metal keeps private textures out of the process
    // footprint entirely (they do not show in `ps` or `vmmap`), so read the
    // renderer's own accounting rather than the OS's.
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!();
    println!("shadow depth textures");
    println!(
        "  after {FRAMES} overlay-only frames  {:7.1} MiB",
        mb(renderer.shadow_allocation_bytes())
    );

    // One frame that actually casts a cascade shadow: this is what promotes the
    // atlas. The step is what a viewport with no casters no longer pays.
    {
        let cube = renderer
            .resources_mut()
            .upload_mesh_data(&device, &vpl::primitives::cube(1.0))
            .expect("upload cube");
        let mut item = vpl::SceneRenderItem::default();
        item.mesh_id = cube;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material = vpl::Material::from_colour([0.8, 0.85, 0.9]);
        let mut shadow_frame = FrameData::default();
        shadow_frame.camera.render_camera = frame.camera.render_camera.clone();
        shadow_frame.camera.viewport_size = frame.camera.viewport_size;
        shadow_frame.scene = vpl::SceneFrame::from_surface_items(vec![item]);
        shadow_frame.scene.generation = 99;
        let cmd = renderer
            .owned()
            .render(&device, &queue, &view, &shadow_frame);
        queue.submit(std::iter::once(cmd));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let s = renderer.last_frame_stats();
        println!(
            "  after one shadow-casting frame  {:7.1} MiB  ({} shadow draw calls)",
            mb(renderer.shadow_allocation_bytes()),
            s.shadow_draw_calls
        );
    }

    // The same breakdown for a frame with one mesh and no lights, which is what
    // the lighting and shadow prepare cost when there is nothing for them to do.
    {
        let cube = renderer
            .resources_mut()
            .upload_mesh_data(&device, &vpl::primitives::cube(1.0))
            .expect("upload cube");
        let mut item = vpl::SceneRenderItem::default();
        item.mesh_id = cube;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material = vpl::Material::from_colour([0.8, 0.85, 0.9]);
        let mut mesh_frame = FrameData::default();
        mesh_frame.camera.render_camera = frame.camera.render_camera.clone();
        mesh_frame.camera.viewport_size = frame.camera.viewport_size;
        mesh_frame.effects.display.mode = frame.effects.display.mode;
        mesh_frame.effects.lighting.lights.clear();
        mesh_frame.effects.lighting.shadows.enabled = false;
        mesh_frame.scene = vpl::SceneFrame::from_surface_items(vec![item]);
        mesh_frame.scene.generation = 100;
        const MESH_FRAMES: usize = 120;
        let mut sums = [0.0f64; 8];
        let mut counted = 0.0f64;
        for i in 0..MESH_FRAMES {
            let cmd = renderer.owned().render(&device, &queue, &view, &mesh_frame);
            queue.submit(std::iter::once(cmd));
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            });
            if i < MESH_FRAMES / 4 {
                continue;
            }
            let b = renderer.last_frame_stats().prepare_breakdown;
            for (slot, v) in sums.iter_mut().zip([
                b.plugin_ms,
                b.plugin_cull_ms,
                b.lighting_ms,
                b.uniforms_ms,
                b.instancing_ms,
                b.geometry_ms,
                b.shadow_ms,
                b.viewport_ms,
            ]) {
                *slot += v as f64;
            }
            counted += 1.0;
        }
        println!();
        println!("one mesh, no lights, shadows off: prepare breakdown, mean ms per frame:");
        for (name, sum) in names.iter().zip(sums.iter()) {
            println!("    {name:<14} {:7.4}", sum / counted);
        }
    }

    let res = renderer.resident_bytes();
    println!();
    println!("resident gpu bytes after {FRAMES} overlay-only frames: {res:?}");
}
