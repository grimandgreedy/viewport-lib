//! Showcase 50: GPU Wave (compute-plugin override path).
//!
//! Demonstrates a `GpuPlugin` driving a mesh through `set_position_override_buffer`.
//! A `WavePlugin` (defined in `examples/plugins/wave_plugin.rs`) runs a compute
//! shader each frame to produce displaced positions for a plane mesh. The
//! standard lit mesh pipeline renders those positions with no CPU readback.
//!
//! End-to-end data flow:
//!   1. Build a flat plane mesh on the CPU and upload it via `upload_mesh_data`.
//!   2. Construct the plugin with the plane's rest-pose positions; it owns a
//!      compute pipeline and an output storage buffer.
//!   3. Call `set_position_override_buffer(plane_id, plugin.output_buffer())`
//!      once at setup. The standard mesh pipeline now reads positions from
//!      that buffer instead of the vertex buffer's position attribute.
//!   4. Each frame, call `plugin.pre_prepare(...)` and submit the returned
//!      command buffer before eframe's paint callback runs. wgpu serialises
//!      submissions, so the renderer sees the latest displaced positions.

use crate::App;
use eframe::egui;
use viewport_lib::{
    LightKind, LightSource, LightingSettings, Material, MeshId, SceneRenderItem,
    ViewportRenderer,
    runtime::GpuPlugin,
};

#[path = "../plugins/wave_plugin.rs"]
mod wave_plugin;
#[path = "../plugins/buoy_plugin.rs"]
mod buoy_plugin;

use buoy_plugin::BuoyPlugin;
use wave_plugin::WavePlugin;

const BUOY_GRID: usize = 4; // 4 x 4 = 16 buoys
const WAVE_HALF_EXTENT: f32 = 4.0;
const BUOY_SPHERE_RADIUS: f32 = 0.2;
const WATERLINE_OFFSET: f32 = 0.18;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Which path drives the per-vertex wave deformation. The radio button in the
/// controls panel switches between them at runtime so the perf difference is
/// directly comparable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeformMode {
    /// `GpuPlugin::pre_prepare` runs a compute shader; the override buffer is
    /// bound to the mesh and the standard pipeline reads from it. No CPU
    /// round-trip per frame.
    Gpu,
    /// CPU computes the wave each frame and uploads via
    /// `write_mesh_positions_normals`. Same visual output, very different
    /// per-frame cost at this vertex count.
    Cpu,
}

pub(crate) struct WaveState {
    pub built: bool,
    pub plane_id: Option<MeshId>,
    /// Buoy mesh: `BUOY_GRID^2` copies of a small sphere baked into one mesh,
    /// positions driven by `BuoyPlugin` reading the wave plugin's GPU output.
    pub buoys_mesh_id: Option<MeshId>,
    /// Boxed so the showcase state stays Sized when the plugin is absent.
    pub plugin: Option<Box<WavePlugin>>,
    pub buoy_plugin: Option<Box<BuoyPlugin>>,
    pub show_buoys: bool,
    /// Cached rest positions retained for the CPU deformation path so we can
    /// compute displacement without re-uploading the mesh.
    pub rest_positions: Vec<[f32; 3]>,
    pub frame_index: u64,
    pub last_dt: f32,
    pub viewport_size: glam::Vec2,
    pub paused: bool,
    /// UI-tunable; pushed into the plugin each frame.
    pub amplitude: f32,
    pub frequency: f32,
    /// Wall-clock seconds since the demo started. Drives the CPU wave so its
    /// phase advances independently of the plugin's `elapsed` counter.
    pub time: f32,
    pub mode: DeformMode,
    /// Whether the override is currently bound on the renderer side. Tracked
    /// because we toggle it when switching modes (CPU mode clears the
    /// override; GPU mode re-binds the plugin output).
    pub override_bound: bool,
    /// Rolling-average wave update cost in milliseconds. Reset on mode
    /// switch so the displayed value reflects the active path.
    pub update_ms_smoothed: f32,
    /// Pending plane dimension. Mesh is rebuilt when this differs from the
    /// dimension used to build the current `plane_id`.
    pub pending_grid_dim: u32,
    /// Dimension actually backing `plane_id` (cols == rows). 0 before build.
    pub current_grid_dim: u32,
}

impl Default for WaveState {
    fn default() -> Self {
        Self {
            built: false,
            plane_id: None,
            buoys_mesh_id: None,
            plugin: None,
            buoy_plugin: None,
            show_buoys: true,
            rest_positions: Vec::new(),
            frame_index: 0,
            last_dt: 1.0 / 60.0,
            viewport_size: glam::Vec2::new(800.0, 600.0),
            paused: false,
            amplitude: 0.35,
            frequency: 1.6,
            time: 0.0,
            mode: DeformMode::Gpu,
            override_bound: false,
            update_ms_smoothed: 0.0,
            pending_grid_dim: 400,
            current_grid_dim: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_wave_scene(&mut self, renderer: &mut ViewportRenderer) {
        // Default is `pending_grid_dim` (400 x 400 = 160,801 verts on first
        // build). At that scale the CPU writeback path costs ~4-8 ms per frame
        // while the GPU compute stays under 100 us; the gap is clearly visible
        // in the smoothed-ms readout. Push the slider higher (up to 800) to
        // make the CPU path start dropping frames.
        let dim = app_grid_dim(self);
        let plane = viewport_lib::primitives::grid_plane(8.0, 8.0, dim, dim);
        // Capture rest positions before move; the override protocol expects a
        // flat `[x, y, z, x, y, z, ...]` layout, which matches `MeshData`.
        let mut rest_flat: Vec<f32> = Vec::with_capacity(plane.positions.len() * 3);
        for p in &plane.positions {
            rest_flat.extend_from_slice(p);
        }
        let rest_positions = plane.positions.clone();

        let plane_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &plane)
            .expect("upload plane mesh");

        let plugin = WavePlugin::new(
            &self.device,
            &self.queue,
            &rest_flat,
            self.wave_state.amplitude,
            self.wave_state.frequency,
        );

        // Initial state matches `DeformMode::Gpu`: overrides bound, plugin
        // drives positions and normals every frame.
        renderer
            .resources_mut()
            .set_position_override_buffer(plane_id, plugin.output_buffer())
            .expect("bind position override");
        renderer
            .resources_mut()
            .set_normal_override_buffer(plane_id, plugin.normal_buffer())
            .expect("bind normal override");

        // ---- Buoy mesh (chained compute) ---------------------------------
        //
        // One small sphere primitive, replicated `BUOY_GRID^2` times into a
        // single MeshData. The mesh's vertex *attributes* (position, normal,
        // colour) are baked once; per-frame the position override buffer
        // produced by `BuoyPlugin` will replace the position attribute.
        let base_sphere =
            viewport_lib::primitives::sphere(BUOY_SPHERE_RADIUS, 12, 8);
        let verts_per_buoy = base_sphere.positions.len();
        let buoy_count = BUOY_GRID * BUOY_GRID;

        let mut buoys = viewport_lib::MeshData::default();
        buoys.positions.reserve(verts_per_buoy * buoy_count);
        buoys.normals.reserve(verts_per_buoy * buoy_count);
        buoys.indices.reserve(base_sphere.indices.len() * buoy_count);
        for _ in 0..buoy_count {
            buoys.positions.extend_from_slice(&base_sphere.positions);
            buoys.normals.extend_from_slice(&base_sphere.normals);
        }
        for b in 0..buoy_count {
            let offset = (b * verts_per_buoy) as u32;
            for &i in &base_sphere.indices {
                buoys.indices.push(i + offset);
            }
        }
        let buoys_mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &buoys)
            .expect("upload buoys mesh");

        // Anchor each buoy on a regular grid inside the wave's extent.
        let mut buoy_anchors: Vec<f32> = Vec::with_capacity(buoy_count * 2);
        let span = WAVE_HALF_EXTENT * 0.7; // total range = 2 * span (~5.6 units, fits inside the 8x8 wave)
        let step = (span * 2.0) / (BUOY_GRID as f32 - 1.0);
        for row in 0..BUOY_GRID {
            for col in 0..BUOY_GRID {
                let x = -span + col as f32 * step;
                let y = -span + row as f32 * step;
                buoy_anchors.push(x);
                buoy_anchors.push(y);
            }
        }

        // Sphere local positions, flat layout (3 floats per vertex), matching
        // the buoy compute's `sphere_local` binding.
        let mut sphere_local_flat: Vec<f32> = Vec::with_capacity(verts_per_buoy * 3);
        for p in &base_sphere.positions {
            sphere_local_flat.extend_from_slice(p);
        }

        let buoy_plugin = BuoyPlugin::new(
            &self.device,
            &self.queue,
            plugin.output_buffer(), // SHARED handle: chained-compute input.
            dim,
            WAVE_HALF_EXTENT,
            &buoy_anchors,
            &sphere_local_flat,
            WATERLINE_OFFSET,
        );
        renderer
            .resources_mut()
            .set_position_override_buffer(buoys_mesh_id, buoy_plugin.output_buffer())
            .expect("bind buoy position override");

        self.wave_state.plane_id = Some(plane_id);
        self.wave_state.buoys_mesh_id = Some(buoys_mesh_id);
        self.wave_state.plugin = Some(Box::new(plugin));
        self.wave_state.buoy_plugin = Some(Box::new(buoy_plugin));
        self.wave_state.rest_positions = rest_positions;
        self.wave_state.frame_index = 0;
        self.wave_state.time = 0.0;
        self.wave_state.override_bound = true;
        self.wave_state.update_ms_smoothed = 0.0;
        self.wave_state.current_grid_dim = dim;
        self.wave_state.built = true;
    }
}

fn app_grid_dim(app: &App) -> u32 {
    app.wave_state.pending_grid_dim.max(8)
}

// ---------------------------------------------------------------------------
// Scene collection (called from the per-frame match in main.rs)
// ---------------------------------------------------------------------------

pub(crate) fn wave_collect(app: &App) -> (Vec<SceneRenderItem>, LightingSettings) {
    let Some(plane_id) = app.wave_state.plane_id else {
        return (Vec::new(), LightingSettings::default());
    };

    let mut item = SceneRenderItem::default();
    item.mesh_id = plane_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    // PBR water-like surface. Dielectric (metallic=0), fairly smooth so the
    // wave crests catch specular highlights.
    item.material = {
        let mut m = Material::pbr([0.18, 0.42, 0.65], 0.0, 0.35);
        m.backface_policy = viewport_lib::BackfacePolicy::Identical;
        m
    };

    let mut items = vec![item];

    // Buoys (only meaningful when the wave is GPU-driven, because the
    // BuoyPlugin samples the wave plugin's GPU output buffer).
    if app.wave_state.show_buoys
        && app.wave_state.mode == DeformMode::Gpu
        && let Some(buoys_id) = app.wave_state.buoys_mesh_id
    {
        let mut buoy_item = SceneRenderItem::default();
        buoy_item.mesh_id = buoys_id;
        buoy_item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        buoy_item.material = Material::pbr([0.95, 0.6, 0.15], 0.4, 0.5);
        items.push(buoy_item);
    }

    // `direction` is the vector toward the light source. Z-up, light shines
    // down from a high-pitch position. Two opposing rim lights so wave crests
    // and troughs both stay readable as the surface oscillates.
    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [0.4, 0.3, 1.0],
    };
    sun.colour = [1.0, 0.96, 0.88];
    sun.intensity = 1.0;

    let mut fill = LightSource::default();
    fill.kind = LightKind::Directional {
        direction: [-0.5, -0.2, 0.4],
    };
    fill.colour = [0.65, 0.78, 1.0];
    fill.intensity = 0.3;

    let lighting = {
        let mut t = LightingSettings::default();
        t.lights = vec![sun, fill];
        t.hemisphere_intensity = 0.35;
        t.sky_colour = [0.85, 0.92, 1.0];
        t.ground_colour = [0.40, 0.35, 0.30];
        t
    };

    (items, lighting)
}

// ---------------------------------------------------------------------------
// Per-frame compute dispatch
// ---------------------------------------------------------------------------

pub(crate) fn submit_wave_items(
    app: &mut App,
    _fd: &mut viewport_lib::FrameData,
    renderer: &mut ViewportRenderer,
) {
    let dt = if app.wave_state.paused {
        0.0
    } else {
        app.wave_state.last_dt.max(1.0 / 240.0)
    };
    app.wave_state.time += dt;

    // Apply any pending mode switch before timing the work below; rebinding /
    // clearing the override happens at most once per switch and is cheap.
    sync_mode_to_renderer(app, renderer);

    // Time the wave-update step in isolation so the perf overlay reflects only
    // the path under test (compute + submit, or CPU math + write_buffer).
    let t0 = std::time::Instant::now();
    match app.wave_state.mode {
        DeformMode::Gpu => run_gpu_path(app, dt),
        DeformMode::Cpu => run_cpu_path(app, renderer),
    }
    let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
    // EMA so the overlay is readable instead of jittering wildly.
    let alpha = 0.1;
    app.wave_state.update_ms_smoothed =
        app.wave_state.update_ms_smoothed * (1.0 - alpha) + elapsed_ms * alpha;

    app.wave_state.frame_index = app.wave_state.frame_index.wrapping_add(1);
}

fn sync_mode_to_renderer(app: &mut App, renderer: &mut ViewportRenderer) {
    let Some(plane_id) = app.wave_state.plane_id else {
        return;
    };
    let want_bound = matches!(app.wave_state.mode, DeformMode::Gpu);
    if want_bound == app.wave_state.override_bound {
        return;
    }
    let resources = renderer.resources_mut();
    if want_bound {
        if let Some(plugin) = app.wave_state.plugin.as_deref() {
            let _ =
                resources.set_position_override_buffer(plane_id, plugin.output_buffer());
            let _ = resources.set_normal_override_buffer(plane_id, plugin.normal_buffer());
        }
    } else {
        let _ = resources.clear_position_override(plane_id);
        let _ = resources.clear_normal_override(plane_id);
    }
    app.wave_state.override_bound = want_bound;
    // Reset the EMA so the overlay snaps to the new path's cost.
    app.wave_state.update_ms_smoothed = 0.0;
}

fn run_gpu_path(app: &mut App, dt: f32) {
    // Drive both compute plugins BEFORE eframe's paint callback runs. The
    // wave plugin writes its position buffer; the buoy plugin reads that same
    // buffer and writes its own. Two separate queue.submit calls would also
    // work; bundling them in one keeps the chained-compute story tight.
    let ctx = viewport_lib::runtime::GpuFrameContext {
        camera: &app.camera,
        viewport_size: app.wave_state.viewport_size,
        dt,
        frame_index: app.wave_state.frame_index,
    };
    let mut bufs: Vec<wgpu::CommandBuffer> = Vec::new();
    if let Some(plugin) = app.wave_state.plugin.as_deref_mut() {
        plugin.set_amplitude(app.wave_state.amplitude);
        plugin.set_frequency(app.wave_state.frequency);
        bufs.extend(plugin.pre_prepare(&app.device, &app.queue, &ctx));
    }
    if app.wave_state.show_buoys {
        if let Some(bp) = app.wave_state.buoy_plugin.as_deref_mut() {
            // Ordering is what matters here: wgpu executes submitted command
            // buffers in submission order, so the buoy compute (appended
            // second) observes the wave compute's writes.
            bufs.extend(bp.pre_prepare(&app.device, &app.queue, &ctx));
        }
    }
    if !bufs.is_empty() {
        app.queue.submit(bufs);
    }
}

fn run_cpu_path(app: &mut App, renderer: &mut ViewportRenderer) {
    // Replicates the WGSL compute shader on the CPU and uploads the result via
    // `write_mesh_positions_normals`. Same math; the visible output is
    // identical to the GPU path. The cost difference is what the demo proves.
    let Some(plane_id) = app.wave_state.plane_id else {
        return;
    };
    let t = app.wave_state.time;
    let amp = app.wave_state.amplitude;
    let freq = app.wave_state.frequency;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(app.wave_state.rest_positions.len());
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(app.wave_state.rest_positions.len());
    for &[x, y, z_rest] in &app.wave_state.rest_positions {
        let phase_x = freq * x + t;
        let phase_y = freq * 1.7 * y + t * 1.3;
        let dz = amp * phase_x.sin() + amp * 0.5 * phase_y.sin();
        positions.push([x, y, z_rest + dz]);

        let dhdx = amp * freq * phase_x.cos();
        let dhdy = amp * 0.5 * freq * 1.7 * phase_y.cos();
        let nx = -dhdx;
        let ny = -dhdy;
        let nz = 1.0;
        let inv_len = (nx * nx + ny * ny + nz * nz).sqrt().recip();
        normals.push([nx * inv_len, ny * inv_len, nz * inv_len]);
    }

    // With overrides cleared, the standard mesh pipeline reads from the vertex
    // buffer attributes that this call updates.
    let _ = renderer
        .resources_mut()
        .write_mesh_positions_normals(&app.queue, plane_id, &positions, &normals);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_wave(app: &mut App, ui: &mut egui::Ui) {
    ui.label("GPU Wave (compute-plugin override)");
    ui.separator();
    ui.label(
        "Same wave math, two different deformation paths. Toggle the mode to\n\
         compare per-frame cost on a 40,401-vertex plane.",
    );
    ui.separator();

    ui.label("Deform path");
    ui.horizontal(|ui| {
        ui.radio_value(&mut app.wave_state.mode, DeformMode::Gpu, "GPU compute");
        ui.radio_value(&mut app.wave_state.mode, DeformMode::Cpu, "CPU writeback");
    });
    match app.wave_state.mode {
        DeformMode::Gpu => ui.label(
            "GpuPlugin::pre_prepare runs a compute shader; output buffer is\n\
             bound via set_position_override_buffer. PCIe traffic per frame:\n\
             one ~16-byte uniform.",
        ),
        DeformMode::Cpu => ui.label(
            "CPU computes positions + normals each frame and uploads via\n\
             write_mesh_positions_normals. PCIe traffic per frame:\n\
             ~944 KB (40,401 verts x 24 bytes).",
        ),
    };

    ui.separator();
    let verts = (app.wave_state.current_grid_dim + 1).pow(2);
    ui.label(format!(
        "wave update step: {:.2} ms (smoothed)",
        app.wave_state.update_ms_smoothed,
    ));
    ui.label(format!("active mesh: {} verts", verts));
    ui.label(
        "Just the deformation work, not the rest of the frame. The render\n\
         pass is the same on both paths.",
    );

    ui.separator();
    ui.label("Mesh resolution");
    // Edit the pending dimension as a buffer; only commit on button press so
    // dragging the slider doesn't trigger a rebuild every frame.
    let mut staged = app.wave_state.pending_grid_dim;
    ui.add(egui::Slider::new(&mut staged, 50..=900).text("grid dim (staged)"));
    if staged != app.wave_state.pending_grid_dim {
        app.wave_state.pending_grid_dim = staged;
    }
    if app.wave_state.pending_grid_dim != app.wave_state.current_grid_dim {
        ui.label(format!(
            "pending: {} verts",
            (app.wave_state.pending_grid_dim + 1).pow(2),
        ));
        if ui.button("Rebuild mesh").clicked() {
            app.wave_state.built = false;
        }
    }

    ui.separator();
    ui.label("Chained compute");
    ui.checkbox(&mut app.wave_state.show_buoys, "Show floating buoys");
    ui.label(format!(
        "{} buoys: a 2nd GpuPlugin reads the wave plugin's GPU output buffer\n\
         and writes per-vertex positions for a separate buoys mesh. No CPU\n\
         intermediate. Hidden when the wave is in CPU mode because the wave's\n\
         GPU buffer is stale in that case.",
        BUOY_GRID * BUOY_GRID,
    ));

    ui.separator();
    ui.checkbox(&mut app.wave_state.paused, "Pause animation");
    ui.add(
        egui::Slider::new(&mut app.wave_state.amplitude, 0.0..=1.5)
            .text("Amplitude")
            .step_by(0.01),
    );
    ui.add(
        egui::Slider::new(&mut app.wave_state.frequency, 0.2..=6.0)
            .text("Frequency")
            .step_by(0.01),
    );
}
