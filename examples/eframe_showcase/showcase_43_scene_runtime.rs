//! Showcase 43: Scene Runtime
//!
//! Demonstrates `ViewportRuntime` in isolation, before any built-in interaction systems
//! exist. Five bodies orbit a common center driven by an external `RuntimePlugin` that
//! writes `NodeTransformOp`s via `TransformWriteback`. All other phases are empty.
//!
//! Controls:
//! - Simulation rate (Hz) slider: change the fixed timestep at runtime.
//! - Interpolation toggle: compare choppy fixed-step rendering vs. smooth interpolation.
//! - Step index and alpha blend factor displayed each frame.

use eframe::egui;
use viewport_lib::{
    FixedTimestep, Material, MeshId,
    RuntimeFrameContext, RuntimePlugin, RuntimeStepContext, SceneRenderItem,
    ViewportRuntime,
    runtime::plugin::phase,
    scene::Scene,
    selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// Orbit plugin
// ---------------------------------------------------------------------------

/// An external `RuntimePlugin` that moves five bodies in circular orbits.
///
/// Each body has a radius, angular speed, and height. On every `Simulate` step
/// the plugin computes the new world position and writes a `NodeTransformOp` via
/// `ctx.writeback.set()`. The scene's local transforms are updated by the runtime
/// during writeback.
struct OrbitPlugin {
    /// Per-body state: (NodeId index, radius, speed rad/s, current angle rad, height).
    bodies: Vec<(usize, f32, f32, f32, f32)>,
}

impl OrbitPlugin {
    fn new() -> Self {
        Self {
            bodies: vec![
                (0, 3.0, 1.2, 0.0, 0.0),
                (1, 4.5, 0.8, 1.26, 1.0),
                (2, 2.0, 2.0, 2.51, -0.5),
                (3, 5.5, 0.5, 3.77, 0.8),
                (4, 3.5, 1.5, 5.03, -1.2),
            ],
        }
    }
}

impl RuntimePlugin for OrbitPlugin {
    fn priority(&self) -> i32 {
        phase::SIMULATE
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext) {
        let node_ids = ctx
            .scene
            .walk_depth_first()
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>();

        for (idx, radius, speed, angle, height) in &mut self.bodies {
            *angle += *speed * ctx.dt;

            let x = angle.cos() * *radius;
            let y = angle.sin() * *radius;
            let z = *height;

            if let Some(&node_id) = node_ids.get(*idx) {
                let t = glam::Affine3A::from_translation(glam::Vec3::new(x, y, z));
                ctx.writeback.set(node_id, t);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct RtDemoState {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub runtime: ViewportRuntime,
    pub mesh_id: Option<MeshId>,
    pub sim_fps: f32,
    pub interpolate: bool,
}

impl Default for RtDemoState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            runtime: ViewportRuntime::new()
                .with_fixed_timestep(FixedTimestep::new(15.0))
                .with_plugin(OrbitPlugin::new()),
            mesh_id: None,
            sim_fps: 15.0,
            interpolate: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

pub(crate) fn build_rt_demo_scene(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    let sphere = viewport_lib::primitives::sphere(0.6, 24, 16);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &sphere)
        .expect("rt demo sphere upload");
    app.rt_state.mesh_id = Some(mesh_id);

    // Five orbit bodies, each a different colour.
    let colours: [[f32; 3]; 5] = [
        [0.9, 0.35, 0.3],
        [0.3, 0.75, 0.9],
        [0.5, 0.9, 0.4],
        [0.9, 0.75, 0.2],
        [0.75, 0.4, 0.9],
    ];
    for colour in &colours {
        let mat = Material::from_colour(*colour);
        app.rt_state
            .scene
            .add(Some(mesh_id), glam::Mat4::IDENTITY, mat);
    }

    app.rt_state.built = true;
}

// ---------------------------------------------------------------------------
// Per-frame update (called from main update loop)
// ---------------------------------------------------------------------------

pub(crate) fn update_rt_demo(app: &mut App, dt: f32) {
    let camera = app.camera.clone();
    let mut frame_ctx = RuntimeFrameContext::default();
    frame_ctx.dt = dt;
    frame_ctx.camera = camera.clone();
    frame_ctx.viewport_size = glam::Vec2::new(800.0, 600.0);
    app.rt_state.runtime.step(
        &mut app.rt_state.scene,
        &mut app.rt_state.selection,
        &frame_ctx,
    );
}

// ---------------------------------------------------------------------------
// Per-frame scene items
// ---------------------------------------------------------------------------

pub(crate) fn rt_demo_scene_items(app: &mut App) -> Vec<SceneRenderItem> {
    let Some(mesh_id) = app.rt_state.mesh_id else {
        return Vec::new();
    };

    if app.rt_state.interpolate {
        // Interpolated: use snapshot table to blend between prev and curr transforms.
        let alpha = app.rt_state.runtime.alpha();
        let node_ids: Vec<_> = app.rt_state.scene.walk_depth_first();
        let mut items = Vec::new();
        for (node_id, _) in &node_ids {
            let node_id = *node_id;
            let (material, world) = match app.rt_state.scene.node(node_id) {
                Some(n) => (n.material().clone(), n.world_transform()),
                None => continue,
            };
            let model = if let Some(t) = app.rt_state.runtime.snapshots().interpolated(node_id, alpha)
            {
                glam::Mat4::from(t)
            } else {
                world
            };
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = model.to_cols_array_2d();
            item.material = material;
            items.push(item);
        }
        items
    } else {
        // No interpolation: render directly from scene transforms (choppy at low Hz).
        app.rt_state
            .scene
            .collect_render_items(&app.rt_state.selection)
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_rt_demo(app: &mut App, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label("Fixed-timestep orbit simulation driven by an external RuntimePlugin.");
        ui.add_space(6.0);

        // Simulation fps slider.
        ui.label("Simulation fps:");
        let mut fps = app.rt_state.sim_fps;
        if ui
            .add(egui::Slider::new(&mut fps, 2.0..=120.0).logarithmic(true))
            .changed()
        {
            app.rt_state.sim_fps = fps;
            app.rt_state
                .runtime
                .set_fixed_timestep(FixedTimestep::new(fps));
        }
        ui.add_space(4.0);

        // Interpolation toggle.
        ui.checkbox(&mut app.rt_state.interpolate, "Interpolate transforms");
        ui.small(if app.rt_state.interpolate {
            "Smooth: renders at display rate using snapshot lerp/slerp."
        } else {
            "Choppy: renders only at simulation rate (jitter visible at low sim fps)."
        });
        ui.add_space(8.0);

        // Live stats.
        ui.separator();
        ui.label("Runtime state:");
        ui.label(format!(
            "Step index : {}",
            app.rt_state.runtime.step_index()
        ));
        ui.label(format!("Alpha      : {:.3}", app.rt_state.runtime.alpha()));
        ui.add_space(4.0);

        ui.separator();
        ui.label("What this shows:");
        ui.small("- RuntimePlugin trait: external code drives transforms via TransformWriteback.");
        ui.small("- FixedTimestep: simulation runs at a fixed Hz independent of frame rate.");
        ui.small("- TransformSnapshotTable: interpolation removes jitter between fixed steps.");
    });
}
