//! Showcase 46: Debug Draw
//!
//! Demonstrates the debug draw runtime visualization system. A post-simulation
//! plugin reads contact events and scene node positions each frame, then submits
//! primitives to a shared DebugDraw resource:
//!
//! - AABB wireframes (green) around each physics body.
//! - Contact normal line segments (red) and contact point markers.
//! - Per-body index labels.
//! - Persistent overlay AABB (yellow) showing the bounding region.
//!
//! Controls:
//! - Dev mode toggle: hides/shows dev-layer visuals (AABBs, normals, labels).
//! - Pause/Resume: freeze the simulation.

use eframe::egui;
use viewport_lib::{
    Aabb, DebugDraw, DebugLayer, DebugPrim, FixedTimestep, Material, MeshId,
    PhysicsBody, PhysicsLitePlugin, RuntimeFrameContext, RuntimePlugin, RuntimeStepContext,
    SceneRenderItem, ViewportRuntime,
    runtime::plugin::phase,
    scene::Scene,
    selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// ID for the persistent overlay AABB
// ---------------------------------------------------------------------------

const BOUNDS_AABB_ID: u64 = 1;

// ---------------------------------------------------------------------------
// Debug draw plugin
// ---------------------------------------------------------------------------

/// Reads scene positions and contact events after physics runs, then writes
/// debug primitives to the shared [`DebugDraw`] resource.
struct DebugOverlayPlugin {
    body_ids: Vec<u64>,
    body_radius: f32,
    bounds: Aabb,
}

impl DebugOverlayPlugin {
    fn new(body_ids: Vec<u64>, body_radius: f32, bounds: Aabb) -> Self {
        Self { body_ids, body_radius, bounds }
    }
}

impl RuntimePlugin for DebugOverlayPlugin {
    fn priority(&self) -> i32 {
        // Run after physics (POST_SIM) so contact events are already in output.
        phase::POST_SIM + 5
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext) {
        let Some(dd) = ctx.resources.get_mut::<DebugDraw>() else {
            return;
        };

        // AABB wireframe and index label around each body (dev layer).
        for (i, &id) in self.body_ids.iter().enumerate() {
            let Some(node) = ctx.scene.node(id) else {
                continue;
            };
            let center = node.world_transform().col(3).truncate();
            let half = glam::Vec3::splat(self.body_radius);
            dd.aabb(center - half, center + half, [0.3, 0.9, 0.4, 1.0]);
            dd.label(
                center + glam::Vec3::Z * (self.body_radius + 0.12),
                format!("body {}", i),
                [1.0, 1.0, 1.0, 0.85],
            );
        }

        // Contact normals and markers (dev layer).
        for contact in &ctx.output.contact_events {
            let cp = contact.contact_point;
            let normal_tip = cp + contact.world_normal * 0.5;
            // Normal direction line.
            dd.line(cp, normal_tip, [1.0, 0.25, 0.25, 1.0]);
            // Contact point marker.
            dd.point(cp, 6.0, [1.0, 0.2, 0.2, 1.0]);
        }

        // Persistent bounding region AABB (overlay layer, always visible).
        if !dd.has_persistent(BOUNDS_AABB_ID) {
            dd.add_persistent(BOUNDS_AABB_ID, DebugPrim::Aabb {
                min: self.bounds.min,
                max: self.bounds.max,
                colour: [0.9, 0.75, 0.2, 0.6],
                layer: DebugLayer::Overlay,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct DbgDrawState {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub runtime: ViewportRuntime,
    pub sphere_mesh: Option<MeshId>,
    pub paused: bool,
    pub dev_enabled: bool,
    pub contact_count: usize,
}

impl Default for DbgDrawState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            runtime: ViewportRuntime::new()
                .with_fixed_timestep(FixedTimestep::new(60.0)),
            sphere_mesh: None,
            paused: false,
            dev_enabled: true,
            contact_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

pub(crate) fn build_dbg_draw_scene(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    let sphere_r = 0.35_f32;
    let sphere = viewport_lib::primitives::sphere(sphere_r, 16, 12);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &sphere)
        .expect("dbg draw sphere upload");
    app.dbg_draw_state.sphere_mesh = Some(mesh_id);

    let scene = &mut app.dbg_draw_state.scene;

    let bounds = Aabb {
        min: glam::Vec3::new(-4.0, -4.0, 0.0),
        max: glam::Vec3::new(4.0, 4.0, 8.0),
    };

    let body_starts: [(f32, f32, f32); 4] = [
        (0.0, 0.0, 5.0),
        (2.0, -1.0, 6.0),
        (-1.5, 1.5, 3.5),
        (1.0, 2.0, 5.5),
    ];
    let body_velocities: [(f32, f32, f32); 4] = [
        (1.2, 0.6, 2.0),
        (-1.0, 1.4, 1.0),
        (1.8, -0.5, 0.5),
        (-0.8, -1.2, 1.5),
    ];
    let colours: [[f32; 3]; 4] = [
        [0.85, 0.35, 0.3],
        [0.3, 0.7, 0.9],
        [0.5, 0.85, 0.4],
        [0.9, 0.7, 0.2],
    ];

    let mut physics = PhysicsLitePlugin::new()
        .with_gravity(glam::Vec3::new(0.0, 0.0, -9.81));

    let mut node_ids = Vec::new();
    for (i, colour) in colours.iter().enumerate() {
        let (x, y, z) = body_starts[i];
        let transform = glam::Mat4::from_translation(glam::Vec3::new(x, y, z));
        let mat = Material::from_colour(*colour);
        let id = scene.add(Some(mesh_id), transform, mat);
        node_ids.push(id);

        let (vx, vy, vz) = body_velocities[i];
        physics.add_body(
            PhysicsBody::new(id)
                .with_velocity(glam::Vec3::new(vx, vy, vz))
                .with_restitution(0.7)
                .with_bounds(bounds),
        );
    }

    let debug_plugin = DebugOverlayPlugin::new(node_ids, sphere_r, bounds);

    app.dbg_draw_state.runtime = ViewportRuntime::new()
        .with_fixed_timestep(FixedTimestep::new(60.0))
        .with_plugin(physics)
        .with_plugin(debug_plugin);

    // Pre-insert the DebugDraw resource so plugins can access it on the first step.
    app.dbg_draw_state.runtime.resources_mut().insert(DebugDraw::new());

    app.dbg_draw_state.built = true;
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

pub(crate) fn update_dbg_draw(app: &mut App, dt: f32) {
    let effective_dt = if app.dbg_draw_state.paused { 0.0 } else { dt };

    // Sync dev_enabled and clear transient draws before the step.
    if let Some(dd) = app.dbg_draw_state.runtime.resources_mut().get_mut::<DebugDraw>() {
        dd.dev_enabled = app.dbg_draw_state.dev_enabled;
        dd.begin_frame();
    }

    let camera = app.camera.clone();
    let mut frame_ctx = RuntimeFrameContext::default();
    frame_ctx.dt = effective_dt;
    frame_ctx.camera = camera.clone();
    frame_ctx.viewport_size = glam::Vec2::new(800.0, 600.0);

    let output = app.dbg_draw_state.runtime.step(
        &mut app.dbg_draw_state.scene,
        &mut app.dbg_draw_state.selection,
        &frame_ctx,
    );

    app.dbg_draw_state.contact_count = output.contact_events.len();
}

// ---------------------------------------------------------------------------
// Scene items
// ---------------------------------------------------------------------------

pub(crate) fn dbg_draw_scene_items(app: &mut App) -> Vec<SceneRenderItem> {
    app.dbg_draw_state
        .scene
        .collect_render_items(&app.dbg_draw_state.selection)
}

// ---------------------------------------------------------------------------
// Render extras
// ---------------------------------------------------------------------------

/// Push debug draw polylines, point cloud, and labels into the frame data.
pub(crate) fn submit_dbg_draw_items(app: &App, fd: &mut viewport_lib::FrameData) {
    let Some(dd) = app.dbg_draw_state.runtime.resources().get::<DebugDraw>() else {
        return;
    };
    fd.scene.polylines.extend(dd.to_polylines());
    if let Some(pc) = dd.to_point_cloud() {
        fd.scene.point_clouds.push(pc);
    }
    fd.overlays.labels.extend(dd.to_labels());
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_dbg_draw(app: &mut App, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label("Four physics spheres with runtime debug visualization.");
        ui.add_space(6.0);

        ui.separator();

        let pause_label = if app.dbg_draw_state.paused { "Resume" } else { "Pause" };
        if ui.button(pause_label).clicked() {
            app.dbg_draw_state.paused = !app.dbg_draw_state.paused;
        }
        ui.add_space(6.0);

        ui.separator();
        ui.label("Visualization layers:");

        let mut dev = app.dbg_draw_state.dev_enabled;
        if ui.checkbox(&mut dev, "Dev layer visuals").changed() {
            app.dbg_draw_state.dev_enabled = dev;
        }
        ui.label("Dev layer: AABB wireframes, contact normals, point markers, body labels.");
        ui.label("The yellow bounding region is overlay-layer and always shown.");
        ui.add_space(6.0);

        ui.separator();
        ui.label("Frame stats:");
        ui.label(format!("Contacts this step: {}", app.dbg_draw_state.contact_count));
        if let Some(dd) = app.dbg_draw_state.runtime.resources().get::<DebugDraw>() {
            ui.label(format!("Transient prims : {}", dd.transient_count()));
            ui.label(format!("Persistent prims: {}", dd.persistent_count()));
        }
        ui.add_space(6.0);

        ui.separator();
        ui.label("What this shows:");
        ui.label("- DebugDraw stored in RuntimeResources, accessed by plugins via ctx.resources.");
        ui.label("- begin_frame() clears transient draws; persistent draws survive across frames.");
        ui.label("- Dev layer suppressed when dev_enabled = false (ship mode).");
        ui.label("- Overlay layer always shown regardless of dev_enabled.");
        ui.label("- to_polylines(), to_point_cloud(), to_labels() convert to render items.");
    });
}
