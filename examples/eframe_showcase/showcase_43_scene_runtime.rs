//! Showcase 43: Scene Runtime
//!
//! Two demos showing the ViewportRuntime layer.
//!
//! Orbit: five bodies orbit a common center driven by an external RuntimePlugin
//! that writes transforms via TransformWriteback. All other phases are empty.
//!
//! Simulation: five spheres fall and bounce under gravity (PhysicsLitePlugin);
//! a sixth follows a circular keyframe path (AnimationPlugin). Camera follow
//! can be bound to any physics body.
//!
//! Both demos use a fixed timestep accumulator with a configurable sim fps.
//! The interpolation toggle compares smooth display-rate rendering against
//! choppy fixed-step rendering.

use eframe::egui;
use viewport_lib::{
    Aabb, AnimationPlugin, AnimationTrack, CameraFollow, FixedTimestep, Keyframe,
    Material, MeshId, PhysicsBody, PhysicsLitePlugin,
    RuntimeFrameContext, RuntimePlugin, RuntimeStepContext, SceneRenderItem,
    ViewportRuntime,
    camera::Camera,
    runtime::plugin::phase,
    scene::Scene,
    selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// Demo selector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeDemo {
    Orbit,
    Simulation,
}

// ---------------------------------------------------------------------------
// Orbit plugin
// ---------------------------------------------------------------------------

struct OrbitPlugin {
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
    // Demo selector
    pub demo: RuntimeDemo,
    pub active_demo: RuntimeDemo,
    // Simulation demo state
    pub paused: bool,
    pub step_once: bool,
    pub camera_follow: bool,
    pub follow_body: usize,
    pub physics_node_ids: Vec<u64>,
    pub anim_node_id: Option<u64>,
}

impl Default for RtDemoState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            runtime: ViewportRuntime::new()
                .with_fixed_timestep(FixedTimestep::new(60.0)),
            mesh_id: None,
            sim_fps: 45.0,
            interpolate: true,
            demo: RuntimeDemo::Orbit,
            active_demo: RuntimeDemo::Orbit,
            paused: false,
            step_once: false,
            camera_follow: false,
            follow_body: 0,
            physics_node_ids: Vec::new(),
            anim_node_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene population helpers
// ---------------------------------------------------------------------------

fn populate_orbit(app: &mut App) {
    app.rt_state.scene = Scene::new();
    app.rt_state.sim_fps = 15.0;
    let Some(mesh_id) = app.rt_state.mesh_id else { return };
    let colours: [[f32; 3]; 5] = [
        [0.9, 0.35, 0.3],
        [0.3, 0.75, 0.9],
        [0.5, 0.9, 0.4],
        [0.9, 0.75, 0.2],
        [0.75, 0.4, 0.9],
    ];
    for colour in &colours {
        app.rt_state
            .scene
            .add(Some(mesh_id), glam::Mat4::IDENTITY, Material::from_colour(*colour));
    }
    app.rt_state.runtime = ViewportRuntime::new()
        .with_fixed_timestep(FixedTimestep::new(app.rt_state.sim_fps))
        .with_plugin(OrbitPlugin::new());
}

fn populate_simulation(app: &mut App) {
    app.rt_state.scene = Scene::new();
    app.rt_state.sim_fps = 60.0;
    app.rt_state.paused = false;
    app.rt_state.step_once = false;
    app.rt_state.camera_follow = false;
    app.rt_state.physics_node_ids.clear();
    app.rt_state.anim_node_id = None;

    let Some(mesh_id) = app.rt_state.mesh_id else { return };
    let scene = &mut app.rt_state.scene;

    let body_colours: [[f32; 3]; 5] = [
        [0.9, 0.3, 0.3],
        [0.3, 0.75, 0.9],
        [0.5, 0.9, 0.4],
        [0.9, 0.75, 0.2],
        [0.75, 0.35, 0.9],
    ];
    let body_starts: [(f32, f32, f32); 5] = [
        (0.0, 0.0, 5.0),
        (1.5, -1.0, 7.0),
        (-2.0, 1.5, 4.0),
        (2.5, 2.0, 6.0),
        (-1.0, -2.5, 3.0),
    ];
    let body_velocities: [(f32, f32, f32); 5] = [
        (1.5, 0.8, 2.0),
        (-1.0, 1.5, 1.0),
        (2.0, -0.5, 0.5),
        (-1.5, -1.0, 1.5),
        (0.5, 2.0, 3.0),
    ];
    let bounds = Aabb {
        min: glam::Vec3::new(-4.5, -4.5, 0.0),
        max: glam::Vec3::new(4.5, 4.5, 9.0),
    };

    let mut physics = PhysicsLitePlugin::new()
        .with_gravity(glam::Vec3::new(0.0, 0.0, -9.81));

    let mut node_ids = Vec::new();
    for (i, colour) in body_colours.iter().enumerate() {
        let (x, y, z) = body_starts[i];
        let pos = glam::Vec3::new(x, y, z);
        let id = scene.add(
            Some(mesh_id),
            glam::Mat4::from_translation(pos),
            Material::from_colour(*colour),
        );
        node_ids.push(id);
        let (vx, vy, vz) = body_velocities[i];
        physics.add_body(
            PhysicsBody::new(id)
                .with_velocity(glam::Vec3::new(vx, vy, vz))
                .with_restitution(0.65 + i as f32 * 0.04)
                .with_bounds(bounds),
        );
    }
    app.rt_state.physics_node_ids = node_ids;

    let anim_id = app.rt_state.scene.add(
        Some(mesh_id),
        glam::Mat4::IDENTITY,
        Material::from_colour([0.95, 0.95, 0.95]),
    );
    app.rt_state.anim_node_id = Some(anim_id);

    let radius = 3.0_f32;
    let height = 5.0_f32;
    let num_kf = 8;
    let duration = 5.0_f32;
    let mut keyframes = Vec::new();
    for i in 0..=num_kf {
        let frac = i as f32 / num_kf as f32;
        let angle = frac * std::f32::consts::TAU;
        let pos = glam::Vec3::new(angle.cos() * radius, angle.sin() * radius, height);
        keyframes.push(Keyframe {
            time: frac * duration,
            transform: glam::Affine3A::from_translation(pos),
        });
    }
    let mut anim = AnimationPlugin::new();
    anim.add_track(AnimationTrack {
        node_id: anim_id,
        keyframes,
        looping: true,
    });

    app.rt_state.runtime = ViewportRuntime::new()
        .with_fixed_timestep(FixedTimestep::new(app.rt_state.sim_fps))
        .with_plugin(physics)
        .with_plugin(anim);
}

// ---------------------------------------------------------------------------
// Scene construction (called once when the showcase is first loaded)
// ---------------------------------------------------------------------------

pub(crate) fn build_rt_demo_scene(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    let sphere = viewport_lib::primitives::sphere(0.5, 20, 14);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &sphere)
        .expect("rt demo sphere upload");
    app.rt_state.mesh_id = Some(mesh_id);
    populate_orbit(app);
    app.rt_state.active_demo = RuntimeDemo::Orbit;
    app.rt_state.built = true;
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

pub(crate) fn update_rt_demo(app: &mut App, dt: f32) {
    // Rebuild if the demo selector changed.
    if app.rt_state.demo != app.rt_state.active_demo {
        match app.rt_state.demo {
            RuntimeDemo::Orbit => {
                populate_orbit(app);
                app.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 14.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            RuntimeDemo::Simulation => {
                populate_simulation(app);
                app.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 4.5),
                    distance: 20.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
        }
        app.rt_state.active_demo = app.rt_state.demo;
    }

    match app.rt_state.demo {
        RuntimeDemo::Orbit => step_orbit(app, dt),
        RuntimeDemo::Simulation => step_simulation(app, dt),
    }
}

fn step_orbit(app: &mut App, dt: f32) {
    let camera = app.camera.clone();
    let mut frame_ctx = RuntimeFrameContext::default();
    frame_ctx.dt = dt;
    frame_ctx.camera = camera;
    frame_ctx.viewport_size = glam::Vec2::new(800.0, 600.0);
    app.rt_state.runtime.step(
        &mut app.rt_state.scene,
        &mut app.rt_state.selection,
        &frame_ctx,
    );
}

fn step_simulation(app: &mut App, dt: f32) {
    let effective_dt = if app.rt_state.step_once {
        app.rt_state.step_once = false;
        app.rt_state.paused = true;
        1.0 / app.rt_state.sim_fps
    } else if app.rt_state.paused {
        0.0
    } else {
        dt
    };

    let camera = app.camera.clone();
    let mut frame_ctx = RuntimeFrameContext::default();
    frame_ctx.dt = effective_dt;
    frame_ctx.camera = camera;
    frame_ctx.viewport_size = glam::Vec2::new(800.0, 600.0);

    let output = app.rt_state.runtime.step(
        &mut app.rt_state.scene,
        &mut app.rt_state.selection,
        &frame_ctx,
    );

    if app.rt_state.camera_follow {
        if let Some(target) = output.camera_follow_target {
            app.camera.center = target;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame scene items
// ---------------------------------------------------------------------------

pub(crate) fn rt_demo_scene_items(app: &mut App) -> Vec<SceneRenderItem> {
    let Some(mesh_id) = app.rt_state.mesh_id else {
        return Vec::new();
    };

    if app.rt_state.interpolate {
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
        // Demo selector.
        ui.horizontal(|ui| {
            for demo in [RuntimeDemo::Orbit, RuntimeDemo::Simulation] {
                let label = match demo {
                    RuntimeDemo::Orbit => "Orbit",
                    RuntimeDemo::Simulation => "Simulation",
                };
                if ui.selectable_label(app.rt_state.demo == demo, label).clicked() {
                    app.rt_state.demo = demo;
                }
            }
        });
        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        match app.rt_state.demo {
            RuntimeDemo::Orbit => controls_orbit(app, ui),
            RuntimeDemo::Simulation => controls_simulation(app, ui),
        }
    });
}

fn controls_orbit(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Five bodies orbit a common center, driven by an external RuntimePlugin.");
    ui.add_space(6.0);

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

    ui.checkbox(&mut app.rt_state.interpolate, "Interpolate transforms");
    ui.label(if app.rt_state.interpolate {
        "Smooth: renders at display rate using snapshot lerp/slerp."
    } else {
        "Choppy: renders only at simulation rate (jitter visible at low sim fps)."
    });
    ui.add_space(8.0);

    ui.separator();
    ui.label("Runtime state:");
    ui.label(format!("Step index : {}", app.rt_state.runtime.step_index()));
    ui.label(format!("Alpha      : {:.3}", app.rt_state.runtime.alpha()));
    ui.add_space(4.0);

    ui.separator();
    ui.label("What this shows:");
    ui.label("- RuntimePlugin trait: external code drives transforms via TransformWriteback.");
    ui.label("- FixedTimestep: simulation runs at a fixed Hz independent of frame rate.");
    ui.label("- TransformSnapshotTable: interpolation removes jitter between fixed steps.");
}

fn controls_simulation(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Five physics spheres fall and bounce (PhysicsLitePlugin).");
    ui.label("White sphere follows a circular keyframe path (AnimationPlugin).");
    ui.add_space(6.0);

    ui.separator();

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

    ui.horizontal(|ui| {
        let pause_label = if app.rt_state.paused { "Resume" } else { "Pause" };
        if ui.button(pause_label).clicked() {
            app.rt_state.paused = !app.rt_state.paused;
        }
        if ui.button("Step Once").clicked() {
            app.rt_state.step_once = true;
            app.rt_state.paused = false;
        }
    });
    ui.add_space(4.0);

    ui.checkbox(&mut app.rt_state.interpolate, "Interpolate transforms");
    ui.label(if app.rt_state.interpolate {
        "Smooth: snapshot lerp/slerp at display rate."
    } else {
        "Choppy: renders only at sim fps (jitter visible at low fps)."
    });
    ui.add_space(6.0);

    ui.separator();

    let mut follow = app.rt_state.camera_follow;
    if ui.checkbox(&mut follow, "Camera follow").changed() {
        app.rt_state.camera_follow = follow;
        let ids = app.rt_state.physics_node_ids.clone();
        let idx = app.rt_state.follow_body;
        if follow {
            if let Some(&id) = ids.get(idx) {
                app.rt_state.runtime.set_camera_follow(CameraFollow::Node {
                    id,
                    offset: glam::Vec3::ZERO,
                    look_at: true,
                });
            }
        } else {
            app.rt_state.runtime.clear_camera_follow();
        }
    }
    if app.rt_state.camera_follow {
        let n = app.rt_state.physics_node_ids.len();
        let mut idx = app.rt_state.follow_body;
        if ui
            .add(egui::Slider::new(&mut idx, 0..=n.saturating_sub(1)).text("Body"))
            .changed()
        {
            app.rt_state.follow_body = idx;
            let ids = app.rt_state.physics_node_ids.clone();
            if let Some(&id) = ids.get(idx) {
                app.rt_state.runtime.set_camera_follow(CameraFollow::Node {
                    id,
                    offset: glam::Vec3::ZERO,
                    look_at: true,
                });
            }
        }
    }
    ui.add_space(6.0);

    ui.separator();
    ui.label("Runtime state:");
    ui.label(format!("Step index : {}", app.rt_state.runtime.step_index()));
    ui.label(format!("Alpha      : {:.3}", app.rt_state.runtime.alpha()));
    if app.rt_state.paused {
        ui.label("Paused");
    }
    ui.add_space(4.0);

    ui.separator();
    ui.label("What this shows:");
    ui.label("- PhysicsLitePlugin: gravity, velocity integration, bounded reflection.");
    ui.label("- AnimationPlugin: keyframed looping path.");
    ui.label("- CameraFollow: orbit camera center tracks a physics body.");
    ui.label("- FixedTimestep + interpolation: smooth rendering at any display fps.");
}
