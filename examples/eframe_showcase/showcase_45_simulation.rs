//! Showcase 45: Simulation and Animation
//!
//! Demonstrates PhysicsLitePlugin, AnimationPlugin, and CameraFollow together.
//!
//! Five spheres fall under gravity and bounce inside a bounding box, driven by
//! PhysicsLitePlugin in the Simulate phase with a fixed timestep. A sixth sphere
//! follows a looping circular path, driven by AnimationPlugin in the Animate phase.
//! Camera follow can be bound to any physics body via the controls panel.
//!
//! Controls:
//! - Simulation fps slider
//! - Pause / step-once buttons
//! - Interpolation toggle (compare smooth vs. choppy at low sim fps)
//! - Camera follow toggle (orbit camera tracks a bouncing sphere)

use eframe::egui;
use viewport_lib::{
    Aabb, AnimationPlugin, AnimationTrack, CameraFollow, FixedTimestep, Keyframe,
    Material, MeshId, PhysicsBody, PhysicsLitePlugin, RuntimeFrameContext, SceneRenderItem,
    ViewportRuntime,
    scene::Scene,
    selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct Sim45State {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub runtime: ViewportRuntime,
    pub sphere_mesh: Option<MeshId>,
    pub sim_fps: f32,
    pub paused: bool,
    pub step_once: bool,
    pub interpolate: bool,
    pub camera_follow: bool,
    pub follow_body: usize,
    pub physics_node_ids: Vec<u64>,
    pub anim_node_id: Option<u64>,
}

impl Default for Sim45State {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            runtime: ViewportRuntime::new()
                .with_fixed_timestep(FixedTimestep::new(60.0)),
            sphere_mesh: None,
            sim_fps: 60.0,
            paused: false,
            step_once: false,
            interpolate: true,
            camera_follow: false,
            follow_body: 0,
            physics_node_ids: Vec::new(),
            anim_node_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

pub(crate) fn build_sim45_scene(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    let sphere = viewport_lib::primitives::sphere(0.4, 16, 12);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &sphere)
        .expect("sim45 sphere upload");
    app.sim45_state.sphere_mesh = Some(mesh_id);

    let scene = &mut app.sim45_state.scene;

    // Physics bodies: five spheres at staggered heights with initial velocities.
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
        let transform = glam::Mat4::from_translation(pos);
        let mat = Material::from_colour(*colour);
        let id = scene.add(Some(mesh_id), transform, mat);
        node_ids.push(id);

        let (vx, vy, vz) = body_velocities[i];
        physics.add_body(
            PhysicsBody::new(id)
                .with_velocity(glam::Vec3::new(vx, vy, vz))
                .with_restitution(0.65 + i as f32 * 0.04)
                .with_bounds(bounds),
        );
    }
    app.sim45_state.physics_node_ids = node_ids;

    // Animated body: a different colour on a circular path.
    let anim_mat = Material::from_colour([0.95, 0.95, 0.95]);
    let anim_id = scene.add(
        Some(mesh_id),
        glam::Mat4::IDENTITY,
        anim_mat,
    );
    app.sim45_state.anim_node_id = Some(anim_id);

    // Circular keyframe path: eight keyframes around a circle of radius 3, at Z=5.
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

    // Build the runtime with physics and animation plugins.
    app.sim45_state.runtime = ViewportRuntime::new()
        .with_fixed_timestep(FixedTimestep::new(app.sim45_state.sim_fps))
        .with_plugin(physics)
        .with_plugin(anim);

    app.sim45_state.built = true;
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

pub(crate) fn update_sim45(app: &mut App, dt: f32) {
    // Determine effective dt.
    let effective_dt = if app.sim45_state.step_once {
        app.sim45_state.step_once = false;
        app.sim45_state.paused = true;
        1.0 / app.sim45_state.sim_fps
    } else if app.sim45_state.paused {
        0.0
    } else {
        dt
    };

    let camera = app.camera.clone();
    let mut frame_ctx = RuntimeFrameContext::default();
    frame_ctx.dt = effective_dt;
    frame_ctx.camera = camera.clone();
    frame_ctx.viewport_size = glam::Vec2::new(800.0, 600.0);

    let output = app.sim45_state.runtime.step(
        &mut app.sim45_state.scene,
        &mut app.sim45_state.selection,
        &frame_ctx,
    );

    // Apply camera follow if enabled.
    if app.sim45_state.camera_follow {
        if let Some(target) = output.camera_follow_target {
            app.camera.center = target;
        }
    }
}

// ---------------------------------------------------------------------------
// Scene items
// ---------------------------------------------------------------------------

pub(crate) fn sim45_scene_items(app: &mut App) -> Vec<SceneRenderItem> {
    let Some(mesh_id) = app.sim45_state.sphere_mesh else {
        return Vec::new();
    };

    if app.sim45_state.interpolate {
        let alpha = app.sim45_state.runtime.alpha();
        let nodes = app.sim45_state.scene.walk_depth_first();
        let mut items = Vec::new();
        for (node_id, _) in &nodes {
            let node_id = *node_id;
            let (material, world) = match app.sim45_state.scene.node(node_id) {
                Some(n) => (n.material().clone(), n.world_transform()),
                None => continue,
            };
            let model = if let Some(t) =
                app.sim45_state.runtime.snapshots().interpolated(node_id, alpha)
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
        app.sim45_state
            .scene
            .collect_render_items(&app.sim45_state.selection)
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_sim45(app: &mut App, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label("Five physics spheres fall and bounce (PhysicsLitePlugin).");
        ui.label("White sphere follows a circular keyframe path (AnimationPlugin).");
        ui.add_space(6.0);

        ui.separator();

        // Sim fps slider.
        ui.label("Simulation fps:");
        let mut fps = app.sim45_state.sim_fps;
        if ui
            .add(egui::Slider::new(&mut fps, 2.0..=120.0).logarithmic(true))
            .changed()
        {
            app.sim45_state.sim_fps = fps;
            app.sim45_state
                .runtime
                .set_fixed_timestep(FixedTimestep::new(fps));
        }
        ui.add_space(4.0);

        // Pause / step.
        ui.horizontal(|ui| {
            let pause_label = if app.sim45_state.paused { "Resume" } else { "Pause" };
            if ui.button(pause_label).clicked() {
                app.sim45_state.paused = !app.sim45_state.paused;
            }
            if ui.button("Step Once").clicked() {
                app.sim45_state.step_once = true;
                app.sim45_state.paused = false;
            }
        });
        ui.add_space(4.0);

        // Interpolation toggle.
        ui.checkbox(&mut app.sim45_state.interpolate, "Interpolate transforms");
        ui.small(if app.sim45_state.interpolate {
            "Smooth: snapshot lerp/slerp at display rate."
        } else {
            "Choppy: renders only at sim fps (jitter visible at low fps)."
        });
        ui.add_space(6.0);

        ui.separator();

        // Camera follow controls.
        let mut follow = app.sim45_state.camera_follow;
        if ui.checkbox(&mut follow, "Camera follow").changed() {
            app.sim45_state.camera_follow = follow;
            let ids = app.sim45_state.physics_node_ids.clone();
            let idx = app.sim45_state.follow_body;
            if follow {
                if let Some(&id) = ids.get(idx) {
                    app.sim45_state.runtime.set_camera_follow(CameraFollow::Node {
                        id,
                        offset: glam::Vec3::ZERO,
                        look_at: true,
                    });
                }
            } else {
                app.sim45_state.runtime.clear_camera_follow();
            }
        }
        if app.sim45_state.camera_follow {
            let n = app.sim45_state.physics_node_ids.len();
            let mut idx = app.sim45_state.follow_body;
            if ui
                .add(egui::Slider::new(&mut idx, 0..=n.saturating_sub(1)).text("Body"))
                .changed()
            {
                app.sim45_state.follow_body = idx;
                let ids = app.sim45_state.physics_node_ids.clone();
                if let Some(&id) = ids.get(idx) {
                    app.sim45_state.runtime.set_camera_follow(CameraFollow::Node {
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
        ui.label(format!(
            "Step index : {}",
            app.sim45_state.runtime.step_index()
        ));
        ui.label(format!("Alpha      : {:.3}", app.sim45_state.runtime.alpha()));
        if app.sim45_state.paused {
            ui.label("Paused");
        }
        ui.add_space(4.0);

        ui.separator();
        ui.label("What this shows:");
        ui.small("- PhysicsLitePlugin: gravity, velocity integration, bounded reflection.");
        ui.small("- AnimationPlugin: keyframed looping path.");
        ui.small("- CameraFollow: orbit camera center tracks a physics body.");
        ui.small("- FixedTimestep + interpolation: smooth rendering at any display fps.");
    });
}
