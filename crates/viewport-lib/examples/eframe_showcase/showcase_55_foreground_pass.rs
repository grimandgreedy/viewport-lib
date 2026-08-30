//! Showcase 55: Foreground Composite Pass
//!
//! The foreground pass draws `SceneFrame::foreground_items` over the finished
//! scene against a cleared depth buffer, so they are never occluded by world
//! geometry and never clip into it. Here the camera flies around a scene on a
//! looping track while depth of field racks focus across the objects; three
//! cubes stay pinned in front of the camera the whole time.
//!
//! What the pass gives you, visible in this scene:
//!
//! - The cubes are never hidden by scene geometry as the camera sweeps past
//!   pillars, because the pass clears depth before drawing them.
//! - The cubes stay sharp while the background blurs: the foreground depth is a
//!   coverage mask, so DoF leaves foreground pixels untouched.
//! - With the projection override on, the cubes use their own field of view and
//!   near plane while sharing the camera's view transform. Off, they reuse the
//!   scene projection.
//! - With "unlit foreground" on, the cubes ignore the scene lighting and read
//!   the same in every part of the scene (the world-independent look a
//!   first-person viewmodel wants); off, they are lit by the scene.
//!
//! The cubes are ordinary `SceneRenderItem`s submitted through
//! `SceneFrame::foreground_items`; nothing here uses the plugin surface.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib as vpl;
use vpl::{
    CameraTarget, CameraTrack, ForegroundPass, ForegroundProjection, LightKind, LightSource,
    LightingSettings, Material, PostProcessSettings, RenderCamera, SceneRenderItem, Selection,
    ViewportRenderer, interpolate_camera, scene::Scene,
};

// Camera-space offsets for the three foreground cubes (RH view space: +X right,
// +Y up, -Z forward). Held a short distance in front of the eye, slightly low.
const CUBE_OFFSETS: [[f32; 3]; 3] = [
    [-0.75, -0.35, -1.7],
    [0.0, -0.28, -1.45],
    [0.75, -0.35, -1.7],
];
const CUBE_COLOURS: [[f32; 3]; 3] = [[0.65, 0.09, 0.07], [0.10, 0.52, 0.18], [0.10, 0.26, 0.68]];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ForegroundState {
    pub built: bool,
    pub scene: Scene,
    pub cube_mesh: vpl::MeshId,
    pub track: CameraTrack,
    pub track_t: f64,
    pub playing: bool,
    /// Wall-clock accumulator driving cube spin and the DoF focus rack.
    pub time: f32,
    pub spin_speed: f32,
    // Depth of field.
    pub dof_enabled: bool,
    pub dof_auto_rack: bool,
    pub dof_focal_distance: f32,
    pub dof_focal_range: f32,
    pub dof_max_blur: f32,
    // Foreground projection override.
    pub projection_override: bool,
    pub fov_y_deg: f32,
    pub near: f32,
    // Foreground appearance.
    pub unlit: bool,
}

impl Default for ForegroundState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            cube_mesh: vpl::MeshId::INVALID,
            track: CameraTrack::new(),
            track_t: 0.0,
            playing: true,
            time: 0.0,
            spin_speed: 1.2,
            dof_enabled: true,
            dof_auto_rack: true,
            dof_focal_distance: 13.0,
            dof_focal_range: 2.0,
            dof_max_blur: 12.0,
            projection_override: true,
            fov_y_deg: 55.0,
            near: 0.02,
            unlit: false,
        }
    }
}

/// A looping orbit around the scene, six keyframes closing back on the first.
fn build_orbit_track() -> CameraTrack {
    let mut track = CameraTrack::new();
    let center = glam::Vec3::new(0.0, 0.0, 1.1);
    let tilt = 1.02;
    let steps = 6;
    for i in 0..=steps {
        let f = i as f32 / steps as f32;
        let az = f * std::f32::consts::TAU;
        // Gentle dolly so the focal-plane relationship keeps shifting.
        let distance = 13.0 + 2.0 * (az * 2.0).cos();
        let orientation = glam::Quat::from_rotation_z(az) * glam::Quat::from_rotation_x(tilt);
        track.push(
            (i as f64) * 2.5,
            CameraTarget {
                center,
                distance,
                orientation,
            },
        );
    }
    track
}

impl App {
    pub(crate) fn build_foreground_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.fg_state.scene = Scene::new();

        // Ground slab, top face at Z = 0.
        let ground = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(30.0, 30.0, 0.4))
            .expect("fg ground");
        self.fg_state.scene.add_named(
            "Ground",
            Some(ground),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.2)),
            {
                let mut m = Material::from_colour([0.82, 0.82, 0.86]);
                m.roughness = 0.9;
                m
            },
        );

        // A scatter of pillars and spheres spread across depth, so any orbit
        // angle shows objects both nearer and farther than the focal plane.
        let pillar = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(1.0, 1.0, 3.0))
            .expect("fg pillar");
        let sphere = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &vpl::primitives::sphere(0.9, 24, 12))
            .expect("fg sphere");

        // (x, y, height-scale, is_sphere, colour)
        let props: [(f32, f32, f32, bool, [f32; 3]); 9] = [
            (-6.0, -5.0, 1.2, false, [0.65, 0.09, 0.07]),
            (-2.5, 4.5, 0.8, true, [0.10, 0.26, 0.68]),
            (5.5, -3.0, 1.5, false, [0.72, 0.42, 0.04]),
            (2.0, 7.5, 1.0, true, [0.38, 0.12, 0.62]),
            (-7.0, 6.0, 1.1, false, [0.10, 0.52, 0.18]),
            (7.0, 5.0, 0.7, true, [0.62, 0.08, 0.35]),
            (0.5, -6.5, 1.3, false, [0.08, 0.20, 0.62]),
            (-4.0, 0.5, 0.9, true, [0.75, 0.28, 0.05]),
            (4.0, 1.0, 1.4, false, [0.35, 0.55, 0.06]),
        ];
        for (i, (x, y, hz, is_sphere, colour)) in props.iter().enumerate() {
            let mat = {
                let mut m = Material::pbr(*colour, 0.05, 0.45);
                m.roughness = 0.4;
                m
            };
            if *is_sphere {
                self.fg_state.scene.add_named(
                    &format!("Sphere {i}"),
                    Some(sphere),
                    glam::Mat4::from_translation(glam::Vec3::new(*x, *y, 0.9)),
                    mat,
                );
            } else {
                self.fg_state.scene.add_named(
                    &format!("Pillar {i}"),
                    Some(pillar),
                    glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::new(1.0, 1.0, *hz),
                        glam::Quat::IDENTITY,
                        glam::Vec3::new(*x, *y, 1.5 * hz),
                    ),
                    mat,
                );
            }
        }

        // Small cube reused for the three foreground items.
        self.fg_state.cube_mesh = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(0.34, 0.34, 0.34))
            .expect("fg cube");

        self.fg_state.track = build_orbit_track();
        self.fg_state.track_t = 0.0;

        // Start the camera on the track.
        let target = interpolate_camera(&self.fg_state.track, 0.0);
        self.camera.center = target.center;
        self.camera.set_distance(target.distance);
        self.camera.set_orientation(target.orientation);

        self.fg_state.built = true;
    }

    pub(crate) fn foreground_lighting() -> LightingSettings {
        let mut l = LightingSettings::default();
        l.lights = vec![{
            let mut s = LightSource::default();
            s.kind = LightKind::Directional {
                direction: [0.35, -0.5, 0.78],
            };
            s.colour = [1.0, 0.97, 0.92];
            s.intensity = 2.0;
            s
        }];
        l.hemisphere_intensity = 0.5;
        l.sky_colour = [0.55, 0.62, 0.78];
        l.ground_colour = [0.28, 0.26, 0.24];
        l
    }
}

/// World-scene render items (the background the camera flies around).
pub(crate) fn foreground_scene_items(app: &mut App) -> Vec<SceneRenderItem> {
    if !app.fg_state.built {
        return Vec::new();
    }
    app.fg_state.scene.collect_render_items(&Selection::new())
}

/// Advance the camera track, the cube spin, and the auto focus rack.
pub(crate) fn update_foreground(app: &mut App, dt: f32) {
    app.fg_state.time += dt;

    if app.fg_state.playing {
        let dur = app.fg_state.track.duration();
        if dur > 0.0 {
            app.fg_state.track_t += dt as f64;
            if app.fg_state.track_t > dur {
                app.fg_state.track_t -= dur;
            }
            let target = interpolate_camera(&app.fg_state.track, app.fg_state.track_t);
            app.camera.center = target.center;
            app.camera.set_distance(target.distance);
            app.camera.set_orientation(target.orientation);
        }
    }

    if app.fg_state.dof_auto_rack {
        // Rack the focal plane back and forth across the object depth range.
        let phase = app.fg_state.time * 0.5;
        app.fg_state.dof_focal_distance = 12.0 + 7.0 * phase.sin();
    }
}

/// Set the camera, post-process (DoF), and foreground pass + items for the frame.
pub(crate) fn configure_frame(app: &App, fd: &mut vpl::FrameData) {
    // Cap the far plane so cascades and the depth range stay useful.
    let mut rc = RenderCamera::from_camera(&app.camera);
    rc.far = (app.camera.distance * 3.0).max(60.0);
    rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
    fd.camera.render_camera = rc;

    // The foreground pass runs in the HDR path, so keep the HDR pipeline active
    // (also required for DoF). DoF itself is toggled separately.
    let mut pp = PostProcessSettings::default();
    if app.fg_state.dof_enabled {
        pp.dof.enabled = true;
        pp.dof.focal_distance = app.fg_state.dof_focal_distance;
        pp.dof.focal_range = app.fg_state.dof_focal_range;
        pp.dof.max_blur_radius = app.fg_state.dof_max_blur;
    }
    fd.effects.post_process = pp;

    // Enable the pass; the projection override is optional.
    fd.effects.foreground = Some(ForegroundPass {
        projection: app
            .fg_state
            .projection_override
            .then_some(ForegroundProjection {
                fov_y: app.fg_state.fov_y_deg.to_radians(),
                near: app.fg_state.near,
                far: None,
            }),
    });

    // Three cubes pinned in front of the camera. The pass shares the main view
    // transform, so a camera-relative world transform keeps them on screen; the
    // cleared depth keeps them over the scene.
    let c2w = app.camera.view_matrix().inverse();
    let t = app.fg_state.time * app.fg_state.spin_speed;
    let mut cubes = Vec::with_capacity(3);
    for (i, off) in CUBE_OFFSETS.iter().enumerate() {
        let spin = glam::Mat4::from_rotation_y(t + i as f32 * 1.3)
            * glam::Mat4::from_rotation_x(t * 0.6 + i as f32);
        let model = c2w * glam::Mat4::from_translation(glam::Vec3::from_array(*off)) * spin;
        let mut item = SceneRenderItem::default();
        item.mesh_id = app.fg_state.cube_mesh;
        item.model = model.to_cols_array_2d();
        item.material = Material::pbr(CUBE_COLOURS[i], 0.1, 0.35);
        item.settings.unlit = app.fg_state.unlit;
        cubes.push(item);
    }
    fd.scene.foreground_items = cubes;
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_foreground(app: &mut App, ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Foreground Composite Pass").strong());
    ui.label(
        "The camera flies a looping track while depth of field racks focus. \
         Three cubes stay pinned in front of the camera, drawn over the scene \
         on a cleared depth buffer.",
    );

    ui.separator();
    ui.horizontal(|ui| {
        let label = if app.fg_state.playing {
            "Pause fly-around"
        } else {
            "Play fly-around"
        };
        if ui.button(label).clicked() {
            app.fg_state.playing = !app.fg_state.playing;
        }
        if ui.button("Restart").clicked() {
            app.fg_state.track_t = 0.0;
        }
    });

    ui.separator();
    ui.label(egui::RichText::new("Depth of Field").strong());
    ui.checkbox(&mut app.fg_state.dof_enabled, "Enable DoF");
    if app.fg_state.dof_enabled {
        ui.checkbox(&mut app.fg_state.dof_auto_rack, "Auto rack focus");
        ui.add_enabled(
            !app.fg_state.dof_auto_rack,
            egui::Slider::new(&mut app.fg_state.dof_focal_distance, 4.0..=24.0)
                .text("Focal distance"),
        );
        ui.add(egui::Slider::new(&mut app.fg_state.dof_focal_range, 0.5..=8.0).text("Focal range"));
        ui.add(egui::Slider::new(&mut app.fg_state.dof_max_blur, 2.0..=20.0).text("Max blur (px)"));
        ui.label(
            egui::RichText::new("The cubes stay sharp: foreground depth masks them out of DoF.")
                .weak()
                .small(),
        );
    }

    ui.separator();
    ui.label(egui::RichText::new("Foreground projection").strong());
    ui.checkbox(
        &mut app.fg_state.projection_override,
        "Override projection (own FOV / near)",
    );
    if app.fg_state.projection_override {
        ui.add(egui::Slider::new(&mut app.fg_state.fov_y_deg, 25.0..=90.0).text("FOV (deg)"));
        ui.add(egui::Slider::new(&mut app.fg_state.near, 0.005..=0.5).text("Near plane"));
    } else {
        ui.label(
            egui::RichText::new("Off: the cubes reuse the scene projection.")
                .weak()
                .small(),
        );
    }

    ui.separator();
    ui.label(egui::RichText::new("Foreground appearance").strong());
    ui.checkbox(&mut app.fg_state.unlit, "Unlit (world-independent look)");
    ui.add(egui::Slider::new(&mut app.fg_state.spin_speed, 0.0..=4.0).text("Cube spin speed"));
    ui.label(
        egui::RichText::new(
            "Unlit is one line on the item's material: the pass bakes no lighting, \
             so a world-independent look is a material choice.",
        )
        .weak()
        .small(),
    );
}
