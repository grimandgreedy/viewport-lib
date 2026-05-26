//! Showcase 27: Camera Framing, Animation & Turntable
//!
//! Three sub-modes share the same walled-platform scene:
//!
//! **Framing** - three named cameras with frustum overlays and HUD brackets.
//! Look-through buttons adopt each camera's POV via an animated fly-to.
//!
//! **Turntable** - `TurntableController` continuously orbits the scene at a
//! configurable speed and elevation. Start/Stop while adjusting the sliders
//! to tune the look before exporting.
//!
//! **Track** - `CameraTrack` stores four keyframes (overview -> Camera A ->
//! Camera B -> Camera C), played back with Catmull-Rom interpolation. Add the
//! current view as a new keyframe, clear the track, or scrub the playback
//! position manually.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    CameraTarget, CameraTrack, ImageAnchor, LightKind, LightSource,
    LightingSettings, Material, PolylineItem, ScreenImageItem, TurntableController,
    ViewportRenderer, interpolate_camera,
};

/// Sub-mode for Showcase 27 (camera framing).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxSubMode {
    /// Camera framing and HUD (original showcase content).
    Framing,
    /// Continuous turntable orbit.
    Turntable,
    /// Keyframe track playback.
    Track,
}

const COLOUR_A: [f32; 4] = [0.4, 0.7, 1.0, 1.0];
const COLOUR_B: [f32; 4] = [1.0, 0.6, 0.3, 1.0];
const COLOUR_C: [f32; 4] = [0.5, 1.0, 0.5, 1.0];

/// Build a camera-to-world pose from eye/target/up (Z-up world).
fn camera_pose(eye: glam::Vec3, target: glam::Vec3, up: glam::Vec3) -> [[f32; 4]; 4] {
    glam::Mat4::look_at_rh(eye, target, up)
        .inverse()
        .to_cols_array_2d()
}

/// Local camera frustum data for showcase_27.
#[derive(Clone)]
pub(crate) struct FrustumData {
    pub pose: [[f32; 4]; 4],
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub colour: [f32; 4],
    pub line_width: f32,
}

/// Compute world-space corners of a frustum cross-section at depth d.
fn frustum_plane_corners(pose: [[f32; 4]; 4], fov_y: f32, aspect: f32, d: f32) -> [[f32; 3]; 4] {
    let half_h = (fov_y * 0.5).tan() * d;
    let half_w = half_h * aspect;
    let mat = glam::Mat4::from_cols_array_2d(&pose);
    let cam = [
        glam::vec3(-half_w, half_h, -d),
        glam::vec3(half_w, half_h, -d),
        glam::vec3(half_w, -half_h, -d),
        glam::vec3(-half_w, -half_h, -d),
    ];
    cam.map(|c| {
        let w = mat.transform_point3(c);
        [w.x, w.y, w.z]
    })
}

/// Build a PolylineItem wireframe for a camera frustum.
pub(crate) fn frustum_to_polyline(f: &FrustumData) -> PolylineItem {
    let near = frustum_plane_corners(f.pose, f.fov_y, f.aspect, f.near);
    let far = frustum_plane_corners(f.pose, f.fov_y, f.aspect, f.far);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();
    // Near quad (closed loop)
    positions.extend_from_slice(&[near[0], near[1], near[2], near[3], near[0]]);
    strip_lengths.push(5);
    // Far quad
    positions.extend_from_slice(&[far[0], far[1], far[2], far[3], far[0]]);
    strip_lengths.push(5);
    // Lateral edges
    for i in 0..4 {
        positions.extend_from_slice(&[near[i], far[i]]);
        strip_lengths.push(2);
    }
    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.default_colour = f.colour;
    item.line_width = f.line_width;
    item
}

/// Compute a CameraTarget for "look through this camera" fly-to.
pub(crate) fn frustum_view_target(f: &FrustumData) -> viewport_lib::CameraTarget {
    let mat = glam::Mat4::from_cols_array_2d(&f.pose);
    let eye = mat.transform_point3(glam::Vec3::ZERO);
    let rot = glam::Mat3::from_cols(
        mat.x_axis.truncate(),
        mat.y_axis.truncate(),
        mat.z_axis.truncate(),
    );
    let orientation = glam::Quat::from_mat3(&rot).normalize();
    let distance = (f.near * 0.5_f32).max(0.01);
    let center = eye + orientation * (-glam::Vec3::Z) * distance;
    viewport_lib::CameraTarget {
        center,
        distance,
        orientation,
    }
}

/// Build the default demo track: overview -> A -> B -> C -> overview.
fn build_demo_track(frustums: &[FrustumData]) -> CameraTrack {
    let overview = CameraTarget {
        center: glam::Vec3::new(0.0, 0.0, 0.5),
        distance: 30.0,
        orientation: glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.0),
    };

    let mut track = CameraTrack::new();
    track.push(0.0, overview);
    if frustums.len() >= 3 {
        track.push(2.5, frustum_view_target(&frustums[0]));
        track.push(5.0, frustum_view_target(&frustums[1]));
        track.push(7.5, frustum_view_target(&frustums[2]));
        track.push(10.0, overview);
    }
    track
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct AuxState {
    pub built: bool,
    pub scene: viewport_lib::scene::Scene,
    pub frustums: Vec<FrustumData>,
    pub img_alpha: f32,
    pub img_scale: f32,
    pub active_frustum: Option<usize>,
    pub sub_mode: AuxSubMode,
    pub turntable: TurntableController,
    pub turntable_running: bool,
    pub track: CameraTrack,
    pub track_t: f64,
    pub track_playing: bool,
}

impl Default for AuxState {
    fn default() -> Self {
        Self {
            built: false,
            scene: viewport_lib::scene::Scene::new(),
            frustums: Vec::new(),
            img_alpha: 1.0,
            img_scale: 1.0,
            active_frustum: None,
            sub_mode: AuxSubMode::Framing,
            turntable: TurntableController::new(0.5, 1.1),
            turntable_running: false,
            track: CameraTrack::new(),
            track_t: 0.0,
            track_playing: false,
        }
    }
}

impl App {
    pub(crate) fn build_aux_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::scene::Scene;
        self.aux_state.scene = Scene::new();
        let up = glam::Vec3::Z;

        // Platform : 14 x 12 slab, top face at Z = 0.
        let platform_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(14.0, 12.0, 0.25))
            .expect("aux platform");
        self.aux_state.scene.add_named(
            "Platform",
            Some(platform_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.125)),
            {
                let mut m = Material::from_colour([1.0, 1.0, 1.0]);
                m.roughness = 0.9;
                m
            },
        );

        // Wall : 12 wide (X), 0.35 deep (Y), 2.4 tall (Z), centred at Z=1.2.
        let wall_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(12.0, 0.35, 2.4))
            .expect("aux wall");
        self.aux_state.scene.add_named(
            "Wall",
            Some(wall_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 1.2)),
            {
                let mut m = Material::from_colour([1.0, 1.0, 1.0]);
                m.roughness = 0.8;
                m
            },
        );

        // Shared sphere + box meshes.
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &viewport_lib::primitives::sphere(0.6, 24, 12))
            .expect("aux sphere");
        let box_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(1.2, 1.2, 1.2))
            .expect("aux box");

        // South side (Y < 0) : warm colours.
        self.aux_state.scene.add_named(
            "South Sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, -3.0, 0.6)),
            {
                let mut m = Material::from_colour([0.85, 0.22, 0.18]);
                m.roughness = 0.3;
                m
            },
        );
        self.aux_state.scene.add_named(
            "South Box",
            Some(box_id),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, -2.5, 0.6)),
            {
                let mut m = Material::from_colour([0.95, 0.55, 0.10]);
                m.roughness = 0.5;
                m
            },
        );

        // North side (Y > 0) : cool colours.
        self.aux_state.scene.add_named(
            "North Sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 3.0, 0.6)),
            {
                let mut m = Material::from_colour([0.15, 0.65, 0.80]);
                m.roughness = 0.3;
                m
            },
        );
        self.aux_state.scene.add_named(
            "North Box",
            Some(box_id),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, 3.0, 0.6)),
            {
                let mut m = Material::from_colour([0.55, 0.20, 0.80]);
                m.roughness = 0.5;
                m
            },
        );

        // Camera A : south of the scene, angled down at the warm objects.
        let eye_a = glam::vec3(0.0, -14.0, 2.5);
        let fa = FrustumData {
            pose: camera_pose(eye_a, glam::vec3(0.0, -2.0, 0.7), up),
            fov_y: 42_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.5,
            far: 3.5,
            colour: COLOUR_A,
            line_width: 2.0,
        };

        // Camera B : north of the scene, angled down at the cool objects.
        let eye_b = glam::vec3(0.0, 14.0, 2.5);
        let fb = FrustumData {
            pose: camera_pose(eye_b, glam::vec3(0.0, 2.0, 0.7), up),
            fov_y: 42_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.5,
            far: 3.5,
            colour: COLOUR_B,
            line_width: 2.0,
        };

        // Camera C : west end, aimed at wall mid-height.
        let eye_c = glam::vec3(-18.0, 0.0, 2.0);
        let fc = FrustumData {
            pose: camera_pose(eye_c, glam::vec3(0.0, 0.0, 1.2), up),
            fov_y: 48_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.5,
            far: 4.5,
            colour: COLOUR_C,
            line_width: 2.0,
        };

        self.aux_state.frustums = vec![fa, fb, fc];

        // Pre-build the demo track from the three frustum positions.
        self.aux_state.track = build_demo_track(&self.aux_state.frustums);

        self.aux_state.built = true;
    }

    pub(crate) fn aux_lighting() -> LightingSettings {
        {
            let mut _t = LightingSettings::default();
            _t.lights = vec![{
                let mut _t = LightSource::default();
                _t.kind = LightKind::Directional {
                    direction: [0.3, -0.5, 0.8],
                };
                _t.colour = [1.0, 0.97, 0.90];
                _t.intensity = 1.8;
                _t
            }];
            _t.shadows_enabled = true;
            _t.shadow_cascade_count = 4;
            _t.hemisphere_intensity = 0.2;
            _t
        }
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_aux(app: &mut App, ui: &mut egui::Ui) {
    // Sub-mode selector.
    ui.horizontal(|ui| {
        for (mode, label) in [
            (AuxSubMode::Framing, "Framing"),
            (AuxSubMode::Turntable, "Turntable"),
            (AuxSubMode::Track, "Track"),
        ] {
            if ui
                .selectable_label(app.aux_state.sub_mode == mode, label)
                .clicked()
                && app.aux_state.sub_mode != mode
            {
                app.aux_state.sub_mode = mode;
                // Stop any active motion when switching modes.
                app.aux_state.turntable_running = false;
                app.aux_state.track_playing = false;
            }
        }
    });
    ui.separator();

    match app.aux_state.sub_mode {
        AuxSubMode::Framing => controls_aux_framing(app, ui),
        AuxSubMode::Turntable => controls_aux_turntable(app, ui),
        AuxSubMode::Track => controls_aux_track(app, ui),
    }
}

fn controls_aux_framing(app: &mut App, ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Camera Framing").strong());
    ui.label(
        "A walled platform with warm objects (south) and cool objects (north). \
             Each camera sees only one side; Camera C looks along the wall.",
    );

    #[derive(Clone, Copy)]
    enum AuxAction {
        LookThrough(usize),
        Overview,
    }
    let mut action: Option<AuxAction> = None;

    ui.add_space(4.0);
    let names = [
        "A - blue (south)",
        "B - orange (north)",
        "C - green (along wall)",
    ];
    for i in 0..app.aux_state.frustums.len() {
        let active = app.aux_state.active_frustum == Some(i);
        if ui
            .selectable_label(active, format!("Look through {}", names[i]))
            .clicked()
        {
            action = Some(AuxAction::LookThrough(i));
        }
    }
    if ui
        .selectable_label(app.aux_state.active_frustum.is_none(), "Overview")
        .clicked()
    {
        action = Some(AuxAction::Overview);
    }
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new("Tab: cycle cameras   Esc: exit to overview")
            .weak()
            .small(),
    );

    match action {
        Some(AuxAction::LookThrough(i)) => {
            let t = frustum_view_target(&app.aux_state.frustums[i]);
            app.cam_animator
                .fly_to(&app.camera, t.center, t.distance, t.orientation, 1.2);
            app.aux_state.active_frustum = Some(i);
        }
        Some(AuxAction::Overview) => {
            app.cam_animator.fly_to(
                &app.camera,
                glam::Vec3::new(0.0, 0.0, 0.5),
                30.0,
                glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.0),
                1.2,
            );
            app.aux_state.active_frustum = None;
        }
        None => {}
    }

    ui.separator();
    ui.label(egui::RichText::new("Active-Camera HUD Overlay").strong());
    ui.label(
        "Corner brackets and crosshair mark the active camera in its colour. \
             Dim brackets indicate overview mode.",
    );
    ui.add_space(2.0);
    match app.aux_state.active_frustum {
        None => {
            ui.label("(overview : click 'Look through' to activate a camera)");
        }
        Some(i) => {
            ui.label(format!(
                "Active: {}",
                ["A - blue", "B - orange", "C - green"][i]
            ));
        }
    }
    ui.add_space(4.0);
    ui.add(egui::Slider::new(&mut app.aux_state.img_alpha, 0.0..=1.0).text("Overlay alpha"));
    ui.add(egui::Slider::new(&mut app.aux_state.img_scale, 0.25..=4.0).text("Overlay scale"));
}

fn controls_aux_turntable(app: &mut App, ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Turntable Camera").strong());
    ui.label(
        "Continuously orbits around the scene center at a fixed elevation. \
             Adjust speed and tilt, then press Start.",
    );
    ui.add_space(4.0);

    let speed_changed = ui
        .add(
            egui::Slider::new(&mut app.aux_state.turntable.angular_velocity, -3.0..=3.0)
                .text("Speed (rad/s)"),
        )
        .changed();
    let tilt_changed = ui
        .add(egui::Slider::new(&mut app.aux_state.turntable.tilt, 0.1..=1.8).text("Tilt (rad)"))
        .changed();

    // If sliders changed while running, reapply the orientation immediately
    // so the preview updates without waiting for the next frame.
    if (speed_changed || tilt_changed) && app.aux_state.turntable_running {
        let az = app.aux_state.turntable.azimuth;
        let tilt = app.aux_state.turntable.tilt;
        app.camera
            .set_orientation(glam::Quat::from_rotation_z(az) * glam::Quat::from_rotation_x(tilt));
    }

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let label = if app.aux_state.turntable_running {
            "Stop"
        } else {
            "Start"
        };
        if ui.button(label).clicked() {
            if app.aux_state.turntable_running {
                app.aux_state.turntable_running = false;
            } else {
                // Sync azimuth/tilt from the current camera orientation before starting.
                app.aux_state.turntable = TurntableController::from_camera(
                    &app.camera,
                    app.aux_state.turntable.angular_velocity,
                );
                app.aux_state.turntable_running = true;
            }
        }
        if ui.button("Reset to overview").clicked() {
            app.aux_state.turntable_running = false;
            app.cam_animator.fly_to(
                &app.camera,
                glam::Vec3::new(0.0, 0.0, 0.5),
                30.0,
                glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.0),
                1.0,
            );
        }
    });

    ui.add_space(4.0);
    if app.aux_state.turntable_running {
        ui.label(format!(
            "Azimuth: {:.2} rad",
            app.aux_state.turntable.azimuth
        ));
    } else {
        ui.label("(stopped)");
    }
}

fn controls_aux_track(app: &mut App, ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Camera Track").strong());
    ui.label(
        "Keyframe animation with Catmull-Rom interpolation. \
             The demo track visits the overview then each of the three cameras.",
    );
    ui.add_space(4.0);

    let duration = app.aux_state.track.duration();

    // Playback controls.
    ui.horizontal(|ui| {
        let play_label = if app.aux_state.track_playing {
            "Stop"
        } else {
            "Play"
        };
        if ui.button(play_label).clicked() {
            if app.aux_state.track_playing {
                app.aux_state.track_playing = false;
            } else if app.aux_state.track.len() >= 2 {
                app.aux_state.track_t = 0.0;
                app.aux_state.track_playing = true;
            }
        }
        if ui.button("Rewind").clicked() {
            app.aux_state.track_t = 0.0;
            app.aux_state.track_playing = false;
        }
    });

    // Scrub bar.
    if duration > 0.0 {
        let mut t_f32 = app.aux_state.track_t as f32;
        if ui
            .add(egui::Slider::new(&mut t_f32, 0.0..=(duration as f32)).text("Time (s)"))
            .changed()
        {
            app.aux_state.track_t = t_f32 as f64;
            app.aux_state.track_playing = false;
            // Apply scrubbed position immediately.
            let target = interpolate_camera(&app.aux_state.track, app.aux_state.track_t);
            app.camera.center = target.center;
            app.camera.set_distance(target.distance);
            app.camera.set_orientation(target.orientation);
        }
    }

    ui.add_space(4.0);
    ui.label(format!(
        "Track: {} keyframes, {:.1} s duration",
        app.aux_state.track.len(),
        duration
    ));
    if duration > 0.0 {
        ui.label(format!(
            "Playback: {:.2} / {:.2} s",
            app.aux_state.track_t, duration
        ));
    }

    ui.separator();
    ui.label(egui::RichText::new("Edit Track").strong());

    ui.horizontal(|ui| {
        if ui.button("Add current view").clicked() {
            let t = if duration > 0.0 { duration + 2.5 } else { 0.0 };
            let target = CameraTarget {
                center: app.camera.center,
                distance: app.camera.distance,
                orientation: app.camera.orientation,
            };
            app.aux_state.track.push(t, target);
        }
        if ui.button("Reset to demo").clicked() {
            app.aux_state.track_playing = false;
            app.aux_state.track_t = 0.0;
            app.aux_state.track = build_demo_track(&app.aux_state.frustums);
        }
        if ui.button("Clear").clicked() {
            app.aux_state.track_playing = false;
            app.aux_state.track_t = 0.0;
            app.aux_state.track = CameraTrack::new();
        }
    });
}

impl App {
    pub(crate) fn aux_push_screen_images(&self, fd: &mut viewport_lib::FrameData) {
        // HUD overlay only shown in Framing mode.
        if self.aux_state.sub_mode != AuxSubMode::Framing {
            return;
        }

        // Generate pixel buffers at physical resolution for crisp HiDPI rendering.
        // item.width/height stay in logical pixels (the visual size); the renderer
        // infers the physical texture size from the pixel buffer length.
        let ppp = fd.camera.pixels_per_point;
        let px = |n: u32| ((n as f32 * ppp).round() as u32).max(1);

        let (colour, size, arm_len, thick) = match self.aux_state.active_frustum {
            Some(idx) => {
                let fc = self.aux_state.frustums[idx].colour;
                let c = [
                    (fc[0] * 255.0) as u8,
                    (fc[1] * 255.0) as u8,
                    (fc[2] * 255.0) as u8,
                    255u8,
                ];
                (c, 72u32, 28u32, 4u32)
            }
            None => ([160u8, 160u8, 160u8, 80u8], 40u32, 16u32, 3u32),
        };

        for (anchor, fx, fy) in [
            (ImageAnchor::TopLeft, false, false),
            (ImageAnchor::TopRight, true, false),
            (ImageAnchor::BottomLeft, false, true),
            (ImageAnchor::BottomRight, true, true),
        ] {
            let mut item = ScreenImageItem::default();
            item.pixels = bracket_pixels(colour, px(size), px(arm_len), px(thick), fx, fy);
            item.width = size;
            item.height = size;
            item.anchor = anchor;
            item.scale = self.aux_state.img_scale;
            item.alpha = self.aux_state.img_alpha;
            fd.scene.screen_images.push(item);
        }

        if self.aux_state.active_frustum.is_some() {
            let mut item = ScreenImageItem::default();
            item.pixels = crosshair_pixels(colour, px(40), px(3), px(5));
            item.width = 40;
            item.height = 40;
            item.anchor = ImageAnchor::Center;
            item.scale = self.aux_state.img_scale;
            item.alpha = self.aux_state.img_alpha;
            fd.scene.screen_images.push(item);
        }
    }
}

fn bracket_pixels(
    colour: [u8; 4],
    size: u32,
    arm: u32,
    thick: u32,
    fx: bool,
    fy: bool,
) -> Vec<[u8; 4]> {
    let mut p = vec![[0u8; 4]; (size * size) as usize];
    for i in 0..arm {
        for t in 0..thick {
            let x = if fx { size - 1 - i } else { i };
            let y = if fy { size - 1 - t } else { t };
            p[(y * size + x) as usize] = colour;
            let x2 = if fx { size - 1 - t } else { t };
            let y2 = if fy { size - 1 - i } else { i };
            p[(y2 * size + x2) as usize] = colour;
        }
    }
    p
}

fn crosshair_pixels(colour: [u8; 4], size: u32, thick: u32, gap: u32) -> Vec<[u8; 4]> {
    let mut p = vec![[0u8; 4]; (size * size) as usize];
    let c = size / 2;
    let ht = thick / 2;
    for i in 0..size {
        if (i as i32 - c as i32).unsigned_abs() <= gap {
            continue;
        }
        for t in 0..thick {
            let y = c.saturating_sub(ht) + t;
            if y < size {
                p[(y * size + i) as usize] = colour;
            }
            let x = c.saturating_sub(ht) + t;
            if x < size {
                p[(i * size + x) as usize] = colour;
            }
        }
    }
    p
}
