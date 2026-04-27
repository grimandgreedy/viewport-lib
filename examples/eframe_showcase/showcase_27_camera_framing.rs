//! Showcase 27: Camera Framing & HUD
//!
//! A platform split by a central wall. Warm-coloured objects sit on the south
//! side (Y < 0) and cool-coloured objects on the north side (Y > 0).
//!
//! - **Camera A (blue)** : south of the wall, looking north. Only sees the
//!   south-side objects; the wall occludes the north side.
//! - **Camera B (orange)** : north of the wall, looking south. Only sees the
//!   north-side objects.
//! - **Camera C (green)** : west end, looking east along the wall. Sees the
//!   full length of the wall as a tall thin prism with objects on both sides.
//!
//! "Look through" buttons adopt each camera's exact POV. Corner-bracket HUD
//! overlays (drawn as [`ScreenImageItem`]s) colour-code which camera is active.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    CameraFrustumItem, ImageAnchor, LightSource, LightingSettings, Material,
    ScreenImageItem, ViewportRenderer,
    LightKind,
};

const COLOR_A: [f32; 4] = [0.4, 0.7, 1.0, 1.0];
const COLOR_B: [f32; 4] = [1.0, 0.6, 0.3, 1.0];
const COLOR_C: [f32; 4] = [0.5, 1.0, 0.5, 1.0];

/// Build a camera-to-world pose from eye/target/up (Z-up world).
fn camera_pose(eye: glam::Vec3, target: glam::Vec3, up: glam::Vec3) -> [[f32; 4]; 4] {
    glam::Mat4::look_at_rh(eye, target, up)
        .inverse()
        .to_cols_array_2d()
}

impl App {
    pub(crate) fn build_aux_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::scene::Scene;
        self.aux_scene = Scene::new();
        let up = glam::Vec3::Z;

        // Platform : 14 × 12 slab, top face at Z = 0.
        let platform_id = renderer.resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(14.0, 12.0, 0.25))
            .expect("aux platform");
        self.aux_scene.add_named("Platform", Some(platform_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.125)),
            { let mut m = Material::from_color([1.0, 1.0, 1.0]); m.roughness = 0.9; m });

        // Wall : 12 wide (X), 0.35 deep (Y), 2.4 tall (Z), centred at Z=1.2.
        let wall_id = renderer.resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(12.0, 0.35, 2.4))
            .expect("aux wall");
        self.aux_scene.add_named("Wall", Some(wall_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 1.2)),
            { let mut m = Material::from_color([1.0, 1.0, 1.0]); m.roughness = 0.8; m });

        // Shared sphere + box meshes.
        let sphere_id = renderer.resources_mut()
            .upload_mesh_data(&self.device, &viewport_lib::primitives::sphere(0.6, 24, 12))
            .expect("aux sphere");
        let box_id = renderer.resources_mut()
            .upload_mesh_data(&self.device, &make_box_with_uvs(1.2, 1.2, 1.2))
            .expect("aux box");

        // South side (Y < 0) — warm colours.
        self.aux_scene.add_named("South Sphere", Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, -3.0, 0.6)),
            { let mut m = Material::from_color([0.85, 0.22, 0.18]); m.roughness = 0.3; m });
        self.aux_scene.add_named("South Box", Some(box_id),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, -2.5, 0.6)),
            { let mut m = Material::from_color([0.95, 0.55, 0.10]); m.roughness = 0.5; m });

        // North side (Y > 0) — cool colours.
        self.aux_scene.add_named("North Sphere", Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 3.0, 0.6)),
            { let mut m = Material::from_color([0.15, 0.65, 0.80]); m.roughness = 0.3; m });
        self.aux_scene.add_named("North Box", Some(box_id),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, 3.0, 0.6)),
            { let mut m = Material::from_color([0.55, 0.20, 0.80]); m.roughness = 0.5; m });

        // Camera A : set well south of the scene, angled down at the warm objects.
        let eye_a = glam::vec3(0.0, -14.0, 2.5);
        let mut fa = CameraFrustumItem::default();
        fa.pose = camera_pose(eye_a, glam::vec3(0.0, -2.0, 0.7), up);
        fa.fov_y = 42_f32.to_radians(); fa.aspect = 16.0 / 9.0;
        fa.near = 0.5; fa.far = 3.5; fa.color = COLOR_A;

        // Camera B : set well north of the scene, angled down at the cool objects.
        let eye_b = glam::vec3(0.0, 14.0, 2.5);
        let mut fb = CameraFrustumItem::default();
        fb.pose = camera_pose(eye_b, glam::vec3(0.0, 2.0, 0.7), up);
        fb.fov_y = 42_f32.to_radians(); fb.aspect = 16.0 / 9.0;
        fb.near = 0.5; fb.far = 3.5; fb.color = COLOR_B;

        // Camera C : set well west of the scene, aimed at wall mid-height.
        // Eye is below the wall top (2.4 m) so it cannot see over the wall.
        let eye_c = glam::vec3(-18.0, 0.0, 2.0);
        let mut fc = CameraFrustumItem::default();
        fc.pose = camera_pose(eye_c, glam::vec3(0.0, 0.0, 1.2), up);
        fc.fov_y = 48_f32.to_radians(); fc.aspect = 16.0 / 9.0;
        fc.near = 0.5; fc.far = 4.5; fc.color = COLOR_C;

        self.aux_frustums = vec![fa, fb, fc];
        self.aux_built = true;
    }

    pub(crate) fn aux_lighting() -> LightingSettings {
        LightingSettings {
            lights: vec![LightSource {
                // direction = toward-light vector; positive Z -> light from above.
                kind: LightKind::Directional { direction: [0.3, -0.5, 0.8] },
                color: [1.0, 0.97, 0.90],
                intensity: 1.8,
                ..LightSource::default()
            }],
            shadows_enabled: true,
            shadow_cascade_count: 4,
            hemisphere_intensity: 0.2,
            ..LightingSettings::default()
        }
    }

    pub(crate) fn controls_aux(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Camera Framing").strong());
        ui.label(
            "A walled platform with warm objects (south) and cool objects (north). \
             Each camera can only see one side — except Camera C which looks along the wall.",
        );

        #[derive(Clone, Copy)]
        enum AuxAction { LookThrough(usize), Overview }
        let mut action: Option<AuxAction> = None;

        ui.add_space(4.0);
        let names = ["A - blue (south)", "B - orange (north)", "C - green (along wall)"];
        for i in 0..self.aux_frustums.len() {
            let active = self.aux_active_frustum == Some(i);
            if ui.selectable_label(active, format!("Look through {}", names[i])).clicked() {
                action = Some(AuxAction::LookThrough(i));
            }
        }
        if ui.selectable_label(self.aux_active_frustum.is_none(), "Overview").clicked() {
            action = Some(AuxAction::Overview);
        }
        ui.add_space(2.0);
        ui.label(egui::RichText::new("Tab: cycle cameras   Esc: exit to overview").weak().small());

        match action {
            Some(AuxAction::LookThrough(i)) => {
                let t = self.aux_frustums[i].camera_view_target();
                self.cam_animator.fly_to(&self.camera, t.center, t.distance, t.orientation, 1.2);
                self.aux_active_frustum = Some(i);
            }
            Some(AuxAction::Overview) => {
                self.cam_animator.fly_to(
                    &self.camera,
                    glam::Vec3::new(0.0, 0.0, 0.5), 30.0,
                    glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.0),
                    1.2,
                );
                self.aux_active_frustum = None;
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
        match self.aux_active_frustum {
            None => { ui.label("(overview — click 'Look through' to activate a camera)"); }
            Some(i) => {
                ui.label(format!("Active: {}", ["A - blue", "B - orange", "C - green"][i]));
            }
        }
        ui.add_space(4.0);
        ui.add(egui::Slider::new(&mut self.aux_img_alpha, 0.0..=1.0).text("Overlay alpha"));
        ui.add(egui::Slider::new(&mut self.aux_img_scale, 0.25..=4.0).text("Overlay scale"));
    }

    pub(crate) fn aux_push_screen_images(&self, fd: &mut viewport_lib::FrameData) {
        let (color, size, arm_len, thick) = match self.aux_active_frustum {
            Some(idx) => {
                let fc = self.aux_frustums[idx].color;
                let c = [(fc[0]*255.0) as u8, (fc[1]*255.0) as u8, (fc[2]*255.0) as u8, 255u8];
                (c, 72u32, 28u32, 4u32)
            }
            None => ([160u8, 160u8, 160u8, 80u8], 40u32, 16u32, 3u32),
        };

        for (anchor, fx, fy) in [
            (ImageAnchor::TopLeft,     false, false),
            (ImageAnchor::TopRight,    true,  false),
            (ImageAnchor::BottomLeft,  false, true),
            (ImageAnchor::BottomRight, true,  true),
        ] {
            let mut item = ScreenImageItem::default();
            item.pixels = bracket_pixels(color, size, arm_len, thick, fx, fy);
            item.width = size; item.height = size; item.anchor = anchor;
            item.scale = self.aux_img_scale; item.alpha = self.aux_img_alpha;
            fd.scene.screen_images.push(item);
        }

        if self.aux_active_frustum.is_some() {
            let mut item = ScreenImageItem::default();
            item.pixels = crosshair_pixels(color, 40, 3, 5);
            item.width = 40; item.height = 40;
            item.anchor = ImageAnchor::Center;
            item.scale = self.aux_img_scale; item.alpha = self.aux_img_alpha;
            fd.scene.screen_images.push(item);
        }
    }
}

fn bracket_pixels(color: [u8; 4], size: u32, arm: u32, thick: u32, fx: bool, fy: bool) -> Vec<[u8; 4]> {
    let mut p = vec![[0u8; 4]; (size * size) as usize];
    for i in 0..arm {
        for t in 0..thick {
            let x  = if fx { size-1-i } else { i };
            let y  = if fy { size-1-t } else { t };
            p[(y*size+x) as usize] = color;
            let x2 = if fx { size-1-t } else { t };
            let y2 = if fy { size-1-i } else { i };
            p[(y2*size+x2) as usize] = color;
        }
    }
    p
}

fn crosshair_pixels(color: [u8; 4], size: u32, thick: u32, gap: u32) -> Vec<[u8; 4]> {
    let mut p = vec![[0u8; 4]; (size * size) as usize];
    let c = size / 2;
    let ht = thick / 2;
    for i in 0..size {
        if (i as i32 - c as i32).unsigned_abs() <= gap { continue; }
        for t in 0..thick {
            let y = c.saturating_sub(ht) + t;
            if y < size { p[(y*size+i) as usize] = color; }
            let x = c.saturating_sub(ht) + t;
            if x < size { p[(i*size+x) as usize] = color; }
        }
    }
    p
}
