//! Showcase 27: Auxiliary Scene Structures
//!
//! Demonstrates Phase 10 of the polyscope gap plan:
//!
//! - **Camera frustum wireframes** : three renderable frustums at different poses,
//!   each with its own FOV and aspect ratio. "Fly to" buttons animate the camera
//!   to view each frustum using [`CameraFrustumItem::camera_target`].
//!
//! - **Screen-space image overlays** : solid-color RGBA patches anchored to each
//!   viewport corner and the center. Alpha and scale sliders update live.

use crate::App;
use eframe::egui;
use viewport_lib::{CameraFrustumItem, CameraTarget, ImageAnchor, ScreenImageItem};

// ---------------------------------------------------------------------------
// App state fields (stored in main App struct):
//   aux_frustums: Vec<CameraFrustumItem>   : the three frustum items
//   aux_img_alpha: f32
//   aux_img_scale: f32
// ---------------------------------------------------------------------------

impl App {
    // -------------------------------------------------------------------------
    // One-time build
    // -------------------------------------------------------------------------

    /// Build the three camera frustum items and store them.
    pub(crate) fn build_aux_scene(&mut self) {
        // Frustum A : facing forward (identity pose), narrow FOV, 16:9
        let mut frustum_a = CameraFrustumItem::default();
        frustum_a.pose = glam::Mat4::IDENTITY.to_cols_array_2d();
        frustum_a.fov_y = std::f32::consts::FRAC_PI_4;
        frustum_a.aspect = 16.0 / 9.0;
        frustum_a.near = 0.2;
        frustum_a.far = 4.0;
        frustum_a.color = [0.4, 0.7, 1.0, 1.0];
        frustum_a.image_plane_depth = Some(1.0);

        // Frustum B : offset left and rotated 30° around Y, wider FOV, 4:3
        let pose_b = glam::Mat4::from_translation(glam::vec3(-3.0, 0.0, 0.0))
            * glam::Mat4::from_rotation_y(30_f32.to_radians());
        let mut frustum_b = CameraFrustumItem::default();
        frustum_b.pose = pose_b.to_cols_array_2d();
        frustum_b.fov_y = 70_f32.to_radians();
        frustum_b.aspect = 4.0 / 3.0;
        frustum_b.near = 0.3;
        frustum_b.far = 5.0;
        frustum_b.color = [1.0, 0.6, 0.3, 1.0];
        frustum_b.image_plane_depth = Some(1.5);

        // Frustum C : offset right, rotated –25° around Y, square-ish
        let pose_c = glam::Mat4::from_translation(glam::vec3(3.0, 0.0, 0.0))
            * glam::Mat4::from_rotation_y(-25_f32.to_radians());
        let mut frustum_c = CameraFrustumItem::default();
        frustum_c.pose = pose_c.to_cols_array_2d();
        frustum_c.fov_y = 55_f32.to_radians();
        frustum_c.aspect = 1.0;
        frustum_c.near = 0.1;
        frustum_c.far = 3.0;
        frustum_c.color = [0.5, 1.0, 0.5, 1.0];

        self.aux_frustums = vec![frustum_a, frustum_b, frustum_c];
        self.aux_built = true;
    }

    // -------------------------------------------------------------------------
    // Controls panel (UI only : fd is populated in the main update loop)
    // -------------------------------------------------------------------------

    pub(crate) fn controls_aux(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Camera Frustums").strong());
        ui.horizontal(|ui| {
            for (i, frustum) in self.aux_frustums.iter().enumerate() {
                let label = format!("Fly to #{}", i + 1);
                if ui.button(&label).clicked() {
                    let target: CameraTarget = frustum.camera_target(2.5);
                    self.cam_animator.fly_to(
                        &self.camera,
                        target.center,
                        target.distance,
                        target.orientation,
                        1.2,
                    );
                }
            }
        });

        ui.separator();
        ui.label(egui::RichText::new("Screen-Space Image Overlays").strong());
        ui.add(egui::Slider::new(&mut self.aux_img_alpha, 0.0..=1.0).text("Alpha"));
        ui.add(egui::Slider::new(&mut self.aux_img_scale, 0.25..=4.0).text("Scale"));
        ui.label("One 64×64 patch per anchor point (TL/TR/BL/BR/Center).");
    }

    /// Push screen-space image items into fd for the current frame.
    pub(crate) fn aux_push_screen_images(&self, fd: &mut viewport_lib::FrameData) {
        let configs: &[(ImageAnchor, [u8; 4])] = &[
            (ImageAnchor::TopLeft, [220, 80, 80, 255]),
            (ImageAnchor::TopRight, [80, 200, 80, 255]),
            (ImageAnchor::BottomLeft, [80, 120, 220, 255]),
            (ImageAnchor::BottomRight, [220, 200, 60, 255]),
            (ImageAnchor::Center, [200, 80, 200, 255]),
        ];
        for (anchor, color) in configs {
            let pixels: Vec<[u8; 4]> = vec![*color; 64 * 64];
            let mut item = ScreenImageItem::default();
            item.pixels = pixels;
            item.width = 64;
            item.height = 64;
            item.anchor = *anchor;
            item.scale = self.aux_img_scale;
            item.alpha = self.aux_img_alpha;
            fd.scene.screen_images.push(item);
        }
    }
}
