//! Showcase 27: Auxiliary Scene Structures
//!
//! Demonstrates two auxiliary overlay features:
//!
//! - **Camera frustum wireframes** : three placed cameras at different poses.
//!   "Look through" buttons animate the viewport into each camera's exact POV
//!   using [`CameraFrustumItem::camera_view_target`]. "Overview" flies back out
//!   to see all three frustums from outside using [`CameraFrustumItem::camera_target`].
//!
//! - **Active-camera HUD overlay** : screen-space corner brackets and a centre
//!   crosshair drawn in the active camera's colour. This mirrors how production
//!   CAD and VFX tools indicate which camera the viewport is currently looking
//!   through. The brackets are pushed as [`ScreenImageItem`]s every frame, so
//!   they remain fixed in screen space regardless of the 3-D view.

use crate::App;
use eframe::egui;
use viewport_lib::{CameraFrustumItem, ImageAnchor, ScreenImageItem};

// Frustum colours (matches the wireframe colour so the HUD tint is consistent).
const COLOR_A: [f32; 4] = [0.4, 0.7, 1.0, 1.0]; // blue
const COLOR_B: [f32; 4] = [1.0, 0.6, 0.3, 1.0]; // orange
const COLOR_C: [f32; 4] = [0.5, 1.0, 0.5, 1.0]; // green

impl App {
    // -------------------------------------------------------------------------
    // One-time build
    // -------------------------------------------------------------------------

    pub(crate) fn build_aux_scene(&mut self) {
        // Frustum A : identity pose (looking along -Z), narrow FOV, 16:9
        let mut frustum_a = CameraFrustumItem::default();
        frustum_a.pose = glam::Mat4::IDENTITY.to_cols_array_2d();
        frustum_a.fov_y = std::f32::consts::FRAC_PI_4;
        frustum_a.aspect = 16.0 / 9.0;
        frustum_a.near = 0.2;
        frustum_a.far = 4.0;
        frustum_a.color = COLOR_A;
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
        frustum_b.color = COLOR_B;
        frustum_b.image_plane_depth = Some(1.5);

        // Frustum C : offset right, rotated -25° around Y, square-ish
        let pose_c = glam::Mat4::from_translation(glam::vec3(3.0, 0.0, 0.0))
            * glam::Mat4::from_rotation_y(-25_f32.to_radians());
        let mut frustum_c = CameraFrustumItem::default();
        frustum_c.pose = pose_c.to_cols_array_2d();
        frustum_c.fov_y = 55_f32.to_radians();
        frustum_c.aspect = 1.0;
        frustum_c.near = 0.1;
        frustum_c.far = 3.0;
        frustum_c.color = COLOR_C;

        self.aux_frustums = vec![frustum_a, frustum_b, frustum_c];
        self.aux_built = true;
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_aux(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Camera Frustums").strong());
        ui.label(
            "Three placed cameras. 'Look through' adopts that camera's exact POV; \
             'Overview' steps back to see all three frustums as objects in the scene.",
        );

        // Collect which button was clicked before doing any borrowing of other
        // fields. This avoids a borrow conflict between the frustum iterator
        // and the mutable fields (cam_animator, aux_active_frustum) that are
        // modified in the same frame.
        #[derive(Clone, Copy)]
        enum AuxAction { LookThrough(usize), Overview }

        let mut action: Option<AuxAction> = None;
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let names = ["A (blue)", "B (orange)", "C (green)"];
            for i in 0..self.aux_frustums.len() {
                if ui.button(format!("Look through {}", names[i])).clicked() {
                    action = Some(AuxAction::LookThrough(i));
                }
            }
            if ui.button("Overview").clicked() {
                action = Some(AuxAction::Overview);
            }
        });

        // Apply the action now that the UI loop is done.
        match action {
            Some(AuxAction::LookThrough(i)) => {
                // camera_view_target places the orbit camera AT the frustum's
                // eye position looking along the frustum's own forward axis.
                let t = self.aux_frustums[i].camera_view_target();
                self.cam_animator.fly_to(
                    &self.camera,
                    t.center,
                    t.distance,
                    t.orientation,
                    1.2,
                );
                self.aux_active_frustum = Some(i);
            }
            Some(AuxAction::Overview) => {
                // camera_target frames the frustum wireframes from outside.
                let t = self.aux_frustums[1].camera_target(5.0);
                self.cam_animator.fly_to(
                    &self.camera,
                    glam::Vec3::ZERO,
                    t.distance,
                    t.orientation,
                    1.2,
                );
                self.aux_active_frustum = None;
            }
            None => {}
        }

        ui.separator();
        ui.label(egui::RichText::new("Active-Camera HUD Overlay").strong());
        ui.label(
            "Corner brackets and crosshair mark the active camera in its wireframe colour. \
             This is the same pattern used by CAD and VFX tools to show which camera \
             the viewport is currently looking through. \
             Dim brackets indicate overview mode (no active camera).",
        );

        ui.add_space(2.0);
        match self.aux_active_frustum {
            None => {
                ui.label("(overview - click 'Look through' to activate a camera)");
            }
            Some(i) => {
                let label = ["A - blue", "B - orange", "C - green"][i];
                ui.label(format!("Active camera: {label}"));
            }
        }

        ui.add_space(4.0);
        ui.add(egui::Slider::new(&mut self.aux_img_alpha, 0.0..=1.0).text("Overlay alpha"));
        ui.add(egui::Slider::new(&mut self.aux_img_scale, 0.25..=4.0).text("Overlay scale"));
    }

    // -------------------------------------------------------------------------
    // Per-frame overlay push
    // -------------------------------------------------------------------------

    /// Push the active-camera HUD overlay into `fd` for the current frame.
    ///
    /// Always pushes something: dim grey brackets in overview mode,
    /// vivid coloured brackets + crosshair when looking through a camera.
    pub(crate) fn aux_push_screen_images(&self, fd: &mut viewport_lib::FrameData) {
        // Pick colour and size based on whether a camera is active.
        let (color, size, arm_len, thick) = match self.aux_active_frustum {
            Some(idx) => {
                let fc = self.aux_frustums[idx].color;
                let c: [u8; 4] = [
                    (fc[0] * 255.0) as u8,
                    (fc[1] * 255.0) as u8,
                    (fc[2] * 255.0) as u8,
                    255u8,
                ];
                (c, 72u32, 28u32, 4u32)
            }
            None => {
                // Dim neutral brackets hint at the feature without being distracting.
                ([160u8, 160u8, 160u8, 80u8], 40u32, 16u32, 3u32)
            }
        };

        // Corner bracket images: L-shaped marks opening toward the viewport centre.
        let brackets: [(ImageAnchor, bool, bool); 4] = [
            (ImageAnchor::TopLeft, false, false),
            (ImageAnchor::TopRight, true, false),
            (ImageAnchor::BottomLeft, false, true),
            (ImageAnchor::BottomRight, true, true),
        ];
        for (anchor, flip_x, flip_y) in brackets {
            let mut item = ScreenImageItem::default();
            item.pixels = bracket_pixels(color, size, arm_len, thick, flip_x, flip_y);
            item.width = size;
            item.height = size;
            item.anchor = anchor;
            item.scale = self.aux_img_scale;
            item.alpha = self.aux_img_alpha;
            fd.scene.screen_images.push(item);
        }

        // Centre crosshair: only shown when a camera is active.
        if self.aux_active_frustum.is_some() {
            const XS: u32 = 40;
            let mut item = ScreenImageItem::default();
            item.pixels = crosshair_pixels(color, XS, 3, 5);
            item.width = XS;
            item.height = XS;
            item.anchor = ImageAnchor::Center;
            item.scale = self.aux_img_scale;
            item.alpha = self.aux_img_alpha;
            fd.scene.screen_images.push(item);
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel-drawing helpers
// ---------------------------------------------------------------------------

/// Generate an L-shaped corner bracket.
///
/// `flip_x` mirrors horizontally (corner opens left instead of right).
/// `flip_y` mirrors vertically (corner opens up instead of down).
fn bracket_pixels(
    color: [u8; 4],
    size: u32,
    arm_len: u32,
    thickness: u32,
    flip_x: bool,
    flip_y: bool,
) -> Vec<[u8; 4]> {
    let mut pixels = vec![[0u8; 4]; (size * size) as usize];
    for i in 0..arm_len {
        for t in 0..thickness {
            // Horizontal arm: rows 0..thickness, cols 0..arm_len
            let x = if flip_x { size - 1 - i } else { i };
            let y = if flip_y { size - 1 - t } else { t };
            pixels[(y * size + x) as usize] = color;

            // Vertical arm: cols 0..thickness, rows 0..arm_len
            let x2 = if flip_x { size - 1 - t } else { t };
            let y2 = if flip_y { size - 1 - i } else { i };
            pixels[(y2 * size + x2) as usize] = color;
        }
    }
    pixels
}

/// Generate a plus-shaped crosshair with a transparent gap at the centre.
fn crosshair_pixels(color: [u8; 4], size: u32, thickness: u32, gap: u32) -> Vec<[u8; 4]> {
    let mut pixels = vec![[0u8; 4]; (size * size) as usize];
    let cx = size / 2;
    let cy = size / 2;
    let half_t = thickness / 2;
    for i in 0..size {
        let d = (i as i32 - cx as i32).unsigned_abs();
        if d <= gap {
            continue;
        }
        // Horizontal bar: column i, rows cy±half_t
        for t in 0..thickness {
            let y = cy.saturating_sub(half_t) + t;
            if y < size {
                pixels[(y * size + i) as usize] = color;
            }
        }
        // Vertical bar: row i, cols cx±half_t
        for t in 0..thickness {
            let x = cx.saturating_sub(half_t) + t;
            if x < size {
                pixels[(i * size + x) as usize] = color;
            }
        }
    }
    pixels
}
