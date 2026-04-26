//! Showcase 10: Camera Tools : build method.
//!
//! Six coloured boxes arranged along the cardinal axes so that every named
//! view preset shows a clearly different face of the layout:
//!   Front (+Y): red/green split visible
//!   Right (+X): green/blue split visible
//!   Top  (+Z): full cross visible from above

use crate::App;
use viewport_lib::{Material, ViewportRenderer, scene::Scene};

impl App {
    /// Build the scene for Showcase 10 (Camera Tools demo).
    pub(crate) fn build_camera_tools_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.cam_tools_scene = Scene::new();

        let objects: &[(&str, glam::Vec3, [f32; 3])] = &[
            ("Origin", glam::Vec3::ZERO, [0.70, 0.70, 0.70]),
            (
                "+X Right",
                glam::Vec3::new(4.0, 0.0, 0.0),
                [0.85, 0.22, 0.22],
            ),
            (
                "-X Left",
                glam::Vec3::new(-4.0, 0.0, 0.0),
                [0.50, 0.10, 0.10],
            ),
            (
                "+Y Front",
                glam::Vec3::new(0.0, 4.0, 0.0),
                [0.22, 0.78, 0.22],
            ),
            (
                "-Y Back",
                glam::Vec3::new(0.0, -4.0, 0.0),
                [0.10, 0.40, 0.10],
            ),
            ("+Z Up", glam::Vec3::new(0.0, 0.0, 4.0), [0.25, 0.50, 0.90]),
        ];

        for (name, pos, color) in objects {
            let mesh = self.upload_box(renderer);
            self.cam_tools_scene.add_named(
                name,
                Some(mesh),
                glam::Mat4::from_translation(*pos),
                Material::from_color(*color),
            );
        }

        self.cam_tools_built = true;
    }
}
