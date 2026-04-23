//! Showcase 5: Advanced Rendering — build method.

use crate::App;
use viewport_lib::{Material, ViewportRenderer};

impl App {
    /// Build the scene for Showcase 5 (PBR vs Blinn-Phong, clip planes, outlines, x-ray).
    pub(crate) fn build_adv_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.adv_scene = viewport_lib::scene::Scene::new();
        self.adv_selection.clear();

        let m = self.upload_box(renderer);
        let id = self.adv_scene.add_named(
            "Gold (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, -1.5, 0.0)),
            Material {
                base_color: [1.0, 0.78, 0.2],
                use_pbr: true,
                metallic: 0.95,
                roughness: 0.05,
                ..Material::default()
            },
        );
        self.adv_selection.select_one(id);

        let m = self.upload_box(renderer);
        self.adv_scene.add_named(
            "Brushed Steel (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 1.5, 0.0)),
            Material {
                base_color: [0.82, 0.82, 0.86],
                use_pbr: true,
                metallic: 0.75,
                roughness: 0.35,
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.adv_scene.add_named(
            "Shiny Blue (Blinn-Phong)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, -1.5, 0.0)),
            Material {
                base_color: [0.2, 0.4, 0.9],
                specular: 0.9,
                shininess: 128.0,
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.adv_scene.add_named(
            "Matte Green (Blinn-Phong)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, 1.5, 0.0)),
            Material {
                base_color: [0.2, 0.7, 0.3],
                specular: 0.05,
                diffuse: 0.95,
                shininess: 4.0,
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.adv_scene.add_named(
            "Wall (occluder)",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(7.5, 0.25, 2.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 3.5, 0.25),
            ),
            Material {
                base_color: [0.55, 0.55, 0.55],
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.adv_scene.add_named(
            "Hidden Magenta (x-ray target)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 5.5, 0.0)),
            Material {
                base_color: [0.9, 0.3, 0.7],
                ..Material::default()
            },
        );

        self.adv_built = true;
    }
}
