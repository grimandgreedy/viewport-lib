//! Showcase 5: Advanced Rendering - build method.

use crate::AppState;
use viewport_lib::Material;

impl AppState {
    /// Build the scene for Showcase 5 (Phase 6 features).
    pub(crate) fn build_adv_scene(&mut self) {
        self.adv_scene = viewport_lib::scene::Scene::new();
        self.adv_selection.clear();

        let m = self.upload_box();
        let id = self.adv_scene.add_named(
            "Gold (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, -1.5)),
            Material::pbr([1.0, 0.78, 0.2], 0.95, 0.05),
        );
        self.adv_selection.select_one(id);

        let m = self.upload_box();
        self.adv_scene.add_named(
            "Brushed Steel (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, 1.5)),
            Material::pbr([0.82, 0.82, 0.86], 0.75, 0.35),
        );

        let m = self.upload_box();
        self.adv_scene.add_named(
            "Shiny Blue (Blinn-Phong)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, 0.0, -1.5)),
            {
                let mut mat = Material::from_color([0.2, 0.4, 0.9]);
                mat.specular = 0.9;
                mat.shininess = 128.0;
                mat
            },
        );

        let m = self.upload_box();
        self.adv_scene.add_named(
            "Matte Green (Blinn-Phong)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, 0.0, 1.5)),
            {
                let mut mat = Material::from_color([0.2, 0.7, 0.3]);
                mat.specular = 0.05;
                mat.diffuse = 0.95;
                mat.shininess = 4.0;
                mat
            },
        );

        let m = self.upload_box();
        self.adv_scene.add_named(
            "Wall (occluder)",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(7.5, 2.5, 0.25),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.25, 3.5),
            ),
            Material::from_color([0.55, 0.55, 0.55]),
        );

        let m = self.upload_box();
        self.adv_scene.add_named(
            "Hidden Magenta (x-ray target)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 5.5)),
            Material::from_color([0.9, 0.3, 0.7]),
        );

        self.adv_built = true;
    }
}
