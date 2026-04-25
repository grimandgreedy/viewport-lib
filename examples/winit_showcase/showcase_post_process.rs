//! Showcase 6: Post-Processing - build method.

use crate::AppState;
use crate::geometry::make_uv_sphere;
use viewport_lib::Material;

impl AppState {
    /// Build the scene for Showcase 6 (post-processing).
    pub(crate) fn build_pp_scene(&mut self) {
        self.pp_scene = viewport_lib::scene::Scene::new();

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Ground",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(10.0, 0.15, 10.0),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, -0.575, 0.0),
            ),
            Material::pbr([0.6, 0.6, 0.6], 0.0, 0.9),
        );

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Gold (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, 0.0, -1.2)),
            Material::pbr([1.0, 0.78, 0.2], 0.95, 0.05),
        );

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Brushed Steel (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, 0.0, -1.2)),
            Material::pbr([0.82, 0.82, 0.86], 0.75, 0.35),
        );

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Chrome (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, 0.0, 1.2)),
            Material::pbr([0.9, 0.9, 0.95], 1.0, 0.02),
        );

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Ceramic (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, 0.0, 1.2)),
            Material::pbr([0.85, 0.7, 0.6], 0.0, 0.9),
        );

        let sphere = make_uv_sphere(32, 16, 0.6);
        let sphere_idx = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("post-process sphere upload");
        let sphere_id = viewport_lib::MeshId::from_index(sphere_idx);
        self.pp_scene.add_named(
            "Sphere Test",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.1, 0.0)),
            Material::pbr([0.85, 0.45, 0.35], 0.0, 0.55),
        );

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Pillar",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.5, 2.5, 0.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 1.0, 0.0),
            ),
            Material::pbr([0.7, 0.7, 0.75], 0.1, 0.7),
        );

        self.pp_built = true;
    }
}
