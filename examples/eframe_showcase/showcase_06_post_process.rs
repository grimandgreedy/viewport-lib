//! Showcase 6: Post-Processing — build method.
//!
//! Note: the full HDR pipeline (`renderer.render()`) requires direct access to the
//! surface texture view, which is not available via eframe's paint callback.
//! Post-process toggles (bloom, SSAO, FXAA, tone mapping) are reflected in
//! `PostProcessSettings` but the HDR pass is not applied in this path.

use crate::App;
use crate::geometry::make_uv_sphere;
use viewport_lib::{Material, MeshId, ViewportRenderer};

impl App {
    /// Build the scene for Showcase 6 (post-processing / PBR scene).
    pub(crate) fn build_pp_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.pp_scene = viewport_lib::scene::Scene::new();

        let m = self.upload_box(renderer);
        self.pp_scene.add_named(
            "Ground",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(10.0, 10.0, 0.15),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, -0.575),
            ),
            Material {
                base_color: [0.6, 0.6, 0.6],
                use_pbr: true,
                metallic: 0.0,
                roughness: 0.9,
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.pp_scene.add_named(
            "Gold (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, -1.2, 0.0)),
            Material {
                base_color: [1.0, 0.78, 0.2],
                use_pbr: true,
                metallic: 0.95,
                roughness: 0.05,
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.pp_scene.add_named(
            "Brushed Steel (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, -1.2, 0.0)),
            Material {
                base_color: [0.82, 0.82, 0.86],
                use_pbr: true,
                metallic: 0.75,
                roughness: 0.35,
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.pp_scene.add_named(
            "Chrome (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, 1.2, 0.0)),
            Material {
                base_color: [0.9, 0.9, 0.95],
                use_pbr: true,
                metallic: 1.0,
                roughness: 0.02,
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.pp_scene.add_named(
            "Ceramic (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, 1.2, 0.0)),
            Material {
                base_color: [0.85, 0.7, 0.6],
                use_pbr: true,
                metallic: 0.0,
                roughness: 0.9,
                ..Material::default()
            },
        );

        let sphere = make_uv_sphere(32, 16, 0.6);
        let sphere_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("pp sphere upload");
        let sphere_id = MeshId::from_index(sphere_idx);
        self.pp_scene.add_named(
            "Sphere Test",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.1)),
            Material {
                base_color: [0.85, 0.45, 0.35],
                use_pbr: true,
                metallic: 0.0,
                roughness: 0.55,
                ..Material::default()
            },
        );

        let m = self.upload_box(renderer);
        self.pp_scene.add_named(
            "Pillar",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.5, 0.5, 2.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, 1.0),
            ),
            Material {
                base_color: [0.7, 0.7, 0.75],
                use_pbr: true,
                metallic: 0.1,
                roughness: 0.7,
                ..Material::default()
            },
        );

        self.pp_built = true;
    }
}
