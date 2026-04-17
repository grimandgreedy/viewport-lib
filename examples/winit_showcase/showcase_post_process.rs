//! Showcase 6: Post-Processing - build method.

use crate::AppState;
use crate::geometry::make_uv_sphere;
use viewport_lib::Material;

impl AppState {
    /// Build the scene for Showcase 6 (post-processing).
    pub(crate) fn build_pp_scene(&mut self) {
        self.pp_scene = viewport_lib::scene::Scene::new();

        // Ground plane - wide flat slab so boxes cast contact shadows onto it.
        let m = self.upload_box();
        self.pp_scene.add_named(
            "Ground",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(10.0, 0.15, 10.0),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, -0.575, 0.0),
            ),
            Material {
                base_color: [0.6, 0.6, 0.6],
                use_pbr: true,
                metallic: 0.0,
                roughness: 0.9,
                ..Material::default()
            },
        );

        // Four boxes sitting on the ground, close together so sides occlude each other.
        let m = self.upload_box();
        self.pp_scene.add_named(
            "Gold (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, 0.0, -1.2)),
            Material {
                base_color: [1.0, 0.78, 0.2],
                use_pbr: true,
                metallic: 0.95,
                roughness: 0.05,
                ..Material::default()
            },
        );

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Brushed Steel (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, 0.0, -1.2)),
            Material {
                base_color: [0.82, 0.82, 0.86],
                use_pbr: true,
                metallic: 0.75,
                roughness: 0.35,
                ..Material::default()
            },
        );

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Chrome (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, 0.0, 1.2)),
            Material {
                base_color: [0.9, 0.9, 0.95],
                use_pbr: true,
                metallic: 1.0,
                roughness: 0.02,
                ..Material::default()
            },
        );

        let m = self.upload_box();
        self.pp_scene.add_named(
            "Ceramic (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, 0.0, 1.2)),
            Material {
                base_color: [0.85, 0.7, 0.6],
                use_pbr: true,
                metallic: 0.0,
                roughness: 0.9,
                ..Material::default()
            },
        );

        // Sphere added for comparing curved-surface lighting against the boxes.
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
            Material {
                base_color: [0.85, 0.45, 0.35],
                use_pbr: true,
                metallic: 0.0,
                roughness: 0.55,
                ..Material::default()
            },
        );

        // Tall central pillar - creates strong occlusion between itself and the surrounding boxes.
        let m = self.upload_box();
        self.pp_scene.add_named(
            "Pillar",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.5, 2.5, 0.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 1.0, 0.0),
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
