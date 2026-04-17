//! Showcase 8: Shadow Demo - build method.

use crate::AppState;
use crate::geometry::{make_box_with_uvs, make_uv_sphere};
use viewport_lib::{Material, MeshId};

impl AppState {
    pub(crate) fn build_shadow_scene(&mut self) {
        self.shd_scene = viewport_lib::scene::Scene::new();

        // Ground plane: large flat box.
        let ground_mesh = make_box_with_uvs(20.0, 0.2, 20.0);
        let ground_idx = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("ground mesh upload");
        let ground_id = MeshId::from_index(ground_idx);
        self.shd_scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, -0.1, 0.0)),
            Material {
                base_color: [0.55, 0.52, 0.48],
                use_pbr: true,
                roughness: 0.9,
                ..Material::default()
            },
        );

        // Sphere mesh for casters.
        let sphere_mesh = make_uv_sphere(24, 12, 0.5);
        let sphere_idx = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("sphere mesh upload");
        let sphere_id = MeshId::from_index(sphere_idx);

        // Box mesh for casters.
        let box_mesh = make_box_with_uvs(1.0, 1.0, 1.0);
        let box_idx = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &box_mesh)
            .expect("box mesh upload");
        let box_id = MeshId::from_index(box_idx);

        // Several objects at varying distances to demonstrate cascade transitions.
        let object_data: &[(&str, glam::Vec3, [f32; 3])] = &[
            (
                "Sphere Near",
                glam::Vec3::new(-1.5, 0.5, 1.0),
                [0.85, 0.35, 0.25],
            ),
            (
                "Box Near",
                glam::Vec3::new(1.5, 0.5, 1.0),
                [0.25, 0.55, 0.85],
            ),
            (
                "Sphere Mid",
                glam::Vec3::new(-3.0, 0.5, -3.0),
                [0.8, 0.7, 0.3],
            ),
            (
                "Box Mid",
                glam::Vec3::new(3.0, 0.5, -3.0),
                [0.35, 0.75, 0.45],
            ),
        ];
        for (name, pos, color) in object_data {
            let mesh_id = if name.contains("Sphere") {
                sphere_id
            } else {
                box_id
            };
            self.shd_scene.add_named(
                name,
                Some(mesh_id),
                glam::Mat4::from_translation(*pos),
                Material {
                    base_color: *color,
                    use_pbr: true,
                    roughness: 0.5,
                    ..Material::default()
                },
            );
        }

        // Tall pillar - long shadow to show cascade range.
        let pillar_mesh = make_box_with_uvs(0.4, 3.0, 0.4);
        let pillar_idx = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &pillar_mesh)
            .expect("pillar mesh upload");
        let pillar_id = MeshId::from_index(pillar_idx);
        self.shd_scene.add_named(
            "Tall Pillar",
            Some(pillar_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 1.5, -6.0)),
            Material {
                base_color: [0.65, 0.65, 0.70],
                use_pbr: true,
                roughness: 0.6,
                ..Material::default()
            },
        );

        self.shd_built = true;
    }
}
