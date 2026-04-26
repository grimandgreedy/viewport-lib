//! Showcase 8: Shadow Demo : build method.

use crate::App;
use crate::geometry::{make_box_with_uvs, make_uv_sphere};
use viewport_lib::{Material, MeshId, ViewportRenderer};

impl App {
    pub(crate) fn build_shadow_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.shd_scene = viewport_lib::scene::Scene::new();

        let ground_mesh = make_box_with_uvs(20.0, 20.0, 0.2);
        let ground_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("ground mesh upload");
        let ground_id = MeshId::from_index(ground_idx);
        self.shd_scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.1)),
            Material::pbr([0.55, 0.52, 0.48], 0.0, 0.9),
        );

        let sphere_mesh = make_uv_sphere(24, 12, 0.5);
        let sphere_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("sphere mesh upload");
        let sphere_id = MeshId::from_index(sphere_idx);

        let sphere_dense_mesh = make_uv_sphere(64, 32, 0.5);
        let sphere_dense_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_dense_mesh)
            .expect("dense sphere mesh upload");
        let sphere_dense_id = MeshId::from_index(sphere_dense_idx);

        let box_mesh = make_box_with_uvs(1.0, 1.0, 1.0);
        let box_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &box_mesh)
            .expect("box mesh upload");
        let box_id = MeshId::from_index(box_idx);

        let object_data: &[(&str, glam::Vec3, [f32; 3])] = &[
            (
                "Sphere Near",
                glam::Vec3::new(-1.5, 1.0, 0.5),
                [0.85, 0.35, 0.25],
            ),
            (
                "Box Near",
                glam::Vec3::new(1.5, 1.0, 0.5),
                [0.25, 0.55, 0.85],
            ),
            (
                "Sphere Mid",
                glam::Vec3::new(-3.0, -3.0, 0.5),
                [0.8, 0.7, 0.3],
            ),
            (
                "Box Mid",
                glam::Vec3::new(3.0, -3.0, 0.5),
                [0.35, 0.75, 0.45],
            ),
        ];
        for (name, pos, color) in object_data {
            let mesh_id = if *name == "Sphere Near" {
                sphere_dense_id
            } else if name.contains("Sphere") {
                sphere_id
            } else {
                box_id
            };
            self.shd_scene.add_named(
                name,
                Some(mesh_id),
                glam::Mat4::from_translation(*pos),
                Material::pbr(*color, 0.0, 0.5),
            );
        }

        let pillar_mesh = make_box_with_uvs(0.4, 0.4, 3.0);
        let pillar_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &pillar_mesh)
            .expect("pillar mesh upload");
        let pillar_id = MeshId::from_index(pillar_idx);
        self.shd_scene.add_named(
            "Tall Pillar",
            Some(pillar_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, -6.0, 1.5)),
            Material::pbr([0.65, 0.65, 0.70], 0.0, 0.6),
        );

        self.shd_built = true;
    }
}
