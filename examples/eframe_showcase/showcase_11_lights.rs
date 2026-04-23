//! Showcase 11: Lights — build method.
//!
//! A flat ground plane plus a 3×3 grid of white spheres — neutral surfaces that make
//! light color, cone angle, and attenuation directly visible.

use crate::App;
use crate::geometry::make_box_with_uvs;
use viewport_lib::{Material, MeshId, ViewportRenderer, scene::Scene};

impl App {
    /// Build the scene for Showcase 11 (Lights demo).
    pub(crate) fn build_lights_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.lights_scene = Scene::new();

        // Ground plane — thin slab, Z-up.
        let ground_mesh = make_box_with_uvs(16.0, 16.0, 0.1);
        let ground_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("lights ground mesh");
        let ground_id = MeshId::from_index(ground_idx);
        self.lights_scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.05)),
            Material {
                base_color: [0.45, 0.45, 0.48],
                roughness: 0.9,
                metallic: 0.0,
                ..Material::default()
            },
        );

        // 3×3 grid of spheres.
        let sphere_mesh = viewport_lib::primitives::sphere(0.6, 32, 16);
        let sphere_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("lights sphere mesh");
        let sphere_id = MeshId::from_index(sphere_idx);

        for row in 0..3i32 {
            for col in 0..3i32 {
                let x = (col - 1) as f32 * 4.0;
                let y = (row - 1) as f32 * 4.0;
                let z = 0.6_f32; // rest on ground
                let name = format!("Sphere ({row},{col})");
                self.lights_scene.add_named(
                    &name,
                    Some(sphere_id),
                    glam::Mat4::from_translation(glam::Vec3::new(x, y, z)),
                    Material {
                        base_color: [0.92, 0.92, 0.92],
                        roughness: 0.35,
                        metallic: 0.0,
                        ..Material::default()
                    },
                );
            }
        }

        self.lights_built = true;
    }
}
