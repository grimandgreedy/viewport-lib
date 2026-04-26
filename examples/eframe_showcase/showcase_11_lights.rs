//! Showcase 11: Lights : build method.
//!
//! A flat ground plane plus a 3×3 grid of white spheres : neutral surfaces that make
//! light color, cone angle, and attenuation directly visible.

use crate::App;
use crate::geometry::make_box_with_uvs;
use viewport_lib::{Material, ViewportRenderer, scene::Scene};

impl App {
    /// Build the scene for Showcase 11 (Lights demo).
    pub(crate) fn build_lights_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.lights_scene = Scene::new();

        // Ground plane : thin slab, Z-up.
        let ground_mesh = make_box_with_uvs(16.0, 16.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("lights ground mesh");
        self.lights_scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.05)),
            {
                let mut m = Material::from_color([0.45, 0.45, 0.48]);
                m.roughness = 0.9;
                m
            },
        );

        // 3×3 grid of spheres.
        let sphere_mesh = viewport_lib::primitives::sphere(0.6, 32, 16);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("lights sphere mesh");

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
                    {
                        let mut m = Material::from_color([0.92, 0.92, 0.92]);
                        m.roughness = 0.35;
                        m
                    },
                );
            }
        }

        self.lights_built = true;
    }
}
