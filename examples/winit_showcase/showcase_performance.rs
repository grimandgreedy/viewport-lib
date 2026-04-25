//! Showcase 3: Performance at Scale - build method.

use crate::AppState;
use viewport_lib::{Material, PickAccelerator, scene::Scene};

impl AppState {
    /// Build a 10x10x3 grid (300 boxes) for the Performance showcase.
    /// All boxes share a single mesh to demonstrate GPU instancing.
    pub(crate) fn build_perf_scene(&mut self) {
        self.perf_scene = Scene::new();
        self.perf_selection.clear();

        // Upload one shared mesh (all 300 nodes reference it).
        let mesh = self.upload_box();
        self.perf_mesh = Some(mesh);

        let spacing = 2.5_f32;
        let colors: [[f32; 3]; 6] = [
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.3],
            [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3],
            [0.9, 0.5, 0.2],
            [0.5, 0.3, 0.9],
        ];

        let (nx, ny, nz) = (100, 100, 100); // 1,000,000 boxes
        let mut count = 0u32;
        for y in 0..ny {
            for z in 0..nz {
                for x in 0..nx {
                    let pos = glam::Vec3::new(
                        (x as f32 - nx as f32 / 2.0) * spacing,
                        (y as f32) * spacing,
                        (z as f32 - nz as f32 / 2.0) * spacing,
                    );
                    let transform = glam::Mat4::from_translation(pos);
                    let color = colors[count as usize % colors.len()];
                    let mat = Material::from_color(color);
                    let name = format!("Perf {count}");
                    self.perf_scene.add_named(&name, Some(mesh), transform, mat);
                    count += 1;
                }
            }
        }

        self.perf_total_objects = count;

        // Build the BVH for accelerated picking.
        let resources = self.renderer.resources();
        self.pick_accelerator = Some(PickAccelerator::build_from_scene(&self.perf_scene, |mid| {
            resources.mesh(mid.index()).map(|m| m.aabb)
        }));
    }
}
