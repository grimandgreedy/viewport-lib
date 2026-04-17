//! Showcase 2: Scene Graph + Materials - build method.

use crate::AppState;
use viewport_lib::{Material, scene::Scene};

impl AppState {
    /// Build the initial scene for Showcase 2.
    pub(crate) fn build_scene(&mut self) {
        self.scene = Scene::new();
        self.layer_b = self.scene.add_layer("Layer B");
        self.layer_b_visible = true;
        self.selection.clear();

        // Add 4 boxes in a grid on the default layer.
        let positions = [
            [-1.5, 0.0, -1.5],
            [1.5, 0.0, -1.5],
            [-1.5, 0.0, 1.5],
            [1.5, 0.0, 1.5],
        ];
        let colors = [
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.3],
            [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3],
        ];
        for (i, (pos, color)) in positions.iter().zip(&colors).enumerate() {
            let mesh = self.upload_box();
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material {
                base_color: *color,
                ..Material::default()
            };
            let name = format!("Box {}", i + 1);
            let id = self.scene.add_named(&name, Some(mesh), transform, mat);

            // Put boxes 3 & 4 on layer B.
            if i >= 2 {
                self.scene.set_layer(id, self.layer_b);
            }
        }
    }
}
