//! Showcase 2: Scene Graph + Materials : build method.

use crate::App;
use viewport_lib::{Material, ViewportRenderer};

impl App {
    /// Build the initial scene for Showcase 2.
    pub(crate) fn build_scene_graph(&mut self, renderer: &mut ViewportRenderer) {
        self.scene = viewport_lib::scene::Scene::new();
        self.layer_b = Some(self.scene.add_layer("Layer B"));
        self.layer_b_visible = true;
        self.selection.clear();

        let positions = [
            [-1.5, -1.5, 0.0],
            [1.5, -1.5, 0.0],
            [-1.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
        ];
        let colors = [
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.3],
            [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3],
        ];
        for (i, (pos, color)) in positions.iter().zip(&colors).enumerate() {
            let mesh = self.upload_box(renderer);
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material::from_color(*color);
            let name = format!("Box {}", i + 1);
            let id = self.scene.add_named(&name, Some(mesh), transform, mat);
            if i >= 2 {
                self.scene.set_layer(id, self.layer_b.unwrap());
            }
        }

        self.scene_built = true;
    }
}
