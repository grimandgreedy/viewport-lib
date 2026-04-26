//! Showcase 4: Professional Interaction : build method.

use crate::App;
use viewport_lib::{Material, ViewportRenderer};

impl App {
    /// Build a small cross-shaped scene for the Interaction showcase.
    pub(crate) fn build_interact_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.interact_scene = viewport_lib::scene::Scene::new();
        self.interact_selection.clear();

        let positions = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
        ];
        let colors = [
            [0.85, 0.85, 0.85],
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.3],
            [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3],
        ];
        let names = ["Center", "Right", "Left", "Front", "Back"];

        for (i, ((pos, color), name)) in positions.iter().zip(&colors).zip(&names).enumerate() {
            let mesh = self.upload_box(renderer);
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material::from_color(*color);
            let id = self
                .interact_scene
                .add_named(name, Some(mesh), transform, mat);
            if i == 0 {
                self.interact_selection.select_one(id);
            }
        }

        self.interact_built = true;
    }
}
