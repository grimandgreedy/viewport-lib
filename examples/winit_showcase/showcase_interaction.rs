//! Showcase 4: Professional Interaction - build method.

use crate::AppState;
use viewport_lib::Material;

impl AppState {
    /// Build a small scene for the Interaction showcase (5 boxes in a cross).
    pub(crate) fn build_interact_scene(&mut self) {
        self.interact_scene = viewport_lib::scene::Scene::new();
        self.interact_selection.clear();

        let positions = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
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
            let mesh = self.upload_box();
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material::from_color(*color);
            let id = self
                .interact_scene
                .add_named(name, Some(mesh), transform, mat);
            // Select the first box by default.
            if i == 0 {
                self.interact_selection.select_one(id);
            }
        }

        self.interact_built = true;
    }
}
