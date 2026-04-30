//! Showcase 9: Annotation Labels : build and render methods.
//!
//! Labels now render natively via `LabelItem` in `OverlayFrame`, replacing the
//! previous egui painter approach.

use crate::App;
use crate::geometry::make_box_with_uvs;
use viewport_lib::{Camera, LabelItem, Material, ViewportRenderer};

impl App {
    /// Build the scene for Showcase 9 (Annotation Labels demo).
    pub(crate) fn build_annotation_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::scene::Scene;

        self.ann_scene = Scene::new();

        let mut place_marker =
            |renderer: &mut ViewportRenderer, pos: glam::Vec3, color: [f32; 3]| {
                let mesh = make_box_with_uvs(0.3, 0.3, 0.3);
                let id = renderer
                    .resources_mut()
                    .upload_mesh_data(&self.device, &mesh)
                    .expect("annotation marker upload");
                self.ann_scene.add_named(
                    "Marker",
                    Some(id),
                    glam::Mat4::from_translation(pos),
                    Material::from_color(color),
                );
            };

        place_marker(renderer, glam::Vec3::ZERO, [1.0, 1.0, 1.0]);
        place_marker(renderer, glam::Vec3::new(2.0, 3.0, 0.0), [1.0, 0.9, 0.1]);
        place_marker(renderer, glam::Vec3::new(-3.0, 2.0, 0.0), [0.4, 0.8, 1.0]);
        place_marker(renderer, glam::Vec3::new(0.0, 300.0, 0.0), [1.0, 0.0, 0.0]);

        self.ann_labels = vec![
            LabelItem {
                world_anchor: Some([0.0, 0.0, 0.0]),
                text: "Origin (0,0,0)".into(),
                color: [1.0, 1.0, 1.0, 1.0],
                background: true,
                ..Default::default()
            },
            LabelItem {
                world_anchor: Some([2.0, 3.0, 0.0]),
                text: "Peak Pressure: 101.3 kPa".into(),
                color: [1.0, 0.9, 0.1, 1.0],
                leader_line: true,
                background: true,
                ..Default::default()
            },
            LabelItem {
                world_anchor: Some([-3.0, 2.0, 0.0]),
                text: "Outlet".into(),
                color: [0.4, 0.8, 1.0, 1.0],
                background: true,
                ..Default::default()
            },
            LabelItem {
                world_anchor: Some([0.0, 300.0, 0.0]),
                text: "Behind camera (clipped)".into(),
                color: [1.0, 0.0, 0.0, 1.0],
                background: true,
                ..Default::default()
            },
        ];

        self.ann_built = true;
    }

    /// Reset the camera to a good viewing angle for the annotation demo.
    pub(crate) fn reset_annotation_camera(&mut self) {
        self.camera = Camera {
            center: glam::Vec3::new(0.0, 0.5, 1.0),
            distance: 12.0,
            orientation: glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.1),
            ..Camera::default()
        };
    }
}
