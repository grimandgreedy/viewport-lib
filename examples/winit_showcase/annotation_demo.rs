//! Annotation label showcase.
//!
//! This mode exercises [`AnnotationLabel`] and [`world_to_screen`] with a few
//! fixed anchors. In the `winit` showcase the labels are projected but not drawn
//! as text; for on-screen text rendering, use the `egui` path.

use crate::AppState;
use viewport_lib::{AnnotationLabel, Camera, Material, MeshId, world_to_screen};

impl AppState {
    /// Build the scene for Showcase 9 (Annotation Labels demo).
    ///
    /// Places a small coloured marker box at each annotation's `world_pos` so the
    /// anchor is visible in the 3D view.  The annotation data is stored on `AppState`
    /// and projected each frame inside [`AppState::annotation_title`].
    pub(crate) fn build_annotation_scene(&mut self) {
        use viewport_lib::scene::Scene;

        self.ann_scene = Scene::new();

        // Helper: upload a small box at a given position with the given colour.
        let mut place_marker = |pos: glam::Vec3, color: [f32; 3]| {
            let mesh = crate::geometry::make_box_with_uvs(0.3, 0.3, 0.3);
            let idx = self
                .renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &mesh)
                .expect("annotation marker upload");
            let id = MeshId::from_index(idx);
            self.ann_scene.add_named(
                "Marker",
                Some(id),
                glam::Mat4::from_translation(pos),
                Material::from_color(color),
            );
        };

        // Label 0 - Origin marker (white).
        place_marker(glam::Vec3::ZERO, [1.0, 1.0, 1.0]);

        // Label 1 - Peak pressure with leader line (yellow).
        place_marker(glam::Vec3::new(2.0, 3.0, 0.0), [1.0, 0.9, 0.1]);

        // Label 2 - Outlet (light blue).
        place_marker(glam::Vec3::new(-3.0, 0.0, 2.0), [0.4, 0.8, 1.0]);

        // Label 3 - Placed behind the default camera: should project to None.
        // Default camera looks toward -Z with the scene at the origin, so
        // a point at Z = +300 is well behind the near plane and will not be visible.
        place_marker(glam::Vec3::new(0.0, 0.0, 300.0), [1.0, 0.0, 0.0]);

        // Build the matching AnnotationLabel vec.
        self.ann_labels = vec![
            {
                let mut l = AnnotationLabel::default();
                l.world_pos = glam::Vec3::ZERO;
                l.text = "Origin (0,0,0)".to_string();
                l.color = [1.0, 1.0, 1.0, 1.0];
                l
            },
            {
                let mut l = AnnotationLabel::default();
                l.world_pos = glam::Vec3::new(2.0, 3.0, 0.0);
                l.text = "Peak Pressure: 101.3 kPa".to_string();
                l.leader_end = Some(glam::Vec3::new(1.0, 1.0, 0.0));
                l.color = [1.0, 0.9, 0.1, 1.0];
                l
            },
            {
                let mut l = AnnotationLabel::default();
                l.world_pos = glam::Vec3::new(-3.0, 0.0, 2.0);
                l.text = "Outlet".to_string();
                l.color = [0.4, 0.8, 1.0, 1.0];
                l
            },
            {
                let mut l = AnnotationLabel::default();
                l.world_pos = glam::Vec3::new(0.0, 0.0, 300.0);
                l.text = "Behind camera (should be clipped)".to_string();
                l.color = [1.0, 0.0, 0.0, 1.0];
                l
            },
        ];
    }

    /// Project all annotation labels and return a title-bar summary string.
    ///
    /// This demonstrates [`world_to_screen`] being called each frame.  In an egui-based
    /// application you would instead call [`viewport_lib::draw_annotation_labels`]
    /// with an egui [`Painter`] to render the text directly on screen.
    pub(crate) fn annotation_title(&self) -> String {
        let view = self.camera.view_matrix();
        let proj = self.camera.proj_matrix();
        let [w, h] = [
            self.surface_config.width as f32,
            self.surface_config.height as f32,
        ];

        let mut parts: Vec<String> = Vec::new();
        for (i, label) in self.ann_labels.iter().enumerate() {
            match world_to_screen(label.world_pos, &view, &proj, [w, h]) {
                Some(s) => parts.push(format!("L{}=({:.0},{:.0})", i, s.x, s.y)),
                None => parts.push(format!("L{}=clipped", i)),
            }
        }
        parts.join("  ")
    }

    /// Reset the camera to a good viewing angle for the annotation demo.
    pub(crate) fn reset_annotation_camera(&mut self) {
        self.camera = Camera {
            center: glam::Vec3::new(0.0, 1.0, 0.5),
            distance: 12.0,
            orientation: glam::Quat::from_rotation_y(0.4) * glam::Quat::from_rotation_x(-0.35),
            ..Camera::default()
        };
    }
}
