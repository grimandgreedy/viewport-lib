//! Showcase 9: Annotation Labels : build and render methods.
//!
//! Unlike the winit showcase (which only projects labels to title-bar coordinates),
//! the eframe version draws annotation text directly on the viewport using the egui
//! painter, demonstrating `world_to_screen` with real on-screen overlays.

use crate::App;
use crate::geometry::make_box_with_uvs;
use viewport_lib::{AnnotationLabel, Camera, Material, MeshId, ViewportRenderer, world_to_screen};

impl App {
    /// Build the scene for Showcase 9 (Annotation Labels demo).
    pub(crate) fn build_annotation_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::scene::Scene;

        self.ann_scene = Scene::new();

        let mut place_marker =
            |renderer: &mut ViewportRenderer, pos: glam::Vec3, color: [f32; 3]| {
                let mesh = make_box_with_uvs(0.3, 0.3, 0.3);
                let idx = renderer
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

        place_marker(renderer, glam::Vec3::ZERO, [1.0, 1.0, 1.0]);
        place_marker(renderer, glam::Vec3::new(2.0, 3.0, 0.0), [1.0, 0.9, 0.1]);
        place_marker(renderer, glam::Vec3::new(-3.0, 2.0, 0.0), [0.4, 0.8, 1.0]);
        place_marker(renderer, glam::Vec3::new(0.0, 300.0, 0.0), [1.0, 0.0, 0.0]);

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
                l.world_pos = glam::Vec3::new(-3.0, 2.0, 0.0);
                l.text = "Outlet".to_string();
                l.color = [0.4, 0.8, 1.0, 1.0];
                l
            },
            {
                let mut l = AnnotationLabel::default();
                l.world_pos = glam::Vec3::new(0.0, 300.0, 0.0);
                l.text = "Behind camera (clipped)".to_string();
                l.color = [1.0, 0.0, 0.0, 1.0];
                l
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

    /// Draw annotation labels on top of the 3D viewport using the egui painter.
    ///
    /// Labels are projected via `world_to_screen` and rendered as text + circles.
    pub(crate) fn draw_annotation_labels(&self, ui: &eframe::egui::Ui, rect: eframe::egui::Rect) {
        let painter = ui.painter_at(rect);
        let view = self.camera.view_matrix();
        let proj = self.camera.proj_matrix();
        let vp_size = [rect.width(), rect.height()];

        for label in &self.ann_labels {
            let Some(screen) = world_to_screen(label.world_pos, &view, &proj, vp_size) else {
                continue;
            };
            let screen_pos = eframe::egui::pos2(rect.left() + screen.x, rect.top() + screen.y);
            let color = eframe::egui::Color32::from_rgba_unmultiplied(
                (label.color[0] * 255.0) as u8,
                (label.color[1] * 255.0) as u8,
                (label.color[2] * 255.0) as u8,
                (label.color[3] * 255.0) as u8,
            );

            // Draw leader line if present.
            if let Some(leader_end) = label.leader_end {
                if let Some(end_screen) = world_to_screen(leader_end, &view, &proj, vp_size) {
                    let end_pos =
                        eframe::egui::pos2(rect.left() + end_screen.x, rect.top() + end_screen.y);
                    painter.line_segment(
                        [screen_pos, end_pos],
                        eframe::egui::Stroke::new(1.0, color.linear_multiply(0.6)),
                    );
                }
            }

            // Anchor dot.
            painter.circle_filled(screen_pos, 3.0, color);

            // Label text with a subtle background.
            let text_pos = screen_pos + eframe::egui::vec2(6.0, -8.0);
            let galley = painter.layout_no_wrap(
                label.text.clone(),
                eframe::egui::FontId::proportional(13.0),
                color,
            );
            let bg_rect = eframe::egui::Rect::from_min_size(
                text_pos - eframe::egui::vec2(2.0, 2.0),
                galley.size() + eframe::egui::vec2(4.0, 4.0),
            );
            painter.rect_filled(bg_rect, 2.0, eframe::egui::Color32::from_black_alpha(140));
            painter.galley(text_pos, galley, color);
        }
    }
}
