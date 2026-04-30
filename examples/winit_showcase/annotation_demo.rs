//! Annotation label showcase.
//!
//! This mode exercises [`LabelItem`] and `OverlayFrame` with a few fixed anchors.
//! Labels render natively inside the viewport : no egui painter is needed.

use crate::AppState;
use viewport_lib::{Camera, LabelItem, Material};

/// Project a world-space position to screen space.
///
/// Returns `None` if the point is behind the camera or outside the view frustum.
fn project(pos: glam::Vec3, view: &glam::Mat4, proj: &glam::Mat4, vp: [f32; 2]) -> Option<glam::Vec2> {
    let clip = *proj * *view * pos.extend(1.0);
    if clip.w <= 0.0 { return None; }
    let ndc = glam::Vec3::new(clip.x, clip.y, clip.z) / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 { return None; }
    let x = (ndc.x * 0.5 + 0.5) * vp[0];
    let y = (1.0 - (ndc.y * 0.5 + 0.5)) * vp[1];
    Some(glam::Vec2::new(x, y))
}

impl AppState {
    /// Build the scene for Showcase 9 (Annotation Labels demo).
    ///
    /// Places a small coloured marker box at each label's `world_anchor` so the
    /// anchor is visible in the 3D view.  Labels are stored in `ann_labels` and
    /// rendered natively via `OverlayFrame` each frame.
    pub(crate) fn build_annotation_scene(&mut self) {
        use viewport_lib::scene::Scene;

        self.ann_scene = Scene::new();

        // Helper: upload a small box at a given position with the given colour.
        let mut place_marker = |pos: glam::Vec3, color: [f32; 3]| {
            let mesh = crate::geometry::make_box_with_uvs(0.3, 0.3, 0.3);
            let id = self
                .renderer
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

        // Label 0 - Origin marker (white).
        place_marker(glam::Vec3::ZERO, [1.0, 1.0, 1.0]);

        // Label 1 - Peak pressure with leader line (yellow).
        place_marker(glam::Vec3::new(2.0, 3.0, 0.0), [1.0, 0.9, 0.1]);

        // Label 2 - Outlet (light blue).
        place_marker(glam::Vec3::new(-3.0, 0.0, 2.0), [0.4, 0.8, 1.0]);

        // Label 3 - Placed behind the default camera: should be culled by the renderer.
        place_marker(glam::Vec3::new(0.0, 0.0, 300.0), [1.0, 0.0, 0.0]);

        // Build the matching LabelItem vec for OverlayFrame.
        self.ann_labels = vec![
            LabelItem {
                world_anchor: Some([0.0, 0.0, 0.0]),
                text: "Origin (0,0,0)".to_string(),
                color: [1.0, 1.0, 1.0, 1.0],
                background: true,
                ..LabelItem::default()
            },
            LabelItem {
                world_anchor: Some([2.0, 3.0, 0.0]),
                text: "Peak Pressure: 101.3 kPa".to_string(),
                color: [1.0, 0.9, 0.1, 1.0],
                leader_line: true,
                background: true,
                ..LabelItem::default()
            },
            LabelItem {
                world_anchor: Some([-3.0, 0.0, 2.0]),
                text: "Outlet".to_string(),
                color: [0.4, 0.8, 1.0, 1.0],
                background: true,
                ..LabelItem::default()
            },
            LabelItem {
                world_anchor: Some([0.0, 0.0, 300.0]),
                text: "Behind camera (should be clipped)".to_string(),
                color: [1.0, 0.0, 0.0, 1.0],
                background: true,
                ..LabelItem::default()
            },
        ];
    }

    /// Project all annotation labels and return a title-bar summary string.
    ///
    /// Demonstrates projecting world anchors to screen space for diagnostics.
    /// The actual label rendering is handled by `OverlayFrame` inside the renderer.
    pub(crate) fn annotation_title(&self) -> String {
        let view = self.camera.view_matrix();
        let proj = self.camera.proj_matrix();
        let [w, h] = [
            self.surface_config.width as f32,
            self.surface_config.height as f32,
        ];

        let mut parts: Vec<String> = Vec::new();
        for (i, label) in self.ann_labels.iter().enumerate() {
            if let Some(wa) = label.world_anchor {
                match project(glam::Vec3::from(wa), &view, &proj, [w, h]) {
                    Some(s) => parts.push(format!("L{}=({:.0},{:.0})", i, s.x, s.y)),
                    None => parts.push(format!("L{}=clipped", i)),
                }
            } else {
                parts.push(format!("L{}=screen-anchored", i));
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
