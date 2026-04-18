//! World-space annotation labels that project 3D positions to 2D screen coordinates.
//!
//! # Purpose
//!
//! Callers can pin text labels to arbitrary 3D world positions (peak pressure values,
//! boundary condition names, measurement callouts) that are reprojected to screen space
//! each frame. Rendering is handled by the caller via egui (or any 2D drawing API) so
//! there is no wgpu pipeline involved.
//!
//! # Typical usage
//!
//! ```rust,ignore
//! // Build a label once (or each frame if the data changes).
//! let mut label = AnnotationLabel::default();
//! label.world_pos = glam::Vec3::new(2.0, 3.0, 0.0);
//! label.text = "Peak pressure: 101.3 kPa".to_string();
//! label.leader_end = Some(glam::Vec3::new(1.0, 1.0, 0.0));
//!
//! // Each frame, project to screen.
//! let view = camera.view_matrix();
//! let proj = camera.proj_matrix();
//! if let Some(screen_pos) = world_to_screen(label.world_pos, &view, &proj, viewport_size) {
//!     // Draw with your 2D API here.
//! }
//! ```

use crate::renderer::FrameData;

/// A text label pinned to a 3D world position.
///
/// Labels are rendered by the caller (typically via egui's painter) after projecting
/// the world position to screen space using [`world_to_screen`] or [`world_to_screen_from_frame`].
/// No wgpu pipeline changes are required.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::annotation::AnnotationLabel;
/// let mut label = AnnotationLabel::default();
/// label.world_pos = glam::Vec3::new(0.0, 5.0, 0.0);
/// label.text = "Inlet".to_string();
/// label.color = [0.4, 0.8, 1.0, 1.0]; // light blue
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AnnotationLabel {
    /// 3D world-space position the label is pinned to.
    pub world_pos: glam::Vec3,

    /// Text content to display.
    pub text: String,

    /// Optional world-space endpoint for an offset callout leader line.
    ///
    /// When `Some(end)`, callers should project `end` to screen space and draw
    /// a line segment from the leader end to the label's screen position.
    pub leader_end: Option<glam::Vec3>,

    /// RGBA colour in linear float format. Default: opaque white `[1.0, 1.0, 1.0, 1.0]`.
    pub color: [f32; 4],

    /// Font size in egui points. Default: `14.0`.
    pub font_size: f32,

    /// Whether to draw a semi-transparent background rectangle behind the text.
    /// Default: `true`.
    pub background: bool,
}

impl Default for AnnotationLabel {
    fn default() -> Self {
        Self {
            world_pos: glam::Vec3::ZERO,
            text: String::new(),
            leader_end: None,
            color: [1.0, 1.0, 1.0, 1.0],
            font_size: 14.0,
            background: true,
        }
    }
}

/// Projects a 3D world-space position to 2D screen-space coordinates.
///
/// Returns `None` if the point is behind the camera (`clip.w <= 0`) or
/// outside the view frustum (NDC outside `[-1, 1]` on X or Y).
///
/// # Arguments
///
/// * `pos` — World-space position to project.
/// * `view` — Camera view matrix (world -> view space).
/// * `proj` — Camera projection matrix (view space -> clip space).
/// * `viewport_size` — Viewport dimensions in physical pixels `[width, height]`.
///
/// # Returns
///
/// Screen-space position in physical pixels with the origin at the top-left corner,
/// or `None` if the point would not be visible on screen.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::annotation::world_to_screen;
/// # use glam::{Mat4, Vec2, Vec3};
/// let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
/// let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
/// let screen = world_to_screen(Vec3::ZERO, &view, &proj, [800.0, 600.0]);
/// assert!(screen.is_some()); // origin is in front of the camera
/// ```
pub fn world_to_screen(
    pos: glam::Vec3,
    view: &glam::Mat4,
    proj: &glam::Mat4,
    viewport_size: [f32; 2],
) -> Option<glam::Vec2> {
    // Transform to clip space: proj * view * pos (homogeneous).
    let clip = *proj * *view * pos.extend(1.0);

    // Points at or behind the camera have w <= 0 and are not visible.
    if clip.w <= 0.0 {
        return None;
    }

    // Perspective division -> NDC in [-1, 1] on each axis.
    let ndc = glam::Vec3::new(clip.x, clip.y, clip.z) / clip.w;

    // Clip any point outside the view frustum on X or Y.
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 {
        return None;
    }

    // Convert NDC to screen pixels.
    // NDC X: -1 = left edge,  +1 = right edge.
    // NDC Y: -1 = bottom, +1 = top (OpenGL convention).
    // Screen Y: 0 = top, viewport_height = bottom -> flip Y.
    let x = (ndc.x * 0.5 + 0.5) * viewport_size[0];
    let y = (1.0 - (ndc.y * 0.5 + 0.5)) * viewport_size[1];

    Some(glam::Vec2::new(x, y))
}

/// Convenience wrapper that extracts camera matrices and viewport size from a [`FrameData`].
///
/// Equivalent to calling [`world_to_screen`] with `frame.camera.render_camera.view`,
/// `frame.camera.render_camera.projection`, and `frame.camera.viewport_size`.
///
/// # Arguments
///
/// * `pos` — World-space position to project.
/// * `frame` — Current frame data containing camera matrices and viewport dimensions.
///
/// # Returns
///
/// Screen-space position in physical pixels (top-left origin), or `None` if not visible.
pub fn world_to_screen_from_frame(pos: glam::Vec3, frame: &FrameData) -> Option<glam::Vec2> {
    world_to_screen(
        pos,
        &frame.camera.render_camera.view,
        &frame.camera.render_camera.projection,
        frame.camera.viewport_size,
    )
}

/// Draws a slice of [`AnnotationLabel`]s using an egui `Painter`.
///
/// For each label:
///
/// 1. Projects `world_pos` to screen space; skips the label if not visible.
/// 2. Converts `color` to `egui::Color32`.
/// 3. Draws the text with an optional semi-transparent background.
/// 4. If `leader_end` is `Some` and also projects successfully, draws a 1 px leader line
///    from the leader endpoint to the label anchor.
///
/// Clipping is handled automatically by egui: the painter clips to its own rect, so
/// passing a tight `clip_rect` (the viewport rectangle) is sufficient.
///
/// # Arguments
///
/// * `painter` — An egui painter whose clip rect covers the viewport.
/// * `labels` — Slice of labels to draw.
/// * `view` — Camera view matrix.
/// * `proj` — Camera projection matrix.
/// * `viewport_size` — Viewport in physical pixels `[width, height]`.
/// * `_clip_rect` — Viewport rectangle (egui clips automatically; passed for explicitness).
///
/// # Feature
///
/// This function is only available when the `egui` feature of `viewport-lib` is enabled.
#[cfg(feature = "egui")]
pub fn draw_annotation_labels(
    painter: &egui::Painter,
    labels: &[AnnotationLabel],
    view: &glam::Mat4,
    proj: &glam::Mat4,
    viewport_size: [f32; 2],
    clip_rect: egui::Rect,
) {
    use egui::{Color32, FontId, Stroke, Vec2, pos2};

    // `world_to_screen` returns positions relative to the viewport top-left corner.
    // Add `clip_rect.min` to convert to window-absolute egui coordinates.
    let ox = clip_rect.min.x;
    let oy = clip_rect.min.y;

    for label in labels {
        let Some(screen) = world_to_screen(label.world_pos, view, proj, viewport_size) else {
            continue;
        };

        let [r, g, b, a] = label.color;
        let color = Color32::from_rgba_unmultiplied(
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
            (a * 255.0) as u8,
        );

        let anchor = pos2(ox + screen.x, oy + screen.y);

        // Draw leader line first (underneath the text).
        if let Some(end_world) = label.leader_end {
            if let Some(end_screen) = world_to_screen(end_world, view, proj, viewport_size) {
                let end_pos = pos2(ox + end_screen.x, oy + end_screen.y);
                painter.line_segment([end_pos, anchor], Stroke::new(1.0, color));
            }
        }

        // Draw label text.
        let font = FontId::proportional(label.font_size);

        if label.background {
            // Measure the text first so we can draw the background rectangle.
            let galley = painter.layout_no_wrap(label.text.clone(), font.clone(), color);
            let text_size = galley.size();
            let bg_rect =
                egui::Rect::from_min_size(anchor, Vec2::new(text_size.x + 4.0, text_size.y + 2.0));
            painter.rect_filled(bg_rect, 2.0, Color32::from_rgba_unmultiplied(0, 0, 0, 160));
            painter.galley(pos2(anchor.x + 2.0, anchor.y + 1.0), galley, color);
        } else {
            painter.text(anchor, egui::Align2::LEFT_TOP, &label.text, font, color);
        }
    }
}
