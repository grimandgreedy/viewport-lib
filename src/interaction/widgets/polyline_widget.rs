//! Polyline widget: N draggable waypoints connected by straight line segments.

use crate::interaction::clip_plane::ray_plane_intersection;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem};

use super::{WidgetContext, WidgetResult, ctx_ray, handle_world_radius, ray_point_dist};

/// An interactive polyline widget with N draggable control points.
///
/// Unlike [`crate::SplineWidget`], this uses straight line segments between points (no
/// interpolation). Control points can be added and removed programmatically, and via
/// double-click when `ctx.double_clicked` is set by the host.
///
/// # Usage
///
/// ```rust,ignore
/// let mut pw = PolylineWidget::new(vec![
///     glam::Vec3::new(-2.0, 0.0, 0.0),
///     glam::Vec3::new( 0.0, 1.0, 0.0),
///     glam::Vec3::new( 2.0, 0.0, 0.0),
/// ]);
///
/// // Each frame:
/// let result = pw.update(&ctx);
/// fd.scene.polylines.push(pw.polyline_item(PL_ID));
/// fd.scene.glyphs.push(pw.handle_glyphs(HANDLE_ID, &ctx));
/// ```
pub struct PolylineWidget {
    /// Control point positions in world space.
    pub points: Vec<glam::Vec3>,
    /// RGBA color for the line segments.
    pub color: [f32; 4],
    /// Line width in pixels.
    pub line_width: f32,
    /// RGBA color for the drag handles.
    pub handle_color: [f32; 4],
    /// Index of the currently hovered control point.
    pub hovered_point: Option<usize>,
    /// Index of the point actively being dragged.
    pub active_point: Option<usize>,

    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
}

impl PolylineWidget {
    /// Create a new polyline widget. Must have at least two points; extras are ignored
    /// if fewer are provided by adding a default second point.
    pub fn new(mut points: Vec<glam::Vec3>) -> Self {
        if points.is_empty() {
            points.push(glam::Vec3::ZERO);
        }
        if points.len() < 2 {
            points.push(points[0] + glam::Vec3::X);
        }
        Self {
            points,
            color: [0.9, 0.5, 0.1, 1.0],
            line_width: 2.0,
            handle_color: [0.0; 4],
            hovered_point: None,
            active_point: None,
            drag_plane_normal: glam::Vec3::Y,
            drag_plane_d: 0.0,
        }
    }

    /// True while a control point drag is in progress.
    pub fn is_active(&self) -> bool {
        self.active_point.is_some()
    }

    /// Append a point at the end of the polyline.
    pub fn add_point(&mut self, pos: glam::Vec3) {
        self.points.push(pos);
    }

    /// Remove the point at `index`. No-op if the polyline would drop below two points.
    pub fn remove_point(&mut self, index: usize) {
        if self.points.len() > 2 && index < self.points.len() {
            self.points.remove(index);
            // Fix up hover/active indices to avoid dangling references.
            if self.hovered_point == Some(index) {
                self.hovered_point = None;
            } else if let Some(h) = self.hovered_point {
                if h > index {
                    self.hovered_point = Some(h - 1);
                }
            }
            if self.active_point == Some(index) {
                self.active_point = None;
            } else if let Some(a) = self.active_point {
                if a > index {
                    self.active_point = Some(a - 1);
                }
            }
        }
    }

    /// Process input for this frame. Returns `Updated` if state changed.
    ///
    /// Double-click behavior (requires `ctx.double_clicked`):
    /// - On a hovered control point: removes that point (minimum 2 enforced).
    /// - On a line segment: inserts a new point at the projected position.
    pub fn update(&mut self, ctx: &WidgetContext) -> WidgetResult {
        let (ro, rd) = ctx_ray(ctx);
        let ref_pos = self.points.first().copied().unwrap_or(glam::Vec3::ZERO);
        let hit_radius = handle_world_radius(ref_pos, &ctx.camera, ctx.viewport_size.y, 12.0);

        // --- Hover detection ---
        if !ctx.dragging {
            self.hovered_point = None;
            let mut best_dist = hit_radius;
            for (i, &pt) in self.points.iter().enumerate() {
                let d = ray_point_dist(ro, rd, pt);
                if d < best_dist {
                    best_dist = d;
                    self.hovered_point = Some(i);
                }
            }
        }

        // --- Double-click: add or remove points ---
        if ctx.double_clicked {
            if let Some(idx) = self.hovered_point {
                // Double-click on handle: remove that point.
                if self.points.len() > 2 {
                    self.points.remove(idx);
                    self.hovered_point = None;
                    return WidgetResult::Updated;
                }
            } else {
                // Double-click on a segment: find the closest segment and insert a point.
                let seg_threshold = hit_radius * 2.0;
                if let Some((seg_idx, insert_pos)) = self.closest_segment(ro, rd, seg_threshold) {
                    self.points.insert(seg_idx + 1, insert_pos);
                    self.hovered_point = Some(seg_idx + 1);
                    return WidgetResult::Updated;
                }
            }
        }

        // --- Drag start ---
        if ctx.drag_started {
            if let Some(idx) = self.hovered_point {
                self.active_point = Some(idx);
                self.drag_plane_normal = -glam::Vec3::from(ctx.camera.forward);
                self.drag_plane_d = -self.drag_plane_normal.dot(self.points[idx]);
            }
        }

        // --- Drag in progress ---
        if ctx.dragging {
            if let Some(idx) = self.active_point {
                if let Some(hit) =
                    ray_plane_intersection(ro, rd, self.drag_plane_normal, self.drag_plane_d)
                {
                    if (hit - self.points[idx]).length_squared() > 1e-10 {
                        self.points[idx] = hit;
                        return WidgetResult::Updated;
                    }
                }
            }
        }

        if ctx.released {
            self.active_point = None;
        }

        WidgetResult::None
    }

    /// Build a `PolylineItem` through all control points.
    pub fn polyline_item(&self, id: u64) -> PolylineItem {
        let n = self.points.len() as u32;
        PolylineItem {
            positions: self.points.iter().map(|p| p.to_array()).collect(),
            strip_lengths: if n > 0 { vec![n] } else { vec![] },
            default_color: self.color,
            line_width: self.line_width,
            id,
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` with sphere handles for each control point.
    ///
    /// Hovered and active handles appear brighter (scalar = 1.0 vs 0.2).
    /// `id_base` is the pick ID for the first point; each subsequent point uses `id_base + i`.
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let mut positions = Vec::with_capacity(self.points.len());
        let mut vectors = Vec::with_capacity(self.points.len());
        let mut scalars = Vec::with_capacity(self.points.len());

        for (i, pt) in self.points.iter().enumerate() {
            let r = handle_world_radius(*pt, &ctx.camera, ctx.viewport_size.y, 9.0);
            let s = if self.hovered_point == Some(i) || self.active_point == Some(i) {
                1.0_f32
            } else {
                0.2
            };
            positions.push(pt.to_array());
            vectors.push([r, 0.0, 0.0]);
            scalars.push(s);
        }
        GlyphItem {
            positions,
            vectors,
            scale: 1.0,
            scale_by_magnitude: true,
            scalars,
            scalar_range: Some((0.0, 1.0)),
            glyph_type: GlyphType::Sphere,
            id: id_base,
            default_color: self.handle_color,
            use_default_color: self.handle_color[3] > 0.0,
            ..GlyphItem::default()
        }
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Find the closest segment to the ray, within `threshold` world units.
    /// Returns `(segment_index, insertion_point_on_segment)` or `None`.
    fn closest_segment(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        threshold: f32,
    ) -> Option<(usize, glam::Vec3)> {
        let mut best: Option<(f32, usize, glam::Vec3)> = None;

        for i in 0..self.points.len().saturating_sub(1) {
            let a = self.points[i];
            let b = self.points[i + 1];
            let (pt, dist) = closest_point_on_segment_to_ray(ray_origin, ray_dir, a, b);
            if dist < threshold {
                if best.is_none() || dist < best.unwrap().0 {
                    best = Some((dist, i, pt));
                }
            }
        }

        best.map(|(_, i, pt)| (i, pt))
    }
}

/// Returns the closest point on segment [a, b] to the ray, plus the distance.
fn closest_point_on_segment_to_ray(
    ray_o: glam::Vec3,
    ray_d: glam::Vec3,
    seg_a: glam::Vec3,
    seg_b: glam::Vec3,
) -> (glam::Vec3, f32) {
    let seg_d = seg_b - seg_a;
    let r = ray_o - seg_a;
    let b = ray_d.dot(seg_d);
    let c = seg_d.dot(seg_d);
    let d = ray_d.dot(r);
    let e = seg_d.dot(r);

    let denom = c - b * b;
    let t_seg = if denom.abs() > 1e-7 {
        // a * c - b * b  where a = ray_d.dot(ray_d) = 1.0
        ((e - b * d) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let seg_pt = seg_a + seg_d * t_seg;
    let dist = ray_point_dist(ray_o, ray_d, seg_pt);
    (seg_pt, dist)
}
