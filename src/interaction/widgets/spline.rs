//! Spline widget: N draggable control points connected by a Catmull-Rom spline.

use super::{WidgetContext, WidgetResult, ctx_ray, handle_world_radius, ray_point_dist};
use crate::interaction::clip_plane::ray_plane_intersection;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem};

/// An interactive spline widget with N draggable Catmull-Rom control points.
///
/// Each frame call `update()` to advance state, then push `polyline_item()` into
/// `fd.scene.polylines` and `handle_glyphs()` into `fd.scene.glyphs`.
pub struct SplineWidget {
    /// Control point positions.
    pub points: Vec<glam::Vec3>,
    /// Color of the spline curve.
    pub color: [f32; 4],
    /// Width of the spline curve in pixels.
    pub line_width: f32,
    /// Color of the control point handles.
    pub handle_color: [f32; 4],
    /// Number of samples between each pair of adjacent control points.
    pub resolution: u32,
    hovered_point: Option<usize>,
    active_point: Option<usize>,
    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
}

impl SplineWidget {
    /// Create a new spline widget with the given control points.
    pub fn new(points: Vec<glam::Vec3>) -> Self {
        Self {
            points,
            color: [0.4, 0.8, 1.0, 1.0],
            line_width: 2.0,
            handle_color: [1.0, 0.8, 0.2, 1.0],
            resolution: 16,
            hovered_point: None,
            active_point: None,
            drag_plane_normal: glam::Vec3::Y,
            drag_plane_d: 0.0,
        }
    }

    /// Returns the index of the currently hovered control point, if any.
    pub fn hovered_point(&self) -> Option<usize> {
        self.hovered_point
    }

    /// Returns true when a control point drag is in progress.
    pub fn is_active(&self) -> bool {
        self.active_point.is_some()
    }

    /// Advance widget state from cursor input. Call once per frame before pushing render items.
    pub fn update(&mut self, ctx: &WidgetContext) -> WidgetResult {
        let (ray_origin, ray_dir) = ctx_ray(ctx);
        // Use the first control point (or origin) to compute a reference world radius.
        let ref_pos = self.points.first().copied().unwrap_or(glam::Vec3::ZERO);
        let hit_radius = handle_world_radius(ref_pos, &ctx.camera, ctx.viewport_size.y, 12.0);

        if !ctx.dragging {
            self.hovered_point = None;
            let mut best_dist = hit_radius;
            for (i, &pt) in self.points.iter().enumerate() {
                let d = ray_point_dist(ray_origin, ray_dir, pt);
                if d < best_dist {
                    best_dist = d;
                    self.hovered_point = Some(i);
                }
            }
        }

        if ctx.drag_started {
            if let Some(idx) = self.hovered_point {
                self.active_point = Some(idx);
                self.drag_plane_normal = -glam::Vec3::from(ctx.camera.forward);
                let pt = self.points[idx];
                self.drag_plane_d = -self.drag_plane_normal.dot(pt);
            }
        }

        if ctx.dragging {
            if let Some(idx) = self.active_point {
                if let Some(hit) = ray_plane_intersection(
                    ray_origin,
                    ray_dir,
                    self.drag_plane_normal,
                    self.drag_plane_d,
                ) {
                    self.points[idx] = hit;
                    return WidgetResult::Updated;
                }
            }
        }

        if ctx.released {
            self.active_point = None;
        }

        WidgetResult::None
    }

    /// Build a `PolylineItem` with the sampled Catmull-Rom spline.
    pub fn polyline_item(&self, id: u64) -> PolylineItem {
        let sampled = self.sampled_positions();
        let n = sampled.len() as u32;
        PolylineItem {
            positions: sampled,
            strip_lengths: if n > 0 { vec![n] } else { vec![] },
            default_color: self.color,
            line_width: self.line_width,
            id,
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` with sphere handles for each control point.
    ///
    /// Hovered and active handles are brightened via `use_default_color` and `default_color`.
    /// For per-handle highlight, call with a separate GlyphItem for the active handle.
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let ref_pos = self.points.first().copied().unwrap_or(glam::Vec3::ZERO);
        let radius = handle_world_radius(ref_pos, &ctx.camera, ctx.viewport_size.y, 8.0);
        GlyphItem {
            positions: self.points.iter().map(|p| p.to_array()).collect(),
            glyph_type: GlyphType::Sphere,
            scale: radius,
            use_default_color: true,
            default_color: self.handle_color,
            id: id_base,
            ..GlyphItem::default()
        }
    }

    /// Evaluate the Catmull-Rom spline and return sampled world-space positions.
    pub fn sampled_positions(&self) -> Vec<[f32; 3]> {
        let n = self.points.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.points[0].to_array()];
        }
        let res = self.resolution.max(1) as usize;
        let mut out: Vec<[f32; 3]> = Vec::with_capacity((n - 1) * res + 1);
        for seg in 0..(n - 1) {
            let p0 = self.points[if seg > 0 { seg - 1 } else { seg }];
            let p1 = self.points[seg];
            let p2 = self.points[seg + 1];
            let p3 = self.points[if seg + 2 < n { seg + 2 } else { seg + 1 }];
            for s in 0..res {
                let t = s as f32 / res as f32;
                out.push(catmull_rom(p0, p1, p2, p3, t).to_array());
            }
        }
        out.push(self.points[n - 1].to_array());
        out
    }
}

fn catmull_rom(
    p0: glam::Vec3,
    p1: glam::Vec3,
    p2: glam::Vec3,
    p3: glam::Vec3,
    t: f32,
) -> glam::Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + p2) * t
        + 2.0 * p1)
}
