//! Line probe widget: two draggable endpoint handles connected by a line segment.

use crate::geometry::intersect::ray_plane_intersection;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem};

use super::{WidgetContext, WidgetResult, ctx_ray, handle_world_radius, ray_point_dist};

/// A two-endpoint line handle rendered in the viewport.
///
/// Drag either sphere handle to reposition the probe path. Read `start` and `end`
/// each frame to get the current endpoint positions.
///
/// # Usage
///
/// ```rust,ignore
/// // Setup (once):
/// let mut probe = LineProbeWidget::new(
///     glam::Vec3::new(-2.0, 0.0, 0.0),
///     glam::Vec3::new( 2.0, 0.0, 0.0),
/// );
///
/// // Each frame:
/// let ctx = WidgetContext { camera, viewport_size, cursor_viewport,
///                           drag_started, dragging, released };
/// probe.update(&ctx);
///
/// fd.scene.polylines.push(probe.polyline_item(LINE_ID));
/// fd.scene.glyphs.push(probe.handle_glyphs(HANDLE_ID_BASE, &ctx));
///
/// // Suppress orbit while dragging:
/// if probe.is_active() { orbit.resolve(); } else { orbit.apply_to_camera(&mut camera); }
/// ```
pub struct LineProbeWidget {
    /// World-space position of the first endpoint.
    pub start: glam::Vec3,
    /// World-space position of the second endpoint.
    pub end: glam::Vec3,
    /// RGBA line and handle colour.
    pub colour: [f32; 4],
    /// Line width in pixels.
    pub line_width: f32,
    /// RGBA colour for the drag handles. When set (non-zero alpha), overrides the default LUT colouring.
    pub handle_colour: [f32; 4],

    hovered_endpoint: Option<usize>,
    active_endpoint: Option<usize>,
    // Camera-facing drag plane captured at drag start.
    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
}

impl LineProbeWidget {
    /// Create a new probe between two world-space positions.
    pub fn new(start: glam::Vec3, end: glam::Vec3) -> Self {
        Self {
            start,
            end,
            colour: [1.0, 0.6, 0.1, 1.0],
            line_width: 2.0,
            handle_colour: [0.0; 4],
            hovered_endpoint: None,
            active_endpoint: None,
            drag_plane_normal: glam::Vec3::Z,
            drag_plane_d: 0.0,
        }
    }

    /// Index of the currently hovered endpoint (0 = start, 1 = end).
    pub fn hovered_endpoint(&self) -> Option<usize> {
        self.hovered_endpoint
    }

    /// True while a drag session is in progress on either endpoint.
    pub fn is_active(&self) -> bool {
        self.active_endpoint.is_some()
    }

    /// Process input for this frame. Returns `Updated` if either endpoint moved.
    ///
    /// Call once per frame before building render items.
    pub fn update(&mut self, ctx: &WidgetContext) -> WidgetResult {
        let (ro, rd) = ctx_ray(ctx);
        let mut updated = false;

        // Hover (only when not dragging, to avoid flicker during drag).
        if self.active_endpoint.is_none() {
            let hit = self.hit_test(ro, rd, ctx);
            // On the drag_started frame the cursor can be right at the edge and the
            // hit test may miss by a hair. Keep the previous hover so the drag still
            // registers if the handle was highlighted on the frame before the click.
            if hit.is_some() || !ctx.drag_started {
                self.hovered_endpoint = hit;
            }
        }

        if ctx.drag_started {
            if let Some(ep) = self.hovered_endpoint {
                let ep_world = self.endpoint_pos(ep);
                let fwd = glam::Vec3::from(ctx.camera.forward);
                let n = -fwd;
                self.drag_plane_normal = n;
                self.drag_plane_d = -n.dot(ep_world);
                self.active_endpoint = Some(ep);
            }
        }

        if let Some(ep) = self.active_endpoint {
            if ctx.released || (!ctx.dragging && !ctx.drag_started) {
                self.active_endpoint = None;
                self.hovered_endpoint = None;
            } else if let Some(hit) =
                ray_plane_intersection(ro, rd, self.drag_plane_normal, self.drag_plane_d)
            {
                let prev = self.endpoint_pos(ep);
                if (hit - prev).length_squared() > 1e-10 {
                    self.set_endpoint(ep, hit);
                    updated = true;
                }
            }
        }

        if updated {
            WidgetResult::Updated
        } else {
            WidgetResult::None
        }
    }

    /// Build the `PolylineItem` for the line segment between the two endpoints.
    ///
    /// `id` is used as the pick ID for the line body (0 = not pickable).
    pub fn polyline_item(&self, id: u64) -> PolylineItem {
        PolylineItem {
            positions: vec![self.start.to_array(), self.end.to_array()],
            strip_lengths: vec![2],
            default_colour: self.colour,
            line_width: self.line_width,

            settings: {
                let mut s = crate::scene::material::ItemSettings::default();
                s.pick_id = crate::renderer::PickId(id);
                s
            },
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` containing sphere handles at both endpoints.
    ///
    /// Handle size is constant in screen space (approximately 10 px radius).
    /// `id_base` is the pick ID for the start handle; the end handle uses `id_base + 1`.
    ///
    /// Colour is driven by the colourmap (viridis by default). The scalar for each
    /// handle is `0.0` when idle and `1.0` when hovered or active, so the two
    /// states map to distinct colourmap colours.
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let r0 = handle_world_radius(self.start, &ctx.camera, ctx.viewport_size.y, 10.0);
        let r1 = handle_world_radius(self.end, &ctx.camera, ctx.viewport_size.y, 10.0);

        let s0 = if self.hovered_endpoint == Some(0) || self.active_endpoint == Some(0) {
            1.0_f32
        } else {
            0.0
        };
        let s1 = if self.hovered_endpoint == Some(1) || self.active_endpoint == Some(1) {
            1.0_f32
        } else {
            0.0
        };

        let mut g = GlyphItem::default();
        g.positions = vec![self.start.to_array(), self.end.to_array()];
        g.vectors = vec![[r0, 0.0, 0.0], [r1, 0.0, 0.0]];
        g.scalars = vec![s0, s1];
        g.scalar_range = Some((0.0, 1.0));
        g.glyph_type = GlyphType::Sphere;
        g.settings = {
            let mut s = crate::scene::material::ItemSettings::default();
            s.pick_id = crate::renderer::PickId(id_base);
            s
        };
        g.default_colour = self.handle_colour;
        g.use_default_colour = self.handle_colour[3] > 0.0;
        g
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn endpoint_pos(&self, ep: usize) -> glam::Vec3 {
        if ep == 0 { self.start } else { self.end }
    }

    fn set_endpoint(&mut self, ep: usize, pos: glam::Vec3) {
        if ep == 0 {
            self.start = pos;
        } else {
            self.end = pos;
        }
    }

    fn hit_test(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        ctx: &WidgetContext,
    ) -> Option<usize> {
        let r0 = handle_world_radius(self.start, &ctx.camera, ctx.viewport_size.y, 10.0);
        let r1 = handle_world_radius(self.end, &ctx.camera, ctx.viewport_size.y, 10.0);

        let d0 = ray_point_dist(ray_origin, ray_dir, self.start);
        let d1 = ray_point_dist(ray_origin, ray_dir, self.end);

        let h0 = d0 < r0;
        let h1 = d1 < r1;

        match (h0, h1) {
            (true, true) => {
                // Prefer the endpoint closer along the ray.
                let t0 = (self.start - ray_origin).dot(ray_dir);
                let t1 = (self.end - ray_origin).dot(ray_dir);
                Some(if t0 <= t1 { 0 } else { 1 })
            }
            (true, false) => Some(0),
            (false, true) => Some(1),
            (false, false) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::widgets::test_support::{CENTRE, ctx_at, point_on_cursor_ray};
    use glam::{Vec2, Vec3};

    #[test]
    fn new_sets_endpoints() {
        let w = LineProbeWidget::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(w.start, Vec3::new(-2.0, 0.0, 0.0));
        assert_eq!(w.end, Vec3::new(2.0, 0.0, 0.0));
        assert!(!w.is_active());
        assert_eq!(w.hovered_endpoint(), None);
    }

    #[test]
    fn polyline_and_handle_items_track_endpoints() {
        let ctx = ctx_at(CENTRE);
        let w = LineProbeWidget::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 0.0));
        let line = w.polyline_item(7);
        assert_eq!(line.positions, vec![[-1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]);
        let glyphs = w.handle_glyphs(9, &ctx);
        assert_eq!(glyphs.positions.len(), 2, "one glyph per endpoint");
    }

    #[test]
    fn endpoint_under_cursor_hovers() {
        let ctx = ctx_at(CENTRE);
        // start sits on the cursor ray; end well off to the side.
        let start = point_on_cursor_ray(&ctx, 10.0);
        let end = start + Vec3::new(5.0, 0.0, 0.0);
        let mut w = LineProbeWidget::new(start, end);
        let r = w.update(&ctx);
        assert_eq!(
            w.hovered_endpoint(),
            Some(0),
            "start should hover under cursor"
        );
        assert_eq!(r, WidgetResult::None, "hover alone does not change state");
    }

    #[test]
    fn dragging_moves_the_active_endpoint_then_release_ends_it() {
        let ctx = ctx_at(CENTRE);
        let start = point_on_cursor_ray(&ctx, 10.0);
        let end = start + Vec3::new(5.0, 0.0, 0.0);
        let mut w = LineProbeWidget::new(start, end);

        // Hover, then begin the drag on the same cursor.
        w.update(&ctx);
        let mut begin = ctx.clone();
        begin.drag_started = true;
        w.update(&begin);
        assert!(
            w.is_active(),
            "drag should have started on the hovered endpoint"
        );

        // Move the cursor while dragging: the endpoint follows onto the drag plane.
        let mut drag = ctx_at(CENTRE + Vec2::new(60.0, 0.0));
        drag.dragging = true;
        let r = w.update(&drag);
        assert_eq!(r, WidgetResult::Updated);
        assert_ne!(w.start, start, "start endpoint should have moved");
        assert_eq!(w.end, end, "the other endpoint stays put");

        // Release ends the drag.
        let mut release = drag.clone();
        release.released = true;
        w.update(&release);
        assert!(!w.is_active());
    }
}
