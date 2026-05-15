//! Line probe widget: two draggable endpoint handles connected by a line segment.

use crate::interaction::clip_plane::ray_plane_intersection;
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
    /// RGBA line and handle color.
    pub color: [f32; 4],
    /// Line width in pixels.
    pub line_width: f32,
    /// RGBA color for the drag handles. When set (non-zero alpha), overrides the default LUT coloring.
    pub handle_color: [f32; 4],

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
            color: [1.0, 0.6, 0.1, 1.0],
            line_width: 2.0,
            handle_color: [0.0; 4],
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
            default_color: self.color,
            line_width: self.line_width,
            id,
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` containing sphere handles at both endpoints.
    ///
    /// Handle size is constant in screen space (approximately 10 px radius).
    /// `id_base` is the pick ID for the start handle; the end handle uses `id_base + 1`.
    ///
    /// Color is driven by the colormap (viridis by default). The scalar for each
    /// handle is `0.0` when idle and `1.0` when hovered or active, so the two
    /// states map to distinct colormap colors.
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

        GlyphItem {
            positions: vec![self.start.to_array(), self.end.to_array()],
            vectors: vec![[r0, 0.0, 0.0], [r1, 0.0, 0.0]],
            scale: 1.0,
            scale_by_magnitude: true,
            scalars: vec![s0, s1],
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
