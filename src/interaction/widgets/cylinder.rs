//! Cylinder widget: two endpoint handles controlling the axis, plus a radius handle.

use crate::interaction::clip_plane::ray_plane_intersection;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem, PickId};
use parry3d::math::{Pose, Vector};
use parry3d::query::{Ray, RayCast};

use super::{
    WidgetContext, WidgetResult, any_perpendicular, any_perpendicular_pair, ctx_ray,
    handle_world_radius, ray_point_dist,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CylinderHandle {
    Start,
    End,
    Radius,
}

/// An interactive cylinder widget: two endpoint handles define the axis, a third
/// handle on the rim controls the radius.
///
/// # Usage
///
/// ```rust,ignore
/// let mut cyl = CylinderWidget::new(
///     glam::Vec3::new(0.0, -2.0, 0.0),
///     glam::Vec3::new(0.0,  2.0, 0.0),
///     1.0,
/// );
///
/// // Each frame:
/// cyl.update(&ctx);
/// fd.scene.polylines.push(cyl.wireframe_item(CYL_ID));
/// fd.scene.glyphs.push(cyl.handle_glyphs(HANDLE_ID, &ctx));
/// ```
pub struct CylinderWidget {
    /// World-space position of the first endpoint (bottom cap center).
    pub start: glam::Vec3,
    /// World-space position of the second endpoint (top cap center).
    pub end: glam::Vec3,
    /// Cylinder radius in world units.
    pub radius: f32,
    /// RGBA colour for the wireframe outline.
    pub colour: [f32; 4],
    /// RGBA colour for the drag handles. Non-zero alpha overrides LUT colouring.
    pub handle_colour: [f32; 4],

    hovered_handle: Option<CylinderHandle>,
    active_handle: Option<CylinderHandle>,
    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
    drag_anchor: glam::Vec3,
}

impl CylinderWidget {
    /// Create a new cylinder widget.
    pub fn new(start: glam::Vec3, end: glam::Vec3, radius: f32) -> Self {
        Self {
            start,
            end,
            radius: radius.max(0.01),
            colour: [0.4, 0.9, 0.5, 1.0],
            handle_colour: [0.0; 4],
            hovered_handle: None,
            active_handle: None,
            drag_plane_normal: glam::Vec3::Z,
            drag_plane_d: 0.0,
            drag_anchor: glam::Vec3::ZERO,
        }
    }

    /// True while a drag is in progress on any handle.
    pub fn is_active(&self) -> bool {
        self.active_handle.is_some()
    }

    /// The axis direction (from start to end).
    pub fn axis(&self) -> glam::Vec3 {
        self.end - self.start
    }

    /// Process input for this frame. Returns `Updated` if state changed.
    pub fn update(&mut self, ctx: &WidgetContext) -> WidgetResult {
        let (ro, rd) = ctx_ray(ctx);
        let mut updated = false;

        if self.active_handle.is_none() {
            let hit = self.hit_test(ro, rd, ctx);
            if hit.is_some() || !ctx.drag_started {
                self.hovered_handle = hit;
            }
        }

        if ctx.drag_started {
            if let Some(handle) = self.hovered_handle {
                let anchor = self.handle_pos(handle);
                let n = -glam::Vec3::from(ctx.camera.forward);
                self.drag_plane_normal = n;
                self.drag_plane_d = -n.dot(anchor);
                self.drag_anchor = anchor;
                self.active_handle = Some(handle);
            }
        }

        if let Some(handle) = self.active_handle {
            if ctx.released || (!ctx.dragging && !ctx.drag_started) {
                self.active_handle = None;
                self.hovered_handle = None;
            } else if let Some(hit) =
                ray_plane_intersection(ro, rd, self.drag_plane_normal, self.drag_plane_d)
            {
                match handle {
                    CylinderHandle::Start => {
                        let delta = hit - self.drag_anchor;
                        if delta.length_squared() > 1e-10 {
                            self.start += delta;
                            self.drag_anchor = hit;
                            updated = true;
                        }
                    }
                    CylinderHandle::End => {
                        let delta = hit - self.drag_anchor;
                        if delta.length_squared() > 1e-10 {
                            self.end += delta;
                            self.drag_anchor = hit;
                            updated = true;
                        }
                    }
                    CylinderHandle::Radius => {
                        // Project hit onto the plane perpendicular to the axis at the midpoint,
                        // then measure distance from midpoint as the new radius.
                        let mid = (self.start + self.end) * 0.5;
                        let new_r = (hit - mid).length().max(0.01);
                        if (new_r - self.radius).abs() > 1e-5 {
                            self.radius = new_r;
                            updated = true;
                        }
                    }
                }
            }
        }

        if updated {
            WidgetResult::Updated
        } else {
            WidgetResult::None
        }
    }

    /// Build a `PolylineItem` with two end-cap circles and four longitudinal lines.
    ///
    /// The end caps have 32 segments each. Four evenly-spaced lines connect the caps.
    pub fn wireframe_item(&self, id: u64) -> PolylineItem {
        const STEPS: usize = 32;

        let axis = self.axis();
        let axis_len = axis.length();
        let axis_dir = if axis_len > 1e-6 {
            axis / axis_len
        } else {
            glam::Vec3::Z
        };
        let (u, v) = any_perpendicular_pair(axis_dir);
        let r = self.radius;

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut strip_lengths: Vec<u32> = Vec::new();

        // Bottom cap circle
        for i in 0..=STEPS {
            let a = i as f32 * std::f32::consts::TAU / STEPS as f32;
            let (s, c) = a.sin_cos();
            positions.push((self.start + u * (c * r) + v * (s * r)).to_array());
        }
        strip_lengths.push((STEPS + 1) as u32);

        // Top cap circle
        for i in 0..=STEPS {
            let a = i as f32 * std::f32::consts::TAU / STEPS as f32;
            let (s, c) = a.sin_cos();
            positions.push((self.end + u * (c * r) + v * (s * r)).to_array());
        }
        strip_lengths.push((STEPS + 1) as u32);

        // Four longitudinal lines at 0, 90, 180, 270 degrees
        for k in 0..4 {
            let a = k as f32 * std::f32::consts::FRAC_PI_2;
            let (s, c) = a.sin_cos();
            let offset = u * (c * r) + v * (s * r);
            positions.push((self.start + offset).to_array());
            positions.push((self.end + offset).to_array());
            strip_lengths.push(2);
        }

        PolylineItem {
            positions,
            strip_lengths,
            default_colour: self.colour,
            line_width: 1.5,

            settings: crate::scene::material::ItemSettings { pick_id: crate::renderer::PickId(id), ..Default::default() },
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` with three sphere handles: start, end, and radius.
    ///
    /// `id_base` = start, `id_base + 1` = end, `id_base + 2` = radius handle.
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let rh = self.handle_pos(CylinderHandle::Radius);

        let rs = handle_world_radius(self.start, &ctx.camera, ctx.viewport_size.y, 10.0);
        let re = handle_world_radius(self.end, &ctx.camera, ctx.viewport_size.y, 10.0);
        let rr = handle_world_radius(rh, &ctx.camera, ctx.viewport_size.y, 8.0);

        let scalar = |h: CylinderHandle| {
            if self.hovered_handle == Some(h) || self.active_handle == Some(h) {
                1.0_f32
            } else {
                0.2
            }
        };

        GlyphItem {
            positions: vec![self.start.to_array(), self.end.to_array(), rh.to_array()],
            vectors: vec![[rs, 0.0, 0.0], [re, 0.0, 0.0], [rr, 0.0, 0.0]],
            scale: 1.0,
            scale_by_magnitude: true,
            scalars: vec![
                scalar(CylinderHandle::Start),
                scalar(CylinderHandle::End),
                scalar(CylinderHandle::Radius),
            ],
            scalar_range: Some((0.0, 1.0)),
            glyph_type: GlyphType::Sphere,

            settings: crate::scene::material::ItemSettings { pick_id: crate::renderer::PickId(id_base), ..Default::default() },
            default_colour: self.handle_colour,
            use_default_colour: self.handle_colour[3] > 0.0,
            ..GlyphItem::default()
        }
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn handle_pos(&self, handle: CylinderHandle) -> glam::Vec3 {
        match handle {
            CylinderHandle::Start => self.start,
            CylinderHandle::End => self.end,
            CylinderHandle::Radius => {
                let mid = (self.start + self.end) * 0.5;
                let axis = self.axis();
                let axis_dir = if axis.length() > 1e-6 {
                    axis.normalize()
                } else {
                    glam::Vec3::Z
                };
                mid + any_perpendicular(axis_dir) * self.radius
            }
        }
    }

    fn hit_test(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        ctx: &WidgetContext,
    ) -> Option<CylinderHandle> {
        let ray = Ray::new(
            Vector::new(ray_origin.x, ray_origin.y, ray_origin.z),
            Vector::new(ray_dir.x, ray_dir.y, ray_dir.z),
        );

        let handles = [
            CylinderHandle::Start,
            CylinderHandle::End,
            CylinderHandle::Radius,
        ];

        let radii = [
            handle_world_radius(self.start, &ctx.camera, ctx.viewport_size.y, 10.0),
            handle_world_radius(self.end, &ctx.camera, ctx.viewport_size.y, 10.0),
            handle_world_radius(
                self.handle_pos(CylinderHandle::Radius),
                &ctx.camera,
                ctx.viewport_size.y,
                8.0,
            ),
        ];

        let positions = [
            self.start,
            self.end,
            self.handle_pos(CylinderHandle::Radius),
        ];

        let mut best: Option<(f32, CylinderHandle)> = None;
        for (i, handle) in handles.iter().enumerate() {
            let pos = positions[i];
            let r = radii[i];
            let d = ray_point_dist(ray_origin, ray_dir, pos);
            if d < r {
                let ball = parry3d::shape::Ball::new(r);
                let pose = Pose::from_parts([pos.x, pos.y, pos.z].into(), glam::Quat::IDENTITY);
                if let Some(t) = ball.cast_ray(&pose, &ray, f32::MAX, true) {
                    if best.is_none() || t < best.unwrap().0 {
                        best = Some((t, *handle));
                    }
                }
            }
        }

        best.map(|(_, h)| h)
    }
}
