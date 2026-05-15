//! Plane widget: a draggable infinite plane defined by a center point and normal.

use crate::interaction::clip_plane::ray_plane_intersection;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem};
use parry3d::math::{Pose, Vector};
use parry3d::query::{Ray, RayCast};

use super::{
    WidgetContext, WidgetResult, any_perpendicular_pair, ctx_ray, handle_world_radius,
    ray_point_dist,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PlaneHandle {
    Center,
    NormalTip,
}

/// An interactive plane widget defined by a center point and unit normal.
///
/// Two handles: the center (moves the plane freely in 3D) and a normal tip
/// (orbits the normal direction around the center by dragging).
///
/// # Usage
///
/// ```rust,ignore
/// let mut plane = PlaneWidget::new(glam::Vec3::ZERO, glam::Vec3::Z);
///
/// // Each frame:
/// plane.update(&ctx);
/// fd.scene.polylines.push(plane.plane_item(PLANE_ID));
/// fd.scene.glyphs.push(plane.handle_glyphs(HANDLE_ID, &ctx));
///
/// // Suppress orbit while dragging:
/// if plane.is_active() { orbit.resolve(); } else { orbit.apply_to_camera(&mut camera); }
/// ```
pub struct PlaneWidget {
    /// World-space center of the plane.
    pub center: glam::Vec3,
    /// Unit normal vector of the plane (always normalized on output).
    pub normal: glam::Vec3,
    /// RGBA color for the wireframe square outline and normal line.
    pub color: [f32; 4],
    /// RGBA color for the drag handles. Non-zero alpha overrides LUT coloring.
    pub handle_color: [f32; 4],
    /// Half-size of the visual square in world space.
    pub display_half_size: f32,
    /// Distance from center to the normal-tip handle sphere.
    pub normal_display_length: f32,

    hovered_handle: Option<PlaneHandle>,
    active_handle: Option<PlaneHandle>,
    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
    drag_anchor: glam::Vec3,
}

impl PlaneWidget {
    /// Create a new plane widget.
    ///
    /// `normal` is normalized internally; if zero, defaults to `Vec3::Z`.
    pub fn new(center: glam::Vec3, normal: glam::Vec3) -> Self {
        let len = normal.length();
        let normal = if len > 1e-6 {
            normal / len
        } else {
            glam::Vec3::Z
        };
        Self {
            center,
            normal,
            color: [0.3, 0.7, 1.0, 1.0],
            handle_color: [0.0; 4],
            display_half_size: 1.5,
            normal_display_length: 2.0,
            hovered_handle: None,
            active_handle: None,
            drag_plane_normal: glam::Vec3::Z,
            drag_plane_d: 0.0,
            drag_anchor: glam::Vec3::ZERO,
        }
    }

    /// True while a drag is in progress on either handle.
    pub fn is_active(&self) -> bool {
        self.active_handle.is_some()
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
                let anchor = match handle {
                    PlaneHandle::Center => self.center,
                    PlaneHandle::NormalTip => self.normal_tip_pos(),
                };
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
                    PlaneHandle::Center => {
                        let delta = hit - self.drag_anchor;
                        if delta.length_squared() > 1e-10 {
                            self.center += delta;
                            self.drag_anchor = hit;
                            updated = true;
                        }
                    }
                    PlaneHandle::NormalTip => {
                        let dir = hit - self.center;
                        let len = dir.length();
                        if len > 1e-3 {
                            let new_normal = dir / len;
                            if (new_normal - self.normal).length_squared() > 1e-8 {
                                self.normal = new_normal;
                                updated = true;
                            }
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

    /// Build a `PolylineItem` for the wireframe square outline and normal indicator.
    ///
    /// The square is oriented by `normal` and centered at `center`.
    /// A line segment from the center to the normal-tip handle is included.
    pub fn plane_item(&self, id: u64) -> PolylineItem {
        let (u, v) = any_perpendicular_pair(self.normal);
        let s = self.display_half_size;
        let c = self.center;
        let tip = self.normal_tip_pos();

        // 5 points close the square, then 2 points for the normal indicator line.
        let positions = vec![
            (c + u * s + v * s).to_array(),
            (c - u * s + v * s).to_array(),
            (c - u * s - v * s).to_array(),
            (c + u * s - v * s).to_array(),
            (c + u * s + v * s).to_array(),
            c.to_array(),
            tip.to_array(),
        ];

        PolylineItem {
            positions,
            strip_lengths: vec![5, 2],
            default_color: self.color,
            line_width: 1.5,
            id,
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` with two sphere handles: center and normal tip.
    ///
    /// `id_base` is the pick ID for the center handle; `id_base + 1` for the normal-tip handle.
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let tip = self.normal_tip_pos();
        let rc = handle_world_radius(self.center, &ctx.camera, ctx.viewport_size.y, 10.0);
        let rt = handle_world_radius(tip, &ctx.camera, ctx.viewport_size.y, 8.0);

        let sc = if matches!(self.hovered_handle, Some(PlaneHandle::Center))
            || matches!(self.active_handle, Some(PlaneHandle::Center))
        {
            1.0_f32
        } else {
            0.2
        };
        let st = if matches!(self.hovered_handle, Some(PlaneHandle::NormalTip))
            || matches!(self.active_handle, Some(PlaneHandle::NormalTip))
        {
            1.0_f32
        } else {
            0.2
        };

        GlyphItem {
            positions: vec![self.center.to_array(), tip.to_array()],
            vectors: vec![[rc, 0.0, 0.0], [rt, 0.0, 0.0]],
            scale: 1.0,
            scale_by_magnitude: true,
            scalars: vec![sc, st],
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

    fn normal_tip_pos(&self) -> glam::Vec3 {
        self.center + self.normal * self.normal_display_length
    }

    fn hit_test(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        ctx: &WidgetContext,
    ) -> Option<PlaneHandle> {
        let tip = self.normal_tip_pos();
        let ray = Ray::new(
            Vector::new(ray_origin.x, ray_origin.y, ray_origin.z),
            Vector::new(ray_dir.x, ray_dir.y, ray_dir.z),
        );

        let rc = handle_world_radius(self.center, &ctx.camera, ctx.viewport_size.y, 10.0);
        let rt = handle_world_radius(tip, &ctx.camera, ctx.viewport_size.y, 8.0);

        let dc = ray_point_dist(ray_origin, ray_dir, self.center);
        let dt = ray_point_dist(ray_origin, ray_dir, tip);

        let center_ball = parry3d::shape::Ball::new(rc);
        let tip_ball = parry3d::shape::Ball::new(rt);
        let center_pose = Pose::from_parts(
            [self.center.x, self.center.y, self.center.z].into(),
            glam::Quat::IDENTITY,
        );
        let tip_pose = Pose::from_parts([tip.x, tip.y, tip.z].into(), glam::Quat::IDENTITY);

        let tc = if dc < rc {
            center_ball.cast_ray(&center_pose, &ray, f32::MAX, true)
        } else {
            None
        };
        let tt = if dt < rt {
            tip_ball.cast_ray(&tip_pose, &ray, f32::MAX, true)
        } else {
            None
        };

        match (tc, tt) {
            (Some(a), Some(b)) => Some(if a <= b {
                PlaneHandle::Center
            } else {
                PlaneHandle::NormalTip
            }),
            (Some(_), None) => Some(PlaneHandle::Center),
            (None, Some(_)) => Some(PlaneHandle::NormalTip),
            (None, None) => None,
        }
    }
}
