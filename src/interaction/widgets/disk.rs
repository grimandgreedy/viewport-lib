//! Disk widget: a bounded circular plane with center, normal, and radius handles.

use crate::interaction::clip_plane::ray_plane_intersection;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem};
use parry3d::math::{Pose, Vector};
use parry3d::query::{Ray, RayCast};

use super::{
    WidgetContext, WidgetResult, any_perpendicular, any_perpendicular_pair, ctx_ray,
    handle_world_radius, ray_point_dist,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DiskHandle {
    Center,
    NormalTip,
    Radius,
}

/// An interactive disk widget: a bounded circular plane with center, normal, and radius.
///
/// Three handles: center (moves freely), normal tip (orbits the normal), and a radius
/// handle on the disk rim (scales the radius).
///
/// # Usage
///
/// ```rust,ignore
/// let mut disk = DiskWidget::new(glam::Vec3::ZERO, glam::Vec3::Z, 2.0);
///
/// // Each frame:
/// disk.update(&ctx);
/// fd.scene.polylines.push(disk.wireframe_item(DISK_ID));
/// fd.scene.glyphs.push(disk.handle_glyphs(HANDLE_ID, &ctx));
/// ```
pub struct DiskWidget {
    /// World-space center of the disk.
    pub center: glam::Vec3,
    /// Unit normal vector of the disk plane.
    pub normal: glam::Vec3,
    /// Radius in world units.
    pub radius: f32,
    /// RGBA colour for the wireframe circle and normal line.
    pub colour: [f32; 4],
    /// RGBA colour for the drag handles. Non-zero alpha overrides LUT colouring.
    pub handle_colour: [f32; 4],
    /// Distance from center to the normal-tip handle sphere.
    pub normal_display_length: f32,

    hovered_handle: Option<DiskHandle>,
    active_handle: Option<DiskHandle>,
    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
    drag_anchor: glam::Vec3,
}

impl DiskWidget {
    /// Create a new disk widget.
    ///
    /// `normal` is normalized internally; if zero, defaults to `Vec3::Z`.
    pub fn new(center: glam::Vec3, normal: glam::Vec3, radius: f32) -> Self {
        let len = normal.length();
        let normal = if len > 1e-6 {
            normal / len
        } else {
            glam::Vec3::Z
        };
        Self {
            center,
            normal,
            radius: radius.max(0.01),
            colour: [0.9, 0.6, 0.1, 1.0],
            handle_colour: [0.0; 4],
            normal_display_length: 2.0,
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
                    DiskHandle::Center => self.center,
                    DiskHandle::NormalTip => self.normal_tip_pos(),
                    DiskHandle::Radius => self.radius_handle_pos(),
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
                    DiskHandle::Center => {
                        let delta = hit - self.drag_anchor;
                        if delta.length_squared() > 1e-10 {
                            self.center += delta;
                            self.drag_anchor = hit;
                            updated = true;
                        }
                    }
                    DiskHandle::NormalTip => {
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
                    DiskHandle::Radius => {
                        let new_r = (hit - self.center).length().max(0.01);
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

    /// Build a `PolylineItem` for the wireframe circle and normal indicator.
    ///
    /// The circle uses 32 segments. A line from center to the normal tip is also included.
    pub fn wireframe_item(&self, id: u64) -> PolylineItem {
        const STEPS: usize = 32;
        let (u, v) = any_perpendicular_pair(self.normal);
        let c = self.center;
        let r = self.radius;

        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(STEPS + 1 + 2);
        for i in 0..=STEPS {
            let a = i as f32 * std::f32::consts::TAU / STEPS as f32;
            let (s, co) = a.sin_cos();
            positions.push((c + u * (co * r) + v * (s * r)).to_array());
        }
        positions.push(c.to_array());
        positions.push(self.normal_tip_pos().to_array());

        PolylineItem {
            positions,
            strip_lengths: vec![(STEPS + 1) as u32, 2],
            default_colour: self.colour,
            line_width: 1.5,

            settings: crate::scene::material::ItemSettings { pick_id: crate::renderer::PickId(id), ..Default::default() },
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` with three sphere handles: center, normal tip, and radius edge.
    ///
    /// `id_base` = center, `id_base + 1` = normal tip, `id_base + 2` = radius handle.
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let tip = self.normal_tip_pos();
        let rh = self.radius_handle_pos();

        let rc = handle_world_radius(self.center, &ctx.camera, ctx.viewport_size.y, 10.0);
        let rt = handle_world_radius(tip, &ctx.camera, ctx.viewport_size.y, 8.0);
        let rr = handle_world_radius(rh, &ctx.camera, ctx.viewport_size.y, 8.0);

        let scalar = |h: DiskHandle| {
            if self.hovered_handle == Some(h) || self.active_handle == Some(h) {
                1.0_f32
            } else {
                0.2
            }
        };

        GlyphItem {
            positions: vec![self.center.to_array(), tip.to_array(), rh.to_array()],
            vectors: vec![[rc, 0.0, 0.0], [rt, 0.0, 0.0], [rr, 0.0, 0.0]],
            scale: 1.0,
            scale_by_magnitude: true,
            scalars: vec![
                scalar(DiskHandle::Center),
                scalar(DiskHandle::NormalTip),
                scalar(DiskHandle::Radius),
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

    fn normal_tip_pos(&self) -> glam::Vec3 {
        self.center + self.normal * self.normal_display_length
    }

    fn radius_handle_pos(&self) -> glam::Vec3 {
        // Radius handle sits on the disk rim along any perpendicular direction.
        self.center + any_perpendicular(self.normal) * self.radius
    }

    fn hit_test(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        ctx: &WidgetContext,
    ) -> Option<DiskHandle> {
        let tip = self.normal_tip_pos();
        let rh = self.radius_handle_pos();

        let ray = Ray::new(
            Vector::new(ray_origin.x, ray_origin.y, ray_origin.z),
            Vector::new(ray_dir.x, ray_dir.y, ray_dir.z),
        );

        let rc = handle_world_radius(self.center, &ctx.camera, ctx.viewport_size.y, 10.0);
        let rt = handle_world_radius(tip, &ctx.camera, ctx.viewport_size.y, 8.0);
        let rr = handle_world_radius(rh, &ctx.camera, ctx.viewport_size.y, 8.0);

        let handles = [
            (DiskHandle::Center, self.center, rc),
            (DiskHandle::NormalTip, tip, rt),
            (DiskHandle::Radius, rh, rr),
        ];

        let mut best: Option<(f32, DiskHandle)> = None;
        for (handle, pos, radius) in handles {
            let d = ray_point_dist(ray_origin, ray_dir, pos);
            if d < radius {
                let ball = parry3d::shape::Ball::new(radius);
                let pose = Pose::from_parts([pos.x, pos.y, pos.z].into(), glam::Quat::IDENTITY);
                if let Some(t) = ball.cast_ray(&pose, &ray, f32::MAX, true) {
                    if best.is_none() || t < best.unwrap().0 {
                        best = Some((t, handle));
                    }
                }
            }
        }

        best.map(|(_, h)| h)
    }
}
