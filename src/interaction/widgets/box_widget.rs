//! Box widget: draggable center handle and six face handles.

use crate::interaction::clip_plane::ray_plane_intersection;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem, aabb_wireframe_polyline};
use crate::scene::aabb::Aabb;
use parry3d::math::{Pose, Vector};
use parry3d::query::{Ray, RayCast};

use super::{WidgetContext, WidgetResult, ctx_ray, handle_world_radius};

/// Which handle on the box is being interacted with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoxHandle {
    /// Center: moves the whole box.
    Center,
    /// One of the six face handles. Index: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
    Face(usize),
}

/// An interactive axis-aligned box widget with a center handle and six face handles.
///
/// Use `wireframe_item()` for the box outline (push into `fd.scene.polylines`) and
/// `handle_glyphs()` for the 7 draggable sphere handles (push into `fd.scene.glyphs`).
///
/// # Usage
///
/// ```rust,ignore
/// let mut bw = BoxWidget::new(glam::Vec3::ZERO, glam::Vec3::splat(2.0));
///
/// // Each frame:
/// bw.update(&ctx);
/// fd.scene.polylines.push(bw.wireframe_item(BOX_ID));
/// fd.scene.glyphs.push(bw.handle_glyphs(HANDLE_ID, &ctx));
/// ```
pub struct BoxWidget {
    /// World-space center of the box.
    pub center: glam::Vec3,
    /// Half-extents along each world axis.
    pub half_extents: glam::Vec3,
    /// RGBA color for the wireframe outline.
    pub color: [f32; 4],
    /// RGBA color for the drag handles. When set (non-zero alpha), overrides the default LUT coloring.
    pub handle_color: [f32; 4],

    hovered_handle: Option<BoxHandle>,
    active_handle: Option<BoxHandle>,
    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
    drag_anchor_world: glam::Vec3,
}

impl BoxWidget {
    /// Create a new box widget.
    pub fn new(center: glam::Vec3, half_extents: glam::Vec3) -> Self {
        Self {
            center,
            half_extents: half_extents.max(glam::Vec3::splat(0.01)),
            color: [0.3, 0.8, 0.4, 1.0],
            handle_color: [0.0; 4],
            hovered_handle: None,
            active_handle: None,
            drag_plane_normal: glam::Vec3::Z,
            drag_plane_d: 0.0,
            drag_anchor_world: glam::Vec3::ZERO,
        }
    }

    /// True while a drag session is in progress.
    pub fn is_active(&self) -> bool {
        self.active_handle.is_some()
    }

    /// The current AABB of the box (center +/- half_extents).
    pub fn aabb(&self) -> Aabb {
        Aabb {
            min: self.center - self.half_extents,
            max: self.center + self.half_extents,
        }
    }

    /// Process input for this frame. Returns `Updated` if state changed.
    pub fn update(&mut self, ctx: &WidgetContext) -> WidgetResult {
        let (ro, rd) = ctx_ray(ctx);
        let mut updated = false;

        if self.active_handle.is_none() {
            let hit = self.hit_test(ro, rd, ctx);
            // On the drag_started frame the cursor can be right at the edge and the
            // hit test may miss by a hair. Keep the previous hover so the drag still
            // registers if the handle was highlighted on the frame before the click.
            if hit.is_some() || !ctx.drag_started {
                self.hovered_handle = hit;
            }
        }

        if ctx.drag_started {
            if let Some(handle) = self.hovered_handle {
                let anchor = self.handle_pos(handle);
                // Always use a camera-facing drag plane. For face handles the
                // resize amount is derived by projecting the movement onto the
                // face normal after the hit is computed, not by constraining the
                // plane itself to the face axis (which would make proj always 0).
                let n = -glam::Vec3::from(ctx.camera.forward);
                self.drag_plane_normal = n;
                self.drag_plane_d = -n.dot(anchor);
                self.drag_anchor_world = anchor;
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
                    BoxHandle::Center => {
                        let delta = hit - self.drag_anchor_world;
                        if delta.length_squared() > 1e-10 {
                            self.center += delta;
                            self.drag_anchor_world = hit;
                            updated = true;
                        }
                    }
                    BoxHandle::Face(i) => {
                        // Move this face outward while keeping the opposite face fixed.
                        // proj > 0 means movement in the face normal direction (outward)
                        // for both positive and negative faces, so no sign correction needed.
                        let normal = Self::face_normal(i);
                        let proj = (hit - self.drag_anchor_world).dot(normal);
                        if proj.abs() > 1e-5 {
                            let axis = i / 2; // 0=X, 1=Y, 2=Z
                            let new_he = (self.half_extents[axis] + proj).max(0.01);
                            let he_delta = new_he - self.half_extents[axis];
                            self.half_extents[axis] = new_he;
                            // Keep opposite face fixed: shift center by half the change.
                            self.center += normal * (he_delta * 0.5);
                            self.drag_anchor_world = hit;
                            updated = true;
                        }
                    }
                }
            }
        }

        if updated { WidgetResult::Updated } else { WidgetResult::None }
    }

    /// Build a `PolylineItem` for the box wireframe outline.
    ///
    /// `id` is the pick ID (0 = not pickable).
    pub fn wireframe_item(&self, id: u64) -> PolylineItem {
        let mut item = aabb_wireframe_polyline(&self.aabb(), self.color);
        item.id = id;
        item
    }

    /// Build a `GlyphItem` with 7 sphere handles: center + 6 face centers.
    ///
    /// `id_base` is the pick ID for the center handle; face handles use
    /// `id_base + 1` through `id_base + 6`.
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let handles = [
            BoxHandle::Center,
            BoxHandle::Face(0),
            BoxHandle::Face(1),
            BoxHandle::Face(2),
            BoxHandle::Face(3),
            BoxHandle::Face(4),
            BoxHandle::Face(5),
        ];

        let mut positions = Vec::with_capacity(7);
        let mut vectors = Vec::with_capacity(7);
        let mut scalars = Vec::with_capacity(7);

        for handle in handles {
            let pos = self.handle_pos(handle);
            let r = handle_world_radius(pos, &ctx.camera, ctx.viewport_size.y, 9.0);
            let s = if self.hovered_handle == Some(handle) || self.active_handle == Some(handle) {
                1.0_f32
            } else {
                0.2
            };
            positions.push(pos.to_array());
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

    /// World position of a handle.
    fn handle_pos(&self, handle: BoxHandle) -> glam::Vec3 {
        match handle {
            BoxHandle::Center => self.center,
            BoxHandle::Face(i) => self.center + Self::face_normal(i) * self.half_extents[i / 2],
        }
    }

    /// Outward unit normal for face index 0..5 (+X,-X,+Y,-Y,+Z,-Z).
    fn face_normal(i: usize) -> glam::Vec3 {
        match i {
            0 => glam::Vec3::X,
            1 => glam::Vec3::NEG_X,
            2 => glam::Vec3::Y,
            3 => glam::Vec3::NEG_Y,
            4 => glam::Vec3::Z,
            _ => glam::Vec3::NEG_Z,
        }
    }

    fn hit_test(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        ctx: &WidgetContext,
    ) -> Option<BoxHandle> {
        let ray = Ray::new(
            Vector::new(ray_origin.x, ray_origin.y, ray_origin.z),
            Vector::new(ray_dir.x, ray_dir.y, ray_dir.z),
        );
        let identity = glam::Quat::IDENTITY;

        let handles = [
            BoxHandle::Center,
            BoxHandle::Face(0),
            BoxHandle::Face(1),
            BoxHandle::Face(2),
            BoxHandle::Face(3),
            BoxHandle::Face(4),
            BoxHandle::Face(5),
        ];

        let mut best: Option<(f32, BoxHandle)> = None;

        for handle in handles {
            let pos = self.handle_pos(handle);
            let r = handle_world_radius(pos, &ctx.camera, ctx.viewport_size.y, 9.0);
            let ball = parry3d::shape::Ball::new(r);
            let pose = Pose::from_parts([pos.x, pos.y, pos.z].into(), identity);
            if let Some(t) = ball.cast_ray(&pose, &ray, f32::MAX, true) {
                if best.is_none() || t < best.unwrap().0 {
                    best = Some((t, handle));
                }
            }
        }

        best.map(|(_, h)| h)
    }
}
