//! Box widget: draggable center, face, and rotation-arc handles for an oriented box.

use crate::interaction::clip_plane::ray_plane_intersection;
use crate::renderer::{GlyphItem, GlyphType, PolylineItem};
use crate::scene::aabb::Aabb;
use parry3d::math::{Pose, Vector};
use parry3d::query::{Ray, RayCast};

use super::{WidgetContext, WidgetResult, any_perpendicular_pair, ctx_ray, handle_world_radius};

/// Which handle on the box is being interacted with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoxHandle {
    /// Center: moves the whole box.
    Center,
    /// One of the six face handles. Index: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z (in local box space).
    Face(usize),
    /// One of three rotation arc grips. Index: 0=rotate around X, 1=around Y, 2=around Z.
    RotArc(usize),
}

/// An interactive oriented box widget with translation, resize, and rotation handles.
///
/// The box orientation is controlled by `rotation` (default `Quat::IDENTITY` for axis-aligned).
/// Three rotation arc handles (one per world axis) are rendered as circle overlays.
///
/// Use `wireframe_item()` for the box outline, `rotation_arcs_item()` for the arc circles,
/// and `handle_glyphs()` for all 10 draggable sphere handles.
///
/// # Usage
///
/// ```rust,ignore
/// let mut bw = BoxWidget::new(glam::Vec3::ZERO, glam::Vec3::splat(2.0));
///
/// // Each frame:
/// bw.update(&ctx);
/// fd.scene.polylines.push(bw.wireframe_item(BOX_ID));
/// fd.scene.polylines.push(bw.rotation_arcs_item(ARC_ID));
/// fd.scene.glyphs.push(bw.handle_glyphs(HANDLE_ID, &ctx));
/// ```
pub struct BoxWidget {
    /// World-space center of the box.
    pub center: glam::Vec3,
    /// Half-extents along the box's local X, Y, Z axes.
    pub half_extents: glam::Vec3,
    /// Orientation of the box (rotates the local axes).
    pub rotation: glam::Quat,
    /// RGBA colour for the wireframe outline.
    pub colour: [f32; 4],
    /// RGBA colour for the drag handles. When set (non-zero alpha), overrides the default LUT colouring.
    pub handle_colour: [f32; 4],

    hovered_handle: Option<BoxHandle>,
    active_handle: Option<BoxHandle>,
    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
    drag_anchor_world: glam::Vec3,
}

impl BoxWidget {
    /// Create a new axis-aligned box widget (rotation defaults to identity).
    pub fn new(center: glam::Vec3, half_extents: glam::Vec3) -> Self {
        Self {
            center,
            half_extents: half_extents.max(glam::Vec3::splat(0.01)),
            rotation: glam::Quat::IDENTITY,
            colour: [0.3, 0.8, 0.4, 1.0],
            handle_colour: [0.0; 4],
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

    /// Returns `(center, half_extents, rotation)` for this oriented box.
    pub fn obb(&self) -> (glam::Vec3, glam::Vec3, glam::Quat) {
        (self.center, self.half_extents, self.rotation)
    }

    /// World-space AABB that conservatively bounds the oriented box.
    ///
    /// When `rotation` is identity this equals the exact box. Otherwise it is a
    /// tight enclosing axis-aligned box computed from all 8 oriented corners.
    pub fn aabb(&self) -> Aabb {
        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        let h = self.half_extents;
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    let p =
                        self.center + self.rotation * glam::Vec3::new(sx * h.x, sy * h.y, sz * h.z);
                    min = min.min(p);
                    max = max.max(p);
                }
            }
        }
        Aabb { min, max }
    }

    /// Returns true if `point` lies inside the oriented box.
    pub fn contains_point(&self, point: glam::Vec3) -> bool {
        let local = self.rotation.inverse() * (point - self.center);
        local.x.abs() <= self.half_extents.x
            && local.y.abs() <= self.half_extents.y
            && local.z.abs() <= self.half_extents.z
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
                // Camera-facing drag plane. For face handles the resize amount is
                // derived by projecting movement onto the rotated face normal after
                // the hit is found, not by constraining the drag plane to that axis.
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
            } else {
                match handle {
                    BoxHandle::Center => {
                        if let Some(hit) = ray_plane_intersection(
                            ro,
                            rd,
                            self.drag_plane_normal,
                            self.drag_plane_d,
                        ) {
                            let delta = hit - self.drag_anchor_world;
                            if delta.length_squared() > 1e-10 {
                                self.center += delta;
                                self.drag_anchor_world = hit;
                                updated = true;
                            }
                        }
                    }
                    BoxHandle::Face(i) => {
                        if let Some(hit) = ray_plane_intersection(
                            ro,
                            rd,
                            self.drag_plane_normal,
                            self.drag_plane_d,
                        ) {
                            // Rotated face normal in world space.
                            let world_normal = self.rotation * Self::local_face_normal(i);
                            let proj = (hit - self.drag_anchor_world).dot(world_normal);
                            if proj.abs() > 1e-5 {
                                let axis = i / 2; // 0=X, 1=Y, 2=Z
                                let new_he = (self.half_extents[axis] + proj).max(0.01);
                                let he_delta = new_he - self.half_extents[axis];
                                self.half_extents[axis] = new_he;
                                // Keep the opposite face fixed: shift center along the rotated normal.
                                self.center += world_normal * (he_delta * 0.5);
                                self.drag_anchor_world = hit;
                                updated = true;
                            }
                        }
                    }
                    BoxHandle::RotArc(axis_idx) => {
                        // Project ray onto the plane perpendicular to the rotation axis
                        // through the box center. Measure the angle delta from the last
                        // drag anchor and apply it as an incremental rotation.
                        let axis = Self::world_rotation_axis(axis_idx);
                        let plane_d = -axis.dot(self.center);
                        if let Some(plane_hit) = ray_plane_intersection(ro, rd, axis, plane_d) {
                            let start_dir =
                                (self.drag_anchor_world - self.center).normalize_or_zero();
                            let new_dir = (plane_hit - self.center).normalize_or_zero();
                            if start_dir.length_squared() > 0.5 && new_dir.length_squared() > 0.5 {
                                let cos_a = start_dir.dot(new_dir).clamp(-1.0, 1.0);
                                let cross = start_dir.cross(new_dir);
                                let sign = cross.dot(axis).signum();
                                let angle = cos_a.acos() * sign;
                                if angle.abs() > 1e-5 {
                                    let delta_rot = glam::Quat::from_axis_angle(axis, angle);
                                    self.rotation = (delta_rot * self.rotation).normalize();
                                    // Update anchor to the new position on the arc.
                                    self.drag_anchor_world =
                                        self.center + new_dir * self.arc_radius();
                                    updated = true;
                                }
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

    /// Build a `PolylineItem` for the oriented box wireframe (12 edges).
    ///
    /// `id` is the pick ID (0 = not pickable).
    pub fn wireframe_item(&self, id: u64) -> PolylineItem {
        let c = self.center;
        let h = self.half_extents;
        let r = self.rotation;

        let p =
            |x: f32, y: f32, z: f32| -> [f32; 3] { (c + r * glam::Vec3::new(x, y, z)).to_array() };

        PolylineItem {
            positions: vec![
                // Bottom face loop (local z = -h.z)
                p(-h.x, -h.y, -h.z),
                p(h.x, -h.y, -h.z),
                p(h.x, h.y, -h.z),
                p(-h.x, h.y, -h.z),
                p(-h.x, -h.y, -h.z),
                // Top face loop (local z = +h.z)
                p(-h.x, -h.y, h.z),
                p(h.x, -h.y, h.z),
                p(h.x, h.y, h.z),
                p(-h.x, h.y, h.z),
                p(-h.x, -h.y, h.z),
                // Four vertical edges
                p(-h.x, -h.y, -h.z),
                p(-h.x, -h.y, h.z),
                p(h.x, -h.y, -h.z),
                p(h.x, -h.y, h.z),
                p(h.x, h.y, -h.z),
                p(h.x, h.y, h.z),
                p(-h.x, h.y, -h.z),
                p(-h.x, h.y, h.z),
            ],
            strip_lengths: vec![5, 5, 2, 2, 2, 2],
            default_colour: self.colour,
            id,
            ..PolylineItem::default()
        }
    }

    /// Build a `PolylineItem` containing three rotation arcs (one per world axis).
    ///
    /// Each arc is a full circle at radius `arc_radius()` around the box center.
    /// Arc colours: X = red, Y = green, Z = blue (semi-transparent).
    pub fn rotation_arcs_item(&self, id: u64) -> PolylineItem {
        const STEPS: usize = 48;
        let c = self.center;
        let r = self.arc_radius();
        let arc_colours = [
            [0.9_f32, 0.2, 0.2, 0.7], // X: red
            [0.2, 0.9, 0.2, 0.7],     // Y: green
            [0.2, 0.4, 1.0, 0.7],     // Z: blue
        ];

        // All three arcs concatenated into one PolylineItem for a single draw call.
        // Use the overall widget colour -- the showcase can override with separate items if needed.
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity((STEPS + 1) * 3);
        let mut strip_lengths: Vec<u32> = Vec::new();
        let _ = arc_colours; // colour varies per strip, but PolylineItem has a single colour; we use widget colour

        for axis_idx in 0..3_usize {
            let axis = Self::world_rotation_axis(axis_idx);
            let (u, v) = any_perpendicular_pair(axis);
            for i in 0..=STEPS {
                let a = i as f32 * std::f32::consts::TAU / STEPS as f32;
                let (s, co) = a.sin_cos();
                positions.push((c + u * (co * r) + v * (s * r)).to_array());
            }
            strip_lengths.push((STEPS + 1) as u32);
        }

        PolylineItem {
            positions,
            strip_lengths,
            default_colour: self.colour,
            line_width: 1.2,
            id,
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` with 10 sphere handles: center, 6 face handles, and 3 rotation grips.
    ///
    /// Pick IDs: `id_base` = center, `id_base + 1..6` = faces (+X,-X,+Y,-Y,+Z,-Z),
    /// `id_base + 7..9` = rotation arc grips (X, Y, Z).
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let all_handles = [
            BoxHandle::Center,
            BoxHandle::Face(0),
            BoxHandle::Face(1),
            BoxHandle::Face(2),
            BoxHandle::Face(3),
            BoxHandle::Face(4),
            BoxHandle::Face(5),
            BoxHandle::RotArc(0),
            BoxHandle::RotArc(1),
            BoxHandle::RotArc(2),
        ];

        let mut positions = Vec::with_capacity(10);
        let mut vectors = Vec::with_capacity(10);
        let mut scalars = Vec::with_capacity(10);

        for handle in all_handles {
            let pos = self.handle_pos(handle);
            let target_px = if matches!(handle, BoxHandle::RotArc(_)) {
                7.0
            } else {
                9.0
            };
            let r = handle_world_radius(pos, &ctx.camera, ctx.viewport_size.y, target_px);
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
            default_colour: self.handle_colour,
            use_default_colour: self.handle_colour[3] > 0.0,
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
            BoxHandle::Face(i) => {
                self.center
                    + self.rotation * (Self::local_face_normal(i) * self.half_extents[i / 2])
            }
            BoxHandle::RotArc(i) => self.center + self.arc_grip_offset(i),
        }
    }

    /// Local-space outward unit normal for face index 0..5 (+X,-X,+Y,-Y,+Z,-Z).
    fn local_face_normal(i: usize) -> glam::Vec3 {
        match i {
            0 => glam::Vec3::X,
            1 => glam::Vec3::NEG_X,
            2 => glam::Vec3::Y,
            3 => glam::Vec3::NEG_Y,
            4 => glam::Vec3::Z,
            _ => glam::Vec3::NEG_Z,
        }
    }

    /// World-space rotation axis for arc index 0..2 (X, Y, Z).
    fn world_rotation_axis(i: usize) -> glam::Vec3 {
        match i {
            0 => glam::Vec3::X,
            1 => glam::Vec3::Y,
            _ => glam::Vec3::Z,
        }
    }

    /// Radius of the rotation arc circles. Slightly larger than the box diagonal.
    fn arc_radius(&self) -> f32 {
        self.half_extents.length() * 1.4 + 0.1
    }

    /// World-space offset from center to the grip sphere for rotation arc `i`.
    fn arc_grip_offset(&self, i: usize) -> glam::Vec3 {
        let r = self.arc_radius();
        // Each arc's grip is positioned along a world axis perpendicular to the rotation axis,
        // so the three grips sit at distinct positions.
        match i {
            0 => glam::Vec3::new(0.0, r, 0.0), // X-arc grip: +Y
            1 => glam::Vec3::new(0.0, 0.0, r), // Y-arc grip: +Z
            _ => glam::Vec3::new(r, 0.0, 0.0), // Z-arc grip: +X
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

        let all_handles = [
            BoxHandle::Center,
            BoxHandle::Face(0),
            BoxHandle::Face(1),
            BoxHandle::Face(2),
            BoxHandle::Face(3),
            BoxHandle::Face(4),
            BoxHandle::Face(5),
            BoxHandle::RotArc(0),
            BoxHandle::RotArc(1),
            BoxHandle::RotArc(2),
        ];

        let mut best: Option<(f32, BoxHandle)> = None;

        for handle in all_handles {
            let pos = self.handle_pos(handle);
            let target_px = if matches!(handle, BoxHandle::RotArc(_)) {
                7.0
            } else {
                9.0
            };
            let r = handle_world_radius(pos, &ctx.camera, ctx.viewport_size.y, target_px);
            let ball = parry3d::shape::Ball::new(r);
            let pose = Pose::from_parts([pos.x, pos.y, pos.z].into(), glam::Quat::IDENTITY);
            if let Some(t) = ball.cast_ray(&pose, &ray, f32::MAX, true) {
                if best.is_none() || t < best.unwrap().0 {
                    best = Some((t, handle));
                }
            }
        }

        best.map(|(_, h)| h)
    }
}
