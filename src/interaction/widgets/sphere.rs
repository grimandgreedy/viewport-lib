//! Sphere widget: draggable center handle and radius handle.

use crate::interaction::clip_plane::ray_plane_intersection;
use crate::renderer::{ClipObject, ClipShape, GlyphItem, GlyphType, PolylineItem};
use parry3d::math::{Pose, Vector};
use parry3d::query::{Ray, RayCast};

use super::{WidgetContext, WidgetResult, ctx_ray, handle_world_radius};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SphereHandle {
    Center,
    Radius,
}

/// An interactive sphere widget with a draggable center and radius handle.
///
/// Use `clip_object()` to get the visual fill/outline (push into
/// `fd.effects.clip_objects` with `clip_geometry: false`), and `handle_glyphs()`
/// for the draggable handle spheres (push into `fd.scene.glyphs`).
///
/// # Usage
///
/// ```rust,ignore
/// let mut sphere = SphereWidget::new(glam::Vec3::ZERO, 2.0);
///
/// // Each frame:
/// sphere.update(&ctx);
/// fd.effects.clip_objects.push(sphere.clip_object());
/// fd.scene.glyphs.push(sphere.handle_glyphs(HANDLE_ID, &ctx));
/// ```
pub struct SphereWidget {
    /// World-space center of the sphere.
    pub center: glam::Vec3,
    /// Radius in world units.
    pub radius: f32,
    /// RGBA fill colour (alpha controls transparency of the fill).
    pub colour: [f32; 4],
    /// RGBA colour for the drag handles. When set (non-zero alpha), overrides the default LUT colouring.
    pub handle_colour: [f32; 4],

    hovered_handle: Option<SphereHandle>,
    active_handle: Option<SphereHandle>,
    drag_plane_normal: glam::Vec3,
    drag_plane_d: f32,
    drag_anchor_world: glam::Vec3,
    drag_anchor_radius: f32,
}

impl SphereWidget {
    /// Create a new sphere widget.
    pub fn new(center: glam::Vec3, radius: f32) -> Self {
        Self {
            center,
            radius: radius.max(0.01),
            colour: [0.3, 0.6, 1.0, 0.25],
            handle_colour: [0.0; 4],
            hovered_handle: None,
            active_handle: None,
            drag_plane_normal: glam::Vec3::Z,
            drag_plane_d: 0.0,
            drag_anchor_world: glam::Vec3::ZERO,
            drag_anchor_radius: 0.0,
        }
    }

    /// True while a drag session is in progress.
    pub fn is_active(&self) -> bool {
        self.active_handle.is_some()
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
                let anchor = match handle {
                    SphereHandle::Center => self.center,
                    SphereHandle::Radius => self.radius_handle_pos(),
                };
                let fwd = glam::Vec3::from(ctx.camera.forward);
                let n = -fwd;
                self.drag_plane_normal = n;
                self.drag_plane_d = -n.dot(anchor);
                self.drag_anchor_world = anchor;
                self.drag_anchor_radius = self.radius;
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
                    SphereHandle::Center => {
                        let delta = hit - self.drag_anchor_world;
                        let new_center = self.center + delta;
                        if (new_center - self.center).length_squared() > 1e-10 {
                            self.center = new_center;
                            self.drag_anchor_world = hit;
                            updated = true;
                        }
                    }
                    SphereHandle::Radius => {
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

    /// Build a `ClipObject` for the sphere visual (fill + outline).
    ///
    /// Push into `fd.effects.clip_objects`. `clip_geometry` is set to `false` so
    /// the sphere only renders as a visual indicator and does not clip geometry.
    pub fn clip_object(&self) -> ClipObject {
        let edge = [self.colour[0], self.colour[1], self.colour[2], 1.0];
        ClipObject {
            shape: ClipShape::Sphere {
                center: self.center.to_array(),
                radius: self.radius,
            },
            colour: Some(self.colour),
            edge_colour: Some(edge),
            clip_geometry: false,
            enabled: true,
            hovered: self.hovered_handle.is_some(),
            active: self.active_handle.is_some(),
            ..ClipObject::default()
        }
    }

    /// Build a `PolylineItem` of three great circle rings (XY, XZ, YZ planes).
    ///
    /// Push into `fd.scene.polylines` for a scene-pass outline consistent with
    /// how `BoxWidget::wireframe_item` is used. `id` is the pick ID (0 = not pickable).
    pub fn wireframe_item(&self, id: u64) -> PolylineItem {
        const STEPS: usize = 64;
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(STEPS * 3 + 3);
        let mut strip_lengths: Vec<u32> = Vec::with_capacity(3);
        let c = self.center;
        let r = self.radius;

        for ring in 0..3_usize {
            for i in 0..=STEPS {
                let a = i as f32 * std::f32::consts::TAU / STEPS as f32;
                let (s, co) = a.sin_cos();
                let p = match ring {
                    0 => glam::Vec3::new(co * r, s * r, 0.0),
                    1 => glam::Vec3::new(co * r, 0.0, s * r),
                    _ => glam::Vec3::new(0.0, co * r, s * r),
                };
                positions.push((c + p).to_array());
            }
            strip_lengths.push((STEPS + 1) as u32);
        }

        let line_colour = [self.colour[0], self.colour[1], self.colour[2], 1.0];
        PolylineItem {
            positions,
            strip_lengths,
            default_colour: line_colour,
            line_width: 1.5,
            id,
            ..PolylineItem::default()
        }
    }

    /// Build a `GlyphItem` with sphere handles: one at the center, one at the
    /// radius edge (along +X from center).
    ///
    /// `id_base` is the pick ID for the center handle; `id_base + 1` for the radius handle.
    pub fn handle_glyphs(&self, id_base: u64, ctx: &WidgetContext) -> GlyphItem {
        let rp = self.radius_handle_pos();
        let r_center = handle_world_radius(self.center, &ctx.camera, ctx.viewport_size.y, 10.0);
        let r_rh = handle_world_radius(rp, &ctx.camera, ctx.viewport_size.y, 8.0);

        let sc = if self.hovered_handle == Some(SphereHandle::Center)
            || self.active_handle == Some(SphereHandle::Center)
        {
            1.0_f32
        } else {
            0.2
        };
        let sr = if self.hovered_handle == Some(SphereHandle::Radius)
            || self.active_handle == Some(SphereHandle::Radius)
        {
            1.0_f32
        } else {
            0.2
        };

        GlyphItem {
            positions: vec![self.center.to_array(), rp.to_array()],
            vectors: vec![[r_center, 0.0, 0.0], [r_rh, 0.0, 0.0]],
            scale: 1.0,
            scale_by_magnitude: true,
            scalars: vec![sc, sr],
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

    fn radius_handle_pos(&self) -> glam::Vec3 {
        self.center + glam::Vec3::X * self.radius
    }

    fn hit_test(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        ctx: &WidgetContext,
    ) -> Option<SphereHandle> {
        let ray = Ray::new(
            Vector::new(ray_origin.x, ray_origin.y, ray_origin.z),
            Vector::new(ray_dir.x, ray_dir.y, ray_dir.z),
        );

        let rp = self.radius_handle_pos();
        let rh_r = handle_world_radius(rp, &ctx.camera, ctx.viewport_size.y, 8.0);
        let ch_r = handle_world_radius(self.center, &ctx.camera, ctx.viewport_size.y, 10.0);

        let rh_ball = parry3d::shape::Ball::new(rh_r);
        let ch_ball = parry3d::shape::Ball::new(ch_r);

        let rh_pose = Pose::from_parts([rp.x, rp.y, rp.z].into(), glam::Quat::IDENTITY);
        let ch_pose = Pose::from_parts(
            [self.center.x, self.center.y, self.center.z].into(),
            glam::Quat::IDENTITY,
        );

        let t_rh = rh_ball.cast_ray(&rh_pose, &ray, f32::MAX, true).map(|i| i);
        let t_ch = ch_ball.cast_ray(&ch_pose, &ray, f32::MAX, true).map(|i| i);

        match (t_ch, t_rh) {
            (Some(tc), Some(tr)) => Some(if tc <= tr {
                SphereHandle::Center
            } else {
                SphereHandle::Radius
            }),
            (Some(_), None) => Some(SphereHandle::Center),
            (None, Some(_)) => Some(SphereHandle::Radius),
            (None, None) => None,
        }
    }
}
