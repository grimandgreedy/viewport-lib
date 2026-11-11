//! Overlay items placed in the world: a [`WorldOverlayLayer`] holds an
//! [`OverlayFrame`] whose coordinates are world units on a [`WorldPlane`].

use crate::overlay::*;
use glam::{DVec2, DVec3, Mat4, Vec4};

/// A world plane for 2D content.
///
/// The same two planes, with the same axes, as `Camera2d`'s `Plane2d` and the
/// tilemap's `TilemapPlane`, so a world layer, a tilemap and a 2D camera on one
/// plane line up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum WorldPlane {
    /// The world XY plane seen from above, looking along -Z. Plane x is world
    /// +X and plane y is world +Y; elevation is world +Z.
    #[default]
    Xy,
    /// The world XZ plane seen from the front, looking along +Y. Plane x is
    /// world +X and plane y is world +Z; elevation is world -Y, towards the
    /// camera.
    Xz,
}

impl WorldPlane {
    /// World direction of plane +x.
    pub fn x_axis(self) -> glam::Vec3 {
        glam::Vec3::X
    }

    /// World direction of plane +y.
    pub fn y_axis(self) -> glam::Vec3 {
        match self {
            Self::Xy => glam::Vec3::Y,
            Self::Xz => glam::Vec3::Z,
        }
    }

    /// World direction of increasing elevation: out of the plane, towards a
    /// camera looking at it.
    pub fn normal(self) -> glam::Vec3 {
        match self {
            Self::Xy => glam::Vec3::Z,
            Self::Xz => glam::Vec3::NEG_Y,
        }
    }

    /// The world point at plane position `p` and `elevation`.
    pub fn to_world(self, p: DVec2, elevation: f64) -> DVec3 {
        self.x_axis().as_dvec3() * p.x
            + self.y_axis().as_dvec3() * p.y
            + self.normal().as_dvec3() * elevation
    }

    /// The plane position of a world point, dropping its elevation.
    pub fn to_plane(self, p: DVec3) -> DVec2 {
        DVec2::new(
            p.dot(self.x_axis().as_dvec3()),
            p.dot(self.y_axis().as_dvec3()),
        )
    }

    /// The elevation of a world point above the plane.
    pub fn elevation_of(self, p: DVec3) -> f64 {
        p.dot(self.normal().as_dvec3())
    }
}

/// Overlay items drawn in the world, on a plane, under the scene camera.
///
/// Submit layers through `SceneFrame::world_overlays`. Each layer's `frame` is
/// an ordinary [`OverlayFrame`] (shapes, labels, glyph runs, polylines,
/// meshes, retained groups) whose coordinates are world units in the layer's
/// own frame: layer x runs along plane +x and layer y runs along plane -y,
/// down the plane the way screen y runs down the screen, from the plane point
/// `origin`. Layer point `(x, y)` is plane point `(origin.x + x, origin.y -
/// y)` at `elevation`. Text reads upright and rectangles grow right and down,
/// as on screen; [`layer_point`](Self::layer_point) and
/// [`plane_point`](Self::plane_point) convert.
///
/// The layer draws inside the scene, after the opaque scene and before
/// transparency, post effects and screen overlays, so it is covered by bloom,
/// tone mapping and a 2D light layer like other scene content. Depth writes
/// are off; `depth_test` decides whether nearer scene geometry covers it.
///
/// Lengths in the frame's items are world units: positions, sizes, corner
/// radii, stroke widths, shadow blur, offset and spread, font sizes, clip
/// rects and `OverlayTransform`. What is defined in device pixels stays in
/// device pixels, measured on the plane at the camera's current zoom: a
/// hairline (a stroke of width `0.0`) is one device pixel wide, the
/// anti-aliasing fringe and clip edges are one device pixel, and text and
/// small vector shapes rasterise at their size on screen, so they re-rasterise
/// as the camera zooms. Text whose `text_style.sizing` is
/// [`TextSizing::World`](crate::overlay::TextSizing::World) rasterises at
/// stepped sizes instead, so a zoom reuses a bounded set of glyphs, and can
/// clamp or fade by its size on screen. Under a perspective camera the zoom
/// is taken where the view's centre ray meets the plane.
///
/// Inside a world layer, `OverlayFrame::layers` composite as they do on the
/// screen, nested ones included: a layer's members draw into a target the size
/// of the viewport, which composites into the scene with the layer's blur,
/// shadow, colour filter and backdrop. A layer's clip does not apply to its
/// composite there (its members keep their own clips), and members drawn into
/// a layer's target do not test scene depth.
///
/// Blend modes that read the destination and backdrop blurs read the scene as
/// drawn so far: the renderer ends the scene pass where one is needed, copies
/// what is under it and resumes. In a host-owned pass (`paint`,
/// `paint_viewport`), on the compatibility tier, or at a render scale below
/// one, the scene pass cannot be ended there: those blends draw as `SourceOver`
/// and backdrops are not drawn.
///
/// Viewport-anchored items resolve against the rectangle `size`
/// from `origin`; world-anchored items resolve to the point's place on the
/// layer and are dropped outside that rectangle.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct WorldOverlayLayer {
    /// The plane the layer lies on. Default `WorldPlane::Xy`.
    pub plane: WorldPlane,
    /// Plane position of the layer's `(0, 0)`. Default `[0, 0]`.
    pub origin: [f32; 2],
    /// Height above the plane, along [`WorldPlane::normal`]. Default `0.0`.
    pub elevation: f32,
    /// The layer rectangle viewport-anchored items resolve against, in world
    /// units right and down from `origin`. Default `[0, 0]`, which puts every
    /// viewport anchor at `origin`. It does not clip.
    pub size: [f32; 2],
    /// Painter's order against the scene's other painter's-order content
    /// (painter sprites and blended tilemap layers): a layer draws after that
    /// content at its own and lower layer numbers, and before content at
    /// higher ones. Layers with equal `order` draw in submission order.
    /// Default `0`.
    pub order: u32,
    /// When `true` the layer tests against scene depth, so nearer geometry
    /// covers it; when `false` it draws over everything already in the scene.
    /// Default `false`.
    pub depth_test: bool,
    /// Whether a painter mask cuts the layer: it is cut when `order` is in
    /// the range of the mask in the slot it names. Items composited through a
    /// nested overlay layer are not cut. Default [`MaskTest::None`].
    pub mask: crate::render_item::MaskTest,
    /// The items, in world units on the layer.
    pub frame: OverlayFrame,
}

impl Default for WorldOverlayLayer {
    fn default() -> Self {
        Self {
            plane: WorldPlane::Xy,
            origin: [0.0, 0.0],
            elevation: 0.0,
            size: [0.0, 0.0],
            order: 0,
            depth_test: false,
            mask: crate::render_item::MaskTest::None,
            frame: OverlayFrame::default(),
        }
    }
}

/// What a [`WorldOverlayLayer::pick`] hit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct WorldOverlayHit {
    /// The item that was hit.
    pub item: WorldOverlayItem,
    /// Where the ray crossed the layer, in layer units.
    pub layer_point: [f32; 2],
    /// The same point in the world.
    pub world_point: DVec3,
    /// Distance along the ray, in units of the ray direction's length.
    pub distance: f64,
}

/// An item of a world layer's frame, by list and index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WorldOverlayItem {
    /// `frame.shapes[i]`.
    Shape(usize),
    /// `frame.meshes[i]`.
    Mesh(usize),
}

impl WorldOverlayLayer {
    /// A layer on `plane` with the defaults for everything else.
    pub fn new(plane: WorldPlane) -> Self {
        Self {
            plane,
            ..Self::default()
        }
    }

    /// Builder: set the plane position of the layer's `(0, 0)`.
    pub fn with_origin(mut self, origin: [f32; 2]) -> Self {
        self.origin = origin;
        self
    }

    /// Builder: set the height above the plane.
    pub fn with_elevation(mut self, elevation: f32) -> Self {
        self.elevation = elevation;
        self
    }

    /// Builder: set the rectangle viewport anchors resolve against.
    pub fn with_size(mut self, size: [f32; 2]) -> Self {
        self.size = size;
        self
    }

    /// Builder: set the painter's order.
    pub fn with_order(mut self, order: u32) -> Self {
        self.order = order;
        self
    }

    /// Builder: test against scene depth.
    pub fn with_depth_test(mut self, depth_test: bool) -> Self {
        self.depth_test = depth_test;
        self
    }

    /// Builder: set the items.
    pub fn with_frame(mut self, frame: OverlayFrame) -> Self {
        self.frame = frame;
        self
    }

    /// The layer point at plane position `p`.
    pub fn layer_point(&self, p: [f32; 2]) -> [f32; 2] {
        [p[0] - self.origin[0], self.origin[1] - p[1]]
    }

    /// The plane position of layer point `p`.
    pub fn plane_point(&self, p: [f32; 2]) -> [f32; 2] {
        [self.origin[0] + p[0], self.origin[1] - p[1]]
    }

    /// The world point at layer point `p`.
    pub fn world_point(&self, p: [f32; 2]) -> DVec3 {
        let q = self.plane_point(p);
        self.plane
            .to_world(DVec2::new(q[0] as f64, q[1] as f64), self.elevation as f64)
    }

    /// The world point of layer `(0, 0)`.
    pub fn world_origin_point(&self) -> DVec3 {
        self.world_point([0.0, 0.0])
    }

    /// The map from layer coordinates to world coordinates relative to
    /// `world_origin`: column 0 is the world step of one layer unit in x,
    /// column 1 of one unit in y (down the plane), column 2 the plane normal
    /// and column 3 the layer origin.
    pub fn layer_to_world(&self, world_origin: DVec3) -> Mat4 {
        let o = (self.world_origin_point() - world_origin).as_vec3();
        Mat4::from_cols(
            self.plane.x_axis().extend(0.0),
            (-self.plane.y_axis()).extend(0.0),
            self.plane.normal().extend(0.0),
            o.extend(1.0),
        )
    }

    /// The extent anchors resolve against: `size`, with a zero side kept just
    /// above zero so it stays a valid viewport.
    pub fn anchor_extent(&self) -> [f32; 2] {
        [self.size[0].max(1.0e-6), self.size[1].max(1.0e-6)]
    }

    /// A view and projection that take a world point (relative to
    /// `world_origin`) to the layer the way a camera takes it to the screen,
    /// for resolving anchors with
    /// [`resolve_anchor_origin_at`](crate::overlay::resolve_anchor_origin_at)
    /// over [`anchor_extent`](Self::anchor_extent): the view drops the point
    /// onto the layer and the projection maps the extent to clip space.
    pub fn anchor_view_projection(&self, world_origin: DVec3) -> (Mat4, Mat4) {
        let view = self.layer_to_world(world_origin).inverse();
        let [w, h] = self.anchor_extent();
        let proj = Mat4::from_cols(
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -2.0 / h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
            Vec4::new(-1.0, 1.0, 0.5, 1.0),
        );
        (view, proj)
    }

    /// Where a ray from `origin` along `dir` crosses the layer, in layer
    /// units, and how far along the ray. `None` when the ray runs parallel to
    /// the plane or crosses it behind `origin`.
    pub fn ray_point(&self, origin: DVec3, dir: DVec3) -> Option<([f32; 2], f64)> {
        let n = self.plane.normal().as_dvec3();
        let denom = dir.dot(n);
        if denom.abs() < 1.0e-12 {
            return None;
        }
        let t = (self.elevation as f64 - origin.dot(n)) / denom;
        if !(t >= 0.0) {
            return None;
        }
        let p = self.plane.to_plane(origin + dir * t);
        Some((self.layer_point([p.x as f32, p.y as f32]), t))
    }

    /// The topmost shape or mesh of the frame under a ray from `origin` along
    /// `dir`, tested on the CPU in layer units with the items' own `contains`.
    ///
    /// Topmost is the highest `z_order`; at an equal `z_order` meshes and
    /// vector shapes are above analytic shapes, and a later item is above an
    /// earlier one. Mask-only shapes are skipped, and anchors resolve as when
    /// the layer draws. Labels, glyph runs, polylines and retained groups are
    /// not hit-tested.
    pub fn pick(&self, origin: DVec3, dir: DVec3) -> Option<WorldOverlayHit> {
        let (point, t) = self.ray_point(origin, dir)?;
        let extent = self.anchor_extent();
        let (view, proj) = self.anchor_view_projection(DVec3::ZERO);
        // (z_order, rank, list, index): the largest is on top.
        let mut best: Option<((i32, u8, u8, usize), WorldOverlayItem)> = None;
        let mut offer = |key: (i32, u8, u8, usize), item| {
            if best.as_ref().is_none_or(|(k, _)| key > *k) {
                best = Some((key, item));
            }
        };
        for (i, s) in self.frame.shapes.iter().enumerate() {
            if s.provides_mask.is_some() {
                continue;
            }
            let Some(top_left) = s.resolve_top_left_at(extent, &view, &proj, DVec3::ZERO) else {
                continue;
            };
            let mut placed = s.clone();
            placed.transform.translate = top_left;
            if placed.contains(point) {
                let rank = u8::from(matches!(s.shape, OverlayShape::Vector { .. }));
                offer((s.z_order, rank, 0, i), WorldOverlayItem::Shape(i));
            }
        }
        for (i, m) in self.frame.meshes.iter().enumerate() {
            let Some(offset) = m.resolve_offset_at(extent, &view, &proj, DVec3::ZERO) else {
                continue;
            };
            let mut placed = m.clone();
            placed.transform.translate = offset;
            if placed.contains(point) {
                offer((m.z_order, 1, 1, i), WorldOverlayItem::Mesh(i));
            }
        }
        best.map(|(_, item)| WorldOverlayHit {
            item,
            layer_point: point,
            world_point: self.world_point(point),
            distance: t,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_points_run_down_the_plane() {
        let layer = WorldOverlayLayer::new(WorldPlane::Xy).with_origin([10.0, 5.0]);
        assert_eq!(layer.plane_point([2.0, 3.0]), [12.0, 2.0]);
        assert_eq!(layer.layer_point([12.0, 2.0]), [2.0, 3.0]);
        let w = layer.world_point([2.0, 3.0]);
        assert_eq!(w, DVec3::new(12.0, 2.0, 0.0));
    }

    #[test]
    fn the_xz_plane_puts_layer_y_down_world_z() {
        let layer = WorldOverlayLayer::new(WorldPlane::Xz).with_elevation(1.0);
        let w = layer.world_point([1.0, 2.0]);
        assert_eq!(w, DVec3::new(1.0, -1.0, -2.0));
        let m = layer.layer_to_world(DVec3::ZERO);
        let p = m * Vec4::new(1.0, 2.0, 0.0, 1.0);
        assert!((p.truncate().as_dvec3() - w).length() < 1e-6);
    }

    #[test]
    fn a_ray_down_the_view_axis_crosses_at_its_plane_point() {
        let layer = WorldOverlayLayer::new(WorldPlane::Xy)
            .with_origin([0.0, 10.0])
            .with_elevation(0.5);
        let (p, t) = layer
            .ray_point(DVec3::new(3.0, 4.0, 10.0), DVec3::NEG_Z)
            .unwrap();
        assert_eq!(p, [3.0, 6.0]);
        assert!((t - 9.5).abs() < 1e-9);
        assert!(
            layer
                .ray_point(DVec3::new(0.0, 0.0, 10.0), DVec3::Z)
                .is_none()
        );
    }

    #[test]
    fn viewport_anchors_resolve_against_the_layer_rectangle() {
        let layer = WorldOverlayLayer::new(WorldPlane::Xy)
            .with_origin([4.0, 4.0])
            .with_size([8.0, 2.0]);
        let (view, proj) = layer.anchor_view_projection(DVec3::ZERO);
        let anchor = OverlayOrigin::World(DVec3::new(6.0, 3.0, 0.0));
        let at =
            resolve_anchor_origin_at(&anchor, layer.anchor_extent(), &view, &proj, DVec3::ZERO)
                .unwrap();
        assert!(
            (at[0] - 2.0).abs() < 1e-4 && (at[1] - 1.0).abs() < 1e-4,
            "{at:?}"
        );
    }

    #[test]
    fn pick_finds_the_topmost_shape_under_the_ray() {
        let mut frame = OverlayFrame::default();
        frame.shapes.push(OverlayShapeItem {
            transform: OverlayTransform::at([0.0, 0.0]),
            size: [4.0, 4.0],
            shape: OverlayShape::Rect { corner_radius: 0.0 },
            ..Default::default()
        });
        frame.shapes.push(OverlayShapeItem {
            transform: OverlayTransform::at([1.0, 1.0]),
            size: [2.0, 2.0],
            shape: OverlayShape::Circle,
            z_order: 1,
            ..Default::default()
        });
        let layer = WorldOverlayLayer::new(WorldPlane::Xy)
            .with_origin([0.0, 4.0])
            .with_frame(frame);
        // Layer (2, 2) is plane (2, 2): inside both, the circle on top.
        let hit = layer.pick(DVec3::new(2.0, 2.0, 5.0), DVec3::NEG_Z).unwrap();
        assert_eq!(hit.item, WorldOverlayItem::Shape(1));
        assert_eq!(hit.layer_point, [2.0, 2.0]);
        // Layer (0.5, 0.5): the rect only.
        let hit = layer.pick(DVec3::new(0.5, 3.5, 5.0), DVec3::NEG_Z).unwrap();
        assert_eq!(hit.item, WorldOverlayItem::Shape(0));
        assert!(
            layer
                .pick(DVec3::new(6.0, 2.0, 5.0), DVec3::NEG_Z)
                .is_none()
        );
    }
}
