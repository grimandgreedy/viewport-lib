use crate::overlay::*;

/// How an [`OverlayPolylineItem`] handles each joint between segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LineJoin {
    /// Mitre join: extend both segment edges until they meet. Falls back to
    /// `Bevel` automatically when the join would exceed `mitre_limit`.
    /// Default.
    #[default]
    Mitre,
    /// Bevel join: cut the outer corner flat between the two segments.
    Bevel,
}

/// End-cap style for open [`OverlayPolylineItem`] strokes.
///
/// Also applies to the ends of each dash when the stroke pattern is
/// [`StrokePattern::Dashed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PolylineCap {
    /// Flat cut flush with the endpoint. Default.
    #[default]
    Butt,
    /// Flat cut extended `thickness / 2` beyond the endpoint.
    Square,
    /// Semicircular cap centred on the endpoint.
    Round,
}

/// Stroke pattern for a polyline.
///
/// Dash and dot placement is measured in accumulated arc length along the
/// path. The unit depends on the item: logical pixels for the screen-space
/// [`OverlayPolylineItem`], world-space units for the 3-D
/// [`OverlayPolylineItem`](crate::overlay::OverlayPolylineItem) (measured along the
/// line in its input space). On closed overlay polylines the pattern continues
/// across the final-to-first segment.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StrokePattern {
    /// Continuous stroke (default).
    #[default]
    Solid,
    /// Dashes of `dash_length` pixels separated by `gap_length` pixels.
    Dashed {
        /// Length of each visible dash, in pixels of arc length.
        dash_length: f32,
        /// Length of each gap between dashes, in pixels of arc length.
        gap_length: f32,
        /// Shifts the pattern backwards along the path; animating this
        /// upwards produces a marching-ants effect.
        offset: f32,
    },
    /// Discs of the stroke thickness placed along the path.
    Dotted {
        /// Distance between dot centres, in pixels of arc length.
        spacing: f32,
        /// Shifts the first dot along the path.
        offset: f32,
    },
}

/// A stroked polyline rendered as a screen-space overlay.
///
/// Constructed from a list of waypoints in logical pixels. Tessellated on
/// the CPU into a triangle list each frame; rendered through the same
/// pipeline as overlay rects and labels (no SDF, no shader changes).
///
/// Use `OverlayPolylineItem::from_path` to construct from a closure that
/// samples a curve at N points (Bezier traces, lissajous, custom paths).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OverlayPolylineItem {
    /// Waypoints in logical pixels, relative to the resolved `anchor` origin.
    /// With the default anchor (viewport top-left) and a zero `position` these
    /// are absolute screen coordinates.
    pub points: Vec<[f32; 2]>,
    /// Origin the path hangs from: a viewport corner (default top-left) or a
    /// projected world point. Every point in `points` is relative to this.
    pub anchor: OverlayAnchor,
    /// Placement in logical pixels relative to the resolved `anchor` origin,
    /// added to every point. Default: `[0.0, 0.0]`.
    pub position: [f32; 2],
    /// How the path's bounding box sits horizontally on `anchor` + `position`.
    /// Default `Left` leaves the points as authored.
    pub align_x: AnchorX,
    /// How the path's bounding box sits vertically on `anchor` + `position`.
    /// Default `Top` leaves the points as authored.
    pub align_y: AnchorY,
    /// Stroke thickness in logical pixels.
    pub thickness: f32,
    /// RGBA colour in linear float format.
    pub colour: [f32; 4],
    /// How segment joints are drawn.
    pub join: LineJoin,
    /// Mitre limit: when the mitre extension exceeds this multiple of
    /// `thickness`, the joint auto-falls back to a bevel.
    pub mitre_limit: f32,
    /// End-cap style for open polylines and dash ends. Closed solid
    /// polylines have no free ends, so caps are ignored there.
    pub cap: PolylineCap,
    /// Solid, dashed, or dotted stroke.
    pub stroke_pattern: StrokePattern,
    /// When `true`, the last point connects back to the first.
    pub closed: bool,
    /// Optional interior fill. Only used when `closed` is `true`.
    ///
    /// Texture fills use this the same way [`OverlayShapeItem`] does: when
    /// `texture` is set, `OverlayFill::Solid` acts as a tint. Gradient fills
    /// are ignored for textured interiors.
    pub fill: Option<OverlayFill>,
    /// Optional texture fill for the interior. Only used when `closed` is `true`.
    ///
    /// The polygon is clipped by triangulating the closed path. UVs are
    /// derived from the path bounds unless `uvs` has one entry per point.
    pub texture: Option<OverlayTextureId>,
    /// Optional per-point UVs for textured interiors.
    ///
    /// When set, this must have the same length as `points`. Otherwise the
    /// renderer falls back to bounds-mapped UVs.
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Affine transform applied to texture UVs before sampling.
    pub texture_transform: TextureTransform,
    /// Overall opacity multiplier in `[0, 1]`.
    pub opacity: f32,
    /// Draw order relative to other overlay rects, polylines, and labels.
    /// Lower values render first (further back).
    pub z_order: i32,
}

impl Default for OverlayPolylineItem {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            anchor: OverlayAnchor::default(),
            position: [0.0, 0.0],
            align_x: AnchorX::Left,
            align_y: AnchorY::Top,
            thickness: 2.0,
            colour: [1.0, 1.0, 1.0, 1.0],
            join: LineJoin::Mitre,
            mitre_limit: 4.0,
            cap: PolylineCap::Butt,
            stroke_pattern: StrokePattern::Solid,
            closed: false,
            fill: None,
            texture: None,
            uvs: None,
            texture_transform: TextureTransform::default(),
            opacity: 1.0,
            z_order: 0,
        }
    }
}

impl OverlayPolylineItem {
    /// Create a polyline from `points` (waypoints in logical pixels from the
    /// viewport top-left). All other fields take their defaults; set them with
    /// the `with_*` methods below.
    pub fn new(points: Vec<[f32; 2]>) -> Self {
        Self {
            points,
            ..Default::default()
        }
    }

    /// Set the origin the path hangs from (a viewport corner or a world point).
    pub fn with_anchor(mut self, anchor: OverlayAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Pin the path to a world-space position, reprojected each frame. Sugar for
    /// `with_anchor(OverlayAnchor::World(pos))`.
    pub fn with_world_anchor(mut self, pos: [f32; 3]) -> Self {
        self.anchor = OverlayAnchor::World(pos);
        self
    }

    /// Set the placement in logical pixels relative to the resolved anchor
    /// origin, added to every point.
    pub fn with_position(mut self, position: [f32; 2]) -> Self {
        self.position = position;
        self
    }

    /// Set how the path's bounding box aligns onto the resolved anchor origin.
    pub fn with_align(mut self, align_x: AnchorX, align_y: AnchorY) -> Self {
        self.align_x = align_x;
        self.align_y = align_y;
        self
    }

    /// Resolve the screen-pixel offset added to every point for a frame: the
    /// `anchor` origin, plus `position`, shifted by `align_x` / `align_y` for the
    /// path's bounding box. Returns `None` when a `World` anchor projects behind
    /// the camera or off-screen (the path is skipped that frame). The default
    /// anchor with a zero `position` and `Left` / `Top` alignment resolves to
    /// `[0, 0]`, so absolute points are drawn unchanged.
    pub fn resolve_offset(
        &self,
        viewport_size: [f32; 2],
        view: &glam::Mat4,
        proj: &glam::Mat4,
    ) -> Option<[f32; 2]> {
        let origin = resolve_anchor_origin(&self.anchor, viewport_size, view, proj)?;
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (0.0, 0.0, 0.0, 0.0);
        if let Some((first, rest)) = self.points.split_first() {
            min_x = first[0];
            min_y = first[1];
            max_x = first[0];
            max_y = first[1];
            for p in rest {
                min_x = min_x.min(p[0]);
                min_y = min_y.min(p[1]);
                max_x = max_x.max(p[0]);
                max_y = max_y.max(p[1]);
            }
        }
        Some([
            origin[0] + self.position[0] + self.align_x.align_shift(max_x - min_x),
            origin[1] + self.position[1] + self.align_y.align_shift(max_y - min_y),
        ])
    }

    /// Set the stroke thickness in logical pixels.
    pub fn with_thickness(mut self, thickness: f32) -> Self {
        self.thickness = thickness;
        self
    }

    /// Set the stroke colour.
    pub fn with_colour(mut self, colour: [f32; 4]) -> Self {
        self.colour = colour;
        self
    }

    /// Set how segment joints are drawn.
    pub fn with_join(mut self, join: LineJoin) -> Self {
        self.join = join;
        self
    }

    /// Set the mitre limit as a multiple of `thickness` before a mitre joint
    /// falls back to a bevel.
    pub fn with_mitre_limit(mut self, mitre_limit: f32) -> Self {
        self.mitre_limit = mitre_limit;
        self
    }

    /// Set the end-cap style for open polylines and dash ends.
    pub fn with_cap(mut self, cap: PolylineCap) -> Self {
        self.cap = cap;
        self
    }

    /// Set the stroke pattern (solid, dashed, or dotted).
    pub fn with_stroke_pattern(mut self, stroke_pattern: StrokePattern) -> Self {
        self.stroke_pattern = stroke_pattern;
        self
    }

    /// Set whether the last point connects back to the first.
    pub fn with_closed(mut self, closed: bool) -> Self {
        self.closed = closed;
        self
    }

    /// Set the interior fill. Only used when the polyline is closed.
    pub fn with_fill(mut self, fill: OverlayFill) -> Self {
        self.fill = Some(fill);
        self
    }

    /// Set the interior texture fill. Only used when the polyline is closed.
    pub fn with_texture(mut self, texture: OverlayTextureId) -> Self {
        self.texture = Some(texture);
        self
    }

    /// Set per-point UVs for a textured interior. Must have one entry per point.
    pub fn with_uvs(mut self, uvs: Vec<[f32; 2]>) -> Self {
        self.uvs = Some(uvs);
        self
    }

    /// Set the affine transform applied to texture UVs before sampling.
    pub fn with_texture_transform(mut self, texture_transform: TextureTransform) -> Self {
        self.texture_transform = texture_transform;
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set the draw order. Lower values render first (further back).
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Construct a polyline by sampling the given closure at `samples + 1`
    /// evenly-spaced parameter values across `[0, 1]`. The closure is called
    /// once per sample at construction time; the resulting points are stored
    /// in `self.points`.
    ///
    /// Consumers wanting non-uniform sample density (denser around tight
    /// curvature) should sample manually and build the item via the regular
    /// struct literal.
    pub fn from_path(
        path: impl Fn(f32) -> [f32; 2],
        samples: u32,
        thickness: f32,
        colour: [f32; 4],
    ) -> Self {
        Self {
            points: sample_open_path(path, samples),
            thickness,
            colour,
            ..Default::default()
        }
    }

    /// Construct a closed, filled polygon by sampling `path` at `samples + 1`
    /// evenly-spaced values across `[0, 1)`. The last sample stops short of
    /// `t = 1` so the closing segment (last point back to the first) does not
    /// double up the start point.
    ///
    /// Sets `closed = true` and applies the given fill and stroke. Pass
    /// `None` for `fill` to draw the outline only.
    pub fn closed_from_path(
        path: impl Fn(f32) -> [f32; 2],
        samples: u32,
        fill: Option<OverlayFill>,
        stroke_colour: [f32; 4],
        thickness: f32,
    ) -> Self {
        Self {
            points: sample_closed_path(path, samples),
            thickness,
            colour: stroke_colour,
            closed: true,
            fill,
            ..Default::default()
        }
    }

    /// Replace `points` by resampling `path`. Honours the item's `closed`
    /// flag: closed items sample `[0, 1)` so the closing segment is not
    /// duplicated, open items sample `[0, 1]` inclusive.
    ///
    /// Call this during frame building to animate a function-generated path.
    /// Nothing else on the item changes, and no renderer state is cached, so
    /// the resampled points take effect on the next prepared frame.
    pub fn set_points_from_path(&mut self, path: impl Fn(f32) -> [f32; 2], samples: u32) {
        self.points = if self.closed {
            sample_closed_path(path, samples)
        } else {
            sample_open_path(path, samples)
        };
    }
}

/// Sample `path` at `samples + 1` values across `[0, 1]` inclusive.
fn sample_open_path(path: impl Fn(f32) -> [f32; 2], samples: u32) -> Vec<[f32; 2]> {
    let n = samples.max(1);
    (0..=n).map(|i| path(i as f32 / n as f32)).collect()
}

/// Sample `path` at `samples + 1` values across `[0, 1)`. The final sample
/// lands at `samples / (samples + 1)`, so a closed polyline's wrap-around
/// segment does not repeat the start point.
fn sample_closed_path(path: impl Fn(f32) -> [f32; 2], samples: u32) -> Vec<[f32; 2]> {
    let n = samples.max(1);
    let divisor = (n + 1) as f32;
    (0..=n).map(|i| path(i as f32 / divisor)).collect()
}

/// End-cap style for `OverlayShape::Line`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LineCap {
    /// Round end caps (default). The stroke ends in a semicircle.
    #[default]
    Round,
    /// Square end caps. The stroke ends in a flat perpendicular cut flush
    /// with the segment endpoint (no extension beyond the endpoint).
    Square,
}

#[cfg(test)]
mod path_sample_tests {
    use super::*;

    // A unit circle: path(0) == path(1), so an inclusive sample would repeat
    // the start point at the end.
    fn circle(t: f32) -> [f32; 2] {
        let a = t * std::f32::consts::TAU;
        [a.cos(), a.sin()]
    }

    #[test]
    fn from_path_samples_endpoint_inclusive() {
        let item = OverlayPolylineItem::from_path(circle, 4, 2.0, [1.0; 4]);
        // 5 points, first and last both at the t=0 position.
        assert_eq!(item.points.len(), 5);
        assert!((item.points[0][0] - item.points[4][0]).abs() < 1e-5);
        assert!((item.points[0][1] - item.points[4][1]).abs() < 1e-5);
    }

    #[test]
    fn closed_from_path_skips_duplicate_endpoint() {
        let fill = Some(OverlayFill::Solid([0.2, 0.4, 0.6, 1.0]));
        let item = OverlayPolylineItem::closed_from_path(circle, 4, fill.clone(), [1.0; 4], 3.0);
        assert!(item.closed);
        assert_eq!(item.fill, fill);
        assert_eq!(item.thickness, 3.0);
        // 5 points spanning [0, 1); the last is at 4/5, not back at the start.
        assert_eq!(item.points.len(), 5);
        assert!((item.points[0][0] - item.points[4][0]).abs() > 1e-3);
        let expected_last = circle(4.0 / 5.0);
        assert!((item.points[4][0] - expected_last[0]).abs() < 1e-5);
    }

    #[test]
    fn resolve_offset_default_is_zero() {
        // Default anchor + zero position + Left/Top align resolves to no offset,
        // so absolute points draw unchanged, with no camera needed.
        let p = OverlayPolylineItem::new(vec![[10.0, 20.0], [30.0, 40.0]]);
        let off = p
            .resolve_offset([800.0, 600.0], &glam::Mat4::IDENTITY, &glam::Mat4::IDENTITY)
            .unwrap();
        assert_eq!(off, [0.0, 0.0]);
    }

    #[test]
    fn resolve_offset_pins_bottom_right() {
        // A path spanning [0,0]..[40,20], anchored and aligned bottom-right, has
        // its bounding-box bottom-right corner pinned to the viewport corner.
        let p = OverlayPolylineItem::new(vec![[0.0, 0.0], [40.0, 20.0]])
            .with_anchor(OverlayAnchor::Viewport {
                x: AnchorX::Right,
                y: AnchorY::Bottom,
            })
            .with_align(AnchorX::Right, AnchorY::Bottom);
        let off = p
            .resolve_offset([800.0, 600.0], &glam::Mat4::IDENTITY, &glam::Mat4::IDENTITY)
            .unwrap();
        // origin [800, 600], align shifts by -(width), -(height) = -40, -20.
        assert_eq!(off, [760.0, 580.0]);
    }

    #[test]
    fn set_points_from_path_follows_closed_flag() {
        let mut open = OverlayPolylineItem {
            closed: false,
            ..Default::default()
        };
        open.set_points_from_path(circle, 4);
        assert!((open.points[0][0] - open.points[4][0]).abs() < 1e-5);

        let mut closed = OverlayPolylineItem {
            closed: true,
            ..Default::default()
        };
        closed.set_points_from_path(circle, 4);
        assert!((closed.points[0][0] - closed.points[4][0]).abs() > 1e-3);
    }
}
