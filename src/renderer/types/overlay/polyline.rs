use crate::renderer::types::*;

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

/// Stroke pattern for [`OverlayPolylineItem`].
///
/// Dash and dot placement is measured in accumulated arc length along the
/// path, in logical pixels. On closed polylines the pattern continues across
/// the final-to-first segment.
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
pub struct OverlayPolylineItem {
    /// Waypoints in logical pixels from the viewport top-left.
    pub points: Vec<[f32; 2]>,
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
