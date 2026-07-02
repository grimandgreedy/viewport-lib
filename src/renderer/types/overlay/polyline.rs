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
        let n = samples.max(1);
        let points: Vec<[f32; 2]> = (0..=n).map(|i| path(i as f32 / n as f32)).collect();
        Self {
            points,
            thickness,
            colour,
            ..Default::default()
        }
    }
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
