//! Polyline overlay geometry and its join and cap styles.

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

/// The stroke of an [`OverlayPolylineItem`]: everything about the line itself.
///
/// This is the item's own geometry, not a band drawn around some other shape,
/// so it lives here rather than on [`OverlayStyle`]. An overlay shape has no
/// stroke: an edge band on a shape is a [`ShadowLayer`] with no blur and a
/// spread.
///
/// [`OverlayStyle`]: crate::overlay::OverlayStyle
/// [`ShadowLayer`]: crate::overlay::ShadowLayer
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct OverlayStroke {
    /// Stroke width in logical pixels.
    pub width: f32,
    /// RGBA colour in linear float format.
    pub colour: crate::colour::Colour,
    /// Solid, dashed, or dotted.
    pub pattern: StrokePattern,
    /// How segment joints are drawn.
    pub join: LineJoin,
    /// Mitre limit: when the mitre extension exceeds this multiple of `width`,
    /// the joint auto-falls back to a bevel.
    pub mitre_limit: f32,
    /// End-cap style for open polylines and dash ends. Closed solid polylines
    /// have no free ends, so caps are ignored there.
    pub cap: PolylineCap,
}

impl Default for OverlayStroke {
    fn default() -> Self {
        Self {
            width: 2.0,
            colour: [1.0, 1.0, 1.0, 1.0].into(),
            pattern: StrokePattern::Solid,
            join: LineJoin::Mitre,
            mitre_limit: 4.0,
            cap: PolylineCap::Butt,
        }
    }
}

impl OverlayStroke {
    /// A solid stroke of `width` logical pixels in `colour`, with the default
    /// join, mitre limit, and cap.
    pub fn new(width: f32, colour: impl Into<crate::colour::Colour>) -> Self {
        Self {
            width,
            colour: colour.into(),
            ..Default::default()
        }
    }

    /// Set the dash or dot pattern.
    pub fn with_pattern(mut self, pattern: StrokePattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set how segment joints are drawn.
    pub fn with_join(mut self, join: LineJoin) -> Self {
        self.join = join;
        self
    }

    /// Set the mitre limit as a multiple of `width`.
    pub fn with_mitre_limit(mut self, mitre_limit: f32) -> Self {
        self.mitre_limit = mitre_limit;
        self
    }

    /// Set the end-cap style.
    pub fn with_cap(mut self, cap: PolylineCap) -> Self {
        self.cap = cap;
        self
    }
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
    /// Translate, rotate, and scale, in logical pixels and radians.
    ///
    /// `translate` is the nudge from the resolved `anchor` origin, so with the
    /// default anchor and alignment it is the absolute screen placement.
    /// Rotation turns the item inside its extent box, which stays
    /// axis-aligned, so `align_x` / `align_y` place the unrotated box and the
    /// content turns within it. See [`OverlayTransform`] for how an item's
    /// transform composes with the transform of a retained group containing
    /// it.
    pub transform: OverlayTransform,
    /// Baked appearance: fill, shadow layers, and backdrop effects.
    ///
    /// Shared across the overlay item types, so a field can be present and
    /// inert here. Ask
    /// [`OverlayStyleSupport`](crate::overlay::OverlayStyleSupport) for what
    /// this family draws.
    pub style: OverlayStyle,
    /// Animation tracks resolved each frame against `OverlayFrame::time`.
    ///
    /// Boxed and `None` for a static item: the track block is several times
    /// the size of the rest of the item, so only items that animate pay for
    /// it. See [`OverlayAnimations`] for why the channel list is what it is.
    pub animations: Option<Box<OverlayAnimations>>,
    /// Per-frame colour multiplier applied to the whole item, identity
    /// `[1, 1, 1, 1]`. Composes multiplicatively with the item's own colours
    /// and with the tint of a retained group containing it. Never reaches
    /// shadow layers, on any path: a compiled group's shadow colours are
    /// baked, so honouring it here would make the same content look different
    /// on the two paths.
    pub tint: [f32; 4],
    /// What this item is clipped to: an axis-aligned box, a mask shape, or
    /// both. The default clips nothing.
    pub clip: OverlayClip,
    /// How the path's bounding box sits horizontally on `anchor` + `position`.
    /// Default `Left` leaves the points as authored.
    pub align_x: AnchorX,
    /// How the path's bounding box sits vertically on `anchor` + `position`.
    /// Default `Top` leaves the points as authored.
    pub align_y: AnchorY,
    /// The line itself: width, colour, pattern, joins and caps.
    ///
    /// `None` draws no line at all, which is what a closed polyline with a
    /// fill and no outline wants.
    pub stroke: Option<OverlayStroke>,
    /// When `true`, the last point connects back to the first.
    pub closed: bool,
    /// Optional per-point UVs for textured interiors.
    ///
    /// When set, this must have the same length as `points`. Otherwise the
    /// renderer falls back to bounds-mapped UVs.
    pub uvs: Option<Vec<[f32; 2]>>,
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
            transform: OverlayTransform::IDENTITY,
            style: OverlayStyle::default(),
            animations: None,
            tint: [1.0, 1.0, 1.0, 1.0],
            clip: OverlayClip::default(),
            align_x: AnchorX::Left,
            align_y: AnchorY::Top,
            stroke: Some(OverlayStroke::default()),
            closed: false,
            uvs: None,
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
        self.transform.translate = position;
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
            origin[0] + self.transform.translate[0] + self.align_x.align_shift(max_x - min_x),
            origin[1] + self.transform.translate[1] + self.align_y.align_shift(max_y - min_y),
        ])
    }

    /// Set the whole stroke at once.
    pub fn with_stroke(mut self, stroke: OverlayStroke) -> Self {
        self.stroke = Some(stroke);
        self
    }

    /// Draw no line, leaving only the interior fill of a closed polyline.
    pub fn without_stroke(mut self) -> Self {
        self.stroke = None;
        self
    }

    /// The stroke, inserting the default one if the item currently has none, so
    /// a single-field setter can be called on a bare item.
    fn stroke_mut(&mut self) -> &mut OverlayStroke {
        self.stroke.get_or_insert_with(OverlayStroke::default)
    }

    /// Set the stroke width in logical pixels.
    pub fn with_thickness(mut self, thickness: f32) -> Self {
        self.stroke_mut().width = thickness;
        self
    }

    /// Set the stroke colour.
    pub fn with_colour(mut self, colour: impl Into<crate::colour::Colour>) -> Self {
        self.stroke_mut().colour = colour.into();
        self
    }

    /// Set how segment joints are drawn.
    pub fn with_join(mut self, join: LineJoin) -> Self {
        self.stroke_mut().join = join;
        self
    }

    /// Set the mitre limit as a multiple of the stroke width before a mitre
    /// joint falls back to a bevel.
    pub fn with_mitre_limit(mut self, mitre_limit: f32) -> Self {
        self.stroke_mut().mitre_limit = mitre_limit;
        self
    }

    /// Set the end-cap style for open polylines and dash ends.
    pub fn with_cap(mut self, cap: PolylineCap) -> Self {
        self.stroke_mut().cap = cap;
        self
    }

    /// Set the stroke pattern (solid, dashed, or dotted).
    pub fn with_stroke_pattern(mut self, stroke_pattern: StrokePattern) -> Self {
        self.stroke_mut().pattern = stroke_pattern;
        self
    }

    /// Set whether the last point connects back to the first.
    pub fn with_closed(mut self, closed: bool) -> Self {
        self.closed = closed;
        self
    }

    /// Set the interior fill. Only used when the polyline is closed.
    pub fn with_fill(mut self, fill: OverlayFill) -> Self {
        self.style.fill = fill;
        self
    }

    /// Set the interior texture fill. Only used when the polyline is closed.
    /// Replaces any colour or gradient fill: an item has one fill.
    pub fn with_texture(mut self, texture: OverlayTextureId) -> Self {
        self.style.fill = OverlayFill::texture(texture);
        self
    }

    /// Set per-point UVs for a textured interior. Must have one entry per point.
    pub fn with_uvs(mut self, uvs: Vec<[f32; 2]>) -> Self {
        self.uvs = Some(uvs);
        self
    }

    /// Set the affine transform applied to texture UVs before sampling. Call
    /// it after [`with_texture`](Self::with_texture): a polyline with no
    /// texture fill has nothing to sample and is left alone.
    pub fn with_texture_transform(mut self, texture_transform: TextureTransform) -> Self {
        if let OverlayFill::Texture { transform, .. } = &mut self.style.fill {
            *transform = texture_transform;
        }
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

    /// Clip the path to the mask shape with this id (registered via
    /// [`OverlayShapeItem::with_clip_mask`](crate::overlay::OverlayShapeItem::with_clip_mask)).
    /// Fragments outside the mask are discarded, so a path drawn inside a
    /// scrolling region is contained.
    pub fn with_clip(mut self, clip_id: u32) -> Self {
        self.clip.mask = Some(clip_id);
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
        colour: impl Into<crate::colour::Colour>,
    ) -> Self {
        Self {
            points: sample_open_path(path, samples),
            stroke: Some(OverlayStroke::new(thickness, colour)),
            ..Default::default()
        }
    }

    /// Construct a closed, filled polygon by sampling `path` at `samples + 1`
    /// evenly-spaced values across `[0, 1)`. The last sample stops short of
    /// `t = 1` so the closing segment (last point back to the first) does not
    /// double up the start point.
    ///
    /// Sets `closed = true` and applies the given fill and stroke. Pass
    /// [`OverlayFill::none`] to draw the outline only.
    pub fn closed_from_path(
        path: impl Fn(f32) -> [f32; 2],
        samples: u32,
        fill: OverlayFill,
        stroke_colour: impl Into<crate::colour::Colour>,
        thickness: f32,
    ) -> Self {
        Self {
            points: sample_closed_path(path, samples),
            stroke: Some(OverlayStroke::new(thickness, stroke_colour)),
            closed: true,
            style: OverlayStyle {
                fill,
                ..Default::default()
            },
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

    /// Set the stacked shadow layers drawn behind this item.
    pub fn with_shadows(mut self, shadows: Vec<crate::overlay::ShadowLayer>) -> Self {
        self.style.shadows = shadows;
        self
    }

    /// Add one shadow layer, in front of any already set.
    pub fn with_shadow(mut self, shadow: crate::overlay::ShadowLayer) -> Self {
        self.style.shadows.push(shadow);
        self
    }

    /// Add a contour of `width` logical pixels in `colour` behind this item.
    /// Shorthand for pushing a [`ShadowLayer::outline`].
    ///
    /// [`ShadowLayer::outline`]: crate::overlay::ShadowLayer::outline
    pub fn with_outline(mut self, colour: impl Into<crate::colour::Colour>, width: f32) -> Self {
        self.style
            .shadows
            .push(crate::overlay::ShadowLayer::outline(colour, width));
        self
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
        let fill = OverlayFill::Solid([0.2, 0.4, 0.6, 1.0].into());
        let item = OverlayPolylineItem::closed_from_path(circle, 4, fill.clone(), [1.0; 4], 3.0);
        assert!(item.closed);
        assert_eq!(item.style.fill, fill);
        assert_eq!(item.stroke.as_ref().unwrap().width, 3.0);
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

impl OverlayPolylineItem {
    /// Set the transform: translate, rotate, scale, and pivot at once.
    pub fn with_transform(mut self, transform: OverlayTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Set the rotation in radians about the pivot.
    ///
    /// The points are the caller's, so a consumer drawing immediately could
    /// always rotate them itself. Under retention it could not: the points are
    /// baked into the compiled buffer, and turning them means re-compiling.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.transform.rotation = radians;
        self
    }

    /// Set the point to rotate and scale around, in logical pixels from the
    /// centre of the path's bounding box.
    pub fn with_rotation_pivot(mut self, pivot: [f32; 2]) -> Self {
        self.transform.pivot = pivot;
        self
    }

    /// Set the uniform scale about the transform pivot.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.transform.scale = scale;
        self
    }

    /// Set the per-frame colour multiplier (identity `[1, 1, 1, 1]`).
    pub fn with_tint(mut self, tint: [f32; 4]) -> Self {
        self.tint = tint;
        self
    }

    /// Set the animation tracks.
    pub fn with_animations(mut self, animations: OverlayAnimations) -> Self {
        self.animations = Some(Box::new(animations));
        self
    }

    /// Clip to an axis-aligned box in logical pixels `[x0, y0, x1, y1]`.
    pub fn with_clip_rect(mut self, clip_rect: [f32; 4]) -> Self {
        self.clip.rect = Some(clip_rect);
        self
    }
}
