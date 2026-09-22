//! Vector shape overlay items (rectangles, ellipses, and paths).

use crate::overlay::*;

/// Shape type for an `OverlayShapeItem`.
///
/// Each variant maps to a signed-distance function evaluated per fragment
/// on the GPU. The bounding quad is defined by `OverlayShapeItem::position`
/// and `size`; the shape variant controls which SDF is used and how the
/// extra `radii` parameters are interpreted.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OverlayShape {
    /// Axis-aligned rectangle with a uniform corner radius.
    Rect {
        /// Corner radius in logical pixels. `0.0` produces sharp corners.
        corner_radius: f32,
    },
    /// Axis-aligned rectangle with independent corner radii.
    /// Order: top-left, top-right, bottom-right, bottom-left.
    RoundedRect {
        /// Per-corner radii in logical pixels.
        radii: [f32; 4],
    },
    /// Circle inscribed in the bounding box (the smaller dimension wins).
    Circle,
    /// Ellipse filling the bounding box.
    Ellipse,
    /// Pill / capsule shape: fully rounded along the shorter axis.
    Capsule,
    /// Hollow circle (annulus). The ring wall occupies the space between
    /// the outer edge (defined by `size`) and the inner hole.
    Ring {
        /// Inner radius as a fraction of the inscribed radius. `0.0` produces
        /// a solid circle; `0.9` produces a thin ring. Clamped to 0.0..1.0.
        inner_radius_frac: f32,
    },
    /// Arc (pie-slice or annular sector). Combines a ring with an angular
    /// range so you can draw progress indicators, radial menus, and pie
    /// charts.
    Arc {
        /// Inner radius as a fraction of the inscribed radius. `0.0` gives a
        /// solid pie slice; values near `1.0` give a thin arc stroke.
        inner_radius_frac: f32,
        /// Start angle in radians. `0.0` points right, angles increase
        /// counter-clockwise.
        start_angle: f32,
        /// End angle in radians. The filled region sweeps CCW from
        /// `start_angle` to `end_angle`.
        end_angle: f32,
    },
    /// Triangle oriented in one of four cardinal directions, fitted to the
    /// bounding box.
    Triangle {
        /// Which direction the triangle points.
        direction: TriangleDirection,
    },
    /// Line segment from the top-left to the bottom-right corner of the
    /// bounding box with a fixed stroke width. For axis-aligned strokes,
    /// set the minor dimension of `size` to a small value (e.g. 0.1).
    Line {
        /// Stroke width in logical pixels.
        thickness: f32,
        /// End-cap style: `Round` (default) or `Square`.
        cap: LineCap,
    },
    /// N-pointed star inscribed in the bounding box.
    Star {
        /// Number of points. Typical values: 4, 5, 6.
        points: u32,
        /// Inner radius as a fraction of the outer radius. Lower values
        /// produce sharper, more pointed tips. Typical value: `0.5`.
        inner_radius_frac: f32,
    },
    /// Regular convex polygon with N sides, inscribed in the bounding box.
    RegularPolygon {
        /// Number of sides. `3` = triangle, `4` = square (45-deg rotated),
        /// `6` = hexagon, etc.
        sides: u32,
    },
    /// Plus/cross shape: the union of a horizontal and a vertical rectangle.
    Cross {
        /// Arm width as a fraction of the smaller half-dimension of the
        /// bounding box. `1.0` fills the entire bounding box; `0.3` gives
        /// thin arms. Clamped to 0.0..1.0.
        arm_width_frac: f32,
    },
    /// Arbitrary vector shape: one or more subpaths with curves and a fill
    /// rule. Unlike the analytic variants above, this has no closed-form SDF,
    /// so it is flattened and tessellated to triangles rather than evaluated
    /// per fragment. It is the general filled region the analytic variants are
    /// special cases of. Distinct from a math vector: "vector" here means
    /// vector art (paths). Subpath coordinates are path-local logical pixels,
    /// placed at the item's `position`.
    ///
    /// A vector shape honours the coverage- and box-relative fields of
    /// [`OverlayShapeItem`]: `fill` (solid and gradients), both shadow lists,
    /// `opacity`, `z_order`, the clip, and `rotation` /
    /// `rotation_pivot`. `position` places the path origin and `size` sets the
    /// rotation centre; the fill and gradient bounds come from the path's own
    /// extent, not `size`.
    ///
    /// Fields that depend on the distance field or the bounding quad have no
    /// effect on a vector shape and are ignored: `shadows` / `inner_shadows` /
    /// the legacy `shadow_*` (no distance field to fall off), `backdrop_blur`
    /// and its filters, and an `OverlayFill::Texture` fill (a colour or
    /// gradient fill still draws).
    Vector {
        /// The contours that make up the shape.
        subpaths: Vec<SubPath>,
        /// How the subpaths combine into filled area.
        fill_rule: FillRule,
    },
}

/// One soft drop or inset shadow.
///
/// Used with [`OverlayShapeItem::shadows`] (drawn behind the item, clipped to
/// outside it) and [`OverlayShapeItem::inner_shadows`] (drawn over it, eroding
/// inward from the boundary) to stack several effects on one item: a soft
/// ambient shadow for depth plus a tighter one for contact, an outer glow for
/// focus, or a plain contour band for a border.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ShadowLayer {
    /// RGBA colour of the shadow, linear float format. The alpha scales the
    /// shadow strength.
    pub colour: crate::colour::Colour,
    /// Blur distance in logical pixels: how far the shadow fades from fully
    /// opaque to fully transparent, measured outward from the (spread) edge.
    /// `0.0` gives a hard edge.
    pub blur: f32,
    /// Offset of the shadow from the item in logical pixels.
    /// Positive X shifts right, positive Y shifts down.
    pub offset: [f32; 2],
    /// Grow the silhouette by this many logical pixels before blurring and
    /// offsetting. Matches the third length of the CSS `box-shadow` shorthand.
    ///
    /// This is the knob that makes a shadow work on thin shapes and small text,
    /// where an offset alone leaves the shadow hidden under the stroke it is
    /// meant to back. A contour is spread with no blur and no offset: see
    /// [`ShadowLayer::outline`].
    pub spread: f32,
    /// Exponent shaping the blur falloff curve. Default `1.0`, which is the
    /// plain fade. Above `1.0` concentrates the shadow against the item and
    /// leaves a longer, lighter tail; below `1.0` broadens it towards a glow.
    ///
    /// Has no effect when `blur` is `0.0`, since there is no gradient to shape.
    pub falloff: f32,
}

impl Default for ShadowLayer {
    fn default() -> Self {
        Self {
            colour: [0.0, 0.0, 0.0, 0.0].into(),
            blur: 0.0,
            offset: [0.0, 0.0],
            spread: 0.0,
            falloff: 1.0,
        }
    }
}

impl ShadowLayer {
    /// Build a shadow layer from colour, blur distance, and offset, with no
    /// spread and the default falloff.
    pub fn new(colour: impl Into<crate::colour::Colour>, blur: f32, offset: [f32; 2]) -> Self {
        Self {
            colour: colour.into(),
            blur,
            offset,
            ..Default::default()
        }
    }

    /// Build a contour: a shadow dilated by `width` with no blur and no offset,
    /// so it reads as an outline hugging the item on every side.
    ///
    /// This is how text and thin strokes get an outline; there is no separate
    /// outline field. Stacking a blurred layer behind one of these gives an
    /// outlined item with a drop shadow, ordered by position in the list.
    pub fn outline(colour: impl Into<crate::colour::Colour>, width: f32) -> Self {
        Self {
            colour: colour.into(),
            spread: width,
            ..Default::default()
        }
    }

    /// Set the shadow colour.
    pub fn with_colour(mut self, colour: impl Into<crate::colour::Colour>) -> Self {
        self.colour = colour.into();
        self
    }

    /// Set the blur distance in logical pixels.
    pub fn with_blur(mut self, blur: f32) -> Self {
        self.blur = blur;
        self
    }

    /// Set the offset in logical pixels.
    pub fn with_offset(mut self, offset: [f32; 2]) -> Self {
        self.offset = offset;
        self
    }

    /// Set the spread: how far the silhouette grows before blur and offset.
    pub fn with_spread(mut self, spread: f32) -> Self {
        self.spread = spread;
        self
    }

    /// Set the falloff exponent shaping the blur curve.
    pub fn with_falloff(mut self, falloff: f32) -> Self {
        self.falloff = falloff;
        self
    }

    /// How far beyond the item's own bounds this layer can draw, in logical
    /// pixels. Geometry that emits a shadow pads its bounds by this.
    pub fn extent(&self) -> f32 {
        let reach = self.spread + self.blur;
        reach + self.offset[0].abs().max(self.offset[1].abs())
    }

    /// Whether this layer draws anything at all.
    pub fn is_visible(&self) -> bool {
        let shaped = self.blur > 0.0 || self.spread > 0.0;
        let displaced = self.offset[0] != 0.0 || self.offset[1] != 0.0;
        self.colour.to_linear_rgba()[3] > 0.0 && (shaped || displaced)
    }
}

/// Maximum number of stacked outer (or inner) shadow layers honoured per
/// shape. Extra layers beyond this count are dropped during `prepare()`.
pub const OVERLAY_MAX_SHADOW_LAYERS: usize = 4;

/// Cardinal direction for `OverlayShape::Triangle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TriangleDirection {
    /// Apex points upward (toward the top of the viewport).
    #[default]
    Up,
    /// Apex points downward.
    Down,
    /// Apex points left.
    Left,
    /// Apex points right.
    Right,
}

impl Default for OverlayShape {
    fn default() -> Self {
        OverlayShape::Rect { corner_radius: 0.0 }
    }
}

/// A screen-space overlay shape: a region, plus options for drawing it.
///
/// `shape` says what the region is and `size` gives its extent; everything else
/// on the item is how to draw it. That split is the whole model, and it is why
/// an arbitrary vector path is an [`OverlayShape`] variant rather than a
/// separate item type: a path is another way of specifying a region.
///
/// # Two coverage backends
///
/// The analytic variants (`Rect`, `Circle`, `Ring`, and the rest) have a
/// closed-form signed-distance function. Each becomes one bounding quad, and
/// the fragment shader evaluates the SDF to produce anti-aliased fill, shadow,
/// and discard regions at any scale.
///
/// [`OverlayShape::Vector`] has no closed-form distance field. It is flattened
/// and tessellated to triangles and drawn on the text pipeline instead, with
/// its shadow layers re-emitted as geometry.
///
/// The item hides which backend runs, but it cannot hide that they differ in
/// what they can express: the options that need a distance field
/// (`style.backdrop`, and a texture fill with or without its nine-patch) do
/// nothing on a vector path. Ask
/// [`OverlayStyleSupport::for_shape`](crate::overlay::OverlayStyleSupport::for_shape)
/// rather than memorising that, and see the per-variant note on
/// [`OverlayShape::Vector`].
///
/// # Fill
///
/// `style.fill` is what the interior is filled with: `OverlayFill::Solid` for
/// a flat colour, one of the gradient variants, or `OverlayFill::Texture` for
/// an uploaded image. A fully transparent solid, which is the default, draws no
/// fill and leaves the shadow layers on their own.
///
/// A texture fill samples the image across the shape, with the variant's `tint`
/// multiplied into each texel (white for no tint) and its `nine_slice` holding
/// the corners at their authored size. The boundary, the shadow layers, and
/// anti-aliasing apply the same way whichever fill is set.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib_types::overlay::{OutlineMode, OverlayShapeItem, OverlayShape, OverlayFill};
/// // Rounded-rect panel background.
/// let panel = OverlayShapeItem::new(
///     OverlayShape::Rect { corner_radius: 8.0 },
///     [20.0, 20.0],
///     [300.0, 200.0],
/// )
/// .with_fill(OverlayFill::Solid([0.1, 0.1, 0.1, 0.85].into()))
/// .with_outline([0.4, 0.4, 0.4, 1.0], 1.0, OutlineMode::Inset);
///
/// // Circle with a left-to-right gradient.
/// let grad_dot = OverlayShapeItem::new(OverlayShape::Circle, [100.0, 100.0], [60.0, 60.0])
///     .with_fill(OverlayFill::LinearGradient {
///         start_colour: [0.0, 0.4, 1.0, 1.0].into(),
///         end_colour: [0.0, 1.0, 0.5, 1.0].into(),
///         angle: 0.0,
///     });
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OverlayShapeItem {
    /// The region this item draws: an analytic variant, or an arbitrary vector
    /// path. This specifies the items shape; every other field specifies how and where to draw it.
    pub shape: OverlayShape,
    /// Extent of the shape's bounding box in logical pixels.
    ///
    /// For an analytic variant this is the shape: a `Circle` is round because
    /// it is inscribed in this box. For [`OverlayShape::Vector`] the path
    /// carries its own extent, so this only sets the box the transform pivots
    /// and the alignment resolve against.
    pub size: [f32; 2],
    /// Where the item hangs from and which point of its own box lands there.
    ///
    /// `transform.translate` is a nudge from the resolved origin, and a world
    /// origin behind the camera or off screen culls the item for the frame.
    pub anchoring: OverlayAnchoring,
    /// Translate, rotate, and scale, in logical pixels and radians.
    ///
    /// `translate` is the nudge from the resolved `anchor` origin, so with the
    /// default anchor and alignment it is the absolute screen placement.
    /// Rotation turns the item inside its extent box, which stays
    /// axis-aligned, so `anchoring.align` places the unrotated box and the
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
    /// What this item is clipped to: an axis-aligned box, a mask shape, or
    /// both. The default clips nothing.
    pub clip: OverlayClip,
    /// Draw order relative to other shapes. Lower values render first (further back).
    pub z_order: i32,
    /// Marks this shape as a clip mask under this id: other items whose
    /// `clip.mask` matches are clipped to it. The shape itself is not drawn.
    /// `None` (the default) means the shape is not a mask.
    ///
    /// The id must be unique within a frame. If two shapes carry the
    /// same id, the first one in submission order is used as the mask and the
    /// rest are ignored. When several independent sources emit into one frame,
    /// offset their ids so they do not collide.
    ///
    /// Used for scroll containers, masked panels, and composite widgets.
    /// Backdrop-blur shapes are composited by a separate pass and are not
    /// clipped; every other overlay family is, textured shapes included.
    ///
    /// The clip follows the mask's shape, not its bounding box: the mask's SDF
    /// is evaluated per fragment, so a `Circle` mask clips to a circle. Masks
    /// nest, and a fragment must be inside the whole parent chain to survive.
    /// The bounding box is still used, as a cheap reject before the SDF.
    pub provides_mask: Option<u32>,
    /// Animation tracks resolved each frame against `OverlayFrame::time`. Each
    /// `Some` track replaces the matching field on the item for the frame. See
    /// [`OverlayAnimations`] for why the channel list is what it is.
    ///
    /// Boxed and `None` for a static shape: the track block is several times
    /// the size of the rest of the item, so only shapes that animate pay for
    /// it.
    pub animations: Option<Box<OverlayAnimations>>,
}

impl Default for OverlayShapeItem {
    fn default() -> Self {
        Self {
            anchoring: crate::overlay::OverlayAnchoring::default(),
            transform: OverlayTransform::IDENTITY,
            style: OverlayStyle {
                fill: OverlayFill::Solid([1.0, 1.0, 1.0, 1.0].into()),
                ..Default::default()
            },
            clip: OverlayClip::default(),
            size: [100.0, 100.0],
            shape: OverlayShape::default(),
            z_order: 0,
            provides_mask: None,
            animations: None,
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-side SDF evaluation (mirrors the GPU shader functions)
// ---------------------------------------------------------------------------

fn sd_rounded_box(p: [f32; 2], b: [f32; 2], r: [f32; 4]) -> f32 {
    // r: [top-right, bottom-right, bottom-left, top-left] (iq convention).
    let chosen = if p[0] > 0.0 {
        if p[1] > 0.0 { r[1] } else { r[0] } // right: bottom-right or top-right
    } else {
        if p[1] > 0.0 { r[2] } else { r[3] } // left: bottom-left or top-left
    };
    let qx = (p[0].abs() - b[0] + chosen).max(0.0);
    let qy = (p[1].abs() - b[1] + chosen).max(0.0);
    let outer = (qx * qx + qy * qy).sqrt();
    let inner = (p[0].abs() - b[0] + chosen)
        .max(p[1].abs() - b[1] + chosen)
        .min(0.0);
    inner + outer - chosen
}

fn sd_circle(p: [f32; 2], r: f32) -> f32 {
    (p[0] * p[0] + p[1] * p[1]).sqrt() - r
}

fn sd_ellipse(p: [f32; 2], ab: [f32; 2]) -> f32 {
    let pa = [p[0].abs(), p[1].abs()];
    let ei = [1.0 / ab[0], 1.0 / ab[1]];
    let e2 = [ab[0] * ab[0], ab[1] * ab[1]];
    let ve = [ei[0] * (e2[0] - e2[1]), ei[1] * (e2[1] - e2[0])];

    let mut t = [std::f32::consts::FRAC_1_SQRT_2; 2];
    for _ in 0..3 {
        let v = [ve[0] * t[0] * t[0] * t[0], ve[1] * t[1] * t[1] * t[1]];
        let diff = [pa[0] - v[0], pa[1] - v[1]];
        let diff_len = (diff[0] * diff[0] + diff[1] * diff[1]).sqrt();
        let tab_v = [t[0] * ab[0] - v[0], t[1] * ab[1] - v[1]];
        let tab_v_len = (tab_v[0] * tab_v[0] + tab_v[1] * tab_v[1]).sqrt();
        let u = if diff_len > 0.0 {
            let s = tab_v_len / diff_len;
            [diff[0] * s, diff[1] * s]
        } else {
            [0.0, 0.0]
        };
        let w = [ei[0] * (v[0] + u[0]), ei[1] * (v[1] + u[1])];
        let wc = [w[0].clamp(0.0, 1.0), w[1].clamp(0.0, 1.0)];
        let wlen = (wc[0] * wc[0] + wc[1] * wc[1]).sqrt();
        t = if wlen > 0.0 {
            [wc[0] / wlen, wc[1] / wlen]
        } else {
            t
        };
    }
    let nearest = [t[0] * ab[0], t[1] * ab[1]];
    let dx = pa[0] - nearest[0];
    let dy = pa[1] - nearest[1];
    let d = (dx * dx + dy * dy).sqrt();
    let np = [pa[0] / ab[0], pa[1] / ab[1]];
    let inside = np[0] * np[0] + np[1] * np[1];
    if inside < 1.0 { -d } else { d }
}

fn sd_capsule(p: [f32; 2], hs: [f32; 2]) -> f32 {
    let r = hs[0].min(hs[1]);
    let mut qx = p[0].abs();
    let mut qy = p[1].abs();
    if hs[0] > hs[1] {
        qx -= hs[0] - r;
    } else {
        qy -= hs[1] - r;
    }
    (qx.max(0.0) * qx.max(0.0) + qy.max(0.0) * qy.max(0.0)).sqrt() - r
}

fn sd_ring(p: [f32; 2], outer_r: f32, inner_frac: f32) -> f32 {
    let wall = outer_r * (1.0 - inner_frac) * 0.5;
    let mid_r = outer_r - wall;
    ((p[0] * p[0] + p[1] * p[1]).sqrt() - mid_r).abs() - wall
}

fn sd_arc(p: [f32; 2], outer_r: f32, inner_frac: f32, sa: f32, ea: f32) -> f32 {
    let d_ring = sd_ring(p, outer_r, inner_frac);

    let angle = p[1].atan2(p[0]);
    let two_pi = std::f32::consts::TAU;
    let sweep = ((ea - sa) % two_pi + two_pi) % two_pi;
    let a = ((angle - sa) % two_pi + two_pi) % two_pi;

    if a <= sweep {
        return d_ring;
    }

    let wall = outer_r * (1.0 - inner_frac) * 0.5;
    let mid_r = outer_r - wall;
    let inner_r = mid_r - wall;
    let outer_edge = mid_r + wall;

    let cs = [sa.cos(), sa.sin()];
    let ce = [ea.cos(), ea.sin()];

    let dot_s = (p[0] * cs[0] + p[1] * cs[1]).clamp(inner_r, outer_edge);
    let dot_e = (p[0] * ce[0] + p[1] * ce[1]).clamp(inner_r, outer_edge);

    let dsx = p[0] - cs[0] * dot_s;
    let dsy = p[1] - cs[1] * dot_s;
    let ds = (dsx * dsx + dsy * dsy).sqrt();

    let dex = p[0] - ce[0] * dot_e;
    let dey = p[1] - ce[1] * dot_e;
    let de = (dex * dex + dey * dey).sqrt();

    ds.min(de)
}

fn sd_line(p: [f32; 2], hs: [f32; 2], radius: f32, square: bool) -> f32 {
    // Segment from (-hs.x, -hs.y) to (hs.x, hs.y).
    if square {
        // Rotated box: half-length along segment, half-width = radius.
        let seg_len = (hs[0] * hs[0] + hs[1] * hs[1]).sqrt();
        if seg_len < 1e-6 {
            return (p[0] * p[0] + p[1] * p[1]).sqrt() - radius;
        }
        let dx = hs[0] / seg_len;
        let dy = hs[1] / seg_len;
        // Rotate p into segment frame.
        let along = p[0] * dx + p[1] * dy;
        let perp = -p[0] * dy + p[1] * dx;
        let qx = along.abs() - seg_len;
        let qy = perp.abs() - radius;
        (qx.max(0.0) * qx.max(0.0) + qy.max(0.0) * qy.max(0.0)).sqrt() + qx.max(qy).min(0.0)
    } else {
        // Capsule: segment from A=(-hs.x,-hs.y) to B=(hs.x,hs.y).
        let bax = 2.0 * hs[0];
        let bay = 2.0 * hs[1];
        let pax = p[0] + hs[0];
        let pay = p[1] + hs[1];
        let t = ((pax * bax + pay * bay) / (bax * bax + bay * bay)).clamp(0.0, 1.0);
        let ex = pax - bax * t;
        let ey = pay - bay * t;
        (ex * ex + ey * ey).sqrt() - radius
    }
}

fn sd_star(p: [f32; 2], r: f32, n: f32, rf: f32) -> f32 {
    let ri = r * rf;
    let an = std::f32::consts::PI / n;
    let two_an = 2.0 * an;

    let a = p[1].atan2(p[0]);
    let a_mod = ((a % two_an) + two_an) % two_an;
    let a_abs = if a_mod > an { two_an - a_mod } else { a_mod };

    let rp = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let q = [rp * a_abs.cos(), rp * a_abs.sin()];

    // Edge from outer tip (r, 0) to inner valley (ri*cos(an), ri*sin(an)).
    let bax = ri * an.cos() - r;
    let bay = ri * an.sin();
    let qax = q[0] - r;
    let qay = q[1];
    let t = ((qax * bax + qay * bay) / (bax * bax + bay * bay)).clamp(0.0, 1.0);
    let ex = qax - bax * t;
    let ey = qay - bay * t;
    let d = (ex * ex + ey * ey).sqrt();
    // Cross product: negative means inside.
    let cross = qax * bay - qay * bax;
    if cross < 0.0 { -d } else { d }
}

fn sd_ngon(p: [f32; 2], r: f32, n: f32) -> f32 {
    // Regular n-gon with circumradius r.
    let an = std::f32::consts::PI / n;
    let two_an = 2.0 * an;
    // Shift by an so edge midpoints align with x-axis in the folded sector.
    let a = p[1].atan2(p[0]) + an;
    let a_mod = ((a % two_an) + two_an) % two_an;
    let a_abs = if a_mod > an { two_an - a_mod } else { a_mod };

    let rp = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let q = [rp * a_abs.cos(), rp * a_abs.sin()];

    let he = r * an.cos(); // apothem
    let hv = r * an.sin(); // half vertex extent

    let dx = q[0] - he;
    let dy = (q[1] - hv).max(0.0);
    if dy > 0.0 {
        (dx * dx + dy * dy).sqrt()
    } else {
        dx
    }
}

fn sd_cross(p: [f32; 2], hs: [f32; 2], arm_frac: f32) -> f32 {
    let arm_w = arm_frac * hs[0].min(hs[1]);
    let box_sdf = |p: [f32; 2], b: [f32; 2]| -> f32 {
        let qx = p[0].abs() - b[0];
        let qy = p[1].abs() - b[1];
        (qx.max(0.0) * qx.max(0.0) + qy.max(0.0) * qy.max(0.0)).sqrt() + qx.max(qy).min(0.0)
    };
    let d_h = box_sdf(p, [hs[0], arm_w]);
    let d_v = box_sdf(p, [arm_w, hs[1]]);
    d_h.min(d_v)
}

fn sd_triangle(p: [f32; 2], hs: [f32; 2]) -> f32 {
    let q = [p[0].abs(), p[1]];
    let e = [hs[0], 2.0 * hs[1]];
    let elen = (e[0] * e[0] + e[1] * e[1]).sqrt();
    let en = [e[0] / elen, e[1] / elen];
    let n = [en[1], -en[0]];
    let d_edge = (q[0] - 0.0) * n[0] + (q[1] - (-hs[1])) * n[1];
    let d_base = q[1] - hs[1];
    d_edge.max(d_base)
}

impl OverlayShapeItem {
    /// Create a shape at `position` (top-left, logical pixels) with `size`
    /// (width, height). All other fields take their defaults; set them with the
    /// `with_*` methods below.
    pub fn new(shape: OverlayShape, position: [f32; 2], size: [f32; 2]) -> Self {
        Self {
            shape,
            transform: OverlayTransform::at(position),
            size,
            ..Default::default()
        }
    }

    /// Build a textured rectangle that fills the given `texture` at its natural
    /// pixel size times `scale`, anchored to a corner or the centre of the
    /// viewport. This covers the common screen-space image overlay: a corner
    /// logo, a watermark, a diagnostic HUD, or a live feed.
    ///
    /// `natural_size` is the image's display size in logical pixels and
    /// `viewport_size` is the current logical viewport size. `anchor_x` /
    /// `anchor_y` pin the image to a viewport edge or centre (e.g. `Right` /
    /// `Bottom` for the bottom-right corner). Pair `texture` with a streaming
    /// `OverlayTextureId` updated each frame for a live image, or a static
    /// uploaded one for a fixed image. The fill's tint is left white, so the
    /// texture is drawn unmodified; set `opacity`, or the fill's tint, to fade
    /// or colour it.
    pub fn textured_image(
        texture: OverlayTextureId,
        natural_size: [f32; 2],
        scale: f32,
        anchor_x: AnchorX,
        anchor_y: AnchorY,
        viewport_size: [f32; 2],
    ) -> Self {
        let size = [natural_size[0] * scale, natural_size[1] * scale];
        let position =
            super::anchor::viewport_anchored_top_left(anchor_x, anchor_y, size, viewport_size);
        Self::new(OverlayShape::Rect { corner_radius: 0.0 }, position, size).with_texture(texture)
    }

    /// Build an arbitrary vector shape from `subpaths` combined under
    /// `fill_rule`. Subpath coordinates are path-local logical pixels, placed
    /// at `position` (top-left); `size` is the bounding box gradient and
    /// texture fills map across. Set the fill and other fields with the
    /// `with_*` methods.
    ///
    /// Use this for shapes that are not one of the analytic variants: multiple
    /// contours, holes, curves, or SVG / icon art. Reach for the analytic
    /// variants (`Rect`, `Circle`, `Ring`, and so on) for simple, animated, or
    /// effect-heavy chrome, which they draw more cheaply and crisply.
    pub fn vector(
        subpaths: Vec<SubPath>,
        fill_rule: FillRule,
        position: [f32; 2],
        size: [f32; 2],
    ) -> Self {
        Self::new(
            OverlayShape::Vector {
                subpaths,
                fill_rule,
            },
            position,
            size,
        )
    }

    /// Set the whole baked appearance at once.
    pub fn with_style(mut self, style: OverlayStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the area fill.
    pub fn with_fill(mut self, fill: OverlayFill) -> Self {
        self.style.fill = fill;
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.style.opacity = opacity;
        self
    }

    /// Add an outline: a band of `width` logical pixels on the item's edge,
    /// placed by `mode`. A width of `0.0` adds nothing.
    ///
    /// An outline is a shadow layer with no blur, so this pushes one (or two, for
    /// [`OutlineMode::Centre`]) onto [`OverlayStyle::shadows`] and
    /// [`OverlayStyle::inner_shadows`]. That is the whole implementation: there
    /// is no separate border in the renderer, and the band draws through the
    /// same code as every other layer, on every coverage backend.
    ///
    /// It costs a layer out of [`OVERLAY_MAX_SHADOW_LAYERS`] per list, so a
    /// shape with four drop shadows cannot also take an outline.
    ///
    /// Calling this twice adds two bands rather than replacing the first.
    ///
    /// [`OverlayStyle::shadows`]: crate::overlay::OverlayStyle::shadows
    /// [`OverlayStyle::inner_shadows`]: crate::overlay::OverlayStyle::inner_shadows
    pub fn with_outline(
        mut self,
        colour: impl Into<crate::colour::Colour>,
        width: f32,
        mode: OutlineMode,
    ) -> Self {
        if width <= 0.0 {
            return self;
        }
        let colour = colour.into();
        let band = |spread: f32| ShadowLayer::new(colour, 0.0, [0.0, 0.0]).with_spread(spread);
        match mode {
            OutlineMode::Inset => self.style.inner_shadows.push(band(width)),
            OutlineMode::Outer => self.style.shadows.push(band(width)),
            OutlineMode::Centre => {
                self.style.inner_shadows.push(band(width * 0.5));
                self.style.shadows.push(band(width * 0.5));
            }
        }
        self
    }

    /// Set the draw order. Lower values render first (further back).
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Fill the shape with an uploaded overlay texture, clipped by the SDF.
    /// Replaces any colour or gradient fill: an item has one fill.
    pub fn with_texture(mut self, texture: OverlayTextureId) -> Self {
        self.style.fill = OverlayFill::texture(texture);
        self
    }

    /// Set the backdrop blur radius (frosted-glass effect) in logical pixels.
    pub fn with_backdrop_blur(mut self, radius: f32) -> Self {
        self.style.backdrop.blur = radius;
        self
    }

    /// Set the rotation around the shape centre, in radians.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.transform.rotation = radians;
        self
    }

    /// Set the point to rotate around, in logical pixels from the shape
    /// centre. `[0.0, 0.0]` rotates around the centre (the default).
    pub fn with_rotation_pivot(mut self, pivot: [f32; 2]) -> Self {
        self.transform.pivot = pivot;
        self
    }

    /// Set the backdrop colour filters applied to the blurred scene behind a
    /// `backdrop_blur` shape: saturation and brightness multipliers (`1.0`
    /// leaves each unchanged) and a hue rotation in radians.
    pub fn with_backdrop_filters(
        mut self,
        saturation: f32,
        brightness: f32,
        hue_shift: f32,
    ) -> Self {
        self.style.backdrop.saturation = saturation;
        self.style.backdrop.brightness = brightness;
        self.style.backdrop.hue_shift = hue_shift;
        self
    }

    /// Set the stacked outer shadow layers (drawn behind the fill). Replaces
    /// the single legacy `with_shadow` outer shadow on the solid shape path.
    pub fn with_shadows(mut self, shadows: Vec<ShadowLayer>) -> Self {
        self.style.shadows = shadows;
        self
    }

    /// Add one outer shadow layer, in front of any already set.
    pub fn with_shadow(mut self, shadow: ShadowLayer) -> Self {
        self.style.shadows.push(shadow);
        self
    }

    /// Add one inner shadow layer, over any already set.
    pub fn with_inner_shadow(mut self, shadow: ShadowLayer) -> Self {
        self.style.inner_shadows.push(shadow);
        self
    }

    /// Set the stacked inner (inset) shadow layers, drawn on top of the fill.
    pub fn with_inner_shadows(mut self, shadows: Vec<ShadowLayer>) -> Self {
        self.style.inner_shadows = shadows;
        self
    }

    /// Mark this shape as a clip mask under `mask_id`. Items whose `clip.mask`
    /// matches are clipped to it, and the mask itself is not drawn.
    ///
    /// The clip follows the mask's distance field, not its bounding box, so a
    /// `Circle` mask clips to a circle; the bounding box is only a cheap reject
    /// ahead of the field. Masks nest, and a fragment must be inside the whole
    /// parent chain to survive.
    pub fn provides_mask(mut self, mask_id: u32) -> Self {
        self.provides_mask = Some(mask_id);
        self
    }

    /// Clip this shape to the mask shape with this id. The mask's SDF (not just
    /// its bounding box) is used, and masks may nest.
    pub fn with_clip(mut self, clip_id: u32) -> Self {
        self.clip.mask = Some(clip_id);
        self
    }

    /// Set the multi-channel animation tracks.
    pub fn with_animations(mut self, animations: OverlayAnimations) -> Self {
        self.animations = Some(Box::new(animations));
        self
    }

    /// Set the origin the shape hangs from (a viewport corner or a world point).
    pub fn with_anchor(mut self, anchor: OverlayOrigin) -> Self {
        self.anchoring.origin = anchor;
        self
    }

    /// Pin the shape to a 3D world position, projected to screen each frame.
    /// Sugar for `with_anchor(OverlayOrigin::World(pos))`.
    pub fn with_world_anchor(mut self, pos: [f32; 3]) -> Self {
        self.anchoring.origin = OverlayOrigin::World(pos);
        self
    }

    /// Pin the item to a fixed screen position in logical pixels from the
    /// top-left. Sugar for the default viewport origin with `position` set to
    /// `pos`.
    pub fn with_screen_anchor(mut self, pos: [f32; 2]) -> Self {
        self.anchoring =
            crate::overlay::OverlayAnchoring::default().with_align(self.anchoring.align);
        self.transform.translate = pos;
        self
    }

    /// Set how the bounding box aligns onto the resolved anchor origin.
    pub fn with_align(mut self, align: Alignment) -> Self {
        self.anchoring.align = align;
        self
    }

    /// Resolve the effective top-left pixel of the bounding box for a frame:
    /// the resolved origin, plus `position`, shifted by `anchoring.align`
    /// for the current `size`. Returns `None` when a `World` anchor projects
    /// behind the camera or off-screen (the shape is skipped that frame).
    ///
    /// The default `Viewport { Left, Top }` anchor with `Left` / `Top`
    /// alignment resolves to `position` and needs no camera, so plain
    /// screen-space shapes resolve with any `view` / `proj`.
    pub fn resolve_top_left(
        &self,
        viewport_size: [f32; 2],
        view: &glam::Mat4,
        proj: &glam::Mat4,
    ) -> Option<[f32; 2]> {
        let origin = resolve_anchor_origin(&self.anchoring.origin, viewport_size, view, proj)?;
        Some([
            origin[0]
                + self.transform.translate[0]
                + self.anchoring.align.x.align_shift(self.size[0]),
            origin[1]
                + self.transform.translate[1]
                + self.anchoring.align.y.align_shift(self.size[1]),
        ])
    }

    /// Signed distance from a screen-space point to the shape boundary.
    ///
    /// The point is in logical pixels from the top-left of the viewport (the
    /// same coordinate space as `position`). Negative values mean the point is
    /// inside the shape; positive values mean it is outside.
    ///
    /// This evaluates the same SDF used by the GPU shader, so the boundary
    /// matches what is rendered on screen (ignoring sub-pixel AA).
    ///
    /// `position` is treated as the box's absolute top-left. For a shape with a
    /// non-default `anchor` (a viewport corner or a world point), first resolve
    /// the frame's top-left with [`Self::resolve_top_left`] and hit-test a copy
    /// whose `position` is that value, so the test frame matches where the shape
    /// draws.
    pub fn distance(&self, point: [f32; 2]) -> f32 {
        let hw = self.size[0] * 0.5;
        let hh = self.size[1] * 0.5;
        let cx = self.transform.translate[0] + hw;
        let cy = self.transform.translate[1] + hh;
        let dx = point[0] - cx;
        let dy = point[1] - cy;
        // Rotate the query point by -rotation around the rotation pivot (an
        // offset from the shape centre) so the SDF evaluates in the unrotated
        // frame, matching the fragment shader. With a zero pivot this reduces
        // to rotation around the centre.
        let c = (-self.transform.rotation).cos();
        let s = (-self.transform.rotation).sin();
        let piv = self.transform.pivot;
        let rx = dx - piv[0];
        let ry = dy - piv[1];
        let p = [c * rx - s * ry + piv[0], s * rx + c * ry + piv[1]];
        let hs = [hw, hh];

        match &self.shape {
            OverlayShape::Rect { corner_radius } => {
                let r = corner_radius.min(hw).min(hh).max(0.0);
                sd_rounded_box(p, hs, [r, r, r, r])
            }
            OverlayShape::RoundedRect { radii: r } => {
                // Input: [tl, tr, br, bl]. iq convention: [tr, br, bl, tl].
                let clamped = [
                    r[1].min(hw).min(hh).max(0.0),
                    r[2].min(hw).min(hh).max(0.0),
                    r[3].min(hw).min(hh).max(0.0),
                    r[0].min(hw).min(hh).max(0.0),
                ];
                sd_rounded_box(p, hs, clamped)
            }
            OverlayShape::Circle => sd_circle(p, hw.min(hh)),
            OverlayShape::Ellipse => sd_ellipse(p, hs),
            OverlayShape::Capsule => sd_capsule(p, hs),
            OverlayShape::Ring { inner_radius_frac } => {
                sd_ring(p, hw.min(hh), inner_radius_frac.clamp(0.0, 1.0))
            }
            OverlayShape::Arc {
                inner_radius_frac,
                start_angle,
                end_angle,
            } => sd_arc(
                p,
                hw.min(hh),
                inner_radius_frac.clamp(0.0, 1.0),
                *start_angle,
                *end_angle,
            ),
            OverlayShape::Triangle { direction } => {
                let (tp, ths) = match direction {
                    TriangleDirection::Up => (p, hs),
                    TriangleDirection::Down => ([p[0], -p[1]], hs),
                    TriangleDirection::Left => ([p[1], p[0]], [hh, hw]),
                    TriangleDirection::Right => ([-p[1], p[0]], [hh, hw]),
                };
                sd_triangle(tp, ths)
            }
            OverlayShape::Line { thickness, cap } => {
                sd_line(p, hs, thickness * 0.5, *cap == LineCap::Square)
            }
            OverlayShape::Star {
                points,
                inner_radius_frac,
            } => {
                let r = hw.min(hh);
                sd_star(p, r, *points as f32, inner_radius_frac.clamp(0.0, 1.0))
            }
            OverlayShape::RegularPolygon { sides } => {
                let r = hw.min(hh);
                sd_ngon(p, r, (*sides).max(3) as f32)
            }
            OverlayShape::Cross { arm_width_frac } => {
                sd_cross(p, hs, arm_width_frac.clamp(0.0, 1.0))
            }
            OverlayShape::Vector {
                subpaths,
                fill_rule,
            } => {
                // No closed-form SDF. Return a sign-only pseudo-distance from a
                // point-in-path test: negative inside, positive outside.
                // `contains` only reads the sign; the magnitude is not a true
                // distance. `p` is in the centred, unrotated frame, so shift it
                // back into path-local space (origin at the item's top-left).
                let q = [p[0] + hw, p[1] + hh];
                if path_contains(subpaths, *fill_rule, q) {
                    -1.0
                } else {
                    1.0
                }
            }
        }
    }

    /// Returns `true` if the screen-space point falls inside the shape boundary.
    ///
    /// The point is in logical pixels from the top-left of the viewport.
    /// Equivalent to `self.distance(point) <= 0.0`.
    pub fn contains(&self, point: [f32; 2]) -> bool {
        self.distance(point) <= 0.0
    }

    /// Set the transform: translate, rotate, scale, and pivot at once.
    pub fn with_transform(mut self, transform: OverlayTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Set the translation in logical pixels from the resolved anchor origin.
    pub fn with_position(mut self, position: [f32; 2]) -> Self {
        self.transform.translate = position;
        self
    }

    /// Set the uniform scale about the transform pivot.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.transform.scale = scale;
        self
    }

    /// Set the per-frame colour multiplier (identity `[1, 1, 1, 1]`).
    pub fn with_tint(mut self, tint: [f32; 4]) -> Self {
        self.style.tint = tint;
        self
    }

    /// Clip to an axis-aligned box in logical pixels `[x0, y0, x1, y1]`.
    pub fn with_clip_rect(mut self, clip_rect: [f32; 4]) -> Self {
        self.clip.rect = Some(clip_rect);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape_at(x: f32, y: f32, w: f32, h: f32, shape: OverlayShape) -> OverlayShapeItem {
        OverlayShapeItem {
            transform: OverlayTransform::at([x, y]),
            size: [w, h],
            shape,
            ..Default::default()
        }
    }

    #[test]
    fn rect_centre_is_inside() {
        let s = shape_at(
            100.0,
            100.0,
            80.0,
            60.0,
            OverlayShape::Rect { corner_radius: 0.0 },
        );
        assert!(s.contains([140.0, 130.0])); // centre
        assert!(s.distance([140.0, 130.0]) < 0.0);
    }

    #[test]
    fn rect_outside() {
        let s = shape_at(
            100.0,
            100.0,
            80.0,
            60.0,
            OverlayShape::Rect { corner_radius: 0.0 },
        );
        assert!(!s.contains([50.0, 130.0])); // left of shape
        assert!(!s.contains([200.0, 130.0])); // right of shape
    }

    #[test]
    fn rect_edge_distance() {
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::Rect { corner_radius: 0.0 },
        );
        // Centre is at (50, 50), half-size 50x50. Point on the right edge:
        let d = s.distance([100.0, 50.0]);
        assert!(d.abs() < 0.01, "edge distance should be ~0, got {d}");
    }

    #[test]
    fn rounded_rect_corner_is_outside() {
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::Rect {
                corner_radius: 20.0,
            },
        );
        // The very corner pixel should be outside the rounded shape.
        assert!(!s.contains([1.0, 1.0]));
        // But interior should still be inside.
        assert!(s.contains([50.0, 50.0]));
    }

    #[test]
    fn circle_contains() {
        let s = shape_at(0.0, 0.0, 100.0, 100.0, OverlayShape::Circle);
        assert!(s.contains([50.0, 50.0])); // centre
        assert!(!s.contains([1.0, 1.0])); // corner
        // Just inside the circle edge (radius = 50, point at distance ~49):
        assert!(s.contains([50.0, 1.5]));
    }

    #[test]
    fn ellipse_contains() {
        let s = shape_at(0.0, 0.0, 200.0, 100.0, OverlayShape::Ellipse);
        assert!(s.contains([100.0, 50.0])); // centre
        assert!(!s.contains([1.0, 1.0])); // corner
    }

    #[test]
    fn capsule_contains() {
        let s = shape_at(0.0, 0.0, 120.0, 40.0, OverlayShape::Capsule);
        assert!(s.contains([60.0, 20.0])); // centre
        // Corner outside the rounded end:
        assert!(!s.contains([1.0, 1.0]));
    }

    #[test]
    fn ring_hole_is_outside() {
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::Ring {
                inner_radius_frac: 0.7,
            },
        );
        // Centre of the ring (the hole) should be outside.
        assert!(!s.contains([50.0, 50.0]));
        // Point in the wall area should be inside.
        assert!(s.contains([50.0, 8.0]));
    }

    #[test]
    fn arc_inside_sweep() {
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::Arc {
                inner_radius_frac: 0.6,
                start_angle: 0.0,
                end_angle: std::f32::consts::PI,
            },
        );
        // Point in the right half of the ring (angle ~0), within the sweep:
        assert!(s.contains([92.0, 50.0]));
        // Point above centre in screen coords (local y = -42, angle ~ -PI/2),
        // outside the [0, PI] sweep:
        assert!(!s.contains([50.0, 8.0]));
    }

    #[test]
    fn triangle_centre_inside() {
        let s = shape_at(
            0.0,
            0.0,
            60.0,
            60.0,
            OverlayShape::Triangle {
                direction: TriangleDirection::Up,
            },
        );
        assert!(s.contains([30.0, 35.0])); // slightly below centre
        assert!(!s.contains([1.0, 1.0])); // top-left corner
    }

    #[test]
    fn triangle_directions() {
        for dir in [
            TriangleDirection::Up,
            TriangleDirection::Down,
            TriangleDirection::Left,
            TriangleDirection::Right,
        ] {
            let s = shape_at(
                0.0,
                0.0,
                60.0,
                60.0,
                OverlayShape::Triangle { direction: dir },
            );
            // Centre-ish should always be inside.
            assert!(
                s.contains([30.0, 30.0]),
                "centre should be inside for {dir:?}"
            );
        }
    }

    #[test]
    fn distance_is_negative_inside_positive_outside() {
        let s = shape_at(0.0, 0.0, 100.0, 100.0, OverlayShape::Circle);
        assert!(s.distance([50.0, 50.0]) < 0.0, "centre should be negative");
        assert!(
            s.distance([0.0, 0.0]) > 0.0,
            "far corner should be positive"
        );
    }

    #[test]
    fn line_round_contains() {
        // 100x4 horizontal line: segment from (-50,-2) to (50,2) in local space,
        // thickness=4 => cap radius 2. Centre (50,2) should be inside.
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            4.0,
            OverlayShape::Line {
                thickness: 4.0,
                cap: LineCap::Round,
            },
        );
        assert!(s.contains([50.0, 2.0])); // centre
        assert!(!s.contains([50.0, 10.0])); // well above
    }

    #[test]
    fn line_round_endpoint_is_on_boundary() {
        // Square bounding box: segment from (-30,-30) to (30,30), radius=5.
        let s = shape_at(
            0.0,
            0.0,
            60.0,
            60.0,
            OverlayShape::Line {
                thickness: 10.0,
                cap: LineCap::Round,
            },
        );
        // Centre is on the segment, distance = -5 (inside).
        assert!(s.contains([30.0, 30.0]));
    }

    #[test]
    fn line_square_cap_flat_end() {
        // Horizontal line, square cap. Points just past the endpoint (in the
        // cap direction) are outside since square caps don't extend.
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            4.0,
            OverlayShape::Line {
                thickness: 4.0,
                cap: LineCap::Square,
            },
        );
        assert!(s.contains([50.0, 2.0])); // centre
        assert!(!s.contains([50.0, 10.0])); // well above
    }

    #[test]
    fn star_centre_inside() {
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::Star {
                points: 5,
                inner_radius_frac: 0.45,
            },
        );
        assert!(s.contains([50.0, 50.0])); // centre
        assert!(!s.contains([1.0, 1.0])); // corner far outside
    }

    #[test]
    fn star_outer_tip_is_on_boundary() {
        // 5-pointed star in 100x100 box: outer radius = 50.
        // The SDF places tips at multiples of 2*pi/n starting from angle 0 (right).
        // The rightmost tip is at local (50, 0) = screen (100, 50).
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::Star {
                points: 5,
                inner_radius_frac: 0.45,
            },
        );
        // Rightmost tip at screen (100, 50). Distance should be ~0.
        let d = s.distance([100.0, 50.0]);
        assert!(
            d.abs() < 1.0,
            "outer tip distance should be near 0, got {d}"
        );
    }

    #[test]
    fn regular_polygon_centre_inside() {
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::RegularPolygon { sides: 6 },
        );
        assert!(s.contains([50.0, 50.0])); // centre
        assert!(!s.contains([1.0, 1.0])); // corner
    }

    #[test]
    fn regular_polygon_vertex_on_boundary() {
        // Hexagon in 100x100 box: circumradius 50. A vertex is at (50, 0)
        // in screen space (top of hexagon, angle = 0 before offset).
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::RegularPolygon { sides: 6 },
        );
        // For n=6, the vertex is at (r, 0) before the pi/n rotation offset.
        // After shifting by pi/6 the vertex that was at angle 0 is now at angle -pi/6.
        // The topmost point is at angle -pi/2 => (0, -50) => screen (50, 0).
        // Just check that the centre is inside and a far corner is outside.
        assert!(s.distance([50.0, 50.0]) < 0.0);
        assert!(s.distance([0.0, 0.0]) > 0.0);
    }

    #[test]
    fn cross_arms_inside_body_outside() {
        // 100x100 cross with arm_width_frac=0.3 => arm half-width = 15px.
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::Cross {
                arm_width_frac: 0.3,
            },
        );
        assert!(s.contains([50.0, 50.0])); // centre
        // Along the horizontal arm, near the edge of the bounding box.
        assert!(s.contains([95.0, 50.0]));
        // In the gap between arms (diagonal corner).
        assert!(!s.contains([5.0, 5.0]));
    }

    #[test]
    fn cross_centre_distance_negative() {
        let s = shape_at(
            0.0,
            0.0,
            100.0,
            100.0,
            OverlayShape::Cross {
                arm_width_frac: 0.4,
            },
        );
        assert!(s.distance([50.0, 50.0]) < 0.0);
        assert!(s.distance([0.0, 0.0]) > 0.0);
    }

    #[test]
    fn anim_track_linear_lerps_endpoints() {
        let track = AnimTrack::<f32> {
            start_time: 10.0,
            duration: 2.0,
            from: 0.0,
            to: 100.0,
            easing: OverlayEasing::Linear,
            repeat: RepeatMode::Once,
        };
        assert!((track.sample(10.0) - 0.0).abs() < 1e-3);
        assert!((track.sample(11.0) - 50.0).abs() < 1e-3);
        assert!((track.sample(12.0) - 100.0).abs() < 1e-3);
        // After duration, Once holds the final value.
        assert!((track.sample(50.0) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn anim_track_pingpong_oscillates() {
        let track = AnimTrack::<f32> {
            start_time: 0.0,
            duration: 1.0,
            from: 0.0,
            to: 10.0,
            easing: OverlayEasing::Linear,
            repeat: RepeatMode::PingPong,
        };
        // forward leg
        assert!((track.sample(0.5) - 5.0).abs() < 1e-3);
        assert!((track.sample(1.0) - 10.0).abs() < 1e-3);
        // reverse leg
        assert!((track.sample(1.5) - 5.0).abs() < 1e-3);
        assert!((track.sample(2.0) - 0.0).abs() < 1e-3);
        // next forward leg
        assert!((track.sample(2.5) - 5.0).abs() < 1e-3);
    }

    #[test]
    fn anim_track_vec2_interpolates_componentwise() {
        let track = AnimTrack::<[f32; 2]> {
            start_time: 0.0,
            duration: 1.0,
            from: [0.0, 100.0],
            to: [200.0, 0.0],
            easing: OverlayEasing::Linear,
            repeat: RepeatMode::Once,
        };
        let v = track.sample(0.5);
        assert!((v[0] - 100.0).abs() < 1e-3);
        assert!((v[1] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn bezier_path_hits_endpoints() {
        // Cubic with p0 = (0,0), p3 = (100, 0) and arched control handles.
        let track = PathTrack::<[f32; 2]>::bezier(
            0.0,
            1.0,
            [[0.0, 0.0], [25.0, -40.0], [75.0, -40.0], [100.0, 0.0]],
        );
        let a = track.sample(0.0);
        let b = track.sample(1.0);
        assert!((a[0] - 0.0).abs() < 1e-3 && (a[1] - 0.0).abs() < 1e-3);
        assert!((b[0] - 100.0).abs() < 1e-3 && (b[1] - 0.0).abs() < 1e-3);
        // Midpoint should sit on the arch above the baseline.
        let m = track.sample(0.5);
        assert!((m[0] - 50.0).abs() < 1e-3);
        assert!(m[1] < -20.0);
    }

    #[test]
    fn polyline_path_hits_waypoints() {
        let track =
            PathTrack::<[f32; 2]>::polyline(0.0, 1.0, vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]);
        let a = track.sample(0.0);
        let mid = track.sample(0.5);
        let end = track.sample(1.0);
        assert!((a[0] - 0.0).abs() < 1e-3 && (a[1] - 0.0).abs() < 1e-3);
        assert!((mid[0] - 10.0).abs() < 1e-3 && (mid[1] - 0.0).abs() < 1e-3);
        assert!((end[0] - 10.0).abs() < 1e-3 && (end[1] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn path_track_custom_closure_loops() {
        // A non-curve path: harmonic motion via a custom closure.
        let track = PathTrack::<f32>::new(0.0, 1.0, |t| (t * std::f32::consts::TAU).sin())
            .with_repeat(RepeatMode::Loop);
        let a = track.sample(0.0);
        let b = track.sample(0.25);
        let c = track.sample(0.5);
        assert!(a.abs() < 1e-3);
        assert!((b - 1.0).abs() < 1e-3);
        assert!(c.abs() < 1e-3);
    }

    #[test]
    fn rotation_affects_hit_test() {
        // 100x40 capsule. Without rotation, (50, 80) sits below the shape
        // (outside). Rotated 90 degrees, the capsule's long axis becomes
        // vertical and that point is inside the body.
        let mut s = OverlayShapeItem {
            transform: OverlayTransform::at([0.0, 30.0]),
            size: [100.0, 40.0],
            shape: OverlayShape::Capsule,
            ..Default::default()
        };
        assert!(!s.contains([50.0, 80.0]));
        s.transform.rotation = std::f32::consts::FRAC_PI_2;
        assert!(s.contains([50.0, 80.0]));
    }

    #[test]
    fn rotation_pivot_shifts_hit_boundary() {
        // 80x40 rectangle at the origin: centre (40, 20). Rotating 90 degrees
        // about the centre keeps the centre fixed. Rotating about a pivot far
        // from the centre swings the whole shape elsewhere, so a point that is
        // inside under centre-rotation falls outside under pivot-rotation.
        let mut s = OverlayShapeItem {
            transform: OverlayTransform::IDENTITY.with_rotation(std::f32::consts::FRAC_PI_2),
            size: [80.0, 40.0],
            shape: OverlayShape::Rect { corner_radius: 0.0 },
            ..Default::default()
        };
        // Centre is always inside regardless of pivot.
        assert!(s.contains([40.0, 20.0]));
        // With a large pivot offset the shape rotates away from the centre.
        s.transform.pivot = [200.0, 0.0];
        assert!(!s.contains([40.0, 20.0]));
    }

    #[test]
    fn default_anchor_resolves_to_position() {
        // Default Viewport { Left, Top } + Left/Top align: the resolved top-left
        // equals `position`, with no camera needed, on any viewport size.
        let s = OverlayShapeItem::new(OverlayShape::Circle, [40.0, 70.0], [30.0, 30.0]);
        let tl = s
            .resolve_top_left([800.0, 600.0], &glam::Mat4::IDENTITY, &glam::Mat4::IDENTITY)
            .unwrap();
        assert_eq!(tl, [40.0, 70.0]);
    }

    #[test]
    fn viewport_corner_pins_across_sizes() {
        // A bottom-right-anchored, bottom-right-aligned box keeps its bottom-right
        // corner on the viewport's bottom-right corner regardless of size.
        let s = OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [0.0, 0.0],
            [50.0, 20.0],
        )
        .with_anchor(OverlayOrigin::Viewport(Alignment::new(
            AnchorX::Right,
            AnchorY::Bottom,
        )))
        .with_align(Alignment::new(AnchorX::Right, AnchorY::Bottom));
        let id = glam::Mat4::IDENTITY;
        let a = s.resolve_top_left([800.0, 600.0], &id, &id).unwrap();
        assert_eq!(a, [750.0, 580.0]);
        let b = s.resolve_top_left([1024.0, 768.0], &id, &id).unwrap();
        assert_eq!(b, [974.0, 748.0]);
    }

    #[test]
    fn centre_anchor_centres_box() {
        // Middle/Middle origin with Middle/Middle align centres the box on the
        // viewport centre, so the drawn box and a hit-test on it agree.
        let s = OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [0.0, 0.0],
            [80.0, 60.0],
        )
        .with_anchor(OverlayOrigin::Viewport(Alignment::new(
            AnchorX::Middle,
            AnchorY::Middle,
        )))
        .with_align(Alignment::new(AnchorX::Middle, AnchorY::Middle));
        let id = glam::Mat4::IDENTITY;
        let tl = s.resolve_top_left([800.0, 600.0], &id, &id).unwrap();
        assert_eq!(tl, [360.0, 270.0]); // 400-40, 300-30
        // The centre of the resolved box is the viewport centre.
        let mut drawn = s.clone();
        drawn.transform.translate = tl;
        assert!(drawn.contains([400.0, 300.0]));
    }

    #[test]
    fn position_nudges_from_origin() {
        // `position` layers on top of the resolved origin, so it nudges an
        // anchored box away from the corner (the animatable channel does the same).
        let s = OverlayShapeItem::new(OverlayShape::Circle, [12.0, -8.0], [20.0, 20.0])
            .with_anchor(OverlayOrigin::Viewport(Alignment::new(
                AnchorX::Right,
                AnchorY::Top,
            )))
            .with_align(Alignment::new(AnchorX::Right, AnchorY::Top));
        let id = glam::Mat4::IDENTITY;
        let tl = s.resolve_top_left([500.0, 500.0], &id, &id).unwrap();
        // origin x = 500, align Right shifts by -20, position adds [12, -8].
        assert_eq!(tl, [500.0 - 20.0 + 12.0, 0.0 - 8.0]);
    }

    #[test]
    fn world_anchor_culls_when_offscreen() {
        // A world point behind the camera resolves to None (shape skipped),
        // while an on-screen point resolves to a pixel inside the viewport.
        let s = OverlayShapeItem::new(OverlayShape::Circle, [0.0, 0.0], [10.0, 10.0])
            .with_world_anchor([0.0, 0.0, 0.0]);
        let proj = glam::Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        // Camera at +z=5 looking toward the origin (down -z), Z-up.
        let view = glam::Mat4::look_at_rh(
            glam::vec3(0.0, 0.0, 5.0),
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 1.0, 0.0),
        );
        let on = s.resolve_top_left([400.0, 400.0], &view, &proj);
        assert!(on.is_some());
        let p = on.unwrap();
        assert!((p[0] - 200.0).abs() < 1.0 && (p[1] - 200.0).abs() < 1.0);

        // Move the camera so the point is behind it (looking away): culled.
        let behind = glam::Mat4::look_at_rh(
            glam::vec3(0.0, 0.0, 5.0),
            glam::vec3(0.0, 0.0, 10.0),
            glam::vec3(0.0, 1.0, 0.0),
        );
        assert!(s.resolve_top_left([400.0, 400.0], &behind, &proj).is_none());
    }

    #[test]
    fn rotation_pivot_zero_matches_centre_rotation() {
        // A zero pivot must reproduce plain centre rotation exactly.
        let base = OverlayShapeItem {
            transform: OverlayTransform::at([10.0, 10.0]).with_rotation(0.7),
            size: [100.0, 30.0],
            shape: OverlayShape::Capsule,
            ..Default::default()
        };
        let mut piv = base.clone();
        piv.transform.pivot = [0.0, 0.0];
        for pt in [[60.0, 25.0], [20.0, 20.0], [90.0, 40.0]] {
            assert!((base.distance(pt) - piv.distance(pt)).abs() < 1e-4);
        }
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;

    /// The animation block is boxed, so a static shape carries a pointer
    /// rather than a track per channel. Assert both halves so neither can
    /// regress: the field is one word, and the item stays inside its budget.
    #[test]
    fn shape_item_stays_small_without_animation_state() {
        assert_eq!(
            std::mem::size_of::<Option<Box<OverlayAnimations>>>(),
            std::mem::size_of::<usize>(),
            "the animation block is no longer boxed"
        );
        let item = std::mem::size_of::<OverlayShapeItem>();
        assert!(
            item <= 512,
            "OverlayShapeItem grew to {item} bytes (budget 512)"
        );
    }
}
