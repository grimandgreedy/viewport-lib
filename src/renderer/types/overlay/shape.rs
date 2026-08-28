use crate::renderer::types::*;

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
    Vector {
        /// The contours that make up the shape.
        subpaths: Vec<SubPath>,
        /// How the subpaths combine into filled area.
        fill_rule: FillRule,
    },
}

/// One soft drop or inset shadow.
///
/// Used with [`OverlayShapeItem::shadows`] (drawn behind the fill) and
/// [`OverlayShapeItem::inner_shadows`] (drawn on top of the fill, under the
/// border) to stack several shadow effects on a single shape: a soft ambient
/// shadow for depth plus a tighter one for contact, or an outer glow for
/// focus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowLayer {
    /// RGBA colour of the shadow, linear float format. The alpha scales the
    /// shadow strength.
    pub colour: [f32; 4],
    /// Blur spread in logical pixels. `0.0` produces no visible shadow.
    pub radius: f32,
    /// Offset of the shadow from the shape centre in logical pixels.
    /// Positive X shifts right, positive Y shifts down.
    pub offset: [f32; 2],
}

impl ShadowLayer {
    /// Build a shadow layer from colour, blur radius, and offset.
    pub fn new(colour: [f32; 4], radius: f32, offset: [f32; 2]) -> Self {
        Self {
            colour,
            radius,
            offset,
        }
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

/// A screen-space overlay shape rendered with a signed-distance function.
///
/// Each item becomes a single bounding quad on the GPU. The fragment shader
/// evaluates an SDF to produce anti-aliased fill, border, and discard regions.
///
/// `fill` controls the interior colour. Use `OverlayFill::Solid` for a flat
/// colour or `OverlayFill::LinearGradient` for a two-colour gradient.
///
/// When `texture` is set the shape samples the uploaded image as its fill.
/// In that case `fill` must be `OverlayFill::Solid`; the solid colour acts as
/// a tint multiplied with each texel. Use `[1.0, 1.0, 1.0, 1.0]` for no tint.
/// The SDF boundary, border, and AA apply the same way regardless of fill mode.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::{OverlayShapeItem, OverlayShape, OverlayFill};
/// // Rounded-rect panel background.
/// let panel = OverlayShapeItem::new(
///     OverlayShape::Rect { corner_radius: 8.0 },
///     [20.0, 20.0],
///     [300.0, 200.0],
/// )
/// .with_fill(OverlayFill::Solid([0.1, 0.1, 0.1, 0.85]))
/// .with_border([0.4, 0.4, 0.4, 1.0], 1.0);
///
/// // Circle with a left-to-right gradient.
/// let grad_dot = OverlayShapeItem::new(OverlayShape::Circle, [100.0, 100.0], [60.0, 60.0])
///     .with_fill(OverlayFill::LinearGradient {
///         start_colour: [0.0, 0.4, 1.0, 1.0],
///         end_colour: [0.0, 1.0, 0.5, 1.0],
///         angle: 0.0,
///     });
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OverlayShapeItem {
    /// Top-left position in logical pixels from the viewport top-left.
    pub position: [f32; 2],
    /// Width and height in logical pixels.
    pub size: [f32; 2],
    /// Which SDF shape to render.
    pub shape: OverlayShape,
    /// Fill style: solid colour or linear gradient.
    ///
    /// When `texture` is `Some` only `OverlayFill::Solid` is used; the colour
    /// becomes a tint multiplied with each texture sample.
    pub fill: OverlayFill,
    /// Overall opacity multiplier applied to both fill and border. Range 0.0-1.0.
    pub opacity: f32,
    /// RGBA border colour in linear float format.
    pub border_colour: [f32; 4],
    /// Border thickness in logical pixels. `0.0` disables the border.
    pub border_width: f32,
    /// Where the border sits relative to the shape edge. Default: `Inset`.
    pub border_mode: BorderMode,
    /// Draw order relative to other shapes. Lower values render first (further back).
    pub z_order: i32,
    /// Optional texture fill. When set the shape samples the image uploaded
    /// via `DeviceResources::upload_overlay_texture`, clipped by the SDF
    /// boundary. `fill` acts as a tint when this is `Some`.
    pub texture: Option<OverlayTextureId>,
    /// RGBA colour of the outer shadow/glow halo. Default: transparent (no shadow).
    pub shadow_colour: [f32; 4],
    /// Blur spread of the shadow in logical pixels. `0.0` disables the shadow.
    pub shadow_radius: f32,
    /// Offset of the shadow centre from the shape centre in logical pixels.
    /// Positive X shifts right, positive Y shifts down. Default: `[0.0, 0.0]`.
    pub shadow_offset: [f32; 2],
    /// Opacity animation. Resolved each frame during `prepare()` using
    /// `OverlayFrame::time`. Default: `OverlayAnimation::None`.
    pub animation: OverlayAnimation,
    /// Backdrop blur radius in logical pixels. When greater than zero the scene
    /// content behind the shape is blurred (frosted glass effect) and the
    /// `fill` colour is composited on top as a tint. `0.0` disables the
    /// effect. Only active in render paths where the renderer owns the command
    /// encoder (`render`, `render_viewport`); in `paint`/`paint_to` paths
    /// blur shapes fall back to a regular solid fill.
    pub backdrop_blur: f32,
    /// Marks this shape as a clip mask. The shape itself is not drawn; its
    /// bounding box defines a clipping rectangle for any shape whose
    /// `clip_id` equals this value. `None` means the shape is not a mask.
    ///
    /// A `clip_mask_id` must be unique within a frame. If two shapes carry the
    /// same id, the first one in submission order is used as the mask and the
    /// rest are ignored. When several independent sources emit into one frame,
    /// offset their ids so they do not collide.
    ///
    /// Used for scroll containers, masked panels, and composite widgets.
    /// Only the solid (non-textured, non-blur) shape path participates in
    /// clipping; textured and backdrop-blur shapes ignore both
    /// `clip_id` and `clip_mask_id`.
    ///
    /// Current limitation: the clip uses the mask shape's axis-aligned
    /// bounding box, not its SDF. For `Rect` and `RoundedRect` masks this
    /// matches the visible bounds; for `Circle`, `Ellipse`, and other
    /// curved shapes the clip is the enclosing square/rectangle.
    pub clip_mask_id: Option<u32>,
    /// When set, this shape is clipped to the bounding box of the mask shape
    /// whose `clip_mask_id` matches this value. Fragments outside the mask's
    /// bounding rect are discarded. `None` means the shape is drawn
    /// unclipped. If no mask with the matching id is present in the frame,
    /// the shape is also drawn unclipped.
    pub clip_id: Option<u32>,
    /// Rotation around the shape centre in radians. Positive rotates
    /// counter-clockwise in math coordinates. `0.0` keeps the default
    /// orientation. Applies to fill, border, shadow, and gradient direction;
    /// the bounding box (`position` + `size`) stays axis-aligned, so the
    /// rotated shape is drawn inside the unrotated box.
    pub rotation: f32,
    /// 9-slice texture sampling for the shape's `texture` fill. When `None`
    /// the texture stretches to fill the bounding box (default).
    pub nine_slice: Option<NineSlice>,
    /// Affine transform applied to the texture sample before lookup. Lets a
    /// single texture pan, scale, rotate, tile, and flip independently of
    /// the shape it fills. Ignored when `nine_slice` is also set on the
    /// same shape.
    pub texture_transform: TextureTransform,
    /// When `true`, the existing `shadow_*` fields render as an *inset*
    /// (inner) shadow that fades from the edge inward instead of an outer
    /// drop shadow. Default `false` (outer shadow, the legacy behaviour).
    ///
    /// Use for pressed buttons, dropdowns, text inputs, scroll wells, and
    /// other recessed UI surfaces.
    ///
    /// A shape currently carries either an outer or an inner shadow, not
    /// both at once. Stackable outer + inner shadow layers are planned for
    /// a follow-up phase.
    pub shadow_inset: bool,
    /// Multi-channel animation tracks for `position`, `size`, `fill`,
    /// `border_colour`, `rotation`, and `opacity`. Each `Some` track
    /// replaces the matching field on the item for the frame. The
    /// `opacity` track takes precedence over the legacy
    /// [`Self::animation`] field when both are set.
    pub animations: OverlayAnimations,
    /// Saturation multiplier applied to the blurred backdrop. `1.0` leaves
    /// saturation unchanged, `0.0` produces greyscale. Only affects shapes
    /// with `backdrop_blur > 0.0`.
    pub backdrop_saturation: f32,
    /// Brightness multiplier applied to the blurred backdrop. `1.0` is
    /// unchanged. Only affects shapes with `backdrop_blur > 0.0`.
    pub backdrop_brightness: f32,
    /// Hue rotation applied to the blurred backdrop, in radians. `0.0` is
    /// unchanged. Only affects shapes with `backdrop_blur > 0.0`.
    pub backdrop_hue_shift: f32,
    /// Point to rotate around, in logical pixels measured from the shape
    /// centre. `[0.0, 0.0]` (default) rotates around the centre. Positive X
    /// is right, positive Y is down, matching the screen-space axes. Applies
    /// together with [`Self::rotation`] on solid (non-textured) shapes.
    pub rotation_pivot: [f32; 2],
    /// Stacked outer shadow layers, drawn behind the fill in order (first
    /// entry furthest back). Up to [`OVERLAY_MAX_SHADOW_LAYERS`] are honoured.
    ///
    /// When non-empty this replaces the single legacy `shadow_*` outer
    /// shadow. Only the solid (non-textured, non-blur) shape path draws
    /// stacked layers; textured and backdrop-blur shapes fall back to the
    /// single legacy `shadow_*` shadow.
    pub shadows: Vec<ShadowLayer>,
    /// Stacked inner (inset) shadow layers, drawn on top of the fill and
    /// under the border, in order. Up to [`OVERLAY_MAX_SHADOW_LAYERS`] are
    /// honoured. Same path limitation as [`Self::shadows`].
    pub inner_shadows: Vec<ShadowLayer>,
}

impl Default for OverlayShapeItem {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            size: [100.0, 100.0],
            shape: OverlayShape::default(),
            fill: OverlayFill::default(),
            opacity: 1.0,
            border_colour: [1.0, 1.0, 1.0, 1.0],
            border_width: 0.0,
            border_mode: BorderMode::Inset,
            z_order: 0,
            texture: None,
            shadow_colour: [0.0, 0.0, 0.0, 0.0],
            shadow_radius: 0.0,
            shadow_offset: [0.0, 0.0],
            animation: OverlayAnimation::None,
            backdrop_blur: 0.0,
            clip_mask_id: None,
            clip_id: None,
            rotation: 0.0,
            nine_slice: None,
            shadow_inset: false,
            texture_transform: TextureTransform::default(),
            animations: OverlayAnimations::default(),
            backdrop_saturation: 1.0,
            backdrop_brightness: 1.0,
            backdrop_hue_shift: 0.0,
            rotation_pivot: [0.0, 0.0],
            shadows: Vec::new(),
            inner_shadows: Vec::new(),
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
            position,
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
    /// `viewport_size` is the current logical viewport size. Pair `texture` with
    /// a streaming `OverlayTextureId` updated each frame for a live image, or a
    /// static uploaded one for a fixed image. The fill is left as a white tint,
    /// so the texture is drawn unmodified; set `opacity` or a non-white fill
    /// afterwards to tint or fade it.
    pub fn textured_image(
        texture: OverlayTextureId,
        natural_size: [f32; 2],
        scale: f32,
        anchor: ImageAnchor,
        viewport_size: [f32; 2],
    ) -> Self {
        let size = [natural_size[0] * scale, natural_size[1] * scale];
        let position = match anchor {
            ImageAnchor::TopLeft => [0.0, 0.0],
            ImageAnchor::TopRight => [viewport_size[0] - size[0], 0.0],
            ImageAnchor::BottomLeft => [0.0, viewport_size[1] - size[1]],
            ImageAnchor::BottomRight => [viewport_size[0] - size[0], viewport_size[1] - size[1]],
            ImageAnchor::Center => [
                (viewport_size[0] - size[0]) * 0.5,
                (viewport_size[1] - size[1]) * 0.5,
            ],
        };
        Self::new(OverlayShape::Rect { corner_radius: 0.0 }, position, size)
            .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
            .with_texture(texture)
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

    /// Set the fill style (solid colour or gradient).
    pub fn with_fill(mut self, fill: OverlayFill) -> Self {
        self.fill = fill;
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set the border colour and width. A width of `0.0` disables the border.
    pub fn with_border(mut self, colour: [f32; 4], width: f32) -> Self {
        self.border_colour = colour;
        self.border_width = width;
        self
    }

    /// Set where the border sits relative to the shape edge.
    pub fn with_border_mode(mut self, mode: BorderMode) -> Self {
        self.border_mode = mode;
        self
    }

    /// Set the draw order. Lower values render first (further back).
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Fill the shape with an uploaded overlay texture, clipped by the SDF.
    pub fn with_texture(mut self, texture: OverlayTextureId) -> Self {
        self.texture = Some(texture);
        self
    }

    /// Set the outer (or inset) shadow colour, blur radius, and offset.
    pub fn with_shadow(mut self, colour: [f32; 4], radius: f32, offset: [f32; 2]) -> Self {
        self.shadow_colour = colour;
        self.shadow_radius = radius;
        self.shadow_offset = offset;
        self
    }

    /// Render the shadow as an inner (inset) shadow instead of an outer one.
    pub fn with_shadow_inset(mut self, inset: bool) -> Self {
        self.shadow_inset = inset;
        self
    }

    /// Set the backdrop blur radius (frosted-glass effect) in logical pixels.
    pub fn with_backdrop_blur(mut self, radius: f32) -> Self {
        self.backdrop_blur = radius;
        self
    }

    /// Set the rotation around the shape centre, in radians.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }

    /// Set the point to rotate around, in logical pixels from the shape
    /// centre. `[0.0, 0.0]` rotates around the centre (the default).
    pub fn with_rotation_pivot(mut self, pivot: [f32; 2]) -> Self {
        self.rotation_pivot = pivot;
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
        self.backdrop_saturation = saturation;
        self.backdrop_brightness = brightness;
        self.backdrop_hue_shift = hue_shift;
        self
    }

    /// Set the stacked outer shadow layers (drawn behind the fill). Replaces
    /// the single legacy `with_shadow` outer shadow on the solid shape path.
    pub fn with_shadows(mut self, shadows: Vec<ShadowLayer>) -> Self {
        self.shadows = shadows;
        self
    }

    /// Set the stacked inner (inset) shadow layers (drawn on top of the fill,
    /// under the border).
    pub fn with_inner_shadows(mut self, shadows: Vec<ShadowLayer>) -> Self {
        self.inner_shadows = shadows;
        self
    }

    /// Flip the texture fill horizontally and/or vertically. Convenience over
    /// [`TextureTransform::flip_x`] / [`TextureTransform::flip_y`]; sets those
    /// fields on the shape's texture transform.
    pub fn with_texture_flip(mut self, flip_x: bool, flip_y: bool) -> Self {
        self.texture_transform.flip_x = flip_x;
        self.texture_transform.flip_y = flip_y;
        self
    }

    /// Mark this shape as a clip mask with the given id. Other shapes whose
    /// clip id matches are clipped to this shape's bounding box.
    pub fn with_clip_mask(mut self, mask_id: u32) -> Self {
        self.clip_mask_id = Some(mask_id);
        self
    }

    /// Clip this shape to the mask shape with this id. The mask's SDF (not just
    /// its bounding box) is used, and masks may nest.
    pub fn with_clip(mut self, clip_id: u32) -> Self {
        self.clip_id = Some(clip_id);
        self
    }

    /// Set 9-slice sampling for the texture fill.
    pub fn with_nine_slice(mut self, nine_slice: NineSlice) -> Self {
        self.nine_slice = Some(nine_slice);
        self
    }

    /// Set the affine transform applied to the texture sample before lookup.
    pub fn with_texture_transform(mut self, transform: TextureTransform) -> Self {
        self.texture_transform = transform;
        self
    }

    /// Set the single-property opacity animation.
    pub fn with_animation(mut self, animation: OverlayAnimation) -> Self {
        self.animation = animation;
        self
    }

    /// Set the multi-channel animation tracks.
    pub fn with_animations(mut self, animations: OverlayAnimations) -> Self {
        self.animations = animations;
        self
    }

    /// Signed distance from a screen-space point to the shape boundary.
    ///
    /// The point is in logical pixels from the top-left of the viewport (the
    /// same coordinate space as `position`). Negative values mean the point is
    /// inside the shape; positive values mean it is outside.
    ///
    /// This evaluates the same SDF used by the GPU shader, so the boundary
    /// matches what is rendered on screen (ignoring sub-pixel AA).
    pub fn distance(&self, point: [f32; 2]) -> f32 {
        let hw = self.size[0] * 0.5;
        let hh = self.size[1] * 0.5;
        let cx = self.position[0] + hw;
        let cy = self.position[1] + hh;
        let dx = point[0] - cx;
        let dy = point[1] - cy;
        // Rotate the query point by -rotation around the rotation pivot (an
        // offset from the shape centre) so the SDF evaluates in the unrotated
        // frame, matching the fragment shader. With a zero pivot this reduces
        // to rotation around the centre.
        let c = (-self.rotation).cos();
        let s = (-self.rotation).sin();
        let piv = self.rotation_pivot;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape_at(x: f32, y: f32, w: f32, h: f32, shape: OverlayShape) -> OverlayShapeItem {
        OverlayShapeItem {
            position: [x, y],
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
            position: [0.0, 30.0],
            size: [100.0, 40.0],
            shape: OverlayShape::Capsule,
            ..Default::default()
        };
        assert!(!s.contains([50.0, 80.0]));
        s.rotation = std::f32::consts::FRAC_PI_2;
        assert!(s.contains([50.0, 80.0]));
    }

    #[test]
    fn rotation_pivot_shifts_hit_boundary() {
        // 80x40 rectangle at the origin: centre (40, 20). Rotating 90 degrees
        // about the centre keeps the centre fixed. Rotating about a pivot far
        // from the centre swings the whole shape elsewhere, so a point that is
        // inside under centre-rotation falls outside under pivot-rotation.
        let mut s = OverlayShapeItem {
            position: [0.0, 0.0],
            size: [80.0, 40.0],
            shape: OverlayShape::Rect { corner_radius: 0.0 },
            rotation: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        // Centre is always inside regardless of pivot.
        assert!(s.contains([40.0, 20.0]));
        // With a large pivot offset the shape rotates away from the centre.
        s.rotation_pivot = [200.0, 0.0];
        assert!(!s.contains([40.0, 20.0]));
    }

    #[test]
    fn rotation_pivot_zero_matches_centre_rotation() {
        // A zero pivot must reproduce plain centre rotation exactly.
        let base = OverlayShapeItem {
            position: [10.0, 10.0],
            size: [100.0, 30.0],
            shape: OverlayShape::Capsule,
            rotation: 0.7,
            ..Default::default()
        };
        let mut piv = base.clone();
        piv.rotation_pivot = [0.0, 0.0];
        for pt in [[60.0, 25.0], [20.0, 20.0], [90.0, 40.0]] {
            assert!((base.distance(pt) - piv.distance(pt)).abs() < 1e-4);
        }
    }
}
