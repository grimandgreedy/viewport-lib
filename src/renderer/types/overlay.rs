use super::*;

// ---------------------------------------------------------------------------
// OverlayFrame and overlay item stubs
// ---------------------------------------------------------------------------

/// Handle to a texture uploaded via `ViewportGpuResources::upload_overlay_texture`.
///
/// Pass this to `OverlayShapeItem::texture` to use the image as fill. The
/// handle remains valid for the lifetime of the `ViewportGpuResources` it
/// came from; using it after the resources are dropped is a logic error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OverlayTextureId(pub(crate) u64);

/// Horizontal alignment of a label relative to its anchor point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LabelAnchor {
    /// Text positioned to the right of the anchor (default).
    #[default]
    Leading,
    /// Text centered horizontally on the anchor.
    Center,
    /// Text positioned to the left of the anchor.
    Trailing,
}

/// A text label rendered as a screen-space overlay.
///
/// Anchored to a world-space or screen-space position with optional leader line
/// and background box.
///
/// # Anchoring
///
/// Set `world_anchor` to pin the label to a 3D position that is reprojected
/// each frame.  Set `screen_anchor` for a fixed screen position in logical
/// pixels from the top-left corner.  When both are set, `screen_anchor` takes
/// precedence.  World-anchored labels are frustum-culled: they are not drawn
/// when the anchor is behind the camera or outside the viewport.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::LabelItem;
/// let label = LabelItem {
///     world_anchor: Some([2.0, 3.0, 0.0]),
///     text: "Peak Pressure: 101.3 kPa".into(),
///     leader_line: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LabelItem {
    /// World-space anchor.  Projected to screen by the renderer each frame.
    /// Set `screen_anchor` instead for fixed screen positions.
    pub world_anchor: Option<[f32; 3]>,

    /// Screen-space anchor in logical pixels from top-left.
    /// Takes precedence over `world_anchor` when both are set.
    pub screen_anchor: Option<[f32; 2]>,

    /// Text content to display.
    pub text: String,

    /// RGBA text colour in linear float format.
    pub colour: [f32; 4],

    /// Font size in logical pixels.
    pub font_size: f32,

    /// Font to use.  `None` uses the built-in default font.
    pub font: Option<crate::resources::font::FontHandle>,

    /// Draw a filled rectangle behind the text.
    pub background: bool,

    /// RGBA colour of the background rectangle.
    pub background_colour: [f32; 4],

    /// Padding between the text and the background rectangle edge in logical
    /// pixels.  Only used when `background` is `true`.  Default: `3.0`.
    pub padding: f32,

    /// Draw a line from the projected `world_anchor` to the label text origin.
    /// Only drawn when `world_anchor` is set.
    pub leader_line: bool,

    /// RGBA colour of the leader line.
    pub leader_colour: [f32; 4],

    /// Horizontal alignment of the label text relative to its anchor.
    pub anchor_align: LabelAnchor,

    /// Pixel offset applied after anchor resolution and alignment.
    /// Useful for nudging a label away from its anchor without moving the
    /// leader line endpoint.  Default: `[0.0, 0.0]`.
    pub offset: [f32; 2],

    /// Overall opacity multiplier applied to text, background, and leader
    /// line colours.  Range 0.0 (invisible) to 1.0 (fully opaque).
    pub opacity: f32,

    /// Maximum text width in logical pixels.  When set, text that exceeds
    /// this width is wrapped to multiple lines.  `None` disables wrapping.
    pub max_width: Option<f32>,

    /// Corner radius of the background rectangle in logical pixels.
    /// Only used when `background` is `true`.  Default: `0.0` (sharp corners).
    pub border_radius: f32,

    /// Explicit draw order.  Labels with lower values are drawn first
    /// (further back).  Labels with equal `z_order` are drawn in list order.
    pub z_order: i32,

    /// Reserved for depth-based occlusion.  Not implemented yet: when `true`
    /// the label is still rendered; behaviour will be defined in a follow-up.
    pub occlude: bool,
}

impl Default for LabelItem {
    fn default() -> Self {
        Self {
            world_anchor: None,
            screen_anchor: None,
            text: String::new(),
            colour: [1.0, 1.0, 1.0, 1.0],
            font_size: 14.0,
            font: None,
            background: false,
            background_colour: [0.0, 0.0, 0.0, 0.55],
            padding: 3.0,
            leader_line: false,
            leader_colour: [1.0, 1.0, 1.0, 0.6],
            anchor_align: LabelAnchor::Leading,
            offset: [0.0, 0.0],
            opacity: 1.0,
            max_width: None,
            border_radius: 0.0,
            z_order: 0,
            occlude: false,
        }
    }
}

/// Corner of the viewport where a [`ScalarBarItem`] is anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalarBarAnchor {
    /// Top-left corner of the viewport.
    TopLeft,
    /// Top-right corner of the viewport.
    TopRight,
    /// Bottom-left corner of the viewport.
    BottomLeft,
    /// Bottom-right corner of the viewport (default).
    #[default]
    BottomRight,
}

/// Long-axis orientation of a [`ScalarBarItem`] gradient strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalarBarOrientation {
    /// Gradient runs top (max) to bottom (min).  Default.
    #[default]
    Vertical,
    /// Gradient runs left (min) to right (max).
    Horizontal,
}

/// A colour-legend (scalar bar) rendered as a screen-space overlay.
///
/// References an already-uploaded [`crate::resources::ColourmapId`] and draws a
/// gradient strip with evenly-spaced tick labels directly in the overlay pass,
/// without requiring any application-side painting.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::{ScalarBarItem, ScalarBarAnchor, ScalarBarOrientation};
/// let bar = ScalarBarItem {
///     colourmap_id: viewport_lib::ColourmapId(0),
///     scalar_min: 0.0,
///     scalar_max: 1.0,
///     title: Some("Height (m)".into()),
///     anchor: ScalarBarAnchor::BottomRight,
///     orientation: ScalarBarOrientation::Vertical,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ScalarBarItem {
    /// Colourmap to sample for the gradient strip.
    pub colourmap_id: crate::resources::ColourmapId,

    /// Scalar value at the low end (bottom or left) of the gradient.
    pub scalar_min: f32,

    /// Scalar value at the high end (top or right) of the gradient.
    pub scalar_max: f32,

    /// Optional title drawn above the gradient strip.
    pub title: Option<String>,

    /// Viewport corner to anchor the bar to.
    pub anchor: ScalarBarAnchor,

    /// Long-axis orientation of the gradient strip.
    pub orientation: ScalarBarOrientation,

    /// Short-axis size of the gradient strip in logical pixels.  Default: `20.0`.
    pub bar_width_px: f32,

    /// Long-axis size of the gradient strip in logical pixels.  Default: `200.0`.
    pub bar_length_px: f32,

    /// Distance from the viewport edge in logical pixels.  Default: `16.0`.
    pub margin_px: f32,

    /// Font to use for tick labels and title.  `None` uses the built-in default.
    pub font: Option<crate::resources::font::FontHandle>,

    /// Font size for tick labels and title in logical pixels.  Default: `12.0`.
    pub font_size: f32,

    /// RGBA colour for tick labels and title.  Default: white.
    pub label_colour: [f32; 4],

    /// Number of evenly-spaced labelled ticks (including min and max).  Default: `5`.
    pub tick_count: u32,

    /// RGBA background box colour (including alpha).
    ///
    /// Default: semi-transparent black `[0.0, 0.0, 0.0, 0.63]`.
    pub background_colour: [f32; 4],

    /// Reverse the value direction of the gradient.
    ///
    /// When `false` (default): vertical bars run max-at-top / min-at-bottom;
    /// horizontal bars run min-at-left / max-at-right.
    /// When `true` the direction is flipped for both orientations.
    pub ticks_reversed: bool,

    /// Font size used exclusively for the title text.
    ///
    /// `None` (default) falls back to `font_size`.
    pub title_font_size: Option<f32>,
}

impl Default for ScalarBarItem {
    fn default() -> Self {
        Self {
            colourmap_id: crate::resources::ColourmapId(0),
            scalar_min: 0.0,
            scalar_max: 1.0,
            title: None,
            anchor: ScalarBarAnchor::BottomRight,
            orientation: ScalarBarOrientation::Vertical,
            bar_width_px: 20.0,
            bar_length_px: 200.0,
            margin_px: 16.0,
            font: None,
            font_size: 12.0,
            label_colour: [1.0, 1.0, 1.0, 1.0],
            tick_count: 5,
            background_colour: [0.0, 0.0, 0.0, 0.63],
            ticks_reversed: false,
            title_font_size: None,
        }
    }
}

/// A two-point measurement overlay that displays the distance between two
/// world-space positions, with a distance readout at the segment midpoint.
///
/// Both endpoints are projected each frame by the renderer; the item culls
/// cleanly when both endpoints are behind the camera.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::RulerItem;
/// let ruler = RulerItem {
///     start: [0.0, 0.0, 0.0],
///     end: [2.5, 0.0, 0.0],
///     colour: [1.0, 1.0, 1.0, 1.0],
///     label_format: Some("{:.2} m".into()),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RulerItem {
    /// World-space start endpoint.
    pub start: [f32; 3],
    /// World-space end endpoint.
    pub end: [f32; 3],
    /// RGBA colour for the ruler line and end caps. Default: white.
    pub colour: [f32; 4],
    /// Line thickness in screen pixels. Default: `1.5`.
    pub line_width_px: f32,
    /// Font for the distance label. `None` = built-in default.
    pub font: Option<crate::resources::FontHandle>,
    /// Font size for the distance label in logical pixels. Default: `13.0`.
    pub font_size: f32,
    /// RGBA colour for the distance label text. Default: white.
    pub label_colour: [f32; 4],
    /// Format string for the distance value using Rust `format!` syntax.
    ///
    /// The `{}` placeholder is replaced with the computed distance.
    /// Accepts precision specifiers like `"{:.3}"` or unit suffixes like
    /// `"{:.2} m"`. Default (`None`): `"{:.3}"` (3 decimal places).
    pub label_format: Option<String>,
    /// Draw small perpendicular tick marks at each endpoint. Default: `true`.
    pub end_caps: bool,
}

impl Default for RulerItem {
    fn default() -> Self {
        Self {
            start: [0.0; 3],
            end: [1.0, 0.0, 0.0],
            colour: [1.0, 1.0, 1.0, 1.0],
            line_width_px: 1.5,
            font: None,
            font_size: 13.0,
            label_colour: [1.0, 1.0, 1.0, 1.0],
            label_format: None,
            end_caps: true,
        }
    }
}

/// A pixel image composited over the viewport in screen space.
///
/// Unlike [`ScreenImageItem`] (which lives in [`SceneFrame`] and supports
/// depth compositing with world geometry), `OverlayImageItem` is a pure
/// screen-space overlay with no depth field. It renders after post-processing,
/// on top of all scene content, labels, scalar bars, and rulers.
///
/// Consumers using [`ScreenImageItem`] without a `depth` buffer (corner logos,
/// diagnostic HUDs, watermarks) should migrate to `OverlayImageItem`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct OverlayImageItem {
    /// RGBA8 pixel data, row-major, top-to-bottom.
    pub pixels: Vec<[u8; 4]>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Which corner (or center) of the viewport to anchor the image to.
    pub anchor: ImageAnchor,
    /// Scale factor relative to natural pixel size (`1.0` = one pixel per screen pixel).
    pub scale: f32,
    /// Overall opacity multiplier applied on top of per-pixel alpha. Default: `1.0`.
    pub alpha: f32,
}

impl Default for OverlayImageItem {
    fn default() -> Self {
        Self {
            pixels: Vec::new(),
            width: 0,
            height: 0,
            anchor: ImageAnchor::TopLeft,
            scale: 1.0,
            alpha: 1.0,
        }
    }
}

/// Anchor position for a [`LoadingBarItem`] overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadingBarAnchor {
    /// Anchored at the top center of the viewport.
    TopCenter,
    /// Anchored at the center of the viewport.
    Center,
    /// Anchored at the bottom center of the viewport (default).
    #[default]
    BottomCenter,
}

/// A progress bar drawn over the viewport in screen space.
///
/// Render via [`OverlayFrame::loading_bars`].
///
/// ```no_run
/// use viewport_lib::{LoadingBarItem, LoadingBarAnchor};
/// let bar = LoadingBarItem {
///     progress: 0.42,
///     label: Some("Building scene… 420 000 / 1 000 000".into()),
///     anchor: LoadingBarAnchor::BottomCenter,
///     ..LoadingBarItem::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LoadingBarItem {
    /// Progress fraction in [0.0, 1.0].
    pub progress: f32,
    /// Optional label displayed above (or below for `TopCenter`) the bar.
    pub label: Option<String>,
    /// Viewport anchor for the bar.
    pub anchor: LoadingBarAnchor,
    /// Bar width in logical pixels.
    pub width_px: f32,
    /// Bar height in logical pixels.
    pub height_px: f32,
    /// Distance from the anchored viewport edge in logical pixels.
    pub margin_px: f32,
    /// Background (unfilled) colour.
    pub background_colour: [f32; 4],
    /// Fill (progress) colour.
    pub fill_colour: [f32; 4],
    /// Label text colour.
    pub label_colour: [f32; 4],
    /// Font size for the label in logical pixels.
    pub font_size: f32,
    /// Corner radius of the bar rectangles in logical pixels.
    pub corner_radius: f32,
    /// Font for the label text. `None` uses the built-in default font.
    pub font: Option<crate::resources::font::FontHandle>,
}

impl Default for LoadingBarItem {
    fn default() -> Self {
        Self {
            progress: 0.0,
            label: None,
            anchor: LoadingBarAnchor::default(),
            width_px: 300.0,
            height_px: 16.0,
            margin_px: 24.0,
            background_colour: [0.12, 0.12, 0.12, 0.88],
            fill_colour: [0.22, 0.60, 1.0, 1.0],
            label_colour: [1.0, 1.0, 1.0, 1.0],
            font_size: 13.0,
            corner_radius: 4.0,
            font: None,
        }
    }
}

/// A solid filled rectangle rendered in screen space.
///
/// Position and size are in logical pixels from the top-left corner of the
/// viewport. Renders after post-processing, in the same pass as other overlay
/// items, before labels so it can serve as a background panel.
///
/// The border is drawn as a slightly expanded rect behind the fill, so
/// `border_width` pixels of it are visible on each side of the fill rect.
/// Set `border_width` to `0.0` for no border.
#[derive(Debug, Clone)]
pub struct OverlayRectItem {
    /// Top-left position in logical pixels from the viewport top-left.
    pub position: [f32; 2],
    /// Width and height in logical pixels.
    pub size: [f32; 2],
    /// RGBA fill colour in linear float format.
    pub colour: [f32; 4],
    /// Overall opacity multiplier applied to both fill and border. Range 0.0-1.0.
    pub opacity: f32,
    /// Corner radius in logical pixels. `0.0` produces sharp corners.
    pub corner_radius: f32,
    /// RGBA border colour in linear float format.
    pub border_colour: [f32; 4],
    /// Border thickness in logical pixels. `0.0` disables the border.
    pub border_width: f32,
    /// Draw order relative to other rects. Lower values render first (further back).
    pub z_order: i32,
}

impl Default for OverlayRectItem {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            size: [100.0, 100.0],
            colour: [0.0, 0.0, 0.0, 0.55],
            opacity: 1.0,
            corner_radius: 0.0,
            border_colour: [1.0, 1.0, 1.0, 1.0],
            border_width: 0.0,
            z_order: 0,
        }
    }
}

/// Fill style for an [`OverlayShapeItem`].
///
/// `Solid` is the default and matches the previous single-colour behaviour.
/// `LinearGradient` interpolates between two colours along an angled axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlayFill {
    /// Uniform solid colour in linear RGBA float format.
    Solid([f32; 4]),
    /// Linear gradient between two colours.
    ///
    /// The gradient runs along `angle` across the bounding box. `angle = 0.0`
    /// goes left-to-right (`start_colour` on the left, `end_colour` on the
    /// right). Positive angles rotate the direction counter-clockwise in math
    /// coordinates; because screen Y points downward, `angle = PI/2` produces
    /// a top-to-bottom gradient (start at top, end at bottom).
    LinearGradient {
        /// RGBA colour at the start of the gradient (left when angle is 0).
        start_colour: [f32; 4],
        /// RGBA colour at the end of the gradient (right when angle is 0).
        end_colour: [f32; 4],
        /// Gradient direction in radians. `0.0` = left-to-right.
        angle: f32,
    },
}

impl Default for OverlayFill {
    fn default() -> Self {
        OverlayFill::Solid([0.0, 0.0, 0.0, 0.55])
    }
}

/// Shape type for an `OverlayShapeItem`.
///
/// Each variant maps to a signed-distance function evaluated per fragment
/// on the GPU. The bounding quad is defined by `OverlayShapeItem::position`
/// and `size`; the shape variant controls which SDF is used and how the
/// extra `radii` parameters are interpreted.
#[derive(Debug, Clone, Copy, PartialEq)]
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
}

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
/// let panel = OverlayShapeItem {
///     position: [20.0, 20.0],
///     size: [300.0, 200.0],
///     shape: OverlayShape::Rect { corner_radius: 8.0 },
///     fill: OverlayFill::Solid([0.1, 0.1, 0.1, 0.85]),
///     border_width: 1.0,
///     border_colour: [0.4, 0.4, 0.4, 1.0],
///     ..Default::default()
/// };
///
/// // Circle with a left-to-right gradient.
/// let grad_dot = OverlayShapeItem {
///     position: [100.0, 100.0],
///     size: [60.0, 60.0],
///     shape: OverlayShape::Circle,
///     fill: OverlayFill::LinearGradient {
///         start_colour: [0.0, 0.4, 1.0, 1.0],
///         end_colour: [0.0, 1.0, 0.5, 1.0],
///         angle: 0.0,
///     },
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
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
    /// Draw order relative to other shapes. Lower values render first (further back).
    pub z_order: i32,
    /// Optional texture fill. When set the shape samples the image uploaded
    /// via `ViewportGpuResources::upload_overlay_texture`, clipped by the SDF
    /// boundary. `fill` acts as a tint when this is `Some`.
    pub texture: Option<OverlayTextureId>,
    /// RGBA colour of the outer shadow/glow halo. Default: transparent (no shadow).
    pub shadow_colour: [f32; 4],
    /// Blur spread of the shadow in logical pixels. `0.0` disables the shadow.
    pub shadow_radius: f32,
    /// Offset of the shadow centre from the shape centre in logical pixels.
    /// Positive X shifts right, positive Y shifts down. Default: `[0.0, 0.0]`.
    pub shadow_offset: [f32; 2],
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
            z_order: 0,
            texture: None,
            shadow_colour: [0.0, 0.0, 0.0, 0.0],
            shadow_radius: 0.0,
            shadow_offset: [0.0, 0.0],
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
    let inner = (p[0].abs() - b[0] + chosen).max(p[1].abs() - b[1] + chosen).min(0.0);
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
        t = if wlen > 0.0 { [wc[0] / wlen, wc[1] / wlen] } else { t };
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
        let p = [point[0] - cx, point[1] - cy];
        let hs = [hw, hh];

        match self.shape {
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
                start_angle,
                end_angle,
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
        let s = shape_at(100.0, 100.0, 80.0, 60.0, OverlayShape::Rect { corner_radius: 0.0 });
        assert!(s.contains([140.0, 130.0])); // centre
        assert!(s.distance([140.0, 130.0]) < 0.0);
    }

    #[test]
    fn rect_outside() {
        let s = shape_at(100.0, 100.0, 80.0, 60.0, OverlayShape::Rect { corner_radius: 0.0 });
        assert!(!s.contains([50.0, 130.0])); // left of shape
        assert!(!s.contains([200.0, 130.0])); // right of shape
    }

    #[test]
    fn rect_edge_distance() {
        let s = shape_at(0.0, 0.0, 100.0, 100.0, OverlayShape::Rect { corner_radius: 0.0 });
        // Centre is at (50, 50), half-size 50x50. Point on the right edge:
        let d = s.distance([100.0, 50.0]);
        assert!(d.abs() < 0.01, "edge distance should be ~0, got {d}");
    }

    #[test]
    fn rounded_rect_corner_is_outside() {
        let s = shape_at(0.0, 0.0, 100.0, 100.0, OverlayShape::Rect { corner_radius: 20.0 });
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
        let s = shape_at(0.0, 0.0, 100.0, 100.0, OverlayShape::Ring { inner_radius_frac: 0.7 });
        // Centre of the ring (the hole) should be outside.
        assert!(!s.contains([50.0, 50.0]));
        // Point in the wall area should be inside.
        assert!(s.contains([50.0, 8.0]));
    }

    #[test]
    fn arc_inside_sweep() {
        let s = shape_at(
            0.0, 0.0, 100.0, 100.0,
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
        let s = shape_at(0.0, 0.0, 60.0, 60.0, OverlayShape::Triangle {
            direction: TriangleDirection::Up,
        });
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
            let s = shape_at(0.0, 0.0, 60.0, 60.0, OverlayShape::Triangle { direction: dir });
            // Centre-ish should always be inside.
            assert!(s.contains([30.0, 30.0]), "centre should be inside for {dir:?}");
        }
    }

    #[test]
    fn distance_is_negative_inside_positive_outside() {
        let s = shape_at(0.0, 0.0, 100.0, 100.0, OverlayShape::Circle);
        assert!(s.distance([50.0, 50.0]) < 0.0, "centre should be negative");
        assert!(s.distance([0.0, 0.0]) > 0.0, "far corner should be positive");
    }
}

/// Semantic overlays rendered after post-processing: labels, scalar bars,
/// rulers, screen-space images, and loading bars.
///
/// This frame section is the right place for any visual element that belongs
/// in front of the 3D scene and must not be affected by tone-mapping or bloom.
#[derive(Debug, Clone, Default)]
pub struct OverlayFrame {
    /// SDF-based shapes rendered before labels. Supports rounded rects,
    /// circles, ellipses, and capsules with anti-aliased edges and borders.
    pub shapes: Vec<OverlayShapeItem>,
    /// Solid filled rectangles rendered before labels. Useful for panel
    /// backgrounds, scrims, and full-screen fades.
    pub rects: Vec<OverlayRectItem>,
    /// Text labels anchored to world-space or screen-space positions.
    pub labels: Vec<LabelItem>,
    /// Colour-legend (scalar bar) overlays.
    pub scalar_bars: Vec<ScalarBarItem>,
    /// Two-point distance measurement overlays.
    pub rulers: Vec<RulerItem>,
    /// Pixel images composited over the viewport in screen space.
    pub images: Vec<OverlayImageItem>,
    /// Progress bar overlays (loading indicators, progress feedback).
    pub loading_bars: Vec<LoadingBarItem>,
}
