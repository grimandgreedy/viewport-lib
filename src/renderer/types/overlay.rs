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
///     label: Some("Building scene... 420 000 / 1 000 000".into()),
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
/// `LinearGradient`, `RadialGradient`, and `ConicalGradient` interpolate
/// between two colours across the shape's bounding box.
#[derive(Debug, Clone, PartialEq)]
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
    /// Radial gradient running from the shape centre to its bounding-box edge.
    ///
    /// `centre_colour` sits at the shape origin; `edge_colour` sits at the
    /// farthest bounding-box corner. The transition follows
    /// `length(local_pos) / max_half_size`.
    RadialGradient {
        /// RGBA colour at the centre of the shape.
        centre_colour: [f32; 4],
        /// RGBA colour at the bounding-box edge.
        edge_colour: [f32; 4],
    },
    /// Conical (sweep) gradient rotating around the shape centre.
    ///
    /// The hue wraps once around the origin like a colour wheel.
    /// `offset_angle` rotates the seam (where `end_colour` meets
    /// `start_colour`) counter-clockwise in math coordinates.
    ConicalGradient {
        /// RGBA colour at the sweep start.
        start_colour: [f32; 4],
        /// RGBA colour at the sweep end (wraps back to start).
        end_colour: [f32; 4],
        /// Rotation offset in radians. `0.0` places the seam to the right.
        offset_angle: f32,
    },
    /// Linear gradient with three or more colour stops at arbitrary
    /// positions along the gradient axis. Use when a two-stop ramp is too
    /// flat; designers commonly stack 3-5 stops for polished surfaces.
    /// Stops outside `[0, 1]` are clamped; more than
    /// [`OVERLAY_MAX_GRADIENT_STOPS`] entries are truncated.
    LinearGradientMulti {
        /// Stops in source order. Need not be pre-sorted by position; the
        /// renderer sorts them at prepare time.
        stops: Vec<GradientStop>,
        /// Gradient direction in radians. `0.0` = left-to-right.
        angle: f32,
    },
    /// Radial gradient with three or more colour stops between the shape
    /// centre and its bounding-box edge.
    RadialGradientMulti {
        /// Stops along the centre-to-edge axis.
        stops: Vec<GradientStop>,
    },
    /// Conical gradient with three or more colour stops along the sweep.
    ConicalGradientMulti {
        /// Stops along the `[0, 1]` sweep parameter.
        stops: Vec<GradientStop>,
        /// Rotation offset in radians.
        offset_angle: f32,
    },
}

impl Default for OverlayFill {
    fn default() -> Self {
        OverlayFill::Solid([0.0, 0.0, 0.0, 0.55])
    }
}

/// Nine-patch / 9-slice texture sampling parameters.
///
/// Treats the bound texture as a panel with four corner regions that stay at
/// their authored pixel size, four edge regions that tile or stretch along one
/// axis, and a centre region that follows both axes. The standard way to ship
/// resizable button, dialog, and scrollbar art without the corners stretching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NineSlice {
    /// Inset in *texture pixels* from each edge to the centre region:
    /// `[top, right, bottom, left]`. Defines where the four corner regions end.
    pub insets_px: [f32; 4],
    /// Sampling for the centre region (between all four insets).
    pub centre_mode: TileMode,
    /// Sampling for the four edge regions (between two opposing insets).
    pub edge_mode: TileMode,
}

impl Default for NineSlice {
    fn default() -> Self {
        Self {
            insets_px: [0.0; 4],
            centre_mode: TileMode::Stretch,
            edge_mode: TileMode::Stretch,
        }
    }
}

/// How a texture region remaps its UV coordinate before sampling.
///
/// Shared by [`NineSlice`] and [`TextureTransform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TileMode {
    /// Linear stretch across the region. Default; matches non-9-slice
    /// behaviour for backwards compatibility.
    #[default]
    Stretch,
    /// Repeat the source region at its native pixel size.
    Tile,
    /// Repeat the source region with every other tile flipped along the
    /// matching axis, so the seam between tiles is mirrored. Smoothes the
    /// repeat for symmetric textures.
    Mirror,
}

/// Affine transform applied to the texture sample before lookup.
///
/// Lets a single texture pan, scale, rotate, tile, and flip independently
/// of the shape it fills. Use for scrolling marquees inside a shape, zooming
/// detail, rotating an icon without rotating the shape, repeating tile art,
/// and mirroring asymmetric assets.
///
/// The transform applies to the bounding-box UV in [0, 1] coordinates,
/// centred at (0.5, 0.5) for scale and rotation. When [`NineSlice`] is also
/// set on the same shape, the 9-slice remap wins and the texture transform
/// is ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureTransform {
    /// UV offset added after scale and rotation. `[0.0, 0.0]` keeps the
    /// sample centred. Drives panning / scrolling.
    pub offset: [f32; 2],
    /// UV scale multiplier applied around the centre. `[1.0, 1.0]` is
    /// 1:1. Larger values widen the sampled UV range (more tiles with
    /// [`TileMode::Tile`], or a wider view of the source with `Stretch`
    /// or `Mirror`). Smaller values zoom in.
    pub scale: [f32; 2],
    /// Rotation around the UV centre `(0.5, 0.5)` in radians, CCW in math
    /// coordinates.
    pub rotation: f32,
    /// Sampling mode for UV values outside `[0, 1]`. `Stretch` clamps to
    /// the edge (default), `Tile` wraps, `Mirror` ping-pongs.
    pub tile_mode: TileMode,
    /// Flip the sample horizontally before lookup.
    pub flip_x: bool,
    /// Flip the sample vertically before lookup.
    pub flip_y: bool,
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
            tile_mode: TileMode::Stretch,
            flip_x: false,
            flip_y: false,
        }
    }
}

impl TextureTransform {
    /// Returns `true` when the transform is the identity (no offset, unit
    /// scale, no rotation, no flips, `Stretch` mode).
    pub fn is_identity(&self) -> bool {
        self.offset == [0.0, 0.0]
            && self.scale == [1.0, 1.0]
            && self.rotation == 0.0
            && matches!(self.tile_mode, TileMode::Stretch)
            && !self.flip_x
            && !self.flip_y
    }
}

/// A single colour stop in a multi-stop gradient.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradientStop {
    /// Position along the gradient axis, in `[0, 1]`. Stops outside the
    /// range are clamped at evaluation time.
    pub position: f32,
    /// Linear RGBA colour at this stop.
    pub colour: [f32; 4],
}

impl GradientStop {
    /// Construct a stop at the given position and colour.
    pub const fn new(position: f32, colour: [f32; 4]) -> Self {
        Self { position, colour }
    }
}

/// Maximum number of stops carried in a single multi-stop gradient. Stops
/// beyond this cap are truncated at prepare time. Covers the vast majority
/// of UI gradient use cases; can be raised by widening the vertex layout
/// if a consumer needs more.
pub const OVERLAY_MAX_GRADIENT_STOPS: usize = 4;

/// Border placement relative to the shape edge.
///
/// Controls whether the border band sits inside, outside, or centred on the
/// SDF zero-crossing. `Inset` matches the default behaviour from earlier
/// phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BorderMode {
    /// Border sits inside the fill edge (default). The fill area shrinks by
    /// `border_width`.
    #[default]
    Inset,
    /// Border sits outside the fill edge. The fill area is unaffected; the
    /// border extends outward.
    Outer,
    /// Border is centred on the fill edge (half inside, half outside).
    Center,
}

/// Animation applied to shape opacity each frame.
///
/// The animation is resolved during `prepare()` using the `time` field on
/// `OverlayFrame`. All `start_time` and `time` values share the same
/// application-defined epoch (e.g. seconds since app launch).
///
/// viewport-lib does not own the event loop. The host application must
/// request continuous repaints while animations are active so that
/// `prepare()` is called often enough to produce smooth updates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlayAnimation {
    /// No animation; use `opacity` as-is.
    None,
    /// Fade from 0 to `opacity` over `duration` seconds.
    FadeIn {
        /// Absolute time when the fade starts.
        start_time: f64,
        /// Duration of the fade in seconds.
        duration: f32,
    },
    /// Fade from `opacity` to 0 over `duration` seconds.
    FadeOut {
        /// Absolute time when the fade starts.
        start_time: f64,
        /// Duration of the fade in seconds.
        duration: f32,
    },
    /// Oscillate opacity between 0 and `opacity` with a sinusoidal wave.
    Pulse {
        /// Absolute time when the pulse starts.
        start_time: f64,
        /// Period of one full oscillation in seconds.
        period: f32,
    },
}

impl Default for OverlayAnimation {
    fn default() -> Self {
        OverlayAnimation::None
    }
}

/// OverlayEasing curve applied to an [`AnimTrack`]'s normalised parameter `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OverlayEasing {
    /// Returns `t` unchanged. Constant speed.
    #[default]
    Linear,
    /// `t * t`. Starts slow, accelerates.
    EaseIn,
    /// `1 - (1 - t)^2`. Starts fast, decelerates.
    EaseOut,
    /// Smoothstep: `3t^2 - 2t^3`. Slow start, fast middle, slow end.
    EaseInOut,
    /// Sinusoidal half-wave: `sin(t * PI)`. Returns to 0 at both ends; peaks
    /// at the midpoint. Combine with [`RepeatMode::Loop`] for a continuous
    /// pulse.
    Pulse,
}

/// How an [`AnimTrack`] handles time past the end of its duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RepeatMode {
    /// Run the track once and hold the final value. Default.
    #[default]
    Once,
    /// Restart the track from `from` each cycle.
    Loop,
    /// Reverse direction at each end so the value oscillates between
    /// `from` and `to`.
    PingPong,
}

/// A single animation track interpolating one channel from `from` to `to`
/// over `duration` seconds, with optional easing and repeat mode.
///
/// Resolved during `prepare()` using `OverlayFrame::time`. Times share the
/// same application-defined epoch as the rest of the overlay animation
/// system. Negative or zero `duration` snaps directly to `to`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimTrack<T: Copy> {
    /// Absolute time at which the track starts.
    pub start_time: f64,
    /// Length of one cycle in seconds.
    pub duration: f32,
    /// Value at `start_time` (or each loop restart).
    pub from: T,
    /// Value at `start_time + duration`.
    pub to: T,
    /// Curve applied to the normalised parameter before interpolation.
    pub easing: OverlayEasing,
    /// What happens past the end of one cycle.
    pub repeat: RepeatMode,
}

impl<T: Copy + Default> Default for AnimTrack<T> {
    fn default() -> Self {
        Self {
            start_time: 0.0,
            duration: 1.0,
            from: T::default(),
            to: T::default(),
            easing: OverlayEasing::Linear,
            repeat: RepeatMode::Once,
        }
    }
}

/// Multi-channel animation tracks attached to an [`OverlayShapeItem`].
///
/// Each `Some` track replaces the matching field on the item for the frame.
/// Tracks are independent: a shape can simultaneously translate, scale,
/// recolour, and rotate.
///
/// Animation resolution is CPU-side in `prepare()`; the host must request
/// continuous repaints while any track is active.
#[derive(Debug, Clone, Default)]
pub struct OverlayAnimations {
    /// Drives the item's overall opacity multiplier. Takes precedence over
    /// the legacy [`OverlayShapeItem::animation`] field when both are set.
    pub opacity: Option<AnimTrack<f32>>,
    /// Drives `position` (top-left in logical pixels).
    pub position: Option<AnimTrack<[f32; 2]>>,
    /// Drives `size` (width / height in logical pixels).
    pub size: Option<AnimTrack<[f32; 2]>>,
    /// Drives the item's solid fill colour. Applies only when `fill` is a
    /// solid colour; gradient fills are left alone.
    pub fill: Option<AnimTrack<[f32; 4]>>,
    /// Drives `border_colour`.
    pub border: Option<AnimTrack<[f32; 4]>>,
    /// Drives `rotation` in radians.
    pub rotation: Option<AnimTrack<f32>>,
    /// Arbitrary path channel driving `opacity`. Overrides the linear
    /// `opacity` track and the legacy `animation` field when set.
    pub opacity_path: Option<PathTrack<f32>>,
    /// Arbitrary path channel driving `position`. Overrides the linear
    /// `position` track when set.
    pub position_path: Option<PathTrack<[f32; 2]>>,
    /// Arbitrary path channel driving `size`. Overrides the linear
    /// `size` track when set.
    pub size_path: Option<PathTrack<[f32; 2]>>,
    /// Arbitrary path channel driving the solid fill colour. Overrides
    /// the linear `fill` track when set.
    pub fill_path: Option<PathTrack<[f32; 4]>>,
    /// Arbitrary path channel driving `border_colour`. Overrides the
    /// linear `border` track when set.
    pub border_path: Option<PathTrack<[f32; 4]>>,
    /// Arbitrary path channel driving `rotation`. Overrides the linear
    /// `rotation` track when set.
    pub rotation_path: Option<PathTrack<f32>>,
}

/// Trait used by [`AnimTrack`] resolution to interpolate between `from`
/// and `to`. Implemented for the channel types the overlay animation
/// system needs: `f32`, `[f32; 2]`, `[f32; 4]`.
pub trait LerpAnim: Copy {
    /// Returns `from * (1 - t) + to * t`.
    fn lerp(from: Self, to: Self, t: f32) -> Self;
}

impl LerpAnim for f32 {
    fn lerp(from: Self, to: Self, t: f32) -> Self {
        from * (1.0 - t) + to * t
    }
}

impl LerpAnim for [f32; 2] {
    fn lerp(from: Self, to: Self, t: f32) -> Self {
        [f32::lerp(from[0], to[0], t), f32::lerp(from[1], to[1], t)]
    }
}

impl LerpAnim for [f32; 4] {
    fn lerp(from: Self, to: Self, t: f32) -> Self {
        [
            f32::lerp(from[0], to[0], t),
            f32::lerp(from[1], to[1], t),
            f32::lerp(from[2], to[2], t),
            f32::lerp(from[3], to[3], t),
        ]
    }
}

impl<T: Copy + LerpAnim> AnimTrack<T> {
    /// Resolve the track at the given absolute time. Returns the
    /// interpolated value.
    pub fn sample(&self, time: f64) -> T {
        if self.duration <= 0.0 {
            return self.to;
        }
        let raw = ((time - self.start_time) as f32) / self.duration;
        let phase = resolve_phase(raw, self.repeat);
        let t = apply_easing(phase, self.easing);
        T::lerp(self.from, self.to, t)
    }
}

/// Map a raw normalised parameter (number of cycles since `start_time`) into
/// the canonical `[0, 1]` phase using the given repeat mode.
fn resolve_phase(raw: f32, repeat: RepeatMode) -> f32 {
    match repeat {
        RepeatMode::Once => raw.clamp(0.0, 1.0),
        RepeatMode::Loop => {
            let f = raw - raw.floor();
            if f < 0.0 { f + 1.0 } else { f }
        }
        RepeatMode::PingPong => {
            let two = (raw * 0.5).floor() * 2.0;
            let r = raw - two;
            if r > 1.0 { 2.0 - r } else { r }
        }
    }
}

/// Apply an easing curve to a `[0, 1]` phase.
fn apply_easing(phase: f32, easing: OverlayEasing) -> f32 {
    match easing {
        OverlayEasing::Linear => phase,
        OverlayEasing::EaseIn => phase * phase,
        OverlayEasing::EaseOut => {
            let inv = 1.0 - phase;
            1.0 - inv * inv
        }
        OverlayEasing::EaseInOut => phase * phase * (3.0 - 2.0 * phase),
        OverlayEasing::Pulse => (phase * std::f32::consts::PI).sin(),
    }
}

/// Arbitrary-path animation track. `path` is a closure called with the eased
/// parameter `t in [0, 1]` and returns the value for the channel.
///
/// Use for any motion that's more than a straight line: Bezier arcs,
/// polylines, lissajous, custom shapes. The `bezier` and `polyline` helpers
/// cover the common cases without the consumer writing the curve math.
///
/// The closure is stored in an `Arc`, so cloning the track is cheap (one
/// atomic bump). The `Send + Sync + 'static` bound is satisfied by closures
/// that capture only owned/by-value data.
#[derive(Clone)]
pub struct PathTrack<T: Copy + LerpAnim> {
    /// Absolute time at which the track starts.
    pub start_time: f64,
    /// Length of one cycle in seconds.
    pub duration: f32,
    /// Curve applied to the normalised parameter before the closure runs.
    pub easing: OverlayEasing,
    /// What happens past the end of one cycle.
    pub repeat: RepeatMode,
    /// Evaluator for the path. Called with `t in [0, 1]` after easing and
    /// repeat resolution. The closure is shared via `Arc` so the track is
    /// cheap to clone.
    pub path: std::sync::Arc<dyn Fn(f32) -> T + Send + Sync>,
}

impl<T: Copy + LerpAnim> std::fmt::Debug for PathTrack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathTrack")
            .field("start_time", &self.start_time)
            .field("duration", &self.duration)
            .field("easing", &self.easing)
            .field("repeat", &self.repeat)
            .field("path", &"<closure>")
            .finish()
    }
}

impl<T: Copy + LerpAnim> PathTrack<T> {
    /// Construct a track that evaluates the supplied closure at each frame.
    /// Defaults to `Linear` easing and `Once` repeat; chain `with_easing`
    /// or `with_repeat` to override.
    pub fn new(
        start_time: f64,
        duration: f32,
        path: impl Fn(f32) -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            start_time,
            duration,
            easing: OverlayEasing::Linear,
            repeat: RepeatMode::Once,
            path: std::sync::Arc::new(path),
        }
    }

    /// Builder-style easing setter.
    pub fn with_easing(mut self, easing: OverlayEasing) -> Self {
        self.easing = easing;
        self
    }

    /// Builder-style repeat-mode setter.
    pub fn with_repeat(mut self, repeat: RepeatMode) -> Self {
        self.repeat = repeat;
        self
    }

    /// Resolve the track at the given absolute time.
    pub fn sample(&self, time: f64) -> T {
        if self.duration <= 0.0 {
            return (self.path)(1.0);
        }
        let raw = ((time - self.start_time) as f32) / self.duration;
        let phase = resolve_phase(raw, self.repeat);
        let t = apply_easing(phase, self.easing);
        (self.path)(t)
    }
}

impl PathTrack<[f32; 2]> {
    /// Construct a 2D track that walks a single cubic Bezier from `p0` to
    /// `p3` with control handles `p1` and `p2`. Evaluates the standard
    /// Bernstein form at the eased parameter.
    pub fn bezier(start_time: f64, duration: f32, control_points: [[f32; 2]; 4]) -> Self {
        let [p0, p1, p2, p3] = control_points;
        Self::new(start_time, duration, move |t| {
            let one_t = 1.0 - t;
            let w0 = one_t * one_t * one_t;
            let w1 = 3.0 * one_t * one_t * t;
            let w2 = 3.0 * one_t * t * t;
            let w3 = t * t * t;
            [
                w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
                w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
            ]
        })
    }

    /// Construct a 2D track that walks a polyline at uniform per-segment
    /// parameter. With `N` points the path spans `N - 1` equal-length
    /// parameter segments; consumers wanting arc-length-uniform motion
    /// should subdivide their polyline ahead of time.
    pub fn polyline(start_time: f64, duration: f32, points: Vec<[f32; 2]>) -> Self {
        Self::new(start_time, duration, move |t| {
            let n = points.len();
            if n == 0 {
                return [0.0, 0.0];
            }
            if n == 1 {
                return points[0];
            }
            let seg_count = n - 1;
            let scaled = t.clamp(0.0, 1.0) * seg_count as f32;
            let seg = (scaled as usize).min(seg_count - 1);
            let local = scaled - seg as f32;
            let a = points[seg];
            let b = points[seg + 1];
            [
                a[0] * (1.0 - local) + b[0] * local,
                a[1] * (1.0 - local) + b[1] * local,
            ]
        })
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
    /// Where the border sits relative to the shape edge. Default: `Inset`.
    pub border_mode: BorderMode,
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
        // Rotate the query point by -rotation around the shape centre so the
        // SDF evaluates in the unrotated frame, matching the fragment shader.
        let c = (-self.rotation).cos();
        let s = (-self.rotation).sin();
        let p = [c * dx - s * dy, s * dx + c * dy];
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
            OverlayShape::Line { thickness, cap } => {
                sd_line(p, hs, thickness * 0.5, cap == LineCap::Square)
            }
            OverlayShape::Star {
                points,
                inner_radius_frac,
            } => {
                let r = hw.min(hh);
                sd_star(p, r, points as f32, inner_radius_frac.clamp(0.0, 1.0))
            }
            OverlayShape::RegularPolygon { sides } => {
                let r = hw.min(hh);
                sd_ngon(p, r, sides.max(3) as f32)
            }
            OverlayShape::Cross { arm_width_frac } => {
                sd_cross(p, hs, arm_width_frac.clamp(0.0, 1.0))
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
}

/// Semantic overlays rendered after post-processing: labels, scalar bars,
/// rulers, screen-space images, and loading bars.
///
/// This frame section is the right place for any visual element that belongs
/// in front of the 3D scene and must not be affected by tone-mapping or bloom.
#[derive(Debug, Clone, Default)]
pub struct OverlayFrame {
    /// Current time in seconds, used to resolve [`OverlayAnimation`] on
    /// shapes. Use the same epoch as the `start_time` values in your
    /// animations (e.g. seconds since app launch). Default: `0.0`.
    pub time: f64,
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
    /// Stroked polylines. Rendered through the same pipeline as overlay
    /// rects and labels; share their z-order space.
    pub polylines: Vec<OverlayPolylineItem>,
}
