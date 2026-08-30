/// Horizontal alignment of an item's box relative to its anchor point.
///
/// Also names a horizontal position on the viewport rect when used as part of a
/// viewport origin: `Left` is the left edge, `Middle` the centre, `Right` the
/// right edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AnchorX {
    /// Left edge sits at the anchor; the box extends right (default).
    #[default]
    Left,
    /// Centered horizontally on the anchor.
    Middle,
    /// Right edge sits at the anchor; the box extends left.
    Right,
}

/// Vertical alignment of an item's box relative to its anchor point.
///
/// Also names a vertical position on the viewport rect when used as part of a
/// viewport origin: `Top` is the top edge, `Middle` the centre, `Bottom` the
/// bottom edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AnchorY {
    /// Top edge sits at the anchor (default).
    #[default]
    Top,
    /// Centered vertically on the anchor.
    Middle,
    /// Bottom edge sits at the anchor.
    Bottom,
}

/// Former name of [`AnchorX`]. Kept so existing code and serialised data using
/// `LabelAnchor` keep working.
pub type LabelAnchor = AnchorX;

/// Former name of [`AnchorY`]. Kept so existing code and serialised data using
/// `LabelAnchorY` keep working.
pub type LabelAnchorY = AnchorY;

/// Where an overlay item hangs from: exactly one origin, resolved to a screen
/// pixel each frame. The item's `position` is then a screen-pixel nudge from
/// that origin and `align_x` / `align_y` place the item's box onto it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OverlayAnchor {
    /// A point on the viewport rect, re-resolved on resize. `Viewport { x:
    /// Left, y: Top }` is the top-left corner, so with a zero `position` and
    /// `Left` / `Top` alignment it reproduces absolute screen-space placement.
    Viewport {
        /// Horizontal position on the viewport rect.
        x: AnchorX,
        /// Vertical position on the viewport rect.
        y: AnchorY,
    },
    /// A 3D world position, projected to screen each frame. The item is skipped
    /// for the frame when the point is behind the camera or off-screen.
    World([f32; 3]),
}

impl Default for OverlayAnchor {
    fn default() -> Self {
        OverlayAnchor::Viewport {
            x: AnchorX::Left,
            y: AnchorY::Top,
        }
    }
}

/// Resolve an [`OverlayAnchor`] to its origin pixel (top-left origin) on a
/// `viewport` of logical pixels. `Viewport` origins map to the matching point on
/// the viewport rect; `World` origins project through `view` / `proj` and return
/// `None` when behind the camera or outside the frustum, which skips the item
/// for the frame.
#[doc(hidden)]
pub fn resolve_anchor_origin(
    anchor: &OverlayAnchor,
    viewport: [f32; 2],
    view: &glam::Mat4,
    proj: &glam::Mat4,
) -> Option<[f32; 2]> {
    match anchor {
        OverlayAnchor::Viewport { x, y } => Some([x.coord(viewport[0]), y.coord(viewport[1])]),
        OverlayAnchor::World(w) => {
            let clip = *proj * *view * glam::Vec3::from(*w).extend(1.0);
            if clip.w <= 0.0 {
                return None;
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            if !(-1.0..=1.0).contains(&ndc_x) || !(-1.0..=1.0).contains(&ndc_y) {
                return None;
            }
            Some([
                (ndc_x * 0.5 + 0.5) * viewport[0],
                (1.0 - (ndc_y * 0.5 + 0.5)) * viewport[1],
            ])
        }
    }
}

impl AnchorX {
    /// Horizontal coordinate of this alignment on a rect of the given `width`:
    /// `Left` = 0, `Middle` = `width / 2`, `Right` = `width`.
    #[doc(hidden)]
    pub fn coord(self, width: f32) -> f32 {
        match self {
            AnchorX::Left => 0.0,
            AnchorX::Middle => width * 0.5,
            AnchorX::Right => width,
        }
    }

    /// Shift applied to a box of `extent` so that this edge sits at the anchor
    /// point: `Left` = 0, `Middle` = `-extent / 2`, `Right` = `-extent`.
    #[doc(hidden)]
    pub fn align_shift(self, extent: f32) -> f32 {
        match self {
            AnchorX::Left => 0.0,
            AnchorX::Middle => -extent * 0.5,
            AnchorX::Right => -extent,
        }
    }
}

impl AnchorY {
    /// Vertical coordinate of this alignment on a rect of the given `height`:
    /// `Top` = 0, `Middle` = `height / 2`, `Bottom` = `height`.
    #[doc(hidden)]
    pub fn coord(self, height: f32) -> f32 {
        match self {
            AnchorY::Top => 0.0,
            AnchorY::Middle => height * 0.5,
            AnchorY::Bottom => height,
        }
    }

    /// Shift applied to a box of `extent` so that this edge sits at the anchor
    /// point: `Top` = 0, `Middle` = `-extent / 2`, `Bottom` = `-extent`.
    #[doc(hidden)]
    pub fn align_shift(self, extent: f32) -> f32 {
        match self {
            AnchorY::Top => 0.0,
            AnchorY::Middle => -extent * 0.5,
            AnchorY::Bottom => -extent,
        }
    }
}

/// Top-left pixel of a `[width, height]` box placed on the viewport rect so the
/// box's `(x, y)` anchor sits at the matching `(x, y)` point of the viewport.
/// Reproduces the placement of the former `ImageAnchor` (e.g. `(Right, Bottom)`
/// pins the box's bottom-right corner to the viewport's bottom-right).
#[doc(hidden)]
pub fn viewport_anchored_top_left(
    x: AnchorX,
    y: AnchorY,
    size: [f32; 2],
    viewport: [f32; 2],
) -> [f32; 2] {
    [
        x.coord(viewport[0]) + x.align_shift(size[0]),
        y.coord(viewport[1]) + y.align_shift(size[1]),
    ]
}

/// NDC rect `[min_x, max_x, min_y, max_y]` for a viewport-anchored box of `size`
/// logical pixels on a `viewport` of logical pixels. Y is top-left origin in
/// pixel space and flips to the NDC convention here.
#[doc(hidden)]
pub fn viewport_anchored_ndc(
    x: AnchorX,
    y: AnchorY,
    size: [f32; 2],
    viewport: [f32; 2],
) -> [f32; 4] {
    let tl = viewport_anchored_top_left(x, y, size, viewport);
    let w = viewport[0].max(1.0);
    let h = viewport[1].max(1.0);
    [
        2.0 * tl[0] / w - 1.0,
        2.0 * (tl[0] + size[0]) / w - 1.0,
        1.0 - 2.0 * (tl[1] + size[1]) / h,
        1.0 - 2.0 * tl[1] / h,
    ]
}
