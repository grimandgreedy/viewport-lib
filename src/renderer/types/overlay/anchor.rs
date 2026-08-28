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

impl AnchorX {
    /// Horizontal coordinate of this alignment on a rect of the given `width`:
    /// `Left` = 0, `Middle` = `width / 2`, `Right` = `width`.
    pub(crate) fn coord(self, width: f32) -> f32 {
        match self {
            AnchorX::Left => 0.0,
            AnchorX::Middle => width * 0.5,
            AnchorX::Right => width,
        }
    }

    /// Shift applied to a box of `extent` so that this edge sits at the anchor
    /// point: `Left` = 0, `Middle` = `-extent / 2`, `Right` = `-extent`.
    pub(crate) fn align_shift(self, extent: f32) -> f32 {
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
    pub(crate) fn coord(self, height: f32) -> f32 {
        match self {
            AnchorY::Top => 0.0,
            AnchorY::Middle => height * 0.5,
            AnchorY::Bottom => height,
        }
    }

    /// Shift applied to a box of `extent` so that this edge sits at the anchor
    /// point: `Top` = 0, `Middle` = `-extent / 2`, `Bottom` = `-extent`.
    pub(crate) fn align_shift(self, extent: f32) -> f32 {
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
pub(crate) fn viewport_anchored_top_left(
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
pub(crate) fn viewport_anchored_ndc(
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
