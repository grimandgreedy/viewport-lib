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
