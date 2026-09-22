//! Anchor points that place an overlay item relative to the viewport or a reference rect.

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

/// A horizontal and a vertical alignment together: one point on a rect.
///
/// Used for both halves of [`OverlayAnchoring`], which is why it exists: the
/// point on the viewport an item hangs from and the point on the item's own box
/// that lands there are the same kind of thing, asked of two different rects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Alignment {
    /// Horizontal: `Left`, `Middle`, or `Right`.
    pub x: AnchorX,
    /// Vertical: `Top`, `Middle`, or `Bottom`.
    pub y: AnchorY,
}

impl Alignment {
    /// The top-left corner.
    pub const TOP_LEFT: Self = Self {
        x: AnchorX::Left,
        y: AnchorY::Top,
    };
    /// The centre.
    pub const CENTRE: Self = Self {
        x: AnchorX::Middle,
        y: AnchorY::Middle,
    };

    /// An alignment from its two halves.
    pub const fn new(x: AnchorX, y: AnchorY) -> Self {
        Self { x, y }
    }
}

/// Where an overlay item hangs from: exactly one origin, resolved to a screen
/// pixel each frame. The item's `position` is then a screen-pixel nudge from
/// that origin, and the anchoring's `align` places the item's box onto it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum OverlayOrigin {
    /// A point on the viewport rect, re-resolved on resize. The top-left corner
    /// with a zero `position` and top-left alignment reproduces absolute
    /// screen-space placement.
    Viewport(Alignment),
    /// A 3D world position, projected to screen each frame. The item is skipped
    /// for the frame when the point is behind the camera or off-screen.
    World([f32; 3]),
}

impl Default for OverlayOrigin {
    fn default() -> Self {
        OverlayOrigin::Viewport(Alignment::TOP_LEFT)
    }
}

/// How an overlay item is placed: the point it hangs from, and the point of its
/// own box that lands there.
///
/// The two are separate questions on the same item. A readout pinned to the
/// bottom-right corner of the viewport with its own bottom-right corner on that
/// point is `origin: Viewport(bottom-right)` with `align: bottom-right`; the
/// same origin with a top-left `align` hangs the readout off the screen instead.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct OverlayAnchoring {
    /// The point the item hangs from: a point on the viewport rect, or a world
    /// position projected each frame.
    pub origin: OverlayOrigin,
    /// The point of the item's own box that lands on the origin.
    pub align: Alignment,
}

impl OverlayAnchoring {
    /// Hang from a point on the viewport rect.
    pub const fn viewport(x: AnchorX, y: AnchorY) -> Self {
        Self {
            origin: OverlayOrigin::Viewport(Alignment::new(x, y)),
            align: Alignment::TOP_LEFT,
        }
    }

    /// Hang from a world position, projected to the screen each frame.
    pub const fn world(pos: [f32; 3]) -> Self {
        Self {
            origin: OverlayOrigin::World(pos),
            align: Alignment::TOP_LEFT,
        }
    }

    /// Set which point of the item's own box lands on the origin.
    pub const fn with_align(mut self, align: Alignment) -> Self {
        self.align = align;
        self
    }
}

/// Resolve an [`OverlayOrigin`] to its origin pixel (top-left origin) on a
/// `viewport` of logical pixels. `Viewport` origins map to the matching point on
/// the viewport rect; `World` origins project through `view` / `proj` and return
/// `None` when behind the camera or outside the frustum, which skips the item
/// for the frame.
///
/// This is the same call the renderer makes when it places an anchored item, so
/// a consumer working out where an item will land (to size a backing, hit test
/// it, or place something beside it) gets the renderer's answer rather than an
/// approximation of it, including the behind-camera cull and the y flip out of
/// NDC. Pass the camera's `view_matrix()` and `proj_matrix()`.
pub fn resolve_anchor_origin(
    anchor: &OverlayOrigin,
    viewport: [f32; 2],
    view: &glam::Mat4,
    proj: &glam::Mat4,
) -> Option<[f32; 2]> {
    match anchor {
        OverlayOrigin::Viewport(a) => Some([a.x.coord(viewport[0]), a.y.coord(viewport[1])]),
        OverlayOrigin::World(w) => {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The pairing reproduces what the two flat alignment fields did: the
    /// origin picks a point on the viewport rect and the alignment picks the
    /// point of the item's own box that lands there. Both used to be spelled
    /// with the same two enums on the same struct, which is why they are named
    /// apart now.
    #[test]
    fn origin_and_align_place_a_box_the_way_the_flat_fields_did() {
        let anchoring = OverlayAnchoring::viewport(AnchorX::Right, AnchorY::Bottom)
            .with_align(Alignment::new(AnchorX::Right, AnchorY::Bottom));

        let origin = resolve_anchor_origin(
            &anchoring.origin,
            [800.0, 600.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
        )
        .unwrap();
        assert_eq!(origin, [800.0, 600.0]);

        let size = [120.0, 40.0];
        let placed = [
            origin[0] + anchoring.align.x.align_shift(size[0]),
            origin[1] + anchoring.align.y.align_shift(size[1]),
        ];
        assert_eq!(placed, [800.0 - 120.0, 600.0 - 40.0]);
    }

    /// The default is the top-left of the viewport with the box's own top-left
    /// on it, which is absolute screen placement and what every item but the
    /// label starts from.
    #[test]
    fn the_default_anchoring_is_absolute_screen_placement() {
        let anchoring = OverlayAnchoring::default();
        assert_eq!(
            anchoring.origin,
            OverlayOrigin::Viewport(Alignment::TOP_LEFT)
        );
        assert_eq!(anchoring.align, Alignment::TOP_LEFT);
        let origin = resolve_anchor_origin(
            &anchoring.origin,
            [800.0, 600.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
        )
        .unwrap();
        assert_eq!(origin, [0.0, 0.0]);
    }

    /// A world origin culls rather than clamping when it projects behind the
    /// camera, which is what skips the item for the frame.
    #[test]
    fn a_world_origin_behind_the_camera_resolves_to_nothing() {
        let behind = glam::Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, -1.0, //
            0.0, 0.0, 0.0, 0.0,
        ]);
        let anchoring = OverlayAnchoring::world([0.0, 0.0, 10.0]);
        assert!(
            resolve_anchor_origin(
                &anchoring.origin,
                [800.0, 600.0],
                &glam::Mat4::IDENTITY,
                &behind
            )
            .is_none()
        );
    }
}
