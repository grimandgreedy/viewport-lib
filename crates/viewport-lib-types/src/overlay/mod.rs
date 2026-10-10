//! Screen-space overlay item vocabulary: the shapes, labels, glyph runs,
//! polylines, and vector paths a consumer builds to describe overlays, together
//! with the fills, anchors, animations, and texture-sampling parameters they
//! compose from.
//!
//! This is pure data. The renderer rasterises and uploads these items; a
//! consumer (or a feature crate such as the gizmo or ui overlays) only needs to
//! name and build them.

pub mod anchor;
pub mod animation;
pub mod clip;
pub mod content_hash;
pub mod fill;
pub mod font;
pub mod frame;
pub mod geometry;
pub mod glyph_run;
pub mod label;
pub mod polyline;
pub mod shape;
pub mod style;
pub mod texture;
pub mod transform;
pub mod vector;

pub use self::anchor::*;
pub use self::animation::*;
pub use self::clip::*;
pub use self::content_hash::*;
pub use self::fill::*;
pub use self::font::*;
pub use self::frame::*;
pub use self::geometry::*;
pub use self::glyph_run::*;
pub use self::label::*;
pub use self::polyline::*;
pub use self::shape::*;
pub use self::style::*;
pub use self::texture::*;
pub use self::transform::*;
pub use self::vector::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// A consumer that stores overlay items in its own `PartialEq` types, or
    /// diffs a frame's items against the last frame's, needs every item and
    /// the frame that carries them to be comparable.
    #[test]
    fn overlay_items_are_comparable() {
        fn assert_partial_eq<T: PartialEq>() {}
        assert_partial_eq::<OverlayAnimations>();
        assert_partial_eq::<OverlayShapeItem>();
        assert_partial_eq::<OverlayPolylineItem>();
        assert_partial_eq::<LabelItem>();
        assert_partial_eq::<GlyphRunItem>();
        assert_partial_eq::<RetainedOverlay>();
        assert_partial_eq::<OverlayFrame>();
    }
}
