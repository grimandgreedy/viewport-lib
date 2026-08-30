//! Screen-space overlay item vocabulary: the shapes, labels, glyph runs,
//! polylines, and vector paths a consumer builds to describe overlays, together
//! with the fills, anchors, animations, and texture-sampling parameters they
//! compose from.
//!
//! This is pure data. The renderer rasterizes and uploads these items; a
//! consumer (or a feature crate such as the gizmo or ui overlays) only needs to
//! name and build them.

pub mod anchor;
pub mod animation;
pub mod fill;
pub mod font;
pub mod frame;
pub mod glyph_run;
pub mod label;
pub mod polyline;
pub mod shape;
pub mod texture;
pub mod vector;

pub use self::anchor::*;
pub use self::animation::*;
pub use self::fill::*;
pub use self::font::*;
pub use self::frame::*;
pub use self::glyph_run::*;
pub use self::label::*;
pub use self::polyline::*;
pub use self::shape::*;
pub use self::texture::*;
pub use self::vector::*;
