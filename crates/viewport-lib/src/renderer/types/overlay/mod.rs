//! The overlay item vocabulary now lives in `viewport-lib-types`. Re-export it
//! here so `crate::renderer::types::overlay::*` (and the crate-root re-exports
//! that glob through it) keep resolving.
//!
//! `FontHandle` is re-exported from `crate::resources::overlay::font` to
//! preserve its existing path; it is intentionally not globbed here.

pub use viewport_lib_types::overlay::{
    anchor, animation, fill, frame, glyph_run, label, polyline, shape, texture, vector,
};

pub use viewport_lib_types::overlay::anchor::*;
pub use viewport_lib_types::overlay::animation::*;
pub use viewport_lib_types::overlay::fill::*;
pub use viewport_lib_types::overlay::frame::*;
pub use viewport_lib_types::overlay::glyph_run::*;
pub use viewport_lib_types::overlay::label::*;
pub use viewport_lib_types::overlay::polyline::*;
pub use viewport_lib_types::overlay::shape::*;
pub use viewport_lib_types::overlay::texture::*;
pub use viewport_lib_types::overlay::vector::*;
