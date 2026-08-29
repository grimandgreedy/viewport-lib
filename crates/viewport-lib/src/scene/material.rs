//! Material and per-item render settings.
//!
//! `Material`, `ItemSettings`, and the shading/backface/alpha sub-types live in
//! `viewport-lib-types` so CPU-side tools can build materials without the
//! renderer. They are re-exported here so the renderer's `crate::scene::material::*`
//! paths keep resolving.

pub use viewport_lib_types::material::*;
