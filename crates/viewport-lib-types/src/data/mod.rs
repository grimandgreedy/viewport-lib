//! CPU submission payloads: the data a consumer hands to `viewport-lib` to
//! upload. Loaders and mesh processors produce these without touching the GPU.

pub mod attribute;
pub mod mesh;
pub mod point;
pub mod volume;
