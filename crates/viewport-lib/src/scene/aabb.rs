//! Axis-aligned bounding box for frustum culling and spatial queries.
//!
//! `Aabb` lives in `viewport-lib-types` so CPU-side tools can compute and pass
//! bounds without the renderer. It is re-exported here so the renderer's
//! `crate::scene::aabb::Aabb` path keeps resolving.

pub use viewport_lib_types::scene::aabb::Aabb;
