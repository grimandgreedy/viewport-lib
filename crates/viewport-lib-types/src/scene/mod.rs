//! Scene vocabulary: the leaf data types a scene is built from.
//!
//! The `Scene` graph itself (spatial index, render-item collection) stays in
//! `viewport-lib`; this module holds the pure leaves it is composed of.

pub mod aabb;

pub use aabb::Aabb;

/// Node identifier: matches `ViewportObject::id()` return type.
///
/// A scene node's stable handle. Consumers that build or reference scene nodes
/// (loaders, simulation plugins, tools) name this without the renderer.
pub type NodeId = u64;
