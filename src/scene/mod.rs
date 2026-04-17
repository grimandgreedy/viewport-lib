/// Scene graph with parent-child hierarchy and layers.
pub mod scene;
pub use scene::{Group, GroupId, Layer, LayerId, Scene, SceneNode};
/// Per-object material parameters (color, shading, textures).
pub mod material;
/// Core `ViewportObject` trait and render mode types.
pub mod traits;
/// Axis-aligned bounding box.
pub mod aabb;
