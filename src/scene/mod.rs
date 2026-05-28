/// Scene graph with parent-child hierarchy and layers.
pub mod scene;
pub use scene::{DecalHandle, Group, GroupId, Layer, LayerId, LiveDecal, Scene, SceneNode, SceneStats};
/// Axis-aligned bounding box.
pub mod aabb;
/// Per-object material parameters (colour, shading, textures).
pub mod material;
/// Participating-media volume primitive (fog, smoke, clouds).
pub mod scatter_volume;
/// Loose octree spatial index for frustum culling acceleration.
pub(crate) mod spatial_index;
/// Core `ViewportObject` trait and render mode types.
pub mod traits;
