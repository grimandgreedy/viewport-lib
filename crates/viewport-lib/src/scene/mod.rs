/// Scene graph with parent-child hierarchy and layers.
pub mod scene;
pub use scene::{Group, GroupId, Layer, LayerId, Scene, SceneNode, SceneStats};
/// Axis-aligned bounding box.
pub mod aabb;
/// Built-in light glyph + influence-volume wireframe emission for scene-graph lights.
pub mod light_indicators;
/// Per-object material parameters (colour, shading, textures).
pub mod material;
/// Participating-media volume primitive (fog, smoke, clouds).
pub use light_indicators::{LightIndicators, LightMarker, build_light_indicators};
/// Loose octree spatial index for frustum culling acceleration.
pub(crate) mod spatial_index;
/// Core `ViewportObject` trait and render mode types.
pub mod traits;
