//! Core viewport object trait and related types.
//!
//! Applications implement [`ViewportObject`](crate::scene::traits::ViewportObject) on their scene object types
//! so the renderer and picking system can work with them generically.

use crate::scene::material::Material;

/// Render mode for a viewport object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderMode {
    /// Filled solid geometry (triangle list, depth-tested).
    Solid,
    /// Wireframe overlay (edge list).
    Wireframe,
}

/// Trait that application objects must implement to be rendered and picked
/// in the viewport.
pub trait ViewportObject {
    /// Unique object identifier.
    fn id(&self) -> u64;
    /// GPU mesh key (index into mesh storage). None if no mesh uploaded yet.
    fn mesh_id(&self) -> Option<u64>;
    /// World-space model matrix (Translation * Rotation * Scale).
    fn model_matrix(&self) -> glam::Mat4;
    /// World-space position.
    fn position(&self) -> glam::Vec3;
    /// Orientation as a quaternion.
    fn rotation(&self) -> glam::Quat;
    /// Per-axis scale factors. Defaults to (1, 1, 1) if not overridden.
    fn scale(&self) -> glam::Vec3 {
        glam::Vec3::ONE
    }
    /// Whether this object should be rendered.
    fn is_visible(&self) -> bool;
    /// RGB color in linear 0..1 range.
    fn color(&self) -> glam::Vec3;
    /// Whether to render per-vertex normal visualization.
    fn show_normals(&self) -> bool {
        false
    }
    /// Render mode override.
    fn render_mode(&self) -> RenderMode {
        RenderMode::Solid
    }
    /// Alpha transparency (1.0 = fully opaque).
    fn transparency(&self) -> f32 {
        1.0
    }
    /// Material for this object. Default derives from `color()`.
    ///
    /// Override to provide per-object ambient, diffuse, specular, shininess,
    /// opacity, or texture parameters. The default implementation wraps
    /// `self.color()` in a [`Material::from_color`] call so existing
    /// implementations continue to work without any changes.
    fn material(&self) -> Material {
        Material::from_color(self.color().into())
    }
}
