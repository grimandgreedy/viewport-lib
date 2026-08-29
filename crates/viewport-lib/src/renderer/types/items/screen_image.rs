use crate::renderer::types::{AnchorX, AnchorY};
use crate::scene::material::ItemSettings;

/// A floating screen-space RGBA image rendered as a viewport overlay.
///
/// The image is drawn after all 3D geometry and anchored to one of the viewport
/// corners or the center.
///
/// For a pure screen-space image with no depth compositing, use a textured
/// `OverlayShapeItem` (see `OverlayShapeItem::textured_image`) fed by a streaming
/// `OverlayTextureId`, which renders after post-processing alongside the other
/// overlays.
///
/// ## Depth compositing
///
/// When `depth` is `Some`, the image composites against 3D scene geometry:
/// pixels whose depth value exceeds the scene depth at that screen position are
/// discarded, so near geometry occludes the image correctly.
///
/// `depth` must contain exactly `width * height` `f32` values in row-major,
/// top-to-bottom order. Each value is an NDC depth in `[0.0, 1.0]` where
/// `0.0` = near plane and `1.0` = far plane, matching wgpu's depth convention.
///
/// Depth compositing is only active in the full `render()` path. When using
/// `paint()` / `paint_to()` (external render passes), the image is drawn
/// without a depth test regardless of this field.
#[non_exhaustive]
#[derive(Clone)]
pub struct ScreenImageItem {
    /// RGBA8 pixel data, row-major, top-to-bottom.
    pub pixels: Vec<[u8; 4]>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Horizontal viewport anchor: `Left` pins the image's left edge to the
    /// viewport's left edge, `Right` its right edge, `Middle` centres it.
    pub anchor_x: AnchorX,
    /// Vertical viewport anchor: `Top` pins the image's top edge to the
    /// viewport's top edge, `Bottom` its bottom edge, `Middle` centres it.
    pub anchor_y: AnchorY,
    /// Scale factor relative to natural pixel size (`1.0` = one pixel per screen pixel).
    pub scale: f32,
    /// Overall opacity multiplier applied on top of per-pixel alpha. Default: `1.0`.
    pub alpha: f32,
    /// Per-pixel NDC depth values `[0.0, 1.0]` for depth compositing against scene
    /// geometry. Must contain exactly `width * height` values if `Some`.
    /// `None` (default) renders the image on top of all geometry (no depth test).
    pub depth: Option<Vec<f32>>,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for ScreenImageItem {
    fn default() -> Self {
        Self {
            pixels: Vec::new(),
            width: 0,
            height: 0,
            anchor_x: AnchorX::Left,
            anchor_y: AnchorY::Top,
            scale: 1.0,
            alpha: 1.0,
            depth: None,
            settings: ItemSettings::default(),
        }
    }
}
