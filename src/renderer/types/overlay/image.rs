use crate::renderer::types::*;

/// A pixel image composited over the viewport in screen space.
///
/// Unlike [`ScreenImageItem`] (which lives in [`SceneFrame`] and supports
/// depth compositing with world geometry), `OverlayImageItem` is a pure
/// screen-space overlay with no depth field. It renders after post-processing,
/// on top of all scene content, labels, scalar bars, and rulers.
///
/// Consumers using [`ScreenImageItem`] without a `depth` buffer (corner logos,
/// diagnostic HUDs, watermarks) should migrate to `OverlayImageItem`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct OverlayImageItem {
    /// RGBA8 pixel data, row-major, top-to-bottom.
    pub pixels: Vec<[u8; 4]>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Which corner (or center) of the viewport to anchor the image to.
    pub anchor: ImageAnchor,
    /// Scale factor relative to natural pixel size (`1.0` = one pixel per screen pixel).
    pub scale: f32,
    /// Overall opacity multiplier applied on top of per-pixel alpha. Default: `1.0`.
    pub alpha: f32,
    /// Draw order relative to other overlay items. Lower values render first
    /// (further back).
    pub z_order: i32,
}

impl Default for OverlayImageItem {
    fn default() -> Self {
        Self {
            pixels: Vec::new(),
            width: 0,
            height: 0,
            anchor: ImageAnchor::TopLeft,
            scale: 1.0,
            alpha: 1.0,
            z_order: 0,
        }
    }
}

impl OverlayImageItem {
    /// Create an image overlay from RGBA8 `pixels` (row-major, top-to-bottom)
    /// with the given `width` and `height`. All other fields take their
    /// defaults; set them with the `with_*` methods below.
    pub fn new(pixels: Vec<[u8; 4]>, width: u32, height: u32) -> Self {
        Self {
            pixels,
            width,
            height,
            ..Default::default()
        }
    }

    /// Set which corner (or center) of the viewport the image anchors to.
    pub fn with_anchor(mut self, anchor: ImageAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Set the scale factor relative to natural pixel size.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the overall opacity multiplier applied on top of per-pixel alpha.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the draw order relative to other overlay items. Lower values render
    /// first (further back).
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }
}
