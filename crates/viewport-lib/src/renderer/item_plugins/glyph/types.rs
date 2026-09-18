//! Per-frame reference to a pre-uploaded glyph set.

use crate::renderer::types::items::IDENTITY_MAT4;

crate::resources::handle::slot_handle! {
    /// Handle to a glyph set uploaded once through
    /// [`ViewportRenderer::upload_glyph_set`](crate::renderer::ViewportRenderer::upload_glyph_set).
    ///
    /// Name it from a [`GlyphSetRefItem`] to draw the stored set without
    /// rebuilding its instance buffer.
    pub struct GlyphSetId;
}
use crate::scene::material::ItemSettings;

/// Per-frame reference to a pre-uploaded glyph set. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GlyphSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`DeviceResources::upload_glyph_set`](crate::resources::DeviceResources::upload_glyph_set)
    /// or `begin_upload_glyph_set`.
    pub source: crate::resources::GlyphSetId,
    /// Per-frame model matrix. Composes on top of the per-instance
    /// transforms baked at upload time.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl GlyphSetRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::GlyphSetId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}
