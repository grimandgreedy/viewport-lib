//! Per-frame reference to a pre-uploaded tensor glyph set.

use crate::renderer::types::items::IDENTITY_MAT4;

crate::resources::handle::slot_handle! {
    /// Handle to a tensor glyph set uploaded once through
    /// [`ViewportRenderer::upload_tensor_glyph_set`](crate::renderer::ViewportRenderer::upload_tensor_glyph_set).
    ///
    /// Name it from a [`TensorGlyphSetRefItem`] to draw the stored set without
    /// rebuilding its instance buffer.
    pub struct TensorGlyphSetId;
}
use crate::scene::material::ItemSettings;

/// Per-frame reference to a pre-uploaded tensor glyph set. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TensorGlyphSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportRenderer::upload_tensor_glyph_set`](crate::renderer::ViewportRenderer::upload_tensor_glyph_set)
    /// or `begin_upload_tensor_glyph_set`.
    pub source: crate::resources::TensorGlyphSetId,
    /// Per-frame model matrix. Composes on top of the per-instance
    /// transforms baked at upload time.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl TensorGlyphSetRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::TensorGlyphSetId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}
