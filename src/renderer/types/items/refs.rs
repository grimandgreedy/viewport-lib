use super::common::IDENTITY_MAT4;
use crate::scene::material::ItemSettings;

/// Per-frame reference to a pre-uploaded polyline.
///
/// Submit one of these on `SceneFrame::polyline_refs` instead of pushing the
/// full `PolylineItem` on `polylines` every frame. The renderer looks up the
/// stored GPU buffers by `id` and applies the per-frame `model` and
/// `settings` without rebuilding the segment buffer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PolylineRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_polyline`](crate::resources::ViewportGpuResources::upload_polyline)
    /// or `begin_upload_polyline`.
    pub id: crate::resources::PolylineId,
    /// Per-frame model matrix. Identity uses the polyline's own world-space
    /// positions.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, wireframe, selection, picking).
    pub settings: ItemSettings,
}

impl PolylineRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::PolylineId) -> Self {
        Self {
            id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded streamtube. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StreamtubeRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_streamtube`](crate::resources::ViewportGpuResources::upload_streamtube)
    /// or `begin_upload_streamtube`.
    pub id: crate::resources::StreamtubeId,
    /// Per-frame model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl StreamtubeRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::StreamtubeId) -> Self {
        Self {
            id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded tube. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TubeRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_tube`](crate::resources::ViewportGpuResources::upload_tube)
    /// or `begin_upload_tube`.
    pub id: crate::resources::TubeId,
    /// Per-frame model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl TubeRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::TubeId) -> Self {
        Self {
            id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded ribbon. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RibbonRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_ribbon`](crate::resources::ViewportGpuResources::upload_ribbon)
    /// or `begin_upload_ribbon`.
    pub id: crate::resources::RibbonId,
    /// Per-frame model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl RibbonRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::RibbonId) -> Self {
        Self {
            id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded point cloud. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PointCloudRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_point_cloud`](crate::resources::ViewportGpuResources::upload_point_cloud)
    /// or `begin_upload_point_cloud`.
    pub id: crate::resources::PointCloudId,
    /// Per-frame model matrix. Composes on top of the model baked into the
    /// upload, so identity here renders the points at their original
    /// transform.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl PointCloudRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::PointCloudId) -> Self {
        Self {
            id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded glyph set. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GlyphSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_glyph_set`](crate::resources::ViewportGpuResources::upload_glyph_set)
    /// or `begin_upload_glyph_set`.
    pub id: crate::resources::GlyphSetId,
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
            id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded tensor glyph set. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TensorGlyphSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_tensor_glyph_set`](crate::resources::ViewportGpuResources::upload_tensor_glyph_set)
    /// or `begin_upload_tensor_glyph_set`.
    pub id: crate::resources::TensorGlyphSetId,
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
            id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded sprite set (static billboards).
///
/// The sprite set's positions, texture, sizes, and colours are baked at
/// upload time. The ref item lets the renderer redraw the same set every
/// frame with no per-frame upload cost.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpriteSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_sprite_set`](crate::resources::ViewportGpuResources::upload_sprite_set)
    /// or `begin_upload_sprite_set`.
    pub id: crate::resources::SpriteSetId,
    /// Per-item render settings (visibility, wireframe, selection).
    pub settings: ItemSettings,
}

impl SpriteSetRefItem {
    /// Visible reference at the upload-time transform.
    pub fn new(id: crate::resources::SpriteSetId) -> Self {
        Self {
            id,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded sprite instance set (entity sprites).
///
/// The sprite definition (texture, blend, size mode, default size) is baked
/// at upload time. The current implementation uses the same upload-time
/// instance transforms as `SpriteSetRefItem`; full per-frame instance
/// transform override against the stable definition is a planned follow-up.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpriteInstanceSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportGpuResources::upload_sprite_instance_set`](crate::resources::ViewportGpuResources::upload_sprite_instance_set)
    /// or `begin_upload_sprite_instance_set`.
    pub id: crate::resources::SpriteInstanceSetId,
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl SpriteInstanceSetRefItem {
    /// Visible reference at the upload-time transform.
    pub fn new(id: crate::resources::SpriteInstanceSetId) -> Self {
        Self {
            id,
            settings: ItemSettings::default(),
        }
    }
}
