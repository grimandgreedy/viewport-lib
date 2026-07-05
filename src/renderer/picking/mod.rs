use super::*;

mod helpers;
use helpers::*;
mod gpu;
mod point;
mod rect;

impl ViewportRenderer {
    /// Copy this frame's pickable items into the CPU pick caches so `pick()` and
    /// `pick_rect()` can run later (e.g. on a mouse click) without the scene data.
    ///
    /// Called from `prepare()` only when the CPU pick cache is enabled. The mesh
    /// entry is cheap (it holds a `mesh_id`), but the data primitives carry inline
    /// geometry, so this is the per-frame cost that `set_cpu_pick_cache` controls.
    pub(crate) fn cache_pick_items(&mut self, frame: &FrameData) {
        let surfaces = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };
        // Opaque volume meshes appear as boundary SceneRenderItems for face/vertex
        // picking; cell-level remapping is driven from `pick_volume_mesh_items`.
        self.pick_scene_items = surfaces
            .iter()
            .cloned()
            .chain(
                frame
                    .scene
                    .volume_meshes
                    .iter()
                    .filter(|item| item.transparency.is_none())
                    .map(|item| item.to_render_item()),
            )
            .collect();
        self.pick_point_cloud_items = frame.scene.point_clouds.clone();
        self.pick_splat_items = frame.scene.gaussian_splats.clone();
        self.pick_volume_items = frame.scene.volumes.clone();
        self.pick_scatter_volume_items = frame.scene.scatter_volumes.clone();
        self.pick_volume_mesh_items = frame.scene.volume_meshes.clone();
        self.pick_polyline_items = frame.scene.polylines.clone();
        self.pick_glyph_items = frame.scene.glyphs.clone();
        self.pick_tensor_glyph_items = frame.scene.tensor_glyphs.clone();
        self.pick_sprite_items = frame.scene.sprite_items.clone();
        self.pick_streamtube_items = frame.scene.streamtube_items.clone();
        self.pick_tube_items = frame.scene.tube_items.clone();
        self.pick_ribbon_items = frame.scene.ribbon_items.clone();
        self.pick_image_slice_items = frame.scene.image_slices.clone();
        self.pick_volume_surface_slice_items = frame.scene.volume_surface_slices.clone();
        self.pick_screen_image_items = frame.scene.screen_images.clone();
        self.pick_decal_items = frame.scene.decals.clone();
    }

    /// Empty the CPU pick caches populated by `cache_pick_items`, freeing their
    /// memory when the cache is turned off.
    pub(crate) fn clear_pick_cache(&mut self) {
        self.pick_scene_items = Vec::new();
        self.pick_point_cloud_items = Vec::new();
        self.pick_splat_items = Vec::new();
        self.pick_volume_items = Vec::new();
        self.pick_scatter_volume_items = Vec::new();
        self.pick_volume_mesh_items = Vec::new();
        self.pick_polyline_items = Vec::new();
        self.pick_glyph_items = Vec::new();
        self.pick_tensor_glyph_items = Vec::new();
        self.pick_sprite_items = Vec::new();
        self.pick_streamtube_items = Vec::new();
        self.pick_tube_items = Vec::new();
        self.pick_ribbon_items = Vec::new();
        self.pick_image_slice_items = Vec::new();
        self.pick_volume_surface_slice_items = Vec::new();
        self.pick_screen_image_items = Vec::new();
        self.pick_decal_items = Vec::new();
    }
}

// ---------------------------------------------------------------------------
// PickRectResult
// ---------------------------------------------------------------------------

/// Result of a [`ViewportRenderer::pick_rect`] call.
#[derive(Clone, Debug, Default)]
pub struct PickRectResult {
    /// IDs of whole items that have geometry inside the pick rect.
    ///
    /// Populated when [`crate::interaction::select::pick_mask::PickMask::OBJECT`] is set.
    pub objects: Vec<u64>,
    /// Sub-elements inside the pick rect as `(item_id, sub_object)` pairs.
    ///
    /// Populated when any sub-element bit is set in the mask. All entries
    /// belong to the same geometric dimension when the mask is
    /// dimension-homogeneous (the common case).
    pub elements: Vec<(u64, crate::interaction::select::sub_object::SubObjectRef)>,
}

impl PickRectResult {
    /// Returns `true` when no objects or elements were found.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.elements.is_empty()
    }
}
