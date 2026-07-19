use super::*;

mod helpers;
use helpers::*;
mod gpu;
pub(crate) use gpu::PendingPick;
/// Pick mask for controlling item types and sub-element levels in pick calls.
pub mod pick_mask;
mod point;
mod rect;
/// Typed sub-object reference and sub-object selection set.
pub mod sub_object;
mod types;

pub use pick_mask::PickMask;
pub use sub_object::{
    CellSelectionInfo, PolylineSelectionInfo, SubObjectRef, SubSelection, SubSelectionRef,
    VolumeSelectionInfo,
};
pub use types::{GpuPickHit, PickHit, PickRectResult};

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
// Backend selection
// ---------------------------------------------------------------------------

/// Which backend the object-level pick entry points run.
///
/// Both backends return the same [`PickHit`]
/// so a caller can switch between them without changing how it reads the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickBackend {
    /// CPU ray-cast. Fills in sub-object identity (face, vertex, edge, cell, ...)
    /// per the mask, and reads the per-frame pick cache, which must be enabled
    /// with [`set_cpu_pick_cache`](ViewportRenderer::set_cpu_pick_cache).
    Cpu,
    /// GPU object-id readback. Resolves sub-object identity from the pass's
    /// second channel: the hit instance (glyphs, sprites), segment (polylines),
    /// point (point clouds), splat, and voxel need no device feature. The
    /// triangle-meshed levels (surface face / cell / vertex, tube-family segment /
    /// strip / node) need `SHADER_PRIMITIVE_INDEX`; without it the GPU backend
    /// returns the object-level hit and a caller that needs the sub-object uses the
    /// [`Cpu`](Self::Cpu) backend explicitly. The GPU backend never runs a
    /// per-click CPU ray-cast. Cost tracks the render rather than the object count,
    /// so it is the backend for large scenes.
    Gpu,
}

/// Outcome of polling a non-blocking GPU pick with
/// [`ViewportRenderer::pick_object_poll`].
#[derive(Clone, Debug)]
pub enum PickPoll {
    /// No pick is in flight: none has been started, or the last one was already
    /// read.
    Idle,
    /// A pick was submitted but its read-back has not completed yet. Poll again
    /// on a later frame.
    Pending,
    /// The pick resolved. `Some` is the hit under the cursor; `None` is empty
    /// space (or a type the GPU pass cannot draw).
    Ready(Option<PickHit>),
}

impl ViewportRenderer {
    /// Pick the nearest object under `cursor`, running `backend` and returning a
    /// shared [`PickHit`].
    ///
    /// The camera and viewport size come from `frame`. `mask` chooses which item
    /// types and sub-object levels participate. Both backends use it to select
    /// item types and to fill the sub-object level of the returned
    /// [`PickHit`]; see
    /// [`PickBackend::Gpu`] for the sub-object levels the GPU backend resolves and
    /// its `SHADER_PRIMITIVE_INDEX` requirement. A type the GPU pass has no
    /// pipeline for yet is simply not drawn, so it returns no hit rather than a
    /// wrong one.
    pub fn pick_object(
        &mut self,
        backend: PickBackend,
        cursor: glam::Vec2,
        frame: &FrameData,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mask: PickMask,
    ) -> Option<PickHit> {
        let viewport_size = glam::Vec2::from(frame.camera.viewport_size);
        let view_proj = frame.camera.render_camera.view_proj();
        let _ = (viewport_size, view_proj);
        match backend {
            PickBackend::Cpu => self.pick(cursor, viewport_size, view_proj, mask),
            PickBackend::Gpu => self.pick_object_gpu_blocking(device, queue, cursor, frame, mask),
        }
    }

    /// Pick every object touching the rect, running `backend` and returning a
    /// [`PickRectResult`].
    ///
    /// The GPU backend is object-level only: it reads back the id region within
    /// the rect and collects the unique object ids, but does not decode the
    /// primitive channel into sub-object identity (that would mean resolving
    /// face / cell / vertex / instance per pixel over the whole rect). A mask
    /// with no `OBJECT` bit gets an empty result from the GPU backend rather
    /// than a silent reinterpretation of what it asked for; use `Cpu` for
    /// sub-object rect selection. `Cpu` is the same as calling
    /// [`pick_rect`](Self::pick_rect).
    pub fn pick_rect_objects(
        &mut self,
        backend: PickBackend,
        rect_min: glam::Vec2,
        rect_max: glam::Vec2,
        frame: &FrameData,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mask: PickMask,
    ) -> PickRectResult {
        match backend {
            PickBackend::Cpu => {
                let viewport_size = glam::Vec2::from(frame.camera.viewport_size);
                let view_proj = frame.camera.render_camera.view_proj();
                self.pick_rect(rect_min, rect_max, viewport_size, view_proj, mask)
            }
            PickBackend::Gpu => self.pick_rect_gpu(device, queue, rect_min, rect_max, frame, mask),
        }
    }
}
