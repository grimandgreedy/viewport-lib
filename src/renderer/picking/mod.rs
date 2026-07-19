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

/// Whether a mask asks for a GPU sub-object level that only the rasterizer's
/// `primitive_index` can name (surface face / cell / vertex / edge, tube-family
/// segment / strip / node), so the GPU backend needs `SHADER_PRIMITIVE_INDEX` to
/// resolve it. The instance / cloud-point / splat / voxel levels come from
/// `instance_index` / `vertex_index` and need no feature, so they are excluded.
///
/// `SEGMENT` / `STRIP` / `POLY_NODE` are treated as needing the feature: polylines
/// answer them without it, but tubes / ribbons / streamtubes need it, and the mask
/// bit alone does not say which is in the scene, so `Auto` errs toward the backend
/// that can always answer.
fn gpu_sub_object_needs_feature(mask: PickMask) -> bool {
    mask.intersects(
        PickMask::FACE
            | PickMask::CELL
            | PickMask::VERTEX
            | PickMask::EDGE
            | PickMask::SEGMENT
            | PickMask::STRIP
            | PickMask::POLY_NODE,
    )
}

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
    /// Let the library choose per query. Uses [`Gpu`](Self::Gpu) for object-level
    /// picks and the sub-object levels the GPU resolves on this device. Only when
    /// the query needs a triangle-mesh sub-object level (surface face / cell /
    /// vertex, tube-family segment / strip / node) on a device without
    /// `SHADER_PRIMITIVE_INDEX` does it fall back to [`Cpu`](Self::Cpu), and then
    /// only if the CPU pick cache is enabled; otherwise it stays on the GPU and
    /// returns the object-level hit. Never trades a fast GPU pick for a hidden
    /// per-click CPU ray-cast the caller did not opt into.
    Auto,
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
        let hit = match self.resolve_pick_backend(backend, device, mask) {
            PickBackend::Cpu => self.pick(cursor, viewport_size, view_proj, mask),
            // `resolve_pick_backend` never returns `Auto`.
            _ => self.pick_object_gpu_blocking(device, queue, cursor, frame, mask),
        };
        // Fill the snap position from the frame's geometry, so both backends
        // return the same feature coordinate. Point-only: the async poll path
        // leaves it `None`.
        hit.map(|mut h| {
            h.sub_object_world_pos = self.snap_world_pos(h.id, h.sub_object, h.world_pos, frame);
            h
        })
    }

    /// World-space position of the resolved sub-object feature, for snapping a
    /// gizmo to it. Reads the feature's coordinate from the frame item plus
    /// `mesh_store` by O(1) index (no ray-cast, no BVH, no per-frame cache clone).
    /// `None` when the feature has no single snap point (cell, voxel, segment,
    /// strip, edge, object-level) or its position is not retained.
    fn snap_world_pos(
        &self,
        id: u64,
        sub_object: Option<SubObjectRef>,
        world_pos: glam::Vec3,
        frame: &FrameData,
    ) -> Option<glam::Vec3> {
        match sub_object? {
            // The chosen corner, from the retained mesh positions times the model.
            // Needs `cpu_positions` retention on the mesh; `None` otherwise.
            SubObjectRef::Vertex(i) => {
                let (mesh_id, model) = self.pick_surface_mesh(id, frame)?;
                let mesh = self.resources.mesh_store.get(mesh_id)?;
                let p = mesh.cpu_positions.as_ref()?.get(i as usize)?;
                Some(glam::Mat4::from_cols_array_2d(&model).transform_point3(glam::Vec3::from(*p)))
            }
            // A curve control node or a point-cloud point: the index is into the
            // item's inline positions. Falls back to the hit point for ref items
            // (positions not inline on the frame).
            SubObjectRef::Point(i) => self.pick_element_position(id, i, frame).or(Some(world_pos)),
            // The closest point on the hit edge to the cursor hit point. The edge
            // id is `face * 3 + local_edge` (see `pick_edge.wgsl`).
            SubObjectRef::Edge(e) => {
                let (mesh_id, model) = self.pick_surface_mesh(id, frame)?;
                let mesh = self.resources.mesh_store.get(mesh_id)?;
                let indices = mesh.cpu_indices.as_ref()?;
                let positions = mesh.cpu_positions.as_ref()?;
                let base = (e / 3) as usize * 3;
                let local = (e % 3) as usize;
                let ia = *indices.get(base + local)? as usize;
                let ib = *indices.get(base + (local + 1) % 3)? as usize;
                let m = glam::Mat4::from_cols_array_2d(&model);
                let a = m.transform_point3(glam::Vec3::from(*positions.get(ia)?));
                let b = m.transform_point3(glam::Vec3::from(*positions.get(ib)?));
                let ab = b - a;
                let t = (world_pos - a).dot(ab) / ab.length_squared().max(1e-12);
                Some(a + ab * t.clamp(0.0, 1.0))
            }
            // The hit point already lies on these features.
            SubObjectRef::Face(_) | SubObjectRef::Splat(_) | SubObjectRef::Instance(_) => {
                Some(world_pos)
            }
            // No single snap point.
            SubObjectRef::Cell(_)
            | SubObjectRef::Voxel(_)
            | SubObjectRef::Segment(_)
            | SubObjectRef::Strip(_) => None,
        }
    }

    /// `(mesh_id, model)` of a pickable surface or volume-mesh boundary named by
    /// `pick_id`, found on the frame.
    fn pick_surface_mesh(
        &self,
        id: u64,
        frame: &FrameData,
    ) -> Option<(crate::resources::mesh::mesh_store::MeshId, [[f32; 4]; 4])> {
        let SurfaceSubmission::Flat(items) = &frame.scene.surfaces;
        if let Some(it) = items
            .iter()
            .find(|i| i.settings.pick_id.0 == id && !i.settings.hidden)
        {
            return Some((it.mesh_id, it.model));
        }
        for vm in &frame.scene.volume_meshes {
            let ri = vm.to_render_item();
            if ri.settings.pick_id.0 == id && !ri.settings.hidden {
                return Some((ri.mesh_id, ri.model));
            }
        }
        None
    }

    /// World position of node / point `i` of a curve control polyline or a point
    /// cloud named by `pick_id`, from its inline positions times the item model.
    /// `None` for ref items (positions live in the store, not on the frame).
    fn pick_element_position(&self, id: u64, i: u32, frame: &FrameData) -> Option<glam::Vec3> {
        let idx = i as usize;
        let at = |positions: &[[f32; 3]], model: &[[f32; 4]; 4]| -> Option<glam::Vec3> {
            positions.get(idx).map(|p| {
                glam::Mat4::from_cols_array_2d(model).transform_point3(glam::Vec3::from(*p))
            })
        };
        for it in &frame.scene.streamtube_items {
            if it.settings.pick_id.0 == id {
                return at(&it.positions, &it.model);
            }
        }
        for it in &frame.scene.tube_items {
            if it.settings.pick_id.0 == id {
                return at(&it.positions, &it.model);
            }
        }
        for it in &frame.scene.ribbon_items {
            if it.settings.pick_id.0 == id {
                return at(&it.positions, &it.model);
            }
        }
        for it in &frame.scene.point_clouds {
            if it.settings.pick_id.0 == id {
                return at(&it.positions, &it.model);
            }
        }
        None
    }

    /// Whether this device can resolve GPU sub-object picking for triangle-meshed
    /// types (surface face / cell / vertex, tube-family segment / strip / node).
    ///
    /// Needs `SHADER_PRIMITIVE_INDEX`. Object-level GPU picking and the instance /
    /// cloud-point / splat / voxel sub-levels work regardless. Use this to decide
    /// whether to enable the CPU pick cache as a sub-object fallback, or just pass
    /// [`PickBackend::Auto`] and let the library route each query.
    pub fn gpu_sub_object_supported(&self, device: &crate::gpu::Device) -> bool {
        device
            .features()
            .contains(crate::gpu::PRIMITIVE_INDEX_FEATURE)
    }

    /// Resolve [`PickBackend::Auto`] to a concrete backend for this query; `Cpu`
    /// and `Gpu` pass through unchanged. Auto stays on the GPU unless the mask
    /// asks for a triangle-mesh sub-object level the device cannot resolve without
    /// `SHADER_PRIMITIVE_INDEX`, in which case it uses the CPU backend when its
    /// cache is on (else the GPU object-level answer).
    fn resolve_pick_backend(
        &self,
        backend: PickBackend,
        device: &crate::gpu::Device,
        mask: PickMask,
    ) -> PickBackend {
        match backend {
            PickBackend::Auto => {
                if gpu_sub_object_needs_feature(mask)
                    && !self.gpu_sub_object_supported(device)
                    && self.cpu_pick_cache_enabled
                {
                    PickBackend::Cpu
                } else {
                    PickBackend::Gpu
                }
            }
            concrete => concrete,
        }
    }

    /// Pick every object touching the rect, running `backend` and returning a
    /// [`PickRectResult`].
    ///
    /// Both backends fill `objects` (when the mask has `OBJECT`) and `elements`
    /// (the sub-object levels the mask asks for). The GPU backend reads the id and
    /// primitive channels over the rect region and decodes each unique
    /// `(object, sub-element)` per pixel; the triangle-mesh sub-levels need
    /// `SHADER_PRIMITIVE_INDEX`, the same as the point path (see
    /// [`PickBackend::Gpu`]). `Cpu` is the same as calling
    /// [`pick_rect`](Self::pick_rect); `Auto` routes per query.
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
        match self.resolve_pick_backend(backend, device, mask) {
            PickBackend::Cpu => {
                let viewport_size = glam::Vec2::from(frame.camera.viewport_size);
                let view_proj = frame.camera.render_camera.view_proj();
                self.pick_rect(rect_min, rect_max, viewport_size, view_proj, mask)
            }
            // `resolve_pick_backend` never returns `Auto`.
            _ => self.pick_rect_gpu(device, queue, rect_min, rect_max, frame, mask),
        }
    }
}
