//! `ViewportRenderer` : the main entry point for the viewport library.
//!
//! Wraps [`DeviceResources`] and provides `prepare()` / `paint()` methods
//! that take raw `wgpu` types. GUI framework adapters (e.g. the egui
//! `CallbackTrait` impl in the application crate) delegate to these methods.

/// Bind the deform sidecar bind group (index 2) for a mesh-family draw.
///
/// On limited devices (`max_bind_groups < 3`, e.g. iced's shared device, which
/// requests 2) the mesh family compiles 2-group noop pipelines and
/// `deform.enabled` is false; issuing `set_bind_group(2, ...)` there is a wgpu
/// validation error. Every mesh-family deform bind goes through this macro so
/// the guard lives in one greppable place; do not open-code a deform group-2
/// bind at a draw site. Group-2 binds for features that fundamentally need a
/// third group (soft body, refraction, scivis LUT, volumes, glyph/tensor
/// instance data, GPU picking) are not deform binds and are not gated: those
/// features do not run on 2-group devices. The invariant is enforced
/// headlessly by `two_bind_group_device_renders_without_validation_errors`.
macro_rules! bind_deform_group {
    ($pass:expr, $resources:expr, $bg:expr) => {
        if $resources.deform.enabled {
            $pass.set_bind_group(2, $bg, &[]);
        }
    };
}

/// Bind a material plugin's group-3 params bind group for a plugin draw.
///
/// Only called with a bind group resolved by `material_plugin_draw`, which
/// only exists on devices that passed the `max_bind_groups >= 4` registration
/// gate, so no runtime capability check is needed here. Route every group-3
/// material bind through this macro (the same choke-point rule as
/// `bind_deform_group!` above).
macro_rules! bind_material_group {
    ($pass:expr, $bg:expr) => {
        $pass.set_bind_group(3, $bg, &[]);
    };
}

/// Minimum scene item count to activate the instanced draw path.
/// Use instancing for any scene with more than 1 object. The per-object path
/// writes uniforms into a per-mesh buffer, so two scene nodes sharing the same
/// mesh would clobber each other. Instancing avoids this by keeping per-item
/// data in a separate instance buffer indexed by draw-call range.
pub(super) const INSTANCING_THRESHOLD: usize = 1;

/// A batch of instances sharing the same mesh and material textures, drawn in one call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InstancedBatch {
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    pub texture_id: Option<crate::resources::TextureId>,
    pub normal_map_id: Option<crate::resources::TextureId>,
    pub ao_map_id: Option<crate::resources::TextureId>,
    pub instance_offset: u32,
    pub instance_count: u32,
    pub is_transparent: bool,
    /// `true` when the batch's material uses the `Identical` backface policy, so
    /// it must be drawn with the two-sided (`cull_mode: None`) instanced pipeline.
    pub two_sided: bool,
    /// `true` when the batch's material is `AlphaMode::Mask`, so its shadow pass
    /// must sample the albedo alpha and discard cut-out fragments instead of
    /// casting a solid silhouette.
    pub is_cutout: bool,
    /// `true` when ANY textured item in the batch is `AlphaMode::Mask` (the
    /// batch key does not include alpha mode, so items can mix). Gates the
    /// discard-free early-Z pipeline: a batch with a masked instance must keep
    /// the full shader so the per-fragment alpha discard still runs.
    pub has_alpha_mask: bool,
}

mod clip;
pub mod debug;
mod frame;
mod items;
mod lighting;
mod overlay;
mod postprocess;

pub use self::clip::*;
pub use self::debug::{AtlasViewerCorner, DebugOutputMode, DebugQuantity, DebugVis};
pub use self::frame::*;
pub use self::items::*;
pub use self::lighting::*;
pub use self::overlay::*;
pub use self::postprocess::*;
// ---------------------------------------------------------------------------

/// All data needed to render one frame of the viewport.
///
/// Fields are grouped by responsibility. Build the sub-objects you need,
/// leave others at their `Default`, then call `prepare()` followed by
/// `paint()` or `paint_to()`.
#[non_exhaustive]
pub struct FrameData {
    /// Camera state, viewport size, and viewport slot.
    pub camera: CameraFrame,
    /// World-space scene content (surfaces, point clouds, glyphs, etc.).
    pub scene: SceneFrame,
    /// Viewport presentation settings (background, grid, axes indicator).
    pub viewport: ViewportFrame,
    /// Interaction and selection visualization (gizmo, outline, x-ray).
    pub interaction: InteractionFrame,
    /// Global rendering effects (lighting, clipping, post-process).
    pub effects: EffectsFrame,
    /// Semantic overlays rendered after post-processing (labels, scalar bars, rulers).
    pub overlays: OverlayFrame,
}

impl Default for FrameData {
    fn default() -> Self {
        Self {
            camera: CameraFrame::default(),
            scene: SceneFrame::default(),
            viewport: ViewportFrame::default(),
            interaction: InteractionFrame::default(),
            effects: EffectsFrame::default(),
            overlays: OverlayFrame::default(),
        }
    }
}

impl FrameData {
    /// Build frame data from the required camera and scene groups.
    pub fn new(camera: CameraFrame, scene: SceneFrame) -> Self {
        Self {
            camera,
            scene,
            ..Self::default()
        }
    }

    /// Build frame data from a camera, scene, and selection in one call.
    ///
    /// This is the preferred constructor for the common single-viewport path.
    /// It collects render items, stamps the scene and selection generation counters,
    /// and leaves viewport chrome and effects at their defaults.
    ///
    /// Override individual settings with the builder methods:
    ///
    /// ```rust,ignore
    /// let frame = FrameData::from_scene(
    ///     CameraFrame::from_camera(&camera, [w, h]),
    ///     &mut scene,
    ///     &selection,
    /// )
    /// .with_background([0.1, 0.1, 0.12, 1.0])
    /// .with_lighting(lighting);
    /// ```
    pub fn from_scene(
        camera: CameraFrame,
        scene: &mut crate::scene::scene::Scene,
        selection: &crate::interaction::select::selection::Selection,
    ) -> Self {
        Self {
            camera,
            scene: SceneFrame::from_scene(scene, selection),
            interaction: InteractionFrame::from_selection(selection),
            ..Self::default()
        }
    }

    /// Set the viewport background clear colour.
    pub fn with_background(mut self, colour: [f32; 4]) -> Self {
        self.viewport.background_colour = Some(colour);
        self
    }

    /// Override the per-frame lighting configuration.
    pub fn with_lighting(mut self, lighting: LightingSettings) -> Self {
        self.effects.lighting = lighting;
        self
    }

    /// Override the post-processing settings.
    pub fn with_post_process(mut self, post: PostProcessSettings) -> Self {
        self.effects.post_process = post;
        self
    }

    /// Override the ground plane configuration.
    pub fn with_ground_plane(mut self, ground: GroundPlane) -> Self {
        self.effects.ground_plane = ground;
        self
    }
}

// ---------------------------------------------------------------------------
// Draw-call macro (must be defined before use in impl block)
// ---------------------------------------------------------------------------

/// Internal macro that emits all draw calls. Used by both `paint` (egui /
/// `'static`) and `paint_to` (iced / any lifetime) to avoid duplicating
/// ~90 lines of rendering code while satisfying Rust's lifetime invariance
/// on `&mut RenderPass<'a>`.
macro_rules! emit_draw_calls {
    ($resources:expr, $render_pass:expr, $frame:expr, $use_instancing:expr, $batches:expr, $camera_bg:expr, $grid_bg:expr, $compute_filter_results:expr, $slot:expr, $wireframe_bgs:expr, $per_item_bgs:expr, $submesh_bgs:expr, $object_indices:expr, $submesh_indices:expr, $scene_items:expr, $po_bundle:expr) => {{
        let resources = $resources;
        let render_pass = $render_pass;
        let frame = $frame;
        let use_instancing: bool = $use_instancing;
        let po_bundle: Option<&crate::renderer::per_object_state::PerObjectBundle> = $po_bundle;
        let _vp_slot: Option<&ViewportSlot> = $slot;
        // Compute filter results: used by per-object path to override index buffers.
        let compute_filter_results: &[crate::resources::ComputeFilterResult] = $compute_filter_results;
        let batches: &[InstancedBatch] = $batches;
        let camera_bg: &crate::gpu::BindGroup = $camera_bg;
        let grid_bg: &crate::gpu::BindGroup = $grid_bg;
        let wireframe_bind_groups: &[crate::gpu::BindGroup] = $wireframe_bgs;
        // Per-item object bind groups indexed by position in scene_items. Each combines
        // a per-item ObjectUniform buffer with the mesh's real textures/LUT/matcap.
        // Items routed through the instanced path have None here and use mesh.object_bind_group.
        let per_item_object_bind_groups: &[Option<crate::gpu::BindGroup>] = $per_item_bgs;
        // Per-range bind groups for items drawn with per-submesh materials,
        // keyed by item position. Empty for frames without range items.
        let submesh_bind_groups: &std::collections::HashMap<
            usize,
            Vec<Option<crate::gpu::BindGroup>>,
        > = $submesh_bgs;
        // Object-data element index for each item's whole-mesh draw (parallel to
        // per_item_object_bind_groups) and for each submesh range. A per-object
        // draw binds the shared object-data buffer and selects its element via
        // this index as @builtin(instance_index); a None bind-group slot falls
        // back to the mesh's single-element buffer and draws at instance 0.
        let object_indices: &[u32] = $object_indices;
        let submesh_indices: &std::collections::HashMap<usize, Vec<u32>> = $submesh_indices;
        // Whole-mesh instance for an item: its object-data index when a shared
        // material bind group is used, else 0 for the mesh's fallback buffer.
        let object_inst = |item_idx: usize| -> u32 {
            match per_item_object_bind_groups
                .get(item_idx)
                .and_then(|o| o.as_ref())
            {
                Some(_) => object_indices.get(item_idx).copied().unwrap_or(0),
                None => 0,
            }
        };

        // The LOD-resolved surface items from prepare: each item's mesh is the
        // level chosen for its on-screen size, and culled items are hidden.
        // Matches `frame.scene.surfaces` in order and length, so the per-object
        // bind groups (indexed by position) line up unchanged.
        let scene_items: &[SceneRenderItem] = $scene_items;

        render_pass.set_bind_group(0, camera_bg, &[]);

        // Grid pass : full-screen analytical shader drawn first so scene geometry
        // occludes it. No vertex buffer; depth is written via @builtin(frag_depth).
        // Camera bind group is restored immediately after for subsequent passes.
        if frame.viewport.show_grid {
            render_pass.set_pipeline(&resources.grid_pipeline);
            render_pass.set_bind_group(0, grid_bg, &[]);
            render_pass.draw(0..3, 0..1);
            render_pass.set_bind_group(0, camera_bg, &[]);
        }

        // Ground plane pass : drawn after grid, before scene geometry.
        // Uses its own bind group (group 0: uniform + shadow atlas + sampler).
        if !matches!(
            frame.effects.ground_plane.mode,
            crate::renderer::types::GroundPlaneMode::None
        ) {
            render_pass.set_pipeline(&resources.ground_plane_pipeline);
            render_pass.set_bind_group(0, &resources.ground_plane_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
            render_pass.set_bind_group(0, camera_bg, &[]);
        }

            if !scene_items.is_empty() {
                if use_instancing && !batches.is_empty() {
                    let excluded_items: Vec<(usize, &SceneRenderItem)> = scene_items
                        .iter()
                        .enumerate()
                        .filter(|(_, item)| {
                            // Per-object set = visible items not admitted to an instanced
                            // batch. Reuse `is_instanceable` so this cannot drift from it.
                            !item.settings.hidden
                                && resources.mesh_store.get(item.mesh_id).is_some()
                                && !crate::renderer::prepare::is_instanceable(
                                    item,
                                    resources,
                                    compute_filter_results,
                                )
                        })
                        .collect();

                // --- Instanced draw path ---
                // Separate opaque and transparent batches.
                let mut opaque_batches: Vec<&InstancedBatch> = Vec::new();
                let mut transparent_batches: Vec<&InstancedBatch> = Vec::new();
                for batch in batches {
                    if batch.is_transparent {
                        transparent_batches.push(batch);
                    } else {
                        opaque_batches.push(batch);
                    }
                }

                    // Draw opaque instanced batches.
                    if !opaque_batches.is_empty() && !frame.viewport.wireframe_mode {
                        if let (Some(pipeline), Some(pipeline_two_sided)) = (
                            &resources.instancing.solid_pipeline,
                            &resources.instancing.solid_two_sided_pipeline,
                        ) {
                            // Early-Z fast path: discard-free pipeline twin for
                            // opaque batches when no clip object or alpha-mask
                            // instance can discard this frame.
                            let clipping_active = frame
                                .effects
                                .clip_objects
                                .iter()
                                .any(|o| o.enabled && o.clip_geometry);
                            let nodiscard_pipes = (
                                resources.instancing.solid_nodiscard_pipeline.as_ref(),
                                resources.instancing.solid_two_sided_nodiscard_pipeline.as_ref(),
                            );
                            bind_deform_group!(render_pass, resources, &resources.deform.dummy_bind_group);
                            // Batches are sorted with two_sided in the key, so one- and
                            // two-sided runs are contiguous; switch pipeline on change.
                            // Geometry is in the shared slab, so the chunk buffers bind
                            // once and each draw carries the mesh's base_vertex /
                            // first_index instead of re-binding a sub-slice per batch.
                            let mut cur_pipe: Option<(bool, bool)> = None;
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            for batch in &opaque_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else { continue };
                                let mat_key = (
                                    batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                );
                                // Combined (instance storage + texture) bind group, primed in prepare().
                                let Some(inst_tex_bg) = resources.instancing.bind_groups.get(&mat_key) else { continue };
                                let no_discard = !clipping_active
                                    && !batch.has_alpha_mask
                                    && nodiscard_pipes.0.is_some()
                                    && nodiscard_pipes.1.is_some();
                                if cur_pipe != Some((batch.two_sided, no_discard)) {
                                    render_pass.set_pipeline(match (no_discard, batch.two_sided) {
                                        (true, true) => nodiscard_pipes.1.unwrap(),
                                        (true, false) => nodiscard_pipes.0.unwrap(),
                                        (false, true) => pipeline_two_sided,
                                        (false, false) => pipeline,
                                    });
                                    cur_pipe = Some((batch.two_sided, no_discard));
                                }
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                                if cur_chunks != Some(chunks) {
                                    render_pass.set_vertex_buffer(0, resources.geometry.vertex_chunk_slice(chunks.0));
                                    render_pass.set_index_buffer(resources.geometry.index_chunk_slice(chunks.1), crate::gpu::IndexFormat::Uint32);
                                    cur_chunks = Some(chunks);
                                }
                                let base_vertex = resources.geometry.base_vertex(mesh.vertex_span);
                                let first_index = resources.geometry.first_index(mesh.index_span);
                                render_pass.draw_indexed(
                                    first_index..first_index + mesh.index_count,
                                    base_vertex,
                                    batch.instance_offset..batch.instance_offset + batch.instance_count,
                                );
                            }
                        }
                    }

                    // Draw transparent instanced batches.
                    if !transparent_batches.is_empty() && !frame.viewport.wireframe_mode {
                        if let Some(ref pipeline) = resources.instancing.transparent_pipeline {
                            render_pass.set_pipeline(pipeline);
                            bind_deform_group!(render_pass, resources, &resources.deform.dummy_bind_group);
                            let mut cur_chunks: Option<(u32, u32)> = None;
                            for batch in &transparent_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else { continue };
                                let mat_key = (
                                    batch.texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                    batch.ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) = resources.instancing.bind_groups.get(&mat_key) else { continue };
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                                if cur_chunks != Some(chunks) {
                                    render_pass.set_vertex_buffer(0, resources.geometry.vertex_chunk_slice(chunks.0));
                                    render_pass.set_index_buffer(resources.geometry.index_chunk_slice(chunks.1), crate::gpu::IndexFormat::Uint32);
                                    cur_chunks = Some(chunks);
                                }
                                let base_vertex = resources.geometry.base_vertex(mesh.vertex_span);
                                let first_index = resources.geometry.first_index(mesh.index_span);
                                render_pass.draw_indexed(
                                    first_index..first_index + mesh.index_count,
                                    base_vertex,
                                    batch.instance_offset..batch.instance_offset + batch.instance_count,
                                );
                            }
                        }
                    }

                    // Wireframe mode fallback: draw per-object using per-item bind
                    // groups so that items sharing a MeshId each get their own uniform.
                    if frame.viewport.wireframe_mode {
                        let mut wf_idx = 0usize;
                        for item in scene_items {
                            if item.settings.hidden { continue; }
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else { continue };
                            render_pass.set_pipeline(&resources.wireframe_pipeline);
                            bind_deform_group!(
                                render_pass,
                                resources,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance)
                            );
                            let bg = wireframe_bind_groups.get(wf_idx)
                                .unwrap_or(&mesh.object_bind_group);
                            render_pass.set_bind_group(1, bg, &[]);
                            render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                            if let Some(edge_buf) = &mesh.edge_index_buffer {
                                render_pass.set_index_buffer(edge_buf.slice(..), crate::gpu::IndexFormat::Uint32);
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            }
                            wf_idx += 1;
                        }
                    } else {
                        // LDR draws all excluded items here, including transparent ones
                        // using the transparent (alpha-blend) pipeline. HDR instead routes
                        // transparent excluded items to the OIT pass in render_frame_internal
                        // so they composite correctly with the HDR transparency model.
                        //
                        // Pipeline, deform bind group, and geometry buffers are only
                        // re-bound when they change between consecutive items: at
                        // thousands of per-object items the redundant re-binds dominate
                        // command-recording time (many scenes share one mesh and one
                        // pipeline across every item).
                        let mut cur_pipeline: Option<*const crate::gpu::RenderPipeline> = None;
                        let mut cur_deform: Option<*const crate::gpu::BindGroup> = None;
                        let mut cur_geometry: Option<(crate::resources::mesh::mesh_store::MeshId, bool)> = None;
                        for (item_idx, item) in &excluded_items {
                            let item_idx = *item_idx;
                            let Some(mesh) = resources
                                .mesh_store
                                .get(item.mesh_id)
                            else {
                                continue;
                            };
                            let is_blended = item.settings.opacity < 1.0
                                || item.material.is_blend();
                            let plug = resources.material_plugin_draw(item.material.shading_plugin);
                            let pipeline: &crate::gpu::RenderPipeline = if let Some((pp, _)) = plug {
                                if is_blended {
                                    &pp.ldr.transparent
                                } else if item.material.is_two_sided() {
                                    &pp.ldr.solid_two_sided
                                } else {
                                    &pp.ldr.solid
                                }
                            } else if is_blended {
                                &resources.transparent_pipeline
                            } else if item.material.is_two_sided() {
                                &resources.solid_two_sided_pipeline
                            } else {
                                &resources.solid_pipeline
                            };
                            if cur_pipeline != Some(pipeline as *const _) {
                                render_pass.set_pipeline(pipeline);
                                cur_pipeline = Some(pipeline as *const _);
                            }
                            let deform_bg = resources
                                .deform
                                .instance_bind_group_for(item.mesh_id, item.deform_instance);
                            if cur_deform != Some(deform_bg as *const _) {
                                bind_deform_group!(render_pass, resources, deform_bg);
                                cur_deform = Some(deform_bg as *const _);
                            }
                            render_pass.set_bind_group(1, per_item_object_bind_groups.get(item_idx).and_then(|opt| opt.as_ref()).unwrap_or(&mesh.object_bind_group), &[]);
                            if let Some((_, mat_bg)) = plug {
                                bind_material_group!(render_pass, mat_bg);
                            }

                            let is_face_attr = item.active_attribute.as_ref().map_or(false, |a| {
                                matches!(
                                    a.kind,
                                    crate::resources::AttributeKind::Face
                                        | crate::resources::AttributeKind::FaceColour
                                        | crate::resources::AttributeKind::Halfedge
                                        | crate::resources::AttributeKind::Corner
                                )
                            });
                            if is_face_attr {
                                if let Some(ref fvb) = mesh.face_vertex_buffer {
                                    render_pass.set_vertex_buffer(0, fvb.slice(..));
                                    let inst = object_inst(item_idx);
                                    render_pass.draw(0..mesh.index_count, inst..inst + 1);
                                    // The face path binds no index buffer, so the
                                    // cached geometry state no longer holds.
                                    cur_geometry = None;
                                }
                            } else if let Some((mats, bgs)) =
                                crate::renderer::prepare::active_submesh_materials(item, mesh)
                                    .zip(submesh_bind_groups.get(&item_idx))
                            {
                                // One draw per range. The LDR path draws blend
                                // ranges inline with the transparent pipeline,
                                // matching how it draws blended items.
                                if cur_geometry != Some((item.mesh_id, false)) {
                                    render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                                    render_pass.set_index_buffer(
                                        resources.geometry.index_slice(mesh.index_span),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    cur_geometry = Some((item.mesh_id, false));
                                }
                                for (r, (mat, range)) in
                                    mats.iter().zip(&mesh.submeshes).enumerate()
                                {
                                    let blended_r =
                                        item.settings.opacity < 1.0 || mat.is_blend();
                                    let plug_r =
                                        resources.material_plugin_draw(mat.shading_plugin);
                                    let pl: &crate::gpu::RenderPipeline =
                                        if let Some((pp, _)) = plug_r {
                                            if blended_r {
                                                &pp.ldr.transparent
                                            } else if mat.is_two_sided() {
                                                &pp.ldr.solid_two_sided
                                            } else {
                                                &pp.ldr.solid
                                            }
                                        } else if blended_r {
                                            &resources.transparent_pipeline
                                        } else if mat.is_two_sided() {
                                            &resources.solid_two_sided_pipeline
                                        } else {
                                            &resources.solid_pipeline
                                        };
                                    if cur_pipeline != Some(pl as *const _) {
                                        render_pass.set_pipeline(pl);
                                        cur_pipeline = Some(pl as *const _);
                                    }
                                    // Prefer the range's own bind group + index;
                                    // fall back to the item's whole-mesh slot, then
                                    // the mesh's single-element buffer at instance 0.
                                    let (bg, inst) = if let Some(rbg) =
                                        bgs.get(r).and_then(|b| b.as_ref())
                                    {
                                        (
                                            rbg,
                                            submesh_indices
                                                .get(&item_idx)
                                                .and_then(|v| v.get(r))
                                                .copied()
                                                .unwrap_or(0),
                                        )
                                    } else if let Some(ibg) = per_item_object_bind_groups
                                        .get(item_idx)
                                        .and_then(|opt| opt.as_ref())
                                    {
                                        (ibg, object_indices.get(item_idx).copied().unwrap_or(0))
                                    } else {
                                        (&mesh.object_bind_group, 0)
                                    };
                                    render_pass.set_bind_group(1, bg, &[]);
                                    if let Some((_, mat_bg)) = plug_r {
                                        bind_material_group!(render_pass, mat_bg);
                                    }
                                    render_pass.draw_indexed(
                                        range.first_index..range.first_index + range.index_count,
                                        0,
                                        inst..inst + 1,
                                    );
                                }
                            } else {
                                if cur_geometry != Some((item.mesh_id, false)) {
                                    render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                                    render_pass.set_index_buffer(
                                        resources.geometry.index_slice(mesh.index_span),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    cur_geometry = Some((item.mesh_id, false));
                                }
                                let inst = object_inst(item_idx);
                                render_pass.draw_indexed(0..mesh.index_count, 0, inst..inst + 1);
                            }
                        }
                    }
            } else {
                // --- Per-object draw path (original) ---
                let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);

                let dist_from_eye = |entry: &(usize, &SceneRenderItem)| -> f32 {
                    let item = entry.1;
                    let pos = glam::Vec3::new(
                        item.model[3][0],
                        item.model[3][1],
                        item.model[3][2],
                    );
                    (pos - eye).length()
                };

                // When prepare cached a render bundle for this item set and
                // this pass uses the camera bind group the bundle was recorded
                // with, replay it instead of re-recording one draw per opaque
                // item; only the blended items still draw immediately (their
                // back-to-front order depends on the camera every frame).
                let bundle_hit: Option<&crate::renderer::per_object_state::PerObjectBundle> =
                    po_bundle.filter(|pb| !pb.hdr && pb.camera_bg == *camera_bg);

                let mut opaque: Vec<(usize, &SceneRenderItem)> = Vec::new();
                let mut transparent: Vec<(usize, &SceneRenderItem)> = Vec::new();
                if let Some(pb) = bundle_hit {
                    for &idx in &pb.transparent {
                        if let Some(item) = scene_items.get(idx) {
                            if !item.settings.hidden
                                && resources.mesh_store.get(item.mesh_id).is_some()
                            {
                                transparent.push((idx, item));
                            }
                        }
                    }
                } else {
                    for (idx, item) in scene_items.iter().enumerate() {
                        if item.settings.hidden || resources.mesh_store.get(item.mesh_id).is_none() {
                            continue;
                        }
                        if item.settings.opacity < 1.0 {
                            transparent.push((idx, item));
                        } else {
                            opaque.push((idx, item));
                        }
                    }
                    opaque.sort_by(|a, b| dist_from_eye(a).partial_cmp(&dist_from_eye(b)).unwrap_or(std::cmp::Ordering::Equal));
                }
                transparent.sort_by(|a, b| dist_from_eye(b).partial_cmp(&dist_from_eye(a)).unwrap_or(std::cmp::Ordering::Equal));

                if let Some(pb) = bundle_hit {
                    render_pass.execute_bundles(std::iter::once(&pb.bundle));
                    // Bundle execution resets all render-pass state; restore
                    // the camera bind group for the draws and passes below.
                    render_pass.set_bind_group(0, camera_bg, &[]);
                }

                // Re-bind pipeline, deform bind group, and geometry buffers only
                // when they change between consecutive items; at thousands of
                // items the redundant re-binds dominate command recording.
                let mut cur_pipeline: Option<*const crate::gpu::RenderPipeline> = None;
                let mut cur_deform: Option<*const crate::gpu::BindGroup> = None;
                let mut cur_geometry: Option<(crate::resources::mesh::mesh_store::MeshId, bool)> = None;
                macro_rules! set_pipeline_cached {
                    ($pl:expr) => {{
                        let pl: &crate::gpu::RenderPipeline = $pl;
                        if cur_pipeline != Some(pl as *const _) {
                            render_pass.set_pipeline(pl);
                            cur_pipeline = Some(pl as *const _);
                        }
                    }};
                }
                macro_rules! set_deform_cached {
                    ($bg:expr) => {{
                        let bg: &crate::gpu::BindGroup = $bg;
                        if cur_deform != Some(bg as *const _) {
                            bind_deform_group!(render_pass, resources, bg);
                            cur_deform = Some(bg as *const _);
                        }
                    }};
                }
                macro_rules! draw_item {
                    ($entry:expr, $pipeline:expr) => {{
                        let (item_idx, item): (usize, &SceneRenderItem) = $entry;
                        let mesh = resources.mesh_store.get(item.mesh_id).unwrap();
                        render_pass.set_bind_group(1, per_item_object_bind_groups.get(item_idx).and_then(|opt| opt.as_ref()).unwrap_or(&mesh.object_bind_group), &[]);
                        if let Some((_, mat_bg)) =
                            resources.material_plugin_draw(item.material.shading_plugin)
                        {
                            bind_material_group!(render_pass, mat_bg);
                        }

                        let deform_bg = resources
                            .deform
                            .instance_bind_group_for(item.mesh_id, item.deform_instance);

                        // mesh.object_bind_group (group 1) already carries the object uniform
                        // and the correct texture views : updated in prepare() if material changed.
                        let is_face_attr = item.active_attribute.as_ref().map_or(false, |a| {
                            matches!(
                                a.kind,
                                crate::resources::AttributeKind::Face
                                    | crate::resources::AttributeKind::FaceColour
                                    | crate::resources::AttributeKind::Halfedge
                                    | crate::resources::AttributeKind::Corner
                            )
                        });

                        if frame.viewport.wireframe_mode {
                            if let Some(edge_buf) = &mesh.edge_index_buffer {
                                set_pipeline_cached!(&resources.wireframe_pipeline);
                                set_deform_cached!(deform_bg);
                                render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                                render_pass.set_index_buffer(
                                    edge_buf.slice(..),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                let inst = object_inst(item_idx);
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, inst..inst + 1);
                                cur_geometry = None;
                            }
                        } else if is_face_attr {
                            if let Some(ref fvb) = mesh.face_vertex_buffer {
                                set_pipeline_cached!($pipeline);
                                set_deform_cached!(deform_bg);
                                render_pass.set_vertex_buffer(0, fvb.slice(..));
                                let inst = object_inst(item_idx);
                                render_pass.draw(0..mesh.index_count, inst..inst + 1);
                                cur_geometry = None;
                            }
                        } else {
                            // Check for a compute-filtered index buffer override.
                            let filter_result = compute_filter_results
                                .iter()
                                .find(|r| r.mesh_id == item.mesh_id);
                            let ranges = if filter_result.is_none() {
                                // A compute-filtered index buffer is compacted,
                                // so the mesh's ranges no longer address it.
                                crate::renderer::prepare::active_submesh_materials(item, mesh)
                                    .zip(submesh_bind_groups.get(&item_idx))
                            } else {
                                None
                            };
                            if let Some((mats, bgs)) = ranges {
                                set_deform_cached!(deform_bg);
                                if cur_geometry != Some((item.mesh_id, false)) {
                                    render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                                    render_pass.set_index_buffer(
                                        resources.geometry.index_slice(mesh.index_span),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    cur_geometry = Some((item.mesh_id, false));
                                }
                                for (r, (mat, range)) in
                                    mats.iter().zip(&mesh.submeshes).enumerate()
                                {
                                    let blended_r =
                                        item.settings.opacity < 1.0 || mat.is_blend();
                                    let plug_r =
                                        resources.material_plugin_draw(mat.shading_plugin);
                                    let pl: &crate::gpu::RenderPipeline =
                                        if let Some((pp, _)) = plug_r {
                                            if blended_r {
                                                &pp.ldr.transparent
                                            } else if mat.is_two_sided() {
                                                &pp.ldr.solid_two_sided
                                            } else {
                                                &pp.ldr.solid
                                            }
                                        } else if blended_r {
                                            &resources.transparent_pipeline
                                        } else if mat.is_two_sided() {
                                            &resources.solid_two_sided_pipeline
                                        } else {
                                            &resources.solid_pipeline
                                        };
                                    set_pipeline_cached!(pl);
                                    // Prefer the range's own bind group + index;
                                    // fall back to the item's whole-mesh slot, then
                                    // the mesh's single-element buffer at instance 0.
                                    let (bg, inst) = if let Some(rbg) =
                                        bgs.get(r).and_then(|b| b.as_ref())
                                    {
                                        (
                                            rbg,
                                            submesh_indices
                                                .get(&item_idx)
                                                .and_then(|v| v.get(r))
                                                .copied()
                                                .unwrap_or(0),
                                        )
                                    } else if let Some(ibg) = per_item_object_bind_groups
                                        .get(item_idx)
                                        .and_then(|opt| opt.as_ref())
                                    {
                                        (ibg, object_indices.get(item_idx).copied().unwrap_or(0))
                                    } else {
                                        (&mesh.object_bind_group, 0)
                                    };
                                    render_pass.set_bind_group(1, bg, &[]);
                                    if let Some((_, mat_bg)) = plug_r {
                                        bind_material_group!(render_pass, mat_bg);
                                    }
                                    render_pass.draw_indexed(
                                        range.first_index..range.first_index + range.index_count,
                                        0,
                                        inst..inst + 1,
                                    );
                                }
                            } else {
                                set_pipeline_cached!($pipeline);
                                set_deform_cached!(deform_bg);
                                let inst = object_inst(item_idx);
                                if let Some(fr) = filter_result {
                                    render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                                    render_pass.set_index_buffer(
                                        fr.index_buffer.slice(..),
                                        crate::gpu::IndexFormat::Uint32,
                                    );
                                    render_pass.draw_indexed(0..fr.index_count, 0, inst..inst + 1);
                                    cur_geometry = None;
                                } else {
                                    if cur_geometry != Some((item.mesh_id, false)) {
                                        render_pass
                                            .set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                                        render_pass.set_index_buffer(
                                            resources.geometry.index_slice(mesh.index_span),
                                            crate::gpu::IndexFormat::Uint32,
                                        );
                                        cur_geometry = Some((item.mesh_id, false));
                                    }
                                    render_pass.draw_indexed(0..mesh.index_count, 0, inst..inst + 1);
                                }
                            }
                        }

                        if item.show_normals {
                            if let Some(ref nl_buf) = mesh.normal_line_buffer {
                                if mesh.normal_line_count > 0 {
                                    set_pipeline_cached!(&resources.wireframe_pipeline);
                                    set_deform_cached!(&resources.deform.dummy_bind_group);
                                    render_pass.set_bind_group(1, &mesh.normal_bind_group, &[]);
                                    render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                                    render_pass.draw(0..mesh.normal_line_count, 0..1);
                                    cur_geometry = None;
                                }
                            }
                        }
                    }};
                }

                for entry in &opaque {
                    let plug = resources.material_plugin_draw(entry.1.material.shading_plugin);
                    let pl = if let Some((pp, _)) = plug {
                        if entry.1.material.is_two_sided() {
                            &pp.ldr.solid_two_sided
                        } else {
                            &pp.ldr.solid
                        }
                    } else if entry.1.material.is_two_sided() {
                        &resources.solid_two_sided_pipeline
                    } else {
                        &resources.solid_pipeline
                    };
                    draw_item!((entry.0, entry.1), pl);
                }
                for entry in &transparent {
                    let pl = if let Some((pp, _)) =
                        resources.material_plugin_draw(entry.1.material.shading_plugin)
                    {
                        &pp.ldr.transparent
                    } else {
                        &resources.transparent_pipeline
                    };
                    draw_item!((entry.0, entry.1), pl);
                }
            }
        }

        // Gizmo pass.
        if let Some(slot) = _vp_slot {
            if frame.interaction.gizmo_model.is_some() && slot.gizmo_index_count > 0 {
                render_pass.set_pipeline(&resources.gizmo_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                render_pass.set_bind_group(1, &slot.gizmo_bind_group, &[]);
                render_pass.set_vertex_buffer(0, slot.gizmo_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    slot.gizmo_index_buffer.slice(..),
                    crate::gpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..slot.gizmo_index_count, 0, 0..1);
            }
        }

        // Constraint guide line pass.
        if let Some(slot) = _vp_slot {
            if !slot.constraint_line_buffers.is_empty() {
                render_pass.set_pipeline(&resources.overlay_line_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, index_count, _ubuf, bg) in &slot.constraint_line_buffers {
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.set_index_buffer(ibuf.slice(..), crate::gpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*index_count, 0, 0..1);
                }
            }
        }

        // Cap fill pass (section view cross-section fill).
        if let Some(slot) = _vp_slot {
            if !slot.cap_buffers.is_empty() {
                render_pass.set_pipeline(&resources.overlay_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.cap_buffers {
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.set_index_buffer(ibuf.slice(..), crate::gpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                }
            }
        }

        // Clip plane handle fill pass (semi-transparent quad fills, alpha blended).
        if let Some(slot) = _vp_slot {
            if !slot.clip_plane_fill_buffers.is_empty() {
                render_pass.set_pipeline(&resources.overlay_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_fill_buffers {
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.set_index_buffer(ibuf.slice(..), crate::gpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                }
            }
        }

        // Clip plane handle border and normal indicator pass (line list).
        if let Some(slot) = _vp_slot {
            if !slot.clip_plane_line_buffers.is_empty() {
                render_pass.set_pipeline(&resources.overlay_line_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_line_buffers {
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.set_index_buffer(ibuf.slice(..), crate::gpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                }
            }
        }

        // X-ray pass: render selected objects as semi-transparent overlay through geometry.
        if let Some(slot) = _vp_slot {
            if !slot.xray_object_buffers.is_empty() {
                render_pass.set_pipeline(&resources.outline.xray_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (mesh_id, _buf, bg) in &slot.xray_object_buffers {
                    let Some(mesh) = resources.mesh_store.get(*mesh_id) else { continue };
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                    render_pass.set_index_buffer(resources.geometry.index_slice(mesh.index_span), crate::gpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        // Axes indicator pass (screen-space, last so it draws on top).
        if let Some(slot) = _vp_slot {
            slot.draw_axes_indicator(
                &mut *render_pass,
                resources,
                frame.viewport.show_axes_indicator,
            );
        }
    }};
}

/// Blit the offscreen outline texture onto the main render target.
///
/// Must run after all scene content (meshes, scivis items, splats, implicit
/// surfaces, marching cubes) so that translucent layers like volumes don't
/// overdraw the outline.
macro_rules! emit_outline_composite {
    ($resources:expr, $render_pass:expr, $vp_slot:expr) => {{
        let resources = $resources;
        let render_pass = $render_pass;
        if let Some(slot) = $vp_slot {
            if !slot.outline_object_buffers.is_empty()
                || !slot.splat_outline_buffers.is_empty()
                || !slot.streamtube_outline_items.is_empty()
                || !slot.tube_outline_items.is_empty()
                || !slot.ribbon_outline_items.is_empty()
                || !slot.polyline_outline_indices.is_empty()
                || !slot.volume_outline_indices.is_empty()
                || !slot.glyph_outline_indices.is_empty()
                || !slot.tensor_glyph_outline_indices.is_empty()
                || !slot.sprite_outline_indices.is_empty()
                || slot.plugin_outline_present
            {
                let composite_bg = slot.hdr.as_ref().map(|h| &h.outline_composite_bind_group);
                let pipeline = resources
                    .outline
                    .composite_pipeline_msaa
                    .as_ref()
                    .or(resources.outline.composite_pipeline_single.as_ref());
                if let (Some(pipeline), Some(bg)) = (pipeline, composite_bg) {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, bg, &[]);
                    render_pass.draw(0..3, 0..1);
                }
            }
        }
    }};
}

/// Draw point cloud and glyph items from per-frame GPU data prepared in `prepare()`.
///
/// Called by both `paint` and `paint_to` after `emit_draw_calls!` to render scivis layers.
macro_rules! emit_scivis_draw_calls {
    ($resources:expr, $render_pass:expr, $pc_gpu_data:expr, $glyph_gpu_data:expr, $polyline_gpu_data:expr, $volume_gpu_data:expr, $streamtube_gpu_data:expr, $camera_bg:expr, $tube_gpu_data:expr, $image_slice_gpu_data:expr, $tensor_glyph_gpu_data:expr, $ribbon_gpu_data:expr, $volume_surface_slice_gpu_data:expr, $sprite_gpu_data:expr, $mesh_instance_gpu_data:expr, $is_hdr:expr) => {{
        let resources = $resources;
        let render_pass = $render_pass;
        let camera_bg: &crate::gpu::BindGroup = $camera_bg;
        let _is_hdr: bool = $is_hdr;

        // Point cloud pass.
        if !$pc_gpu_data.is_empty() {
            if let Some(ref dual) = resources.point_cloud_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for pc in $pc_gpu_data.iter() {
                    render_pass.set_bind_group(1, &pc.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, pc.vertex_buffer.slice(..));
                    // 6 vertices per point (billboard quad = 2 triangles), point_count instances.
                    render_pass.draw(0..6, 0..pc.point_count);
                }
            }
        }

        // Glyph pass.
        if !$glyph_gpu_data.is_empty() {
            render_pass.set_bind_group(0, camera_bg, &[]);
            for glyph in $glyph_gpu_data.iter() {
                let pipeline = if glyph.wireframe {
                    resources
                        .glyph
                        .wireframe_pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                } else {
                    resources
                        .glyph
                        .pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                };
                if let Some(pipeline) = pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(1, &glyph.uniform_bind_group, &[]);
                    render_pass.set_bind_group(2, &glyph.instance_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, glyph.mesh_vertex_buffer.slice(..));
                    if glyph.wireframe {
                        render_pass.set_index_buffer(
                            glyph.mesh_edge_index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(
                            0..glyph.mesh_edge_index_count,
                            0,
                            0..glyph.instance_count,
                        );
                    } else {
                        render_pass.set_index_buffer(
                            glyph.mesh_index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(
                            0..glyph.mesh_index_count,
                            0,
                            0..glyph.instance_count,
                        );
                    }
                }
            }
        }

        // Polyline pass : screen-space thick lines via instanced quad expansion.
        // Each segment instance is drawn as 6 vertices (2 triangles).
        // Items with skip_clip=true (clip object wireframe overlays) use the clip-exempt
        // pipeline so they are always fully visible regardless of active clip volumes.
        // Items with wireframe=true use the thin 1px LineList pipeline instead.
        if !$polyline_gpu_data.is_empty() && resources.polyline.pipeline.is_some() {
            for pl in $polyline_gpu_data.iter() {
                if pl.segment_count == 0 {
                    continue;
                }
                if pl.wireframe {
                    if let (Some(wf_pipeline), Some(wf_bg)) = (
                        resources
                            .polyline
                            .wireframe_pipeline
                            .as_ref()
                            .map(|d| d.for_format(_is_hdr)),
                        pl.wireframe_bind_group.as_ref(),
                    ) {
                        render_pass.set_pipeline(wf_pipeline);
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        render_pass.set_bind_group(1, wf_bg, &[]);
                        render_pass.draw(0..2, 0..pl.segment_count);
                    }
                    continue;
                }
                let pipeline = if pl.skip_clip {
                    resources
                        .polyline
                        .no_clip_pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                } else {
                    resources
                        .polyline
                        .pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                };
                if let Some(pipeline) = pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    render_pass.set_bind_group(1, &pl.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, pl.vertex_buffer.slice(..));
                    render_pass.draw(0..6, 0..pl.segment_count);
                }
            }
        }

        // Volume pass (after glyphs : volumes are translucent, rendered last).
        if !$volume_gpu_data.is_empty() {
            if let Some(ref dual) = resources.volume.pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for vol in $volume_gpu_data.iter() {
                    if vol.wireframe {
                        continue;
                    }
                    render_pass.set_bind_group(1, &vol.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, vol.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        vol.index_buffer.slice(..),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..36, 0, 0..1);
                }
            }
        }

        // Streamtube pass: connected tube mesh per strip set).
        if !$streamtube_gpu_data.is_empty() {
            render_pass.set_bind_group(0, camera_bg, &[]);
            for tube in $streamtube_gpu_data.iter() {
                if tube.index_count == 0 && tube.edge_index_count == 0 {
                    continue;
                }
                let pipeline = if tube.wireframe {
                    resources
                        .streamtube
                        .wireframe_pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                } else {
                    resources
                        .streamtube
                        .pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                };
                if let Some(pipeline) = pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(1, &tube.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, tube.vertex_buffer.slice(..));
                    if tube.wireframe {
                        render_pass.set_index_buffer(
                            tube.edge_index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..tube.edge_index_count, 0, 0..1);
                    } else {
                        render_pass.set_index_buffer(
                            tube.index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..tube.index_count, 0, 0..1);
                    }
                }
            }
        }

        // General tube pass (uses same streamtube pipeline, per-vertex colour).
        if !$tube_gpu_data.is_empty() {
            render_pass.set_bind_group(0, camera_bg, &[]);
            for tube in $tube_gpu_data.iter() {
                if tube.index_count == 0 && tube.edge_index_count == 0 {
                    continue;
                }
                let pipeline = if tube.wireframe {
                    resources
                        .streamtube
                        .wireframe_pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                } else {
                    resources
                        .streamtube
                        .pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                };
                if let Some(pipeline) = pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(1, &tube.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, tube.vertex_buffer.slice(..));
                    if tube.wireframe {
                        render_pass.set_index_buffer(
                            tube.edge_index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..tube.edge_index_count, 0, 0..1);
                    } else {
                        render_pass.set_index_buffer(
                            tube.index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..tube.index_count, 0, 0..1);
                    }
                }
            }
        }

        // Image slice pass (no vertex buffer, 6 vertices generated by shader).
        if !$image_slice_gpu_data.is_empty() {
            if let Some(ref dual) = resources.image_slice.pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for slice in $image_slice_gpu_data.iter() {
                    render_pass.set_bind_group(1, &slice.bind_group, &[]);
                    render_pass.draw(0..6, 0..1);
                }
            }
        }

        // Tensor glyph pass (instanced ellipsoids for stress/strain tensors).
        if !$tensor_glyph_gpu_data.is_empty() {
            render_pass.set_bind_group(0, camera_bg, &[]);
            for tg in $tensor_glyph_gpu_data.iter() {
                let pipeline = if tg.wireframe {
                    resources
                        .tensor_glyph
                        .wireframe_pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                } else {
                    resources
                        .tensor_glyph
                        .pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                };
                if let Some(pipeline) = pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(1, &tg.uniform_bind_group, &[]);
                    render_pass.set_bind_group(2, &tg.instance_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, tg.mesh_vertex_buffer.slice(..));
                    if tg.wireframe {
                        render_pass.set_index_buffer(
                            tg.mesh_edge_index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(
                            0..tg.mesh_edge_index_count,
                            0,
                            0..tg.instance_count,
                        );
                    } else {
                        render_pass.set_index_buffer(
                            tg.mesh_index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..tg.mesh_index_count, 0, 0..tg.instance_count);
                    }
                }
            }
        }

        // Volume surface slice pass (arbitrary mesh sampled from volume).
        if !$volume_surface_slice_gpu_data.is_empty() {
            if let Some(ref dual) = resources.volume.surface_slice_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for slice in $volume_surface_slice_gpu_data.iter() {
                    if let Some(mesh) = resources.mesh_store.get(slice.mesh_id) {
                        render_pass.set_bind_group(1, &slice.bind_group, &[]);
                        render_pass.set_vertex_buffer(
                            0,
                            resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        render_pass.set_index_buffer(
                            resources.geometry.index_slice(mesh.index_span),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }
        }

        // Ribbon pass (flat quad strips, two-sided pipeline). Blend mode
        // routes through the matching pipeline variant; the default
        // AlphaBlend path is unchanged.
        if !$ribbon_gpu_data.is_empty() {
            render_pass.set_bind_group(0, camera_bg, &[]);
            for ribbon in $ribbon_gpu_data.iter() {
                if ribbon.index_count == 0 && ribbon.edge_index_count == 0 {
                    continue;
                }
                let pipeline = if ribbon.wireframe {
                    resources
                        .ribbon
                        .wireframe_pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                } else {
                    match ribbon.blend {
                        crate::renderer::SpriteBlend::Additive => resources
                            .ribbon
                            .pipeline_additive
                            .as_ref()
                            .map(|d| d.for_format(_is_hdr)),
                        crate::renderer::SpriteBlend::Premultiplied => resources
                            .ribbon
                            .pipeline_premultiplied
                            .as_ref()
                            .map(|d| d.for_format(_is_hdr)),
                        crate::renderer::SpriteBlend::AlphaBlend => resources
                            .ribbon
                            .pipeline
                            .as_ref()
                            .map(|d| d.for_format(_is_hdr)),
                    }
                };
                if let Some(pipeline) = pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(1, &ribbon.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, ribbon.vertex_buffer.slice(..));
                    if ribbon.wireframe {
                        render_pass.set_index_buffer(
                            ribbon.edge_index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..ribbon.edge_index_count, 0, 0..1);
                    } else {
                        render_pass.set_index_buffer(
                            ribbon.index_buffer.slice(..),
                            crate::gpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..ribbon.index_count, 0, 0..1);
                    }
                }
            }
        }

        // Mesh-instance pass: one draw call per host-built batch, routed by
        // blend mode. Reuses the scene-graph instanced mesh pipeline family.
        if !$mesh_instance_gpu_data.is_empty() {
            let mesh_buckets: [(
                crate::renderer::SpriteBlend,
                Option<&crate::gpu::RenderPipeline>,
            ); 3] = [
                (
                    crate::renderer::SpriteBlend::AlphaBlend,
                    resources.instancing.hdr_transparent_pipeline.as_ref(),
                ),
                (
                    crate::renderer::SpriteBlend::Additive,
                    resources.instancing.hdr_additive_pipeline.as_ref(),
                ),
                (
                    crate::renderer::SpriteBlend::Premultiplied,
                    resources.instancing.hdr_premultiplied_pipeline.as_ref(),
                ),
            ];
            for (blend, pipeline) in mesh_buckets {
                let Some(pipeline) = pipeline else { continue };
                let mut set = false;
                for batch in $mesh_instance_gpu_data.iter() {
                    if batch.blend != blend {
                        continue;
                    }
                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                        continue;
                    };
                    if mesh.index_count == 0 {
                        continue;
                    }
                    if !set {
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        set = true;
                    }
                    render_pass.set_bind_group(1, &batch.bind_group, &[]);
                    // mesh_instanced.wgsl's pipeline layout includes the deform
                    // bind group at index 2. MeshInstanceItem does not expose
                    // per-instance deform handles, so bind the per-mesh
                    // fallback (or the dummy group when the mesh has no
                    // attached deform data).
                    let deform_bg = resources
                        .deform
                        .instance_bind_group_for(batch.mesh_id, None);
                    bind_deform_group!(render_pass, resources, deform_bg);
                    render_pass
                        .set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                    render_pass.set_index_buffer(
                        resources.geometry.index_slice(mesh.index_span),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..batch.instance_count);
                }
            }
        }

        // Sprite billboard pass: route by (depth_write, blend mode).
        // Depth-write items first (opaque-style markers), then the no-depth-write
        // batches (transparent / additive / premultiplied particle effects).
        if !$sprite_gpu_data.is_empty() {
            let buckets: [(
                bool,
                crate::renderer::SpriteBlend,
                Option<&crate::resources::DualPipeline>,
            ); 6] = [
                (
                    true,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    resources.sprite.pipeline_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Additive,
                    resources.sprite.pipeline_additive_depth_write.as_ref(),
                ),
                (
                    true,
                    crate::renderer::SpriteBlend::Premultiplied,
                    resources.sprite.pipeline_premultiplied_depth_write.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::AlphaBlend,
                    resources.sprite.pipeline.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Additive,
                    resources.sprite.pipeline_additive.as_ref(),
                ),
                (
                    false,
                    crate::renderer::SpriteBlend::Premultiplied,
                    resources.sprite.pipeline_premultiplied.as_ref(),
                ),
            ];
            // Group 2 (sprite_soft_bgl) carries the scene-depth resolve consumed
            // by soft-particle fade. Inline sprite draws inside the main HDR or
            // LDR pass cannot bind the live scene depth (it is still being
            // written), so they bind a placeholder. The shader gates the fade
            // sample on a positive soft_particle_distance, so non-fade items
            // ignore this binding's contents. Soft fade itself is applied in the
            // separate transparent-sprite post-pass driven from `render.rs`.
            let soft_bg = resources.sprite.soft_fallback_bg.as_ref();
            for (depth_write, blend, pipeline) in buckets {
                let Some(dual) = pipeline else { continue };
                let Some(soft_bg) = soft_bg else { continue };
                let mut set = false;
                for sprite in $sprite_gpu_data.iter() {
                    if sprite.wireframe
                        || sprite.depth_write != depth_write
                        || sprite.blend != blend
                    {
                        continue;
                    }
                    if !set {
                        render_pass.set_pipeline(dual.for_format(_is_hdr));
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        render_pass.set_bind_group(2, soft_bg, &[]);
                        set = true;
                    }
                    render_pass.set_bind_group(1, &sprite.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
                    render_pass.draw(0..6, 0..sprite.sprite_count);
                }
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_camera_from_camera_roundtrip() {
        let cam = crate::camera::Camera::default();
        let rc = RenderCamera::from_camera(&cam);
        assert_eq!(rc.eye_position, cam.eye_position().to_array());
        assert_eq!(rc.orientation, cam.orientation);
        assert_eq!(rc.near, cam.effective_znear());
        assert_eq!(rc.far, cam.zfar);
        assert_eq!(rc.fov, cam.fov_y);
        assert_eq!(rc.aspect, cam.aspect);
        // view_proj should match Camera's own method
        let expected_vp = cam.view_proj_matrix();
        let actual_vp = rc.view_proj();
        assert!(
            (expected_vp - actual_vp).abs_diff_eq(glam::Mat4::ZERO, 1e-5),
            "view_proj mismatch"
        );
    }

    #[test]
    fn render_camera_uniform_contains_eye_and_forward() {
        let rc = RenderCamera {
            eye_position: [1.0, 2.0, 3.0],
            forward: [0.0, 0.0, -1.0],
            ..RenderCamera::default()
        };
        let u = rc.camera_uniform();
        assert_eq!(u.eye_pos, [1.0, 2.0, 3.0]);
        assert_eq!(u.forward, [0.0, 0.0, -1.0]);
        assert_eq!(u.view_proj, rc.view_proj().to_cols_array_2d());
    }
}
