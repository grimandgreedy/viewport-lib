//! `ViewportRenderer` : the main entry point for the viewport library.
//!
//! Wraps [`ViewportGpuResources`] and provides `prepare()` / `paint()` methods
//! that take raw `wgpu` types. GUI framework adapters (e.g. the egui
//! `CallbackTrait` impl in the application crate) delegate to these methods.

/// Minimum scene item count to activate the instanced draw path.
/// Use instancing for any scene with more than 1 object. The per-object path
/// writes uniforms into a per-mesh buffer, so two scene nodes sharing the same
/// mesh would clobber each other. Instancing avoids this by keeping per-item
/// data in a separate instance buffer indexed by draw-call range.
pub(super) const INSTANCING_THRESHOLD: usize = 1;

/// A batch of instances sharing the same mesh and material textures, drawn in one call.
#[derive(Debug, Clone)]
pub(crate) struct InstancedBatch {
    pub mesh_id: crate::resources::mesh_store::MeshId,
    pub texture_id: Option<u64>,
    pub normal_map_id: Option<u64>,
    pub ao_map_id: Option<u64>,
    pub instance_offset: u32,
    pub instance_count: u32,
    pub is_transparent: bool,
}

mod clip;
mod frame;
mod items;
mod lighting;
mod overlay;
mod postprocess;

pub use self::clip::*;
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
        selection: &crate::interaction::selection::Selection,
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
    ($resources:expr, $render_pass:expr, $frame:expr, $use_instancing:expr, $batches:expr, $camera_bg:expr, $grid_bg:expr, $compute_filter_results:expr, $slot:expr, $wireframe_bgs:expr) => {{
        let resources = $resources;
        let render_pass = $render_pass;
        let frame = $frame;
        let use_instancing: bool = $use_instancing;
        let _vp_slot: Option<&ViewportSlot> = $slot;
        // Phase G compute filter results: used by per-object path to override index buffers.
        let compute_filter_results: &[crate::resources::ComputeFilterResult] = $compute_filter_results;
        let batches: &[InstancedBatch] = $batches;
        let camera_bg: &wgpu::BindGroup = $camera_bg;
        let grid_bg: &wgpu::BindGroup = $grid_bg;
        let wireframe_bind_groups: &[wgpu::BindGroup] = $wireframe_bgs;

        // Read scene items from the surface submission.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

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
                    let excluded_items: Vec<&SceneRenderItem> = scene_items
                        .iter()
                        .filter(|item| {
                            item.visible
                                && (item.active_attribute.is_some()
                                    || item.material.is_two_sided()
                                    || item.material.param_vis.is_some())
                                && resources
                                    .mesh_store
                                    .get(item.mesh_id)
                                    .is_some()
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
                        if let Some(ref pipeline) = resources.solid_instanced_pipeline {
                            render_pass.set_pipeline(pipeline);
                            for batch in &opaque_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else { continue };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                // Combined (instance storage + texture) bind group, primed in prepare().
                                let Some(inst_tex_bg) = resources.instance_bind_groups.get(&mat_key) else { continue };
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset..batch.instance_offset + batch.instance_count,
                                );
                            }
                        }
                    }

                    // Draw transparent instanced batches.
                    if !transparent_batches.is_empty() && !frame.viewport.wireframe_mode {
                        if let Some(ref pipeline) = resources.transparent_instanced_pipeline {
                            render_pass.set_pipeline(pipeline);
                            for batch in &transparent_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else { continue };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) = resources.instance_bind_groups.get(&mat_key) else { continue };
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
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
                            if !item.visible { continue; }
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else { continue };
                            render_pass.set_pipeline(&resources.wireframe_pipeline);
                            let bg = wireframe_bind_groups.get(wf_idx)
                                .unwrap_or(&mesh.object_bind_group);
                            render_pass.set_bind_group(1, bg, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(mesh.edge_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            wf_idx += 1;
                        }
                    } else {
                        for item in &excluded_items {
                            let Some(mesh) = resources
                                .mesh_store
                                .get(item.mesh_id)
                            else {
                                continue;
                            };
                            let pipeline = if item.material.opacity < 1.0 {
                                &resources.transparent_pipeline
                            } else if item.material.is_two_sided() {
                                &resources.solid_two_sided_pipeline
                            } else {
                                &resources.solid_pipeline
                            };
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);

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
                                    render_pass.draw(0..mesh.index_count, 0..1);
                                }
                            } else {
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            }
                        }
                    }
            } else {
                // --- Per-object draw path (original) ---
                let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);

                let dist_from_eye = |item: &&SceneRenderItem| -> f32 {
                    let pos = glam::Vec3::new(
                        item.model[3][0],
                        item.model[3][1],
                        item.model[3][2],
                    );
                    (pos - eye).length()
                };

                let mut opaque: Vec<&SceneRenderItem> = Vec::new();
                let mut transparent: Vec<&SceneRenderItem> = Vec::new();
                for item in scene_items {
                    if !item.visible || resources.mesh_store.get(item.mesh_id).is_none() {
                        continue;
                    }
                    if item.material.opacity < 1.0 {
                        transparent.push(item);
                    } else {
                        opaque.push(item);
                    }
                }
                opaque.sort_by(|a, b| dist_from_eye(a).partial_cmp(&dist_from_eye(b)).unwrap_or(std::cmp::Ordering::Equal));
                transparent.sort_by(|a, b| dist_from_eye(b).partial_cmp(&dist_from_eye(a)).unwrap_or(std::cmp::Ordering::Equal));

                macro_rules! draw_item {
                    ($item:expr, $pipeline:expr) => {{
                        let item = $item;
                        let mesh = resources.mesh_store.get(item.mesh_id).unwrap();
                        render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);

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
                            render_pass.set_pipeline(&resources.wireframe_pipeline);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(
                                mesh.edge_index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                        } else if is_face_attr {
                            if let Some(ref fvb) = mesh.face_vertex_buffer {
                                render_pass.set_pipeline($pipeline);
                                render_pass.set_vertex_buffer(0, fvb.slice(..));
                                render_pass.draw(0..mesh.index_count, 0..1);
                            }
                        } else {
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            // Phase G: check for a compute-filtered index buffer override.
                            let filter_result = compute_filter_results
                                .iter()
                                .find(|r| r.mesh_id == item.mesh_id);
                            render_pass.set_pipeline($pipeline);
                            if let Some(fr) = filter_result {
                                render_pass.set_index_buffer(
                                    fr.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..fr.index_count, 0, 0..1);
                            } else {
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            }
                        }

                        if item.show_normals {
                            if let Some(ref nl_buf) = mesh.normal_line_buffer {
                                if mesh.normal_line_count > 0 {
                                    render_pass.set_pipeline(&resources.wireframe_pipeline);
                                    render_pass.set_bind_group(1, &mesh.normal_bind_group, &[]);
                                    render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                                    render_pass.draw(0..mesh.normal_line_count, 0..1);
                                }
                            }
                        }
                    }};
                }

                for item in &opaque {
                    let pl = if item.material.is_two_sided() {
                        &resources.solid_two_sided_pipeline
                    } else {
                        &resources.solid_pipeline
                    };
                    draw_item!(item, pl);
                }
                for item in &transparent {
                    draw_item!(item, &resources.transparent_pipeline);
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
                    wgpu::IndexFormat::Uint32,
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
                    render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
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
                    render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
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
                    render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
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
                    render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                }
            }
        }

        // X-ray pass: render selected objects as semi-transparent overlay through geometry.
        if let Some(slot) = _vp_slot {
            if !slot.xray_object_buffers.is_empty() {
                render_pass.set_pipeline(&resources.xray_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (mesh_id, _buf, bg) in &slot.xray_object_buffers {
                    let Some(mesh) = resources.mesh_store.get(*mesh_id) else { continue };
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        // Axes indicator pass (screen-space, last so it draws on top).
        if let Some(slot) = _vp_slot {
            if frame.viewport.show_axes_indicator && slot.axes_vertex_count > 0 {
                render_pass.set_pipeline(&resources.axes_pipeline);
                render_pass.set_vertex_buffer(0, slot.axes_vertex_buffer.slice(..));
                render_pass.draw(0..slot.axes_vertex_count, 0..1);
            }
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
                || !slot.volume_outline_indices.is_empty()
                || !slot.glyph_outline_indices.is_empty()
                || !slot.tensor_glyph_outline_indices.is_empty()
                || !slot.sprite_outline_indices.is_empty()
            {
                let composite_bg = slot.hdr.as_ref().map(|h| &h.outline_composite_bind_group);
                let pipeline = resources
                    .outline_composite_pipeline_msaa
                    .as_ref()
                    .or(resources.outline_composite_pipeline_single.as_ref());
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
    ($resources:expr, $render_pass:expr, $pc_gpu_data:expr, $glyph_gpu_data:expr, $polyline_gpu_data:expr, $volume_gpu_data:expr, $streamtube_gpu_data:expr, $camera_bg:expr, $tube_gpu_data:expr, $image_slice_gpu_data:expr, $tensor_glyph_gpu_data:expr, $ribbon_gpu_data:expr, $volume_surface_slice_gpu_data:expr, $sprite_gpu_data:expr, $is_hdr:expr) => {{
        let resources = $resources;
        let render_pass = $render_pass;
        let camera_bg: &wgpu::BindGroup = $camera_bg;
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
            if let Some(ref dual) = resources.glyph_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for glyph in $glyph_gpu_data.iter() {
                    render_pass.set_bind_group(1, &glyph.uniform_bind_group, &[]);
                    render_pass.set_bind_group(2, &glyph.instance_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, glyph.mesh_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        glyph.mesh_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..glyph.mesh_index_count, 0, 0..glyph.instance_count);
                }
            }
        }

        // Polyline pass : screen-space thick lines via instanced quad expansion.
        // Each segment instance is drawn as 6 vertices (2 triangles).
        // Items with skip_clip=true (clip object wireframe overlays) use the clip-exempt
        // pipeline so they are always fully visible regardless of active clip volumes.
        if !$polyline_gpu_data.is_empty() && resources.polyline_pipeline.is_some() {
            for pl in $polyline_gpu_data.iter() {
                if pl.segment_count == 0 {
                    continue;
                }
                let pipeline = if pl.skip_clip {
                    resources
                        .polyline_no_clip_pipeline
                        .as_ref()
                        .map(|d| d.for_format(_is_hdr))
                } else {
                    resources
                        .polyline_pipeline
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
            if let Some(ref dual) = resources.volume_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for vol in $volume_gpu_data.iter() {
                    render_pass.set_bind_group(1, &vol.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, vol.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(vol.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..36, 0, 0..1);
                }
            }
        }

        // Streamtube pass (SciVis Phase M : connected tube mesh per strip set).
        if !$streamtube_gpu_data.is_empty() {
            if let Some(ref dual) = resources.streamtube_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for tube in $streamtube_gpu_data.iter() {
                    if tube.index_count == 0 {
                        continue;
                    }
                    render_pass.set_bind_group(1, &tube.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, tube.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(tube.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..tube.index_count, 0, 0..1);
                }
            }
        }

        // General tube pass (Phase 3.3 : uses same streamtube pipeline, per-vertex colour).
        if !$tube_gpu_data.is_empty() {
            if let Some(ref dual) = resources.streamtube_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for tube in $tube_gpu_data.iter() {
                    if tube.index_count == 0 {
                        continue;
                    }
                    render_pass.set_bind_group(1, &tube.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, tube.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(tube.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..tube.index_count, 0, 0..1);
                }
            }
        }

        // Image slice pass (Phase 3.2 : no vertex buffer, 6 vertices generated by shader).
        if !$image_slice_gpu_data.is_empty() {
            if let Some(ref dual) = resources.image_slice_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for slice in $image_slice_gpu_data.iter() {
                    render_pass.set_bind_group(1, &slice.bind_group, &[]);
                    render_pass.draw(0..6, 0..1);
                }
            }
        }

        // Tensor glyph pass (Phase 5 : instanced ellipsoids for stress/strain tensors).
        if !$tensor_glyph_gpu_data.is_empty() {
            if let Some(ref dual) = resources.tensor_glyph_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for tg in $tensor_glyph_gpu_data.iter() {
                    render_pass.set_bind_group(1, &tg.uniform_bind_group, &[]);
                    render_pass.set_bind_group(2, &tg.instance_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, tg.mesh_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        tg.mesh_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..tg.mesh_index_count, 0, 0..tg.instance_count);
                }
            }
        }

        // Volume surface slice pass (Phase 10 : arbitrary mesh sampled from volume).
        if !$volume_surface_slice_gpu_data.is_empty() {
            if let Some(ref dual) = resources.volume_surface_slice_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for slice in $volume_surface_slice_gpu_data.iter() {
                    if let Some(mesh) = resources.mesh_store.get(slice.mesh_id) {
                        render_pass.set_bind_group(1, &slice.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }
        }

        // Ribbon pass (Phase 8.1 : flat quad strips, two-sided pipeline).
        if !$ribbon_gpu_data.is_empty() {
            if let Some(ref dual) = resources.ribbon_pipeline {
                render_pass.set_pipeline(dual.for_format(_is_hdr));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for ribbon in $ribbon_gpu_data.iter() {
                    if ribbon.index_count == 0 {
                        continue;
                    }
                    render_pass.set_bind_group(1, &ribbon.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, ribbon.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(ribbon.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..ribbon.index_count, 0, 0..1);
                }
            }
        }

        // Sprite billboard pass: depth-write items first, then transparent items.
        if !$sprite_gpu_data.is_empty() {
            // Depth-write batch (opaque-style markers).
            if let Some(ref dual) = resources.sprite_pipeline_depth_write {
                let mut set = false;
                for sprite in $sprite_gpu_data.iter() {
                    if !sprite.depth_write {
                        continue;
                    }
                    if !set {
                        render_pass.set_pipeline(dual.for_format(_is_hdr));
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        set = true;
                    }
                    render_pass.set_bind_group(1, &sprite.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
                    render_pass.draw(0..6, 0..sprite.sprite_count);
                }
            }
            // No-depth-write batch (transparent effects, default).
            if let Some(ref dual) = resources.sprite_pipeline {
                let mut set = false;
                for sprite in $sprite_gpu_data.iter() {
                    if sprite.depth_write {
                        continue;
                    }
                    if !set {
                        render_pass.set_pipeline(dual.for_format(_is_hdr));
                        render_pass.set_bind_group(0, camera_bg, &[]);
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
