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
/// third group (soft body, refraction, scivis LUT, volumes, vector/tensor
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

/// Bind a material plugin's group-3 texture bind group for a plugin draw.
///
/// Only called with a bind group resolved by `material_plugin_draw`, which
/// only exists on devices that passed the `max_bind_groups >= 4` registration
/// gate, so no runtime capability check is needed here. Route every group-3
/// material bind through this macro (the same choke-point rule as
/// `bind_deform_group!` above).
///
/// Plugin pipeline layouts always carry a group 2 so that group 3 lands at 3.
/// Where deformers are off (a device short of storage buffers) that group is
/// empty, `bind_deform_group!` binds nothing, and this binds the empty group in
/// its place.
macro_rules! bind_material_group {
    ($pass:expr, $resources:expr, $bg:expr) => {
        if !$resources.deform.enabled {
            $pass.set_bind_group(2, &$resources.deform.empty_bind_group, &[]);
        }
        $pass.set_bind_group(3, $bg, &[]);
    };
}

/// A batch of instances sharing the same mesh and material textures, drawn in one call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InstancedBatch {
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    pub texture_id: Option<crate::resources::TextureId>,
    pub normal_map_id: Option<crate::resources::TextureId>,
    pub ao_map_id: Option<crate::resources::TextureId>,
    pub metallic_roughness_id: Option<crate::resources::TextureId>,
    pub emissive_id: Option<crate::resources::TextureId>,
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
    /// The material plugin whose composed instanced shading this batch draws
    /// with, or `None` for built-in shading. All items in the batch share it
    /// (it is part of the batch key). A plugin batch draws through the plugin's
    /// instanced pipeline plus its group-3 texture bind, outside the GPU-cull /
    /// count-multi-draw run-forming, so the field also flags "skip me" in those
    /// loops.
    pub shading_plugin: Option<crate::scene::material::MaterialPluginId>,
    /// The plugin textures every item in the batch names (part of the batch
    /// key): which group-3 bind group the batch draws with.
    pub plugin_textures: crate::resources::mesh_sidecar::shade::PluginTextures,
    /// The sampler bound at group-1 binding 2 for the whole batch: the
    /// material's selected sampler, or the default key. Part of the batch key.
    pub sampler: crate::scene::material::SamplerKey,
    /// `Some` for a fat batch: one that draws through the per-object mesh
    /// pipelines with the twenty-binding group-1 layout, because its items
    /// need a sidecar the lean instanced layout does not carry (a scalar
    /// attribute, matcap, warp, lightmap, override, deform slot, submesh
    /// range, edge or normal lines). A fat batch is still an instance range
    /// of the shared instance buffer; only its pipeline and bind group differ.
    pub fat: Option<FatBatch>,
    /// World-space centre of the batch's instance bounds. The LDR pass sorts
    /// blended batches back to front by it; on an LDR frame each blended item
    /// is its own batch, so the sort is per item.
    pub sort_centre: [f32; 3],
}

/// Which geometry a fat batch draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum FatDraw {
    /// The whole mesh, indexed.
    Triangles,
    /// One submesh index range: `(first_index, index_count)`.
    Range(u32, u32),
    /// The mesh's edge lines: a wireframe item, or viewport wireframe mode.
    Edges,
    /// The per-face expanded vertex buffer (face, corner and half-edge
    /// attributes), drawn unindexed.
    FaceVertices,
    /// The mesh's normal-line buffer, for `show_normals` items.
    NormalLines,
}

/// The per-object-pipeline facts of a fat batch.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FatBatch {
    pub draw: FatDraw,
    /// Key of this batch's group-1 bind group in
    /// `PerObjectState::material_bind_groups`.
    pub bind_key: u64,
    /// The deform instance every item in the batch shares (part of the key).
    pub deform_instance: Option<u32>,
    /// The batch has an active scalar attribute, which keeps the discarding
    /// pipeline (NaN samples discard).
    pub has_attribute: bool,
    /// Scene index of the item the batch's first record came from, and the
    /// submesh range it draws when it is a range record. Every record in the
    /// batch shares the facts its bind group hashes, so the first stands for
    /// all; the bind group is rebuilt from it each frame.
    pub rep_idx: usize,
    pub rep_range: Option<u32>,
}

/// Bound state the fat-batch draw loops carry between batches so a
/// pipeline, bind group or vertex buffer shared by consecutive batches is
/// set once.
#[derive(Default)]
pub(crate) struct FatBound {
    pipeline: Option<*const crate::gpu::RenderPipeline>,
    group1: Option<*const crate::gpu::BindGroup>,
    deform: Option<*const crate::gpu::BindGroup>,
    geometry: Option<(crate::resources::mesh::mesh_store::MeshId, FatDraw)>,
    /// A batch drew through a stand-in pipeline while its own compiles. A
    /// render bundle must not keep that draw, since nothing re-records it
    /// when the compile finishes.
    pub(crate) stand_in: bool,
}

/// The render-pass methods a mesh draw needs, so one draw helper serves both
/// an immediate `RenderPass` and a `RenderBundleEncoder`. `'a` is the
/// lifetime of the resources an encoder may hold on to (a bundle encoder
/// keeps them until `finish`).
pub(crate) trait MeshDrawEncoder<'a> {
    fn set_pipeline(&mut self, pipeline: &'a crate::gpu::RenderPipeline);
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a crate::gpu::BindGroup,
        offsets: &[u32],
    );
    fn set_vertex_buffer(&mut self, slot: u32, slice: crate::gpu::BufferSlice<'a>);
    fn set_index_buffer(
        &mut self,
        slice: crate::gpu::BufferSlice<'a>,
        format: crate::gpu::IndexFormat,
    );
    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>);
    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    );
}

impl<'a> MeshDrawEncoder<'a> for crate::gpu::RenderPass<'_> {
    fn set_pipeline(&mut self, pipeline: &'a crate::gpu::RenderPipeline) {
        crate::gpu::RenderPass::set_pipeline(self, pipeline);
    }
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a crate::gpu::BindGroup,
        offsets: &[u32],
    ) {
        crate::gpu::RenderPass::set_bind_group(self, index, bind_group, offsets);
    }
    fn set_vertex_buffer(&mut self, slot: u32, slice: crate::gpu::BufferSlice<'a>) {
        crate::gpu::RenderPass::set_vertex_buffer(self, slot, slice);
    }
    fn set_index_buffer(
        &mut self,
        slice: crate::gpu::BufferSlice<'a>,
        format: crate::gpu::IndexFormat,
    ) {
        crate::gpu::RenderPass::set_index_buffer(self, slice, format);
    }
    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        crate::gpu::RenderPass::draw(self, vertices, instances);
    }
    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        crate::gpu::RenderPass::draw_indexed(self, indices, base_vertex, instances);
    }
}

impl<'a> MeshDrawEncoder<'a> for crate::gpu::RenderBundleEncoder<'a> {
    fn set_pipeline(&mut self, pipeline: &'a crate::gpu::RenderPipeline) {
        crate::gpu::RenderBundleEncoder::set_pipeline(self, pipeline);
    }
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a crate::gpu::BindGroup,
        offsets: &[u32],
    ) {
        crate::gpu::RenderBundleEncoder::set_bind_group(self, index, bind_group, offsets);
    }
    fn set_vertex_buffer(&mut self, slot: u32, slice: crate::gpu::BufferSlice<'a>) {
        crate::gpu::RenderBundleEncoder::set_vertex_buffer(self, slot, slice);
    }
    fn set_index_buffer(
        &mut self,
        slice: crate::gpu::BufferSlice<'a>,
        format: crate::gpu::IndexFormat,
    ) {
        crate::gpu::RenderBundleEncoder::set_index_buffer(self, slice, format);
    }
    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        crate::gpu::RenderBundleEncoder::draw(self, vertices, instances);
    }
    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        crate::gpu::RenderBundleEncoder::draw_indexed(self, indices, base_vertex, instances);
    }
}

/// Which pass a fat batch is being drawn into, which picks its pipeline set.
pub(crate) enum FatPass<'a> {
    /// A colour pass: the LDR or HDR scene family. `hdr` selects the early-Z
    /// twin where the batch cannot discard.
    Colour {
        family: &'a dyn crate::resources::scene_pipelines::MeshColourFamily,
        hdr: bool,
        clipping_active: bool,
    },
    /// The weighted-blended OIT pass.
    Oit,
}

/// Draw one fat batch: bind its group-1 bind group, deform group and the
/// geometry its `FatDraw` names, select the per-object pipeline for `pass`,
/// and draw its instance range. Returns false when something it needs (a
/// pipeline still compiling, a missing bind group or buffer) kept it from
/// drawing. `bound` carries the state set by the previous batch.
pub(crate) fn draw_fat_batch<'a, E: MeshDrawEncoder<'a>>(
    resources: &'a crate::resources::DeviceResources,
    fat_bind_groups: &'a crate::resources::fast_hash::FastMap<
        u64,
        crate::renderer::per_object_state::MaterialBindGroup,
    >,
    encoder: &mut E,
    batch: &InstancedBatch,
    pass: &FatPass<'a>,
    bound: &mut FatBound,
) -> bool {
    use crate::renderer::pipeline_key::PipelineKey;
    let Some(fat) = batch.fat.as_ref() else {
        return false;
    };
    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
        return false;
    };
    let Some(group1) = fat_bind_groups.get(&fat.bind_key).map(|m| &m.bind_group) else {
        return false;
    };
    let lines = matches!(fat.draw, FatDraw::Edges | FatDraw::NormalLines);
    let two_sided = PipelineKey::two_sided(batch.two_sided);
    let plug = resources.material_plugin_draw(batch.shading_plugin, &batch.plugin_textures);
    let (pipeline, mat_bg): (
        Option<&crate::gpu::RenderPipeline>,
        Option<&crate::gpu::BindGroup>,
    ) = match pass {
        FatPass::Colour {
            family,
            hdr,
            clipping_active,
        } => {
            if lines {
                (family.wireframe(), None)
            } else if let Some((pp, mat_bg)) = plug {
                let pl = if batch.is_transparent {
                    if *hdr {
                        pp.hdr_transparent()
                    } else {
                        pp.ldr_transparent()
                    }
                } else if *hdr {
                    pp.hdr_opaque(two_sided)
                } else {
                    pp.ldr_opaque(two_sided)
                };
                (pl, Some(mat_bg))
            } else if batch.shading_plugin.is_some() {
                // The plugin's per-object set is not composed yet.
                (None, None)
            } else if batch.is_transparent {
                (family.transparent(), None)
            } else {
                // Early-Z: a batch that can never discard draws through
                // the discard-free twin.
                let key = PipelineKey {
                    two_sided: batch.two_sided,
                    no_discard_eligible: *hdr
                        && !clipping_active
                        && !resources.force_po_discard
                        && !batch.has_alpha_mask
                        && !fat.has_attribute
                        && !matches!(fat.draw, FatDraw::Range(..))
                        && !resources
                            .deform
                            .may_discard(batch.mesh_id, fat.deform_instance),
                    ..PipelineKey::default()
                };
                // A mesh with no sidecar record takes the pipeline with the
                // sidecar reads compiled out, or the full one while that
                // compiles: both draw the same image.
                let lean = PipelineKey {
                    no_sidecar: mesh.sidecar.is_none(),
                    ..key
                };
                let pl = if lean.no_sidecar && *hdr {
                    family.opaque(lean).or_else(|| {
                        bound.stand_in = true;
                        family.opaque(key)
                    })
                } else {
                    family.opaque(key)
                };
                (pl, None)
            }
        }
        FatPass::Oit => {
            if lines {
                (None, None)
            } else if let Some((pp, mat_bg)) = plug {
                (pp.oit(two_sided), Some(mat_bg))
            } else if batch.shading_plugin.is_some() {
                (None, None)
            } else {
                (resources.oit.per_object(two_sided), None)
            }
        }
    };
    let Some(pipeline) = pipeline else {
        return false;
    };
    if bound.pipeline != Some(pipeline as *const _) {
        encoder.set_pipeline(pipeline);
        bound.pipeline = Some(pipeline as *const _);
    }
    if bound.group1 != Some(group1 as *const _) {
        encoder.set_bind_group(1, group1, &[]);
        bound.group1 = Some(group1 as *const _);
    }
    if let Some(mat_bg) = mat_bg {
        bind_material_group!(encoder, resources, mat_bg);
    }
    let deform_bg = if matches!(fat.draw, FatDraw::NormalLines) {
        &resources.deform.dummy_bind_group
    } else {
        resources
            .deform
            .instance_bind_group_for(batch.mesh_id, fat.deform_instance)
    };
    if bound.deform != Some(deform_bg as *const _) {
        bind_deform_group!(encoder, resources, deform_bg);
        bound.deform = Some(deform_bg as *const _);
    }
    let instances = batch.instance_offset..batch.instance_offset + batch.instance_count;
    let geometry = (batch.mesh_id, fat.draw);
    let rebind = bound.geometry != Some(geometry);
    match fat.draw {
        FatDraw::Triangles | FatDraw::Range(..) => {
            if rebind {
                encoder.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                encoder.set_index_buffer(
                    resources.geometry.index_slice(mesh.index_span),
                    crate::gpu::IndexFormat::Uint32,
                );
            }
            let range = match fat.draw {
                FatDraw::Range(first, count) => first..first + count,
                _ => 0..mesh.index_count,
            };
            encoder.draw_indexed(range, 0, instances);
        }
        FatDraw::Edges => {
            let Some(edge_buf) = &mesh.edge_index_buffer else {
                return false;
            };
            if rebind {
                encoder.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                encoder.set_index_buffer(edge_buf.slice(..), crate::gpu::IndexFormat::Uint32);
            }
            encoder.draw_indexed(0..mesh.edge_index_count, 0, instances);
        }
        FatDraw::FaceVertices => {
            let Some(fvb) = &mesh.face_vertex_buffer else {
                return false;
            };
            if rebind {
                encoder.set_vertex_buffer(0, fvb.slice(..));
            }
            encoder.draw(0..mesh.index_count, instances);
        }
        FatDraw::NormalLines => {
            let Some(nl_buf) = &mesh.normal_line_buffer else {
                return false;
            };
            if mesh.normal_line_count == 0 {
                return false;
            }
            if rebind {
                encoder.set_vertex_buffer(0, nl_buf.slice(..));
            }
            encoder.draw(0..mesh.normal_line_count, instances);
        }
    }
    bound.geometry = Some(geometry);
    true
}

/// Group-1 bind-group cache key for an instanced batch: the five material
/// texture ids (`u64::MAX` for a fallback slot), the uv1 chunk discriminator
/// (`uv1_chunk_key`) and the sampler code (`sampler_code`).
pub(crate) type BatchBindKey = (u64, u64, u64, u64, u64, u32, u32);

impl InstancedBatch {
    /// This batch's [`BatchBindKey`] for the uv1 chunk discriminator `uv1_key`.
    pub(crate) fn bind_key(&self, uv1_key: u32) -> BatchBindKey {
        let raw = |id: Option<crate::resources::TextureId>| id.map(|t| t.raw()).unwrap_or(u64::MAX);
        (
            raw(self.texture_id),
            raw(self.normal_map_id),
            raw(self.ao_map_id),
            raw(self.metallic_roughness_id),
            raw(self.emissive_id),
            uv1_key,
            crate::resources::material::fallbacks::sampler_code(self.sampler),
        )
    }

    /// The alpha-cutout shadow bind-group cache key for this batch in `cascade`:
    /// the albedo, normal and AO texture ids and the sampler code.
    pub(crate) fn cutout_key(&self, cascade: usize) -> (usize, u64, u64, u64, u32) {
        let raw = |id: Option<crate::resources::TextureId>| id.map(|t| t.raw()).unwrap_or(u64::MAX);
        (
            cascade,
            raw(self.texture_id),
            raw(self.normal_map_id),
            raw(self.ao_map_id),
            crate::resources::material::fallbacks::sampler_code(self.sampler),
        )
    }
}

mod clip;
pub mod debug;
mod frame;
pub(crate) mod items;
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
    /// World-space scene content (surfaces, point clouds, fields, etc.).
    pub scene: SceneFrame,
    /// Viewport presentation settings (background, grid, axes indicator).
    pub viewport: ViewportFrame,
    /// Interaction and selection visualization (gizmo, outline, x-ray).
    pub interaction: InteractionFrame,
    /// Global rendering effects (lighting, clipping, post-process).
    pub effects: EffectsFrame,
    /// Screen-space overlays rendered after post-processing (shapes, labels,
    /// glyph runs, polylines).
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
    pub fn with_background(mut self, colour: impl Into<crate::Colour>) -> Self {
        self.viewport.background_colour = Some(colour.into());
        self
    }

    /// Override the per-frame lighting configuration.
    pub fn with_lighting(mut self, lighting: LightingSettings) -> Self {
        self.effects.lighting = lighting;
        self
    }

    /// Apply a [`LightingPosture`], setting the lighting and exposure together as
    /// a matched pair. See [`EffectsFrame::with_posture`].
    pub fn with_posture(mut self, posture: LightingPosture) -> Self {
        self.effects = core::mem::take(&mut self.effects).with_posture(posture);
        self
    }

    /// Override the post-processing settings.
    pub fn with_post_process(mut self, post: PostProcessSettings) -> Self {
        self.effects.post_process = post;
        self
    }

    /// Override the display transform (pipeline mode, exposure, tone-map operator).
    pub fn with_display(mut self, display: DisplaySettings) -> Self {
        self.effects.display = display;
        self
    }

    /// Override the exposure settings (manual EV, physical camera, or auto).
    pub fn with_exposure(mut self, exposure: ExposureSettings) -> Self {
        self.effects.display.exposure = exposure;
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
    ($resources:expr, $render_pass:expr, $frame:expr, $batches:expr, $camera_bg:expr, $grid_bg:expr, $slot:expr, $fat_bgs:expr, $scene_items:expr, $po_bundle:expr) => {{
        let resources = $resources;
        let render_pass = $render_pass;
        let frame = $frame;
        let po_bundle: Option<&crate::renderer::per_object_state::PerObjectBundle> = $po_bundle;
        let _vp_slot: Option<&ViewportSlot> = $slot;
        let batches: &[InstancedBatch] = $batches;
        let camera_bg: &crate::gpu::BindGroup = $camera_bg;
        let grid_bg: &crate::gpu::BindGroup = $grid_bg;
        // Group-1 bind groups of the fat batches, keyed by `FatBatch::bind_key`.
        let fat_bind_groups: &crate::resources::fast_hash::FastMap<
            u64,
            crate::renderer::per_object_state::MaterialBindGroup,
        > = $fat_bgs;

        // The LOD-resolved surface items from prepare: each item's mesh is the
        // level chosen for its on-screen size, and culled items are hidden.
        // Matches `frame.scene.surfaces` in order and length, so the per-object
        // bind groups (indexed by position) line up unchanged.
        let _scene_items: &[SceneRenderItem] = $scene_items;

        render_pass.set_bind_group(0, camera_bg, &[]);

        // Grid pass : full-screen analytical shader drawn first so scene geometry
        // occludes it. No vertex buffer; depth is written via @builtin(frag_depth).
        // Camera bind group is restored immediately after for subsequent passes.
        if let (true, Some(pipeline)) = (
            frame.viewport.show_grid,
            resources.guide_pipeline(crate::resources::overlay::guides::GUIDE_GRID),
        ) {
            render_pass.set_pipeline(pipeline);
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
            render_pass.set_pipeline(resources.ground.pipeline());
            render_pass.set_bind_group(0, &resources.ground.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
            render_pass.set_bind_group(0, camera_bg, &[]);
        }

        if !batches.is_empty() {
            // When prepare cached a render bundle for this batch list and
            // this pass uses the camera bind group it was recorded with,
            // replay it instead of re-encoding the opaque draws.
            let bundle_hit: Option<&crate::renderer::per_object_state::PerObjectBundle> =
                po_bundle.filter(|pb| !pb.hdr && pb.camera_bg == *camera_bg);
            if let Some(pb) = bundle_hit {
                render_pass.execute_bundles(std::iter::once(&pb.bundle));
                // Bundle execution resets all render-pass state; restore
                // the camera bind group for the draws and passes below.
                render_pass.set_bind_group(0, camera_bg, &[]);
            }
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

            // Draw opaque instanced batches (the bundle replay covers them).
            if bundle_hit.is_none() && !opaque_batches.is_empty() && !frame.viewport.wireframe_mode
            {
                if resources.instancing.ldr.is_some() {
                    // Early-Z fast path: discard-free pipeline twin for
                    // opaque batches when no clip object or alpha-mask
                    // instance can discard this frame.
                    let clipping_active = frame
                        .effects
                        .clip
                        .objects
                        .iter()
                        .any(|o| o.enabled && o.clip_geometry);
                    bind_deform_group!(render_pass, resources, &resources.deform.dummy_bind_group);
                    // Batches are sorted with two_sided in the key, so one- and
                    // two-sided runs are contiguous; switch pipeline on change.
                    // Geometry is in the shared slab, so the chunk buffers bind
                    // once and each draw carries the mesh's base_vertex /
                    // first_index instead of re-binding a sub-slice per batch.
                    let mut cur_pipe: Option<(bool, bool)> = None;
                    let mut cur_chunks: Option<(u32, u32)> = None;
                    for batch in &opaque_batches {
                        // Plugin batches draw in the dedicated plugin
                        // sub-loop below (their own pipeline + group-3
                        // params bind); the built-in pipeline would shade
                        // them without the plugin hook.
                        if batch.shading_plugin.is_some() || batch.fat.is_some() {
                            continue;
                        }
                        let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                            continue;
                        };
                        let mat_key =
                            batch.bind_key(resources.uv1_chunk_key(mesh.vertex_span.chunk));
                        // Combined (instance storage + texture) bind group, primed in prepare().
                        let Some(inst_tex_bg) = resources.instanced_colour_bind_group(mat_key)
                        else {
                            continue;
                        };
                        let no_discard = !clipping_active && !batch.has_alpha_mask;
                        if cur_pipe != Some((batch.two_sided, no_discard)) {
                            // A pipeline still compiling skips the batch this frame.
                            let Some(pipeline) = resources.instancing.ldr_opaque(PipelineKey {
                                two_sided: batch.two_sided,
                                no_discard_eligible: no_discard,
                                ..PipelineKey::default()
                            }) else {
                                cur_pipe = None;
                                continue;
                            };
                            render_pass.set_pipeline(pipeline);
                            cur_pipe = Some((batch.two_sided, no_discard));
                        }
                        render_pass.set_bind_group(1, inst_tex_bg, &[]);
                        let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                        if cur_chunks != Some(chunks) {
                            render_pass.set_vertex_buffer(
                                0,
                                resources.geometry.vertex_chunk_slice(chunks.0),
                            );
                            render_pass.set_index_buffer(
                                resources.geometry.index_chunk_slice(chunks.1),
                                crate::gpu::IndexFormat::Uint32,
                            );
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

            // Material-plugin opaque instanced batches (LDR): one call
            // each through the plugin's composed instanced pipeline plus
            // its group-3 params bind. The built-in opaque loop above
            // skips these. Mirrors the HDR plugin sub-loop; drawn before
            // the transparent batches so depth ordering matches built-in.
            if !frame.viewport.wireframe_mode
                && opaque_batches.iter().any(|b| b.shading_plugin.is_some())
            {
                bind_deform_group!(render_pass, resources, &resources.deform.dummy_bind_group);
                let mut cur_chunks: Option<(u32, u32)> = None;
                let mut cur_pipe: Option<*const crate::gpu::RenderPipeline> = None;
                for batch in &opaque_batches {
                    if batch.shading_plugin.is_none() || batch.fat.is_some() {
                        continue;
                    }
                    let Some((plug_pipes, mat_bg)) = resources.material_plugin_instanced_draw(
                        batch.shading_plugin,
                        &batch.plugin_textures,
                    ) else {
                        continue;
                    };
                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                        continue;
                    };
                    let mat_key = batch.bind_key(resources.uv1_chunk_key(mesh.vertex_span.chunk));
                    let Some(inst_tex_bg) = resources.instanced_colour_bind_group(mat_key) else {
                        continue;
                    };
                    let Some(pipeline) = plug_pipes.ldr_opaque(
                        crate::renderer::pipeline_key::PipelineKey::two_sided(batch.two_sided),
                    ) else {
                        continue;
                    };
                    if cur_pipe != Some(pipeline as *const _) {
                        render_pass.set_pipeline(pipeline);
                        cur_pipe = Some(pipeline as *const _);
                    }
                    render_pass.set_bind_group(1, inst_tex_bg, &[]);
                    bind_material_group!(render_pass, resources, mat_bg);
                    let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                    if cur_chunks != Some(chunks) {
                        render_pass
                            .set_vertex_buffer(0, resources.geometry.vertex_chunk_slice(chunks.0));
                        render_pass.set_index_buffer(
                            resources.geometry.index_chunk_slice(chunks.1),
                            crate::gpu::IndexFormat::Uint32,
                        );
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

            // Fat batches: the per-object pipelines with the twenty-binding
            // group-1 layout, drawn by instance range. Opaque first (the
            // bundle covers these too), then normal lines.
            let ldr_family = resources.scene.ldr_family();
            let fat_pass = crate::renderer::types::FatPass::Colour {
                family: &ldr_family,
                hdr: false,
                clipping_active: false,
            };
            let mut fat_bound = crate::renderer::types::FatBound::default();
            if bundle_hit.is_none() {
                for batch in batches {
                    let Some(fat) = batch.fat.as_ref() else {
                        continue;
                    };
                    if batch.is_transparent
                        || matches!(fat.draw, crate::renderer::types::FatDraw::NormalLines)
                    {
                        continue;
                    }
                    crate::renderer::types::draw_fat_batch(
                        resources,
                        fat_bind_groups,
                        render_pass,
                        batch,
                        &fat_pass,
                        &mut fat_bound,
                    );
                }
            }
            for batch in batches {
                if !batch
                    .fat
                    .as_ref()
                    .is_some_and(|f| matches!(f.draw, crate::renderer::types::FatDraw::NormalLines))
                {
                    continue;
                }
                crate::renderer::types::draw_fat_batch(
                    resources,
                    fat_bind_groups,
                    render_pass,
                    batch,
                    &fat_pass,
                    &mut fat_bound,
                );
            }

            // Blended batches, after every opaque draw. There is no OIT here,
            // so they draw back to front by view depth. Prepare gives each
            // blended item its own batch on this path, so the order is per
            // item. Lean, plugin and fat batches share the one sorted loop,
            // and each kind rebinds its state after a draw of another kind.
            if !transparent_batches.is_empty() {
                let view = frame.camera.render_camera.view;
                let depth =
                    |b: &InstancedBatch| -view.transform_point3(glam::Vec3::from(b.sort_centre)).z;
                transparent_batches.sort_by(|a, b| depth(b).total_cmp(&depth(a)));
                let mut cur_pipe: Option<*const crate::gpu::RenderPipeline> = None;
                let mut cur_chunks: Option<(u32, u32)> = None;
                let mut lean_bound = false;
                for batch in &transparent_batches {
                    if batch.fat.is_some() {
                        if crate::renderer::types::draw_fat_batch(
                            resources,
                            fat_bind_groups,
                            render_pass,
                            batch,
                            &fat_pass,
                            &mut fat_bound,
                        ) {
                            lean_bound = false;
                        }
                        continue;
                    }
                    if frame.viewport.wireframe_mode {
                        continue;
                    }
                    let (pipeline, mat_bg) = if batch.shading_plugin.is_some() {
                        let Some((plug_pipes, mat_bg)) = resources.material_plugin_instanced_draw(
                            batch.shading_plugin,
                            &batch.plugin_textures,
                        ) else {
                            continue;
                        };
                        let Some(pipeline) = plug_pipes.ldr_transparent() else {
                            continue;
                        };
                        (pipeline, Some(mat_bg))
                    } else {
                        let Some(pipeline) = resources.instancing.ldr_transparent() else {
                            continue;
                        };
                        (pipeline, None)
                    };
                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                        continue;
                    };
                    let mat_key = batch.bind_key(resources.uv1_chunk_key(mesh.vertex_span.chunk));
                    let Some(inst_tex_bg) = resources.instanced_colour_bind_group(mat_key) else {
                        continue;
                    };
                    if !lean_bound {
                        bind_deform_group!(
                            render_pass,
                            resources,
                            &resources.deform.dummy_bind_group
                        );
                        cur_pipe = None;
                        cur_chunks = None;
                        fat_bound = crate::renderer::types::FatBound::default();
                        lean_bound = true;
                    }
                    if cur_pipe != Some(pipeline as *const _) {
                        render_pass.set_pipeline(pipeline);
                        cur_pipe = Some(pipeline as *const _);
                    }
                    render_pass.set_bind_group(1, inst_tex_bg, &[]);
                    if let Some(mat_bg) = mat_bg {
                        bind_material_group!(render_pass, resources, mat_bg);
                    }
                    let chunks = (mesh.vertex_span.chunk, mesh.index_span.chunk);
                    if cur_chunks != Some(chunks) {
                        render_pass
                            .set_vertex_buffer(0, resources.geometry.vertex_chunk_slice(chunks.0));
                        render_pass.set_index_buffer(
                            resources.geometry.index_chunk_slice(chunks.1),
                            crate::gpu::IndexFormat::Uint32,
                        );
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

        // The transform gizmo now draws in the 2D overlay pass; see
        // `gizmo_overlay::build_gizmo_overlays` and `prepare_overlay_labels`.

        // Constraint guide line pass.
        if let Some(slot) = _vp_slot {
            if let (false, Some(pipeline)) = (
                slot.constraint_line_buffers.is_empty(),
                resources.guide_pipeline(crate::resources::overlay::guides::GUIDE_LINES),
            ) {
                render_pass.set_pipeline(pipeline);
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
            if let (false, Some(pipeline)) = (
                slot.cap_buffers.is_empty(),
                resources.guide_pipeline(crate::resources::overlay::guides::GUIDE_TRIANGLES),
            ) {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.cap_buffers {
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
                render_pass.set_pipeline(resources.outline.xray_pipeline());
                render_pass.set_bind_group(0, camera_bg, &[]);
                // The x-ray pipeline shares the outline layout, so group 2 is
                // part of it whenever deformers are enabled. X-ray draws the
                // undeformed mesh, so the dummy group is what it wants.
                bind_deform_group!(render_pass, resources, &resources.deform.dummy_bind_group);
                for (mesh_id, xray) in &slot.xray_object_buffers {
                    let Some(mesh) = resources.mesh_store.get(*mesh_id) else {
                        continue;
                    };
                    render_pass.set_bind_group(1, &xray.bind_group, &[]);
                    render_pass
                        .set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                    render_pass.set_index_buffer(
                        resources.geometry.index_slice(mesh.index_span),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        // The axes orientation indicator now draws as screen-space overlay shapes
        // in the shared overlay pass (see `axes_overlay_items`), not here.
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
            if !slot.selection_outlines.outline_object_buffers.is_empty()
                || slot.selection_outlines.plugin_outline_present
            {
                let composite_bg = slot.hdr.as_ref().map(|h| &h.outline_composite_bind_group);
                // Skipped while the pipeline is on a worker.
                let pipeline = resources.outline.composite_ldr(resources.sample_count);
                if let (Some(pipeline), Some(bg)) = (pipeline, composite_bg) {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, bg, &[]);
                    render_pass.draw(0..3, 0..1);
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
