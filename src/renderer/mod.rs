//! `ViewportRenderer` : the main entry point for the viewport library.
//!
//! Wraps [`ViewportGpuResources`] and provides `prepare()` / `paint()` methods
//! that take raw `wgpu` types. GUI framework adapters (e.g. the egui
//! `CallbackTrait` impl in the application crate) delegate to these methods.

#[macro_use]
mod types;
mod picking;
mod prepare;
mod render;
pub mod shader_hashes;
mod shadows;
pub mod stats;

pub(crate) use self::types::ClipPlane;
pub use self::types::{
    CameraFrame, CameraFrustumItem, ClipObject, ClipShape, ComputeFilterItem, ComputeFilterKind,
    EffectsFrame, EnvironmentMap, FilterMode, FrameData, GlyphItem, GlyphType, GroundPlane,
    GroundPlaneMode, ImageAnchor, InteractionFrame, LightKind, LightSource, LightingSettings,
    PickId, PointCloudItem, PointRenderMode, PolylineItem, PostProcessSettings, RenderCamera,
    ScalarBar, ScalarBarAnchor, ScalarBarOrientation, SceneEffects, SceneFrame, SceneRenderItem,
    ScreenImageItem, ShadowFilter, StreamtubeItem, SurfaceSubmission, ToneMapping, ViewportEffects,
    ViewportFrame, VolumeItem,
};

/// An opaque handle to a per-viewport GPU state slot.
///
/// Obtained from [`ViewportRenderer::create_viewport`] and passed to
/// [`ViewportRenderer::prepare_viewport`], [`ViewportRenderer::paint_viewport`],
/// and [`ViewportRenderer::render_viewport`].
///
/// The inner `usize` is the slot index and doubles as the value for
/// [`CameraFrame::with_viewport_index`].  Single-viewport applications that use
/// the legacy [`ViewportRenderer::prepare`] / [`ViewportRenderer::paint`] API do
/// not need this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewportId(pub usize);

use self::shadows::{compute_cascade_matrix, compute_cascade_splits};
use self::types::{INSTANCING_THRESHOLD, InstancedBatch};
use crate::resources::{
    CameraUniform, ClipPlanesUniform, ClipVolumeUniform, GridUniform, InstanceData, LightsUniform,
    ObjectUniform, OutlineEdgeUniform, OutlineObjectBuffers, OutlineUniform, PickInstance,
    ShadowAtlasUniform,
    SingleLightUniform, ViewportGpuResources,
};

/// Per-viewport GPU state: uniform buffers and bind groups that differ per viewport.
///
/// Each viewport slot owns its own camera, clip planes, clip volume, shadow info,
/// and grid buffers, plus the bind groups that reference them. Scene-global
/// resources (lights, shadow atlas texture, IBL) are shared via the bind group
/// pointing to buffers on `ViewportGpuResources`.
pub(crate) struct ViewportSlot {
    pub camera_buf: wgpu::Buffer,
    pub clip_planes_buf: wgpu::Buffer,
    pub clip_volume_buf: wgpu::Buffer,
    pub shadow_info_buf: wgpu::Buffer,
    pub grid_buf: wgpu::Buffer,
    /// Camera bind group (group 0) referencing this slot's per-viewport buffers
    /// plus shared scene-global resources.
    pub camera_bind_group: wgpu::BindGroup,
    /// Grid bind group (group 0 for grid pipeline) referencing this slot's grid buffer.
    pub grid_bind_group: wgpu::BindGroup,
    /// Per-viewport HDR post-process render targets.
    ///
    /// Created lazily on first HDR render call and resized when viewport dimensions change.
    pub hdr: Option<crate::resources::ViewportHdrState>,

    // --- Per-viewport interaction state (Phase 4) ---
    /// Per-frame outline buffers for selected objects, rebuilt in prepare().
    pub outline_object_buffers: Vec<OutlineObjectBuffers>,
    /// Per-frame x-ray buffers for selected objects, rebuilt in prepare().
    pub xray_object_buffers: Vec<(crate::resources::mesh_store::MeshId, wgpu::Buffer, wgpu::BindGroup)>,
    /// Per-frame constraint guide line buffers, rebuilt in prepare().
    pub constraint_line_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Per-frame cap geometry buffers (section view cross-section fill), rebuilt in prepare().
    pub cap_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Per-frame clip plane fill overlay buffers, rebuilt in prepare().
    pub clip_plane_fill_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Per-frame clip plane line overlay buffers, rebuilt in prepare().
    pub clip_plane_line_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Vertex buffer for axes indicator geometry (rebuilt each frame).
    pub axes_vertex_buffer: wgpu::Buffer,
    /// Number of vertices in the axes indicator buffer.
    pub axes_vertex_count: u32,
    /// Gizmo model-matrix uniform buffer.
    pub gizmo_uniform_buf: wgpu::Buffer,
    /// Gizmo bind group (group 1: model matrix uniform).
    pub gizmo_bind_group: wgpu::BindGroup,
    /// Gizmo vertex buffer.
    pub gizmo_vertex_buffer: wgpu::Buffer,
    /// Gizmo index buffer.
    pub gizmo_index_buffer: wgpu::Buffer,
    /// Number of indices in the current gizmo mesh.
    pub gizmo_index_count: u32,
}

/// High-level renderer wrapping all GPU resources and providing framework-agnostic
/// `prepare()` and `paint()` methods.
pub struct ViewportRenderer {
    resources: ViewportGpuResources,
    /// Instanced batches prepared for the current frame. Empty when using per-object path.
    instanced_batches: Vec<InstancedBatch>,
    /// Whether the current frame uses the instanced draw path.
    use_instancing: bool,
    /// Performance counters from the last frame.
    last_stats: crate::renderer::stats::FrameStats,
    /// Last scene generation seen during prepare(). u64::MAX forces rebuild on first frame.
    last_scene_generation: u64,
    /// Last selection generation seen during prepare(). u64::MAX forces rebuild on first frame.
    last_selection_generation: u64,
    /// Last scene_items count seen during prepare(). usize::MAX forces rebuild on first frame.
    /// Included in cache key so that frustum-culling changes (different visible set, different
    /// count) correctly invalidate the instance buffer even when scene_generation is stable.
    last_scene_items_count: usize,
    /// Cached instance data from last rebuild (mirrors the GPU buffer contents).
    cached_instance_data: Vec<InstanceData>,
    /// Cached instanced batch descriptors from last rebuild.
    cached_instanced_batches: Vec<InstancedBatch>,
    /// Per-frame point cloud GPU data, rebuilt in prepare(), consumed in paint().
    point_cloud_gpu_data: Vec<crate::resources::PointCloudGpuData>,
    /// Per-frame glyph GPU data, rebuilt in prepare(), consumed in paint().
    glyph_gpu_data: Vec<crate::resources::GlyphGpuData>,
    /// Per-frame polyline GPU data, rebuilt in prepare(), consumed in paint().
    polyline_gpu_data: Vec<crate::resources::PolylineGpuData>,
    /// Per-frame volume GPU data, rebuilt in prepare(), consumed in paint().
    volume_gpu_data: Vec<crate::resources::VolumeGpuData>,
    /// Per-frame streamtube GPU data, rebuilt in prepare(), consumed in paint().
    streamtube_gpu_data: Vec<crate::resources::StreamtubeGpuData>,
    /// Per-frame GPU implicit surface data, rebuilt in prepare(), consumed in paint() (Phase 16).
    implicit_gpu_data: Vec<crate::resources::implicit::ImplicitGpuItem>,
    /// Per-frame screen-image GPU data, rebuilt in prepare(), consumed in paint() (Phase 10B).
    screen_image_gpu_data: Vec<crate::resources::ScreenImageGpuData>,
    /// Per-viewport GPU state slots.
    ///
    /// Indexed by `FrameData::camera.viewport_index`. Each slot owns independent
    /// uniform buffers and bind groups for camera, clip planes, clip volume,
    /// shadow info, and grid. Slots are grown lazily in `prepare` via
    /// `ensure_viewport_slot`. There are at most 4 in the current UI.
    viewport_slots: Vec<ViewportSlot>,
    /// Phase G : GPU compute filter results from the last `prepare()` call.
    ///
    /// Each entry contains a compacted index buffer + count for one filtered mesh.
    /// Consumed during `paint()` to override the mesh's default index buffer.
    /// Cleared and rebuilt each frame.
    compute_filter_results: Vec<crate::resources::ComputeFilterResult>,
    /// Cascade-0 light-space view-projection matrix from the last shadow prepare.
    /// Cached here so `prepare_viewport_internal` can copy it into the ground plane uniform.
    last_cascade0_shadow_mat: glam::Mat4,
}

impl ViewportRenderer {
    /// Create a new renderer with default settings (no MSAA).
    /// Call once at application startup.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        Self::with_sample_count(device, target_format, 1)
    }

    /// Create a new renderer with the specified MSAA sample count (1, 2, or 4).
    ///
    /// When using MSAA (sample_count > 1), the caller must create multisampled
    /// color and depth textures and use them as render pass attachments with the
    /// final surface texture as the resolve target.
    pub fn with_sample_count(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        Self {
            resources: ViewportGpuResources::new(device, target_format, sample_count),
            instanced_batches: Vec::new(),
            use_instancing: false,
            last_stats: crate::renderer::stats::FrameStats::default(),
            last_scene_generation: u64::MAX,
            last_selection_generation: u64::MAX,
            last_scene_items_count: usize::MAX,
            cached_instance_data: Vec::new(),
            cached_instanced_batches: Vec::new(),
            point_cloud_gpu_data: Vec::new(),
            glyph_gpu_data: Vec::new(),
            polyline_gpu_data: Vec::new(),
            volume_gpu_data: Vec::new(),
            streamtube_gpu_data: Vec::new(),
            implicit_gpu_data: Vec::new(),
            screen_image_gpu_data: Vec::new(),
            viewport_slots: Vec::new(),
            compute_filter_results: Vec::new(),
            last_cascade0_shadow_mat: glam::Mat4::IDENTITY,
        }
    }

    /// Access the underlying GPU resources (e.g. for mesh uploads).
    pub fn resources(&self) -> &ViewportGpuResources {
        &self.resources
    }

    /// Performance counters from the last completed frame.
    pub fn last_frame_stats(&self) -> crate::renderer::stats::FrameStats {
        self.last_stats
    }

    /// Mutable access to the underlying GPU resources (e.g. for mesh uploads).
    pub fn resources_mut(&mut self) -> &mut ViewportGpuResources {
        &mut self.resources
    }

    /// Upload an equirectangular HDR environment map and precompute IBL textures.
    ///
    /// `pixels` is row-major RGBA f32 data (4 floats per texel), `width`×`height`.
    /// This rebuilds camera bind groups so shaders immediately see the new textures.
    pub fn upload_environment_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pixels: &[f32],
        width: u32,
        height: u32,
    ) {
        crate::resources::environment::upload_environment_map(
            &mut self.resources,
            device,
            queue,
            pixels,
            width,
            height,
        );
        self.rebuild_camera_bind_groups(device);
    }

    /// Rebuild the primary + per-viewport camera bind groups.
    ///
    /// Call after IBL textures are uploaded so shaders see the new environment.
    fn rebuild_camera_bind_groups(&mut self, device: &wgpu::Device) {
        self.resources.camera_bind_group = self.resources.create_camera_bind_group(
            device,
            &self.resources.camera_uniform_buf,
            &self.resources.clip_planes_uniform_buf,
            &self.resources.shadow_info_buf,
            &self.resources.clip_volume_uniform_buf,
            "camera_bind_group",
        );

        for slot in &mut self.viewport_slots {
            slot.camera_bind_group = self.resources.create_camera_bind_group(
                device,
                &slot.camera_buf,
                &slot.clip_planes_buf,
                &slot.shadow_info_buf,
                &slot.clip_volume_buf,
                "per_viewport_camera_bg",
            );
        }
    }

    /// Ensure a per-viewport slot exists for `viewport_index`.
    ///
    /// Creates a full `ViewportSlot` with independent uniform buffers for camera,
    /// clip planes, clip volume, shadow info, and grid. The camera bind group
    /// references this slot's per-viewport buffers plus shared scene-global
    /// resources. Slots are created lazily and never destroyed.
    fn ensure_viewport_slot(&mut self, device: &wgpu::Device, viewport_index: usize) {
        while self.viewport_slots.len() <= viewport_index {
            let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_camera_buf"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let clip_planes_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_clip_planes_buf"),
                size: std::mem::size_of::<ClipPlanesUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let clip_volume_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_clip_volume_buf"),
                size: std::mem::size_of::<ClipVolumeUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let shadow_info_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_shadow_info_buf"),
                size: std::mem::size_of::<ShadowAtlasUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_grid_buf"),
                size: std::mem::size_of::<GridUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let camera_bind_group = self.resources.create_camera_bind_group(
                device,
                &camera_buf,
                &clip_planes_buf,
                &shadow_info_buf,
                &clip_volume_buf,
                "per_viewport_camera_bg",
            );

            let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vp_grid_bind_group"),
                layout: &self.resources.grid_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_buf.as_entire_binding(),
                }],
            });

            // Per-viewport gizmo buffers (initial mesh: Translate, no hover, identity orientation).
            let (gizmo_verts, gizmo_indices) = crate::interaction::gizmo::build_gizmo_mesh(
                crate::interaction::gizmo::GizmoMode::Translate,
                crate::interaction::gizmo::GizmoAxis::None,
                glam::Quat::IDENTITY,
            );
            let gizmo_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_gizmo_vertex_buf"),
                size: (std::mem::size_of::<crate::resources::Vertex>() * gizmo_verts.len().max(1))
                    as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            gizmo_vertex_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&gizmo_verts));
            gizmo_vertex_buffer.unmap();
            let gizmo_index_count = gizmo_indices.len() as u32;
            let gizmo_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_gizmo_index_buf"),
                size: (std::mem::size_of::<u32>() * gizmo_indices.len().max(1)) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            gizmo_index_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&gizmo_indices));
            gizmo_index_buffer.unmap();
            let gizmo_uniform = crate::interaction::gizmo::GizmoUniform {
                model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            };
            let gizmo_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_gizmo_uniform_buf"),
                size: std::mem::size_of::<crate::interaction::gizmo::GizmoUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            gizmo_uniform_buf
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&[gizmo_uniform]));
            gizmo_uniform_buf.unmap();
            let gizmo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vp_gizmo_bind_group"),
                layout: &self.resources.gizmo_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gizmo_uniform_buf.as_entire_binding(),
                }],
            });

            // Per-viewport axes vertex buffer (2048 vertices = enough for all axes geometry).
            let axes_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_axes_vertex_buf"),
                size: (std::mem::size_of::<crate::widgets::axes_indicator::AxesVertex>() * 2048)
                    as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.viewport_slots.push(ViewportSlot {
                camera_buf,
                clip_planes_buf,
                clip_volume_buf,
                shadow_info_buf,
                grid_buf,
                camera_bind_group,
                grid_bind_group,
                hdr: None,
                outline_object_buffers: Vec::new(),
                xray_object_buffers: Vec::new(),
                constraint_line_buffers: Vec::new(),
                cap_buffers: Vec::new(),
                clip_plane_fill_buffers: Vec::new(),
                clip_plane_line_buffers: Vec::new(),
                axes_vertex_buffer,
                axes_vertex_count: 0,
                gizmo_uniform_buf,
                gizmo_bind_group,
                gizmo_vertex_buffer,
                gizmo_index_buffer,
                gizmo_index_count,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Multi-viewport public API (Phase 5)
    // -----------------------------------------------------------------------

    /// Create a new viewport slot and return its handle.
    ///
    /// The returned [`ViewportId`] is stable for the lifetime of the renderer.
    /// Pass it to [`prepare_viewport`](Self::prepare_viewport),
    /// [`paint_viewport`](Self::paint_viewport), and
    /// [`render_viewport`](Self::render_viewport) each frame.
    ///
    /// Also set `CameraFrame::viewport_index` to `id.0` when building the
    /// [`FrameData`] for this viewport:
    /// ```rust,ignore
    /// let id = renderer.create_viewport(&device);
    /// let frame = FrameData {
    ///     camera: CameraFrame::from_camera(&cam, size).with_viewport_index(id.0),
    ///     ..Default::default()
    /// };
    /// ```
    pub fn create_viewport(&mut self, device: &wgpu::Device) -> ViewportId {
        let idx = self.viewport_slots.len();
        self.ensure_viewport_slot(device, idx);
        ViewportId(idx)
    }

    /// Release the heavy GPU texture memory (HDR targets, OIT, bloom, SSAO) held
    /// by `id`.
    ///
    /// The slot index is not reclaimed : future calls with this `ViewportId` will
    /// lazily recreate the texture resources as needed.  This is useful when a
    /// viewport is hidden or minimised and you want to reduce VRAM pressure without
    /// invalidating the handle.
    pub fn destroy_viewport(&mut self, id: ViewportId) {
        if let Some(slot) = self.viewport_slots.get_mut(id.0) {
            slot.hdr = None;
        }
    }

    /// Prepare shared scene data.  Call **once per frame**, before any
    /// [`prepare_viewport`](Self::prepare_viewport) calls.
    ///
    /// `frame` provides the scene content (`frame.scene`) and the primary camera
    /// used for shadow cascade framing (`frame.camera`).  In a multi-viewport
    /// setup use any one viewport's `FrameData` here : typically the perspective
    /// view : as the shadow framing reference.
    ///
    /// `scene_effects` carries the scene-global effects: lighting, environment
    /// map, and compute filters.  Obtain it by constructing [`SceneEffects`]
    /// directly or via [`EffectsFrame::split`].
    pub fn prepare_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        scene_effects: &SceneEffects<'_>,
    ) {
        self.prepare_scene_internal(device, queue, frame, scene_effects);
    }

    /// Prepare per-viewport GPU state (camera, clip planes, overlays, axes).
    ///
    /// Call once per viewport per frame, **after** [`prepare_scene`](Self::prepare_scene).
    ///
    /// `id` must have been obtained from [`create_viewport`](Self::create_viewport).
    /// `frame.camera.viewport_index` must equal `id.0`; use
    /// [`CameraFrame::with_viewport_index`] when building the frame.
    pub fn prepare_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ViewportId,
        frame: &FrameData,
    ) {
        debug_assert_eq!(
            frame.camera.viewport_index, id.0,
            "frame.camera.viewport_index ({}) must equal the ViewportId ({}); \
             use CameraFrame::with_viewport_index(id.0)",
            frame.camera.viewport_index, id.0,
        );
        let (_, viewport_fx) = frame.effects.split();
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx);
    }

    /// Issue draw calls for `id` into a `'static` render pass (as provided by egui callbacks).
    ///
    /// This is the method to use from an egui/eframe `CallbackTrait::paint` implementation.
    /// Call [`prepare_scene`](Self::prepare_scene) and [`prepare_viewport`](Self::prepare_viewport)
    /// first (in `CallbackTrait::prepare`), then set the render pass viewport/scissor to confine
    /// drawing to the correct quadrant, and call this method.
    ///
    /// For non-`'static` render passes (winit, iced, manual wgpu), use
    /// [`paint_viewport_to`](Self::paint_viewport_to).
    pub fn paint_viewport(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        id: ViewportId,
        frame: &FrameData,
    ) {
        let vp_idx = id.0;
        let camera_bg = self.viewport_camera_bind_group(vp_idx);
        let grid_bg = self.viewport_grid_bind_group(vp_idx);
        let vp_slot = self.viewport_slots.get(vp_idx);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            grid_bg,
            &self.compute_filter_results,
            vp_slot
        );
        emit_scivis_draw_calls!(
            &self.resources,
            render_pass,
            &self.point_cloud_gpu_data,
            &self.glyph_gpu_data,
            &self.polyline_gpu_data,
            &self.volume_gpu_data,
            &self.streamtube_gpu_data,
            camera_bg
        );
    }

    /// Issue draw calls for `id` into a render pass with any lifetime.
    ///
    /// Identical to [`paint_viewport`](Self::paint_viewport) but accepts a render pass with a
    /// non-`'static` lifetime, making it usable from winit, iced, or raw wgpu where the encoder
    /// creates its own render pass.
    pub fn paint_viewport_to<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        id: ViewportId,
        frame: &FrameData,
    ) {
        let vp_idx = id.0;
        let camera_bg = self.viewport_camera_bind_group(vp_idx);
        let grid_bg = self.viewport_grid_bind_group(vp_idx);
        let vp_slot = self.viewport_slots.get(vp_idx);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            grid_bg,
            &self.compute_filter_results,
            vp_slot
        );
        emit_scivis_draw_calls!(
            &self.resources,
            render_pass,
            &self.point_cloud_gpu_data,
            &self.glyph_gpu_data,
            &self.polyline_gpu_data,
            &self.volume_gpu_data,
            &self.streamtube_gpu_data,
            camera_bg
        );
    }

    /// Return a reference to the camera bind group for the given viewport slot.
    ///
    /// Falls back to `resources.camera_bind_group` if no per-viewport slot
    /// exists (e.g. in single-viewport mode before the first prepare call).
    fn viewport_camera_bind_group(&self, viewport_index: usize) -> &wgpu::BindGroup {
        self.viewport_slots
            .get(viewport_index)
            .map(|slot| &slot.camera_bind_group)
            .unwrap_or(&self.resources.camera_bind_group)
    }

    /// Return a reference to the grid bind group for the given viewport slot.
    ///
    /// Falls back to `resources.grid_bind_group` if no per-viewport slot exists.
    fn viewport_grid_bind_group(&self, viewport_index: usize) -> &wgpu::BindGroup {
        self.viewport_slots
            .get(viewport_index)
            .map(|slot| &slot.grid_bind_group)
            .unwrap_or(&self.resources.grid_bind_group)
    }

    /// Ensure per-viewport HDR state exists for `viewport_index` at dimensions `w`×`h`.
    ///
    /// Calls `ensure_hdr_shared` once to initialise shared pipelines/BGLs/samplers, then
    /// lazily creates or resizes the `ViewportHdrState` inside the slot. Idempotent: if the
    /// slot already has HDR state at the correct size nothing is recreated.
    pub(crate) fn ensure_viewport_hdr(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport_index: usize,
        w: u32,
        h: u32,
        ssaa_factor: u32,
    ) {
        let format = self.resources.target_format;
        // Ensure shared infrastructure (pipelines, BGLs, samplers) exists.
        self.resources.ensure_hdr_shared(device, queue, format);
        // Ensure the slot exists.
        self.ensure_viewport_slot(device, viewport_index);
        let slot = &mut self.viewport_slots[viewport_index];
        // Create or resize the per-viewport HDR state.
        let needs_create = match &slot.hdr {
            None => true,
            Some(h_state) => h_state.size != [w, h] || h_state.ssaa_factor != ssaa_factor,
        };
        if needs_create {
            slot.hdr = Some(self.resources.create_hdr_viewport_state(
                device,
                queue,
                format,
                w,
                h,
                ssaa_factor,
            ));
        }
    }
}
