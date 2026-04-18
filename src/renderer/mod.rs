//! `ViewportRenderer` — the main entry point for the viewport library.
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

pub use self::types::{
    CameraFrame, ClipPlane, ClipVolume, ComputeFilterItem, ComputeFilterKind, EffectsFrame,
    FilterMode, FrameData, GlyphItem, GlyphType, InteractionFrame, LightKind, LightSource,
    LightingSettings, OverlayQuad, PointCloudItem, PointRenderMode, PolylineItem,
    PostProcessSettings, RenderCamera, ScalarBar, ScalarBarAnchor, ScalarBarOrientation,
    SceneFrame, SceneRenderItem, ShadowFilter, StreamtubeItem, SurfaceSubmission, ToneMapping,
    ViewportFrame, VolumeItem,
};

use self::shadows::{compute_cascade_matrix, compute_cascade_splits};
use self::types::{INSTANCING_THRESHOLD, InstancedBatch};
use crate::resources::{
    CameraUniform, ClipPlanesUniform, GridUniform, InstanceData, LightsUniform, ObjectUniform,
    OutlineObjectBuffers, OutlineUniform, PickInstance, SingleLightUniform,
    ViewportGpuResources,
};

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
    /// Last wireframe mode seen during prepare(). Batch layout differs between solid and wireframe.
    last_wireframe_mode: bool,
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
    /// Per-viewport camera uniform buffers and bind groups.
    ///
    /// In single-viewport mode only slot 0 is used (same as the legacy
    /// `resources.camera_bind_group`).  In multi-viewport mode each sub-viewport
    /// has its own slot so concurrent `prepare` calls don't clobber each other.
    ///
    /// The outer Vec is indexed by `FrameData::camera.viewport_index`. Slots are
    /// grown lazily in `prepare` via `ensure_viewport_camera_slot`.
    per_viewport_cameras: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    /// Phase G — GPU compute filter results from the last `prepare()` call.
    ///
    /// Each entry contains a compacted index buffer + count for one filtered mesh.
    /// Consumed during `paint()` to override the mesh's default index buffer.
    /// Cleared and rebuilt each frame.
    compute_filter_results: Vec<crate::resources::ComputeFilterResult>,
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
            last_wireframe_mode: false,
            last_scene_items_count: usize::MAX,
            cached_instance_data: Vec::new(),
            cached_instanced_batches: Vec::new(),
            point_cloud_gpu_data: Vec::new(),
            glyph_gpu_data: Vec::new(),
            polyline_gpu_data: Vec::new(),
            volume_gpu_data: Vec::new(),
            streamtube_gpu_data: Vec::new(),
            per_viewport_cameras: Vec::new(),
            compute_filter_results: Vec::new(),
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

    /// Ensure a per-viewport camera slot exists for `viewport_index`.
    ///
    /// Creates a new `(Buffer, BindGroup)` pair that mirrors the layout of the
    /// shared `resources.camera_bind_group` but with an independent camera
    /// uniform buffer.  Slots are created lazily and never destroyed (there are
    /// at most 4 in the current UI: single, split-h top/bottom, quad × 4).
    fn ensure_viewport_camera_slot(&mut self, device: &wgpu::Device, viewport_index: usize) {
        while self.per_viewport_cameras.len() <= viewport_index {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("per_viewport_camera_buf"),
                size: std::mem::size_of::<crate::resources::CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // The camera bind group (group 0) binds seven resources.  Only binding 0
            // (camera uniform) differs per viewport.  Bindings 1-6 (shadow map,
            // shadow sampler, light uniform, clip planes, shadow atlas info,
            // clip volume) are shared from the primary resources so all viewports
            // see the same lighting, shadow, and clip state.
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("per_viewport_camera_bg"),
                layout: &self.resources.camera_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.resources.shadow_map_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.resources.shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.resources.light_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.resources.clip_planes_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.resources.shadow_info_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.resources.clip_volume_uniform_buf.as_entire_binding(),
                    },
                ],
            });
            self.per_viewport_cameras.push((buf, bg));
        }
    }

    /// Return a reference to the camera bind group for the given viewport slot.
    ///
    /// Falls back to `resources.camera_bind_group` if no per-viewport slot
    /// exists (e.g. in single-viewport mode before the first prepare call).
    fn viewport_camera_bind_group(&self, viewport_index: usize) -> &wgpu::BindGroup {
        self.per_viewport_cameras
            .get(viewport_index)
            .map(|(_, bg)| bg)
            .unwrap_or(&self.resources.camera_bind_group)
    }
}
