//! `ViewportRenderer` — the main entry point for the viewport library.
//!
//! Wraps [`ViewportGpuResources`] and provides `prepare()` / `paint()` methods
//! that take raw `wgpu` types. GUI framework adapters (e.g. the egui
//! `CallbackTrait` impl in the application crate) delegate to these methods.

#[macro_use]
mod types;
mod shadows;
pub mod stats;
pub mod shader_hashes;

pub use self::types::{
    ClipPlane, ClipVolume, ComputeFilterItem, ComputeFilterKind, FilterMode, FrameData,
    GlyphItem, GlyphType, LightKind, LightSource, LightingSettings, OverlayQuad,
    PointCloudItem, PointRenderMode, PolylineItem, PostProcessSettings, ScalarBar,
    ScalarBarAnchor, ScalarBarOrientation, SceneRenderItem, ShadowFilter, StreamtubeItem,
    ToneMapping, VolumeItem,
};

use crate::resources::{
    CameraUniform, ClipPlanesUniform, InstanceData, LightsUniform, ObjectUniform,
    OutlineObjectBuffers, OutlineUniform, OverlayUniform, PickInstance, SingleLightUniform,
    ViewportGpuResources,
};
use self::shadows::{compute_cascade_matrix, compute_cascade_splits};
use self::types::{InstancedBatch, INSTANCING_THRESHOLD};

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
    /// The outer Vec is indexed by `FrameData::viewport_index`.  Slots are
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

    /// Upload per-frame data to GPU buffers and render the shadow pass.
    /// Call before `paint()`.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &FrameData) {
        // Phase G — GPU compute filtering.
        // Dispatch before the render pass. Completely skipped when list is empty (zero overhead).
        if !frame.compute_filter_items.is_empty() {
            self.compute_filter_results =
                self.resources
                    .run_compute_filters(device, queue, &frame.compute_filter_items);
        } else {
            self.compute_filter_results.clear();
        }

        // Ensure built-in colormaps are uploaded on first frame.
        self.resources.ensure_colormaps_initialized(device, queue);

        // Ensure a per-viewport camera slot exists for this viewport index.
        // Must happen before the `resources` borrow below.
        self.ensure_viewport_camera_slot(device, frame.viewport_index);

        let resources = &mut self.resources;
        let lighting = &frame.lighting;

        // Compute scene center / extent for shadow framing.
        let (shadow_center, shadow_extent) =
            if let Some(extent) = frame.lighting.shadow_extent_override {
                (glam::Vec3::ZERO, extent)
            } else if let Some([nx, ny, nz]) = frame.domain_extents {
                let center = glam::Vec3::new(nx / 2.0, ny / 2.0, nz / 2.0);
                let diag = (nx * nx + ny * ny + nz * nz).sqrt();
                (center, diag * 0.75)
            } else {
                (glam::Vec3::ZERO, 20.0)
            };

        /// Build a light-space view-projection matrix for shadow mapping.
        fn compute_shadow_matrix(
            kind: &LightKind,
            shadow_center: glam::Vec3,
            shadow_extent: f32,
        ) -> glam::Mat4 {
            match kind {
                LightKind::Directional { direction } => {
                    let dir = glam::Vec3::from(*direction).normalize();
                    let light_up = if dir.y.abs() > 0.99 {
                        glam::Vec3::Z
                    } else {
                        glam::Vec3::Y
                    };
                    let light_pos = shadow_center + dir * shadow_extent * 2.0;
                    let light_view = glam::Mat4::look_at_rh(light_pos, shadow_center, light_up);
                    let light_proj = glam::Mat4::orthographic_rh(
                        -shadow_extent,
                        shadow_extent,
                        -shadow_extent,
                        shadow_extent,
                        0.01,
                        shadow_extent * 5.0,
                    );
                    light_proj * light_view
                }
                LightKind::Point { position, range } => {
                    let pos = glam::Vec3::from(*position);
                    let to_center = (shadow_center - pos).normalize();
                    let light_up = if to_center.y.abs() > 0.99 {
                        glam::Vec3::Z
                    } else {
                        glam::Vec3::Y
                    };
                    let light_view = glam::Mat4::look_at_rh(pos, shadow_center, light_up);
                    let light_proj =
                        glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, *range);
                    light_proj * light_view
                }
                LightKind::Spot {
                    position,
                    direction,
                    range,
                    ..
                } => {
                    let pos = glam::Vec3::from(*position);
                    let dir = glam::Vec3::from(*direction).normalize();
                    let look_target = pos + dir;
                    let up = if dir.y.abs() > 0.99 {
                        glam::Vec3::Z
                    } else {
                        glam::Vec3::Y
                    };
                    let light_view = glam::Mat4::look_at_rh(pos, look_target, up);
                    let light_proj =
                        glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, *range);
                    light_proj * light_view
                }
            }
        }

        /// Convert a `LightSource` to `SingleLightUniform`, computing shadow matrix for lights[0].
        fn build_single_light_uniform(
            src: &LightSource,
            shadow_center: glam::Vec3,
            shadow_extent: f32,
            compute_shadow: bool,
        ) -> SingleLightUniform {
            let shadow_mat = if compute_shadow {
                compute_shadow_matrix(&src.kind, shadow_center, shadow_extent)
            } else {
                glam::Mat4::IDENTITY
            };

            match &src.kind {
                LightKind::Directional { direction } => SingleLightUniform {
                    light_view_proj: shadow_mat.to_cols_array_2d(),
                    pos_or_dir: *direction,
                    light_type: 0,
                    color: src.color,
                    intensity: src.intensity,
                    range: 0.0,
                    inner_angle: 0.0,
                    outer_angle: 0.0,
                    _pad_align: 0,
                    spot_direction: [0.0, -1.0, 0.0],
                    _pad: [0.0; 5],
                },
                LightKind::Point { position, range } => SingleLightUniform {
                    light_view_proj: shadow_mat.to_cols_array_2d(),
                    pos_or_dir: *position,
                    light_type: 1,
                    color: src.color,
                    intensity: src.intensity,
                    range: *range,
                    inner_angle: 0.0,
                    outer_angle: 0.0,
                    _pad_align: 0,
                    spot_direction: [0.0, -1.0, 0.0],
                    _pad: [0.0; 5],
                },
                LightKind::Spot {
                    position,
                    direction,
                    range,
                    inner_angle,
                    outer_angle,
                } => SingleLightUniform {
                    light_view_proj: shadow_mat.to_cols_array_2d(),
                    pos_or_dir: *position,
                    light_type: 2,
                    color: src.color,
                    intensity: src.intensity,
                    range: *range,
                    inner_angle: *inner_angle,
                    outer_angle: *outer_angle,
                    _pad_align: 0,
                    spot_direction: *direction,
                    _pad: [0.0; 5],
                },
            }
        }

        // Build the LightsUniform for all active lights (max 8).
        let light_count = lighting.lights.len().min(8) as u32;
        let mut lights_arr = [SingleLightUniform {
            light_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            pos_or_dir: [0.0; 3],
            light_type: 0,
            color: [1.0; 3],
            intensity: 1.0,
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            _pad_align: 0,
            spot_direction: [0.0, -1.0, 0.0],
            _pad: [0.0; 5],
        }; 8];

        for (i, src) in lighting.lights.iter().take(8).enumerate() {
            lights_arr[i] = build_single_light_uniform(src, shadow_center, shadow_extent, i == 0);
        }

        // -------------------------------------------------------------------
        // Compute CSM cascade matrices for lights[0] (directional).
        // -------------------------------------------------------------------
        let cascade_count = lighting.shadow_cascade_count.clamp(1, 4) as usize;
        let atlas_res = lighting.shadow_atlas_resolution.max(64);
        let tile_size = atlas_res / 2;

        let cascade_splits = compute_cascade_splits(
            frame.camera_near.max(0.01),
            frame.camera_far.max(1.0),
            cascade_count as u32,
            lighting.cascade_split_lambda,
        );

        let light_dir_for_csm = if light_count > 0 {
            match &lighting.lights[0].kind {
                LightKind::Directional { direction } => glam::Vec3::from(*direction).normalize(),
                LightKind::Point { position, .. } => {
                    (glam::Vec3::from(*position) - shadow_center).normalize()
                }
                LightKind::Spot {
                    position,
                    direction,
                    ..
                } => {
                    let _ = position;
                    glam::Vec3::from(*direction).normalize()
                }
            }
        } else {
            glam::Vec3::new(0.3, 1.0, 0.5).normalize()
        };

        let mut cascade_view_projs = [glam::Mat4::IDENTITY; 4];
        // Distance-based splits for fragment shader cascade selection.
        let mut cascade_split_distances = [0.0f32; 4];

        // Determine if we should use CSM (directional light + valid camera data).
        let use_csm = light_count > 0
            && matches!(lighting.lights[0].kind, LightKind::Directional { .. })
            && frame.camera_view != glam::Mat4::IDENTITY;

        if use_csm {
            // Scale factor converting Z-depth splits to 3D-distance thresholds.
            // For any in-frustum fragment with view Z = z, its 3D dist ≤ z * k_scale.
            // Multiplying splits by k_scale ensures the cascade selection in the fragment
            // shader (which uses distance()) never assigns a fragment to a cascade whose
            // matrix doesn't cover it (eliminates the view-angle-dependent gap bands).
            let tan_hfov = (frame.camera_fov * 0.5).tan();
            let k_scale = (1.0f32
                + tan_hfov * tan_hfov * (1.0 + frame.camera_aspect * frame.camera_aspect))
                .sqrt();

            for i in 0..cascade_count {
                let split_near = if i == 0 {
                    frame.camera_near.max(0.01)
                } else {
                    cascade_splits[i - 1]
                };
                let split_far = cascade_splits[i];
                cascade_view_projs[i] = compute_cascade_matrix(
                    light_dir_for_csm,
                    frame.camera_view,
                    frame.camera_fov,
                    frame.camera_aspect,
                    split_near,
                    split_far,
                    tile_size as f32,
                );
                cascade_split_distances[i] = split_far * k_scale;
            }
        } else {
            // Fallback: single shadow map covering the whole scene (legacy behavior).
            let primary_shadow_mat = if light_count > 0 {
                compute_shadow_matrix(&lighting.lights[0].kind, shadow_center, shadow_extent)
            } else {
                glam::Mat4::IDENTITY
            };
            cascade_view_projs[0] = primary_shadow_mat;
            cascade_split_distances[0] = frame.camera_far;
        }
        let effective_cascade_count = if use_csm { cascade_count } else { 1 };

        // Atlas tile layout (2x2 grid):
        // [0] = top-left, [1] = top-right, [2] = bottom-left, [3] = bottom-right
        let atlas_rects: [[f32; 4]; 8] = [
            [0.0, 0.0, 0.5, 0.5], // cascade 0
            [0.5, 0.0, 1.0, 0.5], // cascade 1
            [0.0, 0.5, 0.5, 1.0], // cascade 2
            [0.5, 0.5, 1.0, 1.0], // cascade 3
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4], // unused slots
        ];

        // Upload ShadowAtlasUniform (binding 5).
        {
            let mut vp_data = [[0.0f32; 4]; 16]; // 4 mat4s flattened
            for c in 0..4 {
                let cols = cascade_view_projs[c].to_cols_array_2d();
                for row in 0..4 {
                    vp_data[c * 4 + row] = cols[row];
                }
            }
            let shadow_atlas_uniform = crate::resources::ShadowAtlasUniform {
                cascade_view_proj: vp_data,
                cascade_splits: cascade_split_distances,
                cascade_count: effective_cascade_count as u32,
                atlas_size: atlas_res as f32,
                shadow_filter: match lighting.shadow_filter {
                    ShadowFilter::Pcf => 0,
                    ShadowFilter::Pcss => 1,
                },
                pcss_light_radius: lighting.pcss_light_radius,
                atlas_rects,
            };
            queue.write_buffer(
                &resources.shadow_info_buf,
                0,
                bytemuck::cast_slice(&[shadow_atlas_uniform]),
            );
        }

        // The primary shadow matrix is still stored in lights[0].light_view_proj for
        // backward compat with the non-instanced shadow pass uniform.
        let _primary_shadow_mat = cascade_view_projs[0];

        // Upload clip planes uniform (binding 4).
        {
            let mut planes = [[0.0f32; 4]; 6];
            let mut count = 0u32;
            for plane in frame.clip_planes.iter().filter(|p| p.enabled).take(6) {
                planes[count as usize] = [
                    plane.normal[0],
                    plane.normal[1],
                    plane.normal[2],
                    plane.distance,
                ];
                count += 1;
            }
            let clip_uniform = ClipPlanesUniform {
                planes,
                count,
                _pad0: 0,
                viewport_width: frame.viewport_size[0].max(1.0),
                viewport_height: frame.viewport_size[1].max(1.0),
            };
            queue.write_buffer(
                &resources.clip_planes_uniform_buf,
                0,
                bytemuck::cast_slice(&[clip_uniform]),
            );
        }

        // Upload clip volume uniform (binding 6).
        {
            use crate::resources::ClipVolumeUniform;
            let clip_vol_uniform = ClipVolumeUniform::from_clip_volume(&frame.clip_volume);
            queue.write_buffer(
                &resources.clip_volume_uniform_buf,
                0,
                bytemuck::cast_slice(&[clip_vol_uniform]),
            );
        }

        // Upload camera uniform (view-proj + eye position only).
        let camera_uniform = CameraUniform {
            view_proj: frame.camera_uniform.view_proj,
            eye_pos: frame.eye_pos,
            _pad: 0.0,
        };
        // Write to the shared buffer for single-viewport / legacy callers.
        queue.write_buffer(
            &resources.camera_uniform_buf,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
        // Also write to the per-viewport slot so each sub-viewport gets its
        // own camera transform even though all prepare() calls happen before
        // any paint() calls (egui-wgpu ordering guarantee).
        // `ensure_viewport_camera_slot` must be called first (done above in prepare).
        if let Some((vp_buf, _)) = self.per_viewport_cameras.get(frame.viewport_index) {
            queue.write_buffer(vp_buf, 0, bytemuck::cast_slice(&[camera_uniform]));
        }

        // Upload lights uniform.
        let lights_uniform = LightsUniform {
            count: light_count,
            shadow_bias: lighting.shadow_bias,
            shadows_enabled: if lighting.shadows_enabled { 1 } else { 0 },
            _pad: 0,
            sky_color: lighting.sky_color,
            hemisphere_intensity: lighting.hemisphere_intensity,
            ground_color: lighting.ground_color,
            _pad2: 0.0,
            lights: lights_arr,
        };
        queue.write_buffer(
            &resources.light_uniform_buf,
            0,
            bytemuck::cast_slice(&[lights_uniform]),
        );

        // Upload all cascade matrices to the shadow uniform buffer before the shadow pass.
        // wgpu batches write_buffer calls before the command buffer, so we must write ALL
        // cascade slots up-front; the cascade loop then selects per-slot via dynamic offset.
        const SHADOW_SLOT_STRIDE: u64 = 256;
        for c in 0..4usize {
            queue.write_buffer(
                &resources.shadow_uniform_buf,
                c as u64 * SHADOW_SLOT_STRIDE,
                bytemuck::cast_slice(&cascade_view_projs[c].to_cols_array_2d()),
            );
        }

        // -- Instancing preparation --
        // Determine instancing mode BEFORE per-object uniforms so we can skip them.
        let visible_count = frame.scene_items.iter().filter(|i| i.visible).count();
        let prev_use_instancing = self.use_instancing;
        self.use_instancing = visible_count > INSTANCING_THRESHOLD;

        // If instancing mode changed (e.g. objects added/removed crossing the threshold),
        // clear batches so the generation check below forces a rebuild.
        if self.use_instancing != prev_use_instancing {
            self.instanced_batches.clear();
            self.last_scene_generation = u64::MAX;
            self.last_scene_items_count = usize::MAX;
        }

        // Per-object uniform writes — needed for the non-instanced path, wireframe mode,
        // and for any items with active scalar attributes or two-sided materials
        // (both bypass the instanced path).
        //
        // Also updates each mesh's `object_bind_group` when the material/attribute key changes,
        // keeping the combined (object-uniform + texture + LUT + scalar-buf) bind group consistent.
        let has_scalar_items = frame
            .scene_items
            .iter()
            .any(|i| i.active_attribute.is_some());
        let has_two_sided_items = frame.scene_items.iter().any(|i| i.two_sided);
        if !self.use_instancing || frame.wireframe_mode || has_scalar_items || has_two_sided_items {
            for item in &frame.scene_items {
                if resources
                    .mesh_store
                    .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                    .is_none()
                {
                    tracing::warn!(
                        mesh_index = item.mesh_index,
                        "scene item mesh_index invalid, skipping"
                    );
                    continue;
                };
                let m = &item.material;
                // Compute scalar attribute range.
                let (has_attr, s_min, s_max) = if let Some(attr_ref) = &item.active_attribute {
                    let range = item
                        .scalar_range
                        .or_else(|| {
                            resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                .and_then(|mesh| mesh.attribute_ranges.get(&attr_ref.name).copied())
                        })
                        .unwrap_or((0.0, 1.0));
                    (1u32, range.0, range.1)
                } else {
                    (0u32, 0.0, 1.0)
                };
                let obj_uniform = ObjectUniform {
                    model: item.model,
                    color: [m.base_color[0], m.base_color[1], m.base_color[2], m.opacity],
                    selected: if item.selected { 1 } else { 0 },
                    wireframe: if frame.wireframe_mode { 1 } else { 0 },
                    ambient: m.ambient,
                    diffuse: m.diffuse,
                    specular: m.specular,
                    shininess: m.shininess,
                    has_texture: if m.texture_id.is_some() { 1 } else { 0 },
                    use_pbr: if m.use_pbr { 1 } else { 0 },
                    metallic: m.metallic,
                    roughness: m.roughness,
                    has_normal_map: if m.normal_map_id.is_some() { 1 } else { 0 },
                    has_ao_map: if m.ao_map_id.is_some() { 1 } else { 0 },
                    has_attribute: has_attr,
                    scalar_min: s_min,
                    scalar_max: s_max,
                    _pad_scalar: 0,
                    nan_color: item.nan_color.unwrap_or([0.0; 4]),
                    use_nan_color: if item.nan_color.is_some() { 1 } else { 0 },
                    _pad_nan: [0; 3],
                };

                let normal_obj_uniform = ObjectUniform {
                    model: item.model,
                    color: [1.0, 1.0, 1.0, 1.0],
                    selected: 0,
                    wireframe: 0,
                    ambient: 0.15,
                    diffuse: 0.75,
                    specular: 0.4,
                    shininess: 32.0,
                    has_texture: 0,
                    use_pbr: 0,
                    metallic: 0.0,
                    roughness: 0.5,
                    has_normal_map: 0,
                    has_ao_map: 0,
                    has_attribute: 0,
                    scalar_min: 0.0,
                    scalar_max: 1.0,
                    _pad_scalar: 0,
                    nan_color: [0.0; 4],
                    use_nan_color: 0,
                    _pad_nan: [0; 3],
                };

                // Write uniform data — use get() to read buffer references, then drop.
                {
                    let mesh = resources
                        .mesh_store
                        .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                        .unwrap();
                    queue.write_buffer(
                        &mesh.object_uniform_buf,
                        0,
                        bytemuck::cast_slice(&[obj_uniform]),
                    );
                    queue.write_buffer(
                        &mesh.normal_uniform_buf,
                        0,
                        bytemuck::cast_slice(&[normal_obj_uniform]),
                    );
                } // mesh borrow dropped here

                // Rebuild the object bind group if material/attribute/LUT changed.
                resources.update_mesh_texture_bind_group(
                    device,
                    item.mesh_index,
                    item.material.texture_id,
                    item.material.normal_map_id,
                    item.material.ao_map_id,
                    item.colormap_id,
                    item.active_attribute.as_ref().map(|a| a.name.as_str()),
                );
            }
        }

        if self.use_instancing {
            resources.ensure_instanced_pipelines(device);

            // Generation-based cache: skip batch rebuild and GPU upload when nothing changed.
            // Also include the scene_items count so that frustum-culling changes (different
            // visible set passed in by the caller) correctly invalidate the cache even when
            // scene_generation is stable (scene not mutated, only camera moved).
            let cache_valid = frame.scene_generation == self.last_scene_generation
                && frame.selection_generation == self.last_selection_generation
                && frame.wireframe_mode == self.last_wireframe_mode
                && frame.scene_items.len() == self.last_scene_items_count;

            if !cache_valid {
                // Cache miss — rebuild batches and upload instance data.

                // Collect visible items with valid meshes, then sort by batch key.
                // Items with active scalar attributes or two-sided rasterization are
                // excluded from instancing — they need per-object draw pipelines.
                let mut sorted_items: Vec<&SceneRenderItem> = frame
                    .scene_items
                    .iter()
                    .filter(|item| {
                        item.visible
                            && item.active_attribute.is_none()
                            && !item.two_sided
                            && resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                .is_some()
                    })
                    .collect();

                // Sort by (mesh_index, texture_id, normal_map_id, ao_map_id) so identical
                // batch keys are contiguous — enables O(N) linear scan instead of HashMap.
                sorted_items.sort_unstable_by_key(|item| {
                    (
                        item.mesh_index,
                        item.material.texture_id,
                        item.material.normal_map_id,
                        item.material.ao_map_id,
                    )
                });

                // Build contiguous instance data array and batch descriptors via linear scan.
                let mut all_instances: Vec<InstanceData> = Vec::with_capacity(sorted_items.len());
                let mut instanced_batches: Vec<InstancedBatch> = Vec::new();

                if !sorted_items.is_empty() {
                    let mut batch_start = 0usize;
                    for i in 1..=sorted_items.len() {
                        let at_end = i == sorted_items.len();
                        let key_changed = !at_end && {
                            let a = sorted_items[batch_start];
                            let b = sorted_items[i];
                            a.mesh_index != b.mesh_index
                                || a.material.texture_id != b.material.texture_id
                                || a.material.normal_map_id != b.material.normal_map_id
                                || a.material.ao_map_id != b.material.ao_map_id
                        };

                        if at_end || key_changed {
                            // Flush the current batch.
                            let batch_items = &sorted_items[batch_start..i];
                            let rep = batch_items[0]; // representative item for batch metadata
                            let instance_offset = all_instances.len() as u32;
                            let is_transparent = rep.material.opacity < 1.0;

                            for item in batch_items {
                                let m = &item.material;
                                all_instances.push(InstanceData {
                                    model: item.model,
                                    color: [
                                        m.base_color[0],
                                        m.base_color[1],
                                        m.base_color[2],
                                        m.opacity,
                                    ],
                                    selected: if item.selected { 1 } else { 0 },
                                    wireframe: if frame.wireframe_mode { 1 } else { 0 },
                                    ambient: m.ambient,
                                    diffuse: m.diffuse,
                                    specular: m.specular,
                                    shininess: m.shininess,
                                    has_texture: if m.texture_id.is_some() { 1 } else { 0 },
                                    use_pbr: if m.use_pbr { 1 } else { 0 },
                                    metallic: m.metallic,
                                    roughness: m.roughness,
                                    has_normal_map: if m.normal_map_id.is_some() { 1 } else { 0 },
                                    has_ao_map: if m.ao_map_id.is_some() { 1 } else { 0 },
                                });
                            }

                            instanced_batches.push(InstancedBatch {
                                mesh_index: rep.mesh_index,
                                texture_id: rep.material.texture_id,
                                normal_map_id: rep.material.normal_map_id,
                                ao_map_id: rep.material.ao_map_id,
                                instance_offset,
                                instance_count: batch_items.len() as u32,
                                is_transparent,
                            });

                            batch_start = i;
                        }
                    }
                }

                // Store to cache.
                self.cached_instance_data = all_instances;
                self.cached_instanced_batches = instanced_batches;

                // Upload instance data to GPU.
                resources.upload_instance_data(device, queue, &self.cached_instance_data);

                // Promote cached batches to active batches.
                self.instanced_batches = self.cached_instanced_batches.clone();

                // Store generations so the next frame can detect staleness.
                self.last_scene_generation = frame.scene_generation;
                self.last_selection_generation = frame.selection_generation;
                self.last_wireframe_mode = frame.wireframe_mode;
                self.last_scene_items_count = frame.scene_items.len();

                // Prime instance+texture bind group cache for all batches.
                // Called here (while resources is &mut) so the draw macro only needs &resources.
                for batch in &self.instanced_batches {
                    resources.get_instance_bind_group(
                        device,
                        batch.texture_id,
                        batch.normal_map_id,
                        batch.ao_map_id,
                    );
                }
            } else {
                // Cache hit: batches unchanged, but instance bind groups must still be primed
                // in case the storage buffer was resized (cache cleared) without batch rebuild.
                for batch in &self.instanced_batches {
                    resources.get_instance_bind_group(
                        device,
                        batch.texture_id,
                        batch.normal_map_id,
                        batch.ao_map_id,
                    );
                }
            }
            // On cache hit: self.instanced_batches is reused unchanged; no GPU upload needed.
        }

        // Non-instanced path: mesh.object_bind_group already carries the texture (updated
        // per-item in the uniform-write loop above). No separate material bind group needed.

        // Rebuild outline / x-ray per-object buffers.
        resources.outline_object_buffers.clear();
        if frame.outline_selected {
            for item in &frame.scene_items {
                if !item.visible || !item.selected {
                    continue;
                }
                let m = &item.material;
                let stencil_uniform = ObjectUniform {
                    model: item.model,
                    color: [m.base_color[0], m.base_color[1], m.base_color[2], m.opacity],
                    selected: 1,
                    wireframe: 0,
                    ambient: m.ambient,
                    diffuse: m.diffuse,
                    specular: m.specular,
                    shininess: m.shininess,
                    has_texture: if m.texture_id.is_some() { 1 } else { 0 },
                    use_pbr: if m.use_pbr { 1 } else { 0 },
                    metallic: m.metallic,
                    roughness: m.roughness,
                    has_normal_map: if m.normal_map_id.is_some() { 1 } else { 0 },
                    has_ao_map: if m.ao_map_id.is_some() { 1 } else { 0 },
                    has_attribute: 0,
                    scalar_min: 0.0,
                    scalar_max: 1.0,
                    _pad_scalar: 0,
                    nan_color: [0.0; 4],
                    use_nan_color: 0,
                    _pad_nan: [0; 3],
                };
                let stencil_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("outline_stencil_object_uniform_buf"),
                    size: std::mem::size_of::<ObjectUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&stencil_buf, 0, bytemuck::cast_slice(&[stencil_uniform]));

                let albedo_view = match m.texture_id {
                    Some(id) if (id as usize) < resources.textures.len() => {
                        &resources.textures[id as usize].view
                    }
                    _ => &resources.fallback_texture.view,
                };
                let normal_view = match m.normal_map_id {
                    Some(id) if (id as usize) < resources.textures.len() => {
                        &resources.textures[id as usize].view
                    }
                    _ => &resources.fallback_normal_map_view,
                };
                let ao_view = match m.ao_map_id {
                    Some(id) if (id as usize) < resources.textures.len() => {
                        &resources.textures[id as usize].view
                    }
                    _ => &resources.fallback_ao_map_view,
                };
                let stencil_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("outline_stencil_object_bg"),
                    layout: &resources.object_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: stencil_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(albedo_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&resources.material_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(ao_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_lut_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: resources.fallback_scalar_buf.as_entire_binding(),
                        },
                    ],
                });

                let uniform = OutlineUniform {
                    model: item.model,
                    color: frame.outline_color,
                    pixel_offset: frame.outline_width_px,
                    _pad: [0.0; 3],
                };
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("outline_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("outline_object_bg"),
                    layout: &resources.outline_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                resources.outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_index: item.mesh_index,
                    _stencil_uniform_buf: stencil_buf,
                    stencil_bind_group: stencil_bg,
                    _outline_uniform_buf: buf,
                    outline_bind_group: bg,
                });
            }
        }

        resources.xray_object_buffers.clear();
        if frame.xray_selected {
            for item in &frame.scene_items {
                if !item.visible || !item.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    color: frame.xray_color,
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                };
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("xray_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("xray_object_bg"),
                    layout: &resources.outline_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                resources
                    .xray_object_buffers
                    .push((item.mesh_index, buf, bg));
            }
        }

        // Update gizmo.
        if let Some(model) = frame.gizmo_model {
            resources.update_gizmo_uniform(queue, model);
            resources.update_gizmo_mesh(
                device,
                queue,
                frame.gizmo_mode,
                frame.gizmo_hovered,
                frame.gizmo_space_orientation,
            );
        }

        // Update domain wireframe.
        if let Some([nx, ny, nz]) = frame.domain_extents {
            resources.upload_domain_wireframe(device, nx, ny, nz);
            let domain_uniform = OverlayUniform {
                model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                color: [1.0, 1.0, 1.0, 1.0],
            };
            queue.write_buffer(
                &resources.domain_uniform_buf,
                0,
                bytemuck::cast_slice(&[domain_uniform]),
            );
        } else {
            resources.domain_vertex_buffer = None;
            resources.domain_index_buffer = None;
            resources.domain_index_count = 0;
        }

        // Upload grid.
        if frame.show_grid {
            let (nx, ny, nz) = if frame.grid_cell_size > 0.0 && frame.grid_half_extent > 0.0 {
                // Direct cell-size path: express as equivalent domain dims so upload_grid
                // produces the right spacing (spacing = nx/20) and extent.
                let nx = frame.grid_cell_size * 20.0;
                let half = frame.grid_half_extent;
                // upload_grid extends from -(nx*5) to nx*(1+5); we want ±half, so set nx = half/5.
                let nx = (half / 5.0).max(nx);
                (nx, nx, nx)
            } else {
                let [nx, ny, nz] = frame.domain_extents.unwrap_or([10.0, 10.0, 10.0]);
                (nx, ny, nz)
            };
            resources.upload_grid(device, nx, ny, nz, frame.is_2d);
            let grid_uniform = OverlayUniform {
                model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                color: [0.3, 0.3, 0.3, 0.5],
            };
            queue.write_buffer(
                &resources.grid_uniform_buf,
                0,
                bytemuck::cast_slice(&[grid_uniform]),
            );
        } else {
            resources.grid_vertex_buffer = None;
            resources.grid_index_buffer = None;
            resources.grid_index_count = 0;
        }

        // Rebuild overlay quad buffers.
        resources.bc_quad_buffers.clear();
        for quad in &frame.overlay_quads {
            let buf = resources.create_overlay_quad(device, &quad.corners, quad.color);
            resources.bc_quad_buffers.push(buf);
        }

        resources.constraint_line_buffers.clear();
        for overlay in &frame.constraint_overlays {
            let buf = resources.create_constraint_overlay(device, overlay);
            resources.constraint_line_buffers.push(buf);
        }

        // Cap geometry for section-view cross-section fill.
        resources.cap_buffers.clear();
        if frame.cap_fill_enabled {
            let active_planes: Vec<_> = frame.clip_planes.iter().filter(|p| p.enabled).collect();
            for plane in &active_planes {
                let plane_n = glam::Vec3::from(plane.normal);
                for item in frame.scene_items.iter().filter(|i| i.visible) {
                    let Some(mesh) = resources
                        .mesh_store
                        .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                    else {
                        continue;
                    };
                    let model = glam::Mat4::from_cols_array_2d(&item.model);
                    let world_aabb = mesh.aabb.transformed(&model);
                    if !world_aabb.intersects_plane(plane_n, plane.distance) {
                        continue;
                    }
                    let (Some(pos), Some(idx)) = (&mesh.cpu_positions, &mesh.cpu_indices) else {
                        continue;
                    };
                    if let Some(cap) = crate::geometry::cap_geometry::generate_cap_mesh(
                        pos,
                        idx,
                        &model,
                        plane_n,
                        plane.distance,
                    ) {
                        let bc = item.material.base_color;
                        let color = plane.cap_color.unwrap_or([bc[0], bc[1], bc[2], 1.0]);
                        let buf = resources.upload_cap_geometry(device, &cap, color);
                        resources.cap_buffers.push(buf);
                    }
                }
            }
        }

        // Axes indicator.
        if frame.show_axes_indicator && frame.viewport_size[0] > 0.0 && frame.viewport_size[1] > 0.0
        {
            let verts = crate::widgets::axes_indicator::build_axes_geometry(
                frame.viewport_size[0],
                frame.viewport_size[1],
                frame.camera_orientation,
            );
            let byte_size = std::mem::size_of_val(verts.as_slice()) as u64;
            if byte_size > resources.axes_vertex_buffer.size() {
                // Reallocate if too small.
                resources.axes_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("axes_vertex_buf"),
                    size: byte_size,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            if !verts.is_empty() {
                queue.write_buffer(
                    &resources.axes_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&verts),
                );
            }
            resources.axes_vertex_count = verts.len() as u32;
        } else {
            resources.axes_vertex_count = 0;
        }

        // ------------------------------------------------------------------
        // SciVis Phase B — point cloud and glyph GPU data upload.
        // Zero-cost when both vecs are empty (no pipelines created, no uploads).
        // ------------------------------------------------------------------
        self.point_cloud_gpu_data.clear();
        if !frame.point_clouds.is_empty() {
            resources.ensure_point_cloud_pipeline(device);
            for item in &frame.point_clouds {
                if item.positions.is_empty() {
                    continue;
                }
                let gpu_data = resources.upload_point_cloud(device, queue, item);
                self.point_cloud_gpu_data.push(gpu_data);
            }
        }

        self.glyph_gpu_data.clear();
        if !frame.glyphs.is_empty() {
            resources.ensure_glyph_pipeline(device);
            for item in &frame.glyphs {
                if item.positions.is_empty() || item.vectors.is_empty() {
                    continue;
                }
                let gpu_data = resources.upload_glyph_set(device, queue, item);
                self.glyph_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase M8 — polyline GPU data upload.
        // Zero-cost when polylines vec is empty (no pipeline created, no uploads).
        // ------------------------------------------------------------------
        self.polyline_gpu_data.clear();
        if !frame.polylines.is_empty() {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.polylines {
                if item.positions.is_empty() {
                    continue;
                }
                let gpu_data = resources.upload_polyline(device, queue, item);
                self.polyline_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase L — isoline extraction and upload via polyline pipeline.
        // Zero-cost when isoline_items is empty (no pipeline init, no uploads).
        // ------------------------------------------------------------------
        if !frame.isoline_items.is_empty() {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.isoline_items {
                if item.positions.is_empty() || item.indices.is_empty() || item.scalars.is_empty() {
                    continue;
                }
                let (positions, strip_lengths) = crate::geometry::isoline::extract_isolines(item);
                if positions.is_empty() {
                    continue;
                }
                let polyline = PolylineItem {
                    positions,
                    scalars: Vec::new(), // solid color — no per-vertex scalar coloring
                    strip_lengths,
                    scalar_range: None,
                    colormap_id: None,
                    default_color: item.color,
                    line_width: item.line_width,
                    id: 0, // isolines are not individually pickable
                };
                let gpu_data = resources.upload_polyline(device, queue, &polyline);
                self.polyline_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase M — streamtube GPU data upload.
        // Zero-cost when streamtube_items is empty (no pipeline init, no uploads).
        // ------------------------------------------------------------------
        self.streamtube_gpu_data.clear();
        if !frame.streamtube_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.streamtube_items {
                if item.positions.is_empty() || item.strip_lengths.is_empty() {
                    continue;
                }
                let gpu_data = resources.upload_streamtube(device, queue, item);
                if gpu_data.instance_count > 0 {
                    self.streamtube_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase D -- volume GPU data upload.
        // Zero-cost when volumes vec is empty (no pipeline created, no uploads).
        // ------------------------------------------------------------------
        self.volume_gpu_data.clear();
        if !frame.volumes.is_empty() {
            resources.ensure_volume_pipeline(device);
            for item in &frame.volumes {
                let gpu = resources.upload_volume_frame(device, queue, item, &frame.clip_planes);
                self.volume_gpu_data.push(gpu);
            }
        }

        // -- Frame stats --
        {
            let total = frame.scene_items.len() as u32;
            let visible = frame.scene_items.iter().filter(|i| i.visible).count() as u32;
            let mut draw_calls = 0u32;
            let mut triangles = 0u64;
            let instanced_batch_count = if self.use_instancing {
                self.instanced_batches.len() as u32
            } else {
                0
            };

            if self.use_instancing {
                for batch in &self.instanced_batches {
                    if let Some(mesh) = resources
                        .mesh_store
                        .get(crate::resources::mesh_store::MeshId(batch.mesh_index))
                    {
                        draw_calls += 1;
                        triangles += (mesh.index_count / 3) as u64 * batch.instance_count as u64;
                    }
                }
            } else {
                for item in &frame.scene_items {
                    if !item.visible {
                        continue;
                    }
                    if let Some(mesh) = resources
                        .mesh_store
                        .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                    {
                        draw_calls += 1;
                        triangles += (mesh.index_count / 3) as u64;
                    }
                }
            }

            self.last_stats = crate::renderer::stats::FrameStats {
                total_objects: total,
                visible_objects: visible,
                culled_objects: total.saturating_sub(visible),
                draw_calls,
                instanced_batches: instanced_batch_count,
                triangles_submitted: triangles,
                shadow_draw_calls: 0, // Updated below in shadow pass.
            };
        }

        // ------------------------------------------------------------------
        // Shadow depth pass — CSM: render each cascade into its atlas tile.
        // Uses set_viewport() to target different regions of the shadow atlas.
        // Submitted as a separate command buffer before the main pass.
        // ------------------------------------------------------------------
        if frame.lighting.shadows_enabled && !frame.scene_items.is_empty() {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_pass_encoder"),
            });
            {
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &resources.shadow_map_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                let mut shadow_draws = 0u32;
                let tile_px = tile_size as f32;

                if self.use_instancing {
                    // Instanced shadow pass: one draw call per InstancedBatch per cascade.
                    // No per-item limit — all instances in the storage buffer are drawn.
                    if let (Some(pipeline), Some(instance_bg)) = (
                        &resources.shadow_instanced_pipeline,
                        self.instanced_batches.first().and_then(|b| {
                            resources.instance_bind_groups.get(&(
                                b.texture_id.unwrap_or(u64::MAX),
                                b.normal_map_id.unwrap_or(u64::MAX),
                                b.ao_map_id.unwrap_or(u64::MAX),
                            ))
                        }),
                    ) {
                        for cascade in 0..effective_cascade_count {
                            let tile_col = (cascade % 2) as f32;
                            let tile_row = (cascade / 2) as f32;
                            shadow_pass.set_viewport(
                                tile_col * tile_px,
                                tile_row * tile_px,
                                tile_px,
                                tile_px,
                                0.0,
                                1.0,
                            );
                            shadow_pass.set_scissor_rect(
                                (tile_col * tile_px) as u32,
                                (tile_row * tile_px) as u32,
                                tile_size,
                                tile_size,
                            );

                            shadow_pass.set_pipeline(pipeline);

                            // Write this cascade's view-projection matrix into its dedicated buffer.
                            queue.write_buffer(
                                resources.shadow_instanced_cascade_bufs[cascade]
                                    .as_ref()
                                    .expect("shadow_instanced_cascade_bufs not allocated"),
                                0,
                                bytemuck::cast_slice(
                                    &cascade_view_projs[cascade].to_cols_array_2d(),
                                ),
                            );

                            let cascade_bg = resources.shadow_instanced_cascade_bgs[cascade]
                                .as_ref()
                                .expect("shadow_instanced_cascade_bgs not allocated");
                            shadow_pass.set_bind_group(0, cascade_bg, &[]);
                            shadow_pass.set_bind_group(1, instance_bg, &[]);

                            for batch in &self.instanced_batches {
                                // OIT: transparent items do not cast shadows.
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = resources
                                    .mesh_store
                                    .get(crate::resources::mesh_store::MeshId(batch.mesh_index))
                                else {
                                    continue;
                                };
                                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                shadow_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                                shadow_draws += 1;
                            }
                        }
                    }
                } else {
                    // Per-item shadow pass (legacy path, used when instancing is disabled).
                    for cascade in 0..effective_cascade_count {
                        // Set viewport to this cascade's tile in the atlas.
                        let tile_col = (cascade % 2) as f32;
                        let tile_row = (cascade / 2) as f32;
                        shadow_pass.set_viewport(
                            tile_col * tile_px,
                            tile_row * tile_px,
                            tile_px,
                            tile_px,
                            0.0,
                            1.0,
                        );
                        shadow_pass.set_scissor_rect(
                            (tile_col * tile_px) as u32,
                            (tile_row * tile_px) as u32,
                            tile_size,
                            tile_size,
                        );

                        shadow_pass.set_pipeline(&resources.shadow_pipeline);
                        // Dynamic offset selects this cascade's pre-uploaded matrix slot.
                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow_bind_group,
                            &[cascade as u32 * 256],
                        );

                        // Frustum-cull against this cascade's frustum.
                        let cascade_frustum =
                            crate::camera::frustum::Frustum::from_view_proj(&cascade_view_projs[cascade]);

                        for item in frame.scene_items.iter() {
                            if !item.visible {
                                continue;
                            }
                            // OIT: transparent items do not cast shadows.
                            if item.material.opacity < 1.0 {
                                continue;
                            }
                            let Some(mesh) = resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                            else {
                                continue;
                            };

                            let world_aabb = mesh
                                .aabb
                                .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                            if cascade_frustum.cull_aabb(&world_aabb) {
                                continue;
                            }

                            // Use the per-mesh object bind group (already uploaded during
                            // the main pass prepare step) to supply the model matrix.
                            shadow_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            shadow_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            shadow_draws += 1;
                        }
                    }
                }
                drop(shadow_pass);
                self.last_stats.shadow_draw_calls = shadow_draws;
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        // ------------------------------------------------------------------
        // Outline offscreen pass — render stencil-based outline ring into a
        // dedicated RGBA texture so the paint() path (which may lack a
        // depth/stencil attachment, e.g. eframe) can composite it later.
        // ------------------------------------------------------------------
        if frame.outline_selected && !resources.outline_object_buffers.is_empty() {
            let w = frame.viewport_size[0] as u32;
            let h = frame.viewport_size[1] as u32;
            resources.ensure_outline_target(device, w, h);

            if let (Some(color_view), Some(depth_view)) =
                (&resources.outline_color_view, &resources.outline_depth_view)
            {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("outline_offscreen_encoder"),
                });
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("outline_offscreen_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0),
                                store: wgpu::StoreOp::Discard,
                            }),
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    // Pass 1: write stencil=1 for selected objects.
                    // mesh.object_bind_group (group 1) contains both the object uniform and
                    // fallback textures — no separate group 2 bind group needed.
                    pass.set_pipeline(&resources.stencil_write_pipeline);
                    pass.set_stencil_reference(1);
                    pass.set_bind_group(0, &resources.camera_bind_group, &[]);
                    for outlined in &resources.outline_object_buffers {
                        let Some(mesh) = resources
                            .mesh_store
                            .get(crate::resources::mesh_store::MeshId(outlined.mesh_index))
                        else {
                            continue;
                        };
                        pass.set_bind_group(1, &outlined.stencil_bind_group, &[]);
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }

                    // Pass 2: draw expanded outline ring where stencil != 1.
                    pass.set_pipeline(&resources.outline_pipeline);
                    pass.set_stencil_reference(1);
                    for outlined in &resources.outline_object_buffers {
                        let Some(mesh) = resources
                            .mesh_store
                            .get(crate::resources::mesh_store::MeshId(outlined.mesh_index))
                        else {
                            continue;
                        };
                        pass.set_bind_group(1, &outlined.outline_bind_group, &[]);
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
                queue.submit(std::iter::once(encoder.finish()));
            }
        }
    }

    /// Issue draw calls for the viewport. Call inside a `wgpu::RenderPass`.
    ///
    /// This method requires a `'static` render pass (as provided by egui's
    /// `CallbackTrait`). For non-static render passes (iced, manual wgpu),
    /// use [`paint_to`](Self::paint_to).
    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'static>, frame: &FrameData) {
        let camera_bg = self.viewport_camera_bind_group(frame.viewport_index);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            &self.compute_filter_results
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

    /// Issue draw calls into a render pass with any lifetime.
    ///
    /// Identical to [`paint`](Self::paint) but accepts a render pass with a
    /// non-`'static` lifetime, making it usable from iced, raw wgpu, or any
    /// framework that creates its own render pass.
    pub fn paint_to<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>, frame: &FrameData) {
        let camera_bg = self.viewport_camera_bind_group(frame.viewport_index);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            &self.compute_filter_results
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

    /// High-level HDR render method. Handles the full post-processing pipeline:
    /// scene → HDR texture → (bloom) → (SSAO) → tone map → output_view.
    ///
    /// When `frame.post_process.enabled` is false, falls back to a simple LDR render
    /// pass targeting `output_view` directly.
    ///
    /// Returns a `CommandBuffer` ready to submit.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        // Always run prepare() to upload uniforms and run the shadow pass.
        self.prepare(device, queue, frame);

        let bg_color = frame.background_color.unwrap_or([
            65.0 / 255.0,
            65.0 / 255.0,
            65.0 / 255.0,
            1.0,
        ]);

        if !frame.post_process.enabled {
            // LDR fallback: render directly to output_view.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ldr_encoder"),
            });
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ldr_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: bg_color[0] as f64,
                                g: bg_color[1] as f64,
                                b: bg_color[2] as f64,
                                a: bg_color[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: self.resources.outline_depth_view.as_ref().map(|v| {
                        wgpu::RenderPassDepthStencilAttachment {
                            view: v,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: None,
                        }
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                let camera_bg = self.viewport_camera_bind_group(frame.viewport_index);
                emit_draw_calls!(
                    &self.resources,
                    &mut render_pass,
                    frame,
                    self.use_instancing,
                    &self.instanced_batches,
                    camera_bg,
                    &self.compute_filter_results
                );
                emit_scivis_draw_calls!(
                    &self.resources,
                    &mut render_pass,
                    &self.point_cloud_gpu_data,
                    &self.glyph_gpu_data,
                    &self.polyline_gpu_data,
                    &self.volume_gpu_data,
                    &self.streamtube_gpu_data,
                    camera_bg
                );
            }
            return encoder.finish();
        }

        // HDR path.
        let w = frame.viewport_size[0] as u32;
        let h = frame.viewport_size[1] as u32;

        self.resources.ensure_hdr_target(
            device,
            queue,
            self.resources.target_format,
            w.max(1),
            h.max(1),
        );

        let pp = &frame.post_process;

        // Upload tone map uniform.
        let mode = match pp.tone_mapping {
            crate::renderer::ToneMapping::Reinhard => 0u32,
            crate::renderer::ToneMapping::Aces => 1u32,
            crate::renderer::ToneMapping::KhronosNeutral => 2u32,
        };
        let tm_uniform = crate::resources::ToneMapUniform {
            exposure: pp.exposure,
            mode,
            bloom_enabled: if pp.bloom { 1 } else { 0 },
            ssao_enabled: if pp.ssao { 1 } else { 0 },
            contact_shadows_enabled: if pp.contact_shadows { 1 } else { 0 },
            _pad_tm: [0; 3],
        };
        if let Some(buf) = &self.resources.tone_map_uniform_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&[tm_uniform]));
        }

        // Upload SSAO uniform if needed.
        if pp.ssao {
            if let Some(buf) = &self.resources.ssao_uniform_buf {
                let proj = frame.camera_proj;
                let inv_proj = proj.inverse();
                let ssao_uniform = crate::resources::SsaoUniform {
                    inv_proj: inv_proj.to_cols_array_2d(),
                    proj: proj.to_cols_array_2d(),
                    radius: 0.5,
                    bias: 0.025,
                    _pad: [0.0; 2],
                };
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&[ssao_uniform]));
            }
        }

        // Upload contact shadow uniform if needed.
        if pp.contact_shadows {
            if let Some(buf) = &self.resources.contact_shadow_uniform_buf {
                let proj = frame.camera_proj;
                let inv_proj = proj.inverse();
                // Transform first light direction to view space.
                let light_dir_world: glam::Vec3 = if let Some(l) = frame.lighting.lights.first() {
                    match l.kind {
                        LightKind::Directional { direction } => {
                            glam::Vec3::from(direction).normalize()
                        }
                        LightKind::Spot { direction, .. } => {
                            glam::Vec3::from(direction).normalize()
                        }
                        _ => glam::Vec3::new(0.0, -1.0, 0.0),
                    }
                } else {
                    glam::Vec3::new(0.0, -1.0, 0.0)
                };
                let view = frame.camera_view;
                let light_dir_view = view.transform_vector3(light_dir_world).normalize();
                let cs_uniform = crate::resources::ContactShadowUniform {
                    inv_proj: inv_proj.to_cols_array_2d(),
                    proj: proj.to_cols_array_2d(),
                    light_dir_view: light_dir_view.to_array(),
                    max_distance: pp.contact_shadow_max_distance,
                    steps: pp.contact_shadow_steps,
                    thickness: pp.contact_shadow_thickness,
                    _pad: [0.0; 2],
                };
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&[cs_uniform]));
            }
        }

        // Upload bloom threshold/intensity uniform (horizontal flag is ignored for threshold pass).
        if pp.bloom {
            if let Some(buf) = &self.resources.bloom_uniform_buf {
                let bloom_u = crate::resources::BloomUniform {
                    threshold: pp.bloom_threshold,
                    intensity: pp.bloom_intensity,
                    horizontal: 0,
                    _pad: 0,
                };
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&[bloom_u]));
            }
        }

        // Rebuild tone-map bind group with correct bloom/AO texture views.
        self.resources.rebuild_tone_map_bind_group_with_device(
            device,
            pp.bloom,
            pp.ssao,
            pp.contact_shadows,
        );

        // -----------------------------------------------------------------------
        // Pre-allocate OIT targets if any transparent items exist.
        // Must happen before camera_bg is borrowed (borrow-checker constraint).
        // -----------------------------------------------------------------------
        {
            let needs_oit = if self.use_instancing && !self.instanced_batches.is_empty() {
                self.instanced_batches.iter().any(|b| b.is_transparent)
            } else {
                frame
                    .scene_items
                    .iter()
                    .any(|i| i.visible && i.material.opacity < 1.0)
            };
            if needs_oit {
                let w = (frame.viewport_size[0] as u32).max(1);
                let h = (frame.viewport_size[1] as u32).max(1);
                self.resources.ensure_oit_targets(device, w, h);
            }
        }

        // -----------------------------------------------------------------------
        // Build the command encoder.
        // -----------------------------------------------------------------------
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdr_encoder"),
        });

        // Per-viewport camera bind group for the HDR path.
        let camera_bg = self.viewport_camera_bind_group(frame.viewport_index);

        // -----------------------------------------------------------------------
        // HDR scene pass: render geometry into the HDR texture.
        // -----------------------------------------------------------------------
        {
            let hdr_view = match &self.resources.hdr_view {
                Some(v) => v,
                None => {
                    return encoder.finish();
                }
            };
            let hdr_depth_view = match &self.resources.hdr_depth_view {
                Some(v) => v,
                None => {
                    return encoder.finish();
                }
            };

            let clear_wgpu = wgpu::Color {
                r: bg_color[0] as f64,
                g: bg_color[1] as f64,
                b: bg_color[2] as f64,
                a: bg_color[3] as f64,
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_scene_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_wgpu),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: hdr_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let resources = &self.resources;
            render_pass.set_bind_group(0, camera_bg, &[]);

            let use_instancing = self.use_instancing;
            let batches = &self.instanced_batches;

            if !frame.scene_items.is_empty() {
                if use_instancing && !batches.is_empty() {
                    let excluded_items: Vec<&SceneRenderItem> = frame
                        .scene_items
                        .iter()
                        .filter(|item| {
                            item.visible
                                && (item.active_attribute.is_some() || item.two_sided)
                                && resources
                                    .mesh_store
                                    .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                    .is_some()
                        })
                        .collect();

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

                    if !opaque_batches.is_empty() && !frame.wireframe_mode {
                        if let Some(ref pipeline) = resources.hdr_solid_instanced_pipeline {
                            render_pass.set_pipeline(pipeline);
                            for batch in &opaque_batches {
                                let Some(mesh) = resources
                                    .mesh_store
                                    .get(crate::resources::mesh_store::MeshId(batch.mesh_index))
                                else {
                                    continue;
                                };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) =
                                    resources.instance_bind_groups.get(&mat_key)
                                else {
                                    continue;
                                };
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                            }
                        }
                    }

                    // NOTE: transparent_batches are now rendered in the OIT pass below,
                    // not in the HDR scene pass. This block intentionally left empty.
                    let _ = &transparent_batches; // suppress unused warning

                    if frame.wireframe_mode {
                        if let Some(ref hdr_wf) = resources.hdr_wireframe_pipeline {
                            render_pass.set_pipeline(hdr_wf);
                            for item in &frame.scene_items {
                                if !item.visible {
                                    continue;
                                }
                                let Some(mesh) = resources
                                    .mesh_store
                                    .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                else {
                                    continue;
                                };
                                render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.edge_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            }
                        }
                    } else if let (Some(hdr_solid), Some(hdr_solid_two_sided)) = (
                        &resources.hdr_solid_pipeline,
                        &resources.hdr_solid_two_sided_pipeline,
                    ) {
                        for item in excluded_items
                            .into_iter()
                            .filter(|item| item.material.opacity >= 1.0)
                        {
                            let Some(mesh) = resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                            else {
                                continue;
                            };
                            let pipeline = if item.two_sided {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
                } else {
                    // Per-object path.
                    let eye = glam::Vec3::from(frame.eye_pos);
                    let dist_from_eye = |item: &&SceneRenderItem| -> f32 {
                        let pos =
                            glam::Vec3::new(item.model[3][0], item.model[3][1], item.model[3][2]);
                        (pos - eye).length()
                    };

                    let mut opaque: Vec<&SceneRenderItem> = Vec::new();
                    let mut transparent: Vec<&SceneRenderItem> = Vec::new();
                    for item in &frame.scene_items {
                        if !item.visible
                            || resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                .is_none()
                        {
                            continue;
                        }
                        if item.material.opacity < 1.0 {
                            transparent.push(item);
                        } else {
                            opaque.push(item);
                        }
                    }
                    opaque.sort_by(|a, b| {
                        dist_from_eye(a)
                            .partial_cmp(&dist_from_eye(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    transparent.sort_by(|a, b| {
                        dist_from_eye(b)
                            .partial_cmp(&dist_from_eye(a))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    let draw_item_hdr =
                        |render_pass: &mut wgpu::RenderPass<'_>,
                         item: &SceneRenderItem,
                         solid_pl: &wgpu::RenderPipeline,
                         trans_pl: &wgpu::RenderPipeline,
                         wf_pl: &wgpu::RenderPipeline| {
                            let mesh = resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                                .unwrap();
                            // mesh.object_bind_group (group 1) already carries the object uniform
                            // and the correct texture views.
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            if frame.wireframe_mode {
                                render_pass.set_pipeline(wf_pl);
                                render_pass.set_index_buffer(
                                    mesh.edge_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            } else if item.material.opacity < 1.0 {
                                render_pass.set_pipeline(trans_pl);
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            } else {
                                render_pass.set_pipeline(solid_pl);
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            }
                        };

                    // NOTE: only opaque items are drawn here. Transparent items are
                    // routed to the OIT pass below.
                    let _ = &transparent; // suppress unused warning
                    if let (
                        Some(hdr_solid),
                        Some(hdr_solid_two_sided),
                        Some(hdr_trans),
                        Some(hdr_wf),
                    ) = (
                        &resources.hdr_solid_pipeline,
                        &resources.hdr_solid_two_sided_pipeline,
                        &resources.hdr_transparent_pipeline,
                        &resources.hdr_wireframe_pipeline,
                    ) {
                        for item in &opaque {
                            let solid_pl = if item.two_sided {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            draw_item_hdr(&mut render_pass, item, solid_pl, hdr_trans, hdr_wf);
                        }
                    }
                }
            }

            // Cap fill pass (HDR path — section view cross-section fill).
            if !resources.cap_buffers.is_empty() {
                if let Some(ref hdr_overlay) = resources.hdr_overlay_pipeline {
                    render_pass.set_pipeline(hdr_overlay);
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &resources.cap_buffers {
                        render_pass.set_bind_group(1, bg, &[]);
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }
            }

            // SciVis Phase B+D+M8+M: point cloud, glyph, polyline, volume, streamtube (HDR path).
            emit_scivis_draw_calls!(
                &self.resources,
                &mut render_pass,
                &self.point_cloud_gpu_data,
                &self.glyph_gpu_data,
                &self.polyline_gpu_data,
                &self.volume_gpu_data,
                &self.streamtube_gpu_data,
                camera_bg
            );
        }

        // -----------------------------------------------------------------------
        // OIT pass: render transparent items into accum + reveal textures.
        // Completely skipped when no transparent items exist (zero overhead).
        // -----------------------------------------------------------------------
        let has_transparent = if self.use_instancing && !self.instanced_batches.is_empty() {
            self.instanced_batches.iter().any(|b| b.is_transparent)
        } else {
            frame
                .scene_items
                .iter()
                .any(|i| i.visible && i.material.opacity < 1.0)
        };

        if has_transparent {
            // OIT targets already allocated in the pre-pass above.
            if let (Some(accum_view), Some(reveal_view), Some(hdr_depth_view)) = (
                self.resources.oit_accum_view.as_ref(),
                self.resources.oit_reveal_view.as_ref(),
                self.resources.hdr_depth_view.as_ref(),
            ) {
                // Clear accum to (0,0,0,0), reveal to 1.0 (no contribution yet).
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: accum_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: reveal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 1.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // reuse opaque depth
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                oit_pass.set_bind_group(0, camera_bg, &[]);

                if self.use_instancing && !self.instanced_batches.is_empty() {
                    if let Some(ref pipeline) = self.resources.oit_instanced_pipeline {
                        oit_pass.set_pipeline(pipeline);
                        for batch in &self.instanced_batches {
                            if !batch.is_transparent {
                                continue;
                            }
                            let Some(mesh) = self
                                .resources
                                .mesh_store
                                .get(crate::resources::mesh_store::MeshId(batch.mesh_index))
                            else {
                                continue;
                            };
                            let mat_key = (
                                batch.texture_id.unwrap_or(u64::MAX),
                                batch.normal_map_id.unwrap_or(u64::MAX),
                                batch.ao_map_id.unwrap_or(u64::MAX),
                            );
                            let Some(inst_tex_bg) =
                                self.resources.instance_bind_groups.get(&mat_key)
                            else {
                                continue;
                            };
                            oit_pass.set_bind_group(1, inst_tex_bg, &[]);
                            oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            oit_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            oit_pass.draw_indexed(
                                0..mesh.index_count,
                                0,
                                batch.instance_offset..batch.instance_offset + batch.instance_count,
                            );
                        }
                    }
                } else if let Some(ref pipeline) = self.resources.oit_pipeline {
                    oit_pass.set_pipeline(pipeline);
                    for item in &frame.scene_items {
                        if !item.visible || item.material.opacity >= 1.0 {
                            continue;
                        }
                        let Some(mesh) = self
                            .resources
                            .mesh_store
                            .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                        else {
                            continue;
                        };
                        oit_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                        oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        oit_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        oit_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // OIT composite pass: blend accum/reveal into HDR buffer.
        // Only executes when transparent items were present.
        // -----------------------------------------------------------------------
        if has_transparent {
            if let (Some(pipeline), Some(bg), Some(hdr_view)) = (
                self.resources.oit_composite_pipeline.as_ref(),
                self.resources.oit_composite_bind_group.as_ref(),
                self.resources.hdr_view.as_ref(),
            ) {
                let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_composite_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                composite_pass.set_pipeline(pipeline);
                composite_pass.set_bind_group(0, bg, &[]);
                composite_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Outline composite pass (HDR path): blit offscreen outline onto hdr_view.
        // Runs after the HDR scene pass (which has depth+stencil) in a separate
        // pass with no depth attachment, so the composite pipeline is compatible.
        // -----------------------------------------------------------------------
        if !self.resources.outline_object_buffers.is_empty() {
            if let (Some(pipeline), Some(bg), Some(hdr_view), Some(hdr_depth_view)) = (
                &self.resources.outline_composite_pipeline_single,
                &self.resources.outline_composite_bind_group,
                &self.resources.hdr_view,
                &self.resources.hdr_depth_view,
            ) {
                let mut outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hdr_outline_composite_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                outline_pass.set_pipeline(pipeline);
                outline_pass.set_bind_group(0, bg, &[]);
                outline_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // SSAO pass.
        // -----------------------------------------------------------------------
        if pp.ssao {
            if let (Some(ssao_bg), Some(ssao_pipeline), Some(ssao_view)) = (
                &self.resources.ssao_bg,
                &self.resources.ssao_pipeline,
                &self.resources.ssao_view,
            ) {
                {
                    let mut ssao_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ssao_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: ssao_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    ssao_pass.set_pipeline(ssao_pipeline);
                    ssao_pass.set_bind_group(0, ssao_bg, &[]);
                    ssao_pass.draw(0..3, 0..1);
                }

                // SSAO blur pass.
                if let (Some(ssao_blur_bg), Some(ssao_blur_pipeline), Some(ssao_blur_view)) = (
                    &self.resources.ssao_blur_bg,
                    &self.resources.ssao_blur_pipeline,
                    &self.resources.ssao_blur_view,
                ) {
                    let mut ssao_blur_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("ssao_blur_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: ssao_blur_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    ssao_blur_pass.set_pipeline(ssao_blur_pipeline);
                    ssao_blur_pass.set_bind_group(0, ssao_blur_bg, &[]);
                    ssao_blur_pass.draw(0..3, 0..1);
                }
            }
        }

        // -----------------------------------------------------------------------
        // Contact shadow pass.
        // -----------------------------------------------------------------------
        if pp.contact_shadows {
            if let (Some(cs_bg), Some(cs_pipeline), Some(cs_view)) = (
                &self.resources.contact_shadow_bg,
                &self.resources.contact_shadow_pipeline,
                &self.resources.contact_shadow_view,
            ) {
                let mut cs_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("contact_shadow_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: cs_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                cs_pass.set_pipeline(cs_pipeline);
                cs_pass.set_bind_group(0, cs_bg, &[]);
                cs_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Bloom passes.
        // -----------------------------------------------------------------------
        if pp.bloom {
            // Threshold pass: extract bright pixels into bloom_threshold_texture.
            if let (
                Some(bloom_threshold_bg),
                Some(bloom_threshold_pipeline),
                Some(bloom_threshold_view),
            ) = (
                &self.resources.bloom_threshold_bg,
                &self.resources.bloom_threshold_pipeline,
                &self.resources.bloom_threshold_view,
            ) {
                {
                    let mut threshold_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("bloom_threshold_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: bloom_threshold_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    threshold_pass.set_pipeline(bloom_threshold_pipeline);
                    threshold_pass.set_bind_group(0, bloom_threshold_bg, &[]);
                    threshold_pass.draw(0..3, 0..1);
                }

                // 4 ping-pong H+V blur passes for a wide glow.
                // Pass 1: threshold → ping → pong. Passes 2-4: pong → ping → pong.
                if let (
                    Some(blur_h_bg),
                    Some(blur_h_pong_bg),
                    Some(blur_v_bg),
                    Some(blur_pipeline),
                    Some(bloom_ping_view),
                    Some(bloom_pong_view),
                ) = (
                    &self.resources.bloom_blur_h_bg,
                    &self.resources.bloom_blur_h_pong_bg,
                    &self.resources.bloom_blur_v_bg,
                    &self.resources.bloom_blur_pipeline,
                    &self.resources.bloom_ping_view,
                    &self.resources.bloom_pong_view,
                ) {
                    const BLUR_ITERATIONS: usize = 4;
                    for i in 0..BLUR_ITERATIONS {
                        // H pass: pass 0 reads threshold, subsequent passes read pong.
                        let h_bg = if i == 0 { blur_h_bg } else { blur_h_pong_bg };
                        {
                            let mut h_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("bloom_blur_h_pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: bloom_ping_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                        depth_slice: None,
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            h_pass.set_pipeline(blur_pipeline);
                            h_pass.set_bind_group(0, h_bg, &[]);
                            h_pass.draw(0..3, 0..1);
                        }
                        // V pass: ping → pong.
                        {
                            let mut v_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("bloom_blur_v_pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: bloom_pong_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                        depth_slice: None,
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            v_pass.set_pipeline(blur_pipeline);
                            v_pass.set_bind_group(0, blur_v_bg, &[]);
                            v_pass.draw(0..3, 0..1);
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Tone map pass: HDR + bloom + AO → (fxaa_texture if FXAA) or output_view.
        // -----------------------------------------------------------------------
        let use_fxaa = pp.fxaa && self.resources.fxaa_view.is_some();
        if let (Some(tone_map_pipeline), Some(tone_map_bg)) = (
            &self.resources.tone_map_pipeline,
            &self.resources.tone_map_bind_group,
        ) {
            let tone_target: &wgpu::TextureView = if use_fxaa {
                self.resources.fxaa_view.as_ref().unwrap()
            } else {
                output_view
            };
            let mut tone_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tone_map_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: tone_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            tone_pass.set_pipeline(tone_map_pipeline);
            tone_pass.set_bind_group(0, tone_map_bg, &[]);
            tone_pass.draw(0..3, 0..1);
        }

        // -----------------------------------------------------------------------
        // FXAA pass: fxaa_texture → output_view (only when FXAA is enabled).
        // -----------------------------------------------------------------------
        if use_fxaa {
            if let (Some(fxaa_pipeline), Some(fxaa_bg)) = (
                &self.resources.fxaa_pipeline,
                &self.resources.fxaa_bind_group,
            ) {
                let mut fxaa_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("fxaa_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                fxaa_pass.set_pipeline(fxaa_pipeline);
                fxaa_pass.set_bind_group(0, fxaa_bg, &[]);
                fxaa_pass.draw(0..3, 0..1);
            }
        }

        encoder.finish()
    }

    /// Render a frame to an offscreen texture and return raw RGBA bytes.
    ///
    /// Creates a temporary [`wgpu::Texture`] render target of the given dimensions,
    /// runs all render passes (shadow, scene, post-processing) into it via
    /// [`render()`](Self::render), then copies the result back to CPU memory.
    ///
    /// No OS window or [`wgpu::Surface`] is required. The caller is responsible for
    /// initialising the wgpu adapter with `compatible_surface: None` and for
    /// constructing a valid [`FrameData`] (including `viewport_size` matching
    /// `width`/`height`).
    ///
    /// Returns `width * height * 4` bytes in RGBA8 layout. The caller encodes to
    /// PNG/EXR independently — no image codec dependency in this crate.
    pub fn render_offscreen(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        // 1. Create offscreen texture with RENDER_ATTACHMENT | COPY_SRC usage.
        let target_format = self.resources.target_format;
        let offscreen_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // 2. Create a texture view for rendering into.
        let output_view = offscreen_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 3. Ensure a depth-stencil buffer exists for the given dimensions.
        //    The LDR render pass uses `resources.outline_depth_view` as its depth
        //    attachment. If none exists, `solid_pipeline` (which expects
        //    Depth24PlusStencil8) would produce a wgpu validation error.
        self.resources
            .ensure_outline_target(device, width.max(1), height.max(1));

        // 4. Render the scene into the offscreen texture.
        //    The caller must set `frame.viewport_size` to `[width as f32, height as f32]`
        //    and `frame.camera_aspect` to `width as f32 / height as f32` for correct
        //    HDR target allocation and scissor rects.
        let cmd_buf = self.render(device, queue, &output_view, frame);
        queue.submit(std::iter::once(cmd_buf));

        // 5. Copy texture → staging buffer (wgpu requires row alignment to 256 bytes).
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) & !(align - 1);
        let buffer_size = (padded_row * height.max(1)) as u64;

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen_copy_encoder"),
        });
        copy_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &offscreen_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height.max(1)),
                },
            },
            wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(copy_encoder.finish()));

        // 6. Map buffer and extract tightly-packed RGBA pixels.
        let (tx, rx) = std::sync::mpsc::channel();
        staging_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        let _ = rx.recv().unwrap_or(Err(wgpu::BufferAsyncError));

        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        {
            let mapped = staging_buf.slice(..).get_mapped_range();
            let data: &[u8] = &mapped;
            if padded_row == unpadded_row {
                // No padding — copy entire slice directly.
                pixels.extend_from_slice(data);
            } else {
                // Strip row padding.
                for row in 0..height as usize {
                    let start = row * padded_row as usize;
                    let end = start + unpadded_row as usize;
                    pixels.extend_from_slice(&data[start..end]);
                }
            }
        }
        staging_buf.unmap();

        // 7. Swizzle BGRA → RGBA if the format stores bytes in BGRA order.
        let is_bgra = matches!(
            target_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        if is_bgra {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2); // B ↔ R
            }
        }

        pixels
    }

    // -----------------------------------------------------------------------
    // Phase K — GPU object-ID picking
    // -----------------------------------------------------------------------

    /// GPU object-ID pick: renders the scene to an offscreen `R32Uint` texture
    /// and reads back the single pixel under `cursor`.
    ///
    /// This is O(1) in mesh complexity — every object is rendered with a flat
    /// `u32` ID, and only one pixel is read back. For triangle-level queries
    /// (barycentric scalar probe, exact world position), use the CPU
    /// [`crate::interaction::picking::pick_scene`] path instead.
    ///
    /// The pipeline is lazily initialized on first call — zero overhead when
    /// this method is never invoked.
    ///
    /// # Arguments
    /// * `device` — wgpu device
    /// * `queue` — wgpu queue
    /// * `cursor` — cursor position in viewport-local pixels (top-left origin)
    /// * `frame` — current frame data (camera, scene_items, viewport_size)
    ///
    /// # Returns
    /// `Some(GpuPickHit)` if an object is under the cursor, `None` if empty space.
    pub fn pick_scene_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cursor: glam::Vec2,
        frame: &FrameData,
    ) -> Option<crate::interaction::picking::GpuPickHit> {
        let vp_w = frame.viewport_size[0] as u32;
        let vp_h = frame.viewport_size[1] as u32;

        // --- bounds check ---
        if cursor.x < 0.0
            || cursor.y < 0.0
            || cursor.x >= frame.viewport_size[0]
            || cursor.y >= frame.viewport_size[1]
            || vp_w == 0
            || vp_h == 0
        {
            return None;
        }

        // --- lazy pipeline init ---
        self.resources.ensure_pick_pipeline(device);

        // --- build PickInstance data ---
        // Sentinel scheme: object_id stored = (scene_items_index + 1) so that
        // clear value 0 unambiguously means "no hit".
        let pick_instances: Vec<PickInstance> = frame
            .scene_items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.visible)
            .map(|(idx, item)| {
                let m = item.model;
                PickInstance {
                    model_c0: m[0],
                    model_c1: m[1],
                    model_c2: m[2],
                    model_c3: m[3],
                    object_id: (idx + 1) as u32,
                    _pad: [0; 3],
                }
            })
            .collect();

        if pick_instances.is_empty() {
            return None;
        }

        // Build a mapping from sentinel object_id → original scene_items index.
        // Also track which scene_items are visible and their scene_items indices
        // so we can issue the right draw calls.
        let visible_items: Vec<(usize, &SceneRenderItem)> = frame
            .scene_items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.visible)
            .collect();

        // --- pick instance storage buffer + bind group ---
        let pick_instance_bytes = bytemuck::cast_slice(&pick_instances);
        let pick_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_instance_buf"),
            size: pick_instance_bytes.len().max(80) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pick_instance_buf, 0, pick_instance_bytes);

        let pick_instance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_instance_bg"),
            layout: self
                .resources
                .pick_bind_group_layout_1
                .as_ref()
                .expect("ensure_pick_pipeline must be called first"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pick_instance_buf.as_entire_binding(),
            }],
        });

        // --- pick camera uniform buffer + bind group ---
        let camera_uniform = CameraUniform {
            view_proj: (frame.camera_proj * frame.camera_view).to_cols_array_2d(),
            eye_pos: frame.camera_uniform.eye_pos,
            _pad: 0.0,
        };
        let camera_bytes = bytemuck::bytes_of(&camera_uniform);
        let pick_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_camera_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pick_camera_buf, 0, camera_bytes);

        let pick_camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_camera_bg"),
            layout: self
                .resources
                .pick_camera_bgl
                .as_ref()
                .expect("ensure_pick_pipeline must be called first"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pick_camera_buf.as_entire_binding(),
            }],
        });

        // --- offscreen pick textures (R32Uint + R32Float) + depth ---
        let pick_id_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_id_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_id_view = pick_id_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let pick_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_depth_color_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_depth_view =
            pick_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_ds_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_stencil_view =
            depth_stencil_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- render pass ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pick_pass_encoder"),
        });
        {
            let mut pick_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pick_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pick_id_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pick_depth_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_stencil_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pick_pass.set_pipeline(
                self.resources
                    .pick_pipeline
                    .as_ref()
                    .expect("ensure_pick_pipeline must be called first"),
            );
            pick_pass.set_bind_group(0, &pick_camera_bg, &[]);
            pick_pass.set_bind_group(1, &pick_instance_bg, &[]);

            // Draw each visible item with its instance slot.
            // Instance index in the storage buffer = position in pick_instances vec.
            for (instance_slot, (_, item)) in visible_items.iter().enumerate() {
                let Some(mesh) = self
                    .resources
                    .mesh_store
                    .get(crate::resources::mesh_store::MeshId(item.mesh_index))
                else {
                    continue;
                };
                pick_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pick_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                let slot = instance_slot as u32;
                pick_pass.draw_indexed(0..mesh.index_count, 0, slot..slot + 1);
            }
        }

        // --- copy 1×1 pixels to staging buffers ---
        // R32Uint: 4 bytes per pixel, min bytes_per_row = 256 (wgpu alignment)
        let bytes_per_row_aligned = 256u32; // wgpu requires multiples of 256

        let id_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_id_staging"),
            size: bytes_per_row_aligned as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let depth_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_depth_staging"),
            size: bytes_per_row_aligned as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let px = cursor.x as u32;
        let py = cursor.y as u32;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &pick_id_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &id_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &pick_depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &depth_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // --- map and read ---
        let (tx_id, rx_id) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let (tx_dep, rx_dep) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        id_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_id.send(r);
            });
        depth_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_dep.send(r);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        let _ = rx_id.recv().unwrap_or(Err(wgpu::BufferAsyncError));
        let _ = rx_dep.recv().unwrap_or(Err(wgpu::BufferAsyncError));

        let object_id = {
            let data = id_staging.slice(..).get_mapped_range();
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        id_staging.unmap();

        let depth = {
            let data = depth_staging.slice(..).get_mapped_range();
            f32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        depth_staging.unmap();

        // --- decode sentinel ---
        // 0 = miss (clear color); anything else is (scene_items_index + 1).
        if object_id == 0 {
            return None;
        }

        Some(crate::interaction::picking::GpuPickHit {
            object_id: (object_id - 1) as u64,
            depth,
        })
    }
}
