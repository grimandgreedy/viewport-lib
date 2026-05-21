use super::types::{ClipShape, SceneEffects, ViewportEffects};
use super::*;
use crate::resources::CurveMeshOutlineItem;
use wgpu::util::DeviceExt;

/// Hash a byte slice for per-batch dirty detection.
///
/// Used by the partial-upload path to avoid reading back the cached instance
/// buffer: a hash mismatch means the batch changed; a match means it is clean.
fn hash_instance_bytes(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}

impl ViewportRenderer {
    /// Scene-global prepare stage: compute filters, lighting, shadow pass, batching, scivis.
    ///
    /// Call once per frame before any `prepare_viewport_internal` calls.
    ///
    /// Reads `scene_fx` for lighting, IBL, and compute filters.  Still reads
    /// `frame.camera` for shadow cascade computation (Phase 1 coupling : see
    /// multi-viewport-plan.md § shadow strategy; decoupled in Phase 2).
    pub(super) fn prepare_scene_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        scene_fx: &SceneEffects<'_>,
    ) {
        // Submit copy commands for async texture uploads queued since last frame,
        // and advance ready state for uploads submitted on the previous frame.
        self.resources
            .submit_pending_texture_uploads(device, queue);

        // Phase G : GPU compute filtering.
        // Dispatch before the render pass. Completely skipped when list is empty (zero overhead).
        if !scene_fx.compute_filter_items.is_empty() {
            self.compute_filter_results =
                self.resources
                    .run_compute_filters(device, queue, scene_fx.compute_filter_items);
        } else {
            self.compute_filter_results.clear();
        }

        // Ensure built-in colourmaps and matcaps are uploaded on first frame.
        self.resources.ensure_colourmaps_initialized(device, queue);
        self.resources.ensure_matcaps_initialized(device, queue);

        let resources = &mut self.resources;
        let lighting = scene_fx.lighting;

        // Read scene items from the surface submission.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

        // Compute scene center / extent for shadow framing.
        let (shadow_center, shadow_extent) = if let Some(extent) = lighting.shadow_extent_override {
            (glam::Vec3::ZERO, extent)
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
                    let light_up = if dir.z.abs() > 0.99 {
                        glam::Vec3::Y
                    } else {
                        glam::Vec3::Z
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
                    let light_up = if to_center.z.abs() > 0.99 {
                        glam::Vec3::Y
                    } else {
                        glam::Vec3::Z
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
                    let up = if dir.z.abs() > 0.99 {
                        glam::Vec3::Y
                    } else {
                        glam::Vec3::Z
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
                    colour: src.colour,
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
                    colour: src.colour,
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
                    colour: src.colour,
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
            colour: [1.0; 3],
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
        // Phase 1 note: uses frame.camera : see multi-viewport-plan.md § shadow strategy.
        // -------------------------------------------------------------------
        let cascade_count = lighting.shadow_cascade_count.clamp(1, 4) as usize;
        let atlas_res = lighting.shadow_atlas_resolution.max(64);
        let tile_size = atlas_res / 2;

        let dist = frame.camera.render_camera.distance;
        let shadow_near = (dist * 0.1).max(frame.camera.render_camera.near);
        let shadow_far = (dist * 1.5).max(10.0).min(frame.camera.render_camera.far);
        let cascade_splits = compute_cascade_splits(
            shadow_near,
            shadow_far,
            cascade_count as u32,
            0.75,
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
            && frame.camera.render_camera.view != glam::Mat4::IDENTITY;

        if use_csm {
            for i in 0..cascade_count {
                let split_near = if i == 0 {
                    frame.camera.render_camera.near.max(0.01)
                } else {
                    cascade_splits[i - 1]
                };
                let split_far = cascade_splits[i];
                cascade_view_projs[i] = compute_cascade_matrix(
                    light_dir_for_csm,
                    frame.camera.render_camera.view,
                    frame.camera.render_camera.fov,
                    frame.camera.render_camera.aspect,
                    split_near,
                    split_far,
                    tile_size as f32,
                );
                cascade_split_distances[i] = split_far;
            }
        } else {
            // Fallback: single shadow map covering the whole scene (legacy behavior).
            let primary_shadow_mat = if light_count > 0 {
                compute_shadow_matrix(&lighting.lights[0].kind, shadow_center, shadow_extent)
            } else {
                glam::Mat4::IDENTITY
            };
            cascade_view_projs[0] = primary_shadow_mat;
            cascade_split_distances[0] = frame.camera.render_camera.far;
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
            let shadow_atlas_uniform = ShadowAtlasUniform {
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
            // Write to all per-viewport slot buffers so each viewport's bind group
            // references correctly populated shadow info.
            for slot in &self.viewport_slots {
                queue.write_buffer(
                    &slot.shadow_info_buf,
                    0,
                    bytemuck::cast_slice(&[shadow_atlas_uniform]),
                );
            }
        }

        // The primary shadow matrix is still stored in lights[0].light_view_proj for
        // backward compat with the non-instanced shadow pass uniform.
        let _primary_shadow_mat = cascade_view_projs[0];
        // Cache for ground plane ShadowOnly mode.
        self.last_cascade0_shadow_mat = cascade_view_projs[0];

        // Upload lights uniform.
        // IBL fields from environment map settings.
        let (ibl_enabled, ibl_intensity, ibl_rotation, show_skybox) =
            if let Some(env) = scene_fx.environment {
                if resources.ibl_irradiance_view.is_some() {
                    (
                        1u32,
                        env.intensity,
                        env.rotation,
                        if env.show_skybox { 1u32 } else { 0 },
                    )
                } else {
                    (0, 0.0, 0.0, 0)
                }
            } else {
                (0, 0.0, 0.0, 0)
            };

        let lights_uniform = LightsUniform {
            count: light_count,
            shadow_bias: lighting.shadow_bias,
            shadows_enabled: if lighting.shadows_enabled { 1 } else { 0 },
            _pad: 0,
            sky_colour: lighting.sky_colour,
            hemisphere_intensity: lighting.hemisphere_intensity,
            ground_colour: lighting.ground_colour,
            _pad2: 0.0,
            lights: lights_arr,
            ibl_enabled,
            ibl_intensity,
            ibl_rotation,
            show_skybox,
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

        // Per-frame batch upload counters.  Populated inside the instancing
        // block and folded into FrameStats at the end of prepare_scene_internal.
        let mut batches_reuploaded = 0u32;
        let mut batches_skipped = 0u32;

        // -- Instancing preparation --
        // Determine instancing mode BEFORE per-object uniforms so we can skip them.
        let visible_count = scene_items.iter().filter(|i| !i.appearance.hidden).count();
        let prev_use_instancing = self.use_instancing;
        self.use_instancing = visible_count > INSTANCING_THRESHOLD;

        // If instancing mode changed (e.g. objects added/removed crossing the threshold),
        // clear batches so the generation check below forces a rebuild.
        if self.use_instancing != prev_use_instancing {
            self.instanced_batches.clear();
            self.last_scene_generation = u64::MAX;
            self.last_scene_items_count = usize::MAX;
        }

        // Per-object uniform writes : needed for the non-instanced path, wireframe mode,
        // and for any items with active scalar attributes or two-sided materials
        // (both bypass the instanced path).
        let has_scalar_items = scene_items.iter().any(|i| i.active_attribute.is_some());
        let has_two_sided_items = scene_items.iter().any(|i| i.material.is_two_sided());
        let has_matcap_items = scene_items.iter().any(|i| i.material.matcap_id.is_some());
        let has_param_vis_items = scene_items.iter().any(|i| i.material.param_vis.is_some());
        let has_wireframe_items = scene_items.iter().any(|i| i.appearance.wireframe);
        let has_normal_vis_items = scene_items.iter().any(|i| i.show_normals);
        // Collect per-item uniforms when wireframe mode is on so we can give each
        // visible item its own bind group (the mesh's shared object_uniform_buf gets
        // overwritten when multiple items reference the same MeshId).
        let mut wireframe_uniforms: Vec<ObjectUniform> = Vec::new();
        let collect_wf_uniforms = frame.viewport.wireframe_mode;
        if !self.use_instancing
            || frame.viewport.wireframe_mode
            || has_scalar_items
            || has_two_sided_items
            || has_matcap_items
            || has_param_vis_items
            || has_wireframe_items
            || has_normal_vis_items
        {
            for item in scene_items {
                // When instancing is active, skip items that will be rendered
                // via the instanced path. They don't need per-object uniform
                // writes; writing them anyway causes O(n) write_buffer calls
                // for the whole scene whenever any single item is two-sided.
                if self.use_instancing
                    && !frame.viewport.wireframe_mode
                    && item.active_attribute.is_none()
                    && !item.material.is_two_sided()
                    && item.material.matcap_id.is_none()
                    && item.material.param_vis.is_none()
                    && !item.appearance.wireframe
                    && item.warp_attribute.is_none()
                    && !item.show_normals
                {
                    continue;
                }

                if resources.mesh_store.get(item.mesh_id).is_none() {
                    tracing::warn!(
                        mesh_index = item.mesh_id.index(),
                        "scene item mesh_index invalid, skipping"
                    );
                    continue;
                };
                let m = &item.material;
                // Compute scalar attribute range.
                let (has_attr, s_min, s_max) = if let Some(attr_ref) = &item.active_attribute {
                    let range =
                        item.scalar_range
                            .or_else(|| {
                                resources.mesh_store.get(item.mesh_id).and_then(|mesh| {
                                    mesh.attribute_ranges.get(&attr_ref.name).copied()
                                })
                            })
                            .unwrap_or((0.0, 1.0));
                    (1u32, range.0, range.1)
                } else {
                    (0u32, 0.0, 1.0)
                };
                let obj_uniform = ObjectUniform {
                    model: item.model,
                    colour: [m.base_colour[0], m.base_colour[1], m.base_colour[2], item.appearance.opacity],
                    selected: if item.selected { 1 } else { 0 },
                    wireframe: if frame.viewport.wireframe_mode || item.appearance.wireframe {
                        1
                    } else {
                        0
                    },
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
                    nan_colour: item.nan_colour.unwrap_or([0.0; 4]),
                    use_nan_colour: if item.nan_colour.is_some() { 1 } else { 0 },
                    use_matcap: if m.matcap_id.is_some() { 1 } else { 0 },
                    matcap_blendable: m.matcap_id.map_or(0, |id| if id.blendable { 1 } else { 0 }),
                    unlit: if item.appearance.unlit { 1 } else { 0 },
                    use_face_colour: u32::from(item.active_attribute.as_ref().map_or(false, |a| {
                        a.kind == crate::resources::AttributeKind::FaceColour
                    })),
                    uv_vis_mode: m.param_vis.map_or(0, |pv| pv.mode as u32),
                    uv_vis_scale: m.param_vis.map_or(8.0, |pv| pv.scale),
                    backface_policy: match m.backface_policy {
                        crate::scene::material::BackfacePolicy::Cull => 0,
                        crate::scene::material::BackfacePolicy::Identical => 1,
                        crate::scene::material::BackfacePolicy::DifferentColour(_) => 2,
                        crate::scene::material::BackfacePolicy::Tint(_) => 3,
                        crate::scene::material::BackfacePolicy::Pattern(cfg) => {
                            4 + cfg.pattern as u32
                        }
                    },
                    backface_colour: match m.backface_policy {
                        crate::scene::material::BackfacePolicy::DifferentColour(c) => {
                            [c[0], c[1], c[2], 1.0]
                        }
                        crate::scene::material::BackfacePolicy::Tint(factor) => {
                            [factor, 0.0, 0.0, 1.0]
                        }
                        crate::scene::material::BackfacePolicy::Pattern(cfg) => {
                            let world_extent = resources
                                .mesh_store
                                .get(item.mesh_id)
                                .map(|mesh| {
                                    mesh.aabb
                                        .transformed(&glam::Mat4::from_cols_array_2d(&item.model))
                                        .longest_side()
                                })
                                .unwrap_or(1.0)
                                .max(1e-6);
                            let world_scale = cfg.scale / world_extent;
                            [cfg.colour[0], cfg.colour[1], cfg.colour[2], world_scale]
                        }
                        _ => [0.0; 4],
                    },
                    has_warp: if item.warp_attribute.is_some() { 1 } else { 0 },
                    warp_scale: item.warp_scale,
                    _pad_warp: [0; 2],
                    emissive: m.emissive,
                    _pad_emissive: 0,
                    alpha_mode: match m.alpha_mode {
                        crate::scene::material::AlphaMode::Opaque => 0,
                        crate::scene::material::AlphaMode::Mask(_) => 1,
                        crate::scene::material::AlphaMode::Blend => 2,
                    },
                    alpha_cutoff: match m.alpha_mode {
                        crate::scene::material::AlphaMode::Mask(c) => c,
                        _ => 0.5,
                    },
                    has_metallic_roughness_tex: if m.metallic_roughness_texture_id.is_some() { 1 } else { 0 },
                    has_emissive_tex: if m.emissive_texture_id.is_some() { 1 } else { 0 },
                };

                let normal_obj_uniform = ObjectUniform {
                    model: item.model,
                    colour: [1.0, 1.0, 1.0, 1.0],
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
                    nan_colour: [0.0; 4],
                    use_nan_colour: 0,
                    use_matcap: 0,
                    matcap_blendable: 0,
                    unlit: 0,
                    use_face_colour: 0,
                    uv_vis_mode: 0,
                    uv_vis_scale: 8.0,
                    backface_policy: 0,
                    backface_colour: [0.0; 4],
                    has_warp: 0,
                    warp_scale: 1.0,
                    _pad_warp: [0; 2],
                    emissive: [0.0; 3],
                    _pad_emissive: 0,
                    alpha_mode: 0,
                    alpha_cutoff: 0.5,
                    has_metallic_roughness_tex: 0,
                    has_emissive_tex: 0,
                };

                // Collect per-item uniform for wireframe per-item bind groups.
                if collect_wf_uniforms && !item.appearance.hidden {
                    wireframe_uniforms.push(obj_uniform);
                }

                // Write uniform data : use get() to read buffer references, then drop.
                {
                    let mesh = resources.mesh_store.get(item.mesh_id).unwrap();
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

                // Rebuild the object bind group if material/attribute/LUT/matcap/warp changed.
                resources.update_mesh_texture_bind_group(
                    device,
                    item.mesh_id,
                    item.material.texture_id,
                    item.material.normal_map_id,
                    item.material.ao_map_id,
                    item.colourmap_id,
                    item.active_attribute.as_ref().map(|a| a.name.as_str()),
                    item.material.matcap_id,
                    item.warp_attribute.as_deref(),
                    item.material.metallic_roughness_texture_id,
                    item.material.emissive_texture_id,
                );
            }
        }

        // Build per-item wireframe bind groups so each visible item gets its own
        // object uniform, avoiding the shared-MeshId overwrite problem.
        if !wireframe_uniforms.is_empty() {
            let n = wireframe_uniforms.len();
            let uniform_size = std::mem::size_of::<ObjectUniform>() as u64;

            // Grow the buffer/bind-group pools if needed. We never shrink them.
            while self.wireframe_uniform_bufs.len() < n {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wireframe_item_uniform"),
                    size: uniform_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("wireframe_item_bg"),
                    layout: &resources.object_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_texture.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&resources.material_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_normal_map_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_ao_map_view,
                            ),
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
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(
                                resources
                                    .fallback_matcap_view
                                    .as_ref()
                                    .unwrap_or(&resources.fallback_texture.view),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: resources.fallback_face_colour_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: resources.fallback_warp_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: wgpu::BindingResource::Sampler(&resources.lut_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_metallic_roughness_texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_emissive_texture_view,
                            ),
                        },
                    ],
                });
                self.wireframe_uniform_bufs.push(buf);
                self.wireframe_bind_groups.push(bg);
            }

            // Write each item's uniform into its dedicated buffer.
            for (i, uniform) in wireframe_uniforms.iter().enumerate() {
                queue.write_buffer(
                    &self.wireframe_uniform_bufs[i],
                    0,
                    bytemuck::cast_slice(std::slice::from_ref(uniform)),
                );
            }
        }

        if self.use_instancing {
            resources.ensure_instanced_pipelines(device);
            resources.ensure_hdr_instanced_pipelines(device);

            // Generation-based cache: skip batch rebuild and GPU upload when nothing changed.
            // Phase 2: wireframe_mode removed from cache key : wireframe rendering
            // uses the per-object wireframe_pipeline, not the instanced path, so
            // instance data is now viewport-agnostic.
            //
            // Items with active_attribute, two-sided policy, matcap, or param_vis are
            // excluded from the instanced batch filter. These flags are set on render
            // items AFTER collect_render_items() (per-frame mutations), so they do NOT
            // bump the scene generation. Use last_instancable_count as a cache key
            // instead of a blanket has_per_frame_mutations flag; this allows scenes
            // that mix instanced and non-instanced items (e.g. one two-sided mesh +
            // many static boxes) to still hit the instanced batch cache on frames
            // where the filtered set is unchanged.
            let instancable_count = scene_items
                .iter()
                .filter(|item| {
                    !item.appearance.hidden
                        && item.active_attribute.is_none()
                        && !item.material.is_two_sided()
                        && item.material.matcap_id.is_none()
                        && item.material.param_vis.is_none()
                        && resources.mesh_store.get(item.mesh_id).is_some()
                })
                .count();
            let cache_valid = instancable_count == self.last_instancable_count
                && frame.scene.generation == self.last_scene_generation
                && frame.interaction.selection_generation == self.last_selection_generation
                && scene_items.len() == self.last_scene_items_count;

            if !cache_valid {
                // Cache miss : rebuild batches and upload instance data.
                let mut sorted_items: Vec<&SceneRenderItem> = scene_items
                    .iter()
                    .filter(|item| {
                        !item.appearance.hidden
                            && item.active_attribute.is_none()
                            && !item.material.is_two_sided()
                            && item.material.matcap_id.is_none()
                            && item.material.param_vis.is_none()
                            && resources.mesh_store.get(item.mesh_id).is_some()
                    })
                    .collect();

                sorted_items.sort_unstable_by(|a, b| {
                    // Batch grouping key (must match the batch-split condition).
                    let batch_ord = (
                        a.mesh_id.index(),
                        a.material.texture_id,
                        a.material.normal_map_id,
                        a.material.ao_map_id,
                    )
                    .cmp(&(
                        b.mesh_id.index(),
                        b.material.texture_id,
                        b.material.normal_map_id,
                        b.material.ao_map_id,
                    ));
                    if batch_ord != std::cmp::Ordering::Equal {
                        return batch_ord;
                    }
                    // Within a batch, sort by model matrix for spatial coherence:
                    // column 3 (translation) first, then columns 0-2.  This keeps
                    // spatially close instances adjacent in the buffer, which
                    // reduces GPU cache pressure through the visibility-index
                    // indirection in the culled draw path.
                    for col in [3, 0, 1, 2] {
                        for row in 0..4 {
                            let ord = a.model[col][row]
                                .to_bits()
                                .cmp(&b.model[col][row].to_bits());
                            if ord != std::cmp::Ordering::Equal {
                                return ord;
                            }
                        }
                    }
                    // Final tiebreaker: pick_id is a stable, application-assigned
                    // per-object identity that is guaranteed unique for every
                    // pickable object. Placing it last (rather than in the batch
                    // key) ensures that any two objects with identical transforms
                    // still sort deterministically, regardless of the order they
                    // appear in the caller's scene_items slice.
                    a.pick_id.0.cmp(&b.pick_id.0)
                });

                let mut all_instances: Vec<InstanceData> = Vec::with_capacity(sorted_items.len());
                let mut all_aabbs: Vec<InstanceAabb> = Vec::with_capacity(sorted_items.len());
                let mut batch_metas: Vec<BatchMeta> = Vec::new();
                let mut instanced_batches: Vec<InstancedBatch> = Vec::new();

                if !sorted_items.is_empty() {
                    let mut batch_start = 0usize;
                    for i in 1..=sorted_items.len() {
                        let at_end = i == sorted_items.len();
                        let key_changed = !at_end && {
                            let a = sorted_items[batch_start];
                            let b = sorted_items[i];
                            a.mesh_id != b.mesh_id
                                || a.material.texture_id != b.material.texture_id
                                || a.material.normal_map_id != b.material.normal_map_id
                                || a.material.ao_map_id != b.material.ao_map_id
                        };

                        if at_end || key_changed {
                            let batch_items = &sorted_items[batch_start..i];
                            let rep = batch_items[0];
                            let instance_offset = all_instances.len() as u32;
                            let is_transparent = rep.appearance.opacity < 1.0;

                            // All items in a batch share the same mesh_id (batch key).
                            // Look up the mesh once and reuse it for both index_count and
                            // per-instance AABB transforms, avoiding N redundant hash map
                            // lookups inside the inner loop.
                            let batch_idx = instanced_batches.len() as u32;
                            let batch_mesh = resources.mesh_store.get(rep.mesh_id);
                            let mesh_index_count =
                                batch_mesh.map(|m| m.index_count).unwrap_or(0);

                            for item in batch_items {
                                let m = &item.material;
                                all_instances.push(InstanceData {
                                    model: item.model,
                                    colour: [
                                        m.base_colour[0],
                                        m.base_colour[1],
                                        m.base_colour[2],
                                        item.appearance.opacity,
                                    ],
                                    selected: if item.selected { 1 } else { 0 },
                                    wireframe: 0, // Phase 2: always 0 : wireframe uses per-object pipeline
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
                                    unlit: if item.appearance.unlit { 1 } else { 0 },
                                    _pad_inst: [0; 3],
                                });
                                if let Some(mesh) = batch_mesh {
                                    let model = glam::Mat4::from_cols_array_2d(&item.model);
                                    let world_aabb = mesh.aabb.transformed(&model);
                                    all_aabbs.push(InstanceAabb {
                                        min: world_aabb.min.into(),
                                        batch_index: batch_idx,
                                        max: world_aabb.max.into(),
                                        _pad: 0,
                                    });
                                }
                            }

                            // vis_offset is the prefix sum of instance counts; since
                            // instances are laid out contiguously per batch, it equals
                            // instance_offset.
                            batch_metas.push(BatchMeta {
                                index_count: mesh_index_count,
                                first_index: 0,
                                instance_offset,
                                instance_count: batch_items.len() as u32,
                                vis_offset: instance_offset,
                                is_transparent: if is_transparent { 1 } else { 0 },
                                _pad: [0, 0],
                            });

                            instanced_batches.push(InstancedBatch {
                                mesh_id: rep.mesh_id,
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

                // Partial upload: when the batch structure is unchanged (same
                // count, same offsets and sizes per batch), compare each
                // batch's instance data against the cached CPU copy and only
                // write the sub-ranges that actually differ.  This avoids
                // re-uploading the full buffer when only a small fraction of
                // objects changed (e.g. one animated object in a large static
                // scene).
                //
                // A forced full upload (via `force_dirty()`) or any structural
                // change (different batch count, different instance counts)
                // falls back to the original full-upload path.
                let structure_preserved = self.cached_instance_count > 0
                    && all_instances.len() == self.cached_instance_count
                    && instanced_batches.len() == self.cached_instanced_batches.len()
                    && instanced_batches
                        .iter()
                        .zip(&self.cached_instanced_batches)
                        .all(|(a, b)| {
                            a.mesh_id == b.mesh_id
                                && a.instance_offset == b.instance_offset
                                && a.instance_count == b.instance_count
                        });
                let force = std::mem::replace(&mut self.force_full_upload, false);

                if structure_preserved && !force {
                    let inst_stride = std::mem::size_of::<InstanceData>() as u64;
                    let aabb_stride = std::mem::size_of::<InstanceAabb>() as u64;
                    // Ensure the hash vec is the right length (it should already be,
                    // but guard against a first-run edge case).
                    if self.cached_instance_hashes.len() != instanced_batches.len() {
                        self.cached_instance_hashes.resize(instanced_batches.len(), 0);
                    }
                    for (bi, batch) in instanced_batches.iter().enumerate() {
                        let start = batch.instance_offset as usize;
                        let end = start + batch.instance_count as usize;
                        let new_bytes = bytemuck::cast_slice::<InstanceData, u8>(
                            &all_instances[start..end],
                        );
                        let new_hash = hash_instance_bytes(new_bytes);
                        if new_hash != self.cached_instance_hashes[bi] {
                            if let Some(buf) = resources.instance_storage_buf.as_ref() {
                                queue.write_buffer(
                                    buf,
                                    batch.instance_offset as u64 * inst_stride,
                                    new_bytes,
                                );
                            }
                            if let Some(aabb_buf) = resources.instance_aabb_buf.as_ref() {
                                let aabb_bytes = bytemuck::cast_slice::<InstanceAabb, u8>(
                                    &all_aabbs[start..end],
                                );
                                queue.write_buffer(
                                    aabb_buf,
                                    batch.instance_offset as u64 * aabb_stride,
                                    aabb_bytes,
                                );
                            }
                            self.cached_instance_hashes[bi] = new_hash;
                            batches_reuploaded += 1;
                        } else {
                            batches_skipped += 1;
                        }
                    }
                } else {
                    resources.upload_instance_data(device, queue, &all_instances);
                    resources.upload_aabb_and_batch_meta(device, queue, &all_aabbs, &batch_metas);
                    batches_reuploaded = instanced_batches.len() as u32;
                    // Rebuild the hash cache so the next partial-upload check is seeded.
                    self.cached_instance_hashes.clear();
                    for batch in &instanced_batches {
                        let start = batch.instance_offset as usize;
                        let end = start + batch.instance_count as usize;
                        let bytes = bytemuck::cast_slice::<InstanceData, u8>(&all_instances[start..end]);
                        self.cached_instance_hashes.push(hash_instance_bytes(bytes));
                    }
                }

                self.cached_instance_count = all_instances.len();
                self.cached_instanced_batches = instanced_batches;
                self.instanced_batches = self.cached_instanced_batches.clone();

                self.last_scene_generation = frame.scene.generation;
                self.last_selection_generation = frame.interaction.selection_generation;
                self.last_scene_items_count = scene_items.len();
                self.last_instancable_count = sorted_items.len();

                for batch in &self.instanced_batches {
                    resources.get_instance_bind_group(
                        device,
                        batch.texture_id,
                        batch.normal_map_id,
                        batch.ao_map_id,
                    );
                }
            } else {
                for batch in &self.instanced_batches {
                    resources.get_instance_bind_group(
                        device,
                        batch.texture_id,
                        batch.normal_map_id,
                        batch.ao_map_id,
                    );
                }
            }

            // ------------------------------------------------------------------
            // GPU cull dispatch (Phase 3)
            //
            // Run `cull_instances` + `write_indirect_args` whenever GPU culling
            // is active and all required buffers are allocated.
            // ------------------------------------------------------------------
            if self.gpu_culling_enabled
                && !self.instanced_batches.is_empty()
                && self.cached_instance_count > 0
            {
                let instance_count = self.cached_instance_count as u32;
                let batch_count = self.instanced_batches.len() as u32;

                // Do all mutable borrows before taking immutable borrows from resources.
                if self.cull_resources.is_none() {
                    self.cull_resources =
                        Some(crate::renderer::indirect::CullResources::new(device));
                }
                resources.ensure_cull_instance_pipelines(device);
                for batch in &self.instanced_batches.clone() {
                    resources.get_instance_cull_bind_group(
                        device,
                        batch.texture_id,
                        batch.normal_map_id,
                        batch.ao_map_id,
                    );
                }

                // Now take immutable borrows to the GPU buffers for dispatch.
                if let (
                    Some(aabb_buf),
                    Some(meta_buf),
                    Some(counter_buf),
                    Some(vis_buf),
                    Some(indirect_buf),
                ) = (
                    resources.instance_aabb_buf.as_ref(),
                    resources.batch_meta_buf.as_ref(),
                    resources.batch_counter_buf.as_ref(),
                    resources.visibility_index_buf.as_ref(),
                    resources.indirect_args_buf.as_ref(),
                ) {
                    // Build the FrustumUniform from the current camera view-projection.
                    let vp_mat = frame.camera.render_camera.view_proj();
                    let cpu_frustum = crate::camera::frustum::Frustum::from_view_proj(&vp_mat);
                    let frustum_uniform = crate::resources::FrustumUniform {
                        planes: std::array::from_fn(|i| crate::resources::FrustumPlane {
                            normal: cpu_frustum.planes[i].normal.into(),
                            distance: cpu_frustum.planes[i].d,
                        }),
                        instance_count,
                        batch_count,
                        _pad: [0; 2],
                    };

                    let cull = self.cull_resources.as_ref().unwrap();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cull_encoder"),
                        });
                    cull.dispatch(
                        &mut encoder,
                        device,
                        queue,
                        &frustum_uniform,
                        aabb_buf,
                        meta_buf,
                        counter_buf,
                        vis_buf,
                        indirect_buf,
                        instance_count,
                        batch_count,
                    );

                    // Copy indirect_args_buf to the CPU-readable staging buffer so the
                    // visible instance count can be read back next frame (one-frame lag).
                    let indirect_bytes = batch_count as u64 * 20;
                    if self.indirect_readback_buf.as_ref().map_or(0, |b| b.size()) < indirect_bytes
                    {
                        self.indirect_readback_buf =
                            Some(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("indirect_readback_buf"),
                                size: indirect_bytes,
                                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                                mapped_at_creation: false,
                            }));
                    }
                    if let Some(ref rb_buf) = self.indirect_readback_buf {
                        encoder.copy_buffer_to_buffer(indirect_buf, 0, rb_buf, 0, indirect_bytes);
                    }
                    queue.submit(std::iter::once(encoder.finish()));
                    self.indirect_readback_batch_count = batch_count;
                    self.indirect_readback_pending = true;
                }
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase B : point cloud and glyph GPU data upload.
        // ------------------------------------------------------------------
        self.point_cloud_gpu_data.clear();
        if !frame.scene.point_clouds.is_empty() {
            resources.ensure_point_cloud_pipeline(device);
            for item in &frame.scene.point_clouds {
                if item.positions.is_empty() {
                    continue;
                }
                let gpu_data = resources.upload_point_cloud(device, queue, item);
                self.point_cloud_gpu_data.push(gpu_data);
            }
        }

        self.glyph_gpu_data.clear();
        if !frame.scene.glyphs.is_empty() {
            resources.ensure_glyph_pipeline(device);
            for item in &frame.scene.glyphs {
                if item.positions.is_empty() || item.vectors.is_empty() {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.appearance.wireframe;
                let gpu_data = resources.upload_glyph_set(device, queue, item, wireframe);
                self.glyph_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // Sprite billboard GPU data upload.
        // ------------------------------------------------------------------
        self.sprite_gpu_data.clear();
        if !frame.scene.sprite_items.is_empty() {
            resources.ensure_sprite_pipelines(device);
            for item in &frame.scene.sprite_items {
                if item.positions.is_empty() {
                    continue;
                }
                let mut gd = resources.upload_sprite(device, queue, item);
                gd.wireframe = frame.viewport.wireframe_mode || item.appearance.wireframe;
                self.sprite_gpu_data.push(gd);
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase 5 : tensor glyph GPU data upload.
        // ------------------------------------------------------------------
        self.tensor_glyph_gpu_data.clear();
        if !frame.scene.tensor_glyphs.is_empty() {
            resources.ensure_tensor_glyph_pipeline(device);
            for item in &frame.scene.tensor_glyphs {
                if item.positions.is_empty() {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.appearance.wireframe;
                let gd = resources.upload_tensor_glyph_set(device, queue, item, wireframe);
                self.tensor_glyph_gpu_data.push(gd);
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase M8 : polyline GPU data upload.
        // ------------------------------------------------------------------
        self.polyline_gpu_data.clear();
        self.polyline_selected_gpu_indices.clear();
        let vp_size = frame.camera.viewport_size;
        if !frame.scene.polylines.is_empty() {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.scene.polylines {
                if item.positions.is_empty() {
                    continue;
                }
                let mut gpu_data = resources.upload_polyline(device, queue, item, vp_size);
                gpu_data.wireframe = frame.viewport.wireframe_mode || item.appearance.wireframe;
                if frame.interaction.outline_selected && item.selected {
                    self.polyline_selected_gpu_indices.push(self.polyline_gpu_data.len());
                }
                self.polyline_gpu_data.push(gpu_data);

                // Phase 11: auto-generate GlyphItems for node/edge vector quantities.
                if !item.node_vectors.is_empty() {
                    resources.ensure_glyph_pipeline(device);
                    let g = crate::quantities::polyline_node_vectors_to_glyphs(item);
                    if !g.positions.is_empty() {
                        let wf = frame.viewport.wireframe_mode || item.appearance.wireframe;
                        let gd = resources.upload_glyph_set(device, queue, &g, wf);
                        self.glyph_gpu_data.push(gd);
                    }
                }
                if !item.edge_vectors.is_empty() {
                    resources.ensure_glyph_pipeline(device);
                    let g = crate::quantities::polyline_edge_vectors_to_glyphs(item);
                    if !g.positions.is_empty() {
                        let wf = frame.viewport.wireframe_mode || item.appearance.wireframe;
                        let gd = resources.upload_glyph_set(device, queue, &g, wf);
                        self.glyph_gpu_data.push(gd);
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase L : isoline extraction and upload via polyline pipeline.
        // ------------------------------------------------------------------
        if !frame.scene.isolines.is_empty() {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.scene.isolines {
                if item.positions.is_empty() || item.indices.is_empty() || item.scalars.is_empty() {
                    continue;
                }
                let (positions, strip_lengths) = crate::geometry::isoline::extract_isolines(item);
                if positions.is_empty() {
                    continue;
                }
                let polyline = PolylineItem {
                    positions,
                    scalars: Vec::new(),
                    strip_lengths,
                    scalar_range: None,
                    colourmap_id: None,
                    default_colour: item.colour,
                    line_width: item.line_width,
                    id: 0,
                    ..Default::default()
                };
                let gpu_data = resources.upload_polyline(device, queue, &polyline, vp_size);
                self.polyline_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // Phase 10A : camera frustum wireframes (converted to polylines).
        // ------------------------------------------------------------------
        if !frame.scene.camera_frustums.is_empty() {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.scene.camera_frustums {
                let polyline = item.to_polyline();
                if !polyline.positions.is_empty() {
                    let gpu_data = resources.upload_polyline(device, queue, &polyline, vp_size);
                    self.polyline_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 16 : GPU implicit surface items.
        // ------------------------------------------------------------------
        self.implicit_gpu_data.clear();
        self.pick_implicit_items.clear();
        if !frame.scene.gpu_implicit.is_empty() {
            resources.ensure_implicit_pipeline(device);
            for item in &frame.scene.gpu_implicit {
                if item.primitives.is_empty() {
                    continue;
                }
                let gpu = resources.upload_implicit_item(device, item);
                self.implicit_gpu_data.push(gpu);
                if item.id != 0 {
                    self.pick_implicit_items.push(GpuImplicitPickItem {
                        id: item.id,
                        primitives: item.primitives.clone(),
                        blend_mode: item.blend_mode,
                        max_steps: item.march_options.max_steps,
                        step_scale: item.march_options.step_scale,
                        hit_threshold: item.march_options.hit_threshold,
                        max_distance: item.march_options.max_distance,
                    });
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 17 : GPU marching cubes compute dispatch.
        // ------------------------------------------------------------------
        self.mc_gpu_data.clear();
        self.pick_mc_items.clear();
        if !frame.scene.gpu_mc_jobs.is_empty() {
            resources.ensure_mc_pipelines(device);
            self.mc_gpu_data = resources.run_mc_jobs(device, queue, &frame.scene.gpu_mc_jobs);
            for job in &frame.scene.gpu_mc_jobs {
                if job.id != 0 {
                    if let Some(cpu_data) = &job.cpu_data {
                        self.pick_mc_items.push(GpuMcPickItem {
                            id: job.id,
                            isovalue: job.isovalue,
                            volume_data: cpu_data.clone(),
                        });
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 10B : screen-space image overlays.
        // ------------------------------------------------------------------
        self.screen_image_gpu_data.clear();
        if !frame.scene.screen_images.is_empty() {
            resources.ensure_screen_image_pipeline(device);
            // Phase 12: ensure dc pipeline if any item carries depth data.
            if frame.scene.screen_images.iter().any(|i| i.depth.is_some()) {
                resources.ensure_screen_image_dc_pipeline(device);
            }
            let vp_w = vp_size[0];
            let vp_h = vp_size[1];
            for item in &frame.scene.screen_images {
                if item.width == 0 || item.height == 0 || item.pixels.is_empty() {
                    continue;
                }
                let gpu = resources.upload_screen_image(device, queue, item, vp_w, vp_h);
                self.screen_image_gpu_data.push(gpu);
            }
        }

        // ------------------------------------------------------------------
        // Phase 7 : overlay image overlays (OverlayFrame).
        // ------------------------------------------------------------------
        self.overlay_image_gpu_data.clear();
        if !frame.overlays.images.is_empty() {
            resources.ensure_screen_image_pipeline(device);
            let vp_w = vp_size[0];
            let vp_h = vp_size[1];
            for item in &frame.overlays.images {
                if item.width == 0 || item.height == 0 || item.pixels.is_empty() {
                    continue;
                }
                let gpu = resources.upload_overlay_image(device, queue, item, vp_w, vp_h);
                self.overlay_image_gpu_data.push(gpu);
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase M : streamtube GPU data upload.
        // ------------------------------------------------------------------
        self.streamtube_gpu_data.clear();
        self.streamtube_selected_gpu_indices.clear();
        if !frame.scene.streamtube_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.scene.streamtube_items {
                if item.positions.is_empty() || item.strip_lengths.is_empty() {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.appearance.wireframe;
                let gpu_data = resources.upload_streamtube(device, queue, item, wireframe);
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && item.selected {
                        self.streamtube_selected_gpu_indices.push(self.streamtube_gpu_data.len());
                    }
                    self.streamtube_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 3.3 : General Tube GPU data upload.
        // ------------------------------------------------------------------
        self.tube_gpu_data.clear();
        self.tube_selected_gpu_indices.clear();
        if !frame.scene.tube_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.scene.tube_items {
                if item.positions.is_empty() || item.strip_lengths.is_empty() {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.appearance.wireframe;
                let gpu_data = resources.upload_tube(device, queue, item, wireframe);
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && item.selected {
                        self.tube_selected_gpu_indices.push(self.tube_gpu_data.len());
                    }
                    self.tube_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 8.1 : Ribbon GPU data upload.
        // ------------------------------------------------------------------
        self.ribbon_gpu_data.clear();
        self.ribbon_selected_gpu_indices.clear();
        if !frame.scene.ribbon_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.scene.ribbon_items {
                if item.positions.is_empty() || item.strip_lengths.is_empty() {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.appearance.wireframe;
                let gpu_data = resources.upload_ribbon(device, queue, item, wireframe);
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && item.selected {
                        self.ribbon_selected_gpu_indices.push(self.ribbon_gpu_data.len());
                    }
                    self.ribbon_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 3.2 : Image Slice GPU data upload.
        // ------------------------------------------------------------------
        self.image_slice_gpu_data.clear();
        if !frame.scene.image_slices.is_empty() {
            resources.ensure_image_slice_pipeline(device);
            for item in &frame.scene.image_slices {
                if let Some(gpu_data) = resources.upload_image_slice(device, queue, item) {
                    self.image_slice_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 10 : Volume Surface Slice GPU data upload.
        // ------------------------------------------------------------------
        self.volume_surface_slice_gpu_data.clear();
        if !frame.scene.volume_surface_slices.is_empty() {
            resources.ensure_volume_surface_slice_pipeline(device);
            for item in &frame.scene.volume_surface_slices {
                if let Some(gpu_data) = resources.upload_volume_surface_slice(device, queue, item) {
                    self.volume_surface_slice_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 4: Surface LIC GPU data upload.
        // ------------------------------------------------------------------
        self.lic_gpu_data.clear();
        if !frame.scene.lic_items.is_empty() {
            // The LIC surface pipeline is created inside ensure_hdr_shared (already called before
            // prepare_scene_internal runs), so no separate ensure call is needed here.
            for item in &frame.scene.lic_items {
                if item.vector_attribute.is_empty() {
                    continue;
                }
                if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                    // Verify the vector attribute buffer exists before committing to this item.
                    if mesh
                        .vector_attribute_buffers
                        .contains_key(&item.vector_attribute)
                    {
                        if let Some(bgl) = &resources.lic_surface_bgl {
                            use crate::resources::LicObjectUniform;
                            let model = item.model;
                            let obj_data = LicObjectUniform { model };
                            let obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("lic_object_uniform"),
                                size: std::mem::size_of::<LicObjectUniform>() as u64,
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            queue.write_buffer(&obj_buf, 0, bytemuck::cast_slice(&[obj_data]));
                            // Bind group (group 1): object uniform only.
                            // Flow vectors are bound as vertex buffer 1 in the render pass.
                            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("lic_surface_item_bg"),
                                layout: bgl,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: obj_buf.as_entire_binding(),
                                }],
                            });
                            self.lic_gpu_data.push(crate::resources::LicSurfaceGpuData {
                                bind_group: bg,
                                _object_uniform_buf: obj_buf,
                                mesh_id: item.mesh_id,
                                vector_attribute: item.vector_attribute.clone(),
                            });
                        }
                    }
                }
            }
            // Write LicAdvectUniform to the per-viewport buffer.
            if let Some(hdr) = self.viewport_slots[frame.camera.viewport_index]
                .hdr
                .as_ref()
            {
                if let Some(first) = frame.scene.lic_items.first() {
                    let [vw, vh] = hdr.scene_size;
                    let u = crate::resources::LicAdvectUniform {
                        steps: first.config.steps,
                        step_size: first.config.step_size,
                        vp_width: vw as f32,
                        vp_height: vh as f32,
                    };
                    queue.write_buffer(&hdr.lic_uniform_buf, 0, bytemuck::cast_slice(&[u]));
                }
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase D : volume GPU data upload.
        // Phase 1 note: clip_planes are per-viewport but passed here for culling.
        // Fix in Phase 2/3: upload clip-plane-agnostic data; apply planes in shader.
        // ------------------------------------------------------------------
        self.volume_gpu_data.clear();
        if !frame.scene.volumes.is_empty() {
            resources.ensure_volume_pipeline(device);
            let clip_objects_for_vol = &frame.effects.clip_objects;
            // Phase 5: under budget pressure with allow_volume_quality_reduction, double the
            // step size (half the sample count) to reduce GPU raymarch cost.
            let vol_step_multiplier = if self.degradation_volume_quality_reduced {
                2.0_f32
            } else {
                1.0_f32
            };
            for item in &frame.scene.volumes {
                let mut gpu = resources.upload_volume_frame(
                    device,
                    queue,
                    item,
                    clip_objects_for_vol,
                    vol_step_multiplier,
                );
                gpu.wireframe = frame.viewport.wireframe_mode || item.appearance.wireframe;
                self.volume_gpu_data.push(gpu);
            }
        }

        // Volume wireframe overlay: OBB from bbox + model matrix.
        let need_vol_wf = frame.viewport.wireframe_mode
            || frame.scene.volumes.iter().any(|v| !v.appearance.hidden && v.appearance.wireframe);
        if need_vol_wf {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.scene.volumes {
                if item.appearance.hidden {
                    continue;
                }
                if !(frame.viewport.wireframe_mode || item.appearance.wireframe) {
                    continue;
                }
                let polyline = volume_obb_polyline(item);
                let gpu = resources.upload_polyline(device, queue, &polyline, vp_size);
                self.polyline_gpu_data.push(gpu);
            }
        }

        // TransparentVolumeMesh wireframe: boundary mesh edge overlay.
        self.tvm_wireframe_draws.clear();
        for item in &frame.scene.transparent_volume_meshes {
            if item.appearance.hidden {
                continue;
            }
            if !(item.appearance.wireframe || frame.viewport.wireframe_mode) {
                continue;
            }
            let Some(mesh_id) = item.boundary_mesh_id else {
                continue;
            };
            if resources.mesh_store.get(mesh_id).is_none() {
                continue;
            }
            self.tvm_wireframe_draws.push(mesh_id);
        }
        if !self.tvm_wireframe_draws.is_empty() && self.tvm_wireframe_bg.is_none() {
            use wgpu::util::DeviceExt;
            let mut tvm_wf_uniform: crate::resources::ObjectUniform =
                bytemuck::Zeroable::zeroed();
            tvm_wf_uniform.model = glam::Mat4::IDENTITY.to_cols_array_2d();
            tvm_wf_uniform.colour = [0.75, 0.75, 0.75, 1.0];
            tvm_wf_uniform.wireframe = 1;
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tvm_wireframe_uniform"),
                contents: bytemuck::cast_slice(&[tvm_wf_uniform]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tvm_wireframe_bg"),
                layout: &resources.object_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_texture.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_normal_map_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_ao_map_view,
                        ),
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
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            resources
                                .fallback_matcap_view
                                .as_ref()
                                .unwrap_or(&resources.fallback_texture.view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: resources.fallback_face_colour_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: resources.fallback_warp_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&resources.lut_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_metallic_roughness_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_emissive_texture_view,
                        ),
                    },
                ],
            });
            self.tvm_wireframe_buf = Some(buf);
            self.tvm_wireframe_bg = Some(bg);
        }

        // -- Frame stats --
        {
            let total = scene_items.len() as u32;
            let visible = scene_items.iter().filter(|i| !i.appearance.hidden).count() as u32;
            let mut draw_calls = 0u32;
            let mut triangles = 0u64;
            let instanced_batch_count = if self.use_instancing {
                self.instanced_batches.len() as u32
            } else {
                0
            };

            if self.use_instancing {
                for batch in &self.instanced_batches {
                    if let Some(mesh) = resources.mesh_store.get(batch.mesh_id) {
                        draw_calls += 1;
                        triangles += (mesh.index_count / 3) as u64 * batch.instance_count as u64;
                    }
                }
            } else {
                for item in scene_items {
                    if item.appearance.hidden {
                        continue;
                    }
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
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
                batches_reuploaded,
                batches_skipped,
                triangles_submitted: triangles,
                shadow_draw_calls: 0, // Updated below in shadow pass.
                gpu_culling_active: self.gpu_culling_enabled,
                // Clear stale readback if GPU culling is off this frame.
                gpu_visible_instances: if self.gpu_culling_enabled {
                    self.last_stats.gpu_visible_instances
                } else {
                    None
                },
                ..self.last_stats
            };
        }

        // ------------------------------------------------------------------
        // Shadow depth pass : CSM: render each cascade into its atlas tile.
        // Phase 5: skip the pass entirely when over budget and shadow reduction is allowed.
        // ------------------------------------------------------------------
        let skip_shadows = self.degradation_shadows_skipped;

        // When skipping the shadow pass (budget pressure or empty scene), clear the
        // atlas to max depth so that stale values from a previous frame or a previous
        // showcase don't produce phantom shadows.
        if lighting.shadows_enabled && (skip_shadows || scene_items.is_empty()) {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_clear_encoder"),
            });
            let _ = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_clear_pass"),
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
            queue.submit(std::iter::once(enc.finish()));
        }

        if lighting.shadows_enabled && !scene_items.is_empty() && !skip_shadows {
            // ------------------------------------------------------------------
            // Shadow GPU cull dispatch (Phase 4)
            //
            // For each active cascade, dispatch `cull_instances` + `write_indirect_args`
            // with the cascade frustum. Results land in `shadow_vis_bufs[c]` and
            // `shadow_indirect_bufs[c]`, consumed by the shadow render pass below.
            // All cascade dispatches share the same `batch_counter_buf`; each
            // `write_indirect_args` dispatch resets the counters for the next cascade.
            // ------------------------------------------------------------------
            if self.gpu_culling_enabled
                && self.use_instancing
                && !self.instanced_batches.is_empty()
                && self.cached_instance_count > 0
            {
                // Mutable operations first.
                if self.cull_resources.is_none() {
                    self.cull_resources =
                        Some(crate::renderer::indirect::CullResources::new(device));
                }
                resources.ensure_cull_instance_pipelines(device);
                for c in 0..effective_cascade_count {
                    resources.get_shadow_cull_instance_bind_group(device, c);
                }

                let instance_count = self.cached_instance_count as u32;
                let batch_count = self.instanced_batches.len() as u32;

                if let (Some(aabb_buf), Some(meta_buf), Some(counter_buf)) = (
                    resources.instance_aabb_buf.as_ref(),
                    resources.batch_meta_buf.as_ref(),
                    resources.batch_counter_buf.as_ref(),
                ) {
                    let cull = self.cull_resources.as_ref().unwrap();
                    let mut shadow_cull_encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("shadow_cull_encoder"),
                        });
                    for c in 0..effective_cascade_count {
                        if let (Some(shadow_vis_buf), Some(shadow_indirect_buf)) = (
                            resources.shadow_vis_bufs[c].as_ref(),
                            resources.shadow_indirect_bufs[c].as_ref(),
                        ) {
                            let cpu_frustum = crate::camera::frustum::Frustum::from_view_proj(
                                &cascade_view_projs[c],
                            );
                            let frustum_uniform = crate::resources::FrustumUniform {
                                planes: std::array::from_fn(|i| crate::resources::FrustumPlane {
                                    normal: cpu_frustum.planes[i].normal.into(),
                                    distance: cpu_frustum.planes[i].d,
                                }),
                                instance_count,
                                batch_count,
                                _pad: [0; 2],
                            };
                            cull.dispatch_shadow(
                                &mut shadow_cull_encoder,
                                device,
                                queue,
                                c,
                                &frustum_uniform,
                                aabb_buf,
                                meta_buf,
                                counter_buf,
                                shadow_vis_buf,
                                shadow_indirect_buf,
                                instance_count,
                                batch_count,
                            );
                        }
                    }
                    queue.submit(std::iter::once(shadow_cull_encoder.finish()));
                }
            }

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
                    let use_shadow_indirect = self.gpu_culling_enabled
                        && resources.shadow_instanced_cull_pipeline.is_some()
                        && resources.shadow_vis_bufs[0].is_some();

                    if use_shadow_indirect {
                        // GPU-culled indirect shadow path (Phase 4).
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

                            // Write cascade view-projection matrix.
                            queue.write_buffer(
                                resources.shadow_instanced_cascade_bufs[cascade]
                                    .as_ref()
                                    .expect("shadow_instanced_cascade_bufs not allocated"),
                                0,
                                bytemuck::cast_slice(
                                    &cascade_view_projs[cascade].to_cols_array_2d(),
                                ),
                            );

                            let Some(pipeline) = resources.shadow_instanced_cull_pipeline.as_ref()
                            else {
                                continue;
                            };
                            let Some(cascade_bg) =
                                resources.shadow_instanced_cascade_bgs[cascade].as_ref()
                            else {
                                continue;
                            };
                            let Some(inst_cull_bg) =
                                resources.shadow_cull_instance_bgs[cascade].as_ref()
                            else {
                                continue;
                            };
                            let Some(shadow_indirect_buf) =
                                resources.shadow_indirect_bufs[cascade].as_ref()
                            else {
                                continue;
                            };

                            shadow_pass.set_pipeline(pipeline);
                            shadow_pass.set_bind_group(0, cascade_bg, &[]);
                            shadow_pass.set_bind_group(1, inst_cull_bg, &[]);

                            for (bi, batch) in self.instanced_batches.iter().enumerate() {
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                shadow_pass
                                    .draw_indexed_indirect(shadow_indirect_buf, bi as u64 * 20);
                                shadow_draws += 1;
                            }
                        }
                    } else if let (Some(pipeline), Some(instance_bg)) = (
                        &resources.shadow_instanced_pipeline,
                        self.instanced_batches.first().and_then(|b| {
                            resources.instance_bind_groups.get(&(
                                b.texture_id.unwrap_or(u64::MAX),
                                b.normal_map_id.unwrap_or(u64::MAX),
                                b.ao_map_id.unwrap_or(u64::MAX),
                            ))
                        }),
                    ) {
                        // Direct draw shadow path (fallback when GPU culling is off).
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
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
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

                        shadow_pass.set_pipeline(&resources.shadow_pipeline);
                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow_bind_group,
                            &[cascade as u32 * 256],
                        );

                        let cascade_frustum = crate::camera::frustum::Frustum::from_view_proj(
                            &cascade_view_projs[cascade],
                        );

                        for item in scene_items.iter() {
                            if item.appearance.hidden {
                                continue;
                            }
                            if item.appearance.opacity < 1.0 {
                                continue;
                            }
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };

                            let world_aabb = mesh
                                .aabb
                                .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                            if cascade_frustum.cull_aabb(&world_aabb) {
                                continue;
                            }

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
    }

    /// Per-viewport prepare stage: camera, clip planes, clip volume, grid, overlays, cap geometry, axes.
    ///
    /// Call once per viewport per frame, after `prepare_scene_internal`.
    /// Reads `viewport_fx` for clip planes, clip volume, cap fill, and post-process settings.
    pub(super) fn prepare_viewport_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
    ) {
        // Ensure a per-viewport camera slot exists for this viewport index.
        // Must happen before the `resources` borrow below.
        self.ensure_viewport_slot(device, frame.camera.viewport_index);

        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

        // Capture before the resources mutable borrow so it's accessible inside the block.
        let gp_cascade0_mat = self.last_cascade0_shadow_mat.to_cols_array_2d();

        {
            let resources = &mut self.resources;

            // Upload clip planes + clip volume uniforms from clip_objects.
            {
                let mut planes = [[0.0f32; 4]; 6];
                let mut count = 0u32;
                let mut clip_vols_uniform: ClipVolumesUniform = bytemuck::Zeroable::zeroed();

                for obj in viewport_fx
                    .clip_objects
                    .iter()
                    .filter(|o| o.enabled && o.clip_geometry)
                {
                    match obj.shape {
                        ClipShape::Plane {
                            normal, distance, ..
                        } if count < 6 => {
                            planes[count as usize] = [normal[0], normal[1], normal[2], distance];
                            count += 1;
                        }
                        ClipShape::Box {
                            center,
                            half_extents,
                            orientation,
                        } if (clip_vols_uniform.count as usize) < CLIP_VOLUME_MAX => {
                            let idx = clip_vols_uniform.count as usize;
                            clip_vols_uniform.volumes[idx] =
                                ClipVolumeEntry::from_box(center, half_extents, orientation);
                            clip_vols_uniform.count += 1;
                        }
                        ClipShape::Sphere { center, radius }
                            if (clip_vols_uniform.count as usize) < CLIP_VOLUME_MAX =>
                        {
                            let idx = clip_vols_uniform.count as usize;
                            clip_vols_uniform.volumes[idx] =
                                ClipVolumeEntry::from_sphere(center, radius);
                            clip_vols_uniform.count += 1;
                        }
                        ClipShape::Cylinder {
                            center,
                            axis,
                            radius,
                            half_length,
                        } if (clip_vols_uniform.count as usize) < CLIP_VOLUME_MAX => {
                            let idx = clip_vols_uniform.count as usize;
                            clip_vols_uniform.volumes[idx] =
                                ClipVolumeEntry::from_cylinder(center, axis, radius, half_length);
                            clip_vols_uniform.count += 1;
                        }
                        _ => {}
                    }
                }

                let clip_uniform = ClipPlanesUniform {
                    planes,
                    count,
                    _pad0: 0,
                    viewport_width: frame.camera.viewport_size[0].max(1.0),
                    viewport_height: frame.camera.viewport_size[1].max(1.0),
                };
                // Write to per-viewport slot buffer.
                if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                    queue.write_buffer(
                        &slot.clip_planes_buf,
                        0,
                        bytemuck::cast_slice(&[clip_uniform]),
                    );
                    queue.write_buffer(
                        &slot.clip_volume_buf,
                        0,
                        bytemuck::cast_slice(&[clip_vols_uniform]),
                    );
                }
                // Also write to shared buffers for legacy single-viewport callers.
                queue.write_buffer(
                    &resources.clip_planes_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[clip_uniform]),
                );
                queue.write_buffer(
                    &resources.clip_volume_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[clip_vols_uniform]),
                );
            }

            // Upload camera uniform to per-viewport slot buffer.
            let camera_uniform = frame.camera.render_camera.camera_uniform();
            // Write to shared buffer for legacy single-viewport callers.
            queue.write_buffer(
                &resources.camera_uniform_buf,
                0,
                bytemuck::cast_slice(&[camera_uniform]),
            );
            // Write to the per-viewport slot buffer.
            if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                queue.write_buffer(&slot.camera_buf, 0, bytemuck::cast_slice(&[camera_uniform]));
            }

            // Upload grid uniform (full-screen analytical shader : no vertex buffers needed).
            if frame.viewport.show_grid {
                let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
                if !eye.is_finite() {
                    tracing::warn!(
                        eye_x = eye.x,
                        eye_y = eye.y,
                        eye_z = eye.z,
                        "grid skipped: eye_position is non-finite (camera distance overflow?)"
                    );
                } else {
                    let view_proj_mat = frame.camera.render_camera.view_proj().to_cols_array_2d();

                    let (spacing, minor_fade) = if frame.viewport.grid_cell_size > 0.0 {
                        (frame.viewport.grid_cell_size, 1.0_f32)
                    } else {
                        let vertical_depth = (eye.z - frame.viewport.grid_z).abs().max(1.0);
                        let world_per_pixel =
                            2.0 * (frame.camera.render_camera.fov / 2.0).tan() * vertical_depth
                                / frame.camera.viewport_size[1].max(1.0);
                        let target = (world_per_pixel * 60.0).max(1e-9_f32);
                        let mut s = 1.0_f32;
                        let mut iters = 0u32;
                        while s < target {
                            s *= 10.0;
                            iters += 1;
                        }
                        let ratio = (target / s).clamp(0.0, 1.0);
                        let fade = if ratio < 0.5 {
                            1.0_f32
                        } else {
                            let t = (ratio - 0.5) * 2.0;
                            1.0 - t * t * (3.0 - 2.0 * t)
                        };
                        tracing::debug!(
                            eye_z = eye.z,
                            vertical_depth,
                            world_per_pixel,
                            target,
                            spacing = s,
                            lod_iters = iters,
                            ratio,
                            minor_fade = fade,
                            "grid LOD"
                        );
                        (s, fade)
                    };

                    let spacing_major = spacing * 10.0;
                    let snap_x = (eye.x / spacing_major).floor() * spacing_major;
                    let snap_y = (eye.y / spacing_major).floor() * spacing_major;
                    tracing::debug!(
                        spacing_minor = spacing,
                        spacing_major,
                        snap_x,
                        snap_y,
                        eye_x = eye.x,
                        eye_y = eye.y,
                        eye_z = eye.z,
                        "grid snap"
                    );

                    let orient = frame.camera.render_camera.orientation;
                    let right = orient * glam::Vec3::X;
                    let up = orient * glam::Vec3::Y;
                    let back = orient * glam::Vec3::Z;
                    let cam_to_world = [
                        [right.x, right.y, right.z, 0.0_f32],
                        [up.x, up.y, up.z, 0.0_f32],
                        [back.x, back.y, back.z, 0.0_f32],
                    ];
                    let aspect =
                        frame.camera.viewport_size[0] / frame.camera.viewport_size[1].max(1.0);
                    let tan_half_fov = (frame.camera.render_camera.fov / 2.0).tan();

                    let uniform = GridUniform {
                        view_proj: view_proj_mat,
                        cam_to_world,
                        tan_half_fov,
                        aspect,
                        _pad_ivp: [0.0; 2],
                        eye_pos: frame.camera.render_camera.eye_position,
                        grid_z: frame.viewport.grid_z,
                        spacing_minor: spacing,
                        spacing_major,
                        snap_origin: [snap_x, snap_y],
                        colour_minor: {
                            let [r, g, b] = frame.viewport.grid_colour.unwrap_or([0.55, 0.55, 0.55]);
                            [r, g, b, 0.4 * minor_fade]
                        },
                        colour_major: {
                            let [r, g, b] = frame.viewport.grid_colour.unwrap_or([0.60, 0.60, 0.60]);
                            [r, g, b, 0.4 + 0.2 * minor_fade]
                        },
                    };
                    // Write to per-viewport slot buffer.
                    if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                        queue.write_buffer(&slot.grid_buf, 0, bytemuck::cast_slice(&[uniform]));
                    }
                    // Also write to shared buffer for legacy callers.
                    queue.write_buffer(
                        &resources.grid_uniform_buf,
                        0,
                        bytemuck::cast_slice(&[uniform]),
                    );
                }
            }
            // ------------------------------------------------------------------
            // Ground plane uniform upload.
            // ------------------------------------------------------------------
            {
                let gp = &viewport_fx.ground_plane;
                let mode_u32: u32 = match gp.mode {
                    crate::renderer::types::GroundPlaneMode::None => 0,
                    crate::renderer::types::GroundPlaneMode::ShadowOnly => 1,
                    crate::renderer::types::GroundPlaneMode::Tile => 2,
                    crate::renderer::types::GroundPlaneMode::SolidColour => 3,
                };
                let orient = frame.camera.render_camera.orientation;
                let right = orient * glam::Vec3::X;
                let up = orient * glam::Vec3::Y;
                let back = orient * glam::Vec3::Z;
                let aspect = frame.camera.viewport_size[0] / frame.camera.viewport_size[1].max(1.0);
                let tan_half_fov = (frame.camera.render_camera.fov / 2.0).tan();
                let vp = frame.camera.render_camera.view_proj().to_cols_array_2d();
                let gp_uniform = crate::resources::GroundPlaneUniform {
                    view_proj: vp,
                    cam_right: [right.x, right.y, right.z, 0.0],
                    cam_up: [up.x, up.y, up.z, 0.0],
                    cam_back: [back.x, back.y, back.z, 0.0],
                    eye_pos: frame.camera.render_camera.eye_position,
                    height: gp.height,
                    colour: gp.colour,
                    shadow_colour: gp.shadow_colour,
                    light_vp: gp_cascade0_mat,
                    tan_half_fov,
                    aspect,
                    tile_size: gp.tile_size,
                    shadow_bias: 0.002,
                    mode: mode_u32,
                    shadow_opacity: gp.shadow_opacity,
                    _pad: [0.0; 2],
                    colour2: gp.tile_colour2,
                };
                queue.write_buffer(
                    &resources.ground_plane_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[gp_uniform]),
                );
            }
        } // `resources` mutable borrow dropped here.

        // ------------------------------------------------------------------
        // Build per-viewport interaction state into local variables.
        // Uses &self.resources (immutable) for BGL lookups; no conflict with
        // the slot borrow that follows.
        // ------------------------------------------------------------------

        let vp_idx = frame.camera.viewport_index;

        // Outline mask buffers for selected objects (one per selected object).
        let mut outline_object_buffers: Vec<OutlineObjectBuffers> = Vec::new();
        if frame.interaction.outline_selected {
            let resources = &self.resources;
            for item in scene_items {
                if item.appearance.hidden || !item.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    colour: [0.0; 4], // unused by mask shader
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                };
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("outline_mask_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("outline_mask_object_bg"),
                    layout: &resources.outline_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.mesh_id,
                    two_sided: item.material.is_two_sided(),
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                });
            }
            // Selected transparent volume meshes: use their boundary surface for the outline.
            for item in &frame.scene.transparent_volume_meshes {
                if !item.selected {
                    continue;
                }
                let Some(mesh_id) = item.boundary_mesh_id else {
                    continue;
                };
                let uniform = OutlineUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                };
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("outline_mask_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("outline_mask_object_bg"),
                    layout: &resources.outline_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id,
                    two_sided: false,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                });
            }
            // Selected volume surface slices: use their mesh directly.
            for item in &frame.scene.volume_surface_slices {
                if !item.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                };
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("outline_mask_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("outline_mask_object_bg"),
                    layout: &resources.outline_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.mesh_id,
                    two_sided: true,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                });
            }
        }

        // Splat outline buffers: point sprite discs for selected Gaussian splat sets.
        let mut splat_outline_buffers: Vec<crate::resources::SplatOutlineBuffers> = Vec::new();
        // Curve mesh outline items: streamtubes, tubes, ribbons rendered via outline_mask_pipeline.
        let mut streamtube_outline_items: Vec<CurveMeshOutlineItem> = Vec::new();
        let mut tube_outline_items: Vec<CurveMeshOutlineItem> = Vec::new();
        let mut ribbon_outline_items: Vec<CurveMeshOutlineItem> = Vec::new();
        // Polyline outline indices: indices into polyline_gpu_data for selected polylines.
        let mut polyline_outline_indices: Vec<usize> = Vec::new();
        // Each entry is (gpu_data_index, instance_ranges).
        // None = draw all instances (object-level selection).
        // Some(vec) = draw only these specific instance indices (sub-object Instance selection).
        let mut glyph_outline_indices: Vec<(usize, Option<Vec<u32>>)> = Vec::new();
        let mut tensor_glyph_outline_indices: Vec<(usize, Option<Vec<u32>>)> = Vec::new();
        let mut sprite_outline_indices: Vec<(usize, Option<Vec<u32>>)> = Vec::new();
        if frame.interaction.outline_selected {
            let resources = &self.resources;
            let view_proj = frame.camera.render_camera.view_proj();
            let [vp_w, vp_h] = frame.camera.viewport_size;
            for item in &frame.scene.gaussian_splats {
                let Some(gpu_set) = resources.gaussian_splat_store.get(item.id.0) else {
                    continue;
                };
                if item.selected && !gpu_set.cpu_positions.is_empty() {
                    // Object-level: outline all splats.
                    // World-space radius covering the visible Gaussian tail (~3 sigma).
                    let mean_max_scale: f32 = if gpu_set.cpu_scales.is_empty() {
                        0.05
                    } else {
                        gpu_set
                            .cpu_scales
                            .iter()
                            .map(|s| s[0].max(s[1]).max(s[2]))
                            .sum::<f32>()
                            / gpu_set.cpu_scales.len() as f32
                    };
                    let world_radius = mean_max_scale * 3.0;

                    // Project the world radius to a pixel half-size at the cloud center.
                    // Use the camera right vector so the offset is always perpendicular
                    // to the view direction, avoiding the collapse when looking along X.
                    let model = glam::Mat4::from_cols_array_2d(&item.model);
                    let center_w = model.transform_point3(glam::Vec3::ZERO);
                    let cam_right = frame
                        .camera
                        .render_camera
                        .view
                        .row(0)
                        .truncate()
                        .normalize();
                    let p0_clip =
                        view_proj * glam::Vec4::new(center_w.x, center_w.y, center_w.z, 1.0);
                    let p1_world = center_w + cam_right * world_radius;
                    let p1_clip =
                        view_proj * glam::Vec4::new(p1_world.x, p1_world.y, p1_world.z, 1.0);
                    let pixel_radius = if p0_clip.w.abs() > 1e-6 && p1_clip.w.abs() > 1e-6 {
                        let p0_ndc = glam::Vec2::new(p0_clip.x, p0_clip.y) / p0_clip.w;
                        let p1_ndc = glam::Vec2::new(p1_clip.x, p1_clip.y) / p1_clip.w;
                        (p1_ndc - p0_ndc).length() * 0.5 * vp_w.max(vp_h)
                    } else {
                        world_radius * 100.0
                    };
                    let pixel_radius = pixel_radius.max(1.0);

                    let position_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("splat_outline_pos_buf"),
                            contents: bytemuck::cast_slice(gpu_set.cpu_positions.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    let uniform = SplatOutlineMaskUniform {
                        model: item.model,
                        viewport_w: vp_w,
                        viewport_h: vp_h,
                        pixel_radius,
                        _pad: [0.0; 5],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("splat_outline_uniform_buf"),
                            contents: bytemuck::cast_slice(&[uniform]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("splat_outline_bg"),
                        layout: &resources.outline_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });

                    let n = gpu_set.cpu_positions.len();
                    let size_data: Vec<f32> = vec![pixel_radius; n];
                    let size_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("splat_outline_size_buf"),
                        contents: bytemuck::cast_slice(&size_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

                    splat_outline_buffers.push(crate::resources::SplatOutlineBuffers {
                        position_buf,
                        size_buf,
                        instance_count: n as u32,
                        _uniform_buf: uniform_buf,
                        bind_group,
                    });
                } else if !item.selected && item.pick_id != 0 {
                    // Per-splat sub-selection: outline only the selected splats.
                    let sub_sel = frame.interaction.sub_selection.as_ref();
                    let selected_indices: Vec<u32> = sub_sel
                        .iter()
                        .flat_map(|s| s.items.iter())
                        .filter_map(|(node_id, sub)| {
                            if *node_id == item.pick_id {
                                if let crate::interaction::sub_object::SubObjectRef::Splat(i) = sub
                                {
                                    return Some(*i);
                                }
                            }
                            None
                        })
                        .collect();
                    if selected_indices.is_empty() {
                        continue;
                    }

                    let model = glam::Mat4::from_cols_array_2d(&item.model);
                    let cam_right = frame
                        .camera
                        .render_camera
                        .view
                        .row(0)
                        .truncate()
                        .normalize();

                    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(selected_indices.len());
                    let mut sizes: Vec<f32> = Vec::with_capacity(selected_indices.len());
                    for &idx in &selected_indices {
                        let i = idx as usize;
                        if let Some(&pos) = gpu_set.cpu_positions.get(i) {
                            positions.push(pos);
                            let world_radius = if let Some(s) = gpu_set.cpu_scales.get(i) {
                                s[0].max(s[1]).max(s[2]) * 3.0
                            } else {
                                0.15
                            };
                            let center_w = model.transform_point3(glam::Vec3::from(pos));
                            let p0_clip = view_proj
                                * glam::Vec4::new(center_w.x, center_w.y, center_w.z, 1.0);
                            let p1_world = center_w + cam_right * world_radius;
                            let p1_clip = view_proj
                                * glam::Vec4::new(p1_world.x, p1_world.y, p1_world.z, 1.0);
                            let px = if p0_clip.w.abs() > 1e-6 && p1_clip.w.abs() > 1e-6 {
                                let p0_ndc = glam::Vec2::new(p0_clip.x, p0_clip.y) / p0_clip.w;
                                let p1_ndc = glam::Vec2::new(p1_clip.x, p1_clip.y) / p1_clip.w;
                                ((p1_ndc - p0_ndc).length() * 0.5 * vp_w.max(vp_h)).max(1.0)
                            } else {
                                world_radius * 100.0
                            };
                            sizes.push(px);
                        }
                    }
                    if positions.is_empty() {
                        continue;
                    }

                    let pixel_radius = sizes
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max)
                        .max(1.0);
                    let uniform = SplatOutlineMaskUniform {
                        model: item.model,
                        viewport_w: vp_w,
                        viewport_h: vp_h,
                        pixel_radius,
                        _pad: [0.0; 5],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("splat_sel_outline_uniform_buf"),
                            contents: bytemuck::cast_slice(&[uniform]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("splat_sel_outline_bg"),
                        layout: &resources.outline_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });
                    let position_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("splat_sel_outline_pos_buf"),
                            contents: bytemuck::cast_slice(&positions),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let size_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("splat_sel_outline_size_buf"),
                        contents: bytemuck::cast_slice(&sizes),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    splat_outline_buffers.push(crate::resources::SplatOutlineBuffers {
                        position_buf,
                        size_buf,
                        instance_count: positions.len() as u32,
                        _uniform_buf: uniform_buf,
                        bind_group,
                    });
                }
            }

            // Point cloud outline buffers: reuse the same point sprite mask pipeline.
            for item in &frame.scene.point_clouds {
                if item.positions.is_empty() {
                    continue;
                }
                let pixel_radius = (item.point_size * 0.5).max(1.0);
                if item.selected {
                    // Object-level: outline all points.
                    let position_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("pc_outline_pos_buf"),
                            contents: bytemuck::cast_slice(item.positions.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let uniform = SplatOutlineMaskUniform {
                        model: item.model,
                        viewport_w: vp_w,
                        viewport_h: vp_h,
                        pixel_radius,
                        _pad: [0.0; 5],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("pc_outline_uniform_buf"),
                            contents: bytemuck::cast_slice(&[uniform]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pc_outline_bg"),
                        layout: &self.resources.outline_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });
                    let n = item.positions.len();
                    let size_data: Vec<f32> = vec![pixel_radius; n];
                    let size_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("pc_outline_size_buf"),
                        contents: bytemuck::cast_slice(&size_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    splat_outline_buffers.push(crate::resources::SplatOutlineBuffers {
                        position_buf,
                        size_buf,
                        instance_count: n as u32,
                        _uniform_buf: uniform_buf,
                        bind_group,
                    });
                } else if item.id != 0 {
                    // Per-point sub-selection: outline only the selected points.
                    let sub_sel = frame.interaction.sub_selection.as_ref();
                    let selected_positions: Vec<[f32; 3]> = sub_sel
                        .iter()
                        .flat_map(|s| s.items.iter())
                        .filter_map(|(node_id, sub)| {
                            if *node_id == item.id {
                                if let crate::interaction::sub_object::SubObjectRef::Point(i) = sub
                                {
                                    return item.positions.get(*i as usize).copied();
                                }
                            }
                            None
                        })
                        .collect();
                    if selected_positions.is_empty() {
                        continue;
                    }
                    let n = selected_positions.len();
                    let uniform = SplatOutlineMaskUniform {
                        model: item.model,
                        viewport_w: vp_w,
                        viewport_h: vp_h,
                        pixel_radius,
                        _pad: [0.0; 5],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("pc_sel_outline_uniform_buf"),
                            contents: bytemuck::cast_slice(&[uniform]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pc_sel_outline_bg"),
                        layout: &self.resources.outline_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });
                    let position_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("pc_sel_outline_pos_buf"),
                            contents: bytemuck::cast_slice(&selected_positions),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let size_data = vec![pixel_radius; n];
                    let size_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("pc_sel_outline_size_buf"),
                        contents: bytemuck::cast_slice(&size_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    splat_outline_buffers.push(crate::resources::SplatOutlineBuffers {
                        position_buf,
                        size_buf,
                        instance_count: n as u32,
                        _uniform_buf: uniform_buf,
                        bind_group,
                    });
                }
            }

            // Glyph outline indices: record which glyph GPU data entries are selected
            // so the mask pass can render the actual instanced mesh.
            {
                let sub_sel = frame.interaction.sub_selection.as_ref();
                let mut gpu_idx = 0usize;
                for item in &frame.scene.glyphs {
                    if item.positions.is_empty() || item.vectors.is_empty() {
                        continue;
                    }
                    if item.selected {
                        self.resources.ensure_glyph_outline_mask_pipeline(device);
                        glyph_outline_indices.push((gpu_idx, None));
                    } else if item.id != 0 {
                        // Check for per-instance sub-selection.
                        let instances: Vec<u32> = sub_sel
                            .iter()
                            .flat_map(|s| s.items.iter())
                            .filter_map(|(node_id, sub)| {
                                if *node_id == item.id {
                                    if let crate::interaction::sub_object::SubObjectRef::Instance(
                                        i,
                                    ) = sub
                                    {
                                        return Some(*i);
                                    }
                                }
                                None
                            })
                            .collect();
                        if !instances.is_empty() {
                            self.resources.ensure_glyph_outline_mask_pipeline(device);
                            glyph_outline_indices.push((gpu_idx, Some(instances)));
                        }
                    }
                    gpu_idx += 1;
                }
            }

            // Polyline outlines: collect indices of selected polylines so the mask
            // pass can draw their segment quads via the polyline_outline_mask_pipeline.
            if !self.polyline_selected_gpu_indices.is_empty() {
                self.resources.ensure_polyline_outline_mask_pipeline(device);
                polyline_outline_indices = self.polyline_selected_gpu_indices.clone();
            }

            // Sprite outline indices: record which sprite GPU data entries are selected
            // so the mask pass can render the actual billboard quads.
            {
                let sub_sel = frame.interaction.sub_selection.as_ref();
                for (i, item) in frame.scene.sprite_items.iter().enumerate() {
                    if item.positions.is_empty() {
                        continue;
                    }
                    if item.selected {
                        self.resources.ensure_sprite_outline_mask_pipeline(device);
                        sprite_outline_indices.push((i, None));
                    } else if item.id != 0 {
                        let instances: Vec<u32> = sub_sel
                            .iter()
                            .flat_map(|s| s.items.iter())
                            .filter_map(|(node_id, sub)| {
                                if *node_id == item.id {
                                    if let crate::interaction::sub_object::SubObjectRef::Instance(
                                        idx,
                                    ) = sub
                                    {
                                        return Some(*idx);
                                    }
                                }
                                None
                            })
                            .collect();
                        if !instances.is_empty() {
                            self.resources.ensure_sprite_outline_mask_pipeline(device);
                            sprite_outline_indices.push((i, Some(instances)));
                        }
                    }
                }
            }

            // Streamtube / Tube / Ribbon outline items: use the actual triangle mesh
            // geometry so the depth-buffer edge detection follows the tube silhouette.
            let make_curve_item = |index: usize, two_sided: bool| -> CurveMeshOutlineItem {
                let uniform = crate::resources::OutlineUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    colour: [1.0, 1.0, 1.0, 1.0],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                };
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("curve_outline_uniform_buf"),
                    contents: bytemuck::cast_slice(&[uniform]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("curve_outline_mask_bg"),
                    layout: &self.resources.outline_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                CurveMeshOutlineItem {
                    index,
                    two_sided,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                }
            };

            for &idx in &self.streamtube_selected_gpu_indices {
                streamtube_outline_items.push(make_curve_item(idx, false));
            }
            for &idx in &self.tube_selected_gpu_indices {
                tube_outline_items.push(make_curve_item(idx, false));
            }
            for &idx in &self.ribbon_selected_gpu_indices {
                ribbon_outline_items.push(make_curve_item(idx, true));
            }


            // Tensor glyph outline indices: same approach as arrow glyphs.
            {
                let sub_sel = frame.interaction.sub_selection.as_ref();
                let mut gpu_idx = 0usize;
                for item in &frame.scene.tensor_glyphs {
                    if item.positions.is_empty() {
                        continue;
                    }
                    if item.selected {
                        self.resources
                            .ensure_tensor_glyph_outline_mask_pipeline(device);
                        tensor_glyph_outline_indices.push((gpu_idx, None));
                    } else if item.id != 0 {
                        let instances: Vec<u32> = sub_sel
                            .iter()
                            .flat_map(|s| s.items.iter())
                            .filter_map(|(node_id, sub)| {
                                if *node_id == item.id {
                                    if let crate::interaction::sub_object::SubObjectRef::Instance(
                                        i,
                                    ) = sub
                                    {
                                        return Some(*i);
                                    }
                                }
                                None
                            })
                            .collect();
                        if !instances.is_empty() {
                            self.resources
                                .ensure_tensor_glyph_outline_mask_pipeline(device);
                            tensor_glyph_outline_indices.push((gpu_idx, Some(instances)));
                        }
                    }
                    gpu_idx += 1;
                }
            }
        }

        // Volume outline: record indices of selected volumes so the mask pass can
        // reuse their VolumeGpuData bind groups (which already contain model, 3D
        // texture, samplers, and LUTs needed by the ray-march mask shader).
        let mut volume_outline_indices: Vec<usize> = Vec::new();
        if frame.interaction.outline_selected {
            self.resources.ensure_volume_cube(device);
            self.resources.ensure_volume_pipeline(device);
            self.resources.ensure_volume_outline_mask_pipeline(device);
            for (i, item) in frame.scene.volumes.iter().enumerate() {
                if item.selected {
                    volume_outline_indices.push(i);
                }
            }
        }

        // Image slice outlines: compute world-space quad corners and create inline vertex/index buffers.
        let mut raw_geom_outline_buffers: Vec<crate::resources::RawGeomOutlineBuffers> = Vec::new();
        if frame.interaction.outline_selected {
            let resources = &self.resources;
            for item in &frame.scene.image_slices {
                if !item.selected {
                    continue;
                }
                use crate::SliceAxis;
                let [bmin, bmax] = [item.bbox_min, item.bbox_max];
                let t = item.offset;
                let (v0, v1, v2, v3) = match item.axis {
                    SliceAxis::X => {
                        let x = bmin[0] + t * (bmax[0] - bmin[0]);
                        (
                            [x, bmin[1], bmin[2]],
                            [x, bmax[1], bmin[2]],
                            [x, bmax[1], bmax[2]],
                            [x, bmin[1], bmax[2]],
                        )
                    }
                    SliceAxis::Y => {
                        let y = bmin[1] + t * (bmax[1] - bmin[1]);
                        (
                            [bmin[0], y, bmin[2]],
                            [bmax[0], y, bmin[2]],
                            [bmax[0], y, bmax[2]],
                            [bmin[0], y, bmax[2]],
                        )
                    }
                    SliceAxis::Z => {
                        let z = bmin[2] + t * (bmax[2] - bmin[2]);
                        (
                            [bmin[0], bmin[1], z],
                            [bmax[0], bmin[1], z],
                            [bmax[0], bmax[1], z],
                            [bmin[0], bmax[1], z],
                        )
                    }
                };
                let verts: [[f32; 3]; 4] = [v0, v1, v2, v3];
                let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
                let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("image_slice_outline_verts"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("image_slice_outline_indices"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let uniform = OutlineUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                };
                let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("outline_mask_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("outline_mask_object_bg"),
                    layout: &resources.outline_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    }],
                });
                raw_geom_outline_buffers.push(crate::resources::RawGeomOutlineBuffers {
                    vertex_buf,
                    index_buf,
                    index_count: 6,
                    two_sided: true,
                    _uniform_buf: uniform_buf,
                    mask_bind_group: bg,
                });
            }
        }

        // Screen image outlines: compute NDC bounds and create outline buffers.
        let mut screen_rect_outline_buffers: Vec<crate::resources::ScreenRectOutlineBuffers> =
            Vec::new();
        if frame.interaction.outline_selected
            && frame.scene.screen_images.iter().any(|i| i.selected)
        {
            self.resources
                .ensure_screen_rect_outline_mask_pipeline(device);
            let [vp_w, vp_h] = frame.camera.viewport_size;
            if let Some(bgl) = self.resources.screen_rect_outline_bgl.as_ref() {
                for item in &frame.scene.screen_images {
                    if !item.selected || item.width == 0 || item.height == 0 {
                        continue;
                    }
                    use crate::ImageAnchor;
                    let img_w_ndc = 2.0 * item.width as f32 * item.scale / vp_w.max(1.0);
                    let img_h_ndc = 2.0 * item.height as f32 * item.scale / vp_h.max(1.0);
                    let (ndc_min_x, ndc_max_x, ndc_min_y, ndc_max_y) = match item.anchor {
                        ImageAnchor::TopLeft => (-1.0, -1.0 + img_w_ndc, 1.0 - img_h_ndc, 1.0),
                        ImageAnchor::TopRight => (1.0 - img_w_ndc, 1.0, 1.0 - img_h_ndc, 1.0),
                        ImageAnchor::BottomLeft => (-1.0, -1.0 + img_w_ndc, -1.0, -1.0 + img_h_ndc),
                        ImageAnchor::BottomRight => (1.0 - img_w_ndc, 1.0, -1.0, -1.0 + img_h_ndc),
                        _ => (
                            -img_w_ndc * 0.5,
                            img_w_ndc * 0.5,
                            -img_h_ndc * 0.5,
                            img_h_ndc * 0.5,
                        ),
                    };
                    #[repr(C)]
                    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
                    struct NdcRectUniform {
                        ndc_min: [f32; 2],
                        ndc_max: [f32; 2],
                    }
                    let uniform_data = NdcRectUniform {
                        ndc_min: [ndc_min_x, ndc_min_y],
                        ndc_max: [ndc_max_x, ndc_max_y],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("screen_rect_outline_uniform"),
                            contents: bytemuck::bytes_of(&uniform_data),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("screen_rect_outline_bg"),
                        layout: bgl,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });
                    screen_rect_outline_buffers.push(crate::resources::ScreenRectOutlineBuffers {
                        _uniform_buf: uniform_buf,
                        bind_group: bg,
                    });
                }
            }
        }

        // Implicit surface outlines: record indices into implicit_gpu_data for selected items.
        let mut implicit_outline_indices: Vec<usize> = Vec::new();
        if frame.interaction.outline_selected {
            let mut gpu_idx = 0usize;
            for item in &frame.scene.gpu_implicit {
                if item.primitives.is_empty() {
                    continue;
                }
                if item.selected {
                    self.resources.ensure_implicit_pipeline(device);
                    self.resources.ensure_implicit_outline_mask_pipeline(device);
                    implicit_outline_indices.push(gpu_idx);
                }
                gpu_idx += 1;
            }
        }

        // MC surface outlines: build per-job outline uniform + bind group.
        let mut mc_outline_data: Vec<crate::resources::gpu_marching_cubes::McOutlineItem> =
            Vec::new();
        if frame.interaction.outline_selected {
            for (i, job) in frame.scene.gpu_mc_jobs.iter().enumerate() {
                if !job.selected {
                    continue;
                }
                self.resources.ensure_mc_pipelines(device);
                self.resources.ensure_mc_outline_mask_pipeline(device);
                let uniform = OutlineUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                };
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc_outline_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mc_outline_bg"),
                    layout: &self.resources.outline_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                mc_outline_data.push(crate::resources::gpu_marching_cubes::McOutlineItem {
                    mc_gpu_idx: i,
                    _uniform_buf: buf,
                    mask_bind_group: bg,
                });
            }
        }

        // X-ray buffers for selected objects.
        let mut xray_object_buffers: Vec<(
            crate::resources::mesh_store::MeshId,
            wgpu::Buffer,
            wgpu::BindGroup,
        )> = Vec::new();
        if frame.interaction.xray_selected {
            let resources = &self.resources;
            for item in scene_items {
                if item.appearance.hidden || !item.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    colour: frame.interaction.xray_colour,
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
                xray_object_buffers.push((item.mesh_id, buf, bg));
            }
        }

        // Constraint guide lines.
        let mut constraint_line_buffers = Vec::new();
        for overlay in &frame.interaction.constraint_overlays {
            constraint_line_buffers.push(self.resources.create_constraint_overlay(device, overlay));
        }

        // Clip plane overlays : generated automatically from clip_objects with a colour set.
        let mut clip_plane_fill_buffers = Vec::new();
        let mut clip_plane_line_buffers = Vec::new();
        for obj in viewport_fx.clip_objects.iter().filter(|o| o.enabled) {
            // Skip if neither fill nor edge colour is set.
            if obj.colour.is_none() && obj.edge_colour.is_none() {
                continue;
            }
            if let ClipShape::Plane {
                normal,
                distance,
                display_center,
                ..
            } = obj.shape
            {
                let n = glam::Vec3::from(normal);
                // Use the caller-supplied display_center when available so that
                // lateral translations (tangent to the plane) are reflected in
                // the overlay quad position.  Fall back to the foot-of-normal
                // from the world origin when none is set.
                let center = display_center
                    .map(glam::Vec3::from)
                    .unwrap_or_else(|| n * (-distance));
                let active = obj.active;
                let hovered = obj.hovered || active;

                // Fill quad: derived from `colour`; transparent if not set.
                let fill_colour = if let Some(base_colour) = obj.colour {
                    if active {
                        [
                            base_colour[0] * 0.5,
                            base_colour[1] * 0.5,
                            base_colour[2] * 0.5,
                            base_colour[3] * 0.5,
                        ]
                    } else if hovered {
                        [
                            base_colour[0] * 0.8,
                            base_colour[1] * 0.8,
                            base_colour[2] * 0.8,
                            base_colour[3] * 0.6,
                        ]
                    } else {
                        [
                            base_colour[0] * 0.5,
                            base_colour[1] * 0.5,
                            base_colour[2] * 0.5,
                            base_colour[3] * 0.3,
                        ]
                    }
                } else {
                    [0.0, 0.0, 0.0, 0.0]
                };

                // Border edge: use `edge_colour` when set, otherwise derive from `colour`.
                let border_base = obj.edge_colour.or(obj.colour).unwrap_or([1.0, 1.0, 1.0, 1.0]);
                let border_colour = if active {
                    [border_base[0], border_base[1], border_base[2], 0.9]
                } else if hovered {
                    [border_base[0], border_base[1], border_base[2], 0.8]
                } else {
                    [
                        border_base[0] * 0.9,
                        border_base[1] * 0.9,
                        border_base[2] * 0.9,
                        0.6,
                    ]
                };

                let overlay = crate::interaction::clip_plane::ClipPlaneOverlay {
                    center,
                    normal: n,
                    extent: obj.extent,
                    fill_colour,
                    border_colour,
                    _hovered: hovered,
                    _active: active,
                };
                if obj.colour.is_some() {
                    clip_plane_fill_buffers.push(
                        self.resources
                            .create_clip_plane_fill_overlay(device, &overlay),
                    );
                }
                clip_plane_line_buffers.push(
                    self.resources
                        .create_clip_plane_line_overlay(device, &overlay),
                );
            } else {
                // Box/Sphere/Cylinder: generate wireframe polyline overlay.
                // These use the clip-exempt pipeline so the outline is always fully visible,
                // even when multiple clip volumes are active (the user needs to see where each
                // clip is positioned to understand the combined result).
                let base_colour = obj.colour.unwrap_or([1.0, 1.0, 1.0, 1.0]);
                self.resources.ensure_polyline_no_clip_pipeline(device);
                match obj.shape {
                    ClipShape::Box {
                        center,
                        half_extents,
                        orientation,
                    } => {
                        let polyline =
                            clip_box_outline(center, half_extents, orientation, base_colour);
                        let vp_size = frame.camera.viewport_size;
                        let mut gpu = self
                            .resources
                            .upload_polyline(device, queue, &polyline, vp_size);
                        gpu.skip_clip = true;
                        self.polyline_gpu_data.push(gpu);
                    }
                    ClipShape::Sphere { center, radius } => {
                        let polyline = clip_sphere_outline(center, radius, base_colour);
                        let vp_size = frame.camera.viewport_size;
                        let mut gpu = self
                            .resources
                            .upload_polyline(device, queue, &polyline, vp_size);
                        gpu.skip_clip = true;
                        self.polyline_gpu_data.push(gpu);
                    }
                    ClipShape::Cylinder {
                        center,
                        axis,
                        radius,
                        half_length,
                    } => {
                        let polyline =
                            clip_cylinder_outline(center, axis, radius, half_length, base_colour);
                        let vp_size = frame.camera.viewport_size;
                        let mut gpu = self
                            .resources
                            .upload_polyline(device, queue, &polyline, vp_size);
                        gpu.skip_clip = true;
                        self.polyline_gpu_data.push(gpu);
                    }
                    _ => {}
                }
            }
        }

        // Cap geometry for section-view cross-section fill.
        let mut cap_buffers = Vec::new();
        if viewport_fx.cap_fill_enabled {
            for obj in viewport_fx.clip_objects.iter().filter(|o| o.enabled) {
                if let ClipShape::Plane {
                    normal,
                    distance,
                    cap_colour,
                    ..
                } = obj.shape
                {
                    let plane_n = glam::Vec3::from(normal);
                    for item in scene_items.iter().filter(|i| !i.appearance.hidden) {
                        let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                            continue;
                        };
                        let model = glam::Mat4::from_cols_array_2d(&item.model);
                        let world_aabb = mesh.aabb.transformed(&model);
                        if !world_aabb.intersects_plane(plane_n, distance) {
                            continue;
                        }
                        let (Some(pos), Some(idx)) = (&mesh.cpu_positions, &mesh.cpu_indices)
                        else {
                            continue;
                        };
                        if let Some(cap) = crate::geometry::cap_geometry::generate_cap_mesh(
                            pos, idx, &model, plane_n, distance,
                        ) {
                            let bc = item.material.base_colour;
                            let colour = cap_colour.unwrap_or([bc[0], bc[1], bc[2], 1.0]);
                            let buf = self.resources.upload_cap_geometry(device, &cap, colour);
                            cap_buffers.push(buf);
                        }
                    }
                }
            }
        }

        // Axes indicator geometry (built here, written to slot buffer below).
        let axes_verts = if frame.viewport.show_axes_indicator
            && frame.camera.viewport_size[0] > 0.0
            && frame.camera.viewport_size[1] > 0.0
        {
            let verts = crate::widgets::axes_indicator::build_axes_geometry(
                frame.camera.viewport_size[0],
                frame.camera.viewport_size[1],
                frame.camera.render_camera.orientation,
            );
            if verts.is_empty() { None } else { Some(verts) }
        } else {
            None
        };

        // Gizmo mesh + uniform (built here, written to slot buffers below).
        let gizmo_update = frame.interaction.gizmo_model.map(|model| {
            let (verts, indices) = crate::interaction::gizmo::build_gizmo_mesh(
                frame.interaction.gizmo_mode,
                frame.interaction.gizmo_hovered,
                frame.interaction.gizmo_space_orientation,
            );
            (verts, indices, model)
        });

        // ------------------------------------------------------------------
        // Assign all interaction state to the per-viewport slot.
        // ------------------------------------------------------------------
        {
            let slot = &mut self.viewport_slots[vp_idx];
            slot.outline_object_buffers = outline_object_buffers;
            slot.splat_outline_buffers = splat_outline_buffers;
            slot.streamtube_outline_items = streamtube_outline_items;
            slot.tube_outline_items = tube_outline_items;
            slot.ribbon_outline_items = ribbon_outline_items;
            slot.polyline_outline_indices = polyline_outline_indices;
            slot.volume_outline_indices = volume_outline_indices;
            slot.glyph_outline_indices = glyph_outline_indices;
            slot.tensor_glyph_outline_indices = tensor_glyph_outline_indices;
            slot.sprite_outline_indices = sprite_outline_indices;
            slot.raw_geom_outline_buffers = raw_geom_outline_buffers;
            slot.screen_rect_outline_buffers = screen_rect_outline_buffers;
            slot.implicit_outline_indices = implicit_outline_indices;
            slot.mc_outline_data = mc_outline_data;
            slot.xray_object_buffers = xray_object_buffers;
            slot.constraint_line_buffers = constraint_line_buffers;
            slot.clip_plane_fill_buffers = clip_plane_fill_buffers;
            slot.clip_plane_line_buffers = clip_plane_line_buffers;
            slot.cap_buffers = cap_buffers;

            // Axes: resize buffer if needed, then upload.
            if let Some(verts) = axes_verts {
                let byte_size = std::mem::size_of_val(verts.as_slice()) as u64;
                if byte_size > slot.axes_vertex_buffer.size() {
                    slot.axes_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("vp_axes_vertex_buf"),
                        size: byte_size,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                queue.write_buffer(&slot.axes_vertex_buffer, 0, bytemuck::cast_slice(&verts));
                slot.axes_vertex_count = verts.len() as u32;
            } else {
                slot.axes_vertex_count = 0;
            }

            // Gizmo: resize buffers if needed, then upload mesh + uniform.
            if let Some((verts, indices, model)) = gizmo_update {
                let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
                let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);
                if vert_bytes.len() as u64 > slot.gizmo_vertex_buffer.size() {
                    slot.gizmo_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("vp_gizmo_vertex_buf"),
                        size: vert_bytes.len() as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                if idx_bytes.len() as u64 > slot.gizmo_index_buffer.size() {
                    slot.gizmo_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("vp_gizmo_index_buf"),
                        size: idx_bytes.len() as u64,
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                queue.write_buffer(&slot.gizmo_vertex_buffer, 0, vert_bytes);
                queue.write_buffer(&slot.gizmo_index_buffer, 0, idx_bytes);
                slot.gizmo_index_count = indices.len() as u32;
                let uniform = crate::interaction::gizmo::GizmoUniform {
                    model: model.to_cols_array_2d(),
                };
                queue.write_buffer(&slot.gizmo_uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
            }
        }

        // ------------------------------------------------------------------
        // Outline offscreen pass : screen-space edge detection.
        //
        // 1. Render selected objects to an R8 mask texture (white on black).
        // 2. Run a fullscreen edge-detection pass reading the mask and writing
        //    an anti-aliased outline ring to the outline colour texture.
        //
        // The outline colour texture is later composited onto the main target
        // by the composite pass in paint()/render().
        // ------------------------------------------------------------------
        if frame.interaction.outline_selected
            && (!self.viewport_slots[vp_idx]
                .outline_object_buffers
                .is_empty()
                || !self.viewport_slots[vp_idx].splat_outline_buffers.is_empty()
                || !self.viewport_slots[vp_idx]
                    .streamtube_outline_items
                    .is_empty()
                || !self.viewport_slots[vp_idx].tube_outline_items.is_empty()
                || !self.viewport_slots[vp_idx].ribbon_outline_items.is_empty()
                || !self.viewport_slots[vp_idx]
                    .polyline_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .volume_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx].glyph_outline_indices.is_empty()
                || !self.viewport_slots[vp_idx]
                    .tensor_glyph_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .sprite_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .raw_geom_outline_buffers
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .screen_rect_outline_buffers
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .implicit_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx].mc_outline_data.is_empty())
        {
            let ppp = frame.camera.pixels_per_point;
            let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
            let h = (frame.camera.viewport_size[1] * ppp).round() as u32;

            // Ensure per-viewport HDR state exists (provides outline textures).
            self.ensure_viewport_hdr(
                device,
                queue,
                vp_idx,
                w.max(1),
                h.max(1),
                frame.effects.post_process.ssaa_factor.max(1),
                self.current_render_scale,
            );

            // Write edge-detection uniform (colour, radius, viewport size).
            {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let [scene_w, scene_h] = slot_hdr.scene_size;
                let edge_uniform = OutlineEdgeUniform {
                    colour: frame.interaction.outline_colour,
                    radius: frame.interaction.outline_width_px,
                    viewport_w: scene_w as f32,
                    viewport_h: scene_h as f32,
                    _pad: 0.0,
                };
                queue.write_buffer(
                    &slot_hdr.outline_edge_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[edge_uniform]),
                );
            }

            // Extract raw pointers for slot fields needed inside the render
            // passes alongside &self.resources borrows.
            let slot_ref = &self.viewport_slots[vp_idx];
            let outlines_ptr = &slot_ref.outline_object_buffers as *const Vec<OutlineObjectBuffers>;
            let splat_outlines_ptr = &slot_ref.splat_outline_buffers
                as *const Vec<crate::resources::SplatOutlineBuffers>;
            let streamtube_outline_items_ptr = &slot_ref.streamtube_outline_items
                as *const Vec<CurveMeshOutlineItem>;
            let tube_outline_items_ptr = &slot_ref.tube_outline_items
                as *const Vec<CurveMeshOutlineItem>;
            let ribbon_outline_items_ptr = &slot_ref.ribbon_outline_items
                as *const Vec<CurveMeshOutlineItem>;
            let polyline_outline_idx_ptr =
                &slot_ref.polyline_outline_indices as *const Vec<usize>;
            let vol_outline_idx_ptr = &slot_ref.volume_outline_indices as *const Vec<usize>;
            let glyph_outline_idx_ptr =
                &slot_ref.glyph_outline_indices as *const Vec<(usize, Option<Vec<u32>>)>;
            let tensor_glyph_outline_idx_ptr =
                &slot_ref.tensor_glyph_outline_indices as *const Vec<(usize, Option<Vec<u32>>)>;
            let sprite_outline_idx_ptr =
                &slot_ref.sprite_outline_indices as *const Vec<(usize, Option<Vec<u32>>)>;
            let raw_geom_outlines_ptr = &slot_ref.raw_geom_outline_buffers
                as *const Vec<crate::resources::RawGeomOutlineBuffers>;
            let screen_rect_outlines_ptr = &slot_ref.screen_rect_outline_buffers
                as *const Vec<crate::resources::ScreenRectOutlineBuffers>;
            let implicit_outline_idx_ptr = &slot_ref.implicit_outline_indices as *const Vec<usize>;
            let mc_outlines_ptr = &slot_ref.mc_outline_data
                as *const Vec<crate::resources::gpu_marching_cubes::McOutlineItem>;
            let glyph_gpu_ptr = &self.glyph_gpu_data as *const Vec<crate::resources::GlyphGpuData>;
            let tensor_glyph_gpu_ptr =
                &self.tensor_glyph_gpu_data as *const Vec<crate::resources::TensorGlyphGpuData>;
            let sprite_gpu_ptr =
                &self.sprite_gpu_data as *const Vec<crate::resources::SpriteGpuData>;
            let streamtube_gpu_ptr =
                &self.streamtube_gpu_data as *const Vec<crate::resources::StreamtubeGpuData>;
            let tube_gpu_ptr =
                &self.tube_gpu_data as *const Vec<crate::resources::StreamtubeGpuData>;
            let ribbon_gpu_ptr =
                &self.ribbon_gpu_data as *const Vec<crate::resources::StreamtubeGpuData>;
            let polyline_gpu_ptr =
                &self.polyline_gpu_data as *const Vec<crate::resources::PolylineGpuData>;
            let implicit_gpu_ptr =
                &self.implicit_gpu_data as *const Vec<crate::resources::implicit::ImplicitGpuItem>;
            let mc_gpu_data_ptr =
                &self.mc_gpu_data as *const Vec<crate::resources::gpu_marching_cubes::McFrameData>;
            let camera_bg_ptr = &slot_ref.camera_bind_group as *const wgpu::BindGroup;
            let slot_hdr = slot_ref.hdr.as_ref().unwrap();
            let mask_view_ptr = &slot_hdr.outline_mask_view as *const wgpu::TextureView;
            let colour_view_ptr = &slot_hdr.outline_colour_view as *const wgpu::TextureView;
            let depth_view_ptr = &slot_hdr.outline_depth_view as *const wgpu::TextureView;
            let edge_bg_ptr = &slot_hdr.outline_edge_bind_group as *const wgpu::BindGroup;
            // SAFETY: slot fields remain valid for the duration of this function;
            // no other code modifies these fields here.
            let (
                outlines,
                splat_outlines,
                streamtube_outline_items,
                tube_outline_items,
                ribbon_outline_items,
                polyline_outline_idxs,
                vol_outline_indices,
                glyph_outline_indices,
                tensor_glyph_outline_indices,
                sprite_outline_indices,
                raw_geom_outlines,
                screen_rect_outlines,
                implicit_outline_idxs,
                mc_outlines,
                glyph_gpu_data,
                tensor_glyph_gpu_data,
                sprite_gpu_data,
                streamtube_gpu_data,
                tube_gpu_data,
                ribbon_gpu_data,
                polyline_gpu_data,
                implicit_gpu_data,
                mc_gpu_frame_data,
                camera_bg,
                mask_view,
                colour_view,
                depth_view,
                edge_bg,
            ) = unsafe {
                (
                    &*outlines_ptr,
                    &*splat_outlines_ptr,
                    &*streamtube_outline_items_ptr,
                    &*tube_outline_items_ptr,
                    &*ribbon_outline_items_ptr,
                    &*polyline_outline_idx_ptr,
                    &*vol_outline_idx_ptr,
                    &*glyph_outline_idx_ptr,
                    &*tensor_glyph_outline_idx_ptr,
                    &*sprite_outline_idx_ptr,
                    &*raw_geom_outlines_ptr,
                    &*screen_rect_outlines_ptr,
                    &*implicit_outline_idx_ptr,
                    &*mc_outlines_ptr,
                    &*glyph_gpu_ptr,
                    &*tensor_glyph_gpu_ptr,
                    &*sprite_gpu_ptr,
                    &*streamtube_gpu_ptr,
                    &*tube_gpu_ptr,
                    &*ribbon_gpu_ptr,
                    &*polyline_gpu_ptr,
                    &*implicit_gpu_ptr,
                    &*mc_gpu_data_ptr,
                    &*camera_bg_ptr,
                    &*mask_view_ptr,
                    &*colour_view_ptr,
                    &*depth_view_ptr,
                    &*edge_bg_ptr,
                )
            };

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("outline_offscreen_encoder"),
            });

            // Pass 1: render selected objects to R8 mask texture.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("outline_mask_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: mask_view,
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
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_bind_group(0, camera_bg, &[]);
                for outlined in outlines {
                    let Some(mesh) = self.resources.mesh_store.get(outlined.mesh_id) else {
                        continue;
                    };
                    let pipeline = if outlined.two_sided {
                        &self.resources.outline_mask_two_sided_pipeline
                    } else {
                        &self.resources.outline_mask_pipeline
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(1, &outlined.mask_bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }

                // Draw Gaussian splat outline discs.  Each splat position expands to
                // a screen-space disc in the vertex shader (6 vertices per instance).
                // Depth is tested (splats behind selected meshes are culled) but not
                // written, so all visible splats in a cloud contribute to the mask.
                pass.set_pipeline(&self.resources.splat_outline_mask_pipeline);
                for splat in splat_outlines {
                    pass.set_bind_group(1, &splat.bind_group, &[]);
                    pass.set_vertex_buffer(0, splat.position_buf.slice(..));
                    pass.set_vertex_buffer(1, splat.size_buf.slice(..));
                    pass.draw(0..6, 0..splat.instance_count);
                }

                // Draw glyph instances into the mask using the actual instanced
                // mesh geometry so the outline follows arrow/sphere shapes.
                if !glyph_outline_indices.is_empty() {
                    if let Some(pipeline) = self.resources.glyph_outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        for (idx, instance_filter) in glyph_outline_indices {
                            if let Some(glyph) = glyph_gpu_data.get(*idx) {
                                pass.set_bind_group(0, camera_bg, &[]);
                                pass.set_bind_group(1, &glyph.uniform_bind_group, &[]);
                                pass.set_bind_group(2, &glyph.instance_bind_group, &[]);
                                pass.set_vertex_buffer(0, glyph.mesh_vertex_buffer.slice(..));
                                pass.set_index_buffer(
                                    glyph.mesh_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                match instance_filter {
                                    None => {
                                        pass.draw_indexed(
                                            0..glyph.mesh_index_count,
                                            0,
                                            0..glyph.instance_count,
                                        );
                                    }
                                    Some(indices) => {
                                        for &i in indices {
                                            pass.draw_indexed(
                                                0..glyph.mesh_index_count,
                                                0,
                                                i..i + 1,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw tensor glyph instances into the mask (instanced ellipsoids).
                if !tensor_glyph_outline_indices.is_empty() {
                    if let Some(pipeline) =
                        self.resources.tensor_glyph_outline_mask_pipeline.as_ref()
                    {
                        pass.set_pipeline(pipeline);
                        for (idx, instance_filter) in tensor_glyph_outline_indices {
                            if let Some(tg) = tensor_glyph_gpu_data.get(*idx) {
                                pass.set_bind_group(0, camera_bg, &[]);
                                pass.set_bind_group(1, &tg.uniform_bind_group, &[]);
                                pass.set_bind_group(2, &tg.instance_bind_group, &[]);
                                pass.set_vertex_buffer(0, tg.mesh_vertex_buffer.slice(..));
                                pass.set_index_buffer(
                                    tg.mesh_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                match instance_filter {
                                    None => {
                                        pass.draw_indexed(
                                            0..tg.mesh_index_count,
                                            0,
                                            0..tg.instance_count,
                                        );
                                    }
                                    Some(indices) => {
                                        for &i in indices {
                                            pass.draw_indexed(0..tg.mesh_index_count, 0, i..i + 1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw sprite billboards into the mask so the outline matches
                // each sprite's actual quad shape and per-instance size.
                if !sprite_outline_indices.is_empty() {
                    if let Some(pipeline) = self.resources.sprite_outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        for (idx, instance_filter) in sprite_outline_indices {
                            if let Some(sprite) = sprite_gpu_data.get(*idx) {
                                pass.set_bind_group(0, camera_bg, &[]);
                                pass.set_bind_group(1, &sprite.bind_group, &[]);
                                pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
                                match instance_filter {
                                    None => {
                                        pass.draw(0..6, 0..sprite.sprite_count);
                                    }
                                    Some(indices) => {
                                        for &i in indices {
                                            pass.draw(0..6, i..i + 1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw volumes into the mask using a simplified ray march so the
                // outline hugs the actual volume silhouette, not the AABB.
                if !vol_outline_indices.is_empty() {
                    if let Some(pipeline) = self.resources.volume_outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        for &idx in vol_outline_indices {
                            if let Some(vol) = self.volume_gpu_data.get(idx) {
                                pass.set_bind_group(1, &vol.bind_group, &[]);
                                pass.set_vertex_buffer(0, vol.vertex_buffer.slice(..));
                                pass.set_index_buffer(
                                    vol.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                pass.draw_indexed(0..36, 0, 0..1);
                            }
                        }
                    }
                }

                // Draw inline-geometry quads for image slices.
                for raw in raw_geom_outlines {
                    let pipeline = if raw.two_sided {
                        &self.resources.outline_mask_two_sided_pipeline
                    } else {
                        &self.resources.outline_mask_pipeline
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(1, &raw.mask_bind_group, &[]);
                    pass.set_vertex_buffer(0, raw.vertex_buf.slice(..));
                    pass.set_index_buffer(raw.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..raw.index_count, 0, 0..1);
                }

                // Draw screen-space rect outlines for screen images.
                if !screen_rect_outlines.is_empty() {
                    if let Some(pipeline) =
                        self.resources.screen_rect_outline_mask_pipeline.as_ref()
                    {
                        pass.set_pipeline(pipeline);
                        for sr in screen_rect_outlines {
                            pass.set_bind_group(0, &sr.bind_group, &[]);
                            pass.draw(0..6, 0..1);
                        }
                    }
                }

                // Draw GPU implicit surface outlines via ray-march to mask.
                if !implicit_outline_idxs.is_empty() {
                    if let Some(pipeline) = self.resources.implicit_outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, camera_bg, &[]);
                        for &idx in implicit_outline_idxs {
                            if let Some(gpu) = implicit_gpu_data.get(idx) {
                                pass.set_bind_group(1, &gpu.bind_group, &[]);
                                pass.draw(0..6, 0..1);
                            }
                        }
                    }
                }

                // Draw GPU marching cubes outlines (stride-24 vertex buffer, draw_indirect).
                if !mc_outlines.is_empty() {
                    if let Some(pipeline) = self.resources.mc_outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, camera_bg, &[]);
                        for mc_out in mc_outlines {
                            pass.set_bind_group(1, &mc_out.mask_bind_group, &[]);
                            if let Some(mc) = mc_gpu_frame_data.get(mc_out.mc_gpu_idx) {
                                if let Some(vol) = self.resources.mc_volumes.get(mc.volume_idx) {
                                    for slab in &vol.slabs {
                                        pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                                        pass.draw_indirect(&slab.indirect_buf, 0);
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw streamtube, tube, and ribbon mesh outlines. Streamtubes and
                // tubes use the back-face-culled pipeline; ribbons use the two-sided
                // pipeline because they are flat surfaces with no clear front face.
                pass.set_bind_group(0, camera_bg, &[]);
                let curve_draw_groups = [
                    (streamtube_outline_items as &[CurveMeshOutlineItem], streamtube_gpu_data as &[crate::resources::StreamtubeGpuData]),
                    (tube_outline_items, tube_gpu_data),
                    (ribbon_outline_items, ribbon_gpu_data),
                ];
                for (items, gpu_data_slice) in &curve_draw_groups {
                    for item in *items {
                        let pipeline = if item.two_sided {
                            &self.resources.outline_mask_two_sided_pipeline
                        } else {
                            &self.resources.outline_mask_pipeline
                        };
                        pass.set_pipeline(pipeline);
                        if let Some(gpu) = gpu_data_slice.get(item.index) {
                            pass.set_bind_group(1, &item.mask_bind_group, &[]);
                            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                            pass.set_index_buffer(
                                gpu.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
                        }
                    }
                }

                // Draw polyline segment quads into the mask using the dedicated
                // polyline_outline_mask_pipeline (instance-expanded quads).
                if !polyline_outline_idxs.is_empty() {
                    if let Some(pipeline) =
                        self.resources.polyline_outline_mask_pipeline.as_ref()
                    {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, camera_bg, &[]);
                        for &idx in polyline_outline_idxs {
                            if let Some(pline) = polyline_gpu_data.get(idx) {
                                pass.set_bind_group(1, &pline.bind_group, &[]);
                                pass.set_vertex_buffer(0, pline.vertex_buffer.slice(..));
                                pass.draw(0..6, 0..pline.segment_count);
                            }
                        }
                    }
                }
            }

            // Pass 2: fullscreen edge detection (reads mask, writes colour).
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("outline_edge_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: colour_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.resources.outline_edge_pipeline);
                pass.set_bind_group(0, edge_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // ------------------------------------------------------------------
        // Sub-object highlight prepare: build GPU geometry from sub-selection
        // snapshot when the version has changed since the last frame.
        // ------------------------------------------------------------------
        {
            let w = frame.camera.viewport_size[0];
            let h = frame.camera.viewport_size[1];

            let has_sub_sel = frame.interaction.sub_selection.is_some();

            if has_sub_sel {
                let needs_rebuild = {
                    let slot = &self.viewport_slots[vp_idx];
                    let sel_version_changed = frame
                        .interaction
                        .sub_selection
                        .as_ref()
                        .map(|s| slot.sub_highlight_generation != s.version)
                        .unwrap_or(slot.sub_highlight_generation != u64::MAX);
                    sel_version_changed || slot.sub_highlight.is_none()
                };
                if needs_rebuild {
                    self.resources.ensure_sub_highlight_pipelines(device);
                    let sel_ref = frame.interaction.sub_selection.as_ref();
                    let data = self.resources.build_sub_highlight(
                        device,
                        queue,
                        sel_ref,
                        &std::collections::HashMap::new(),
                        &[],
                        frame.interaction.sub_highlight_face_fill_colour,
                        frame.interaction.sub_highlight_edge_colour,
                        frame.interaction.sub_highlight_edge_width_px,
                        frame.interaction.sub_highlight_vertex_size_px,
                        w,
                        h,
                    );
                    let new_gen = frame
                        .interaction
                        .sub_selection
                        .as_ref()
                        .map(|s| s.version)
                        .unwrap_or(u64::MAX);
                    let slot = &mut self.viewport_slots[vp_idx];
                    slot.sub_highlight = Some(data);
                    slot.sub_highlight_generation = new_gen;
                }
            } else {
                let slot = &mut self.viewport_slots[vp_idx];
                slot.sub_highlight = None;
                slot.sub_highlight_generation = u64::MAX;
            }
        }

        // ---------------------------------------------------------------
        // Overlay labels
        // ---------------------------------------------------------------
        self.label_gpu_data = None;
        if !frame.overlays.labels.is_empty() {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let view = &frame.camera.render_camera.view;
                let proj = &frame.camera.render_camera.projection;

                // Sort by z_order for correct draw ordering.
                let mut sorted_labels: Vec<&crate::renderer::types::LabelItem> =
                    frame.overlays.labels.iter().collect();
                sorted_labels.sort_by_key(|l| l.z_order);

                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                for label in &sorted_labels {
                    if label.text.is_empty() || label.opacity <= 0.0 {
                        continue;
                    }

                    // Resolve screen position from anchor.
                    let screen_pos = if let Some(sa) = label.screen_anchor {
                        Some(sa)
                    } else if let Some(wa) = label.world_anchor {
                        project_to_screen(wa, view, proj, vp_w, vp_h)
                    } else {
                        continue;
                    };
                    let Some(anchor_px) = screen_pos else {
                        continue;
                    };

                    let opacity = label.opacity.clamp(0.0, 1.0);

                    // Layout text (with optional word wrapping).
                    let layout = if let Some(max_w) = label.max_width {
                        self.resources.glyph_atlas.layout_text_wrapped(
                            &label.text,
                            label.font_size,
                            label.font,
                            max_w,
                            device,
                        )
                    } else {
                        self.resources.glyph_atlas.layout_text(
                            &label.text,
                            label.font_size,
                            label.font,
                            device,
                        )
                    };

                    // Compute ascent so glyphs are positioned below the anchor.
                    let font_index = label.font.map_or(0, |h| h.0);
                    let ascent = self
                        .resources
                        .glyph_atlas
                        .font_ascent(font_index, label.font_size);

                    // Horizontal alignment.
                    let align_offset = match label.anchor_align {
                        crate::renderer::types::LabelAnchor::Leading => 6.0,
                        crate::renderer::types::LabelAnchor::Center => -layout.total_width * 0.5,
                        crate::renderer::types::LabelAnchor::Trailing => -layout.total_width - 6.0,
                    };

                    // Text origin with alignment + user offset.
                    let text_x = anchor_px[0] + align_offset + label.offset[0];
                    let text_y = anchor_px[1] - layout.height * 0.5 + label.offset[1];

                    // Background box (drawn first, behind text).
                    if label.background {
                        let pad = label.padding;
                        let bx0 = text_x - pad;
                        let by0 = text_y - pad;
                        let bx1 = text_x + layout.total_width + pad;
                        let by1 = text_y + layout.height + pad;
                        let bg_colour = apply_opacity(label.background_colour, opacity);
                        if label.border_radius > 0.0 {
                            emit_rounded_quad(
                                &mut verts,
                                bx0,
                                by0,
                                bx1,
                                by1,
                                label.border_radius,
                                bg_colour,
                                vp_w,
                                vp_h,
                            );
                        } else {
                            emit_solid_quad(&mut verts, bx0, by0, bx1, by1, bg_colour, vp_w, vp_h);
                        }
                    }

                    // Leader line.
                    if label.leader_line {
                        if let Some(wa) = label.world_anchor {
                            let world_px = project_to_screen(wa, view, proj, vp_w, vp_h);
                            if let Some(wp) = world_px {
                                emit_line_quad(
                                    &mut verts,
                                    wp[0],
                                    wp[1],
                                    text_x,
                                    text_y + layout.height * 0.5,
                                    1.5,
                                    apply_opacity(label.leader_colour, opacity),
                                    vp_w,
                                    vp_h,
                                );
                            }
                        }
                    }

                    // Glyph quads.
                    let text_colour = apply_opacity(label.colour, opacity);
                    for gq in &layout.quads {
                        let gx = text_x + gq.pos[0];
                        let gy = text_y + ascent + gq.pos[1];
                        emit_textured_quad(
                            &mut verts,
                            gx,
                            gy,
                            gx + gq.size[0],
                            gy + gq.size[1],
                            gq.uv_min,
                            gq.uv_max,
                            text_colour,
                            vp_w,
                            vp_h,
                        );
                    }
                }

                // Upload atlas if new glyphs were rasterized.
                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("overlay_label_vbuf"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("overlay_label_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.label_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }

        // ---------------------------------------------------------------
        // Scalar bars
        // ---------------------------------------------------------------
        self.scalar_bar_gpu_data = None;
        if !frame.overlays.scalar_bars.is_empty() {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                for bar in &frame.overlays.scalar_bars {
                    // Clone the LUT immediately so the immutable borrow on self.resources
                    // is released before the mutable glyph_atlas borrow below.
                    let Some(lut) = self
                        .resources
                        .get_colourmap_rgba(bar.colourmap_id)
                        .map(|l| l.to_vec())
                    else {
                        continue;
                    };

                    let is_vertical = matches!(
                        bar.orientation,
                        crate::renderer::types::ScalarBarOrientation::Vertical
                    );
                    let reversed = bar.ticks_reversed;

                    // Effective font sizes.
                    let tick_fs = bar.font_size;
                    let title_fs = bar.title_font_size.unwrap_or(bar.font_size);
                    let font_index = bar.font.map_or(0, |h| h.0);

                    // Actual pixel dimensions of the gradient strip.
                    let (strip_w, strip_h) = if is_vertical {
                        (bar.bar_width_px, bar.bar_length_px)
                    } else {
                        (bar.bar_length_px, bar.bar_width_px)
                    };

                    // Pre-compute tick texts and their widths so the background box
                    // can be sized to cover the tick labels.
                    let tick_count = bar.tick_count.max(2);
                    let mut tick_data: Vec<(String, f32, f32)> = Vec::new(); // (text, total_w, height)
                    let mut max_tick_w = 0.0f32;
                    let mut tick_h = 0.0f32;
                    for i in 0..tick_count {
                        let t = i as f32 / (tick_count - 1) as f32;
                        let value = bar.scalar_min + t * (bar.scalar_max - bar.scalar_min);
                        let text = format!("{value:.2}");
                        let layout = self
                            .resources
                            .glyph_atlas
                            .layout_text(&text, tick_fs, bar.font, device);
                        max_tick_w = max_tick_w.max(layout.total_width);
                        tick_h = layout.height;
                        tick_data.push((text, layout.total_width, layout.height));
                    }

                    // Vertical space reserved above the gradient strip.
                    // In vertical mode the top/bottom tick labels are centred on the strip
                    // endpoints, so they each overhang by tick_h/2. title_h must absorb the
                    // top overhang AND leave a gap so the title text does not touch the tick.
                    let half_tick = tick_h / 2.0;
                    let title_h = if bar.title.is_some() {
                        // title text height + small gap + top-tick overhang
                        title_fs + 4.0 + half_tick
                    } else {
                        // no title, but still need room for the top-tick overhang
                        half_tick
                    };

                    // Pre-compute title width before bar_x/bar_y so the overhang can
                    // be used to push the strip inward and prevent clipping.
                    let title_w = if let Some(ref t) = bar.title {
                        self.resources
                            .glyph_atlas
                            .layout_text(t, title_fs, bar.font, device)
                            .total_width
                    } else {
                        0.0
                    };

                    // How far title / tick labels spill beyond the strip on each side.
                    // Vertical: title centred on the narrow strip, ticks to the right.
                    //   left side: title overhang only.
                    //   right side: ticks dominate (strip_w + 4 + max_tick_w).
                    // Horizontal: both title and tick labels can overhang left/right equally.
                    let bg_pad = 4.0;
                    let (inset_left, inset_right) = if is_vertical {
                        let title_oh = ((title_w - strip_w) / 2.0).max(0.0);
                        let right_extent = 4.0 + max_tick_w + bg_pad; // relative to strip right edge
                        (title_oh + bg_pad, right_extent)
                    } else {
                        let title_oh = ((title_w - strip_w) / 2.0).max(0.0);
                        let tick_oh = max_tick_w / 2.0;
                        let side = title_oh.max(tick_oh) + bg_pad;
                        (side, side)
                    };

                    // How far content hangs below the strip bottom (used to keep the
                    // background box flush with margin_px on the bottom-anchored side).
                    // Vertical: bottom tick label is centred on the strip endpoint -> half_tick.
                    // Horizontal: tick labels sit fully below the strip -> 3 + tick_h.
                    let bottom_overhang = if is_vertical { half_tick } else { 3.0 + tick_h };

                    // Top-left of the gradient strip.
                    // bg_pad is added/subtracted here so that the background box edge lands
                    // exactly at margin_px from the viewport edge on the anchored side.
                    //   Top anchor:    bg_y0 = bar_y - title_h - bg_pad  =>  set bar_y = margin_px + title_h + bg_pad
                    //   Bottom anchor: bg_y1 = bar_y + strip_h + bottom_overhang + bg_pad  =>  bar_y = vp_h - margin_px - strip_h - bottom_overhang - bg_pad
                    let (bar_x, bar_y) = match bar.anchor {
                        crate::renderer::types::ScalarBarAnchor::TopLeft => {
                            (bar.margin_px + inset_left, bar.margin_px + title_h + bg_pad)
                        }
                        crate::renderer::types::ScalarBarAnchor::TopRight => (
                            vp_w - bar.margin_px - strip_w - inset_right,
                            bar.margin_px + title_h + bg_pad,
                        ),
                        crate::renderer::types::ScalarBarAnchor::BottomLeft => (
                            bar.margin_px + inset_left,
                            vp_h - bar.margin_px - strip_h - bottom_overhang - bg_pad,
                        ),
                        crate::renderer::types::ScalarBarAnchor::BottomRight => (
                            vp_w - bar.margin_px - strip_w - inset_right,
                            vp_h - bar.margin_px - strip_h - bottom_overhang - bg_pad,
                        ),
                    };

                    // Background box: now that bar_x/bar_y are inset, the box stays on screen.
                    let (bg_x0, bg_y0, bg_x1, bg_y1) = if is_vertical {
                        let title_right = bar_x + (strip_w + title_w) / 2.0;
                        let ticks_right = bar_x + strip_w + 4.0 + max_tick_w;
                        (
                            bar_x - bg_pad - ((title_w - strip_w) / 2.0).max(0.0),
                            bar_y - title_h - bg_pad,
                            ticks_right.max(title_right) + bg_pad,
                            bar_y + strip_h + half_tick + bg_pad,
                        )
                    } else {
                        let title_overhang = ((title_w - strip_w) / 2.0).max(0.0);
                        let tick_overhang = max_tick_w / 2.0;
                        let side_pad = title_overhang.max(tick_overhang);
                        let bottom = bar_y + strip_h + 3.0 + tick_h + bg_pad;
                        (
                            bar_x - bg_pad - side_pad,
                            bar_y - title_h - bg_pad,
                            bar_x + strip_w + bg_pad + side_pad,
                            bottom,
                        )
                    };
                    emit_rounded_quad(
                        &mut verts,
                        bg_x0,
                        bg_y0,
                        bg_x1,
                        bg_y1,
                        3.0,
                        bar.background_colour,
                        vp_w,
                        vp_h,
                    );

                    // Gradient strip: 64 solid quads sampled from the colourmap LUT.
                    let steps: usize = 64;
                    for s in 0..steps {
                        let (qx0, qy0, qx1, qy1, t) = if is_vertical {
                            // Default: top = max (t=1). Reversed: top = min (t=0).
                            let t = if reversed {
                                s as f32 / (steps - 1) as f32
                            } else {
                                1.0 - s as f32 / (steps - 1) as f32
                            };
                            let step_h = strip_h / steps as f32;
                            let sy = bar_y + s as f32 * step_h;
                            (bar_x, sy, bar_x + strip_w, sy + step_h + 0.5, t)
                        } else {
                            // Default: left = min (t=0). Reversed: left = max (t=1).
                            let t = if reversed {
                                1.0 - s as f32 / (steps - 1) as f32
                            } else {
                                s as f32 / (steps - 1) as f32
                            };
                            let step_w = strip_w / steps as f32;
                            let sx = bar_x + s as f32 * step_w;
                            (sx, bar_y, sx + step_w + 0.5, bar_y + strip_h, t)
                        };
                        let lut_idx = (t * 255.0).clamp(0.0, 255.0) as usize;
                        let [r, g, b, a] = lut[lut_idx];
                        let colour = [
                            r as f32 / 255.0,
                            g as f32 / 255.0,
                            b as f32 / 255.0,
                            a as f32 / 255.0,
                        ];
                        emit_solid_quad(&mut verts, qx0, qy0, qx1, qy1, colour, vp_w, vp_h);
                    }

                    // Tick labels.
                    let ascent = self.resources.glyph_atlas.font_ascent(font_index, tick_fs);
                    for (i, (text, tw, th)) in tick_data.iter().enumerate() {
                        let t = i as f32 / (tick_count - 1) as f32;
                        let layout = self
                            .resources
                            .glyph_atlas
                            .layout_text(text, tick_fs, bar.font, device);

                        let (lx, ly) = if is_vertical {
                            // Place text to the right of the strip, vertically centered
                            // on its tick position.
                            // Default: top=max -> progress = 1.0-t puts max at top.
                            // Reversed: top=min -> progress = t puts min at top.
                            let progress = if reversed { t } else { 1.0 - t };
                            let tick_y = bar_y + progress * strip_h;
                            (bar_x + strip_w + 4.0, tick_y - th * 0.5)
                        } else {
                            // Place text below the strip, horizontally centered on its tick.
                            // Default: left=min -> tick at t*strip_w.
                            // Reversed: left=max -> tick at (1-t)*strip_w.
                            let frac = if reversed { 1.0 - t } else { t };
                            let tick_x = bar_x + frac * strip_w;
                            (tick_x - tw * 0.5, bar_y + strip_h + 3.0)
                        };
                        let _ = (tw, th); // used above

                        for gq in &layout.quads {
                            let gx = lx + gq.pos[0];
                            let gy = ly + ascent + gq.pos[1];
                            emit_textured_quad(
                                &mut verts,
                                gx,
                                gy,
                                gx + gq.size[0],
                                gy + gq.size[1],
                                gq.uv_min,
                                gq.uv_max,
                                bar.label_colour,
                                vp_w,
                                vp_h,
                            );
                        }
                    }

                    // Optional title above the gradient strip.
                    if let Some(ref title_text) = bar.title {
                        let layout = self
                            .resources
                            .glyph_atlas
                            .layout_text(title_text, title_fs, bar.font, device);
                        let title_ascent =
                            self.resources.glyph_atlas.font_ascent(font_index, title_fs);
                        // Centre the title over the gradient strip.
                        let tx = bar_x + (strip_w - layout.total_width) * 0.5;
                        let ty = bar_y - title_h;
                        for gq in &layout.quads {
                            let gx = tx + gq.pos[0];
                            let gy = ty + title_ascent + gq.pos[1];
                            emit_textured_quad(
                                &mut verts,
                                gx,
                                gy,
                                gx + gq.size[0],
                                gy + gq.size[1],
                                gq.uv_min,
                                gq.uv_max,
                                bar.label_colour,
                                vp_w,
                                vp_h,
                            );
                        }
                    }
                }

                // Upload any newly rasterized glyphs (may overlap with label upload above).
                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("overlay_scalar_bar_vbuf"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("overlay_scalar_bar_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.scalar_bar_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }

        // ---------------------------------------------------------------
        // Rulers
        // ---------------------------------------------------------------
        self.ruler_gpu_data = None;
        if !frame.overlays.rulers.is_empty() {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let view = &frame.camera.render_camera.view;
                let proj = &frame.camera.render_camera.projection;

                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                for ruler in &frame.overlays.rulers {
                    // Project both endpoints to NDC (returns None only if behind camera).
                    let start_ndc = project_to_ndc(ruler.start, view, proj);
                    let end_ndc = project_to_ndc(ruler.end, view, proj);

                    // Cull entirely when either endpoint is behind the camera.
                    let (Some(sndc), Some(endc)) = (start_ndc, end_ndc) else {
                        continue;
                    };

                    // Clip the segment to the viewport NDC box [-1,1]^2.
                    // This keeps the line visible when only one end is off-screen sideways.
                    let Some((csndc, cendc)) = clip_line_ndc(sndc, endc) else {
                        continue;
                    };

                    let [sx, sy] = ndc_to_screen_px(csndc, vp_w, vp_h);
                    let [ex, ey] = ndc_to_screen_px(cendc, vp_w, vp_h);

                    // Track which original endpoints are within the viewport (for end caps).
                    let start_on_screen = ndc_in_viewport(sndc);
                    let end_on_screen = ndc_in_viewport(endc);

                    // Main ruler line.
                    emit_line_quad(
                        &mut verts,
                        sx,
                        sy,
                        ex,
                        ey,
                        ruler.line_width_px,
                        ruler.colour,
                        vp_w,
                        vp_h,
                    );

                    // End caps only at endpoints that are actually on screen.
                    if ruler.end_caps {
                        let dx = ex - sx;
                        let dy = ey - sy;
                        let len = (dx * dx + dy * dy).sqrt().max(0.001);
                        let cap_half = 5.0;
                        let px = -dy / len * cap_half;
                        let py = dx / len * cap_half;

                        if start_on_screen {
                            emit_line_quad(
                                &mut verts,
                                sx - px,
                                sy - py,
                                sx + px,
                                sy + py,
                                ruler.line_width_px,
                                ruler.colour,
                                vp_w,
                                vp_h,
                            );
                        }
                        if end_on_screen {
                            emit_line_quad(
                                &mut verts,
                                ex - px,
                                ey - py,
                                ex + px,
                                ey + py,
                                ruler.line_width_px,
                                ruler.colour,
                                vp_w,
                                vp_h,
                            );
                        }
                    }

                    // Distance label: always shows true 3D distance.
                    // Place it at the midpoint of the visible (clipped) segment.
                    let start_world = glam::Vec3::from(ruler.start);
                    let end_world = glam::Vec3::from(ruler.end);
                    let distance = (end_world - start_world).length();
                    let text = format_ruler_distance(distance, ruler.label_format.as_deref());

                    let mid_x = (sx + ex) * 0.5;
                    let mid_y = (sy + ey) * 0.5;

                    let layout = self.resources.glyph_atlas.layout_text(
                        &text,
                        ruler.font_size,
                        ruler.font,
                        device,
                    );
                    let font_index = ruler.font.map_or(0, |h| h.0);
                    let ascent = self
                        .resources
                        .glyph_atlas
                        .font_ascent(font_index, ruler.font_size);

                    // Center the label above the midpoint with a small gap.
                    let lx = mid_x - layout.total_width * 0.5;
                    let ly = mid_y - layout.height - 6.0;

                    // Semi-transparent background box.
                    let pad = 3.0;
                    emit_solid_quad(
                        &mut verts,
                        lx - pad,
                        ly - pad,
                        lx + layout.total_width + pad,
                        ly + layout.height + pad,
                        [0.0, 0.0, 0.0, 0.55],
                        vp_w,
                        vp_h,
                    );

                    // Glyph quads.
                    for gq in &layout.quads {
                        let gx = lx + gq.pos[0];
                        let gy = ly + ascent + gq.pos[1];
                        emit_textured_quad(
                            &mut verts,
                            gx,
                            gy,
                            gx + gq.size[0],
                            gy + gq.size[1],
                            gq.uv_min,
                            gq.uv_max,
                            ruler.label_colour,
                            vp_w,
                            vp_h,
                        );
                    }
                }

                // Upload any newly rasterized glyphs.
                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("overlay_ruler_vbuf"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("overlay_ruler_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.ruler_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }

        // ---------------------------------------------------------------
        // Loading bars
        // ---------------------------------------------------------------
        self.loading_bar_gpu_data = None;
        if !frame.overlays.loading_bars.is_empty() {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                for bar in &frame.overlays.loading_bars {
                    // Bar top-left corner based on anchor.
                    let bar_x = vp_w * 0.5 - bar.width_px * 0.5;
                    let bar_y = match bar.anchor {
                        crate::renderer::types::LoadingBarAnchor::TopCenter => bar.margin_px,
                        crate::renderer::types::LoadingBarAnchor::Center => {
                            vp_h * 0.5 - bar.height_px * 0.5
                        }
                        crate::renderer::types::LoadingBarAnchor::BottomCenter => {
                            vp_h - bar.margin_px - bar.height_px
                        }
                    };

                    // Label above (TopCenter: below) the bar.
                    if let Some(ref text) = bar.label {
                        let layout = self.resources.glyph_atlas.layout_text(
                            text,
                            bar.font_size,
                            bar.font,
                            device,
                        );
                        let font_index = bar.font.map_or(0, |h| h.0);
                        let ascent =
                            self.resources.glyph_atlas.font_ascent(font_index, bar.font_size);
                        let label_gap = 5.0;
                        let lx = bar_x + bar.width_px * 0.5 - layout.total_width * 0.5;
                        let ly = match bar.anchor {
                            crate::renderer::types::LoadingBarAnchor::TopCenter => {
                                bar_y + bar.height_px + label_gap
                            }
                            _ => bar_y - layout.height - label_gap,
                        };
                        for gq in &layout.quads {
                            let gx = lx + gq.pos[0];
                            let gy = ly + ascent + gq.pos[1];
                            emit_textured_quad(
                                &mut verts,
                                gx,
                                gy,
                                gx + gq.size[0],
                                gy + gq.size[1],
                                gq.uv_min,
                                gq.uv_max,
                                bar.label_colour,
                                vp_w,
                                vp_h,
                            );
                        }
                    }

                    // Background rectangle.
                    emit_rounded_quad(
                        &mut verts,
                        bar_x,
                        bar_y,
                        bar_x + bar.width_px,
                        bar_y + bar.height_px,
                        bar.corner_radius,
                        bar.background_colour,
                        vp_w,
                        vp_h,
                    );

                    // Fill rectangle clipped to progress fraction.
                    let fill_w = bar.width_px * bar.progress.clamp(0.0, 1.0);
                    if fill_w > 0.5 {
                        emit_rounded_quad(
                            &mut verts,
                            bar_x,
                            bar_y,
                            bar_x + fill_w,
                            bar_y + bar.height_px,
                            bar.corner_radius,
                            bar.fill_colour,
                            vp_w,
                            vp_h,
                        );
                    }
                }

                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("loading_bar_vbuf"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("loading_bar_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.loading_bar_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }

        // ------------------------------------------------------------------
        // Overlay rects
        // ------------------------------------------------------------------
        self.overlay_rect_gpu_data = None;
        if !frame.overlays.rects.is_empty() {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                let mut sorted: Vec<&crate::renderer::types::OverlayRectItem> =
                    frame.overlays.rects.iter().collect();
                sorted.sort_by_key(|r| r.z_order);

                for rect in &sorted {
                    if rect.opacity <= 0.0 {
                        continue;
                    }
                    let x0 = rect.position[0];
                    let y0 = rect.position[1];
                    let x1 = x0 + rect.size[0];
                    let y1 = y0 + rect.size[1];

                    // Border: slightly expanded rect drawn behind the fill.
                    if rect.border_width > 0.0 {
                        let bw = rect.border_width;
                        let mut bc = rect.border_colour;
                        bc[3] *= rect.opacity;
                        emit_rounded_quad(
                            &mut verts,
                            x0 - bw,
                            y0 - bw,
                            x1 + bw,
                            y1 + bw,
                            rect.corner_radius + bw,
                            bc,
                            vp_w,
                            vp_h,
                        );
                    }

                    // Fill.
                    let mut fc = rect.colour;
                    fc[3] *= rect.opacity;
                    emit_rounded_quad(
                        &mut verts,
                        x0,
                        y0,
                        x1,
                        y1,
                        rect.corner_radius,
                        fc,
                        vp_w,
                        vp_h,
                    );
                }

                if !verts.is_empty() {
                    let vertex_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("overlay_rect_vbuf"),
                            contents: bytemuck::cast_slice(&verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("overlay_rect_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.overlay_rect_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }

        // ------------------------------------------------------------------
        // Gaussian splat: per-viewport GPU sort.
        // ------------------------------------------------------------------
        self.gaussian_splat_draw_data.clear();
        if !frame.scene.gaussian_splats.is_empty() {
            self.resources.ensure_gaussian_splat_pipelines(device);
            let vp_idx = frame.camera.viewport_index;
            let eye = frame.camera.render_camera.eye_position;
            let vp_w = frame.camera.viewport_size[0].max(1.0);
            let vp_h = frame.camera.viewport_size[1].max(1.0);
            for item in &frame.scene.gaussian_splats {
                let store_index = item.id.0;
                if self
                    .resources
                    .gaussian_splat_store
                    .get(store_index)
                    .is_none()
                {
                    continue;
                }
                let sh_degree = self
                    .resources
                    .gaussian_splat_store
                    .get(store_index)
                    .unwrap()
                    .sh_degree;
                let count = self
                    .resources
                    .gaussian_splat_store
                    .get(store_index)
                    .unwrap()
                    .count;
                self.resources.run_gaussian_splat_sort(
                    device,
                    queue,
                    store_index,
                    vp_idx,
                    eye,
                    item.model,
                    vp_w,
                    vp_h,
                    sh_degree,
                );
                self.gaussian_splat_draw_data
                    .push(crate::resources::GaussianSplatDrawData {
                        store_index,
                        viewport_index: vp_idx,
                        model: item.model,
                        count,
                        wireframe: frame.viewport.wireframe_mode || item.appearance.wireframe,
                    });
            }
        }

        // Gaussian splat wireframe overlay.
        let need_splat_wf = frame.viewport.wireframe_mode
            || frame.scene.gaussian_splats.iter().any(|g| !g.appearance.hidden && g.appearance.wireframe);
        if need_splat_wf {
            self.resources.ensure_polyline_pipeline(device);
            let vp_size = frame.camera.viewport_size;
            for item in &frame.scene.gaussian_splats {
                if item.appearance.hidden {
                    continue;
                }
                if !(frame.viewport.wireframe_mode || item.appearance.wireframe) {
                    continue;
                }
                let store_index = item.id.0;
                let Some(gpu_set) = self.resources.gaussian_splat_store.get(store_index) else {
                    continue;
                };
                let count = gpu_set.count as usize;
                let positions = gpu_set.cpu_positions.clone();
                let scales = gpu_set.cpu_scales.clone();
                let _ = gpu_set;
                let polyline = splat_wireframe_polyline(&positions, &scales, item.model, count);
                if !polyline.positions.is_empty() {
                    let gpu = self.resources.upload_polyline(device, queue, &polyline, vp_size);
                    self.polyline_gpu_data.push(gpu);
                }
            }
        }

        // Sprite wireframe overlay: quad outline per sprite (<=100) or AABB box (>100).
        let need_sprite_wf = frame.viewport.wireframe_mode
            || frame.scene.sprite_items.iter().any(|s| !s.appearance.hidden && s.appearance.wireframe);
        if need_sprite_wf {
            self.resources.ensure_polyline_pipeline(device);
            let vp_size = frame.camera.viewport_size;
            for item in &frame.scene.sprite_items {
                if item.appearance.hidden {
                    continue;
                }
                if !(frame.viewport.wireframe_mode || item.appearance.wireframe) {
                    continue;
                }
                if item.positions.is_empty() {
                    continue;
                }
                let polyline = sprite_wireframe_polyline(item, &frame.camera);
                if !polyline.positions.is_empty() {
                    let gpu = self.resources.upload_polyline(device, queue, &polyline, vp_size);
                    self.polyline_gpu_data.push(gpu);
                }
            }
        }
    }

    /// Upload per-frame data to GPU buffers and render the shadow pass.
    /// Call before `paint()`.
    ///
    /// Returns [`crate::FrameStats`] with per-frame timing and upload metrics.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) -> crate::renderer::stats::FrameStats {
        let prepare_start = std::time::Instant::now();

        // Phase 4 : read back GPU timestamps from the previous frame, if available.
        // By the time prepare() is called, the previous frame's queue.submit() has
        // already happened, so it is safe to initiate the map here.
        if self.ts_needs_readback {
            if let Some(ref stg_buf) = self.ts_staging_buf {
                let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
                stg_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                // Non-blocking poll: flush any completed callbacks. GPU work from the
                // previous frame is almost certainly done by the time CPU reaches here.
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(std::time::Duration::from_millis(100)),
                    })
                    .ok();
                if rx.try_recv().unwrap_or(Err(wgpu::BufferAsyncError)).is_ok() {
                    let data = stg_buf.slice(..).get_mapped_range();
                    let t0 = u64::from_le_bytes(data[0..8].try_into().unwrap());
                    let t1 = u64::from_le_bytes(data[8..16].try_into().unwrap());
                    drop(data);
                    // ts_period is nanoseconds/tick; convert delta to milliseconds.
                    let gpu_ms = t1.saturating_sub(t0) as f32 * self.ts_period / 1_000_000.0;
                    self.last_stats.gpu_frame_ms = Some(gpu_ms);
                }
                stg_buf.unmap();
            }
            self.ts_needs_readback = false;
        }

        // Read back GPU-visible instance count from the previous frame's indirect args copy.
        // The cull pass from the previous frame has already been submitted and is almost
        // certainly done by the time prepare() is called; a short poll is enough.
        if self.indirect_readback_pending {
            if let Some(ref stg_buf) = self.indirect_readback_buf {
                let bytes = self.indirect_readback_batch_count as u64 * 20;
                if bytes > 0 {
                    let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
                    stg_buf
                        .slice(..bytes)
                        .map_async(wgpu::MapMode::Read, move |r| {
                            let _ = tx.send(r);
                        });
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(std::time::Duration::from_millis(100)),
                        })
                        .ok();
                    if rx.try_recv().unwrap_or(Err(wgpu::BufferAsyncError)).is_ok() {
                        let data = stg_buf.slice(..bytes).get_mapped_range();
                        let mut visible: u32 = 0;
                        for i in 0..self.indirect_readback_batch_count as usize {
                            // DrawIndexedIndirect layout: [index_count, instance_count, first_index, base_vertex, first_instance]
                            // instance_count is at byte offset 4 within each 20-byte entry.
                            let off = i * 20 + 4;
                            let n = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            visible = visible.saturating_add(n);
                        }
                        drop(data);
                        self.last_stats.gpu_visible_instances = Some(visible);
                    }
                    stg_buf.unmap();
                }
            }
            self.indirect_readback_pending = false;
        }

        // Wall-clock duration since the previous prepare() call approximates the frame interval.
        let total_frame_ms = self
            .last_prepare_instant
            .map(|t| t.elapsed().as_secs_f32() * 1000.0)
            .unwrap_or(0.0);

        // Snapshot geometry upload bytes accumulated since the last frame, then reset.
        let upload_bytes = self.resources.frame_upload_bytes;
        self.resources.frame_upload_bytes = 0;

        // Resolve effective scale bounds and degradation flags.
        // When a preset is set it overrides the individual fields; the individual
        // fields are preserved so they restore when switching back to None.
        let policy = self.performance_policy;
        let (eff_min_scale, eff_max_scale, eff_allow_shadows, eff_allow_volumes, eff_allow_effects) =
            match policy.preset {
                Some(crate::renderer::stats::QualityPreset::High) => {
                    (1.0_f32, 1.0_f32, false, false, false)
                }
                Some(crate::renderer::stats::QualityPreset::Medium) => {
                    (0.75_f32, 1.0_f32, true, false, true)
                }
                Some(crate::renderer::stats::QualityPreset::Low) => {
                    (0.5_f32, 0.75_f32, true, true, true)
                }
                None => (
                    policy.min_render_scale,
                    policy.max_render_scale,
                    policy.allow_shadow_reduction,
                    policy.allow_volume_quality_reduction,
                    policy.allow_effect_throttling,
                ),
            };

        // Capture mode: force max render scale and suppress all degradation.
        // The adaptation controller is paused for the duration of the frame.
        let in_capture = self.runtime_mode == crate::renderer::stats::RuntimeMode::Capture;
        if in_capture {
            self.current_render_scale = eff_max_scale;
        }

        // When a preset is active, clamp current_render_scale to the preset's bounds
        // immediately, without requiring allow_dynamic_resolution. This ensures the
        // preset has a visible effect even when the adaptation controller is off.
        // The controller can still adjust within these bounds when enabled.
        if !in_capture && policy.preset.is_some() {
            self.current_render_scale = self
                .current_render_scale
                .clamp(eff_min_scale, eff_max_scale);
        }

        // Tiered degradation ladder.
        // Order: render scale -> shadows -> volumes -> effects.
        // The tier advances one step per over-budget frame once render scale has
        // reached its minimum (nothing more the controller can reduce).
        // The tier retreats one step per frame that is comfortably under budget,
        // reversing the ladder in the same order (effects first).
        // Capture mode resets the tier; otherwise advance/retreat based on budget.
        let missed_prev = self.last_stats.missed_budget;
        let under_prev = !self.last_stats.missed_budget
            && policy
                .target_fps
                .map(|fps| {
                    let budget = 1000.0 / fps;
                    let sig = self
                        .last_stats
                        .gpu_frame_ms
                        .unwrap_or(self.last_stats.total_frame_ms);
                    sig < budget * 0.8
                })
                .unwrap_or(true);
        if in_capture {
            self.degradation_tier = 0;
        } else {
            let at_min = !policy.allow_dynamic_resolution
                || self.current_render_scale <= eff_min_scale + 0.001;
            if missed_prev && at_min {
                self.degradation_tier = (self.degradation_tier + 1).min(3);
            } else if under_prev {
                self.degradation_tier = self.degradation_tier.saturating_sub(1);
            }
        }

        // Derive per-pass flags from the current tier and effective policy.
        // All flags are suppressed in Capture mode regardless of tier.
        self.degradation_shadows_skipped =
            !in_capture && self.degradation_tier >= 1 && eff_allow_shadows;
        self.degradation_volume_quality_reduced =
            !in_capture && self.degradation_tier >= 2 && eff_allow_volumes;
        self.degradation_effects_throttled =
            !in_capture && self.degradation_tier >= 3 && eff_allow_effects;

        // Cache scene items for renderer.pick() dispatch.
        {
            let surfaces = match &frame.scene.surfaces {
                SurfaceSubmission::Flat(items) => items.as_ref(),
            };
            self.pick_scene_items = surfaces.to_vec();
            self.pick_point_cloud_items = frame.scene.point_clouds.clone();
            self.pick_splat_items = frame.scene.gaussian_splats.clone();
            self.pick_volume_items = frame.scene.volumes.clone();
            self.pick_tvm_items = frame.scene.transparent_volume_meshes.clone();
            self.pick_volume_mesh_items = frame.scene.volume_mesh_items.clone();
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
        }

        let (scene_fx, viewport_fx) = frame.effects.split();
        self.prepare_scene_internal(device, queue, frame, &scene_fx);
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx);

        let cpu_prepare_ms = prepare_start.elapsed().as_secs_f32() * 1000.0;

        let budget_ms = policy.target_fps.map(|fps| 1000.0 / fps);

        // Controller signal: prefer gpu_frame_ms (excludes vsync wait, one-frame lag is
        // acceptable). Fall back to total_frame_ms when GPU timestamps are unavailable:
        // it reflects wall-clock frame duration and correctly fires over-budget at low
        // frame rates. cpu_prepare_ms is not used as a fallback because it only measures
        // CPU-side work and is low even when the GPU or driver is the bottleneck.
        let controller_ms = self.last_stats.gpu_frame_ms.unwrap_or(total_frame_ms);

        // Capture mode always reports missed_budget = false; degradation is suppressed.
        let missed_budget = !in_capture && budget_ms.map(|b| controller_ms > b).unwrap_or(false);

        // Adaptation controller: adjust render scale within effective bounds when enabled.
        // Uses controller_ms from the previous frame (gpu_frame_ms when available,
        // otherwise total_frame_ms). Paused in Capture mode.
        if policy.allow_dynamic_resolution && !in_capture {
            if let Some(budget) = budget_ms {
                if controller_ms > budget {
                    // Over budget: step down quickly.
                    self.current_render_scale =
                        (self.current_render_scale - 0.1).max(eff_min_scale);
                } else if controller_ms < budget * 0.8 {
                    // Comfortably under budget: recover slowly to avoid oscillation.
                    self.current_render_scale =
                        (self.current_render_scale + 0.05).min(eff_max_scale);
                }
            }
        }

        self.last_prepare_instant = Some(prepare_start);
        self.frame_counter = self.frame_counter.wrapping_add(1);

        let reported_render_scale = self.current_render_scale;

        let stats = crate::renderer::stats::FrameStats {
            cpu_prepare_ms,
            // gpu_frame_ms is updated by the timestamp readback above when available;
            // propagate the most recent value from last_stats.
            gpu_frame_ms: self.last_stats.gpu_frame_ms,
            total_frame_ms,
            render_scale: reported_render_scale,
            missed_budget,
            upload_bytes,
            shadows_skipped: self.degradation_shadows_skipped,
            volume_quality_reduced: self.degradation_volume_quality_reduced,
            // effects_throttled is set by the render path; carry forward here so
            // prepare()-only callers still see the previous frame's value until
            // paint_to()/render() updates it.
            effects_throttled: self.degradation_effects_throttled,
            ..self.last_stats
        };
        self.last_stats = stats;
        stats
    }
}

// ---------------------------------------------------------------------------
// Clip boundary wireframe helpers (used by prepare_viewport_internal)
// ---------------------------------------------------------------------------

/// Wireframe outline for a clip box (12 edges as 2-point polyline strips).
fn clip_box_outline(
    center: [f32; 3],
    half: [f32; 3],
    orientation: [[f32; 3]; 3],
    colour: [f32; 4],
) -> PolylineItem {
    let ax = glam::Vec3::from(orientation[0]) * half[0];
    let ay = glam::Vec3::from(orientation[1]) * half[1];
    let az = glam::Vec3::from(orientation[2]) * half[2];
    let c = glam::Vec3::from(center);

    let corners = [
        c - ax - ay - az,
        c + ax - ay - az,
        c + ax + ay - az,
        c - ax + ay - az,
        c - ax - ay + az,
        c + ax - ay + az,
        c + ax + ay + az,
        c - ax + ay + az,
    ];
    let edges: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0), // bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // verticals
    ];

    let mut positions = Vec::with_capacity(24);
    let mut strip_lengths = Vec::with_capacity(12);
    for (a, b) in edges {
        positions.push(corners[a].to_array());
        positions.push(corners[b].to_array());
        strip_lengths.push(2u32);
    }

    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.default_colour = colour;
    item.line_width = 2.0;
    item
}

/// Wireframe outline for a clip sphere (three great circles).
fn clip_sphere_outline(center: [f32; 3], radius: f32, colour: [f32; 4]) -> PolylineItem {
    let c = glam::Vec3::from(center);
    let segs = 64usize;
    let mut positions = Vec::with_capacity((segs + 1) * 3);
    let mut strip_lengths = Vec::with_capacity(3);

    for axis in 0..3usize {
        let start = positions.len();
        for i in 0..=segs {
            let t = i as f32 / segs as f32 * std::f32::consts::TAU;
            let (s, cs) = t.sin_cos();
            let p = c + match axis {
                0 => glam::Vec3::new(cs * radius, s * radius, 0.0),
                1 => glam::Vec3::new(cs * radius, 0.0, s * radius),
                _ => glam::Vec3::new(0.0, cs * radius, s * radius),
            };
            positions.push(p.to_array());
        }
        strip_lengths.push((positions.len() - start) as u32);
    }

    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.default_colour = colour;
    item.line_width = 2.0;
    item
}

/// Wireframe outline for a clip cylinder (two end-cap circles + longitudinal lines).
fn clip_cylinder_outline(
    center: [f32; 3],
    axis: [f32; 3],
    radius: f32,
    half_length: f32,
    colour: [f32; 4],
) -> PolylineItem {
    let c = glam::Vec3::from(center);
    let ax = glam::Vec3::from(axis).normalize();

    // Build an orthonormal frame around the axis.
    let ref_v = if ax.y.abs() < 0.99 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let perp_u = ref_v.cross(ax).normalize();
    let perp_v = ax.cross(perp_u);

    let segs = 32usize;
    let long_lines = 8usize;
    let cap_verts = segs + 1;
    let total_cap = cap_verts * 2 + long_lines * 2;
    let mut positions = Vec::with_capacity(total_cap);
    let mut strip_lengths = Vec::with_capacity(2 + long_lines);

    // Two end-cap circles.
    for sign in [-1.0f32, 1.0] {
        let cap_center = c + ax * (sign * half_length);
        let start = positions.len();
        for i in 0..=segs {
            let t = i as f32 / segs as f32 * std::f32::consts::TAU;
            let (s, cs) = t.sin_cos();
            let p = cap_center + perp_u * (cs * radius) + perp_v * (s * radius);
            positions.push(p.to_array());
        }
        strip_lengths.push((positions.len() - start) as u32);
    }

    // Longitudinal lines connecting the two caps.
    for i in 0..long_lines {
        let t = i as f32 / long_lines as f32 * std::f32::consts::TAU;
        let (s, cs) = t.sin_cos();
        let offset = perp_u * (cs * radius) + perp_v * (s * radius);
        positions.push((c + ax * (-half_length) + offset).to_array());
        positions.push((c + ax * half_length + offset).to_array());
        strip_lengths.push(2);
    }

    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.default_colour = colour;
    item.line_width = 2.0;
    item
}

// ---------------------------------------------------------------------------
// Overlay label helpers
// ---------------------------------------------------------------------------

/// Project a world-space position to NDC.
/// Returns `None` only if the point is behind the camera (`clip.w <= 0`).
/// Does NOT reject points outside the [-1,1] viewport box.
fn project_to_ndc(pos: [f32; 3], view: &glam::Mat4, proj: &glam::Mat4) -> Option<[f32; 2]> {
    let clip = *proj * *view * glam::Vec3::from(pos).extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    Some([clip.x / clip.w, clip.y / clip.w])
}

/// Convert NDC coordinates to screen pixels (top-left origin).
fn ndc_to_screen_px(ndc: [f32; 2], vp_w: f32, vp_h: f32) -> [f32; 2] {
    [
        (ndc[0] * 0.5 + 0.5) * vp_w,
        (1.0 - (ndc[1] * 0.5 + 0.5)) * vp_h,
    ]
}

/// Returns true when the NDC point lies within the viewport square.
fn ndc_in_viewport(ndc: [f32; 2]) -> bool {
    ndc[0] >= -1.0 && ndc[0] <= 1.0 && ndc[1] >= -1.0 && ndc[1] <= 1.0
}

/// Clip a line segment [a, b] in NDC to the [-1,1]^2 viewport box
/// using the Liang-Barsky algorithm.
/// Returns the clipped endpoints, or `None` if the segment is entirely outside.
fn clip_line_ndc(a: [f32; 2], b: [f32; 2]) -> Option<([f32; 2], [f32; 2])> {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;

    // (p, q) pairs for left, right, bottom, top boundaries.
    for (p, q) in [
        (-dx, a[0] + 1.0),
        (dx, 1.0 - a[0]),
        (-dy, a[1] + 1.0),
        (dy, 1.0 - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }

    if t0 > t1 {
        return None;
    }
    Some((
        [a[0] + t0 * dx, a[1] + t0 * dy],
        [a[0] + t1 * dx, a[1] + t1 * dy],
    ))
}

/// Project a world-space position to screen pixels (top-left origin).
/// Returns `None` if behind the camera or outside the frustum.
fn project_to_screen(
    pos: [f32; 3],
    view: &glam::Mat4,
    proj: &glam::Mat4,
    vp_w: f32,
    vp_h: f32,
) -> Option<[f32; 2]> {
    let p = glam::Vec3::from(pos);
    let clip = *proj * *view * p.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    if ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0 {
        return None;
    }
    let x = (ndc_x * 0.5 + 0.5) * vp_w;
    let y = (1.0 - (ndc_y * 0.5 + 0.5)) * vp_h;
    Some([x, y])
}

/// Convert screen pixel coordinates to NDC.
#[inline]
fn px_to_ndc(px_x: f32, px_y: f32, vp_w: f32, vp_h: f32) -> [f32; 2] {
    [px_x / vp_w * 2.0 - 1.0, 1.0 - px_y / vp_h * 2.0]
}

/// Emit a solid-colour quad (6 vertices) in screen pixel coordinates.
fn emit_solid_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let tl = px_to_ndc(x0, y0, vp_w, vp_h);
    let tr = px_to_ndc(x1, y0, vp_w, vp_h);
    let bl = px_to_ndc(x0, y1, vp_w, vp_h);
    let br = px_to_ndc(x1, y1, vp_w, vp_h);
    let uv = [0.0, 0.0];
    let tex = 0.0;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        _pad: 0.0,
    };
    verts.extend_from_slice(&[v(tl), v(bl), v(tr), v(tr), v(bl), v(br)]);
}

/// Emit a textured quad (6 vertices) for a glyph in screen pixel coordinates.
fn emit_textured_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let tl = px_to_ndc(x0, y0, vp_w, vp_h);
    let tr = px_to_ndc(x1, y0, vp_w, vp_h);
    let bl = px_to_ndc(x0, y1, vp_w, vp_h);
    let br = px_to_ndc(x1, y1, vp_w, vp_h);
    let tex = 1.0;
    let v = |pos: [f32; 2], uv: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        _pad: 0.0,
    };
    // UV layout: top-left = uv_min, bottom-right = uv_max.
    verts.extend_from_slice(&[
        v(tl, uv_min),
        v(bl, [uv_min[0], uv_max[1]]),
        v(tr, [uv_max[0], uv_min[1]]),
        v(tr, [uv_max[0], uv_min[1]]),
        v(bl, [uv_min[0], uv_max[1]]),
        v(br, uv_max),
    ]);
}

/// Emit a thin screen-space line as a quad (6 vertices).
fn emit_line_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 {
        return;
    }
    let half = thickness * 0.5;
    let nx = -dy / len * half;
    let ny = dx / len * half;

    let p0 = px_to_ndc(x0 + nx, y0 + ny, vp_w, vp_h);
    let p1 = px_to_ndc(x0 - nx, y0 - ny, vp_w, vp_h);
    let p2 = px_to_ndc(x1 + nx, y1 + ny, vp_w, vp_h);
    let p3 = px_to_ndc(x1 - nx, y1 - ny, vp_w, vp_h);
    let uv = [0.0, 0.0];
    let tex = 0.0;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        _pad: 0.0,
    };
    verts.extend_from_slice(&[v(p0), v(p1), v(p2), v(p2), v(p1), v(p3)]);
}

/// Apply an opacity multiplier to a colour's alpha channel.
#[inline]
fn apply_opacity(colour: [f32; 4], opacity: f32) -> [f32; 4] {
    [colour[0], colour[1], colour[2], colour[3] * opacity]
}

/// Emit a rounded rectangle as solid quads: one center rect + four edge rects +
/// four corner fans.  This is a CPU tessellation approach that avoids shader
/// changes.
fn emit_rounded_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    radius: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let w = x1 - x0;
    let h = y1 - y0;
    let r = radius.min(w * 0.5).min(h * 0.5).max(0.0);

    if r < 0.5 {
        emit_solid_quad(verts, x0, y0, x1, y1, colour, vp_w, vp_h);
        return;
    }

    // Center cross (two rects that cover everything except the corners).
    // Horizontal bar (full width, inset top/bottom by r).
    emit_solid_quad(verts, x0, y0 + r, x1, y1 - r, colour, vp_w, vp_h);
    // Top bar (inset left/right by r, top edge).
    emit_solid_quad(verts, x0 + r, y0, x1 - r, y0 + r, colour, vp_w, vp_h);
    // Bottom bar.
    emit_solid_quad(verts, x0 + r, y1 - r, x1 - r, y1, colour, vp_w, vp_h);

    // Four corner fans.
    let corners = [
        (
            x0 + r,
            y0 + r,
            std::f32::consts::PI,
            std::f32::consts::FRAC_PI_2 * 3.0,
        ), // top-left
        (
            x1 - r,
            y0 + r,
            std::f32::consts::FRAC_PI_2 * 3.0,
            std::f32::consts::TAU,
        ), // top-right
        (x1 - r, y1 - r, 0.0, std::f32::consts::FRAC_PI_2), // bottom-right
        (
            x0 + r,
            y1 - r,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
        ), // bottom-left
    ];
    let segments = 6;
    let uv = [0.0, 0.0];
    let tex = 0.0;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        _pad: 0.0,
    };
    for (cx, cy, start, end) in corners {
        let center = px_to_ndc(cx, cy, vp_w, vp_h);
        for i in 0..segments {
            let a0 = start + (end - start) * i as f32 / segments as f32;
            let a1 = start + (end - start) * (i + 1) as f32 / segments as f32;
            let p0 = px_to_ndc(cx + a0.cos() * r, cy + a0.sin() * r, vp_w, vp_h);
            let p1 = px_to_ndc(cx + a1.cos() * r, cy + a1.sin() * r, vp_w, vp_h);
            verts.extend_from_slice(&[v(center), v(p0), v(p1)]);
        }
    }
}

/// Generate an OBB wireframe polyline for a VolumeItem by transforming its
/// bbox corners through the model matrix.
fn volume_obb_polyline(item: &crate::renderer::types::VolumeItem) -> crate::renderer::types::PolylineItem {
    let model = glam::Mat4::from_cols_array_2d(&item.model);
    let mn = glam::Vec3::from(item.bbox_min);
    let mx = glam::Vec3::from(item.bbox_max);
    let local = [
        glam::Vec3::new(mn.x, mn.y, mn.z),
        glam::Vec3::new(mx.x, mn.y, mn.z),
        glam::Vec3::new(mn.x, mx.y, mn.z),
        glam::Vec3::new(mx.x, mx.y, mn.z),
        glam::Vec3::new(mn.x, mn.y, mx.z),
        glam::Vec3::new(mx.x, mn.y, mx.z),
        glam::Vec3::new(mn.x, mx.y, mx.z),
        glam::Vec3::new(mx.x, mx.y, mx.z),
    ];
    let c: Vec<[f32; 3]> = local
        .iter()
        .map(|p| model.transform_point3(*p).to_array())
        .collect();
    obb_box_polyline(&c)
}

/// Generate a box wireframe polyline from 8 corners.
/// Corner indexing: bit 0=x, bit 1=y, bit 2=z (0=min, 1=max).
fn obb_box_polyline(c: &[[f32; 3]]) -> crate::renderer::types::PolylineItem {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();
    // Bottom face (z=min): 0,1,3,2,0
    positions.extend_from_slice(&[c[0], c[1], c[3], c[2], c[0]]);
    strip_lengths.push(5);
    // Top face (z=max): 4,5,7,6,4
    positions.extend_from_slice(&[c[4], c[5], c[7], c[6], c[4]]);
    strip_lengths.push(5);
    // Lateral edges
    for (lo, hi) in [(0usize, 4usize), (1, 5), (2, 6), (3, 7)] {
        positions.extend_from_slice(&[c[lo], c[hi]]);
        strip_lengths.push(2);
    }
    crate::renderer::types::PolylineItem {
        positions,
        strip_lengths,
        default_colour: [0.75, 0.75, 0.75, 1.0],
        line_width: 1.0,
        ..crate::renderer::types::PolylineItem::default()
    }
}

/// Generate a wireframe polyline for a Gaussian splat set.
/// <= 100 splats: three orthogonal rings (XY, XZ, YZ) per splat scaled by splat scale.
/// > 100 splats: OBB fitted via PCA on a subsample of positions.
fn splat_wireframe_polyline(
    positions: &[[f32; 3]],
    scales: &[[f32; 3]],
    model: [[f32; 4]; 4],
    count: usize,
) -> crate::renderer::types::PolylineItem {
    if count == 0 || positions.is_empty() {
        return crate::renderer::types::PolylineItem::default();
    }
    let model_mat = glam::Mat4::from_cols_array_2d(&model);
    if count <= 100 {
        splat_rings_polyline(positions, scales, model_mat)
    } else {
        splat_obb_polyline(positions, model_mat)
    }
}

fn splat_rings_polyline(
    positions: &[[f32; 3]],
    scales: &[[f32; 3]],
    model_mat: glam::Mat4,
) -> crate::renderer::types::PolylineItem {
    const SEGMENTS: usize = 32;
    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();
    for (pos, scale) in positions.iter().zip(scales.iter()) {
        let center = glam::Vec3::from(*pos);
        let [sx, sy, sz] = *scale;
        let rings: [(glam::Vec3, glam::Vec3, f32, f32); 3] = [
            (glam::Vec3::X, glam::Vec3::Y, sx, sy),
            (glam::Vec3::X, glam::Vec3::Z, sx, sz),
            (glam::Vec3::Y, glam::Vec3::Z, sy, sz),
        ];
        for (a1, a2, r1, r2) in &rings {
            for i in 0..=SEGMENTS {
                let t = std::f32::consts::TAU * i as f32 / SEGMENTS as f32;
                let p_local = center + (*a1) * (r1 * t.cos()) + (*a2) * (r2 * t.sin());
                let p_world = model_mat.transform_point3(p_local);
                all_positions.push(p_world.to_array());
            }
            strip_lengths.push((SEGMENTS + 1) as u32);
        }
    }
    crate::renderer::types::PolylineItem {
        positions: all_positions,
        strip_lengths,
        default_colour: [0.75, 0.75, 0.75, 1.0],
        line_width: 1.0,
        ..crate::renderer::types::PolylineItem::default()
    }
}

fn splat_obb_polyline(
    positions: &[[f32; 3]],
    model_mat: glam::Mat4,
) -> crate::renderer::types::PolylineItem {
    const N_SUBSAMPLE: usize = 10_000;
    let n = positions.len();
    // Compute centroid from subsample.
    let step = if n > N_SUBSAMPLE { n / N_SUBSAMPLE } else { 1 };
    let samples: Vec<glam::Vec3> = positions
        .iter()
        .step_by(step)
        .map(|p| glam::Vec3::from(*p))
        .collect();
    if samples.is_empty() {
        return crate::renderer::types::PolylineItem::default();
    }
    let centroid = samples.iter().copied().sum::<glam::Vec3>() / samples.len() as f32;
    // Compute 3x3 covariance matrix.
    let mut cov = [[0.0f32; 3]; 3];
    for p in &samples {
        let d = *p - centroid;
        let v = [d.x, d.y, d.z];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += v[i] * v[j];
            }
        }
    }
    let inv_n = 1.0 / samples.len() as f32;
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] *= inv_n;
        }
    }
    // Eigenvectors via Jacobi iteration.
    let (axes, _) = jacobi_eig_3x3(&cov);
    // Project ALL positions onto each axis to find exact extents.
    let mut min_ext = [f32::INFINITY; 3];
    let mut max_ext = [f32::NEG_INFINITY; 3];
    for p in positions {
        let d = glam::Vec3::from(*p) - centroid;
        let dv = [d.x, d.y, d.z];
        for i in 0..3 {
            let proj = dv[0] * axes[i][0] + dv[1] * axes[i][1] + dv[2] * axes[i][2];
            min_ext[i] = min_ext[i].min(proj);
            max_ext[i] = max_ext[i].max(proj);
        }
    }
    // Build 8 OBB corners in world space (object space -> model matrix).
    let axis: [glam::Vec3; 3] = [
        glam::Vec3::from(axes[0]),
        glam::Vec3::from(axes[1]),
        glam::Vec3::from(axes[2]),
    ];
    let center_obj = centroid
        + axis[0] * (min_ext[0] + max_ext[0]) * 0.5
        + axis[1] * (min_ext[1] + max_ext[1]) * 0.5
        + axis[2] * (min_ext[2] + max_ext[2]) * 0.5;
    let half = [
        (max_ext[0] - min_ext[0]) * 0.5,
        (max_ext[1] - min_ext[1]) * 0.5,
        (max_ext[2] - min_ext[2]) * 0.5,
    ];
    let signs: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [ 1.0,  1.0,  1.0],
    ];
    let corners: Vec<[f32; 3]> = signs
        .iter()
        .map(|s| {
            let p = center_obj
                + axis[0] * (s[0] * half[0])
                + axis[1] * (s[1] * half[1])
                + axis[2] * (s[2] * half[2]);
            model_mat.transform_point3(p).to_array()
        })
        .collect();
    obb_box_polyline(&corners)
}

/// Generate a wireframe polyline for a sprite batch.
/// <= 100 sprites: 4-edge quad outline per sprite.
/// > 100 sprites: AABB box from world-space positions.
fn sprite_wireframe_polyline(
    item: &crate::renderer::types::SpriteItem,
    camera: &crate::CameraFrame,
) -> crate::renderer::types::PolylineItem {
    let count = item.positions.len();
    if count == 0 {
        return crate::renderer::types::PolylineItem::default();
    }
    let model = glam::Mat4::from_cols_array_2d(&item.model);
    if count <= 100 {
        sprite_quad_outlines_polyline(item, camera, model)
    } else {
        let mut mn = glam::Vec3::splat(f32::INFINITY);
        let mut mx = glam::Vec3::splat(f32::NEG_INFINITY);
        for pos in &item.positions {
            let wp = model.transform_point3(glam::Vec3::from(*pos));
            mn = mn.min(wp);
            mx = mx.max(wp);
        }
        let corners: Vec<[f32; 3]> = [
            glam::Vec3::new(mn.x, mn.y, mn.z),
            glam::Vec3::new(mx.x, mn.y, mn.z),
            glam::Vec3::new(mn.x, mx.y, mn.z),
            glam::Vec3::new(mx.x, mx.y, mn.z),
            glam::Vec3::new(mn.x, mn.y, mx.z),
            glam::Vec3::new(mx.x, mn.y, mx.z),
            glam::Vec3::new(mn.x, mx.y, mx.z),
            glam::Vec3::new(mx.x, mx.y, mx.z),
        ]
        .iter()
        .map(|p| p.to_array())
        .collect();
        obb_box_polyline(&corners)
    }
}

/// Generate 4-edge quad outlines for each sprite in a batch.
///
/// Mirrors the sprite vertex shader corner computation:
/// - WorldSpace sprites: expand along camera right/up by half-size in world units.
/// - ScreenSpace sprites: convert NDC corners back to world space via inv_view_proj.
fn sprite_quad_outlines_polyline(
    item: &crate::renderer::types::SpriteItem,
    camera: &crate::CameraFrame,
    model: glam::Mat4,
) -> crate::renderer::types::PolylineItem {
    let view = &camera.render_camera.view;
    // Row 0 of the view matrix = camera right in world space.
    // Row 1 of the view matrix = camera up in world space.
    // glam Mat4 is column-major: view[col][row], matching view[0][0]/view[1][0]/view[2][0] in WGSL.
    let cam_right = glam::Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
    let cam_up = glam::Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);

    let view_proj = camera.render_camera.view_proj();
    let inv_view_proj = view_proj.inverse();
    let [vw, vh] = camera.viewport_size;
    let is_world_space =
        matches!(item.size_mode, crate::renderer::types::SpriteSizeMode::WorldSpace);

    // BL -> BR -> TR -> TL -> BL: a closed rectangle (4 edges, 5 positions per strip).
    const CORNERS: [(f32, f32); 5] =
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)];

    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();

    for i in 0..item.positions.len() {
        let world_pos = model.transform_point3(glam::Vec3::from(item.positions[i]));
        let size = if i < item.sizes.len() { item.sizes[i] } else { item.default_size };
        let rotation = if i < item.rotations.len() { item.rotations[i] } else { 0.0 };
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();
        let half = size * 0.5;

        let mut pts: Vec<[f32; 3]> = Vec::with_capacity(5);
        let mut ok = true;

        if is_world_space {
            for (cx, cy) in CORNERS {
                let rx = cos_r * cx - sin_r * cy;
                let ry = sin_r * cx + cos_r * cy;
                let p = world_pos + cam_right * (rx * half) + cam_up * (ry * half);
                pts.push(p.to_array());
            }
        } else {
            let clip_center = view_proj * world_pos.extend(1.0);
            if clip_center.w <= 0.0 {
                // Behind camera -- skip this sprite.
                ok = false;
            } else {
                let ndc_center = glam::Vec3::new(clip_center.x, clip_center.y, clip_center.z) / clip_center.w;
                for (cx, cy) in CORNERS {
                    let rx = cos_r * cx - sin_r * cy;
                    let ry = sin_r * cx + cos_r * cy;
                    let ndc = glam::Vec3::new(
                        ndc_center.x + rx * half / vw,
                        ndc_center.y + ry * half / vh,
                        ndc_center.z,
                    );
                    let world_h = inv_view_proj * ndc.extend(1.0);
                    if world_h.w.abs() < 1e-7 {
                        ok = false;
                        break;
                    }
                    pts.push((glam::Vec3::new(world_h.x, world_h.y, world_h.z) / world_h.w).to_array());
                }
            }
        }

        if ok && pts.len() == 5 {
            all_positions.extend_from_slice(&pts);
            strip_lengths.push(5);
        }
    }

    crate::renderer::types::PolylineItem {
        positions: all_positions,
        strip_lengths,
        default_colour: [0.75, 0.75, 0.75, 1.0],
        line_width: 1.0,
        ..crate::renderer::types::PolylineItem::default()
    }
}

/// Jacobi eigenvalue decomposition for a 3x3 symmetric matrix.
/// Returns (eigenvectors as rows, eigenvalues).
fn jacobi_eig_3x3(a: &[[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut m = *a;
    let mut v: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let mut max_off = 0.0f32;
        let (mut p, mut q) = (0usize, 1usize);
        for i in 0..3usize {
            for j in (i + 1)..3usize {
                let abs = m[i][j].abs();
                if abs > max_off {
                    max_off = abs;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-10 {
            break;
        }
        let tau = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let m_pp = m[p][p];
        let m_qq = m[q][q];
        let m_pq = m[p][q];
        m[p][p] = c * c * m_pp - 2.0 * s * c * m_pq + s * s * m_qq;
        m[q][q] = s * s * m_pp + 2.0 * s * c * m_pq + c * c * m_qq;
        m[p][q] = 0.0;
        m[q][p] = 0.0;
        for r in 0..3usize {
            if r != p && r != q {
                let m_pr = m[p][r];
                let m_qr = m[q][r];
                m[p][r] = c * m_pr - s * m_qr;
                m[r][p] = m[p][r];
                m[q][r] = s * m_pr + c * m_qr;
                m[r][q] = m[q][r];
            }
        }
        for r in 0..3usize {
            let v_rp = v[r][p];
            let v_rq = v[r][q];
            v[r][p] = c * v_rp - s * v_rq;
            v[r][q] = s * v_rp + c * v_rq;
        }
    }
    // Return eigenvectors as rows (transposed from columns of v).
    (
        [
            [v[0][0], v[1][0], v[2][0]],
            [v[0][1], v[1][1], v[2][1]],
            [v[0][2], v[1][2], v[2][2]],
        ],
        [m[0][0], m[1][1], m[2][2]],
    )
}

// ---------------------------------------------------------------------------
// Ruler label formatting
// ---------------------------------------------------------------------------

/// Format a distance value using a caller-supplied format pattern.
///
/// The pattern may contain one `{...}` placeholder with an optional precision
/// specifier, e.g. `"{:.3}"` or `"{:.2} m"`.  Anything outside the braces is
/// treated as a literal prefix / suffix.  Unrecognised patterns fall back to
/// three decimal places.
fn format_ruler_distance(distance: f32, fmt: Option<&str>) -> String {
    let pattern = fmt.unwrap_or("{:.3}");
    // Find the first `{...}` block.
    if let Some(open) = pattern.find('{') {
        if let Some(close_rel) = pattern[open..].find('}') {
            let close = open + close_rel;
            let spec = &pattern[open + 1..close]; // e.g. ":.3" or ""
            let prefix = &pattern[..open];
            let suffix = &pattern[close + 1..];
            let formatted = if let Some(prec_str) = spec.strip_prefix(":.") {
                // Strip trailing 'f' for patterns like "{:.3f}".
                let prec_str = prec_str.trim_end_matches('f');
                if let Ok(prec) = prec_str.parse::<usize>() {
                    format!("{distance:.prec$}")
                } else {
                    format!("{distance:.3}")
                }
            } else if spec.is_empty() || spec == ":" {
                format!("{distance}")
            } else {
                format!("{distance:.3}")
            };
            return format!("{prefix}{formatted}{suffix}");
        }
    }
    format!("{distance:.3}")
}
