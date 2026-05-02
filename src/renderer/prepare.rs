use super::types::{ClipShape, SceneEffects, ViewportEffects};
use super::*;
use wgpu::util::DeviceExt;

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
        // Phase G : GPU compute filtering.
        // Dispatch before the render pass. Completely skipped when list is empty (zero overhead).
        if !scene_fx.compute_filter_items.is_empty() {
            self.compute_filter_results =
                self.resources
                    .run_compute_filters(device, queue, scene_fx.compute_filter_items);
        } else {
            self.compute_filter_results.clear();
        }

        // Ensure built-in colormaps and matcaps are uploaded on first frame.
        self.resources.ensure_colormaps_initialized(device, queue);
        self.resources.ensure_matcaps_initialized(device, queue);

        let resources = &mut self.resources;
        let lighting = scene_fx.lighting;

        // Resolve scene items from the SurfaceSubmission seam.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items,
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
        // Phase 1 note: uses frame.camera : see multi-viewport-plan.md § shadow strategy.
        // -------------------------------------------------------------------
        let cascade_count = lighting.shadow_cascade_count.clamp(1, 4) as usize;
        let atlas_res = lighting.shadow_atlas_resolution.max(64);
        let tile_size = atlas_res / 2;

        let cascade_splits = compute_cascade_splits(
            frame.camera.render_camera.near.max(0.01),
            frame.camera.render_camera.far.max(1.0),
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
            sky_color: lighting.sky_color,
            hemisphere_intensity: lighting.hemisphere_intensity,
            ground_color: lighting.ground_color,
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

        // -- Instancing preparation --
        // Determine instancing mode BEFORE per-object uniforms so we can skip them.
        let visible_count = scene_items.iter().filter(|i| i.visible).count();
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
        let has_two_sided_items = scene_items
            .iter()
            .any(|i| i.material.is_two_sided());
        let has_matcap_items = scene_items.iter().any(|i| i.material.matcap_id.is_some());
        let has_param_vis_items = scene_items.iter().any(|i| i.material.param_vis.is_some());
        let has_wireframe_items = scene_items.iter().any(|i| i.render_as_wireframe);
        if !self.use_instancing
            || frame.viewport.wireframe_mode
            || has_scalar_items
            || has_two_sided_items
            || has_matcap_items
            || has_param_vis_items
            || has_wireframe_items
        {
            for item in scene_items {
                if resources
                    .mesh_store
                    .get(item.mesh_id)
                    .is_none()
                {
                    tracing::warn!(
                        mesh_index = item.mesh_id.index(),
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
                                .get(item.mesh_id)
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
                    wireframe: if frame.viewport.wireframe_mode || item.render_as_wireframe { 1 } else { 0 },
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
                    use_matcap: if m.matcap_id.is_some() { 1 } else { 0 },
                    matcap_blendable: m.matcap_id.map_or(0, |id| if id.blendable { 1 } else { 0 }),
                    _pad2: 0,
                    use_face_color: u32::from(item.active_attribute.as_ref().map_or(false, |a| {
                        a.kind == crate::resources::AttributeKind::FaceColor
                    })),
                    uv_vis_mode: m.param_vis.map_or(0, |pv| pv.mode as u32),
                    uv_vis_scale: m.param_vis.map_or(8.0, |pv| pv.scale),
                    backface_policy: match m.backface_policy {
                        crate::scene::material::BackfacePolicy::Cull => 0,
                        crate::scene::material::BackfacePolicy::Identical => 1,
                        crate::scene::material::BackfacePolicy::DifferentColor(_) => 2,
                        crate::scene::material::BackfacePolicy::Tint(_) => 3,
                        crate::scene::material::BackfacePolicy::Pattern { pattern, .. } => {
                            4 + pattern as u32
                        }
                    },
                    backface_color: match m.backface_policy {
                        crate::scene::material::BackfacePolicy::DifferentColor(c) => {
                            [c[0], c[1], c[2], 1.0]
                        }
                        crate::scene::material::BackfacePolicy::Tint(factor) => {
                            [factor, 0.0, 0.0, 1.0]
                        }
                        crate::scene::material::BackfacePolicy::Pattern { color, .. } => {
                            [color[0], color[1], color[2], 1.0]
                        }
                        _ => [0.0; 4],
                    },
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
                    use_matcap: 0,
                    matcap_blendable: 0,
                    _pad2: 0,
                    use_face_color: 0,
                    uv_vis_mode: 0,
                    uv_vis_scale: 8.0,
                    backface_policy: 0,
                    backface_color: [0.0; 4],
                };

                // Write uniform data : use get() to read buffer references, then drop.
                {
                    let mesh = resources
                        .mesh_store
                        .get(item.mesh_id)
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

                // Rebuild the object bind group if material/attribute/LUT/matcap changed.
                resources.update_mesh_texture_bind_group(
                    device,
                    item.mesh_id,
                    item.material.texture_id,
                    item.material.normal_map_id,
                    item.material.ao_map_id,
                    item.colormap_id,
                    item.active_attribute.as_ref().map(|a| a.name.as_str()),
                    item.material.matcap_id,
                );
            }
        }

        if self.use_instancing {
            resources.ensure_instanced_pipelines(device);

            // Generation-based cache: skip batch rebuild and GPU upload when nothing changed.
            // Phase 2: wireframe_mode removed from cache key : wireframe rendering
            // uses the per-object wireframe_pipeline, not the instanced path, so
            // instance data is now viewport-agnostic.
            let cache_valid = frame.scene.generation == self.last_scene_generation
                && frame.interaction.selection_generation == self.last_selection_generation
                && scene_items.len() == self.last_scene_items_count;

            if !cache_valid {
                // Cache miss : rebuild batches and upload instance data.
                let mut sorted_items: Vec<&SceneRenderItem> = scene_items
                    .iter()
                    .filter(|item| {
                        item.visible
                            && item.active_attribute.is_none()
                            && !item.material.is_two_sided()
                            && item.material.matcap_id.is_none()
                            && item.material.param_vis.is_none()
                            && resources
                                .mesh_store
                                .get(item.mesh_id)
                                .is_some()
                    })
                    .collect();

                sorted_items.sort_unstable_by_key(|item| {
                    (
                        item.mesh_id.index(),
                        item.material.texture_id,
                        item.material.normal_map_id,
                        item.material.ao_map_id,
                    )
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
                                });
                            }

                            // Build per-instance AABBs alongside instance data.
                            // All items in a batch share the same mesh_id (batch key), so
                            // mesh.index_count is the same for every item — look it up once.
                            let batch_idx = instanced_batches.len() as u32;
                            let mesh_index_count = resources
                                .mesh_store
                                .get(rep.mesh_id)
                                .map(|m| m.index_count)
                                .unwrap_or(0);
                            for item in batch_items {
                                if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                                    let model =
                                        glam::Mat4::from_cols_array_2d(&item.model);
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

                self.cached_instance_data = all_instances;
                self.cached_instanced_batches = instanced_batches;

                resources.upload_instance_data(device, queue, &self.cached_instance_data);
                resources.upload_aabb_and_batch_meta(device, queue, &all_aabbs, &batch_metas);

                self.instanced_batches = self.cached_instanced_batches.clone();

                self.last_scene_generation = frame.scene.generation;
                self.last_selection_generation = frame.interaction.selection_generation;
                self.last_scene_items_count = scene_items.len();

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
                && !self.cached_instance_data.is_empty()
            {
                let instance_count = self.cached_instance_data.len() as u32;
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
                    let cpu_frustum =
                        crate::camera::frustum::Frustum::from_view_proj(&vp_mat);
                    let frustum_uniform = crate::resources::FrustumUniform {
                        planes: std::array::from_fn(|i| crate::resources::FrustumPlane {
                            normal: cpu_frustum.planes[i].normal.into(),
                            distance: cpu_frustum.planes[i].d,
                        }),
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
                    queue.submit(std::iter::once(encoder.finish()));
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
                let gpu_data = resources.upload_glyph_set(device, queue, item);
                self.glyph_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // SciVis Phase M8 : polyline GPU data upload.
        // ------------------------------------------------------------------
        self.polyline_gpu_data.clear();
        let vp_size = frame.camera.viewport_size;
        if !frame.scene.polylines.is_empty() {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.scene.polylines {
                if item.positions.is_empty() {
                    continue;
                }
                let gpu_data = resources.upload_polyline(device, queue, item, vp_size);
                self.polyline_gpu_data.push(gpu_data);

                // Phase 11: auto-generate GlyphItems for node/edge vector quantities.
                if !item.node_vectors.is_empty() {
                    resources.ensure_glyph_pipeline(device);
                    let g = crate::quantities::polyline_node_vectors_to_glyphs(item);
                    if !g.positions.is_empty() {
                        let gd = resources.upload_glyph_set(device, queue, &g);
                        self.glyph_gpu_data.push(gd);
                    }
                }
                if !item.edge_vectors.is_empty() {
                    resources.ensure_glyph_pipeline(device);
                    let g = crate::quantities::polyline_edge_vectors_to_glyphs(item);
                    if !g.positions.is_empty() {
                        let gd = resources.upload_glyph_set(device, queue, &g);
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
                    colormap_id: None,
                    default_color: item.color,
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
        if !frame.scene.gpu_implicit.is_empty() {
            resources.ensure_implicit_pipeline(device);
            for item in &frame.scene.gpu_implicit {
                if item.primitives.is_empty() {
                    continue;
                }
                let gpu = resources.upload_implicit_item(device, item);
                self.implicit_gpu_data.push(gpu);
            }
        }

        // ------------------------------------------------------------------
        // Phase 17 : GPU marching cubes compute dispatch.
        // ------------------------------------------------------------------
        self.mc_gpu_data.clear();
        if !frame.scene.gpu_mc_jobs.is_empty() {
            resources.ensure_mc_pipelines(device);
            self.mc_gpu_data =
                resources.run_mc_jobs(device, queue, &frame.scene.gpu_mc_jobs);
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
        if !frame.scene.streamtube_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.scene.streamtube_items {
                if item.positions.is_empty() || item.strip_lengths.is_empty() {
                    continue;
                }
                let gpu_data = resources.upload_streamtube(device, queue, item);
                if gpu_data.index_count > 0 {
                    self.streamtube_gpu_data.push(gpu_data);
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
            // Extract ClipPlane structs from clip_objects for volume cap fill support.
            let clip_planes_for_vol: Vec<crate::renderer::types::ClipPlane> = frame
                .effects
                .clip_objects
                .iter()
                .filter(|o| o.enabled)
                .filter_map(|o| {
                    if let ClipShape::Plane {
                        normal,
                        distance,
                        cap_color,
                    } = o.shape
                    {
                        Some(crate::renderer::types::ClipPlane {
                            normal,
                            distance,
                            enabled: true,
                            cap_color,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            // Phase 5: under budget pressure with allow_volume_quality_reduction, double the
            // step size (half the sample count) to reduce GPU raymarch cost.
            let vol_step_multiplier =
                if self.last_stats.missed_budget
                    && self.performance_policy.allow_volume_quality_reduction
                {
                    2.0_f32
                } else {
                    1.0_f32
                };
            for item in &frame.scene.volumes {
                let gpu = resources.upload_volume_frame(
                    device,
                    queue,
                    item,
                    &clip_planes_for_vol,
                    vol_step_multiplier,
                );
                self.volume_gpu_data.push(gpu);
            }
        }

        // -- Frame stats --
        {
            let total = scene_items.len() as u32;
            let visible = scene_items.iter().filter(|i| i.visible).count() as u32;
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
                        .get(batch.mesh_id)
                    {
                        draw_calls += 1;
                        triangles += (mesh.index_count / 3) as u64 * batch.instance_count as u64;
                    }
                }
            } else {
                for item in scene_items {
                    if !item.visible {
                        continue;
                    }
                    if let Some(mesh) = resources
                        .mesh_store
                        .get(item.mesh_id)
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
                gpu_culling_active: self.gpu_culling_enabled,
                ..self.last_stats
            };
        }

        // ------------------------------------------------------------------
        // Shadow depth pass : CSM: render each cascade into its atlas tile.
        // Phase 5: skip the pass entirely when over budget and shadow reduction is allowed.
        // ------------------------------------------------------------------
        let skip_shadows = self.last_stats.missed_budget
            && self.performance_policy.allow_shadow_reduction;
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
                && !self.cached_instance_data.is_empty()
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

                let instance_count = self.cached_instance_data.len() as u32;
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
                            let cpu_frustum =
                                crate::camera::frustum::Frustum::from_view_proj(
                                    &cascade_view_projs[c],
                                );
                            let frustum_uniform = crate::resources::FrustumUniform {
                                planes: std::array::from_fn(|i| crate::resources::FrustumPlane {
                                    normal: cpu_frustum.planes[i].normal.into(),
                                    distance: cpu_frustum.planes[i].d,
                                }),
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

                            let Some(pipeline) =
                                resources.shadow_instanced_cull_pipeline.as_ref()
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
                                shadow_pass
                                    .set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                shadow_pass.draw_indexed_indirect(
                                    shadow_indirect_buf,
                                    bi as u64 * 20,
                                );
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
                                let Some(mesh) = resources
                                    .mesh_store
                                    .get(batch.mesh_id)
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
                            if !item.visible {
                                continue;
                            }
                            if item.material.opacity < 1.0 {
                                continue;
                            }
                            let Some(mesh) = resources
                                .mesh_store
                                .get(item.mesh_id)
                            else {
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
            SurfaceSubmission::Flat(items) => items,
        };

        // Capture before the resources mutable borrow so it's accessible inside the block.
        let gp_cascade0_mat = self.last_cascade0_shadow_mat.to_cols_array_2d();

        {
            let resources = &mut self.resources;

            // Upload clip planes + clip volume uniforms from clip_objects.
            {
                let mut planes = [[0.0f32; 4]; 6];
                let mut count = 0u32;
                let mut clip_vol_uniform: ClipVolumeUniform = bytemuck::Zeroable::zeroed(); // volume_type=0

                for obj in viewport_fx.clip_objects.iter().filter(|o| o.enabled) {
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
                        } if clip_vol_uniform.volume_type == 0 => {
                            clip_vol_uniform.volume_type = 2;
                            clip_vol_uniform.box_center = center;
                            clip_vol_uniform.box_half_extents = half_extents;
                            clip_vol_uniform.box_col0 = orientation[0];
                            clip_vol_uniform.box_col1 = orientation[1];
                            clip_vol_uniform.box_col2 = orientation[2];
                        }
                        ClipShape::Sphere { center, radius }
                            if clip_vol_uniform.volume_type == 0 =>
                        {
                            clip_vol_uniform.volume_type = 3;
                            clip_vol_uniform.sphere_center = center;
                            clip_vol_uniform.sphere_radius = radius;
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
                        bytemuck::cast_slice(&[clip_vol_uniform]),
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
                    bytemuck::cast_slice(&[clip_vol_uniform]),
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
                        color_minor: [0.35, 0.35, 0.35, 0.4 * minor_fade],
                        color_major: [0.40, 0.40, 0.40, 0.4 + 0.2 * minor_fade],
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
                    crate::renderer::types::GroundPlaneMode::SolidColor => 3,
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
                    color: gp.color,
                    shadow_color: gp.shadow_color,
                    light_vp: gp_cascade0_mat,
                    tan_half_fov,
                    aspect,
                    tile_size: gp.tile_size,
                    shadow_bias: 0.002,
                    mode: mode_u32,
                    shadow_opacity: gp.shadow_opacity,
                    _pad: [0.0; 2],
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
                if !item.visible || !item.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    color: [0.0; 4], // unused by mask shader
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
        }

        // X-ray buffers for selected objects.
        let mut xray_object_buffers: Vec<(crate::resources::mesh_store::MeshId, wgpu::Buffer, wgpu::BindGroup)> = Vec::new();
        if frame.interaction.xray_selected {
            let resources = &self.resources;
            for item in scene_items {
                if !item.visible || !item.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    color: frame.interaction.xray_color,
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

        // Clip plane overlays : generated automatically from clip_objects with a color set.
        let mut clip_plane_fill_buffers = Vec::new();
        let mut clip_plane_line_buffers = Vec::new();
        for obj in viewport_fx.clip_objects.iter().filter(|o| o.enabled) {
            let Some(base_color) = obj.color else {
                continue;
            };
            if let ClipShape::Plane {
                normal, distance, ..
            } = obj.shape
            {
                let n = glam::Vec3::from(normal);
                // Shader plane equation: dot(p, n) + distance = 0, so the plane
                // sits at -n * distance from the origin.
                let center = n * (-distance);
                let active = obj.active;
                let hovered = obj.hovered || active;

                let fill_color = if active {
                    [
                        base_color[0] * 0.5,
                        base_color[1] * 0.5,
                        base_color[2] * 0.5,
                        base_color[3] * 0.5,
                    ]
                } else if hovered {
                    [
                        base_color[0] * 0.8,
                        base_color[1] * 0.8,
                        base_color[2] * 0.8,
                        base_color[3] * 0.6,
                    ]
                } else {
                    [
                        base_color[0] * 0.5,
                        base_color[1] * 0.5,
                        base_color[2] * 0.5,
                        base_color[3] * 0.3,
                    ]
                };
                let border_color = if active {
                    [base_color[0], base_color[1], base_color[2], 0.9]
                } else if hovered {
                    [base_color[0], base_color[1], base_color[2], 0.8]
                } else {
                    [
                        base_color[0] * 0.9,
                        base_color[1] * 0.9,
                        base_color[2] * 0.9,
                        0.6,
                    ]
                };

                let overlay = crate::interaction::clip_plane::ClipPlaneOverlay {
                    center,
                    normal: n,
                    extent: obj.extent,
                    fill_color,
                    border_color,
                    hovered,
                    active,
                };
                clip_plane_fill_buffers.push(
                    self.resources
                        .create_clip_plane_fill_overlay(device, &overlay),
                );
                clip_plane_line_buffers.push(
                    self.resources
                        .create_clip_plane_line_overlay(device, &overlay),
                );
            } else {
                // Box/Sphere: generate wireframe polyline.
                // ensure_polyline_pipeline must be called before upload_polyline; it is a
                // no-op if already initialised, so calling it here is always safe.
                self.resources.ensure_polyline_pipeline(device);
                match obj.shape {
                    ClipShape::Box {
                        center,
                        half_extents,
                        orientation,
                    } => {
                        let polyline =
                            clip_box_outline(center, half_extents, orientation, base_color);
                        let vp_size = frame.camera.viewport_size;
                        let gpu = self
                            .resources
                            .upload_polyline(device, queue, &polyline, vp_size);
                        self.polyline_gpu_data.push(gpu);
                    }
                    ClipShape::Sphere { center, radius } => {
                        let polyline = clip_sphere_outline(center, radius, base_color);
                        let vp_size = frame.camera.viewport_size;
                        let gpu = self
                            .resources
                            .upload_polyline(device, queue, &polyline, vp_size);
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
                    cap_color,
                } = obj.shape
                {
                    let plane_n = glam::Vec3::from(normal);
                    for item in scene_items.iter().filter(|i| i.visible) {
                        let Some(mesh) = self
                            .resources
                            .mesh_store
                            .get(item.mesh_id)
                        else {
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
                            let bc = item.material.base_color;
                            let color = cap_color.unwrap_or([bc[0], bc[1], bc[2], 1.0]);
                            let buf = self.resources.upload_cap_geometry(device, &cap, color);
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
        //    an anti-aliased outline ring to the outline color texture.
        //
        // The outline color texture is later composited onto the main target
        // by the composite pass in paint()/render().
        // ------------------------------------------------------------------
        if frame.interaction.outline_selected
            && !self.viewport_slots[vp_idx]
                .outline_object_buffers
                .is_empty()
        {
            let w = frame.camera.viewport_size[0] as u32;
            let h = frame.camera.viewport_size[1] as u32;

            // Ensure per-viewport HDR state exists (provides outline textures).
            self.ensure_viewport_hdr(
                device,
                queue,
                vp_idx,
                w.max(1),
                h.max(1),
                frame.effects.post_process.ssaa_factor.max(1),
            );

            // Write edge-detection uniform (color, radius, viewport size).
            {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let edge_uniform = OutlineEdgeUniform {
                    color: frame.interaction.outline_color,
                    radius: frame.interaction.outline_width_px,
                    viewport_w: w as f32,
                    viewport_h: h as f32,
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
            let outlines_ptr =
                &slot_ref.outline_object_buffers as *const Vec<OutlineObjectBuffers>;
            let camera_bg_ptr = &slot_ref.camera_bind_group as *const wgpu::BindGroup;
            let slot_hdr = slot_ref.hdr.as_ref().unwrap();
            let mask_view_ptr = &slot_hdr.outline_mask_view as *const wgpu::TextureView;
            let color_view_ptr = &slot_hdr.outline_color_view as *const wgpu::TextureView;
            let depth_view_ptr = &slot_hdr.outline_depth_view as *const wgpu::TextureView;
            let edge_bg_ptr = &slot_hdr.outline_edge_bind_group as *const wgpu::BindGroup;
            // SAFETY: slot fields remain valid for the duration of this function;
            // no other code modifies these fields here.
            let (outlines, camera_bg, mask_view, color_view, depth_view, edge_bg) = unsafe {
                (
                    &*outlines_ptr,
                    &*camera_bg_ptr,
                    &*mask_view_ptr,
                    &*color_view_ptr,
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
                    let Some(mesh) = self
                        .resources
                        .mesh_store
                        .get(outlined.mesh_id)
                    else {
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
            }

            // Pass 2: fullscreen edge detection (reads mask, writes color).
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("outline_edge_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
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
            if let Some(sel_ref) = &frame.interaction.sub_selection {
                let needs_rebuild = {
                    let slot = &self.viewport_slots[vp_idx];
                    slot.sub_highlight_generation != sel_ref.version
                        || slot.sub_highlight.is_none()
                };
                if needs_rebuild {
                    self.resources.ensure_sub_highlight_pipelines(device);
                    let data = self.resources.build_sub_highlight(
                        device,
                        queue,
                        sel_ref,
                        frame.interaction.sub_highlight_face_fill_color,
                        frame.interaction.sub_highlight_edge_color,
                        frame.interaction.sub_highlight_edge_width_px,
                        frame.interaction.sub_highlight_vertex_size_px,
                        w,
                        h,
                    );
                    let slot = &mut self.viewport_slots[vp_idx];
                    slot.sub_highlight = Some(data);
                    slot.sub_highlight_generation = sel_ref.version;
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
                    let ascent = self.resources.glyph_atlas.font_ascent(font_index, label.font_size);

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
                        let bg_color = apply_opacity(label.background_color, opacity);
                        if label.border_radius > 0.0 {
                            emit_rounded_quad(
                                &mut verts,
                                bx0, by0, bx1, by1,
                                label.border_radius,
                                bg_color,
                                vp_w, vp_h,
                            );
                        } else {
                            emit_solid_quad(
                                &mut verts,
                                bx0, by0, bx1, by1,
                                bg_color,
                                vp_w, vp_h,
                            );
                        }
                    }

                    // Leader line.
                    if label.leader_line {
                        if let Some(wa) = label.world_anchor {
                            let world_px = project_to_screen(wa, view, proj, vp_w, vp_h);
                            if let Some(wp) = world_px {
                                emit_line_quad(
                                    &mut verts,
                                    wp[0], wp[1],
                                    text_x, text_y + layout.height * 0.5,
                                    1.5,
                                    apply_opacity(label.leader_color, opacity),
                                    vp_w, vp_h,
                                );
                            }
                        }
                    }

                    // Glyph quads.
                    let text_color = apply_opacity(label.color, opacity);
                    for gq in &layout.quads {
                        let gx = text_x + gq.pos[0];
                        let gy = text_y + ascent + gq.pos[1];
                        emit_textured_quad(
                            &mut verts,
                            gx, gy,
                            gx + gq.size[0], gy + gq.size[1],
                            gq.uv_min, gq.uv_max,
                            text_color,
                            vp_w, vp_h,
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
                    let Some(lut) = self.resources.get_colormap_rgba(bar.colormap_id).map(|l| l.to_vec()) else {
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
                        let layout = self.resources.glyph_atlas.layout_text(
                            &text, tick_fs, bar.font, device,
                        );
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
                        self.resources.glyph_atlas.layout_text(t, title_fs, bar.font, device).total_width
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
                        let tick_oh  = max_tick_w / 2.0;
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
                        crate::renderer::types::ScalarBarAnchor::TopLeft => (
                            bar.margin_px + inset_left,
                            bar.margin_px + title_h + bg_pad,
                        ),
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
                        let tick_overhang  = max_tick_w / 2.0;
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
                        bg_x0, bg_y0, bg_x1, bg_y1,
                        3.0,
                        bar.background_color,
                        vp_w, vp_h,
                    );

                    // Gradient strip: 64 solid quads sampled from the colormap LUT.
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
                        let color = [
                            r as f32 / 255.0,
                            g as f32 / 255.0,
                            b as f32 / 255.0,
                            a as f32 / 255.0,
                        ];
                        emit_solid_quad(&mut verts, qx0, qy0, qx1, qy1, color, vp_w, vp_h);
                    }

                    // Tick labels.
                    let ascent = self.resources.glyph_atlas.font_ascent(font_index, tick_fs);
                    for (i, (text, tw, th)) in tick_data.iter().enumerate() {
                        let t = i as f32 / (tick_count - 1) as f32;
                        let layout = self.resources.glyph_atlas.layout_text(
                            text, tick_fs, bar.font, device,
                        );

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
                                gx, gy,
                                gx + gq.size[0], gy + gq.size[1],
                                gq.uv_min, gq.uv_max,
                                bar.label_color,
                                vp_w, vp_h,
                            );
                        }
                    }

                    // Optional title above the gradient strip.
                    if let Some(ref title_text) = bar.title {
                        let layout = self.resources.glyph_atlas.layout_text(
                            title_text, title_fs, bar.font, device,
                        );
                        let title_ascent = self.resources.glyph_atlas.font_ascent(font_index, title_fs);
                        // Centre the title over the gradient strip.
                        let tx = bar_x + (strip_w - layout.total_width) * 0.5;
                        let ty = bar_y - title_h;
                        for gq in &layout.quads {
                            let gx = tx + gq.pos[0];
                            let gy = ty + title_ascent + gq.pos[1];
                            emit_textured_quad(
                                &mut verts,
                                gx, gy,
                                gx + gq.size[0], gy + gq.size[1],
                                gq.uv_min, gq.uv_max,
                                bar.label_color,
                                vp_w, vp_h,
                            );
                        }
                    }
                }

                // Upload any newly rasterized glyphs (may overlap with label upload above).
                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                    let end_ndc   = project_to_ndc(ruler.end,   view, proj);

                    // Cull entirely when either endpoint is behind the camera.
                    let (Some(sndc), Some(endc)) = (start_ndc, end_ndc) else { continue };

                    // Clip the segment to the viewport NDC box [-1,1]^2.
                    // This keeps the line visible when only one end is off-screen sideways.
                    let Some((csndc, cendc)) = clip_line_ndc(sndc, endc) else { continue };

                    let [sx, sy] = ndc_to_screen_px(csndc, vp_w, vp_h);
                    let [ex, ey] = ndc_to_screen_px(cendc, vp_w, vp_h);

                    // Track which original endpoints are within the viewport (for end caps).
                    let start_on_screen = ndc_in_viewport(sndc);
                    let end_on_screen   = ndc_in_viewport(endc);

                    // Main ruler line.
                    emit_line_quad(
                        &mut verts,
                        sx, sy, ex, ey,
                        ruler.line_width_px,
                        ruler.color,
                        vp_w, vp_h,
                    );

                    // End caps only at endpoints that are actually on screen.
                    if ruler.end_caps {
                        let dx = ex - sx;
                        let dy = ey - sy;
                        let len = (dx * dx + dy * dy).sqrt().max(0.001);
                        let cap_half = 5.0;
                        let px = -dy / len * cap_half;
                        let py =  dx / len * cap_half;

                        if start_on_screen {
                            emit_line_quad(
                                &mut verts,
                                sx - px, sy - py,
                                sx + px, sy + py,
                                ruler.line_width_px,
                                ruler.color,
                                vp_w, vp_h,
                            );
                        }
                        if end_on_screen {
                            emit_line_quad(
                                &mut verts,
                                ex - px, ey - py,
                                ex + px, ey + py,
                                ruler.line_width_px,
                                ruler.color,
                                vp_w, vp_h,
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
                    let ascent = self.resources.glyph_atlas.font_ascent(font_index, ruler.font_size);

                    // Center the label above the midpoint with a small gap.
                    let lx = mid_x - layout.total_width * 0.5;
                    let ly = mid_y - layout.height - 6.0;

                    // Semi-transparent background box.
                    let pad = 3.0;
                    emit_solid_quad(
                        &mut verts,
                        lx - pad, ly - pad,
                        lx + layout.total_width + pad, ly + layout.height + pad,
                        [0.0, 0.0, 0.0, 0.55],
                        vp_w, vp_h,
                    );

                    // Glyph quads.
                    for gq in &layout.quads {
                        let gx = lx + gq.pos[0];
                        let gy = ly + ascent + gq.pos[1];
                        emit_textured_quad(
                            &mut verts,
                            gx, gy,
                            gx + gq.size[0], gy + gq.size[1],
                            gq.uv_min, gq.uv_max,
                            ruler.label_color,
                            vp_w, vp_h,
                        );
                    }
                }

                // Upload any newly rasterized glyphs.
                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

        // Wall-clock duration since the previous prepare() call approximates the frame interval.
        let total_frame_ms = self
            .last_prepare_instant
            .map(|t| t.elapsed().as_secs_f32() * 1000.0)
            .unwrap_or(0.0);

        // Snapshot geometry upload bytes accumulated since the last frame, then reset.
        let upload_bytes = self.resources.frame_upload_bytes;
        self.resources.frame_upload_bytes = 0;

        let (scene_fx, viewport_fx) = frame.effects.split();
        self.prepare_scene_internal(device, queue, frame, &scene_fx);
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx);

        let cpu_prepare_ms = prepare_start.elapsed().as_secs_f32() * 1000.0;

        let policy = self.performance_policy;
        let budget_ms = policy.target_fps.map(|fps| 1000.0 / fps);
        let missed_budget = budget_ms
            .map(|b| total_frame_ms > b)
            .unwrap_or(false);

        // Adaptation controller: adjust render scale within policy bounds when enabled.
        // Uses total_frame_ms from the *previous* frame so the controller reacts one
        // frame after the overrun, which is the earliest it can have the measurement.
        if policy.allow_dynamic_resolution {
            if let Some(budget) = budget_ms {
                if total_frame_ms > budget {
                    // Over budget: step down quickly.
                    self.current_render_scale =
                        (self.current_render_scale - 0.1).max(policy.min_render_scale);
                } else if total_frame_ms < budget * 0.8 {
                    // Comfortably under budget: recover slowly to avoid oscillation.
                    self.current_render_scale =
                        (self.current_render_scale + 0.05).min(policy.max_render_scale);
                }
            }
        }

        self.last_prepare_instant = Some(prepare_start);
        self.frame_counter = self.frame_counter.wrapping_add(1);

        let stats = crate::renderer::stats::FrameStats {
            cpu_prepare_ms,
            // gpu_frame_ms is updated by the timestamp readback above when available;
            // propagate the most recent value from last_stats.
            gpu_frame_ms: self.last_stats.gpu_frame_ms,
            total_frame_ms,
            render_scale: self.current_render_scale,
            missed_budget,
            upload_bytes,
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
    color: [f32; 4],
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
    item.default_color = color;
    item.line_width = 2.0;
    item
}

/// Wireframe outline for a clip sphere (three great circles).
fn clip_sphere_outline(center: [f32; 3], radius: f32, color: [f32; 4]) -> PolylineItem {
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
    item.default_color = color;
    item.line_width = 2.0;
    item
}

// ---------------------------------------------------------------------------
// Overlay label helpers
// ---------------------------------------------------------------------------

/// Project a world-space position to NDC.
/// Returns `None` only if the point is behind the camera (`clip.w <= 0`).
/// Does NOT reject points outside the [-1,1] viewport box.
fn project_to_ndc(
    pos: [f32; 3],
    view: &glam::Mat4,
    proj: &glam::Mat4,
) -> Option<[f32; 2]> {
    let clip = *proj * *view * glam::Vec3::from(pos).extend(1.0);
    if clip.w <= 0.0 { return None; }
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
        ( dx, 1.0 - a[0]),
        (-dy, a[1] + 1.0),
        ( dy, 1.0 - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 { return None; }
        } else {
            let r = q / p;
            if p < 0.0 { t0 = t0.max(r); } else { t1 = t1.min(r); }
        }
    }

    if t0 > t1 { return None; }
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
    [
        px_x / vp_w * 2.0 - 1.0,
        1.0 - px_y / vp_h * 2.0,
    ]
}

/// Emit a solid-colour quad (6 vertices) in screen pixel coordinates.
fn emit_solid_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    color: [f32; 4],
    vp_w: f32, vp_h: f32,
) {
    let tl = px_to_ndc(x0, y0, vp_w, vp_h);
    let tr = px_to_ndc(x1, y0, vp_w, vp_h);
    let bl = px_to_ndc(x0, y1, vp_w, vp_h);
    let br = px_to_ndc(x1, y1, vp_w, vp_h);
    let uv = [0.0, 0.0];
    let tex = 0.0;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos, uv, color, use_texture: tex, _pad: 0.0,
    };
    verts.extend_from_slice(&[v(tl), v(bl), v(tr), v(tr), v(bl), v(br)]);
}

/// Emit a textured quad (6 vertices) for a glyph in screen pixel coordinates.
fn emit_textured_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    color: [f32; 4],
    vp_w: f32, vp_h: f32,
) {
    let tl = px_to_ndc(x0, y0, vp_w, vp_h);
    let tr = px_to_ndc(x1, y0, vp_w, vp_h);
    let bl = px_to_ndc(x0, y1, vp_w, vp_h);
    let br = px_to_ndc(x1, y1, vp_w, vp_h);
    let tex = 1.0;
    let v = |pos: [f32; 2], uv: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos, uv, color, use_texture: tex, _pad: 0.0,
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
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    thickness: f32,
    color: [f32; 4],
    vp_w: f32, vp_h: f32,
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
        position: pos, uv, color, use_texture: tex, _pad: 0.0,
    };
    verts.extend_from_slice(&[v(p0), v(p1), v(p2), v(p2), v(p1), v(p3)]);
}

/// Apply an opacity multiplier to a colour's alpha channel.
#[inline]
fn apply_opacity(color: [f32; 4], opacity: f32) -> [f32; 4] {
    [color[0], color[1], color[2], color[3] * opacity]
}

/// Emit a rounded rectangle as solid quads: one center rect + four edge rects +
/// four corner fans.  This is a CPU tessellation approach that avoids shader
/// changes.
fn emit_rounded_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    radius: f32,
    color: [f32; 4],
    vp_w: f32, vp_h: f32,
) {
    let w = x1 - x0;
    let h = y1 - y0;
    let r = radius.min(w * 0.5).min(h * 0.5).max(0.0);

    if r < 0.5 {
        emit_solid_quad(verts, x0, y0, x1, y1, color, vp_w, vp_h);
        return;
    }

    // Center cross (two rects that cover everything except the corners).
    // Horizontal bar (full width, inset top/bottom by r).
    emit_solid_quad(verts, x0, y0 + r, x1, y1 - r, color, vp_w, vp_h);
    // Top bar (inset left/right by r, top edge).
    emit_solid_quad(verts, x0 + r, y0, x1 - r, y0 + r, color, vp_w, vp_h);
    // Bottom bar.
    emit_solid_quad(verts, x0 + r, y1 - r, x1 - r, y1, color, vp_w, vp_h);

    // Four corner fans.
    let corners = [
        (x0 + r, y0 + r, std::f32::consts::PI, std::f32::consts::FRAC_PI_2 * 3.0),       // top-left
        (x1 - r, y0 + r, std::f32::consts::FRAC_PI_2 * 3.0, std::f32::consts::TAU),      // top-right
        (x1 - r, y1 - r, 0.0, std::f32::consts::FRAC_PI_2),                               // bottom-right
        (x0 + r, y1 - r, std::f32::consts::FRAC_PI_2, std::f32::consts::PI),              // bottom-left
    ];
    let segments = 6;
    let uv = [0.0, 0.0];
    let tex = 0.0;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos, uv, color, use_texture: tex, _pad: 0.0,
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
