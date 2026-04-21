use super::*;

impl ViewportRenderer {
    /// Upload per-frame data to GPU buffers and render the shadow pass.
    /// Call before `paint()`.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &FrameData) {
        // Phase G — GPU compute filtering.
        // Dispatch before the render pass. Completely skipped when list is empty (zero overhead).
        if !frame.effects.compute_filter_items.is_empty() {
            self.compute_filter_results =
                self.resources
                    .run_compute_filters(device, queue, &frame.effects.compute_filter_items);
        } else {
            self.compute_filter_results.clear();
        }

        // Ensure built-in colormaps are uploaded on first frame.
        self.resources.ensure_colormaps_initialized(device, queue);

        // Ensure a per-viewport camera slot exists for this viewport index.
        // Must happen before the `resources` borrow below.
        self.ensure_viewport_camera_slot(device, frame.camera.viewport_index);

        let resources = &mut self.resources;
        let lighting = &frame.effects.lighting;

        // Resolve scene items from the SurfaceSubmission seam.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items,
        };

        // Compute scene center / extent for shadow framing.
        let (shadow_center, shadow_extent) =
            if let Some(extent) = frame.effects.lighting.shadow_extent_override {
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
            for plane in frame.effects.clip_planes.iter().filter(|p| p.enabled).take(6) {
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
                viewport_width: frame.camera.viewport_size[0].max(1.0),
                viewport_height: frame.camera.viewport_size[1].max(1.0),
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
            let clip_vol_uniform = ClipVolumeUniform::from_clip_volume(&frame.effects.clip_volume);
            queue.write_buffer(
                &resources.clip_volume_uniform_buf,
                0,
                bytemuck::cast_slice(&[clip_vol_uniform]),
            );
        }

        // Upload camera uniform.
        let camera_uniform = frame.camera.render_camera.camera_uniform();
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
        if let Some((vp_buf, _)) = self.per_viewport_cameras.get(frame.camera.viewport_index) {
            queue.write_buffer(vp_buf, 0, bytemuck::cast_slice(&[camera_uniform]));
        }

        // Upload lights uniform.
        // IBL fields from environment map settings.
        let (ibl_enabled, ibl_intensity, ibl_rotation, show_skybox) =
            if let Some(env) = &frame.effects.environment {
                if resources.ibl_irradiance_view.is_some() {
                    (1u32, env.intensity, env.rotation, if env.show_skybox { 1u32 } else { 0 })
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

        // Per-object uniform writes — needed for the non-instanced path, wireframe mode,
        // and for any items with active scalar attributes or two-sided materials
        // (both bypass the instanced path).
        //
        // Also updates each mesh's `object_bind_group` when the material/attribute key changes,
        // keeping the combined (object-uniform + texture + LUT + scalar-buf) bind group consistent.
        let has_scalar_items = scene_items
            .iter()
            .any(|i| i.active_attribute.is_some());
        let has_two_sided_items = scene_items.iter().any(|i| i.two_sided);
        if !self.use_instancing || frame.viewport.wireframe_mode || has_scalar_items || has_two_sided_items {
            for item in scene_items {
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
                    wireframe: if frame.viewport.wireframe_mode { 1 } else { 0 },
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
            let cache_valid = frame.scene.generation == self.last_scene_generation
                && frame.interaction.selection_generation == self.last_selection_generation
                && frame.viewport.wireframe_mode == self.last_wireframe_mode
                && scene_items.len() == self.last_scene_items_count;

            if !cache_valid {
                // Cache miss — rebuild batches and upload instance data.

                // Collect visible items with valid meshes, then sort by batch key.
                // Items with active scalar attributes or two-sided rasterization are
                // excluded from instancing — they need per-object draw pipelines.
                let mut sorted_items: Vec<&SceneRenderItem> = scene_items
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
                                    wireframe: if frame.viewport.wireframe_mode { 1 } else { 0 },
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
                self.last_scene_generation = frame.scene.generation;
                self.last_selection_generation = frame.interaction.selection_generation;
                self.last_wireframe_mode = frame.viewport.wireframe_mode;
                self.last_scene_items_count = scene_items.len();

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
        if frame.interaction.outline_selected {
            for item in scene_items {
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
                    color: frame.interaction.outline_color,
                    pixel_offset: frame.interaction.outline_width_px,
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
        if frame.interaction.xray_selected {
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
                resources
                    .xray_object_buffers
                    .push((item.mesh_index, buf, bg));
            }
        }

        // Update gizmo.
        if let Some(model) = frame.interaction.gizmo_model {
            resources.update_gizmo_uniform(queue, model);
            resources.update_gizmo_mesh(
                device,
                queue,
                frame.interaction.gizmo_mode,
                frame.interaction.gizmo_hovered,
                frame.interaction.gizmo_space_orientation,
            );
        }

        // Upload grid uniform (full-screen analytical shader — no vertex buffers needed).
        if frame.viewport.show_grid {
            let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
            if !eye.is_finite() {
                tracing::warn!(eye_x = eye.x, eye_y = eye.y, eye_z = eye.z,
                    "grid skipped: eye_position is non-finite (camera distance overflow?)");
            } else {
            let view_proj_mat = frame.camera.render_camera.view_proj().to_cols_array_2d();

            // Adaptive LOD spacing — snap to next power of 10 above the target world
            // coverage.  Avoid log10/powf: they are imprecise near exact decade boundaries
            // (e.g. log10(10.0) may return 0.9999999 or 1.0000001 in f32, making ceil
            // flip between 1 and 2 each frame and causing the grid to oscillate).
            // A multiply loop is exact and has no boundary ambiguity.
            let (spacing, minor_fade) = if frame.viewport.grid_cell_size > 0.0 {
                (frame.viewport.grid_cell_size, 1.0_f32)
            } else {
                let vertical_depth = (eye.y - frame.viewport.grid_y).abs().max(1.0);
                let world_per_pixel = 2.0 * (frame.camera.render_camera.fov / 2.0).tan() * vertical_depth
                    / frame.camera.viewport_size[1].max(1.0);
                let target = (world_per_pixel * 60.0).max(1e-9_f32);
                let mut s = 1.0_f32;
                let mut iters = 0u32;
                while s < target {
                    s *= 10.0;
                    iters += 1;
                }
                // Fade minor lines out as we approach the LOD boundary so that the
                // 10× spacing jump is gradual rather than a sudden pop.
                // ratio ∈ (0.1, 1.0]: 0.1 = just entered this LOD, 1.0 = about to leave.
                let ratio = (target / s).clamp(0.0, 1.0);
                let fade = if ratio < 0.5 {
                    1.0_f32
                } else {
                    let t = (ratio - 0.5) * 2.0; // 0..1
                    1.0 - t * t * (3.0 - 2.0 * t) // smooth step down
                };
                tracing::debug!(
                    eye_y = eye.y,
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

            // Snap eye.xz to the nearest spacing_major multiple so the GPU works
            // with hit.xz - snap_origin (small offset) rather than raw world coords.
            // spacing_major is a power of 10, so snap_origin is exactly representable in f32.
            let spacing_major = spacing * 10.0;
            let snap_x = (eye.x / spacing_major).floor() * spacing_major;
            let snap_z = (eye.z / spacing_major).floor() * spacing_major;
            tracing::debug!(
                spacing_minor = spacing,
                spacing_major,
                snap_x,
                snap_z,
                eye_x = eye.x,
                eye_y = eye.y,
                eye_z = eye.z,
                "grid snap"
            );

            // Camera-to-world rotation: compute from orientation quaternion.
            // Columns are [right, up, back] where back = camera +Z (away from scene).
            // This is exact (no matrix inversion) and stable at any camera distance.
            let orient = frame.camera.render_camera.orientation;
            let right = orient * glam::Vec3::X;
            let up    = orient * glam::Vec3::Y;
            let back  = orient * glam::Vec3::Z;
            let cam_to_world = [
                [right.x, right.y, right.z, 0.0_f32],
                [up.x,    up.y,    up.z,    0.0_f32],
                [back.x,  back.y,  back.z,  0.0_f32],
            ];
            let aspect = frame.camera.viewport_size[0] / frame.camera.viewport_size[1].max(1.0);
            let tan_half_fov = (frame.camera.render_camera.fov / 2.0).tan();

            let uniform = GridUniform {
                view_proj: view_proj_mat,
                cam_to_world,
                tan_half_fov,
                aspect,
                _pad_ivp: [0.0; 2],
                eye_pos: frame.camera.render_camera.eye_position,
                grid_y: frame.viewport.grid_y,
                spacing_minor: spacing,
                spacing_major,
                snap_origin: [snap_x, snap_z],
                // Minor lines fade out as we approach the LOD boundary.
                // Major lines dim from 0.8 -> 0.4 in sync so that at the transition
                // the old major lines (which become new minor lines) are already at
                // the new minor alpha — no visible alpha jump.
                color_minor: [0.35, 0.35, 0.35, 0.4 * minor_fade],
                color_major: [0.40, 0.40, 0.40, 0.4 + 0.2 * minor_fade],
            };
            queue.write_buffer(
                &resources.grid_uniform_buf,
                0,
                bytemuck::cast_slice(&[uniform]),
            );
            } // end else (eye is finite)
        }

        resources.constraint_line_buffers.clear();
        for overlay in &frame.interaction.constraint_overlays {
            let buf = resources.create_constraint_overlay(device, overlay);
            resources.constraint_line_buffers.push(buf);
        }

        resources.clip_plane_fill_buffers.clear();
        resources.clip_plane_line_buffers.clear();
        for overlay in &frame.interaction.clip_plane_overlays {
            let fill = resources.create_clip_plane_fill_overlay(device, overlay);
            resources.clip_plane_fill_buffers.push(fill);
            let lines = resources.create_clip_plane_line_overlay(device, overlay);
            resources.clip_plane_line_buffers.push(lines);
        }

        // Cap geometry for section-view cross-section fill.
        resources.cap_buffers.clear();
        if frame.effects.cap_fill_enabled {
            let active_planes: Vec<_> = frame.effects.clip_planes.iter().filter(|p| p.enabled).collect();
            for plane in &active_planes {
                let plane_n = glam::Vec3::from(plane.normal);
                for item in scene_items.iter().filter(|i| i.visible) {
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
        if frame.viewport.show_axes_indicator && frame.camera.viewport_size[0] > 0.0 && frame.camera.viewport_size[1] > 0.0
        {
            let verts = crate::widgets::axes_indicator::build_axes_geometry(
                frame.camera.viewport_size[0],
                frame.camera.viewport_size[1],
                frame.camera.render_camera.orientation,
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
        // SciVis Phase M8 — polyline GPU data upload.
        // Zero-cost when polylines vec is empty (no pipeline created, no uploads).
        // ------------------------------------------------------------------
        self.polyline_gpu_data.clear();
        if !frame.scene.polylines.is_empty() {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.scene.polylines {
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
        if !frame.scene.streamtube_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.scene.streamtube_items {
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
        if !frame.scene.volumes.is_empty() {
            resources.ensure_volume_pipeline(device);
            for item in &frame.scene.volumes {
                let gpu = resources.upload_volume_frame(device, queue, item, &frame.effects.clip_planes);
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
                        .get(crate::resources::mesh_store::MeshId(batch.mesh_index))
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
        if frame.effects.lighting.shadows_enabled && !scene_items.is_empty() {
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
                        let cascade_frustum = crate::camera::frustum::Frustum::from_view_proj(
                            &cascade_view_projs[cascade],
                        );

                        for item in scene_items.iter() {
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
        if frame.interaction.outline_selected && !resources.outline_object_buffers.is_empty() {
            let w = frame.camera.viewport_size[0] as u32;
            let h = frame.camera.viewport_size[1] as u32;
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
}
