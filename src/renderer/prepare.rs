use super::types::{ClipShape, SceneEffects, ViewportEffects};
use super::*;

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
        if !self.use_instancing
            || frame.viewport.wireframe_mode
            || has_scalar_items
            || has_two_sided_items
            || has_matcap_items
            || has_param_vis_items
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
            for item in &frame.scene.volumes {
                let gpu = resources.upload_volume_frame(device, queue, item, &clip_planes_for_vol);
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
            };
        }

        // ------------------------------------------------------------------
        // Shadow depth pass : CSM: render each cascade into its atlas tile.
        // ------------------------------------------------------------------
        if lighting.shadows_enabled && !scene_items.is_empty() {
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
    }

    /// Upload per-frame data to GPU buffers and render the shadow pass.
    /// Call before `paint()`.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &FrameData) {
        let (scene_fx, viewport_fx) = frame.effects.split();
        self.prepare_scene_internal(device, queue, frame, &scene_fx);
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx);
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
