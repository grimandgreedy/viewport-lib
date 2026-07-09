//! Non-instanced (per-object) mesh draw preparation.

use super::*;

impl ViewportRenderer {
    /// Non-instanced (per-object) mesh draw preparation: compute the per-item
    /// material/feature flags, write one `ObjectUniform` per scene item, and build
    /// or reuse the per-item bind groups (plus the wireframe-mode uniform pool).
    pub(super) fn prepare_per_object(
        resources: &mut DeviceResources,
        mesh_uniforms: &mut PerObjectState,
        use_instancing: bool,
        scene_items: &[SceneRenderItem],
        instanceable: &[bool],
        frame_index: u64,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) -> u32 {
        // Count of per-item bind groups actually (re)built this frame, reported
        // in FrameStats so a cache that is silently missing is visible.
        let mut bind_groups_built = 0u32;
        // Collect per-item uniforms when wireframe mode is on so we can give each
        // visible item its own bind group (the mesh's shared object_uniform_buf gets
        // overwritten when multiple items reference the same MeshId).
        let mut wireframe_uniforms: Vec<ObjectUniform> = Vec::new();
        let collect_wf_uniforms = frame.viewport.wireframe_mode;
        // Enter the per-object write loop when at least one item needs it. An
        // item is non-instanceable for any of the reasons is_instanceable
        // tests (scalar colouring, two-sided, matcap, param-vis, a position or
        // normal override, skinning, a compute filter, or hidden), so
        // `instanceable` already folds all of those in: scanning it is cheaper
        // than rescanning the materials once per flag. Wireframe and
        // normal-visualization items take the per-object path even when they
        // would otherwise instance, so test those two directly.
        let any_non_instanceable = instanceable.iter().any(|&b| !b);
        let any_wireframe_or_normals = scene_items
            .iter()
            .any(|i| i.settings.wireframe || i.show_normals);
        if !use_instancing
            || frame.viewport.wireframe_mode
            || any_non_instanceable
            || any_wireframe_or_normals
        {
            // The per-frame slot Vec maps this frame's item index to a bind
            // group for the render path. Clear it and size it to the item list:
            // slots for items on the instanced path stay None, and stale entries
            // from a previous frame's different item must not leak through.
            if mesh_uniforms.bind_groups.len() < scene_items.len() {
                mesh_uniforms
                    .bind_groups
                    .resize_with(scene_items.len(), || None);
            }
            for slot in mesh_uniforms.bind_groups.iter_mut() {
                *slot = None;
            }
            // Counts how many per-object items this frame have shared each
            // pick_id, so items with a duplicate pick_id (or the shared
            // PickId::NONE default) get distinct cache entries instead of
            // stomping one another's uniform buffer.
            let mut pick_occurrences: std::collections::HashMap<u64, u32> =
                std::collections::HashMap::new();
            for (item_idx, item) in scene_items.iter().enumerate() {
                // When instancing is active, skip items that will be rendered
                // via the instanced path. They don't need per-object uniform
                // writes; writing them anyway causes O(n) write_buffer calls
                // for the whole scene whenever any single item is two-sided.
                //
                // Items whose mesh has a GPU position or normal override bound
                // (`set_position_override_buffer` / `set_normal_override_buffer`)
                // MUST go through this per-item write path. The override
                // binding at group 1 binding 13/14 is selected by the shader's
                // `has_position_override` / `has_normal_override` flag, which
                // lives in the per-item `ObjectUniform`. Skipping the write
                // leaves the flag at the default 0, the shader ignores the
                // override binding, and the consumer's compute output silently
                // does nothing.
                // Items that go through the instanced path do not need a per-item
                // uniform write, unless something forces them back to per-object:
                // wireframe mode, a wireframe or warp item, or normal visualization.
                if use_instancing
                    && instanceable[item_idx]
                    && !frame.viewport.wireframe_mode
                    && !item.settings.wireframe
                    && item.warp_attribute.is_none()
                    && !item.show_normals
                {
                    continue;
                }

                // Wireframe edge and normal-line buffers are built lazily on
                // the first frame a view needs them, not at upload time.
                if item.show_normals {
                    resources.ensure_normal_lines(device, item.mesh_id);
                }
                if item.settings.wireframe || frame.viewport.wireframe_mode {
                    resources.ensure_edge_indices(device, item.mesh_id);
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
                let cm = common_material(item);
                let obj_uniform = ObjectUniform {
                    model: cm.model,
                    colour: cm.colour,
                    selected: cm.selected,
                    wireframe: if frame.viewport.wireframe_mode || item.settings.wireframe {
                        1
                    } else {
                        0
                    },
                    ambient: cm.ambient,
                    diffuse: cm.diffuse,
                    specular: cm.specular,
                    shininess: cm.shininess,
                    has_texture: cm.has_texture,
                    use_pbr: cm.use_pbr,
                    metallic: cm.metallic,
                    roughness: cm.roughness,
                    has_normal_map: cm.has_normal_map,
                    has_ao_map: cm.has_ao_map,
                    has_attribute: has_attr,
                    scalar_min: s_min,
                    scalar_max: s_max,
                    receive_shadows: cm.receive_shadows,
                    nan_colour: item.nan_colour.unwrap_or([0.0; 4]),
                    use_nan_colour: if item.nan_colour.is_some() { 1 } else { 0 },
                    use_matcap: if m.matcap_id().is_some() { 1 } else { 0 },
                    matcap_blendable: m
                        .matcap_id()
                        .map_or(0, |id| if id.blendable { 1 } else { 0 }),
                    unlit: cm.unlit,
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
                    has_position_override: {
                        let mesh = resources.mesh_store.get(item.mesh_id);
                        if mesh.map_or(false, |m| m.position_override_buffer.is_some()) {
                            1
                        } else {
                            0
                        }
                    },
                    has_normal_override: {
                        let mesh = resources.mesh_store.get(item.mesh_id);
                        if mesh.map_or(false, |m| m.normal_override_buffer.is_some()) {
                            1
                        } else {
                            0
                        }
                    },
                    emissive: m.emissive,
                    use_flat: cm.use_flat,
                    alpha_mode: match m.alpha_mode {
                        crate::scene::material::AlphaMode::Opaque => 0,
                        crate::scene::material::AlphaMode::Mask(_) => 1,
                        crate::scene::material::AlphaMode::Blend => 2,
                    },
                    alpha_cutoff: match m.alpha_mode {
                        crate::scene::material::AlphaMode::Mask(c) => c,
                        _ => 0.5,
                    },
                    has_metallic_roughness_tex: if m.metallic_roughness_texture_id.is_some() {
                        1
                    } else {
                        0
                    },
                    has_emissive_tex: if m.emissive_texture_id.is_some() {
                        1
                    } else {
                        0
                    },
                    uv_transform: cm.uv_transform,
                    deform_flags: resources.deform.flag_bits(item.mesh_id),
                    _pad_after_deform: 0,
                    ao_range: cm.ao_range,
                    metallic_range: m.metallic_range,
                    roughness_range: m.roughness_range,
                };

                // Collect per-item uniform for wireframe per-item bind groups.
                if collect_wf_uniforms && !item.settings.hidden {
                    wireframe_uniforms.push(obj_uniform);
                }

                // The normal-visualization uniform feeds only the normal-line
                // pass, so assemble and write it only for items showing normals.
                if item.show_normals {
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
                        receive_shadows: 1,
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
                        has_position_override: 0,
                        has_normal_override: 0,
                        emissive: [0.0; 3],
                        use_flat: 0,
                        alpha_mode: 0,
                        alpha_cutoff: 0.5,
                        has_metallic_roughness_tex: 0,
                        has_emissive_tex: 0,
                        uv_transform: [0.0, 0.0, 1.0, 1.0],
                        deform_flags: 0,
                        _pad_after_deform: 0,
                        ao_range: [0.0, 1.0],
                        metallic_range: [0.0, 1.0],
                        roughness_range: [0.0, 1.0],
                    };
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                        queue.write_buffer(
                            &mesh.normal_uniform_buf,
                            0,
                            bytemuck::cast_slice(&[normal_obj_uniform]),
                        );
                    }
                }

                // Rebuild the object bind group if material/attribute/LUT/matcap/warp changed.
                resources.update_mesh_texture_bind_group(
                    device,
                    item.mesh_id,
                    item.material.texture_id,
                    item.material.normal_map_id,
                    item.material.ao_map_id,
                    item.colourmap_id,
                    item.active_attribute.as_ref().map(|a| a.name.as_str()),
                    item.material.matcap_id(),
                    item.warp_attribute.as_deref(),
                    item.material.metallic_roughness_texture_id,
                    item.material.emissive_texture_id,
                );

                // Per-object draw resources are cached so each item keeps its own
                // uniform buffer and bind group. Multiple scene items can share
                // the same MeshId; the mesh's shared object_uniform_buf is stomped
                // by whichever item wrote last, so the per-object cache guarantees
                // each item draws with its own transform/material.
                //
                // The cache key is pick_id plus an occurrence counter. A single
                // pickable object keys on (pick_id, 0) so the entry survives the
                // host reordering the item list. The occurrence disambiguates
                // items that share a pick_id: every non-pickable item defaults to
                // PickId::NONE, and one object can be drawn by several items (such
                // as a volume mesh submitted as both a coloured surface and a
                // pickable cell mesh). Without it they would collide on one entry
                // and all draw with the last item's transform and material.
                {
                    use crate::renderer::per_object_state::PerObjectKey;
                    let pick = item.settings.pick_id.0;
                    let occurrence = {
                        let counter = pick_occurrences.entry(pick).or_insert(0);
                        let n = *counter;
                        *counter += 1;
                        n
                    };
                    let key = PerObjectKey {
                        pick_id: pick,
                        occurrence,
                    };
                    let uniform_size = std::mem::size_of::<ObjectUniform>() as u64;
                    let entry = mesh_uniforms.cache.entry(key).or_insert_with(|| {
                        let buf = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("per_item_object_uniform"),
                            size: uniform_size,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        crate::renderer::per_object_state::PerObjectCacheEntry {
                            uniform_buf: buf,
                            bind_group: None,
                            cache_key: 0,
                            last_uniform: None,
                            last_frame: frame_index,
                        }
                    });
                    entry.last_frame = frame_index;

                    // Skip the uniform write when nothing about this object's
                    // ObjectUniform changed (the common case for static scene
                    // geometry the camera only moves past). Writing it is the
                    // bulk of the per-object cost at scale.
                    let uniform_changed = entry.last_uniform.as_ref().map_or(true, |u| {
                        bytemuck::bytes_of(u) != bytemuck::bytes_of(&obj_uniform)
                    });
                    if uniform_changed {
                        queue.write_buffer(
                            &entry.uniform_buf,
                            0,
                            bytemuck::cast_slice(&[obj_uniform]),
                        );
                        entry.last_uniform = Some(obj_uniform);
                        // Keep the mesh's shared object uniform in sync too. It
                        // feeds the fallback bind group and the point-shadow
                        // pass; per-object items draw via their own bind group,
                        // so a stale shared buffer for an unchanged item is
                        // harmless, but a changed one must be propagated.
                        if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                            queue.write_buffer(
                                &mesh.object_uniform_buf,
                                0,
                                bytemuck::cast_slice(&[obj_uniform]),
                            );
                        }
                    }

                    // Pass the cached key so the build skips create_bind_group
                    // when nothing changed. Only treat the stored key as valid
                    // when a bind group has already been built for this object.
                    let prev_key = entry.bind_group.as_ref().map(|_| entry.cache_key);
                    let built = resources.build_per_item_object_bind_group(
                        device,
                        item.mesh_id,
                        &entry.uniform_buf,
                        item.material.texture_id,
                        item.material.normal_map_id,
                        item.material.ao_map_id,
                        item.colourmap_id,
                        item.active_attribute.as_ref().map(|a| a.name.as_str()),
                        item.material.matcap_id(),
                        item.warp_attribute.as_deref(),
                        item.material.metallic_roughness_texture_id,
                        item.material.emissive_texture_id,
                        prev_key,
                    );
                    // `built` is Some only on a miss (or first build); on a hit
                    // the build returns None and the cached bind group is kept.
                    if let Some((bg, key)) = built {
                        bind_groups_built += 1;
                        entry.bind_group = Some(bg);
                        entry.cache_key = key;
                    }

                    // Populate this frame's slot for the render path (cheap
                    // reference-counted clone of the cached bind group). Clone
                    // into a local first so the `entry` borrow of `cache` ends
                    // before the disjoint `bind_groups` field is written.
                    let slot_bg = entry.bind_group.clone();
                    mesh_uniforms.bind_groups[item_idx] = slot_bg;
                }
            }
        }

        // Drop cache entries for objects not seen for a while, so a long
        // streaming session does not carry resources for evicted objects
        // forever. The grace window tolerates an object briefly leaving the
        // per-object path (a frame of frustum-edge culling) without losing its
        // bind group.
        const CACHE_GRACE_FRAMES: u64 = 60;
        mesh_uniforms
            .cache
            .retain(|_, e| e.last_frame + CACHE_GRACE_FRAMES >= frame_index);

        // Build per-item wireframe bind groups so each visible item gets its own
        // object uniform, avoiding the shared-MeshId overwrite problem.
        if !wireframe_uniforms.is_empty() {
            let n = wireframe_uniforms.len();
            let uniform_size = std::mem::size_of::<ObjectUniform>() as u64;

            // Grow the buffer/bind-group pools if needed. We never shrink them.
            while mesh_uniforms.wireframe_uniform_bufs.len() < n {
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
                                &resources.content.fallback_lut_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: resources.content.fallback_scalar_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(
                                resources
                                    .content
                                    .fallback_matcap_view
                                    .as_ref()
                                    .unwrap_or(&resources.fallback_texture.view),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: resources
                                .content
                                .fallback_face_colour_buf
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: resources.content.fallback_warp_buf.as_entire_binding(),
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
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: resources
                                .content
                                .fallback_position_override_buf
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: resources
                                .content
                                .fallback_normal_override_buf
                                .as_entire_binding(),
                        },
                    ],
                });
                mesh_uniforms.wireframe_uniform_bufs.push(buf);
                mesh_uniforms.wireframe_bind_groups.push(bg);
            }

            // Write each item's uniform into its dedicated buffer.
            for (i, uniform) in wireframe_uniforms.iter().enumerate() {
                queue.write_buffer(
                    &mesh_uniforms.wireframe_uniform_bufs[i],
                    0,
                    bytemuck::cast_slice(std::slice::from_ref(uniform)),
                );
            }
        }

        bind_groups_built
    }

    /// Rebuild or drop the cached per-object render bundle for this frame.
    ///
    /// Eligible frames are the all-per-object case (instancing selected but no
    /// batch formed) with plain solid meshes: no wireframe mode or per-item
    /// wireframe/normals/attribute/warp/deform features, no compute-filter
    /// index overrides, no registered deformers, and the LDR path (the HDR
    /// path records its own scene pass). The bundle stores the opaque draws
    /// in item order; blended items are listed for immediate depth-sorted
    /// drawing after the bundle. Per-item transforms and colours flow through
    /// the uniform buffers referenced by the recorded bind groups, so camera
    /// and object motion do not re-record; changes to the item set, material
    /// bind groups, or LOD-resolved meshes do (via the key hash and the
    /// bind-groups-built counter).
    pub(super) fn update_per_object_bundle(&mut self, device: &wgpu::Device, frame: &FrameData) {
        /// Below this many opaque items the recording cost is not worth
        /// caching against.
        const MIN_BUNDLE_ITEMS: usize = 64;
        self.last_stats.per_object_bundle_cached = false;
        // Measurement kill-switch so the bundle can be A/B'd in one binary.
        static DISABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DISABLE.get_or_init(|| std::env::var_os("VIEWPORT_DISABLE_PER_OBJECT_BUNDLE").is_some())
        {
            self.per_object_bundle = None;
            return;
        }

        let plan = 'plan: {
            if frame.effects.post_process.enabled
                || frame.viewport.wireframe_mode
                || !self.compute_filter_results.is_empty()
                || !self.instancing.use_instancing
                || !self.instancing.batches.is_empty()
                || !self.resources.deform.meshes.is_empty()
                || self.prepared_surfaces.len() < MIN_BUNDLE_ITEMS
            {
                break 'plan None;
            }
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            frame.camera.viewport_index.hash(&mut h);
            self.prepared_surfaces.len().hash(&mut h);
            let mut transparent: Vec<usize> = Vec::new();
            let mut opaque = 0usize;
            for (i, item) in self.prepared_surfaces.iter().enumerate() {
                if item.settings.hidden {
                    (i, 0u8).hash(&mut h);
                    continue;
                }
                if item.settings.wireframe
                    || item.show_normals
                    || item.warp_attribute.is_some()
                    || item.active_attribute.is_some()
                    || item.deform_instance.is_some()
                {
                    break 'plan None;
                }
                if self.resources.mesh_store.get(item.mesh_id).is_none() {
                    (i, 1u8).hash(&mut h);
                    continue;
                }
                if item.settings.opacity < 1.0 || item.material.is_blend() {
                    (i, 2u8).hash(&mut h);
                    transparent.push(i);
                    continue;
                }
                (
                    i,
                    3u8,
                    item.mesh_id,
                    item.material.is_two_sided(),
                    item.settings.pick_id.0,
                )
                    .hash(&mut h);
                opaque += 1;
            }
            if opaque < MIN_BUNDLE_ITEMS {
                break 'plan None;
            }
            Some((h.finish(), transparent))
        };

        let Some((key, transparent)) = plan else {
            self.per_object_bundle = None;
            return;
        };

        let camera_bg = self
            .viewport_camera_bind_group(frame.camera.viewport_index)
            .clone();
        // A rebuilt per-item bind group means the recorded one is stale even
        // though the key (which hashes item facts, not resource identity)
        // still matches.
        let reusable = self.last_stats.per_object_bind_groups_built == 0
            && self
                .per_object_bundle
                .as_ref()
                .is_some_and(|pb| pb.key == key && pb.camera_bg == camera_bg);
        if !reusable {
            self.per_object_bundle =
                Some(self.record_per_object_bundle(device, key, camera_bg, transparent));
        }
        self.last_stats.per_object_bundle_cached = true;
    }

    /// Record the opaque per-object draws into a render bundle. Mirrors the
    /// plain-solid-mesh subset of the paint path's per-object loop; the
    /// eligibility checks in `update_per_object_bundle` guarantee no item
    /// needs the wireframe/attribute/filter/deform variants.
    fn record_per_object_bundle(
        &self,
        device: &wgpu::Device,
        key: u64,
        camera_bg: wgpu::BindGroup,
        transparent: Vec<usize>,
    ) -> crate::renderer::per_object_state::PerObjectBundle {
        let resources = &self.resources;
        let mut enc = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label: Some("per_object_bundle"),
            color_formats: &[Some(resources.target_format)],
            depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                format: crate::resources::SCENE_DEPTH_FORMAT,
                depth_read_only: false,
                // The scene passes attach the stencil aspect with
                // `stencil_ops: None` (read-only); the bundle must declare the
                // same or replay fails validation. The mesh pipelines never
                // touch stencil.
                stencil_read_only: true,
            }),
            sample_count: resources.sample_count,
            multiview: None,
        });
        enc.set_bind_group(0, &camera_bg, &[]);
        enc.set_bind_group(2, &resources.deform.dummy_bind_group, &[]);

        let mut cur_two_sided: Option<bool> = None;
        let mut cur_mesh = None;
        for (i, item) in self.prepared_surfaces.iter().enumerate() {
            if item.settings.hidden || item.settings.opacity < 1.0 || item.material.is_blend() {
                continue;
            }
            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                continue;
            };
            let two_sided = item.material.is_two_sided();
            if cur_two_sided != Some(two_sided) {
                enc.set_pipeline(if two_sided {
                    &resources.solid_two_sided_pipeline
                } else {
                    &resources.solid_pipeline
                });
                cur_two_sided = Some(two_sided);
            }
            enc.set_bind_group(
                1,
                self.mesh_uniforms
                    .bind_groups
                    .get(i)
                    .and_then(|opt| opt.as_ref())
                    .unwrap_or(&mesh.object_bind_group),
                &[],
            );
            if cur_mesh != Some(item.mesh_id) {
                enc.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                enc.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                cur_mesh = Some(item.mesh_id);
            }
            enc.draw_indexed(0..mesh.index_count, 0, 0..1);
        }

        let bundle = enc.finish(&wgpu::RenderBundleDescriptor {
            label: Some("per_object_bundle"),
        });
        crate::renderer::per_object_state::PerObjectBundle {
            bundle,
            key,
            camera_bg,
            transparent,
        }
    }
}
