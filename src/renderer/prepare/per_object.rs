//! Non-instanced (per-object) mesh draw preparation.

use super::*;

impl ViewportRenderer {
    /// Non-instanced (per-object) mesh draw preparation: compute the per-item
    /// material/feature flags, write one `ObjectUniform` per scene item, and build
    /// or reuse the per-item bind groups (plus the wireframe-mode uniform pool).
    pub(super) fn prepare_per_object(
        resources: &mut ViewportGpuResources,
        mesh_uniforms: &mut PerObjectState,
        use_instancing: bool,
        scene_items: &[SceneRenderItem],
        compute_filter_results: &[crate::resources::ComputeFilterResult],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) {
        let has_scalar_items = scene_items.iter().any(|i| i.active_attribute.is_some());
        let has_two_sided_items = scene_items.iter().any(|i| i.material.is_two_sided());
        let has_matcap_items = scene_items.iter().any(|i| i.material.matcap_id().is_some());
        let has_param_vis_items = scene_items.iter().any(|i| i.material.param_vis.is_some());
        let has_wireframe_items = scene_items.iter().any(|i| i.settings.wireframe);
        let has_normal_vis_items = scene_items.iter().any(|i| i.show_normals);
        // Items whose mesh has a position or normal override bound. The shader
        // flag that drives the override binding lives in the per-item
        // ObjectUniform, so we must enter the write loop for those items even
        // if no other "special" flag is set anywhere in the scene.
        let has_override_items = scene_items.iter().any(|i| {
            resources.mesh_store.get(i.mesh_id).map_or(false, |m| {
                m.position_override_buffer.is_some() || m.normal_override_buffer.is_some()
            })
        });
        // Skinned items are excluded from the batched-instanced path (the
        // instanced shader does not consume the skin palette), so they need
        // per-item object uniform writes too.
        let has_skinned_items = scene_items.iter().any(|i| {
            resources
                .deform
                .has_per_instance_deform_data(i.mesh_id, i.deform_instance)
        });
        // Collect per-item uniforms when wireframe mode is on so we can give each
        // visible item its own bind group (the mesh's shared object_uniform_buf gets
        // overwritten when multiple items reference the same MeshId).
        let mut wireframe_uniforms: Vec<ObjectUniform> = Vec::new();
        let collect_wf_uniforms = frame.viewport.wireframe_mode;
        if !use_instancing
            || frame.viewport.wireframe_mode
            || has_scalar_items
            || has_two_sided_items
            || has_matcap_items
            || has_param_vis_items
            || has_wireframe_items
            || has_normal_vis_items
            || has_override_items
            || has_skinned_items
        {
            // Make sure the per-item object pool can index every scene item.
            // Pool entries for items that go through the instanced path stay None.
            if mesh_uniforms.bind_groups.len() < scene_items.len() {
                mesh_uniforms.bind_groups
                    .resize_with(scene_items.len(), || None);
            }
            if mesh_uniforms.cache_keys.len() < scene_items.len() {
                mesh_uniforms.cache_keys.resize(scene_items.len(), 0);
            }
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
                    && is_instanceable(item, resources, compute_filter_results)
                    && !frame.viewport.wireframe_mode
                    && !item.settings.wireframe
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

                // Collect per-item uniform for wireframe per-item bind groups.
                if collect_wf_uniforms && !item.settings.hidden {
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
                    item.material.matcap_id(),
                    item.warp_attribute.as_deref(),
                    item.material.metallic_roughness_texture_id,
                    item.material.emissive_texture_id,
                );

                // Per-item object pool: ensure this slot has its own uniform buffer +
                // bind group keyed on the item's position in scene_items. Multiple
                // scene items can share the same MeshId; the mesh's shared
                // object_uniform_buf above is stomped by whichever item wrote last,
                // so we maintain this parallel pool to guarantee each item draws
                // with its own transform/material.
                {
                    let uniform_size = std::mem::size_of::<ObjectUniform>() as u64;
                    while mesh_uniforms.uniform_bufs.len() <= item_idx {
                        let buf = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("per_item_object_uniform"),
                            size: uniform_size,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        mesh_uniforms.uniform_bufs.push(buf);
                    }
                    queue.write_buffer(
                        &mesh_uniforms.uniform_bufs[item_idx],
                        0,
                        bytemuck::cast_slice(&[obj_uniform]),
                    );

                    let built = resources.build_per_item_object_bind_group(
                        device,
                        item.mesh_id,
                        &mesh_uniforms.uniform_bufs[item_idx],
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
                    if let Some((bg, key)) = built {
                        let need_rebuild = mesh_uniforms.bind_groups[item_idx].is_none()
                            || mesh_uniforms.cache_keys[item_idx] != key;
                        if need_rebuild {
                            mesh_uniforms.bind_groups[item_idx] = Some(bg);
                            mesh_uniforms.cache_keys[item_idx] = key;
                        }
                    }
                }
            }
        }

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
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: resources.fallback_position_override_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: resources.fallback_normal_override_buf.as_entire_binding(),
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
    }
}
