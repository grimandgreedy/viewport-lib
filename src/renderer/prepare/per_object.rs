//! Non-instanced (per-object) mesh draw preparation.

use super::*;
use crate::renderer::per_object_state::PerObjectState;

/// Ensure the shared object-data storage buffer holds at least `needed`
/// `ObjectUniform` elements. Grows by reallocating to the next power of two
/// (never shrinks), bumps `object_data_gen`, and clears the material bind-group
/// map (whose binding-0 reference points at the old buffer) when it does.
fn ensure_object_data_buffer(
    state: &mut PerObjectState,
    device: &crate::gpu::Device,
    needed: usize,
) {
    let needed = needed.max(1);
    if state.object_data_buf.is_some() && state.object_data_capacity >= needed {
        return;
    }
    let new_cap = needed.next_power_of_two().max(256);
    let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("object_data_buf"),
        size: (new_cap * std::mem::size_of::<ObjectUniform>()) as u64,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    state.object_data_buf = Some(buf);
    state.object_data_capacity = new_cap;
    state.object_data_gen += 1;
    // The previously built bind groups reference the old buffer at binding 0.
    state.material_bind_groups.clear();
}

/// Assemble the per-item `ObjectUniform` for a scene item: the model
/// transform, material and feature flags the mesh shaders read at group 1
/// binding 0. Shared by the per-object scene prepare and the foreground
/// item prepare.
pub(super) fn build_object_uniform(
    resources: &DeviceResources,
    item: &SceneRenderItem,
    wireframe_mode: bool,
    light_probe_index: Option<u32>,
) -> ObjectUniform {
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
    let cm = common_material(item);
    ObjectUniform {
        model: cm.model,
        colour: cm.colour,
        selected: cm.selected,
        wireframe: if wireframe_mode || item.settings.wireframe {
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
            crate::scene::material::BackfacePolicy::Pattern(cfg) => 4 + cfg.pattern as u32,
        },
        backface_colour: match m.backface_policy {
            crate::scene::material::BackfacePolicy::DifferentColour(c) => [c[0], c[1], c[2], 1.0],
            crate::scene::material::BackfacePolicy::Tint(factor) => [factor, 0.0, 0.0, 1.0],
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
        normal_strength: cm.normal_strength,
        ao_range: cm.ao_range,
        metallic_range: m.metallic_range,
        roughness_range: m.roughness_range,
        position_override_base: {
            let mesh = resources.mesh_store.get(item.mesh_id);
            mesh.and_then(|m| m.position_override_slice)
                .map_or(0, |s| s.base_element)
        },
        position_override_len: {
            let mesh = resources.mesh_store.get(item.mesh_id);
            mesh.and_then(|m| m.position_override_slice)
                .map_or(u32::MAX, |s| s.element_count)
        },
        normal_override_base: {
            let mesh = resources.mesh_store.get(item.mesh_id);
            mesh.and_then(|m| m.normal_override_slice)
                .map_or(0, |s| s.base_element)
        },
        normal_override_len: {
            let mesh = resources.mesh_store.get(item.mesh_id);
            mesh.and_then(|m| m.normal_override_slice)
                .map_or(u32::MAX, |s| s.element_count)
        },
        has_light_probe: light_probe_index.map_or(0, |_| 1),
        light_probe_index: light_probe_index.unwrap_or(0),
        lightmap_mode: resources
            .mesh_store
            .get(item.mesh_id)
            .and_then(|mesh| mesh.lightmap.as_ref())
            .map_or(0, |lm| lm.mode),
        lightmap_directional: resources
            .mesh_store
            .get(item.mesh_id)
            .and_then(|mesh| mesh.lightmap.as_ref())
            .map_or(0, |lm| {
                (lm.direction_texture_id.is_some() && !lm.is_shadowmask) as u32
            }),
        // Scene-atlas placement: a per-mesh lightmap owns its whole atlas
        // (identity scale/bias, layer 0); a scene lightmap sits in a sub-rect of a
        // shared page. Both read straight off the registration.
        lightmap_scale_bias: resources
            .mesh_store
            .get(item.mesh_id)
            .and_then(|mesh| mesh.lightmap.as_ref())
            .map_or([1.0, 1.0, 0.0, 0.0], |lm| lm.scale_bias),
        lightmap_index: resources
            .mesh_store
            .get(item.mesh_id)
            .and_then(|mesh| mesh.lightmap.as_ref())
            .map_or(0, |lm| lm.layer),
        has_shadowmask: resources
            .mesh_store
            .get(item.mesh_id)
            .and_then(|mesh| mesh.lightmap.as_ref())
            .map_or(0, |lm| lm.is_shadowmask as u32),
        _pad_ls: [0; 2],
    }
}

/// Light-probe SH prepass, shared by the per-object and instanced draw paths.
///
/// For each item that opts into a light probe (`indirect_light == LightProbe`),
/// blend the nearby probes' SH at the object position and pack it into the
/// shared SH buffer (group 0 binding 18). The returned vector maps this frame's
/// scene-item index to that object's SH block index; both paths read it so an
/// item's `light_probe_index` is identical whichever path draws it. Returns all
/// `None` (and writes nothing) when no probes are uploaded, so it costs nothing
/// for consumers that never call `set_light_probes`.
pub(super) fn prepare_light_probe_sh(
    resources: &DeviceResources,
    scene_items: &[SceneRenderItem],
    queue: &crate::gpu::Queue,
) -> Vec<Option<u32>> {
    use crate::renderer::IndirectLightSource;
    let mut indices = vec![None; scene_items.len()];
    let probes = resources.lighting.probes.as_ref().filter(|p| !p.is_empty());
    let mut sh_gpu: Vec<[f32; 4]> = Vec::new();
    let mut count = 0u32;
    for (idx, item) in scene_items.iter().enumerate() {
        match item.indirect_light {
            IndirectLightSource::LightProbe => {
                // Blend the nearby point probes' SH at the object position and
                // pack a block; the block index rides light_probe_index.
                if let Some(probes) = probes {
                    if (count as usize) < crate::resources::light_probes::MAX_LIGHT_PROBE_OBJECTS {
                        let center = [item.model[3][0], item.model[3][1], item.model[3][2]];
                        sh_gpu.extend_from_slice(&probes.blend_sh_at(center).to_gpu());
                        indices[idx] = Some(count);
                        count += 1;
                    }
                }
            }
            IndirectLightSource::ProbeVolume => {
                // The volume is sampled per fragment in the shader, so no CPU
                // blend and no SH block: the sentinel index selects it. Works
                // whether or not a point-probe set is also uploaded.
                indices[idx] = Some(crate::resources::light_probes::PROBE_VOLUME_INDEX);
            }
            IndirectLightSource::GlobalIbl => {}
        }
    }
    if !sh_gpu.is_empty() {
        queue.write_buffer(
            &resources.lighting.indirect_buf,
            0,
            bytemuck::cast_slice(&sh_gpu),
        );
    }
    indices
}

impl ViewportRenderer {
    /// Non-instanced (per-object) mesh draw preparation: compute the per-item
    /// material/feature flags, write one `ObjectUniform` per scene item, and build
    /// or reuse the per-item bind groups (plus the wireframe-mode uniform pool).
    ///
    /// `probe_indices` is index-aligned with `scene_items` and maps each item to
    /// its light-probe SH block (see [`prepare_light_probe_sh`]).
    pub(super) fn prepare_per_object(
        resources: &mut DeviceResources,
        mesh_uniforms: &mut PerObjectState,
        use_instancing: bool,
        scene_items: &[SceneRenderItem],
        instanceable: &[bool],
        probe_indices: &[Option<u32>],
        frame_index: u64,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> u32 {
        // Count of per-item bind groups actually (re)built this frame, reported
        // in FrameStats so a cache that is silently missing is visible.
        let mut bind_groups_built = 0u32;

        // Purge the deduped material bind-group map when a texture or mesh was
        // freed, so no bind group keeps freed GPU memory alive.
        if mesh_uniforms.free_epoch != resources.resource_free_epoch {
            mesh_uniforms.free_epoch = resources.resource_free_epoch;
            mesh_uniforms.material_bind_groups.clear();
        }

        // Reset this frame's per-item slot maps. Slots for items on the
        // instanced path stay None; a None slot at draw time falls back to the
        // mesh's single-element bind group drawn at instance 0.
        mesh_uniforms.bind_groups.clear();
        mesh_uniforms
            .bind_groups
            .resize_with(scene_items.len(), || None);
        mesh_uniforms.object_indices.clear();
        mesh_uniforms.object_indices.resize(scene_items.len(), 0);
        mesh_uniforms.submesh_bind_groups.clear();
        mesh_uniforms.submesh_indices.clear();

        // Collect per-item uniforms when wireframe mode is on so we can give each
        // visible item its own wireframe bind group.
        let mut wireframe_uniforms: Vec<ObjectUniform> = Vec::new();
        let collect_wf_uniforms = frame.viewport.wireframe_mode;
        // Enter the per-object path when at least one item needs it. An item is
        // non-instanceable for any of the reasons is_instanceable tests, so
        // `instanceable` already folds those in; wireframe and normal
        // visualization force the per-object path directly.
        let any_non_instanceable = instanceable.iter().any(|&b| !b);
        let any_wireframe_or_normals = scene_items
            .iter()
            .any(|i| i.settings.wireframe || i.show_normals);
        if !use_instancing
            || frame.viewport.wireframe_mode
            || any_non_instanceable
            || any_wireframe_or_normals
        {
            // Pass 1 gathers every per-object draw's ObjectUniform into one
            // contiguous Vec and records its element index; the group-1 bind
            // groups are built in pass 2, after the shared storage buffer is
            // sized, so a build never references a buffer a later grow
            // reallocated. `pending` lists the draws (whole-mesh entry, then each
            // submesh range) to bind in pass 2.
            let mut object_data: Vec<ObjectUniform> = Vec::with_capacity(scene_items.len());
            let mut pending: Vec<(usize, u32)> = Vec::new();
            let mut range_mats_by_item: std::collections::HashMap<
                usize,
                Vec<crate::scene::material::Material>,
            > = std::collections::HashMap::new();
            // Meshes whose single-element `object_uniform_buf` fallback has been
            // written this frame, so a mesh shared by many per-object items pays
            // one write instead of one per item (the write is only a shadow /
            // fallback buffer; the items themselves draw from the shared
            // object-data buffer).
            let mut shared_written: std::collections::HashSet<
                crate::resources::mesh::mesh_store::MeshId,
            > = std::collections::HashSet::new();
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
                let obj_uniform = build_object_uniform(
                    resources,
                    item,
                    frame.viewport.wireframe_mode,
                    probe_indices[item_idx],
                );

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
                        normal_strength: 1.0,
                        ao_range: [0.0, 1.0],
                        metallic_range: [0.0, 1.0],
                        roughness_range: [0.0, 1.0],
                        position_override_base: 0,
                        position_override_len: u32::MAX,
                        normal_override_base: 0,
                        normal_override_len: u32::MAX,
                        has_light_probe: 0,
                        light_probe_index: 0,
                        lightmap_mode: 0,
                        lightmap_directional: 0,
                        lightmap_scale_bias: [1.0, 1.0, 0.0, 0.0],
                        lightmap_index: 0,
                        has_shadowmask: 0,
                        _pad_ls: [0; 2],
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

                // Keep the mesh's single-element object buffer in sync: it backs
                // the mesh's fallback bind group and the shadow (directional and
                // point) caster passes, which draw at instance 0. Written once
                // per mesh per frame; when several items share a MeshId it holds
                // the first item's transform (the multi-item-per-mesh shadow
                // caster case was already a shared-buffer approximation).
                if shared_written.insert(item.mesh_id) {
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                        queue.write_buffer(
                            &mesh.object_uniform_buf,
                            0,
                            bytemuck::cast_slice(&[obj_uniform]),
                        );
                        resources.frame_upload_bytes += std::mem::size_of::<ObjectUniform>() as u64;
                    }
                }

                // Whole-mesh draw: record this item's element in the shared
                // object-data array; the group-1 bind group is built in pass 2.
                let idx = object_data.len() as u32;
                object_data.push(obj_uniform);
                mesh_uniforms.object_indices[item_idx] = idx;
                pending.push((item_idx, 0));

                // Items drawn with per-submesh materials get one extra array
                // element and one pending bind per range.
                let range_mats: Option<Vec<crate::scene::material::Material>> = resources
                    .mesh_store
                    .get(item.mesh_id)
                    .and_then(|m| super::mesh_material::active_submesh_materials(item, m))
                    .map(|mats| mats.to_vec());
                if let Some(mats) = range_mats {
                    let mut range_item = item.clone();
                    range_item.submesh_materials = None;
                    let mut range_indices: Vec<u32> = Vec::with_capacity(mats.len());
                    for (r, mat) in mats.iter().enumerate() {
                        range_item.material = mat.clone();
                        let range_uniform = build_object_uniform(
                            resources,
                            &range_item,
                            frame.viewport.wireframe_mode,
                            probe_indices[item_idx],
                        );
                        let ridx = object_data.len() as u32;
                        object_data.push(range_uniform);
                        range_indices.push(ridx);
                        pending.push((item_idx, r as u32 + 1));
                    }
                    mesh_uniforms
                        .submesh_bind_groups
                        .insert(item_idx, vec![None; mats.len()]);
                    mesh_uniforms
                        .submesh_indices
                        .insert(item_idx, range_indices);
                    range_mats_by_item.insert(item_idx, mats);
                }
            }

            // Size and fill the shared object-data buffer, then build pass-2
            // bind groups against it (deduped by mesh+material fingerprint).
            ensure_object_data_buffer(mesh_uniforms, device, object_data.len());
            if !object_data.is_empty() {
                if let Some(buf) = mesh_uniforms.object_data_buf.as_ref() {
                    queue.write_buffer(buf, 0, bytemuck::cast_slice(&object_data));
                    resources.frame_upload_bytes +=
                        (object_data.len() * std::mem::size_of::<ObjectUniform>()) as u64;
                }
            }
            let data_gen = mesh_uniforms.object_data_gen;
            let buf = mesh_uniforms.object_data_buf.clone();
            if let Some(buf) = buf {
                for &(item_idx, submesh) in &pending {
                    let item = &scene_items[item_idx];
                    let mat: &crate::scene::material::Material = if submesh == 0 {
                        &item.material
                    } else {
                        &range_mats_by_item[&item_idx][(submesh - 1) as usize]
                    };
                    let Some(key) = resources.per_item_object_bg_key(
                        item.mesh_id,
                        mat.texture_id,
                        mat.normal_map_id,
                        mat.ao_map_id,
                        item.colourmap_id,
                        item.active_attribute.as_ref().map(|a| a.name.as_str()),
                        mat.matcap_id(),
                        item.warp_attribute.as_deref(),
                        mat.metallic_roughness_texture_id,
                        mat.emissive_texture_id,
                    ) else {
                        continue;
                    };
                    let fresh = mesh_uniforms
                        .material_bind_groups
                        .get(&key)
                        .is_some_and(|m| m.data_gen == data_gen);
                    if !fresh {
                        if let Some((bg, _)) = resources.build_per_item_object_bind_group(
                            device,
                            item.mesh_id,
                            &buf,
                            mat.texture_id,
                            mat.normal_map_id,
                            mat.ao_map_id,
                            item.colourmap_id,
                            item.active_attribute.as_ref().map(|a| a.name.as_str()),
                            mat.matcap_id(),
                            item.warp_attribute.as_deref(),
                            mat.metallic_roughness_texture_id,
                            mat.emissive_texture_id,
                            None,
                        ) {
                            bind_groups_built += 1;
                            mesh_uniforms.material_bind_groups.insert(
                                key,
                                crate::renderer::per_object_state::MaterialBindGroup {
                                    bind_group: bg,
                                    data_gen,
                                    last_frame: frame_index,
                                },
                            );
                        }
                    }
                    let bg = mesh_uniforms.material_bind_groups.get_mut(&key).map(|m| {
                        m.last_frame = frame_index;
                        m.bind_group.clone()
                    });
                    if let Some(bg) = bg {
                        if submesh == 0 {
                            mesh_uniforms.bind_groups[item_idx] = Some(bg);
                        } else if let Some(slot) =
                            mesh_uniforms.submesh_bind_groups.get_mut(&item_idx)
                        {
                            slot[(submesh - 1) as usize] = Some(bg);
                        }
                    }
                }
            }
        }

        // Capacity-based eviction of the deduped material bind-group map. A
        // material whose items left the frame keeps its bind group until the map
        // exceeds its budget, so re-showing a large set does not rebuild every
        // bind group in one frame (~11 us per build measured). Entries used this
        // frame are never evicted: the budget is at least twice the live set, so
        // eviction only removes stale entries, oldest first. (A resource free
        // already cleared the whole map at the top of this function.)
        const CACHE_MIN_CAPACITY: usize = 8192;
        if mesh_uniforms.material_bind_groups.len() > CACHE_MIN_CAPACITY {
            let live = mesh_uniforms
                .material_bind_groups
                .values()
                .filter(|e| e.last_frame == frame_index)
                .count();
            let cap = CACHE_MIN_CAPACITY.max(live.saturating_mul(2));
            let len = mesh_uniforms.material_bind_groups.len();
            if len > cap {
                let excess = len - cap;
                let mut stale_frames: Vec<u64> = mesh_uniforms
                    .material_bind_groups
                    .values()
                    .map(|e| e.last_frame)
                    .filter(|f| *f != frame_index)
                    .collect();
                stale_frames.sort_unstable();
                let threshold = stale_frames[excess.min(stale_frames.len()) - 1];
                mesh_uniforms
                    .material_bind_groups
                    .retain(|_, e| e.last_frame == frame_index || e.last_frame > threshold);
            }
        }

        // Build per-item wireframe bind groups so each visible item gets its own
        // object uniform, avoiding the shared-MeshId overwrite problem.
        if !wireframe_uniforms.is_empty() {
            let n = wireframe_uniforms.len();
            let uniform_size = std::mem::size_of::<ObjectUniform>() as u64;

            // Grow the buffer/bind-group pools if needed. We never shrink them.
            while mesh_uniforms.wireframe_uniform_bufs.len() < n {
                let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("wireframe_item_uniform"),
                    size: uniform_size,
                    usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("wireframe_item_bg"),
                    layout: &resources.binds.object_bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::TextureView(
                                &resources.material.texture.view,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 2,
                            resource: crate::gpu::BindingResource::Sampler(
                                &resources.material.sampler,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 3,
                            resource: crate::gpu::BindingResource::TextureView(
                                &resources.material.normal_map_view,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 4,
                            resource: crate::gpu::BindingResource::TextureView(
                                &resources.material.ao_map_view,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 5,
                            resource: crate::gpu::BindingResource::TextureView(
                                &resources.content.fallback_lut_view,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 6,
                            resource: resources.content.fallback_scalar_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 7,
                            resource: crate::gpu::BindingResource::TextureView(
                                resources
                                    .content
                                    .fallback_matcap_view
                                    .as_ref()
                                    .unwrap_or(&resources.material.texture.view),
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 8,
                            resource: resources
                                .content
                                .fallback_face_colour_buf
                                .as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 9,
                            resource: resources.content.fallback_warp_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 10,
                            resource: crate::gpu::BindingResource::Sampler(
                                &resources.material.lut_sampler,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 11,
                            resource: crate::gpu::BindingResource::TextureView(
                                &resources.material.metallic_roughness_view,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 12,
                            resource: crate::gpu::BindingResource::TextureView(
                                &resources.material.emissive_view,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 13,
                            resource: resources
                                .content
                                .fallback_position_override_buf
                                .as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 14,
                            resource: resources
                                .content
                                .fallback_normal_override_buf
                                .as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 15,
                            resource: resources
                                .content
                                .fallback_extension_attr_buf
                                .as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 17,
                            resource: crate::gpu::BindingResource::TextureView(
                                &resources.material.texture_array_view,
                            ),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 18,
                            resource: crate::gpu::BindingResource::TextureView(
                                &resources.material.texture_array_view,
                            ),
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

    /// Write the per-item uniforms and build the group-1 bind groups for one
    /// viewport's foreground items.
    ///
    /// Foreground items live outside the shared [`PerObjectState`] cache: the
    /// list is small and per-viewport, and running the `(pick_id, occurrence)`
    /// keyed cache over a second list would collide with the scene items'
    /// entries. `entries` is index-aligned with `items`; an entry whose mesh
    /// is missing keeps `bind_group: None` and the draw path skips it.
    pub(super) fn prepare_foreground_objects(
        resources: &mut DeviceResources,
        entries: &mut Vec<crate::renderer::per_object_state::ForegroundObjectEntry>,
        items: &[SceneRenderItem],
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        entries.truncate(items.len());
        let uniform_size = std::mem::size_of::<ObjectUniform>() as u64;
        for (idx, item) in items.iter().enumerate() {
            if entries.len() <= idx {
                let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("foreground_object_uniform"),
                    size: uniform_size,
                    usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                entries.push(crate::renderer::per_object_state::ForegroundObjectEntry {
                    uniform_buf: buf,
                    bind_group: None,
                    cache_key: 0,
                    last_uniform: None,
                });
            }
            if resources.mesh_store.get(item.mesh_id).is_none() {
                tracing::warn!(
                    mesh_index = item.mesh_id.index(),
                    "foreground item mesh_index invalid, skipping"
                );
                entries[idx].bind_group = None;
                continue;
            }

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

            let obj_uniform = build_object_uniform(resources, item, false, None);
            let entry = &mut entries[idx];
            let uniform_changed = entry.last_uniform.as_ref().map_or(true, |u| {
                bytemuck::bytes_of(u) != bytemuck::bytes_of(&obj_uniform)
            });
            if uniform_changed {
                queue.write_buffer(&entry.uniform_buf, 0, bytemuck::cast_slice(&[obj_uniform]));
                entry.last_uniform = Some(obj_uniform);
                resources.frame_upload_bytes += uniform_size;
            }

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
            if let Some((bg, key)) = built {
                entry.bind_group = Some(bg);
                entry.cache_key = key;
            }
        }
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
    pub(super) fn update_per_object_bundle(
        &mut self,
        device: &crate::gpu::Device,
        frame: &FrameData,
    ) {
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

        // Post-processing selects the HDR path; the bundle records against
        // that pass's formats and pipelines instead of being disabled. The
        // HDR pipelines build on the first HDR frame, so frame one records
        // nothing and the bundle starts on frame two.
        let hdr = frame.effects.post_process.enabled;
        // Clip geometry needs the discarding pipeline (its clip test is a
        // discard); with it active the whole bundle keeps discards.
        let clipping_active = frame
            .effects
            .clip_objects
            .iter()
            .any(|o| o.enabled && o.clip_geometry);
        let plan = 'plan: {
            if frame.viewport.wireframe_mode
                || !self.compute_filter_results.is_empty()
                || !self.instancing.use_instancing
                || !self.instancing.batches.is_empty()
                || !self.resources.deform.meshes.is_empty()
                || self.prepared_surfaces.len() < MIN_BUNDLE_ITEMS
            {
                break 'plan None;
            }
            if hdr
                && (self.resources.scene.hdr_solid.is_none()
                    || self.resources.scene.hdr_solid_two_sided.is_none())
            {
                break 'plan None;
            }
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            hdr.hash(&mut h);
            frame.camera.viewport_index.hash(&mut h);
            self.prepared_surfaces.len().hash(&mut h);
            let mut transparent: Vec<usize> = Vec::new();
            let mut opaque = 0usize;
            // An alpha-mask item's fragment shader discards below the cutoff, so
            // one in the opaque set forces the whole bundle to keep discards.
            let mut any_mask = false;
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
                    // Material-plugin items need a per-item pipeline and a
                    // group-3 bind the bundle recorder does not capture.
                    || item.material.shading_plugin.is_some()
                    // Per-submesh materials need one draw and one bind group
                    // per index range; the bundle records a single
                    // full-range draw per mesh.
                    || item.submesh_materials.is_some()
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
                if !matches!(
                    item.material.alpha_mode,
                    crate::scene::material::AlphaMode::Opaque
                ) {
                    any_mask = true;
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
            // Early-Z: a plain-opaque HDR set with no clip geometry and no
            // alpha mask can never hit a discard, so record it with the
            // discard-free pipeline twins and let hidden fragments be depth
            // rejected before shading. The choice is hashed so flipping it
            // (clip toggled, the force-discard measurement knob) re-records.
            let no_discard = hdr
                && !clipping_active
                && !any_mask
                && !self.resources.force_po_discard
                && self.resources.scene.hdr_solid_nodiscard.is_some()
                && self.resources.scene.hdr_solid_two_sided_nodiscard.is_some();
            no_discard.hash(&mut h);
            Some((h.finish(), transparent, no_discard))
        };

        let Some((key, transparent, no_discard)) = plan else {
            self.per_object_bundle = None;
            return;
        };

        // Churn gate. A single isolated change re-records immediately (the
        // measured cost of one re-record is a wash against immediate draws),
        // but a set that changes twice in short succession backs the bundle
        // off entirely: record + replay encodes every draw twice per frame,
        // and dropping a just-recorded bundle every frame leaks it in
        // wgpu 27 (gfx-rs/wgpu#8656). The gate re-arms once the set has
        // been stable for a stretch.
        const SUPPRESS_WINDOW: u32 = 8;
        const REARM_STABLE: u32 = 30;
        // A rebuilt per-item bind group means the recorded one is stale even
        // though the key (which hashes item facts, not resource identity)
        // still matches, so it counts as a change too.
        let dirty = self.last_stats.per_object_bind_groups_built > 0;
        let changed = dirty || self.per_object_bundle_gate.last_key != Some(key);
        self.per_object_bundle_gate.last_key = Some(key);
        if changed {
            if self.per_object_bundle_gate.frames_since_change < SUPPRESS_WINDOW {
                self.per_object_bundle_gate.suppressed = true;
            }
            self.per_object_bundle_gate.frames_since_change = 0;
        } else {
            self.per_object_bundle_gate.frames_since_change = self
                .per_object_bundle_gate
                .frames_since_change
                .saturating_add(1);
            if self.per_object_bundle_gate.suppressed
                && self.per_object_bundle_gate.frames_since_change >= REARM_STABLE
            {
                self.per_object_bundle_gate.suppressed = false;
            }
        }
        // Diagnostic kill-switch for the churn gate: with it set, sustained
        // churn re-records the bundle every frame instead of backing off.
        // Exists for leak retests on wgpu versions that ship the
        // render-bundle drop fix (gfx-rs/wgpu#8661); leave it unset
        // otherwise, since per-frame re-record costs more than it saves.
        static GATE_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let gate_disabled = *GATE_DISABLED
            .get_or_init(|| std::env::var_os("VIEWPORT_DISABLE_BUNDLE_CHURN_GATE").is_some());
        if self.per_object_bundle_gate.suppressed && !gate_disabled {
            self.per_object_bundle = None;
            return;
        }

        let camera_bg = self
            .viewport_camera_bind_group(frame.camera.viewport_index)
            .clone();
        let reusable = !dirty
            && self
                .per_object_bundle
                .as_ref()
                .is_some_and(|pb| pb.key == key && pb.camera_bg == camera_bg);
        if !reusable {
            self.per_object_bundle = Some(self.record_per_object_bundle(
                device,
                key,
                camera_bg,
                transparent,
                hdr,
                no_discard,
            ));
        }
        self.last_stats.per_object_bundle_cached = true;
    }

    /// Record the opaque per-object draws into a render bundle. Mirrors the
    /// plain-solid-mesh subset of the paint path's per-object loop; the
    /// eligibility checks in `update_per_object_bundle` guarantee no item
    /// needs the wireframe/attribute/filter/deform variants.
    fn record_per_object_bundle(
        &self,
        device: &crate::gpu::Device,
        key: u64,
        camera_bg: crate::gpu::BindGroup,
        transparent: Vec<usize>,
        hdr: bool,
        no_discard: bool,
    ) -> crate::renderer::per_object_state::PerObjectBundle {
        let resources = &self.resources;
        // The HDR scene pass renders Rgba16Float at sample count 1 with a
        // writable stencil aspect; the LDR passes render the surface format
        // at the configured sample count with the stencil attached read-only
        // (`stencil_ops: None`). A read-only stencil declaration is valid in
        // both (the mesh pipelines never touch stencil).
        let (colour_format, sample_count) = if hdr {
            (crate::gpu::TextureFormat::Rgba16Float, 1)
        } else {
            (resources.target_format, resources.sample_count)
        };
        let (solid, solid_two_sided) = if hdr && no_discard {
            (
                self.resources.scene.hdr_solid_nodiscard.as_ref().unwrap(),
                self.resources
                    .scene
                    .hdr_solid_two_sided_nodiscard
                    .as_ref()
                    .unwrap(),
            )
        } else if hdr {
            (
                self.resources.scene.hdr_solid.as_ref().unwrap(),
                self.resources.scene.hdr_solid_two_sided.as_ref().unwrap(),
            )
        } else {
            (&resources.scene.solid, &resources.scene.solid_two_sided)
        };
        let mut enc = crate::resources::builders::render_bundle_encoder(
            device,
            "per_object_bundle",
            &[Some(colour_format)],
            Some(crate::gpu::RenderBundleDepthStencil {
                format: crate::resources::SCENE_DEPTH_FORMAT,
                depth_read_only: false,
                stencil_read_only: true,
            }),
            sample_count,
        );
        enc.set_bind_group(0, &camera_bg, &[]);
        bind_deform_group!(enc, resources, &resources.deform.dummy_bind_group);

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
                enc.set_pipeline(if two_sided { solid_two_sided } else { solid });
                cur_two_sided = Some(two_sided);
            }
            // A per-item slot means this item draws with the shared material
            // bind group and selects its data at object_indices[i]; a None slot
            // falls back to the mesh's single-element bind group at instance 0.
            let (obj_bg, inst) = match self
                .mesh_uniforms
                .bind_groups
                .get(i)
                .and_then(|opt| opt.as_ref())
            {
                Some(bg) => (bg, self.mesh_uniforms.object_indices[i]),
                None => (&mesh.object_bind_group, 0),
            };
            enc.set_bind_group(1, obj_bg, &[]);
            if cur_mesh != Some(item.mesh_id) {
                enc.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                enc.set_index_buffer(
                    resources.geometry.index_slice(mesh.index_span),
                    crate::gpu::IndexFormat::Uint32,
                );
                cur_mesh = Some(item.mesh_id);
            }
            enc.draw_indexed(0..mesh.index_count, 0, inst..inst + 1);
        }

        let bundle = enc.finish(&crate::gpu::RenderBundleDescriptor {
            label: Some("per_object_bundle"),
        });
        crate::renderer::per_object_state::PerObjectBundle {
            bundle,
            key,
            hdr,
            camera_bg,
            transparent,
        }
    }
}
