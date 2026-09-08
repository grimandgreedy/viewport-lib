//! Instanced (GPU-driven) mesh draw preparation.

use super::*;
use crate::resources::mesh::instanced_bindless::MaterialTextureBinding;
use viewport_lib_types::ids::TextureId;

/// The batch grouping key for one instanced item: items with an equal key share
/// a batch (same pipeline state and, under `PerBatch`, the same texture bind
/// group). `mesh_id` and `two_sided` are always part of the key; the five
/// material texture ids are only included under `PerBatch` binding. Under
/// `Bindless` the shader indexes a texture array by a per-material index, so the
/// texture ids collapse to `None` and instances of one mesh with different
/// materials batch together.
///
/// One exception under `Bindless`: an alpha-masked material keeps its albedo id
/// in the key. The shadow-cutout pass stays on the per-batch binding (it does not
/// bind the material buffer), so it needs one albedo per batch to alpha-test
/// against. Opaque materials cast shadows through the discard-free pipeline that
/// samples no texture, so they collapse fully.
///
/// Used by both the sort comparator and the batch-split predicate below, so the
/// two cannot drift out of sync.
type BatchGroupKey = (
    usize,
    Option<TextureId>,
    Option<TextureId>,
    Option<TextureId>,
    Option<TextureId>,
    Option<TextureId>,
    bool,
);

/// Form the opaque draw groups for GPU-driven submission and size + upload the
/// buffers the compaction pass and the count-multi-draw need. Returns whether the
/// path is active (bindless + native multi-draw + built cull pipelines + at least
/// one group); when inactive it clears `draw_groups` so the draw loop falls back
/// to CPU run-forming.
///
/// The group key mirrors the draw loop's run key exactly (two-sidedness,
/// discard-free eligibility, and geometry chunk over contiguous opaque batches),
/// so the groups drawn here match the runs the CPU path would have formed. The
/// texture bind group is constant under bindless, so it is not part of the key.
fn build_and_upload_draw_groups(
    resources: &DeviceResources,
    instancing: &mut InstancingState,
    cull_state: &mut crate::resources::ViewportCullState,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    frame: &FrameData,
) -> bool {
    let bindless =
        resources.instancing.material_texture_binding == MaterialTextureBinding::Bindless;
    // `multi_draw_supported` (native MULTI_DRAW_INDIRECT_COUNT), not
    // `multi_draw_active()`: the count variant cannot be emulated, so the
    // `multi_draw_forced` diagnostic override must not reach this path.
    let active =
        bindless && instancing.multi_draw_supported && resources.cull.hdr_solid_pipeline.is_some();
    if !active {
        instancing.draw_groups.clear();
        return false;
    }
    let clipping_active = frame
        .effects
        .clip
        .objects
        .iter()
        .any(|o| o.enabled && o.clip_geometry);
    let nodiscard = resources.cull.hdr_solid_nodiscard_pipeline.is_some()
        && resources
            .cull
            .hdr_solid_two_sided_nodiscard_pipeline
            .is_some();

    let n = instancing.batches.len();
    let mut group_id = vec![crate::renderer::indirect::NO_GROUP; n];
    let mut group_arg_base = vec![0u32; n];
    let mut opaque_groups: Vec<crate::renderer::instancing_state::DrawGroup> = Vec::new();
    let mut oit_groups: Vec<crate::renderer::instancing_state::DrawGroup> = Vec::new();
    // Opaque and transparent batches are grouped in one pass so both draws share a
    // single compaction and one `draw_counts` slot space. The run key includes
    // `is_transparent`, so a group is always a maximal contiguous batch range of
    // one kind: consecutive groups have adjacent arg ranges and never overlap in
    // the compacted buffer, whichever way the two kinds interleave. `next_group`
    // is the global slot each group takes in `draw_counts` (its `count_index`).
    let mut next_group = 0u32;
    // (run key, index within the group vec the key's transparency bit selects).
    let mut cur: Option<((bool, bool, bool, u32, u32), usize)> = None;
    for (b, batch) in instancing.batches.iter().enumerate() {
        let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
            cur = None;
            continue;
        };
        let transparent = batch.is_transparent;
        // OIT has no discard-free variant; only the opaque pass keys on it.
        let no_discard = !transparent && !clipping_active && !batch.has_alpha_mask && nodiscard;
        let key = (
            transparent,
            batch.two_sided,
            no_discard,
            mesh.vertex_span.chunk,
            mesh.index_span.chunk,
        );
        let vec_ref = if transparent {
            &mut oit_groups
        } else {
            &mut opaque_groups
        };
        let gv = match cur {
            Some((k, gv)) if k == key => {
                vec_ref[gv].size += 1;
                gv
            }
            _ => {
                let count_index = next_group;
                next_group += 1;
                vec_ref.push(crate::renderer::instancing_state::DrawGroup {
                    arg_base: b as u32,
                    size: 1,
                    two_sided: batch.two_sided,
                    no_discard,
                    count_index,
                });
                let gv = vec_ref.len() - 1;
                cur = Some((key, gv));
                gv
            }
        };
        group_id[b] = vec_ref[gv].count_index;
        group_arg_base[b] = vec_ref[gv].arg_base;
    }

    if opaque_groups.is_empty() && oit_groups.is_empty() {
        instancing.draw_groups.clear();
        instancing.oit_draw_groups.clear();
        return false;
    }

    // Per-batch group metadata (scene-global): grow to `n` batches.
    if instancing.group_buf_capacity < n {
        let cap = (n * 2).max(64);
        let mk = |label: &str| {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size: (cap * std::mem::size_of::<u32>()) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        instancing.group_id_buf = Some(mk("draw_group_id_buf"));
        instancing.group_arg_base_buf = Some(mk("draw_group_arg_base_buf"));
        instancing.group_buf_capacity = cap;
    }
    queue.write_buffer(
        instancing.group_id_buf.as_ref().unwrap(),
        0,
        bytemuck::cast_slice(&group_id),
    );
    queue.write_buffer(
        instancing.group_arg_base_buf.as_ref().unwrap(),
        0,
        bytemuck::cast_slice(&group_arg_base),
    );

    // Per-viewport compaction outputs: compacted args (one DrawIndexedIndirect =
    // 20 bytes per batch slot) and per-group survivor counts (u32; at most `n`).
    if cull_state.compact_capacity < n {
        let cap = (n * 2).max(64);
        cull_state.compacted_args_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("compacted_draw_args_buf"),
            size: (cap * 20) as u64,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::INDIRECT
                | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        cull_state.draw_counts_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("draw_counts_buf"),
            size: (cap * std::mem::size_of::<u32>()) as u64,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::INDIRECT
                | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        cull_state.compact_capacity = cap;
    }

    instancing.draw_groups = opaque_groups;
    instancing.oit_draw_groups = oit_groups;
    true
}

fn batch_group_key(item: &SceneRenderItem, binding: MaterialTextureBinding) -> BatchGroupKey {
    let m = &item.material;
    let textured = binding == MaterialTextureBinding::PerBatch;
    // Under bindless the texture ids drop out of the key (the array is bound
    // once per frame); keep them only for the per-batch binding path.
    let keep = |id: Option<TextureId>| if textured { id } else { None };
    // Albedo is kept for an alpha-masked material even under bindless, so the
    // per-batch shadow-cutout pass has a single albedo to sample per batch.
    let is_masked = matches!(m.alpha_mode, crate::scene::material::AlphaMode::Mask(_));
    let albedo = if textured || is_masked {
        m.texture_id
    } else {
        None
    };
    (
        item.mesh_id.index(),
        albedo,
        keep(m.normal_map_id),
        keep(m.ao_map_id),
        keep(m.metallic_roughness_texture_id),
        keep(m.emissive_texture_id),
        m.is_two_sided(),
    )
}

impl ViewportRenderer {
    /// Build instanced batches for the current frame: filter eligible items,
    /// pack instance/AABB/batch-meta buffers (partial upload when the structure
    /// is preserved), and dispatch the GPU frustum cull + indirect-args passes.
    /// Returns `(batches_reuploaded, batches_skipped)` for frame stats.
    pub(super) fn prepare_instanced(
        resources: &mut DeviceResources,
        instancing: &mut InstancingState,
        instanceable: &[bool],
        scene_items: &[SceneRenderItem],
        probe_indices: &[Option<u32>],
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> (u32, u32) {
        let mut batches_reuploaded = 0u32;
        let mut batches_skipped = 0u32;
        resources.ensure_instanced_pipelines(device);
        resources.ensure_hdr_instanced_pipelines(device);
        resources.ensure_oit_instanced_pipeline(device);

        // Generation-based cache: skip batch rebuild and GPU upload when nothing changed.
        // wireframe_mode removed from cache key : wireframe rendering
        // uses the per-object wireframe_pipeline, not the instanced path, so
        // instance data is now viewport-agnostic.
        //
        // Items with active_attribute, matcap, an emissive texture, a shading
        // plugin, warp, deform, submesh materials, or overrides are excluded from
        // the instanced batch filter (see `is_instanceable`). Items whose mesh has
        // an active compute filter result are also excluded so the per-object path
        // can apply the filtered index buffer (instanced draws always use the full
        // index buffer).
        // These flags are set on render items AFTER collect_render_items() (per-frame
        // mutations), so they do NOT bump the scene generation. Use last_instancable_count
        // as a cache key instead of a blanket has_per_frame_mutations flag; this allows
        // scenes that mix instanced and non-instanced items (e.g. one two-sided mesh +
        // many static boxes) to still hit the instanced batch cache on frames where the
        // filtered set is unchanged.
        let instancable_count = instanceable.iter().filter(|&&b| b).count();
        let cache_valid = instancable_count == instancing.last_instancable_count
            && frame.scene.generation == instancing.last_scene_generation
            && frame.interaction.selection_generation == instancing.last_selection_generation
            && scene_items.len() == instancing.last_scene_items_count
            // Cached batches reference mesh ids by slot; a free (which bumps this
            // epoch) can leave them pointing at freed meshes, so every instanced
            // draw is skipped. Rebuild when it moves, as the per-object path does.
            && resources.resource_free_epoch == instancing.last_resource_free_epoch
            // The global wireframe toggle is baked into each instance's
            // per-instance wireframe flag (see the `InstanceData` push below), so
            // flipping it with no other scene change must still force a rebuild.
            && frame.viewport.wireframe_mode == instancing.last_wireframe_mode;

        if !cache_valid {
            // Cache miss : rebuild batches and upload instance data.
            // Each entry keeps its original scene-item index alongside the item
            // reference so the light-probe SH block (keyed by that index) can be
            // recovered after sorting reorders the list.
            let mut sorted_items: Vec<(usize, &SceneRenderItem)> = scene_items
                .iter()
                .enumerate()
                .filter(|(idx, _)| instanceable[*idx])
                .collect();

            let binding = resources.instancing.material_texture_binding;
            sorted_items.sort_unstable_by(|(_, a), (_, b)| {
                // Batch grouping key (shared with the batch-split condition
                // below). two_sided is part of the key because the two pipelines
                // differ in cull mode, so a batch must not mix one- and
                // two-sided items.
                let batch_ord = batch_group_key(a, binding).cmp(&batch_group_key(b, binding));
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
                a.settings.pick_id.0.cmp(&b.settings.pick_id.0)
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
                        let a = sorted_items[batch_start].1;
                        let b = sorted_items[i].1;
                        batch_group_key(a, binding) != batch_group_key(b, binding)
                    };

                    if at_end || key_changed {
                        let batch_items = &sorted_items[batch_start..i];
                        let rep = batch_items[0].1;
                        let instance_offset = all_instances.len() as u32;
                        let is_transparent = rep.settings.opacity < 1.0;

                        // All items in a batch share the same mesh_id (batch key).
                        // Look up the mesh once and reuse it for both index_count and
                        // per-instance AABB transforms, avoiding N redundant hash map
                        // lookups inside the inner loop.
                        let batch_idx = instanced_batches.len() as u32;
                        let batch_mesh = resources.mesh_store.get(rep.mesh_id);
                        let mesh_index_count = batch_mesh.map(|m| m.index_count).unwrap_or(0);
                        // The draw binds the whole slab chunk once and offsets
                        // per mesh; carry those offsets so the cull kernel writes
                        // them into the indirect args (and the direct path reads
                        // them from the batch).
                        let (mesh_first_index, mesh_base_vertex) = batch_mesh
                            .map(|m| {
                                (
                                    resources.geometry.first_index(m.index_span),
                                    resources.geometry.base_vertex(m.vertex_span),
                                )
                            })
                            .unwrap_or((0, 0));

                        for (orig_idx, item) in batch_items {
                            let cm = common_material(item);
                            let material_id = resources.material_gpu_builder.intern(&item.material);
                            let custom_data_id = resources
                                .custom_data_builder
                                .intern(item.settings.custom_data);
                            // Styled back-face `Pattern` world scale: transform the
                            // mesh AABB by this instance's model to get its world
                            // extent, then divide the pattern scale by it. Mirrors
                            // the per-object derivation (`per_object.rs`). Zero for
                            // every non-Pattern policy.
                            let backface_pattern_scale = match item.material.backface_policy {
                                crate::scene::material::BackfacePolicy::Pattern(cfg) => {
                                    let world_extent = batch_mesh
                                        .map(|m| {
                                            let model = glam::Mat4::from_cols_array_2d(&item.model);
                                            m.aabb.transformed(&model).longest_side()
                                        })
                                        .unwrap_or(1.0)
                                        .max(1e-6);
                                    cfg.scale / world_extent
                                }
                                _ => 0.0,
                            };
                            // Recover this item's light-probe SH block (assigned
                            // in the shared prepass, keyed by scene-item index).
                            let probe = probe_indices[*orig_idx];
                            all_instances.push(InstanceData {
                                model: cm.model,
                                colour: cm.colour,
                                selected: cm.selected,
                                // The instanced shader shades wireframe the same way the
                                // per-object shader does: a per-instance flag that swaps
                                // the fragment colour for flat grey, not a separate
                                // pipeline or topology (see `mesh_instanced.wgsl`'s
                                // `inst.wireframe` check, mirroring `mesh.wgsl`'s
                                // `object.wireframe`). Previously hardcoded to 0 here, so
                                // a batched item's own `settings.wireframe` was silently
                                // dropped -- it rendered solid instead.
                                wireframe: (frame.viewport.wireframe_mode
                                    || item.settings.wireframe)
                                    as u32,
                                has_texture: cm.has_texture,
                                has_normal_map: cm.has_normal_map,
                                has_ao_map: cm.has_ao_map,
                                unlit: cm.unlit,
                                receive_shadows: cm.receive_shadows,
                                // Shading scalars (PBR terms, ranges, emissive,
                                // use_pbr/use_flat) live in material_gpu_buf, read
                                // via material_id; alpha stays per-instance for the
                                // shadow-cutout pass.
                                material_id,
                                alpha_cutoff: match item.material.alpha_mode {
                                    crate::scene::material::AlphaMode::Mask(c) => c,
                                    _ => 0.5,
                                },
                                alpha_flag: matches!(
                                    item.material.alpha_mode,
                                    crate::scene::material::AlphaMode::Mask(_)
                                ) as u32,
                                has_light_probe: probe.map_or(0, |_| 1),
                                light_probe_index: probe.unwrap_or(0),
                                ignore_clip: item.settings.ignore_clip as u32,
                                custom_data_id,
                                backface_pattern_scale,
                                _pad: 0,
                            });
                            if let Some(mesh) = batch_mesh {
                                let model = glam::Mat4::from_cols_array_2d(&item.model);
                                let world_aabb = mesh.aabb.transformed(&model);
                                all_aabbs.push(InstanceAabb {
                                    min: world_aabb.min.into(),
                                    batch_index: batch_idx,
                                    max: world_aabb.max.into(),
                                    cast_shadows: if item.settings.cast_shadows { 1 } else { 0 },
                                });
                            }
                        }

                        // vis_offset is the prefix sum of instance counts; since
                        // instances are laid out contiguously per batch, it equals
                        // instance_offset.
                        batch_metas.push(BatchMeta {
                            index_count: mesh_index_count,
                            first_index: mesh_first_index,
                            instance_offset,
                            instance_count: batch_items.len() as u32,
                            vis_offset: instance_offset,
                            is_transparent: if is_transparent { 1 } else { 0 },
                            base_vertex: mesh_base_vertex,
                            _pad: 0,
                        });

                        instanced_batches.push(InstancedBatch {
                            mesh_id: rep.mesh_id,
                            texture_id: rep.material.texture_id,
                            normal_map_id: rep.material.normal_map_id,
                            ao_map_id: rep.material.ao_map_id,
                            metallic_roughness_id: rep.material.metallic_roughness_texture_id,
                            emissive_id: rep.material.emissive_texture_id,
                            instance_offset,
                            instance_count: batch_items.len() as u32,
                            is_transparent,
                            two_sided: rep.material.is_two_sided(),
                            is_cutout: matches!(
                                rep.material.alpha_mode,
                                crate::scene::material::AlphaMode::Mask(_)
                            ),
                            // Texture presence is batch-uniform (texture_id is
                            // in the batch key); the mask discard only fires on
                            // textured instances.
                            has_alpha_mask: rep.material.texture_id.is_some()
                                && batch_items.iter().any(|(_, it)| {
                                    matches!(
                                        it.material.alpha_mode,
                                        crate::scene::material::AlphaMode::Mask(_)
                                    )
                                }),
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
            let structure_preserved = instancing.cached_instance_count > 0
                && all_instances.len() == instancing.cached_instance_count
                && instanced_batches.len() == instancing.cached_batches.len()
                && instanced_batches
                    .iter()
                    .zip(&instancing.cached_batches)
                    .all(|(a, b)| {
                        a.mesh_id == b.mesh_id
                            && a.instance_offset == b.instance_offset
                            && a.instance_count == b.instance_count
                            && a.two_sided == b.two_sided
                    });
            let force = std::mem::replace(&mut instancing.force_full_upload, false);

            if structure_preserved && !force {
                let inst_stride = std::mem::size_of::<InstanceData>() as u64;
                let aabb_stride = std::mem::size_of::<InstanceAabb>() as u64;
                // Ensure the hash vec is the right length (it should already be,
                // but guard against a first-run edge case).
                if instancing.cached_instance_hashes.len() != instanced_batches.len() {
                    instancing
                        .cached_instance_hashes
                        .resize(instanced_batches.len(), 0);
                }
                for (bi, batch) in instanced_batches.iter().enumerate() {
                    let start = batch.instance_offset as usize;
                    let end = start + batch.instance_count as usize;
                    let new_bytes =
                        bytemuck::cast_slice::<InstanceData, u8>(&all_instances[start..end]);
                    let new_hash = hash_instance_bytes(new_bytes);
                    if new_hash != instancing.cached_instance_hashes[bi] {
                        if let Some(buf) = resources.instancing.storage_buf.as_ref() {
                            queue.write_buffer(
                                buf,
                                batch.instance_offset as u64 * inst_stride,
                                new_bytes,
                            );
                            resources.frame_upload_bytes += new_bytes.len() as u64;
                        }
                        if let Some(aabb_buf) = resources.cull.aabb_buf.as_ref() {
                            let aabb_bytes =
                                bytemuck::cast_slice::<InstanceAabb, u8>(&all_aabbs[start..end]);
                            queue.write_buffer(
                                aabb_buf,
                                batch.instance_offset as u64 * aabb_stride,
                                aabb_bytes,
                            );
                            resources.frame_upload_bytes += aabb_bytes.len() as u64;
                        }
                        instancing.cached_instance_hashes[bi] = new_hash;
                        batches_reuploaded += 1;
                    } else {
                        batches_skipped += 1;
                    }
                }
            } else {
                resources.upload_instance_data(device, queue, &all_instances);
                resources.upload_cull_inputs(device, queue, &all_aabbs, &batch_metas);
                // The instance storage buffer was rebuilt, so every viewport's
                // cull bind groups now reference a stale binding-0 buffer.
                instancing.instance_gen = instancing.instance_gen.wrapping_add(1);
                batches_reuploaded = instanced_batches.len() as u32;
                // Rebuild the hash cache so the next partial-upload check is seeded.
                instancing.cached_instance_hashes.clear();
                for batch in &instanced_batches {
                    let start = batch.instance_offset as usize;
                    let end = start + batch.instance_count as usize;
                    let bytes =
                        bytemuck::cast_slice::<InstanceData, u8>(&all_instances[start..end]);
                    instancing
                        .cached_instance_hashes
                        .push(hash_instance_bytes(bytes));
                }
            }

            instancing.cached_instance_count = all_instances.len();
            instancing.cached_aabbs = all_aabbs;
            // The cached shadow render bundles replay the batch draw sequence;
            // any change to the batch list (not just structure: textures and
            // cutout flags pick pipelines and bind groups too) invalidates them.
            if instanced_batches != instancing.cached_batches {
                instancing.batches_gen = instancing.batches_gen.wrapping_add(1);
            }
            instancing.cached_batches = instanced_batches;
            instancing.batches = instancing.cached_batches.clone();

            instancing.last_scene_generation = frame.scene.generation;
            instancing.last_wireframe_mode = frame.viewport.wireframe_mode;
            instancing.last_selection_generation = frame.interaction.selection_generation;
            instancing.last_scene_items_count = scene_items.len();
            instancing.last_instancable_count = sorted_items.len();
            instancing.last_resource_free_epoch = resources.resource_free_epoch;

            for batch in &instancing.batches {
                resources.get_instance_bind_group(
                    device,
                    batch.texture_id,
                    batch.normal_map_id,
                    batch.ao_map_id,
                    batch.metallic_roughness_id,
                    batch.emissive_id,
                );
            }
        } else {
            for batch in &instancing.batches {
                resources.get_instance_bind_group(
                    device,
                    batch.texture_id,
                    batch.normal_map_id,
                    batch.ao_map_id,
                    batch.metallic_roughness_id,
                    batch.emissive_id,
                );
            }
        }

        // Under bindless, the direct colour draws share one frame-constant
        // texture-array bind group. Rebuild it when the instance buffer or the
        // texture set changed (a no-op on the per-batch binding).
        resources.ensure_bindless_colour_bind_group(device, instancing.instance_gen);

        (batches_reuploaded, batches_skipped)
    }

    /// Run the main-camera GPU cull for one viewport, writing this viewport's
    /// visibility list and indirect draw args into `cull_state`.
    ///
    /// The cull inputs (per-instance AABBs, per-batch meta) are shared across
    /// viewports and were uploaded by `prepare_instanced` in scene scope. This
    /// runs once per viewport against that viewport's camera, so two viewports
    /// on different cameras get independent visibility results.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_viewport_cull(
        resources: &mut DeviceResources,
        cull_state: &mut crate::resources::ViewportCullState,
        instancing: &mut InstancingState,
        ts_query_set: Option<&crate::gpu::QuerySet>,
        ts_written_mask: &std::sync::atomic::AtomicU32,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        if !instancing.gpu_culling_enabled
            || !instancing.use_instancing
            || instancing.batches.is_empty()
            || instancing.cached_instance_count == 0
        {
            return;
        }

        let instance_count = instancing.cached_instance_count as u32;
        let batch_count = instancing.batches.len() as u32;

        // Do all mutable borrows before taking immutable borrows from resources.
        if instancing.cull_resources.is_none() {
            instancing.cull_resources = Some(crate::renderer::indirect::CullResources::new(device));
        }
        resources.ensure_cull_instance_pipelines(device);
        cull_state.ensure_outputs(device, instance_count, batch_count);
        // Drop cull bind groups whose binding-0 instance storage buffer was
        // rebuilt this frame; `ensure_outputs` already handles a resized vis
        // buffer. Also drop them when the free epoch moved: these bind groups
        // sample the albedo/normal/ao views (bindings 1/3/4) and double as the
        // indirect draw's group-1, and `replace_texture` swaps the view under a
        // stable id without changing the cache key, so a texture update would
        // otherwise keep drawing the old pixels. Mirrors the eviction
        // `replace_texture` already does for the non-culled instance bind groups.
        if cull_state.built_gen != instancing.instance_gen
            || cull_state.built_free_epoch != resources.resource_free_epoch
        {
            cull_state.instance_cull_bind_groups.clear();
            cull_state.bindless_cull_bind_group = None;
            cull_state.built_gen = instancing.instance_gen;
            cull_state.built_free_epoch = resources.resource_free_epoch;
        }
        for batch in &instancing.batches.clone() {
            resources.get_instance_cull_bind_group(
                cull_state,
                device,
                batch.texture_id,
                batch.normal_map_id,
                batch.ao_map_id,
                batch.metallic_roughness_id,
                batch.emissive_id,
            );
        }
        // Under bindless the culled colour draws share one texture-array bind
        // group per viewport (the per-batch groups above still serve the
        // shadow-cutout cull path); a no-op on the per-batch binding.
        resources.get_bindless_cull_bind_group(cull_state, device);

        // GPU-driven submission: form the opaque draw groups and size the
        // per-viewport compaction buffers. Active only under the bindless +
        // native-multi-draw path; leaves `draw_groups` empty otherwise, so the
        // draw loop keeps the CPU run-forming path.
        let gpu_driven =
            build_and_upload_draw_groups(resources, instancing, cull_state, device, queue, frame);

        // Now take immutable borrows to the GPU buffers for dispatch.
        if let (
            Some(aabb_buf),
            Some(meta_buf),
            Some(counter_buf),
            Some(vis_buf),
            Some(indirect_buf),
        ) = (
            resources.cull.aabb_buf.as_ref(),
            resources.cull.batch_meta_buf.as_ref(),
            cull_state.batch_counter_buf.as_ref(),
            cull_state.visibility_index_buf.as_ref(),
            cull_state.indirect_args_buf.as_ref(),
        ) {
            let vp_mat = frame.camera.render_camera.view_proj();
            let cpu_frustum = crate::camera::frustum::Frustum::from_view_proj(&vp_mat);

            let cull = instancing.cull_resources.as_ref().unwrap();
            let mut encoder =
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("cull_encoder"),
                });
            let sub = crate::plugin_api::CullSubmission {
                instance_aabbs: aabb_buf,
                instance_count,
                batch_meta: meta_buf,
                batch_count,
                counter: counter_buf,
                visible_out: vis_buf,
                indirect_out: indirect_buf,
                shadow_pass: false,
            };
            let cull_ts = ts_query_set.map(|qs| (qs, ts_written_mask));
            // HiZ occlusion: reproject last frame's depth into this camera
            // and build the pyramid into the cull encoder, before the cull
            // dispatch that samples it. `build` is false on the first frame
            // (nothing to reproject yet) and after a resize, which leaves
            // the cull frustum-only for that frame.
            let vp_cols = vp_mat.to_cols_array_2d();
            let built = resources.occlusion_culling_enabled()
                && cull_state.build_hiz_reprojected(queue, &mut encoder, vp_cols);
            let (hiz_view, hiz_dims) = if built {
                let (view, dims) = cull_state.hiz_cull_view().unwrap();
                (Some(view), dims)
            } else {
                (None, [1.0, 1.0])
            };
            let extras = crate::renderer::indirect::MainCullExtras {
                view_proj: vp_cols,
                viewport: hiz_dims,
                hiz_view,
                do_occlusion: built,
            };
            cull.dispatch(
                &mut encoder,
                device,
                queue,
                &cpu_frustum,
                None,
                &sub,
                cull_ts,
                Some(&extras),
            );

            // GPU-driven submission: compact the per-batch cull args into one
            // multi-draw range per group. Same encoder, after the cull, so the
            // storage barrier orders it after the args were written.
            if gpu_driven {
                if let (Some(group_id), Some(group_arg_base), Some(compacted), Some(counts)) = (
                    instancing.group_id_buf.as_ref(),
                    instancing.group_arg_base_buf.as_ref(),
                    cull_state.compacted_args_buf.as_ref(),
                    cull_state.draw_counts_buf.as_ref(),
                ) {
                    cull.compact_draws(
                        &mut encoder,
                        device,
                        queue,
                        batch_count,
                        indirect_buf,
                        group_arg_base,
                        group_id,
                        compacted,
                        counts,
                    );
                }
            }

            // Copy indirect_args_buf to the CPU-readable staging buffer so the
            // visible instance count can be read back on a later frame. The
            // readback holds a single set of counters, so it tracks the primary
            // viewport (index 0) only. Skip while a map is in flight or unread so
            // the buffer is not overwritten before `prepare()` has read it.
            let do_readback = frame.camera.viewport_index == 0
                && !instancing.indirect_readback_pending
                && !instancing.indirect_map_inflight;
            if do_readback {
                // Stage the per-batch indirect args followed by the 8-byte
                // cull breakdown counters ([total, frustum_visible]) in one
                // buffer, read back together on a later frame.
                let indirect_bytes = batch_count as u64 * 20;
                let total_bytes = indirect_bytes + 8;
                if instancing
                    .indirect_readback_buf
                    .as_ref()
                    .map_or(0, |b| b.size())
                    < total_bytes
                {
                    instancing.indirect_readback_buf =
                        Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                            label: Some("indirect_readback_buf"),
                            size: total_bytes,
                            usage: crate::gpu::BufferUsages::COPY_DST
                                | crate::gpu::BufferUsages::MAP_READ,
                            mapped_at_creation: false,
                        }));
                }
                if let Some(ref rb_buf) = instancing.indirect_readback_buf {
                    if indirect_bytes > 0 {
                        encoder.copy_buffer_to_buffer(indirect_buf, 0, rb_buf, 0, indirect_bytes);
                    }
                    encoder.copy_buffer_to_buffer(
                        cull.main_stats_buf(),
                        0,
                        rb_buf,
                        indirect_bytes,
                        8,
                    );
                }
            }
            sink.push(encoder.finish());
            if do_readback {
                instancing.indirect_readback_batch_count = batch_count;
                instancing.indirect_readback_pending = true;
            }
        }
    }
}

#[cfg(test)]
mod batch_key_tests {
    use super::*;

    fn item_with(mesh: usize, tex: Option<u64>, two_sided: bool) -> SceneRenderItem {
        let mut item = SceneRenderItem::default();
        // MeshId is a generational slot handle; the key only reads its index.
        item.mesh_id = crate::resources::mesh::mesh_store::MeshId::from_index(mesh as u32);
        item.material.texture_id = tex.map(TextureId::from_raw);
        if two_sided {
            item.material.backface_policy = crate::scene::material::BackfacePolicy::Identical;
        }
        item
    }

    #[test]
    fn per_batch_splits_on_texture_id() {
        let a = item_with(0, Some(1), false);
        let b = item_with(0, Some(2), false);
        assert_ne!(
            batch_group_key(&a, MaterialTextureBinding::PerBatch),
            batch_group_key(&b, MaterialTextureBinding::PerBatch),
            "per-batch binding must split two textures into separate batches",
        );
    }

    #[test]
    fn bindless_collapses_texture_id() {
        let a = item_with(0, Some(1), false);
        let b = item_with(0, Some(2), false);
        assert_eq!(
            batch_group_key(&a, MaterialTextureBinding::Bindless),
            batch_group_key(&b, MaterialTextureBinding::Bindless),
            "bindless binding must collapse different textures on one mesh into one batch",
        );
    }

    #[test]
    fn mesh_and_two_sided_always_split() {
        // Different mesh never batches together, in either mode.
        for binding in [
            MaterialTextureBinding::PerBatch,
            MaterialTextureBinding::Bindless,
        ] {
            let a = item_with(0, None, false);
            let b = item_with(1, None, false);
            assert_ne!(batch_group_key(&a, binding), batch_group_key(&b, binding));
        }
        // two_sided stays in the key even under bindless (the pipelines differ
        // in cull mode), so it splits regardless of texture collapse.
        let one_sided = item_with(0, Some(5), false);
        let two_sided = item_with(0, Some(5), true);
        assert_ne!(
            batch_group_key(&one_sided, MaterialTextureBinding::Bindless),
            batch_group_key(&two_sided, MaterialTextureBinding::Bindless),
        );
    }
}
