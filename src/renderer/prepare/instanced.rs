//! Instanced (GPU-driven) mesh draw preparation.

use super::*;

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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        // Items with active_attribute, two-sided policy, matcap, or param_vis are
        // excluded from the instanced batch filter. Items whose mesh has an active
        // compute filter result are also excluded so the per-object path can apply
        // the filtered index buffer (instanced draws always use the full index buffer).
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
            && scene_items.len() == instancing.last_scene_items_count;

        if !cache_valid {
            // Cache miss : rebuild batches and upload instance data.
            let mut sorted_items: Vec<&SceneRenderItem> = scene_items
                .iter()
                .enumerate()
                .filter(|(idx, _)| instanceable[*idx])
                .map(|(_, item)| item)
                .collect();

            sorted_items.sort_unstable_by(|a, b| {
                // Batch grouping key (must match the batch-split condition).
                // two_sided is part of the key because the two pipelines differ
                // in cull mode, so a batch must not mix one- and two-sided items.
                let batch_ord = (
                    a.mesh_id.index(),
                    a.material.texture_id,
                    a.material.normal_map_id,
                    a.material.ao_map_id,
                    a.material.is_two_sided(),
                )
                    .cmp(&(
                        b.mesh_id.index(),
                        b.material.texture_id,
                        b.material.normal_map_id,
                        b.material.ao_map_id,
                        b.material.is_two_sided(),
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
                        let a = sorted_items[batch_start];
                        let b = sorted_items[i];
                        a.mesh_id != b.mesh_id
                            || a.material.texture_id != b.material.texture_id
                            || a.material.normal_map_id != b.material.normal_map_id
                            || a.material.ao_map_id != b.material.ao_map_id
                            || a.material.is_two_sided() != b.material.is_two_sided()
                    };

                    if at_end || key_changed {
                        let batch_items = &sorted_items[batch_start..i];
                        let rep = batch_items[0];
                        let instance_offset = all_instances.len() as u32;
                        let is_transparent = rep.settings.opacity < 1.0;

                        // All items in a batch share the same mesh_id (batch key).
                        // Look up the mesh once and reuse it for both index_count and
                        // per-instance AABB transforms, avoiding N redundant hash map
                        // lookups inside the inner loop.
                        let batch_idx = instanced_batches.len() as u32;
                        let batch_mesh = resources.mesh_store.get(rep.mesh_id);
                        let mesh_index_count = batch_mesh.map(|m| m.index_count).unwrap_or(0);

                        for item in batch_items {
                            let cm = common_material(item);
                            all_instances.push(InstanceData {
                                model: cm.model,
                                colour: cm.colour,
                                selected: cm.selected,
                                wireframe: 0, // always 0 : wireframe uses per-object pipeline
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
                                unlit: cm.unlit,
                                receive_shadows: cm.receive_shadows,
                                use_flat: cm.use_flat,
                                _pad_inst: 0,
                                uv_transform: cm.uv_transform,
                                ao_range: cm.ao_range,
                                _pad_ao_range: [0.0, 0.0],
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
                            two_sided: rep.material.is_two_sided(),
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
                        }
                        if let Some(aabb_buf) = resources.cull.aabb_buf.as_ref() {
                            let aabb_bytes =
                                bytemuck::cast_slice::<InstanceAabb, u8>(&all_aabbs[start..end]);
                            queue.write_buffer(
                                aabb_buf,
                                batch.instance_offset as u64 * aabb_stride,
                                aabb_bytes,
                            );
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
            instancing.cached_batches = instanced_batches;
            instancing.batches = instancing.cached_batches.clone();

            instancing.last_scene_generation = frame.scene.generation;
            instancing.last_selection_generation = frame.interaction.selection_generation;
            instancing.last_scene_items_count = scene_items.len();
            instancing.last_instancable_count = sorted_items.len();

            for batch in &instancing.batches {
                resources.get_instance_bind_group(
                    device,
                    batch.texture_id,
                    batch.normal_map_id,
                    batch.ao_map_id,
                );
            }
        } else {
            for batch in &instancing.batches {
                resources.get_instance_bind_group(
                    device,
                    batch.texture_id,
                    batch.normal_map_id,
                    batch.ao_map_id,
                );
            }
        }

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
        ts_query_set: Option<&wgpu::QuerySet>,
        ts_written_mask: &std::sync::atomic::AtomicU32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
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
        // buffer.
        if cull_state.built_gen != instancing.instance_gen {
            cull_state.instance_cull_bind_groups.clear();
            cull_state.built_gen = instancing.instance_gen;
        }
        for batch in &instancing.batches.clone() {
            resources.get_instance_cull_bind_group(
                cull_state,
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
            resources.cull.aabb_buf.as_ref(),
            resources.cull.batch_meta_buf.as_ref(),
            cull_state.batch_counter_buf.as_ref(),
            cull_state.visibility_index_buf.as_ref(),
            cull_state.indirect_args_buf.as_ref(),
        ) {
            let vp_mat = frame.camera.render_camera.view_proj();
            let cpu_frustum = crate::camera::frustum::Frustum::from_view_proj(&vp_mat);

            let cull = instancing.cull_resources.as_ref().unwrap();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                        Some(device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("indirect_readback_buf"),
                            size: total_bytes,
                            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
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
            queue.submit(std::iter::once(encoder.finish()));
            if do_readback {
                instancing.indirect_readback_batch_count = batch_count;
                instancing.indirect_readback_pending = true;
            }
        }
    }
}
