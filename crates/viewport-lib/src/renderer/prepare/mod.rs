use super::types::{ClipShape, SceneEffects, ViewportEffects};
use super::*;

mod instanced;
mod lighting;
mod math;
mod mesh_material;
mod overlay_compile;
mod overlay_geometry;
mod overlay_style_check;
mod overlay_vector;
mod per_object;
mod projection;
mod scene_uploads;
mod shadow_pass;
mod viewport_interaction;
mod viewport_misc;
mod viewport_overlays;

use math::*;
use mesh_material::*;
pub(crate) use mesh_material::{
    active_submesh_materials, has_opaque_draws, has_transparent_draws, is_instanceable,
};
use overlay_geometry::*;
use projection::*;

/// One cube-map face of a point-light shadow: which atlas slot and face it
/// occupies, the light-space view-projection used to render it, and the light
/// position and range for depth reconstruction.
pub(super) struct PointShadowFace {
    pub(super) slot: u32,
    pub(super) face: u32,
    pub(super) view_proj: glam::Mat4,
    pub(super) light_pos: glam::Vec3,
    pub(super) range: f32,
}

/// Byte stride between consecutive point-light cube-map face entries in the
/// per-face dynamic-offset uniform buffer.
pub(super) const POINT_FACE_STRIDE: u64 = 256;

/// Transient per-frame lighting results produced by the lighting phase and
/// consumed by the shadow depth pass: the directional cascade matrices and the
/// point-light cube-map faces to render this frame.
pub(super) struct LightingFrame {
    pub(super) point_shadow_faces: Vec<PointShadowFace>,
    pub(super) tile_size: u32,
    pub(super) cascade_view_projs: [glam::Mat4; 4],
    pub(super) effective_cascade_count: usize,
}

impl ViewportRenderer {
    /// Pick a level for every item with a LOD group and overwrite its `mesh_id`
    /// with the chosen mesh. Items without a group are left alone.
    ///
    /// Returns `(items_resolved, switches, culled, reduced)`: how many LOD items
    /// drew, how many tracked items (those with a pick id) changed level since
    /// last frame, how many fell below their group's cull size and were hidden,
    /// and how many drew at a reduced level (below full detail).
    ///
    /// Screen size is measured from the group's full-detail mesh AABB, so the
    /// metric does not depend on which level happens to be drawn. Items with a
    /// pick id use hysteresis through `lod_levels`; items without one resolve
    /// fresh each frame.
    fn resolve_lod(
        resources: &crate::resources::DeviceResources,
        lod_levels: &mut std::collections::HashMap<u64, usize>,
        camera: &RenderCamera,
        items: &mut [SceneRenderItem],
    ) -> (u32, u32, u32, u32) {
        let mut resolved = 0u32;
        let mut switches = 0u32;
        let mut culled = 0u32;
        let mut reduced = 0u32;
        let mut seen: Vec<u64> = Vec::new();

        for item in items.iter_mut() {
            let Some(group_id) = item.lod_group else {
                continue;
            };
            let Some(group) = resources.lod_group(group_id) else {
                continue;
            };
            let Some(mesh) = resources.mesh(group.mesh_at(0)) else {
                continue;
            };

            let model = glam::Mat4::from_cols_array_2d(&item.model);
            let size = crate::resources::projected_screen_size(&mesh.aabb, &model, camera);

            // Too small to bother drawing: hide it for this frame. Hidden items
            // are skipped by every draw and shadow pass.
            if group.should_cull(size) {
                item.settings.hidden = true;
                culled += 1;
                continue;
            }

            let pick = item.settings.pick_id.0;
            let level = if pick == 0 {
                group.level_for_size(size)
            } else {
                let current = lod_levels.get(&pick).copied().unwrap_or(0);
                let next = group.select(size, current);
                if next != current {
                    switches += 1;
                }
                lod_levels.insert(pick, next);
                seen.push(pick);
                next
            };

            item.mesh_id = group.mesh_at(level);
            resolved += 1;
            if level > 0 {
                reduced += 1;
            }
        }

        // Drop tracking state for items that are no longer present so the map
        // does not grow without bound as pick ids come and go.
        if seen.is_empty() {
            lod_levels.clear();
        } else if lod_levels.len() > seen.len() {
            let keep: std::collections::HashSet<u64> = seen.into_iter().collect();
            lod_levels.retain(|k, _| keep.contains(k));
        }

        (resolved, switches, culled, reduced)
    }

    /// Scene-global prepare stage: lighting, shadow pass, batching, scivis.
    ///
    /// Call once per frame before any `prepare_viewport_internal` calls.
    ///
    /// Reads `scene_fx` for lighting and IBL, `frame.scene` for the items, and
    /// `frame.camera` for shadow cascade computation.
    pub(super) fn prepare_scene_internal(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        scene_fx: &SceneEffects<'_>,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        // Start of frame: rotate the double-buffered timestamp query sets.
        // Passes write this frame's timestamps into `ts_query_set`, setting
        // their slot bit as they encode (shadow here in prepare, scene/oit/post
        // later in render). The set written last frame moves to
        // `ts_query_set_prev` together with its written mask, and the render
        // path resolves it from this frame's encoder: one submission after the
        // passes that wrote it, which Metal's stage-boundary counters need to
        // return settled end-of-pass samples for short passes.
        std::mem::swap(&mut self.ts_query_set, &mut self.ts_query_set_prev);
        self.ts_prev_mask = self
            .ts_written_mask
            .swap(0, std::sync::atomic::Ordering::Relaxed);

        self.ensure_frame_pipelines(device, frame);
        if scene_fx.environment.as_ref().is_some_and(|e| e.show_skybox) {
            self.resources.ensure_skybox_pipeline(device);
        }

        // Reset the per-material transform interner for this frame. The per-object
        // and instanced passes below intern each item's material into it; the
        // buffer is uploaded at the end of this function (and again after
        // per-viewport foreground objects intern).
        self.resources.material_gpu_builder.reset();
        self.resources.custom_data_builder.reset();

        // Drain the upload-job runner. Worker results received since the last
        // frame are observed, GPU submissions are polled for completion, and
        // any registered completion callbacks fire on this thread.
        //
        // Only a presented frame pumps the pipeline. A derivative render (a
        // capture / bake) reads the currently resident scene and must not
        // advance promotions: doing so drains the one-cycle promotion window a
        // consumer polls to fire a deferred bind, which strands the bind
        // permanently (the runner clears its `finished` table every call).
        if self.render_advances_state() {
            match self.upload_budget {
                Some(d) => self.resources.process_uploads_with_budget(
                    device,
                    queue,
                    crate::resources::FrameBudget::from_now(d),
                ),
                None => self.resources.process_uploads(device, queue),
            }
        }

        // Run the mesh-family pipeline rebuild that deformer registration
        // deferred, before anything draws or builds against those pipelines.
        // A burst of registrations since the last frame costs one rebuild
        // here instead of one per call.
        self.resources.flush_mesh_pipeline_rebuild(device);

        // Ensure built-in colourmaps and matcaps are uploaded on first frame.
        self.resources.ensure_colourmaps_initialized(device, queue);
        self.resources.ensure_matcaps_initialized(device, queue);

        let plugin_frame_index = self.plugin_frame_index;

        let hdr_family = self.draws_hdr(frame);
        let resources = &mut self.resources;
        let lighting = scene_fx.lighting;

        // Read scene items from the surface submission, then extend with the
        // boundary draws contributed by opaque volume meshes (items in
        // `volume_meshes` whose `transparency` is `None`). The owned vector
        // keeps these extra items alive for the whole prepare pass.
        let mut scene_items_owned: Vec<SceneRenderItem> = {
            let surfaces = match &frame.scene.surfaces {
                SurfaceSubmission::Flat(items) => items.as_ref(),
            };
            let extra = frame
                .scene
                .volume_meshes
                .iter()
                .filter(|item| item.transparency.is_none())
                .map(|item| item.to_render_item());
            surfaces.iter().cloned().chain(extra).collect()
        };

        // Per-camera layer cull. Drop any mesh-family item whose
        // `visibility_mask` shares no bit with this camera's `cull_mask`
        // (the AND is zero). Runs before LOD, instancing, shadow, and picking
        // read the list, so a layer-culled item is absent from every pass.
        // This is the CPU half of the shared layer mask: it costs one AND per
        // item, is inert at the default (`!0 & anything != 0`), and needs no
        // GPU carrier. The GPU-driven instanced cull does not yet honour the
        // mask; items dropped here never reach it, but a batch visible to the
        // camera still runs its full GPU cull regardless of per-light channels.
        let cull_mask = frame.camera.cull_mask;
        scene_items_owned.retain(|item| (item.settings.visibility_mask & cull_mask) != 0);

        // Resolve LOD groups to concrete meshes before anything reads the draw
        // list. Items with a `lod_group` get their `mesh_id` overwritten with the
        // level for their on-screen size; items without one are untouched. Both
        // the instanced and per-object paths see the resolved meshes, and so do
        // the shadow and viewport passes, since this runs once on the shared list.
        let (mut lod_items_resolved, mut lod_switches, mut lod_culled, mut lod_items_reduced) =
            Self::resolve_lod(
                resources,
                &mut self.lod_levels,
                &frame.camera.render_camera,
                &mut scene_items_owned,
            );

        let scene_items: &[SceneRenderItem] = &scene_items_owned;

        let lighting_start = web_time::Instant::now();
        let lighting_frame = Self::prepare_lighting(
            resources,
            &mut self.shadow,
            &mut self.last_cluster_stats,
            &mut self.last_frustum_culled_lights,
            &self.viewport_slots,
            &self.item_type_plugins,
            plugin_frame_index,
            scene_fx,
            device,
            queue,
            frame,
            sink,
            self.ts_query_set.as_ref(),
            &self.ts_written_mask,
        );
        let lighting_ms = lighting_start.elapsed().as_secs_f32() * 1000.0;

        // -- Instancing preparation --
        // Determine instancing mode BEFORE per-object uniforms so we can skip them.
        let visible_count = scene_items.iter().filter(|i| !i.settings.hidden).count();
        let prev_use_instancing = self.instancing.use_instancing;
        self.instancing.use_instancing = visible_count > INSTANCING_THRESHOLD;

        // If instancing mode changed (e.g. objects added/removed crossing the threshold),
        // clear batches so the generation check below forces a rebuild.
        if self.instancing.use_instancing != prev_use_instancing {
            self.instancing.batches.clear();
            self.instancing.last_scene_generation = u64::MAX;
            self.instancing.last_scene_items_count = usize::MAX;
        }
        if self.instancing.use_instancing != prev_use_instancing {
            tracing::debug!(
                visible_objects = visible_count,
                threshold = INSTANCING_THRESHOLD,
                instanced = self.instancing.use_instancing,
                "instancing mode changed"
            );
        }

        let per_object_start = web_time::Instant::now();
        // Set up the pipeline sets of the material plugins the frame references
        // before draw time (paint has no mutable access). This compiles
        // nothing: each pipeline is built by the first draw that selects it.
        // Unknown ids are ignored and those items fall back to built-in
        // shading.
        let mut seen: Vec<u32> = Vec::new();
        for item in scene_items.iter() {
            let Some(pid) = item.material.shading_plugin else {
                continue;
            };
            if seen.contains(&pid.plugin_index()) {
                continue;
            }
            seen.push(pid.plugin_index());
            // The per-object set draws the items that cannot join a batch; the
            // instanced set lets the rest draw one call per batch (see
            // `is_instanceable`).
            if resources.material_plugin_needs_build(pid) {
                resources.ensure_material_plugin_pipelines(device, pid);
            }
            if !resources.material_plugin_instanced_ready(pid) {
                // The instanced set sits on the built-in instanced group-1
                // layout, and its GPU-culled twins on the cull layout, so both
                // come first. Both calls are idempotent.
                resources.ensure_instanced_pipelines(device);
                resources.ensure_cull_instance_pipelines(device);
                resources.ensure_material_plugin_instanced_pipelines(device, pid);
            }
        }
        // Evaluate instanceability once per frame and share the result. Each
        // `is_instanceable` call does several mesh-store and deform lookups, so
        // computing it once here instead of separately in the per-object skip
        // test and the instanced batch filter keeps this O(items) rather than running the
        // same per-item work three times over. At city scale (tens of thousands
        // of resident meshes) that is the difference between a few milliseconds
        // and a few hundred.
        let instanceable: Vec<bool> = scene_items
            .iter()
            .map(|item| is_instanceable(item, resources))
            .collect();
        // Blend each light-probe-lit item's SH into the shared buffer once, so
        // the per-object and instanced paths that draw those items index the
        // same block. No probes uploaded -> all None and nothing written.
        let probe_indices = per_object::prepare_light_probe_sh(resources, scene_items, queue);
        let per_object_bind_groups_built = Self::prepare_per_object(
            resources,
            &mut self.mesh_uniforms,
            self.instancing.use_instancing,
            scene_items,
            &instanceable,
            &probe_indices,
            self.frame_counter,
            device,
            queue,
            frame,
        );
        let uniforms_ms = per_object_start.elapsed().as_secs_f32() * 1000.0;

        let instanced_start = web_time::Instant::now();
        let (batches_reuploaded, batches_skipped) = if self.instancing.use_instancing {
            Self::prepare_instanced(
                resources,
                &mut self.instancing,
                &instanceable,
                scene_items,
                &probe_indices,
                device,
                queue,
                frame,
                hdr_family,
            )
        } else {
            (0, 0)
        };
        settle_gpu_culling(&mut self.instancing, &resources.pipeline_compiler, device);
        let instancing_ms = instanced_start.elapsed().as_secs_f32() * 1000.0;

        let geometry_start = web_time::Instant::now();
        // The particle systems' draw bind groups bake a texture view in at
        // creation, so a free or replace since the last frame has to be picked
        // up before the item type draws from them. The store stays here, so
        // this does too.
        let (inst_resolved, inst_switches, inst_culled, inst_reduced) = Self::upload_mesh_instances(
            resources,
            &mut self.mesh_instance_gpu_data,
            &mut self.mesh_instance_lod_levels,
            device,
            queue,
            frame,
            hdr_family,
        );
        lod_items_resolved += inst_resolved;
        lod_switches += inst_switches;
        lod_culled += inst_culled;
        lod_items_reduced += inst_reduced;
        // The shared line substrate is refilled each frame: the plugin
        // wireframes below are its producers.
        self.polyline_gpu_data.clear();
        // Refresh any deform slots bound to a same-device consumer buffer,
        // GPU-to-GPU, before the mesh render pass reads them.
        resources.run_deform_slot_copies(device, queue);
        // Volume wireframe overlay: OBB from bbox + model matrix.
        // Transparent volume meshes wireframe: boundary mesh edge overlay.
        // Items rendering as opaque already participate in the standard
        // wireframe pass via the surface submission; here we only need to
        // gather boundary edges for items rendering through the projected-tet
        // path so they still get a wireframe overlay.
        self.mesh_uniforms.tvm_wireframe_draws.clear();
        let mut tvm_wireframe_records: Vec<crate::resources::ObjectUniform> = Vec::new();
        for item in &frame.scene.volume_meshes {
            if item.settings.hidden || item.transparency.is_none() {
                continue;
            }
            if !(item.settings.wireframe || frame.viewport.wireframe_mode) {
                continue;
            }
            if resources.mesh_store.get(item.boundary_mesh_id).is_none() {
                continue;
            }
            // The edge buffer is built lazily on first wireframe use.
            resources.ensure_edge_indices(device, item.boundary_mesh_id);
            self.mesh_uniforms
                .tvm_wireframe_draws
                .push(item.boundary_mesh_id);
            let mut record: crate::resources::ObjectUniform = bytemuck::Zeroable::zeroed();
            record.model = item.model;
            record.colour = [0.75, 0.75, 0.75, 1.0];
            record.wireframe = 1;
            tvm_wireframe_records.push(record);
        }
        // One record per draw, selected by instance index. The buffer and its
        // bind group are rebuilt only when a frame needs more records than
        // the buffer holds.
        if tvm_wireframe_records.len() > self.mesh_uniforms.tvm_wireframe_capacity {
            let capacity = tvm_wireframe_records.len().next_power_of_two();
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("tvm_wireframe_uniform"),
                size: (capacity * std::mem::size_of::<crate::resources::ObjectUniform>()) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.mesh_uniforms.tvm_wireframe_capacity = capacity;
            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("tvm_wireframe_bg"),
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
                        resource: crate::gpu::BindingResource::Sampler(&resources.material.sampler),
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
                    crate::gpu::BindGroupEntry {
                        binding: 19,
                        resource: resources.content.fallback_uv1_buf.as_entire_binding(),
                    },
                ],
            });
            self.mesh_uniforms.tvm_wireframe_buf = Some(buf);
            self.mesh_uniforms.tvm_wireframe_bg = Some(bg);
        }
        if let (Some(buf), false) = (
            &self.mesh_uniforms.tvm_wireframe_buf,
            tvm_wireframe_records.is_empty(),
        ) {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&tvm_wireframe_records));
        }

        let geometry_ms = geometry_start.elapsed().as_secs_f32() * 1000.0;

        // -- Frame stats --
        {
            let total = scene_items.len() as u32;
            let visible = scene_items.iter().filter(|i| !i.settings.hidden).count() as u32;
            let mut draw_calls = 0u32;
            let mut triangles = 0u64;
            let instanced_batch_count = if self.instancing.use_instancing {
                self.instancing.batches.len() as u32
            } else {
                0
            };

            if self.instancing.use_instancing {
                for batch in &self.instancing.batches {
                    if let Some(mesh) = resources.mesh_store.get(batch.mesh_id) {
                        draw_calls += 1;
                        triangles += (mesh.index_count / 3) as u64 * batch.instance_count as u64;
                    }
                }
                // Items outside the batches still issue one draw each through
                // the per-object path (one per range for per-submesh-material
                // items); count them so `draw_calls` and `triangles_submitted`
                // cover both paths.
                for (item, inst) in scene_items.iter().zip(instanceable.iter()) {
                    if item.settings.hidden || *inst {
                        continue;
                    }
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                        draw_calls += active_submesh_materials(item, mesh)
                            .map_or(1, |mats| mats.len() as u32);
                        triangles += (mesh.index_count / 3) as u64;
                    }
                }
            } else {
                for item in scene_items {
                    if item.settings.hidden {
                        continue;
                    }
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                        draw_calls += active_submesh_materials(item, mesh)
                            .map_or(1, |mats| mats.len() as u32);
                        triangles += (mesh.index_count / 3) as u64;
                    }
                }
            }

            // Visible items that miss the instanced fast path. Reuses the
            // instanceable bitset computed above, so this is just a count.
            let per_object_items = scene_items
                .iter()
                .zip(instanceable.iter())
                .filter(|(item, inst)| !item.settings.hidden && !**inst)
                .count() as u32;

            // Items drawn instanced whose mesh carries per-mesh deform slot
            // data. The instanced draws bind the empty deform group and the
            // instanced shader passes no flags, so that data does not reach the
            // draw and the item renders undeformed. Per-instance data is
            // excluded from instancing by `is_instanceable`, so it is not at
            // risk and is not counted here.
            //
            // Gated on `use_instancing` as well as the per-item predicate:
            // below the threshold every item draws per-object whatever
            // `is_instanceable` says about it, and the per-object path binds
            // the mesh's real deform group.
            let deform_slots_ignored = if self.instancing.use_instancing {
                scene_items
                    .iter()
                    .zip(instanceable.iter())
                    .filter(|(item, inst)| {
                        **inst
                            && !item.settings.hidden
                            && resources.deform.has_mesh_slot_data(item.mesh_id)
                    })
                    .count() as u32
            } else {
                0
            };

            let deform = resources.deform.take_counters();
            let plugin_draw_calls = self
                .item_type_plugins
                .values()
                .map(|p| p.draw_calls())
                .sum();
            let plugin_upload_bytes = self
                .item_type_plugins
                .values()
                .map(|p| p.upload_bytes())
                .sum();
            self.last_stats = crate::renderer::stats::FrameStats {
                total_objects: total,
                visible_objects: visible,
                culled_objects: total.saturating_sub(visible),
                draw_calls,
                instanced_batches: instanced_batch_count,
                per_object_items,
                per_object_bind_groups_built,
                batches_reuploaded,
                batches_skipped,
                plugin_draw_calls,
                plugin_upload_bytes,
                deform_slots_ignored,
                deform_buffer_reallocations: deform.buffer_reallocations,
                deform_mesh_bind_groups_rebuilt: deform.mesh_bind_groups_rebuilt,
                deform_instance_bind_groups_rebuilt: deform.instance_bind_groups_rebuilt,
                triangles_submitted: triangles,
                shadow_draw_calls: 0,    // Updated below in shadow pass.
                shadow_draw_commands: 0, // Updated below in shadow pass.
                shadow_buffer_binds: 0,  // Updated below in shadow pass.
                slab_chunk_count: resources.geometry.chunk_count(),
                slab_resident_bytes: resources.geometry.resident_bytes(),
                lod_items_resolved,
                lod_switches,
                lod_culled,
                lod_items_reduced,
                gpu_culling_active: self.instancing.gpu_culling_enabled,
                // Clear stale readback if GPU culling is off this frame.
                gpu_visible_instances: if self.instancing.gpu_culling_enabled {
                    self.last_stats.gpu_visible_instances
                } else {
                    None
                },
                gpu_culled_total: if self.instancing.gpu_culling_enabled {
                    self.last_stats.gpu_culled_total
                } else {
                    None
                },
                gpu_frustum_visible: if self.instancing.gpu_culling_enabled {
                    self.last_stats.gpu_frustum_visible
                } else {
                    None
                },
                ..self.last_stats
            };
        }

        // ------------------------------------------------------------------
        let shadow_start = web_time::Instant::now();
        let colour_hdr = frame.effects.display.is_hdr() && !self.direct_paint;
        Self::prepare_shadow_pass(
            resources,
            &mut self.instancing,
            &mut self.shadow,
            &self.item_type_plugins,
            plugin_frame_index,
            lighting,
            scene_items,
            &lighting_frame,
            self.degradation_shadows_skipped,
            &mut self.last_stats,
            self.ts_query_set.as_ref(),
            &self.ts_written_mask,
            device,
            queue,
            frame,
            colour_hdr,
            sink,
        );
        let shadow_ms = shadow_start.elapsed().as_secs_f32() * 1000.0;

        // Hand the LOD-resolved items to the paint pass. `scene_items_owned`
        // carries the surface submission with each item's level mesh selected
        // and culled items hidden, plus the opaque volume boundary draws
        // appended. The draw path reads `self.prepared_surfaces`, so dropping
        // the volume tail keeps it a drop-in match for `frame.scene.surfaces`
        // in both order and length (the per-object bind groups index the same
        // way). Without this the paint pass re-derived from the raw surfaces
        // and every non-instanced LOD swap and cull was silently lost.
        let surface_count = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.len(),
        };
        scene_items_owned.truncate(surface_count);
        self.prepared_surfaces = scene_items_owned;

        // The `resources` borrow ends with the shadow pass above, so record the
        // scene-phase timings into the breakdown now.
        self.prepare_breakdown.lighting_ms = lighting_ms;
        self.prepare_breakdown.uniforms_ms = uniforms_ms;
        self.prepare_breakdown.instancing_ms = instancing_ms;
        self.prepare_breakdown.geometry_ms = geometry_ms;
        self.prepare_breakdown.shadow_ms = shadow_ms;

        // All scene items have interned their material transforms; upload the
        // block buffer so the scene pass can index it. Foreground objects
        // re-upload after they intern in `prepare_viewport_internal`.
        self.resources.upload_material_gpu(queue);
        self.resources.upload_custom_data(device, queue);

        // Item-type wireframes fill the shared line substrate, cleared above.
        // Placed at the end of scene prepare because the plugin context borrows
        // `resources` shared while the upload above holds it mutably.
        self.dispatch_plugin_wireframes(device, queue, frame);

        self.flush_camera_bind_group_rebuild(device);
    }

    /// Rebuild the camera bind groups if something they name was replaced this
    /// prepare: a shadow texture promoted out of its placeholder, or the
    /// custom-data buffer grown. Run at the end of each prepare phase, before
    /// any render pass, so the frame that made the change already draws with
    /// the new resource and not one frame late.
    fn flush_camera_bind_group_rebuild(&mut self, device: &crate::gpu::Device) {
        if self.resources.camera_bind_groups_dirty {
            self.resources.camera_bind_groups_dirty = false;
            self.rebuild_camera_bind_groups(device);
            // The per-object bundle recorded the old camera bind group. Drop it
            // so this frame draws item by item; the next prepare records again.
            self.per_object_bundle = None;
        }
    }

    /// Whether `frame` carries anything the mesh pipeline families draw.
    pub(crate) fn has_mesh_content(frame: &FrameData) -> bool {
        (match &frame.scene.surfaces {
            crate::renderer::SurfaceSubmission::Flat(items) => !items.is_empty(),
        }) || !frame.scene.volume_meshes.is_empty()
            || !frame.scene.mesh_instances.is_empty()
            || !frame.scene.foreground_items.is_empty()
    }

    /// Whether `frame` is drawn through the HDR path. A frame that asks for
    /// HDR is still drawn with the LDR pipelines when the caller paints it
    /// straight into its own render pass.
    pub(crate) fn draws_hdr(&self, frame: &FrameData) -> bool {
        frame.effects.display.is_hdr() && !self.direct_paint
    }

    /// Build the pipelines `frame`'s own passes draw with, if this is the first
    /// frame to ask for them.
    ///
    /// The mesh pipelines, the ground plane and the skybox are not built at
    /// construction. The draw sites test the frame they are handed, so this
    /// runs for the scene frame and again for each viewport's frame: under the
    /// split API those can differ, and a viewport may be the only one asking.
    ///
    /// Only the mesh family the frame is drawn with is built: the HDR one for a
    /// frame on the HDR path, the LDR one otherwise. A renderer that stays on
    /// one path never compiles the other.
    fn ensure_frame_pipelines(&mut self, device: &crate::gpu::Device, frame: &FrameData) {
        if Self::has_mesh_content(frame) {
            if self.draws_hdr(frame) {
                self.resources.ensure_hdr_mesh_pipelines(device);
            } else {
                self.resources.ensure_ldr_mesh_pipelines(device);
            }
        }
        if !matches!(
            frame.effects.ground_plane.mode,
            crate::renderer::types::GroundPlaneMode::None
        ) {
            self.resources.ensure_ground_plane_pipeline(device);
        }
        if frame
            .effects
            .environment
            .as_ref()
            .is_some_and(|e| e.show_skybox)
        {
            self.resources.ensure_skybox_pipeline(device);
        }
    }

    /// Per-viewport prepare stage: camera, clip planes, clip volume, grid, overlays, cap geometry, axes.
    ///
    /// Call once per viewport per frame, after `prepare_scene_internal`.
    /// Reads `viewport_fx` for clip planes, clip volume, cap fill, and post-process settings.
    pub(super) fn prepare_viewport_internal(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        // Ensure a per-viewport camera slot exists for this viewport index.
        self.ensure_viewport_slot(device, frame.camera.viewport_index);
        self.ensure_frame_pipelines(device, frame);

        // Run the main-camera GPU cull for this viewport against its own camera,
        // writing this slot's visibility list and indirect args.
        let vp_idx = frame.camera.viewport_index;
        let hdr_family = self.draws_hdr(frame);
        Self::run_viewport_cull(
            &mut self.resources,
            &mut self.viewport_slots[vp_idx].cull,
            &mut self.instancing,
            self.ts_query_set.as_ref(),
            &self.ts_written_mask,
            device,
            queue,
            frame,
            hdr_family,
            sink,
        );

        self.prepare_clip_uniforms(queue, frame, viewport_fx);
        self.prepare_interaction_state(device, queue, frame, viewport_fx);
        Self::prepare_foreground_objects(
            &mut self.resources,
            &mut self.viewport_slots[vp_idx].foreground_objects,
            &frame.scene.foreground_items,
            device,
            queue,
        );
        // Foreground objects just interned their materials; re-upload the block
        // buffer so any new entries past the scene set are resident.
        self.resources.upload_material_gpu(queue);
        self.resources.upload_custom_data(device, queue);
        // That upload can have grown the custom-data buffer.
        self.flush_camera_bind_group_rebuild(device);
        self.prepare_outline_pass(device, queue, frame, sink);
        self.prepare_sub_highlight(device, queue, frame);

        // Overlay families each build a per-family vertex buffer. When any item
        // carries a non-zero `z_order`, they also record their draw segments into
        // `overlay_draw_segments`, which `finalize_overlay_draw_order` sorts into
        // a single cross-family paint order; otherwise the emit path keeps its
        // fixed family order and the list stays empty.
        // Retained groups always draw through the sorted segment list (they carry
        // their own z_order and are separate draws), so force the ordered path
        // whenever any are present, even if no immediate item sets a z_order.
        let overlay_start = web_time::Instant::now();
        self.overlay_uses_zorder =
            frame.overlays.uses_nonzero_z_order() || !frame.overlays.retained.is_empty();
        self.overlay_draw_segments.clear();
        self.overlay_retained_draws.clear();
        self.overlay_retained_shape_draws.clear();
        self.overlay_instances_ready = false;
        self.overlay_retained_counters = Default::default();
        self.prepare_overlay_labels(device, queue, frame);
        self.prepare_overlay_shapes(device, queue, frame);
        self.finalize_overlay_draw_order(frame);
        self.prepare_breakdown.overlay_ms = overlay_start.elapsed().as_secs_f32() * 1000.0;
        // Publish the retained counters here rather than in the stats assembly:
        // the split API reaches this phase through `prepare_viewport` without
        // assembling a `FrameStats`, and `last_frame_stats()` should still show
        // what the overlay drew. Per viewport, like `overlay_ms` above.
        let overlay_counters = self.overlay_retained_counters;
        self.last_stats.overlay_retained_submitted = overlay_counters.submitted;
        self.last_stats.overlay_retained_drawn = overlay_counters.drawn;
        self.last_stats.overlay_retained_reemitted = overlay_counters.reemitted;
        self.last_stats.overlay_retained_bytes =
            self.resources.content.overlay_geometry.allocated_bytes();
        self.prepare_debug_buffer(frame);
        self.prepare_atlas_blit(queue, frame, viewport_fx);
    }

    /// Upload per-frame data to GPU buffers and render the shadow pass.
    /// Call before `paint()`.
    ///
    /// Returns [`crate::FrameStats`] with per-frame timing and upload metrics.
    /// Submits every prepare pass on `queue` inline, as before; use
    /// [`prepare_deferred`](Self::prepare_deferred) to collect the buffers and
    /// submit them on the device-driving thread instead.
    pub(crate) fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> crate::renderer::stats::FrameStats {
        let mut sink = crate::renderer::SubmitSink::inline(queue);
        self.prepare_into(device, queue, frame, &mut sink)
    }

    /// As [`prepare`](Self::prepare), but the prepare passes push their command
    /// buffers into a returned `Vec` rather than submitting them. The caller must
    /// submit the buffers, in order, on the device-driving thread. Intended for a
    /// render worker that encodes off-thread (see the `SubmitSink` docs for why
    /// submission cannot leave the driving thread).
    ///
    /// Pair with [`render_to_texture_deferred`](Self::render_to_texture_deferred)
    /// for the scene pass and [`submit_frame`](Self::submit_frame) to submit the
    /// combined batch in order.
    pub fn prepare_deferred(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> (
        crate::renderer::stats::FrameStats,
        Vec<crate::gpu::CommandBuffer>,
    ) {
        let mut sink = crate::renderer::SubmitSink::deferred();
        let stats = self.prepare_into(device, queue, frame, &mut sink);
        (stats, sink.into_buffers())
    }

    /// The shared prepare body. `sink` decides whether each pass submits inline or
    /// is collected for a later main-thread submit; `queue` is still needed for
    /// non-submitting queue work (buffer writes, upload-job polling, readbacks).
    fn prepare_into(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) -> crate::renderer::stats::FrameStats {
        let prepare_start = web_time::Instant::now();
        self.prepare_breakdown = crate::renderer::stats::PrepareBreakdown::default();

        let plugin_start = web_time::Instant::now();

        // Dispatch item-type plugin prepare work first so any GPU outputs
        // the plugin produces are visible to the rest of `prepare`.
        let plugin_bufs = self.dispatch_plugin_prepare(device, queue, frame);
        if !plugin_bufs.is_empty() {
            sink.extend(plugin_bufs);
        }

        self.prepare_breakdown.plugin_ms = plugin_start.elapsed().as_secs_f32() * 1000.0;

        // Run plugin culling for the current camera frustum so subsequent
        // plugin paint/shadow calls can skip culled items. Timed apart from
        // prepare above: the two scale with different things.
        let cull_start = web_time::Instant::now();
        if !self.item_type_plugins.is_empty() {
            let vp = frame.camera.render_camera.view_proj();
            let frustum = crate::camera::frustum::Frustum::from_view_proj(&vp);
            self.dispatch_plugin_cull(&frustum, frame);
        }
        self.prepare_breakdown.plugin_cull_ms = cull_start.elapsed().as_secs_f32() * 1000.0;

        // Rebuild or drop the cached per-object render bundle now that the
        // prepared item list, LOD resolve, and per-item bind groups are final.
        self.update_per_object_bundle(device, frame);

        // Read back GPU timestamps without blocking. The staging buffer is
        // mapped on one frame and read on a later one; a non-blocking poll
        // pumps the map callback, and the value is consumed once the map
        // completes. A blocking wait here would stall the CPU on the previous
        // frame's GPU work, which on this workload is most of the frame.
        use std::sync::atomic::Ordering;
        if self.ts_map_inflight {
            let _ = device.poll(crate::gpu::PollType::Poll);
            match self.ts_map_status.load(Ordering::Acquire) {
                1 => {
                    if let Some(ref stg_buf) = self.ts_staging_buf {
                        let data = crate::gpu::mapped_range(stg_buf.slice(..));
                        // Read one begin/end pair per slot. Only slots whose bit is
                        // set in the resolved mask hold valid data this frame; the
                        // rest are passes that did not run, left at 0 ms.
                        let mask = self.ts_pending_mask;
                        let slot_pair = |slot: u32| -> Option<(u64, u64)> {
                            if mask & (1 << slot) == 0 {
                                return None;
                            }
                            // 256-byte stride per slot (resolve alignment).
                            let base = slot as usize * 256;
                            let t0 = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
                            let t1 =
                                u64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap());
                            // Metal can report equal (or unsettled zero)
                            // timestamps at pass boundaries; treat those as no
                            // sample rather than real values.
                            (t0 > 0 && t1 > t0).then_some((t0, t1))
                        };
                        let to_ms = |ticks: u64| ticks as f32 * self.ts_period / 1_000_000.0;
                        let slot_ms = |slot: u32| -> f32 {
                            slot_pair(slot).map_or(0.0, |(t0, t1)| to_ms(t1 - t0))
                        };
                        self.last_stats.gpu_breakdown = crate::renderer::stats::GpuBreakdown {
                            scene_ms: slot_ms(crate::renderer::GPU_TS_SCENE),
                            shadow_ms: slot_ms(crate::renderer::GPU_TS_SHADOW),
                            oit_ms: slot_ms(crate::renderer::GPU_TS_OIT),
                            post_ms: slot_ms(crate::renderer::GPU_TS_POST),
                            cull_ms: slot_ms(crate::renderer::GPU_TS_CULL),
                            cull_plan_ms: slot_ms(crate::renderer::GPU_TS_CULL_PLAN),
                            cull_count_ms: slot_ms(crate::renderer::GPU_TS_CULL_COUNT),
                            cull_scatter_ms: slot_ms(crate::renderer::GPU_TS_CULL_SCATTER),
                            point_shadow_ms: slot_ms(crate::renderer::GPU_TS_POINT_SHADOW),
                            cluster_ms: slot_ms(crate::renderer::GPU_TS_CLUSTER),
                            ssao_ms: slot_ms(crate::renderer::GPU_TS_SSAO),
                            bloom_ms: slot_ms(crate::renderer::GPU_TS_BLOOM),
                            fxaa_ms: slot_ms(crate::renderer::GPU_TS_FXAA),
                            overlay_ms: slot_ms(crate::renderer::GPU_TS_OVERLAY),
                        };
                        // GPU frame time: the span from the first to the last
                        // measured pass. All slots were written on the same
                        // queue in one frame, so min(begin)..max(end) covers
                        // the measured passes and everything submitted between
                        // them (previously this reported only the scene pass,
                        // badly under-reporting shadow/compute-heavy frames).
                        let mut span: Option<(u64, u64)> = None;
                        for slot in 0..crate::renderer::GPU_TS_SLOTS {
                            if let Some((t0, t1)) = slot_pair(slot) {
                                span = Some(match span {
                                    None => (t0, t1),
                                    Some((lo, hi)) => (lo.min(t0), hi.max(t1)),
                                });
                            }
                        }
                        drop(data);
                        if let Some((lo, hi)) = span {
                            self.last_stats.gpu_frame_ms = Some(to_ms(hi - lo));
                            self.last_stats.gpu_sample_generation += 1;
                        }
                        stg_buf.unmap();
                    }
                    self.ts_map_inflight = false;
                }
                2 => {
                    // Map failed; the buffer is left unmapped. Drop this sample
                    // and let the next frame start a fresh readback.
                    self.ts_map_inflight = false;
                }
                _ => {}
            }
        } else if self.ts_data_ready {
            if let Some(ref stg_buf) = self.ts_staging_buf {
                self.ts_map_status.store(0, Ordering::Release);
                let status = self.ts_map_status.clone();
                stg_buf
                    .slice(..)
                    .map_async(crate::gpu::MapMode::Read, move |r| {
                        status.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release);
                    });
                // Pump once so the map can complete immediately when the GPU has
                // already finished; otherwise it completes on a later frame.
                let _ = device.poll(crate::gpu::PollType::Poll);
                self.ts_map_inflight = true;
                self.ts_data_ready = false;
            }
        }

        // Read back the GPU-visible instance count without blocking, using the
        // same map-on-one-frame, read-on-a-later-frame scheme as the timestamps.
        // The staging buffer holds the per-batch indirect args followed by the
        // 8-byte cull breakdown counters ([total, frustum_visible]).
        let indirect_bytes = self.instancing.indirect_readback_batch_count as u64 * 20;
        let bytes = indirect_bytes + 8;
        if self.instancing.indirect_map_inflight {
            let _ = device.poll(crate::gpu::PollType::Poll);
            match self.instancing.indirect_map_status.load(Ordering::Acquire) {
                1 => {
                    if let Some(ref stg_buf) = self.instancing.indirect_readback_buf {
                        let data = crate::gpu::mapped_range(stg_buf.slice(..bytes));
                        let mut visible: u32 = 0;
                        for i in 0..self.instancing.indirect_readback_batch_count as usize {
                            // DrawIndexedIndirect layout: [index_count, instance_count, first_index, base_vertex, first_instance]
                            // instance_count is at byte offset 4 within each 20-byte entry.
                            let off = i * 20 + 4;
                            let n = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            visible = visible.saturating_add(n);
                        }
                        // Cull breakdown counters appended after the indirect args.
                        let stats_off = indirect_bytes as usize;
                        let total =
                            u32::from_le_bytes(data[stats_off..stats_off + 4].try_into().unwrap());
                        let frustum_visible = u32::from_le_bytes(
                            data[stats_off + 4..stats_off + 8].try_into().unwrap(),
                        );
                        drop(data);
                        self.last_stats.gpu_visible_instances = Some(visible);
                        self.last_stats.gpu_culled_total = Some(total);
                        self.last_stats.gpu_frustum_visible = Some(frustum_visible);
                        stg_buf.unmap();
                    }
                    self.instancing.indirect_map_inflight = false;
                }
                2 => {
                    self.instancing.indirect_map_inflight = false;
                }
                _ => {}
            }
        } else if self.instancing.indirect_readback_pending {
            // Clear the flag whether or not we map, so a zero-batch frame does
            // not leave it stuck and block the cull pass from copying again.
            self.instancing.indirect_readback_pending = false;
            if bytes > 0 {
                if let Some(ref stg_buf) = self.instancing.indirect_readback_buf {
                    self.instancing
                        .indirect_map_status
                        .store(0, Ordering::Release);
                    let status = self.instancing.indirect_map_status.clone();
                    stg_buf
                        .slice(..bytes)
                        .map_async(crate::gpu::MapMode::Read, move |r| {
                            status.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release);
                        });
                    let _ = device.poll(crate::gpu::PollType::Poll);
                    self.instancing.indirect_map_inflight = true;
                }
            }
        }

        // Wall-clock duration since the previous prepare() call approximates the frame interval.
        let total_frame_ms = self
            .last_prepare_instant
            .map(|t| t.elapsed().as_secs_f32() * 1000.0)
            .unwrap_or(0.0);

        // Snapshot geometry upload bytes accumulated since the last frame, then reset.
        let upload_bytes = self.resources.frame_upload_bytes;
        self.resources.frame_upload_bytes = 0;
        let pipelines_built_this_frame = self
            .resources
            .frame_pipelines_built
            .swap(0, std::sync::atomic::Ordering::Relaxed);

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

        // Cache pickable items for the CPU pick path. Disabled by default: this copies
        // all inline point/glyph/curve geometry each frame, so scenes that do not use
        // renderer.pick()/pick_rect() leave it off (see set_cpu_pick_cache).
        if self.cpu_pick_cache_enabled {
            self.cache_pick_items(frame);
        }

        let (scene_fx, viewport_fx) = frame.effects.split();
        self.prepare_scene_internal(device, queue, frame, &scene_fx, sink);

        let viewport_start = web_time::Instant::now();
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx, sink);
        // Overlays are timed inside the viewport phase and reported on their own,
        // so take them back out of `viewport_ms`: the two fields are siblings, not
        // a field and its subset, and every field here sums into `cpu_prepare_ms`.
        let viewport_total = viewport_start.elapsed().as_secs_f32() * 1000.0;
        self.prepare_breakdown.viewport_ms =
            (viewport_total - self.prepare_breakdown.overlay_ms).max(0.0);

        let cpu_prepare_ms = prepare_start.elapsed().as_secs_f32() * 1000.0;
        // Remainder: timestamp readback, scatter sort, degradation logic, stats
        // assembly, and anything else not bracketed by a phase timer above.
        let b = &mut self.prepare_breakdown;
        b.other_ms = (cpu_prepare_ms
            - b.plugin_ms
            - b.lighting_ms
            - b.uniforms_ms
            - b.instancing_ms
            - b.geometry_ms
            - b.shadow_ms
            - b.viewport_ms
            - b.overlay_ms)
            .max(0.0);

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
        // Only a presented frame advances the counter. A derivative capture
        // render must not perturb the presented frame's temporal phase (the
        // counter drives scatter jitter and the pick cadence).
        if self.render_advances_state() {
            self.frame_counter = self.frame_counter.wrapping_add(1);
        }

        let reported_render_scale = self.current_render_scale;

        let stats = crate::renderer::stats::FrameStats {
            cpu_prepare_ms,
            prepare_breakdown: self.prepare_breakdown,
            // gpu_frame_ms is updated by the timestamp readback above when available;
            // propagate the most recent value from last_stats.
            gpu_frame_ms: self.last_stats.gpu_frame_ms,
            total_frame_ms,
            render_scale: reported_render_scale,
            missed_budget,
            upload_bytes,
            pipelines_built_this_frame,
            pipelines_pending: self.resources.pipeline_compiler.pending() as u32,
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

/// Decide whether this frame runs the GPU cull. It needs the cull compute
/// pipelines; the first frame that wants them with instances to cull asks
/// for them under the compilation policy, and under `Background` the
/// frames until they are built take the CPU path, which draws the same
/// image.
fn settle_gpu_culling(
    inst: &mut InstancingState,
    compiler: &crate::resources::pipeline_slot::PipelineCompiler,
    device: &crate::gpu::Device,
) {
    let has_work =
        inst.use_instancing && !inst.batches.is_empty() && inst.cached_instance_count > 0;
    inst.gpu_culling_enabled = inst.gpu_culling_wanted
        && (!has_work || {
            let dev = device.clone();
            inst.cull_resources
                .get(compiler, move || {
                    crate::renderer::indirect::CullResources::new(&dev)
                })
                .is_some()
        });
}

#[cfg(test)]
mod lod_resolve_tests {
    use super::ViewportRenderer;
    use crate::geometry::primitives;
    use crate::renderer::{RenderCamera, SceneRenderItem};
    use crate::resources::{DeviceResources, LodGroupId, MeshData};
    use std::collections::HashMap;

    /// Upload each level mesh, then register the group.
    fn register(
        res: &mut DeviceResources,
        device: &crate::gpu::Device,
        levels: &[(MeshData, f32)],
    ) -> crate::error::ViewportResult<LodGroupId> {
        let mut ids = Vec::with_capacity(levels.len());
        let mut sizes = Vec::with_capacity(levels.len());
        for (data, size) in levels {
            ids.push(res.upload_mesh_data(device, data)?);
            sizes.push(*size);
        }
        res.register_lod_group(&ids, &sizes)
    }

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn looking_down_z() -> RenderCamera {
        let mut camera = RenderCamera::default();
        camera.eye_position = [0.0, 0.0, 0.0];
        camera.forward = [0.0, 0.0, -1.0];
        camera
    }

    fn item_at(z: f32, group: crate::resources::LodGroupId, pick: u64) -> SceneRenderItem {
        let mut item = SceneRenderItem::default();
        item.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, z)).to_cols_array_2d();
        item.lod_group = Some(group);
        item.settings.pick_id = crate::renderer::PickId(pick);
        item
    }

    #[test]
    fn resolve_swaps_mesh_by_distance() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let group = register(
            &mut res,
            &device,
            &[
                (primitives::icosphere(1.0, 3), 0.5),
                (primitives::icosphere(1.0, 1), 0.2),
                (primitives::icosphere(1.0, 0), 0.0),
            ],
        )
        .unwrap();
        let full = res.lod_group(group).unwrap().mesh_at(0);
        let crude = res.lod_group(group).unwrap().mesh_at(2);

        let camera = looking_down_z();
        let mut levels = HashMap::new();
        let mut items = vec![item_at(-3.0, group, 1), item_at(-300.0, group, 2)];

        let (resolved, _switches, _culled, reduced) =
            ViewportRenderer::resolve_lod(&res, &mut levels, &camera, &mut items);

        assert_eq!(resolved, 2);
        assert_eq!(reduced, 1, "the far object draws at a reduced level");
        assert_eq!(items[0].mesh_id, full, "near object uses full detail");
        assert_eq!(items[1].mesh_id, crude, "far object uses crudest level");
    }

    #[test]
    fn items_without_a_group_are_untouched() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let group = register(
            &mut res,
            &device,
            &[(primitives::cube(1.0), 0.5), (primitives::cube(1.0), 0.0)],
        )
        .unwrap();

        let plain_mesh = res
            .upload_mesh_data(&device, &primitives::cube(2.0))
            .unwrap();
        let mut plain = SceneRenderItem::default();
        plain.mesh_id = plain_mesh;

        let camera = looking_down_z();
        let mut levels = HashMap::new();
        let mut items = vec![plain, item_at(-3.0, group, 1)];

        let (resolved, _, _, _) =
            ViewportRenderer::resolve_lod(&res, &mut levels, &camera, &mut items);

        assert_eq!(resolved, 1, "only the LOD item is counted");
        assert_eq!(items[0].mesh_id, plain_mesh, "plain item is untouched");
    }

    #[test]
    fn switches_count_only_level_changes() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let group = register(
            &mut res,
            &device,
            &[
                (primitives::icosphere(1.0, 3), 0.5),
                (primitives::icosphere(1.0, 1), 0.2),
                (primitives::icosphere(1.0, 0), 0.0),
            ],
        )
        .unwrap();

        let camera = looking_down_z();
        let mut levels = HashMap::new();
        let mut items = vec![item_at(-3.0, group, 1)];

        // First frame: the object appears, landing on its level. That is a change
        // from the assumed starting level 0, but here it is already level 0.
        let (_, switches_first, _, _) =
            ViewportRenderer::resolve_lod(&res, &mut levels, &camera, &mut items);
        assert_eq!(switches_first, 0, "near object starts and stays at level 0");

        // Second frame at the same distance: no change.
        let (_, switches_second, _, _) =
            ViewportRenderer::resolve_lod(&res, &mut levels, &camera, &mut items);
        assert_eq!(switches_second, 0);
    }

    #[test]
    fn stale_pick_ids_are_pruned() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let group = register(
            &mut res,
            &device,
            &[(primitives::cube(1.0), 0.5), (primitives::cube(1.0), 0.0)],
        )
        .unwrap();

        let camera = looking_down_z();
        let mut levels = HashMap::new();

        let mut frame_one = vec![item_at(-3.0, group, 1), item_at(-3.0, group, 2)];
        ViewportRenderer::resolve_lod(&res, &mut levels, &camera, &mut frame_one);
        assert_eq!(levels.len(), 2);

        let mut frame_two = vec![item_at(-3.0, group, 1)];
        ViewportRenderer::resolve_lod(&res, &mut levels, &camera, &mut frame_two);
        assert_eq!(levels.len(), 1, "pick id 2 dropped out and was pruned");
        assert!(levels.contains_key(&1));
    }

    #[test]
    fn cull_below_hides_tiny_items() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let group = register(
            &mut res,
            &device,
            &[
                (primitives::icosphere(1.0, 3), 0.5),
                (primitives::icosphere(1.0, 0), 0.0),
            ],
        )
        .unwrap();
        res.set_lod_cull_below(group, Some(0.05)).unwrap();

        let camera = looking_down_z();
        let mut levels = HashMap::new();
        // Near item stays; far item drops below the cull size.
        let mut items = vec![item_at(-3.0, group, 1), item_at(-400.0, group, 2)];

        let (resolved, _, culled, _) =
            ViewportRenderer::resolve_lod(&res, &mut levels, &camera, &mut items);

        assert_eq!(resolved, 1, "only the near item draws");
        assert_eq!(culled, 1, "the far item is culled");
        assert!(!items[0].settings.hidden, "near item visible");
        assert!(items[1].settings.hidden, "far item hidden");
    }
}
