use super::types::{ClipShape, SceneEffects, ViewportEffects};
use super::*;
use crate::resources::CurveMeshOutlineItem;
use wgpu::util::DeviceExt;

mod instanced;
mod lighting;
mod math;
mod mesh_material;
mod overlay_geometry;
mod per_object;
mod projection;
mod scene_uploads;
mod shadow_pass;
mod viewport_interaction;
mod viewport_misc;
mod viewport_overlays;
mod wireframe;

use math::*;
pub(crate) use mesh_material::backface_needs_per_object;
use mesh_material::*;
use overlay_geometry::*;
use projection::*;
use wireframe::*;

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
    /// Scene-global prepare stage: compute filters, lighting, shadow pass, batching, scivis.
    ///
    /// Call once per frame before any `prepare_viewport_internal` calls.
    ///
    /// Reads `scene_fx` for lighting, IBL, and compute filters.  Also reads
    /// `frame.camera` for shadow cascade computation.
    pub(super) fn prepare_scene_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        scene_fx: &SceneEffects<'_>,
    ) {
        // Drain the upload-job runner. Worker results received since the last
        // frame are observed, GPU submissions are polled for completion, and
        // any registered completion callbacks fire on this thread.
        match self.upload_budget {
            Some(d) => self.resources.process_uploads_with_budget(
                device,
                queue,
                crate::resources::FrameBudget::from_now(d),
            ),
            None => self.resources.process_uploads(device, queue),
        }

        // GPU compute filtering.
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

        let plugin_frame_index = self.plugin_frame_index;

        let resources = &mut self.resources;
        let lighting = scene_fx.lighting;

        // Read scene items from the surface submission, then extend with the
        // boundary draws contributed by opaque volume meshes (items in
        // `volume_meshes` whose `transparency` is `None`). The owned vector
        // keeps these extra items alive for the whole prepare pass.
        let scene_items_owned: Vec<SceneRenderItem> = {
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
        let scene_items: &[SceneRenderItem] = &scene_items_owned;

        let lighting_start = std::time::Instant::now();
        let lighting_frame = Self::prepare_lighting(
            resources,
            &mut self.shadow,
            &mut self.last_cluster_stats,
            &mut self.last_frustum_culled_lights,
            &self.viewport_slots,
            scene_fx,
            device,
            queue,
            frame,
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

        let per_object_start = std::time::Instant::now();
        // Evaluate instanceability once per frame and share the result. Each
        // `is_instanceable` call does several mesh-store and deform lookups plus
        // a linear scan over the compute-filter results, so computing it once
        // here instead of separately in the per-object skip test and the
        // instanced batch filter keeps this O(items) rather than running the
        // same per-item work three times over. At city scale (tens of thousands
        // of resident meshes) that is the difference between a few milliseconds
        // and a few hundred.
        let instanceable: Vec<bool> = scene_items
            .iter()
            .map(|item| is_instanceable(item, resources, &self.compute_filter_results))
            .collect();
        let per_object_bind_groups_built = Self::prepare_per_object(
            resources,
            &mut self.mesh_uniforms,
            self.instancing.use_instancing,
            scene_items,
            &instanceable,
            self.frame_counter,
            device,
            queue,
            frame,
        );
        let uniforms_ms = per_object_start.elapsed().as_secs_f32() * 1000.0;

        let instanced_start = std::time::Instant::now();
        let (batches_reuploaded, batches_skipped) = if self.instancing.use_instancing {
            Self::prepare_instanced(
                resources,
                &mut self.instancing,
                &instanceable,
                scene_items,
                device,
                queue,
                frame,
            )
        } else {
            (0, 0)
        };
        let instancing_ms = instanced_start.elapsed().as_secs_f32() * 1000.0;

        let geometry_start = std::time::Instant::now();
        Self::upload_geometry_glyphs(
            resources,
            &mut self.point_cloud_gpu_data,
            &mut self.glyph_gpu_data,
            &mut self.sprite_gpu_data,
            &mut self.mesh_instance_gpu_data,
            &mut self.particle_gpu_data,
            &mut self.tensor_glyph_gpu_data,
            device,
            queue,
            frame,
        );
        Self::upload_polylines(
            resources,
            &mut self.polyline_gpu_data,
            &mut self.polyline_selected_gpu_indices,
            &mut self.glyph_gpu_data,
            device,
            queue,
            frame,
        );
        Self::upload_implicit_decals_mc(
            resources,
            &mut self.implicit_gpu_data,
            &mut self.pick_implicit_items,
            &mut self.decal_gpu_data,
            &mut self.decal_exclude_items,
            &mut self.mc_gpu_data,
            &mut self.pick_mc_items,
            device,
            queue,
            frame,
        );
        Self::upload_images(
            resources,
            &mut self.screen_image_gpu_data,
            &mut self.overlay_image_gpu_data,
            device,
            queue,
            frame,
        );
        Self::upload_tubes_ribbons(
            resources,
            &mut self.streamtube_gpu_data,
            &mut self.streamtube_selected_gpu_indices,
            &mut self.tube_gpu_data,
            &mut self.tube_selected_gpu_indices,
            &mut self.ribbon_gpu_data,
            &mut self.ribbon_selected_gpu_indices,
            device,
            queue,
            frame,
        );
        Self::upload_slices(
            resources,
            &mut self.image_slice_gpu_data,
            &mut self.volume_surface_slice_gpu_data,
            device,
            queue,
            frame,
        );
        let vp_size = frame.camera.viewport_size;
        // Surface LIC GPU data upload.
        // ------------------------------------------------------------------
        self.lic_gpu_data.clear();
        {
            let lic_scene_items: Vec<(&SceneRenderItem, &LicOverlay)> = scene_items
                .iter()
                .filter(|i| !i.settings.hidden)
                .filter_map(|i| i.lic.as_ref().map(|l| (i, l)))
                .collect();
            if !lic_scene_items.is_empty() {
                // The LIC surface pipeline is created inside ensure_hdr_shared (already called
                // before prepare_scene_internal runs), so no separate ensure call is needed here.
                for (item, lic) in &lic_scene_items {
                    if lic.vector_attribute.is_empty() {
                        continue;
                    }
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                        // Verify the vector attribute buffer exists before committing to this item.
                        if mesh
                            .vector_attribute_buffers
                            .contains_key(&lic.vector_attribute)
                        {
                            if let Some(bgl) = &resources.lic_surface_bgl {
                                use crate::resources::LicObjectUniform;
                                let model = item.model;
                                let obj_data = LicObjectUniform { model };
                                let obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some("lic_object_uniform"),
                                    size: std::mem::size_of::<LicObjectUniform>() as u64,
                                    usage: wgpu::BufferUsages::UNIFORM
                                        | wgpu::BufferUsages::COPY_DST,
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
                                    vector_attribute: lic.vector_attribute.clone(),
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
                    if let Some((_, first_lic)) = lic_scene_items.first() {
                        let [vw, vh] = hdr.scene_size;
                        let u = crate::resources::LicAdvectUniform {
                            steps: first_lic.config.steps,
                            step_size: first_lic.config.step_size,
                            vp_width: vw as f32,
                            vp_height: vh as f32,
                        };
                        queue.write_buffer(&hdr.lic_uniform_buf, 0, bytemuck::cast_slice(&[u]));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Volume GPU data upload.
        // Note: clip_planes are per-viewport but passed here for culling.
        // ------------------------------------------------------------------
        self.volume_gpu_data.clear();
        if !frame.scene.volumes.is_empty() {
            resources.ensure_volume_pipeline(device);
            let clip_objects_for_vol = &frame.effects.clip_objects;
            // Under budget pressure with allow_volume_quality_reduction, double the
            // step size (half the sample count) to reduce GPU raymarch cost.
            let vol_step_multiplier = if self.degradation_volume_quality_reduced {
                2.0_f32
            } else {
                1.0_f32
            };
            for item in &frame.scene.volumes {
                if item.settings.hidden {
                    continue;
                }
                let mut gpu = resources.upload_volume_frame(
                    device,
                    queue,
                    item,
                    clip_objects_for_vol,
                    vol_step_multiplier,
                );
                gpu.wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                self.volume_gpu_data.push(gpu);
            }
        }

        // Volume wireframe overlay: OBB from bbox + model matrix.
        let need_vol_wf = frame.viewport.wireframe_mode
            || frame
                .scene
                .volumes
                .iter()
                .any(|v| !v.settings.hidden && v.settings.wireframe);
        if need_vol_wf {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.scene.volumes {
                if item.settings.hidden {
                    continue;
                }
                if !(frame.viewport.wireframe_mode || item.settings.wireframe) {
                    continue;
                }
                let polyline = volume_obb_polyline(item);
                let gpu = resources.upload_polyline_per_frame(device, queue, &polyline, vp_size);
                self.polyline_gpu_data.push(gpu);
            }
        }

        // Transparent volume meshes wireframe: boundary mesh edge overlay.
        // Items rendering as opaque already participate in the standard
        // wireframe pass via the surface submission; here we only need to
        // gather boundary edges for items rendering through the projected-tet
        // path so they still get a wireframe overlay.
        self.mesh_uniforms.tvm_wireframe_draws.clear();
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
            self.mesh_uniforms
                .tvm_wireframe_draws
                .push(item.boundary_mesh_id);
        }
        if !self.mesh_uniforms.tvm_wireframe_draws.is_empty()
            && self.mesh_uniforms.tvm_wireframe_bg.is_none()
        {
            use wgpu::util::DeviceExt;
            let mut tvm_wf_uniform: crate::resources::ObjectUniform = bytemuck::Zeroable::zeroed();
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
                        resource: wgpu::BindingResource::TextureView(&resources.fallback_lut_view),
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
            self.mesh_uniforms.tvm_wireframe_buf = Some(buf);
            self.mesh_uniforms.tvm_wireframe_bg = Some(bg);
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
            } else {
                for item in scene_items {
                    if item.settings.hidden {
                        continue;
                    }
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                        draw_calls += 1;
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
                triangles_submitted: triangles,
                shadow_draw_calls: 0, // Updated below in shadow pass.
                gpu_culling_active: self.instancing.gpu_culling_enabled,
                // Clear stale readback if GPU culling is off this frame.
                gpu_visible_instances: if self.instancing.gpu_culling_enabled {
                    self.last_stats.gpu_visible_instances
                } else {
                    None
                },
                ..self.last_stats
            };
        }

        // ------------------------------------------------------------------
        let shadow_start = std::time::Instant::now();
        Self::prepare_shadow_pass(
            resources,
            &mut self.instancing,
            &self.compute_filter_results,
            &self.item_type_plugins,
            plugin_frame_index,
            lighting,
            scene_items,
            &lighting_frame,
            self.degradation_shadows_skipped,
            &mut self.last_stats,
            device,
            queue,
            frame,
        );
        let shadow_ms = shadow_start.elapsed().as_secs_f32() * 1000.0;

        // The `resources` borrow ends with the shadow pass above, so record the
        // scene-phase timings into the breakdown now.
        self.prepare_breakdown.lighting_ms = lighting_ms;
        self.prepare_breakdown.uniforms_ms = uniforms_ms;
        self.prepare_breakdown.instancing_ms = instancing_ms;
        self.prepare_breakdown.geometry_ms = geometry_ms;
        self.prepare_breakdown.shadow_ms = shadow_ms;
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
        self.ensure_viewport_slot(device, frame.camera.viewport_index);
        self.prepare_clip_uniforms(queue, frame, viewport_fx);
        self.prepare_interaction_state(device, queue, frame, viewport_fx);
        self.prepare_outline_pass(device, queue, frame);
        self.prepare_sub_highlight(device, queue, frame);

        self.prepare_overlay_labels(device, queue, frame);
        self.prepare_scalar_bars(device, queue, frame);
        self.prepare_rulers(device, queue, frame);
        self.prepare_loading_bars(device, queue, frame);
        self.prepare_overlay_shapes(device, frame);
        self.prepare_splat_sort(device, queue, frame);
        self.prepare_splat_wireframe(device, queue, frame);
        self.prepare_sprite_wireframe(device, queue, frame);
        self.prepare_debug_buffer(device, frame);
        self.prepare_atlas_blit(queue, frame, viewport_fx);
    }

    /// Upload per-frame data to GPU buffers and render the shadow pass.
    /// Call before `paint()`.
    ///
    /// Returns [`crate::FrameStats`] with per-frame timing and upload metrics.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) -> crate::renderer::stats::FrameStats {
        let prepare_start = std::time::Instant::now();
        self.prepare_breakdown = crate::renderer::stats::PrepareBreakdown::default();

        let plugin_start = std::time::Instant::now();

        // Dispatch item-type plugin prepare work first so any GPU outputs
        // the plugin produces are visible to the rest of `prepare`.
        let plugin_bufs = self.dispatch_plugin_prepare(device, queue, frame);
        if !plugin_bufs.is_empty() {
            queue.submit(plugin_bufs);
        }

        // Run plugin culling for the current camera frustum so subsequent
        // plugin paint/shadow calls can skip culled items.
        if !self.item_type_plugins.is_empty() && !frame.scene.plugin_items.is_empty() {
            let vp = frame.camera.render_camera.view_proj();
            let frustum = crate::camera::frustum::Frustum::from_view_proj(&vp);
            self.dispatch_plugin_cull(&frustum, frame);
        }
        self.prepare_breakdown.plugin_ms = plugin_start.elapsed().as_secs_f32() * 1000.0;

        // Read back GPU timestamps without blocking. The staging buffer is
        // mapped on one frame and read on a later one; a non-blocking poll
        // pumps the map callback, and the value is consumed once the map
        // completes. A blocking wait here would stall the CPU on the previous
        // frame's GPU work, which on this workload is most of the frame.
        use std::sync::atomic::Ordering;
        if self.ts_map_inflight {
            let _ = device.poll(wgpu::PollType::Poll);
            match self.ts_map_status.load(Ordering::Acquire) {
                1 => {
                    if let Some(ref stg_buf) = self.ts_staging_buf {
                        let data = stg_buf.slice(..).get_mapped_range();
                        let t0 = u64::from_le_bytes(data[0..8].try_into().unwrap());
                        let t1 = u64::from_le_bytes(data[8..16].try_into().unwrap());
                        drop(data);
                        // ts_period is nanoseconds/tick; convert delta to milliseconds.
                        let gpu_ms = t1.saturating_sub(t0) as f32 * self.ts_period / 1_000_000.0;
                        // Metal can report equal begin/end timestamps at pass
                        // boundaries; treat a zero delta as no sample rather than
                        // a real 0 ms frame.
                        if gpu_ms > 0.0 {
                            self.last_stats.gpu_frame_ms = Some(gpu_ms);
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
                stg_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    status.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release);
                });
                // Pump once so the map can complete immediately when the GPU has
                // already finished; otherwise it completes on a later frame.
                let _ = device.poll(wgpu::PollType::Poll);
                self.ts_map_inflight = true;
                self.ts_data_ready = false;
            }
        }

        // Read back the GPU-visible instance count without blocking, using the
        // same map-on-one-frame, read-on-a-later-frame scheme as the timestamps.
        let bytes = self.instancing.indirect_readback_batch_count as u64 * 20;
        if self.instancing.indirect_map_inflight {
            let _ = device.poll(wgpu::PollType::Poll);
            match self.instancing.indirect_map_status.load(Ordering::Acquire) {
                1 => {
                    if let Some(ref stg_buf) = self.instancing.indirect_readback_buf {
                        let data = stg_buf.slice(..bytes).get_mapped_range();
                        let mut visible: u32 = 0;
                        for i in 0..self.instancing.indirect_readback_batch_count as usize {
                            // DrawIndexedIndirect layout: [index_count, instance_count, first_index, base_vertex, first_instance]
                            // instance_count is at byte offset 4 within each 20-byte entry.
                            let off = i * 20 + 4;
                            let n = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            visible = visible.saturating_add(n);
                        }
                        drop(data);
                        self.last_stats.gpu_visible_instances = Some(visible);
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
                        .map_async(wgpu::MapMode::Read, move |r| {
                            status.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release);
                        });
                    let _ = device.poll(wgpu::PollType::Poll);
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

        // Prepare scatter volumes for rendering. Independent of picking: always runs.
        {
            self.prepared_scatter_volumes.clear();
            self.prepared_refraction_volumes.clear();
            let global_wireframe = frame.viewport.wireframe_mode;
            let eye = frame.camera.render_camera.eye_position;
            for item in &frame.scene.scatter_volumes {
                if item.settings.hidden || item.settings.wireframe || global_wireframe {
                    continue;
                }
                let mut flags: u32 = 0;
                if item.settings.unlit {
                    flags |= crate::scene::scatter_volume::SCATTER_FLAG_UNLIT;
                }
                if item.settings.receive_shadows {
                    flags |= crate::scene::scatter_volume::SCATTER_FLAG_RECEIVE_SHADOWS;
                }
                self.prepared_scatter_volumes.push((
                    item.volume.clone(),
                    item.settings.opacity,
                    flags,
                ));
                if item.volume.refraction.is_some() {
                    self.prepared_refraction_volumes
                        .push((item.volume.clone(), item.settings.opacity));
                }
            }
            // Sort back-to-front for the per-volume scatter draws. The
            // metric is the maximum corner distance of the volume's world
            // AABB from the eye, descending. Centroid distance flips order
            // when one volume contains another (huge fog containing a small
            // fire) -- the fire centroid can land on either side of the fog
            // centroid as the camera orbits, causing the alpha-over composite
            // to swap visibly. Sorting by far-corner distance keeps
            // containers (whose far corner is much further from the eye)
            // strictly behind contained volumes regardless of camera angle.
            self.prepared_scatter_volumes.sort_by(|a, b| {
                let aabb_a = a.0.world_aabb();
                let aabb_b = b.0.world_aabb();
                let far_corner = |aabb: &crate::Aabb| -> f32 {
                    let cx = if (aabb.min.x - eye[0]).abs() > (aabb.max.x - eye[0]).abs() {
                        aabb.min.x
                    } else {
                        aabb.max.x
                    };
                    let cy = if (aabb.min.y - eye[1]).abs() > (aabb.max.y - eye[1]).abs() {
                        aabb.min.y
                    } else {
                        aabb.max.y
                    };
                    let cz = if (aabb.min.z - eye[2]).abs() > (aabb.max.z - eye[2]).abs() {
                        aabb.min.z
                    } else {
                        aabb.max.z
                    };
                    (cx - eye[0]).powi(2) + (cy - eye[1]).powi(2) + (cz - eye[2]).powi(2)
                };
                let da = far_corner(&aabb_a);
                let db = far_corner(&aabb_b);
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let (scene_fx, viewport_fx) = frame.effects.split();
        self.prepare_scene_internal(device, queue, frame, &scene_fx);

        let viewport_start = std::time::Instant::now();
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx);
        self.prepare_breakdown.viewport_ms = viewport_start.elapsed().as_secs_f32() * 1000.0;

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
            - b.viewport_ms)
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
        self.frame_counter = self.frame_counter.wrapping_add(1);

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
