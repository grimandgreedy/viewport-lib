//! Scivis per-frame GPU upload passes called from `prepare_scene_internal`.
//!
//! Associated functions (no `self`): the caller holds a long-lived
//! `&mut self.resources` borrow across the whole scene prepare, so each takes
//! `resources` plus the disjoint `self` fields it fills.

use super::*;

/// Decal resource cache tallies for one frame: `uploads` counts cache misses
/// (a buffer + bind group were built), `reused` counts cache hits.
#[derive(Default, Clone, Copy)]
pub(super) struct DecalCacheStats {
    pub uploads: u32,
    pub reused: u32,
}

impl ViewportRenderer {
    /// Upload mesh-instance batches, resolving any LOD group per instance.
    ///
    /// Items without a `lod_group` upload exactly as before: one batch, one
    /// draw. Items with one have their instances grouped by level, and each
    /// occupied level uploads as its own batch drawn with that level's mesh.
    /// Instances below the group's cull size are dropped from every batch.
    /// Returns `(instances_resolved, switches, culled, reduced)` for the LOD
    /// stats, where `reduced` counts instances drawn below full detail.
    pub(super) fn upload_mesh_instances(
        resources: &mut DeviceResources,
        mesh_instance_gpu_data: &mut Vec<crate::resources::MeshInstanceGpuData>,
        lod_levels: &mut std::collections::HashMap<(u64, u32), usize>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> (u32, u32, u32, u32) {
        mesh_instance_gpu_data.clear();
        if frame.scene.mesh_instances.is_empty() {
            if !lod_levels.is_empty() {
                lod_levels.clear();
            }
            return (0, 0, 0, 0);
        }
        resources.ensure_instanced_pipelines(device);
        resources.ensure_hdr_instanced_pipelines(device);

        let mut resolved = 0u32;
        let mut switches = 0u32;
        let mut culled = 0u32;
        let mut reduced = 0u32;
        let mut seen: Vec<(u64, u32)> = Vec::new();

        for item in &frame.scene.mesh_instances {
            if item.settings.hidden || item.transforms.is_empty() {
                continue;
            }

            // The full-detail AABB sizes every instance; fetching it also tells
            // us whether the group is usable. No group, missing group, or a
            // removed full-detail mesh all fall back to the plain single-batch
            // draw, exactly the old path.
            let base_aabb = match item.lod_group {
                Some(id) => resources
                    .lod_group(id)
                    .and_then(|g| resources.mesh(g.mesh_at(0)))
                    .map(|m| m.aabb),
                None => None,
            };
            let Some(aabb) = base_aabb else {
                if let Some(gd) = resources.upload_mesh_instance(device, queue, item) {
                    mesh_instance_gpu_data.push(gd);
                }
                continue;
            };

            // Bucket instance indices by level. Everything that needs the
            // group's immutable borrow happens in this block, which then yields
            // owned data so the uploads below can borrow `resources` mutably.
            let group_id = item.lod_group.unwrap();
            let pick = item.settings.pick_id.0;
            let (level_meshes, buckets) = {
                let group = resources.lod_group(group_id).unwrap();
                let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); group.level_count()];
                for (idx, transform) in item.transforms.iter().enumerate() {
                    let model = glam::Mat4::from_cols_array_2d(transform);
                    let size = crate::resources::projected_screen_size(
                        &aabb,
                        &model,
                        &frame.camera.render_camera,
                    );
                    if group.should_cull(size) {
                        culled += 1;
                        continue;
                    }
                    let level = if pick == 0 {
                        group.level_for_size(size)
                    } else {
                        let key = (pick, idx as u32);
                        let current = lod_levels.get(&key).copied().unwrap_or(0);
                        let next = group.select(size, current);
                        if next != current {
                            switches += 1;
                        }
                        lod_levels.insert(key, next);
                        seen.push(key);
                        next
                    };
                    buckets[level].push(idx as u32);
                    resolved += 1;
                    if level > 0 {
                        reduced += 1;
                    }
                }
                let level_meshes: Vec<crate::resources::mesh::mesh_store::MeshId> =
                    group.levels().iter().map(|l| l.mesh).collect();
                (level_meshes, buckets)
            };

            for (level, idxs) in buckets.iter().enumerate() {
                if idxs.is_empty() {
                    continue;
                }
                if let Some(gd) = resources.upload_mesh_instance_from(
                    device,
                    queue,
                    item,
                    level_meshes[level],
                    Some(idxs),
                ) {
                    mesh_instance_gpu_data.push(gd);
                }
            }
        }

        // Drop tracking state for instances no longer present so the map does
        // not grow without bound.
        if seen.is_empty() {
            if !lod_levels.is_empty() {
                lod_levels.clear();
            }
        } else if lod_levels.len() > seen.len() {
            let keep: std::collections::HashSet<(u64, u32)> = seen.into_iter().collect();
            lod_levels.retain(|k, _| keep.contains(k));
        }

        (resolved, switches, culled, reduced)
    }

    pub(super) fn upload_geometry_glyphs(
        resources: &mut DeviceResources,
        point_cloud_gpu_data: &mut Vec<crate::resources::PointCloudGpuData>,
        glyph_gpu_data: &mut Vec<crate::resources::GlyphGpuData>,
        sprite_gpu_data: &mut Vec<crate::resources::SpriteGpuData>,
        particle_gpu_data: &mut Vec<crate::resources::gpu::gpu_particles::ParticleFrameData>,
        tensor_glyph_gpu_data: &mut Vec<crate::resources::TensorGlyphGpuData>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        // ------------------------------------------------------------------
        // point cloud and glyph GPU data upload.
        // ------------------------------------------------------------------
        point_cloud_gpu_data.clear();
        if !frame.scene.point_clouds.is_empty() {
            resources.ensure_point_cloud_pipeline(device);
            for item in &frame.scene.point_clouds {
                if item.settings.hidden || item.positions.is_empty() {
                    continue;
                }
                let gpu_data = resources.upload_point_cloud_per_frame(device, queue, item);
                point_cloud_gpu_data.push(gpu_data);
            }
        }

        // Pre-uploaded point cloud references. Model matrix lives at offset 0
        // of PointCloudUniform.
        if !frame.scene.point_cloud_refs.is_empty() {
            resources.ensure_point_cloud_pipeline(device);
            for ref_item in &frame.scene.point_cloud_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources.content.point_cloud_store.get(ref_item.source) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                queue.write_buffer(&entry._uniform_buf, 0, bytemuck::bytes_of(&ref_item.model));
                point_cloud_gpu_data.push(entry);
            }
        }

        glyph_gpu_data.clear();
        if !frame.scene.glyphs.is_empty() {
            resources.ensure_glyph_pipeline(device);
            for item in &frame.scene.glyphs {
                if item.settings.hidden || item.positions.is_empty() || item.vectors.is_empty() {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                let gpu_data = resources.upload_glyph_set_per_frame(device, queue, item, wireframe);
                glyph_gpu_data.push(gpu_data);
            }
        }

        // Pre-uploaded glyph set references. Model matrix lives at offset 0
        // of GlyphUniform and is composed on top of per-instance positions
        // in the vertex shader.
        if !frame.scene.glyph_set_refs.is_empty() {
            resources.ensure_glyph_pipeline(device);
            for ref_item in &frame.scene.glyph_set_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources.content.glyph_set_store.get(ref_item.source) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                queue.write_buffer(&entry._uniform_buf, 0, bytemuck::bytes_of(&ref_item.model));
                let mut gpu_data = entry;
                gpu_data.wireframe = frame.viewport.wireframe_mode || ref_item.settings.wireframe;
                glyph_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // Sprite billboard GPU data upload.
        // ------------------------------------------------------------------
        sprite_gpu_data.clear();
        if !frame.scene.sprite_items.is_empty() {
            resources.ensure_sprite_pipelines(device);
            for item in &frame.scene.sprite_items {
                if item.settings.hidden || item.positions.is_empty() {
                    continue;
                }
                let mut gd = resources.upload_sprite(device, queue, item);
                gd.wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                sprite_gpu_data.push(gd);
            }
        }

        // Pre-uploaded sprite set references.
        if !frame.scene.sprite_set_refs.is_empty() {
            resources.ensure_sprite_pipelines(device);
            for ref_item in &frame.scene.sprite_set_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources.content.sprite_set_store.get(ref_item.source) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                let mut gd = entry;
                gd.wireframe = frame.viewport.wireframe_mode || ref_item.settings.wireframe;
                sprite_gpu_data.push(gd);
            }
        }

        // Pre-uploaded sprite instance set references.
        if !frame.scene.sprite_instance_set_refs.is_empty() {
            resources.ensure_sprite_pipelines(device);
            for ref_item in &frame.scene.sprite_instance_set_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources
                    .content
                    .sprite_instance_set_store
                    .get(ref_item.source)
                {
                    Some(e) => e.clone(),
                    None => continue,
                };
                let mut gd = entry;
                gd.wireframe = frame.viewport.wireframe_mode || ref_item.settings.wireframe;
                sprite_gpu_data.push(gd);
            }
        }

        // Mesh-instance batches are uploaded by `upload_mesh_instances`, called
        // separately so it can resolve LOD groups per instance.

        // ------------------------------------------------------------------
        // GPU particle systems: dispatch emit + sim compute passes.
        // ------------------------------------------------------------------
        particle_gpu_data.clear();
        if !frame.scene.gpu_particle_systems.is_empty() {
            *particle_gpu_data =
                resources.run_particle_jobs(device, queue, &frame.scene.gpu_particle_systems, sink);
        }

        // ------------------------------------------------------------------
        // Tensor glyph GPU data upload.
        // ------------------------------------------------------------------
        tensor_glyph_gpu_data.clear();
        if !frame.scene.tensor_glyphs.is_empty() {
            resources.ensure_tensor_glyph_pipeline(device);
            for item in &frame.scene.tensor_glyphs {
                if item.settings.hidden || item.positions.is_empty() {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                let gd =
                    resources.upload_tensor_glyph_set_per_frame(device, queue, item, wireframe);
                tensor_glyph_gpu_data.push(gd);
            }
        }

        // Pre-uploaded tensor glyph set references. Model matrix lives at
        // offset 0 of TensorGlyphUniform and is composed on top of the
        // per-instance ellipsoid model in the vertex shader.
        if !frame.scene.tensor_glyph_set_refs.is_empty() {
            resources.ensure_tensor_glyph_pipeline(device);
            for ref_item in &frame.scene.tensor_glyph_set_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources
                    .content
                    .tensor_glyph_set_store
                    .get(ref_item.source)
                {
                    Some(e) => e.clone(),
                    None => continue,
                };
                queue.write_buffer(&entry._uniform_buf, 0, bytemuck::bytes_of(&ref_item.model));
                let mut gpu_data = entry;
                gpu_data.wireframe = frame.viewport.wireframe_mode || ref_item.settings.wireframe;
                tensor_glyph_gpu_data.push(gpu_data);
            }
        }
    }

    pub(super) fn upload_polylines(
        resources: &mut DeviceResources,
        polyline_gpu_data: &mut Vec<crate::resources::PolylineGpuData>,
        polyline_selected_gpu_indices: &mut Vec<usize>,
        glyph_gpu_data: &mut Vec<crate::resources::GlyphGpuData>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // ------------------------------------------------------------------
        // polyline GPU data upload.
        // ------------------------------------------------------------------
        polyline_gpu_data.clear();
        polyline_selected_gpu_indices.clear();
        let vp_size = frame.camera.viewport_size;
        if !frame.scene.polylines.is_empty() {
            resources.ensure_polyline_pipeline(device);
            // Clip-exempt polylines (ItemSettings::ignore_clip, e.g. clip-object
            // outlines) draw through the no-clip pipeline; create it on demand.
            if frame.scene.polylines.iter().any(|p| p.settings.ignore_clip) {
                resources.ensure_polyline_no_clip_pipeline(device);
            }
            for item in &frame.scene.polylines {
                if item.settings.hidden || item.positions.is_empty() {
                    continue;
                }
                let mut gpu_data =
                    resources.upload_polyline_per_frame(device, queue, item, vp_size);
                gpu_data.wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                gpu_data.skip_clip = item.settings.ignore_clip;
                if frame.interaction.outline_selected && item.settings.selected {
                    polyline_selected_gpu_indices.push(polyline_gpu_data.len());
                }
                polyline_gpu_data.push(gpu_data);

                // Auto-generate GlyphItems for node/edge vector quantities.
                if !item.node_vectors.is_empty() {
                    resources.ensure_glyph_pipeline(device);
                    let g = crate::quantities::polyline_node_vectors_to_glyphs(item);
                    if !g.positions.is_empty() {
                        let wf = frame.viewport.wireframe_mode || item.settings.wireframe;
                        let gd = resources.upload_glyph_set_per_frame(device, queue, &g, wf);
                        glyph_gpu_data.push(gd);
                    }
                }
                if !item.edge_vectors.is_empty() {
                    resources.ensure_glyph_pipeline(device);
                    let g = crate::quantities::polyline_edge_vectors_to_glyphs(item);
                    if !g.positions.is_empty() {
                        let wf = frame.viewport.wireframe_mode || item.settings.wireframe;
                        let gd = resources.upload_glyph_set_per_frame(device, queue, &g, wf);
                        glyph_gpu_data.push(gd);
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Pre-uploaded polyline references.
        // ------------------------------------------------------------------
        if !frame.scene.polyline_refs.is_empty() {
            resources.ensure_polyline_pipeline(device);
            // viewport_width/height live at offset 96 of PolylineUniform;
            // they drive the screen-space miter expansion and must reflect
            // the current viewport size, not the placeholder used at upload.
            let vp = [
                frame.camera.viewport_size[0].max(1.0),
                frame.camera.viewport_size[1].max(1.0),
            ];
            for ref_item in &frame.scene.polyline_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources.content.polyline_store.get(ref_item.source) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                // Model matrix at offset 0.
                queue.write_buffer(&entry._uniform_buf, 0, bytemuck::bytes_of(&ref_item.model));
                queue.write_buffer(&entry._uniform_buf, 96, bytemuck::bytes_of(&vp));
                let mut gpu_data = entry;
                gpu_data.wireframe = frame.viewport.wireframe_mode || ref_item.settings.wireframe;
                if frame.interaction.outline_selected && ref_item.settings.selected {
                    polyline_selected_gpu_indices.push(polyline_gpu_data.len());
                }
                polyline_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // Scatter-volume bounds outlines: emit a polyline of the volume
        // shape for each volume whose `selected` or `wireframe` flag is set,
        // or when global wireframe mode is on. Scatter volumes have no other
        // selection feedback, so the outline is independent of
        // `interaction.outline_selected` (which gates surface-mesh outlines).
        // ------------------------------------------------------------------
        if !frame.scene.scatter_volumes.is_empty() {
            for item in &frame.scene.scatter_volumes {
                if item.settings.hidden {
                    continue;
                }
                let show_outline = item.settings.selected
                    || item.settings.wireframe
                    || frame.viewport.wireframe_mode;
                if !show_outline {
                    continue;
                }
                resources.ensure_polyline_pipeline(device);
                let colour = if item.settings.selected {
                    [1.0_f32, 0.9, 0.2, 1.0]
                } else {
                    [0.8_f32, 0.85, 0.95, 1.0]
                };
                let polyline = match item.volume.shape {
                    crate::scene::scatter_volume::ScatterShape::Box(b) => {
                        crate::renderer::aabb_wireframe_polyline(&b, colour)
                    }
                    crate::scene::scatter_volume::ScatterShape::Sphere { center, radius } => {
                        crate::renderer::sphere_wireframe_polyline(center, radius, 48, colour)
                    }
                };
                let mut gpu_data =
                    resources.upload_polyline_per_frame(device, queue, &polyline, vp_size);
                gpu_data.wireframe = true;
                polyline_gpu_data.push(gpu_data);
            }
        }

        // ------------------------------------------------------------------
        // isoline extraction and upload via polyline pipeline.
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
                    colourmap_id: None,
                    default_colour: item.colour,
                    line_width: item.line_width,
                    ..Default::default()
                };
                let gpu_data =
                    resources.upload_polyline_per_frame(device, queue, &polyline, vp_size);
                polyline_gpu_data.push(gpu_data);
            }
        }
    }

    pub(super) fn upload_implicit_decals_mc(
        resources: &mut DeviceResources,
        implicit_gpu_data: &mut Vec<crate::resources::volume::implicit::ImplicitGpuItem>,
        pick_implicit_items: &mut Vec<GpuImplicitPickItem>,
        decal_gpu_data: &mut Vec<crate::resources::decal::DecalGpuItem>,
        decal_cache: &mut std::collections::HashMap<u64, crate::resources::decal::DecalGpuItem>,
        decal_exclude_items: &mut Vec<crate::resources::decal::DecalExcludeGpuItem>,
        mc_gpu_data: &mut Vec<crate::resources::volume::gpu_marching_cubes::McFrameData>,
        pick_mc_items: &mut Vec<GpuMcPickItem>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> DecalCacheStats {
        // ------------------------------------------------------------------
        // GPU implicit surface items.
        // ------------------------------------------------------------------
        implicit_gpu_data.clear();
        pick_implicit_items.clear();
        if !frame.scene.gpu_implicit.is_empty() {
            resources.ensure_implicit_pipeline(device);
            for item in &frame.scene.gpu_implicit {
                if item.settings.hidden || item.primitives.is_empty() {
                    continue;
                }
                let mut gpu = resources.upload_implicit_item(device, item);
                gpu.pick_id = item.settings.pick_id;
                implicit_gpu_data.push(gpu);
                if item.settings.pick_id != PickId::NONE {
                    pick_implicit_items.push(GpuImplicitPickItem {
                        id: item.settings.pick_id.0,
                        primitives: item.primitives.clone(),
                        blend_mode: item.blend_mode,
                        max_steps: item.march_options.max_steps,
                        step_scale: item.march_options.step_scale,
                        hit_threshold: item.march_options.hit_threshold,
                        max_distance: item.march_options.max_distance,
                    });
                }
            }
        }

        // ------------------------------------------------------------------
        // Screen-space decals, sorted by sort_key.
        // ------------------------------------------------------------------
        decal_gpu_data.clear();
        let mut decal_stats = DecalCacheStats::default();
        if frame.scene.decals.is_empty() {
            // No decals this frame: drop any cached GPU resources.
            decal_cache.clear();
        } else {
            resources.ensure_decal_pipeline(device);
            // Stable sort so equal-key decals stay in submission order.
            let mut sorted: Vec<&crate::renderer::DecalItem> = frame.scene.decals.iter().collect();
            sorted.sort_by_key(|d| d.sort_key);
            // Reuse cached GPU resources for unchanged decals; only new or
            // changed decals rebuild a uniform buffer and bind group. Decals are
            // static per submission, so this is a hit in steady state.
            let mut seen: std::collections::HashSet<u64> =
                std::collections::HashSet::with_capacity(sorted.len());
            for item in sorted {
                if item.settings.hidden {
                    continue;
                }
                if item.settings.opacity <= 0.0 {
                    continue;
                }
                // Apply appearance.opacity on top of the item's own alpha.
                let mut effective = item.clone();
                effective.alpha *= item.settings.opacity;
                let key = crate::resources::decal::hash_decal_item(&effective);
                match decal_cache.entry(key) {
                    std::collections::hash_map::Entry::Occupied(e) => {
                        // `selected` is not part of the cache key, so refresh it
                        // on the reused clone to reflect this frame's selection.
                        let mut gpu = e.get().clone();
                        gpu.selected = effective.settings.selected;
                        decal_gpu_data.push(gpu);
                        decal_stats.reused += 1;
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        let gpu = resources.upload_decal_item(device, &effective);
                        decal_gpu_data.push(gpu.clone());
                        e.insert(gpu);
                        decal_stats.uploads += 1;
                    }
                }
                seen.insert(key);
            }
            // Evict decals that were not part of this frame's submission.
            decal_cache.retain(|k, _| seen.contains(k));
        }

        // ------------------------------------------------------------------
        // Collect non-receiver surfaces for the decal exclude pass.
        // ------------------------------------------------------------------
        decal_exclude_items.clear();
        {
            let crate::SurfaceSubmission::Flat(ref surfaces) = frame.scene.surfaces;
            let has_exclude = surfaces
                .iter()
                .any(|item| !item.receives_decals && !item.settings.hidden);
            if has_exclude {
                resources.ensure_decal_exclude_pipeline(device);
                for item in surfaces.iter() {
                    if !item.receives_decals && !item.settings.hidden {
                        let gpu =
                            resources.upload_decal_exclude_item(device, item.mesh_id, item.model);
                        decal_exclude_items.push(gpu);
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // GPU marching cubes compute dispatch.
        // ------------------------------------------------------------------
        mc_gpu_data.clear();
        pick_mc_items.clear();
        if !frame.scene.gpu_mc_items.is_empty() {
            resources.ensure_mc_pipelines(device);
            *mc_gpu_data = resources.run_mc_jobs(device, queue, &frame.scene.gpu_mc_items);
            for job in &frame.scene.gpu_mc_items {
                if job.settings.pick_id != PickId::NONE {
                    if let Some(cpu_data) = &job.cpu_data {
                        pick_mc_items.push(GpuMcPickItem {
                            id: job.settings.pick_id.0,
                            isovalue: job.isovalue,
                            volume_data: cpu_data.clone(),
                        });
                    }
                }
            }
        }

        decal_stats
    }

    pub(super) fn upload_images(
        resources: &mut DeviceResources,
        screen_image_gpu_data: &mut Vec<crate::resources::ScreenImageGpuData>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        let vp_size = frame.camera.viewport_size;

        // ------------------------------------------------------------------
        // Screen-space image overlays.
        // ------------------------------------------------------------------
        screen_image_gpu_data.clear();
        if !frame.scene.screen_images.is_empty() {
            resources.ensure_screen_image_pipeline(device);
            // Ensure dc pipeline if any item carries depth data.
            if frame.scene.screen_images.iter().any(|i| i.depth.is_some()) {
                resources.ensure_screen_image_dc_pipeline(device);
            }
            let vp_w = vp_size[0];
            let vp_h = vp_size[1];
            for item in &frame.scene.screen_images {
                if item.settings.hidden
                    || item.width == 0
                    || item.height == 0
                    || item.pixels.is_empty()
                {
                    continue;
                }
                let gpu = resources.upload_screen_image(device, queue, item, vp_w, vp_h);
                screen_image_gpu_data.push(gpu);
            }
        }
    }

    pub(super) fn upload_tubes_ribbons(
        resources: &mut DeviceResources,
        streamtube_gpu_data: &mut Vec<crate::resources::StreamtubeGpuData>,
        streamtube_selected_gpu_indices: &mut Vec<usize>,
        tube_gpu_data: &mut Vec<crate::resources::StreamtubeGpuData>,
        tube_selected_gpu_indices: &mut Vec<usize>,
        ribbon_gpu_data: &mut Vec<crate::resources::StreamtubeGpuData>,
        ribbon_selected_gpu_indices: &mut Vec<usize>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // ------------------------------------------------------------------
        // streamtube GPU data upload.
        // ------------------------------------------------------------------
        streamtube_gpu_data.clear();
        streamtube_selected_gpu_indices.clear();
        if !frame.scene.streamtube_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.scene.streamtube_items {
                if item.settings.hidden
                    || item.positions.is_empty()
                    || item.strip_lengths.is_empty()
                {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                let mut gpu_data =
                    resources.upload_streamtube_per_frame(device, queue, item, wireframe);
                gpu_data.pick_id = item.settings.pick_id;
                gpu_data.model = item.model;
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && item.settings.selected {
                        streamtube_selected_gpu_indices.push(streamtube_gpu_data.len());
                    }
                    streamtube_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Pre-uploaded streamtube references.
        // ------------------------------------------------------------------
        if !frame.scene.streamtube_refs.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for ref_item in &frame.scene.streamtube_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources.content.streamtube_store.get(ref_item.source) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                queue.write_buffer(&entry._uniform_buf, 0, bytemuck::bytes_of(&ref_item.model));
                let mut gpu_data = entry;
                gpu_data.pick_id = ref_item.settings.pick_id;
                gpu_data.model = ref_item.model;
                gpu_data.wireframe = frame.viewport.wireframe_mode || ref_item.settings.wireframe;
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && ref_item.settings.selected {
                        streamtube_selected_gpu_indices.push(streamtube_gpu_data.len());
                    }
                    streamtube_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // General Tube GPU data upload.
        // ------------------------------------------------------------------
        tube_gpu_data.clear();
        tube_selected_gpu_indices.clear();
        if !frame.scene.tube_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.scene.tube_items {
                if item.settings.hidden
                    || item.positions.is_empty()
                    || item.strip_lengths.is_empty()
                {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                let mut gpu_data = resources.upload_tube_per_frame(device, queue, item, wireframe);
                gpu_data.pick_id = item.settings.pick_id;
                gpu_data.model = item.model;
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && item.settings.selected {
                        tube_selected_gpu_indices.push(tube_gpu_data.len());
                    }
                    tube_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Pre-uploaded tube references.
        // ------------------------------------------------------------------
        if !frame.scene.tube_refs.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for ref_item in &frame.scene.tube_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources.content.tube_store.get(ref_item.source) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                queue.write_buffer(&entry._uniform_buf, 0, bytemuck::bytes_of(&ref_item.model));
                let mut gpu_data = entry;
                gpu_data.pick_id = ref_item.settings.pick_id;
                gpu_data.model = ref_item.model;
                gpu_data.wireframe = frame.viewport.wireframe_mode || ref_item.settings.wireframe;
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && ref_item.settings.selected {
                        tube_selected_gpu_indices.push(tube_gpu_data.len());
                    }
                    tube_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Ribbon GPU data upload.
        // ------------------------------------------------------------------
        ribbon_gpu_data.clear();
        ribbon_selected_gpu_indices.clear();
        if !frame.scene.ribbon_items.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for item in &frame.scene.ribbon_items {
                if item.settings.hidden
                    || item.positions.is_empty()
                    || item.strip_lengths.is_empty()
                {
                    continue;
                }
                let wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                let mut gpu_data =
                    resources.upload_ribbon_per_frame(device, queue, item, wireframe);
                gpu_data.pick_id = item.settings.pick_id;
                gpu_data.model = item.model;
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && item.settings.selected {
                        ribbon_selected_gpu_indices.push(ribbon_gpu_data.len());
                    }
                    ribbon_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Pre-uploaded ribbon references.
        // ------------------------------------------------------------------
        if !frame.scene.ribbon_refs.is_empty() {
            resources.ensure_streamtube_pipeline(device);
            for ref_item in &frame.scene.ribbon_refs {
                if ref_item.settings.hidden {
                    continue;
                }
                let entry = match resources.content.ribbon_store.get(ref_item.source) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                queue.write_buffer(&entry._uniform_buf, 0, bytemuck::bytes_of(&ref_item.model));
                let mut gpu_data = entry;
                gpu_data.pick_id = ref_item.settings.pick_id;
                gpu_data.model = ref_item.model;
                gpu_data.wireframe = frame.viewport.wireframe_mode || ref_item.settings.wireframe;
                if gpu_data.index_count > 0 {
                    if frame.interaction.outline_selected && ref_item.settings.selected {
                        ribbon_selected_gpu_indices.push(ribbon_gpu_data.len());
                    }
                    ribbon_gpu_data.push(gpu_data);
                }
            }
        }
    }

    pub(super) fn upload_slices(
        resources: &mut DeviceResources,
        image_slice_gpu_data: &mut Vec<crate::resources::ImageSliceGpuData>,
        volume_surface_slice_gpu_data: &mut Vec<crate::resources::VolumeSurfaceSliceGpuData>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // ------------------------------------------------------------------
        // Image Slice GPU data upload.
        // ------------------------------------------------------------------
        image_slice_gpu_data.clear();
        if !frame.scene.image_slices.is_empty() {
            resources.ensure_image_slice_pipeline(device);
            for item in &frame.scene.image_slices {
                if item.settings.hidden {
                    continue;
                }
                if let Some(gpu_data) = resources.upload_image_slice(device, queue, item) {
                    image_slice_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
        // Volume Surface Slice GPU data upload.
        // ------------------------------------------------------------------
        volume_surface_slice_gpu_data.clear();
        if !frame.scene.volume_surface_slices.is_empty() {
            resources.ensure_volume_surface_slice_pipeline(device);
            for item in &frame.scene.volume_surface_slices {
                if item.settings.hidden {
                    continue;
                }
                if let Some(gpu_data) = resources.upload_volume_surface_slice(device, queue, item) {
                    volume_surface_slice_gpu_data.push(gpu_data);
                }
            }
        }

        // ------------------------------------------------------------------
    }
}
