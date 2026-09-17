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
            resources.check_texture_slot(
                item.texture_id,
                crate::resources::TextureSlot::MeshInstanceAlbedo,
            );

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

    pub(super) fn upload_polylines(
        resources: &mut DeviceResources,
        polyline_gpu_data: &mut Vec<crate::resources::PolylineGpuData>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // ------------------------------------------------------------------
        // The shared line substrate: everything that renders through the
        // polyline pipelines without being a polyline item. The polyline item
        // type prepares its own draws in its plugin.
        // ------------------------------------------------------------------
        polyline_gpu_data.clear();
        let vp_size = frame.camera.viewport_size;

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

    pub(super) fn upload_decals(
        resources: &mut DeviceResources,
        decal_gpu_data: &mut Vec<crate::resources::decal::DecalGpuItem>,
        decal_cache: &mut std::collections::HashMap<
            u64,
            (
                crate::resources::decal::DecalGpuItem,
                crate::resources::resource_deps::ResourceDeps,
            ),
        >,
        decal_deps_gate: &mut crate::resources::resource_deps::DepsGate,
        decal_exclude_items: &mut Vec<crate::resources::decal::DecalExcludeGpuItem>,
        device: &crate::gpu::Device,
        frame: &FrameData,
    ) -> DecalCacheStats {
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
            // Cached entries hold bind groups over texture views, so a free or
            // a replace since the last frame invalidates them: a free drops
            // only the entries whose deps no longer resolve (they rebuild
            // against the fallback below), a replace drops everything, since a
            // view swapped behind a live id cannot be detected per entry.
            match decal_deps_gate.poll(resources) {
                crate::resources::resource_deps::Revalidate::RebuildAll => decal_cache.clear(),
                crate::resources::resource_deps::Revalidate::CheckEach => {
                    decal_cache.retain(|_, (_, deps)| deps.resolves(resources));
                }
                crate::resources::resource_deps::Revalidate::Valid => {}
            }
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
                let key = crate::resources::decal::hash_decal_item(
                    &effective,
                    &resources.content.textures,
                );
                match decal_cache.entry(key) {
                    std::collections::hash_map::Entry::Occupied(e) => {
                        // `selected` is not part of the cache key, so refresh it
                        // on the reused clone to reflect this frame's selection.
                        let mut gpu = e.get().0.clone();
                        gpu.selected = effective.settings.selected;
                        decal_gpu_data.push(gpu);
                        decal_stats.reused += 1;
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        use crate::resources::TextureSlot;
                        resources.check_texture_slot(
                            Some(effective.texture_id),
                            TextureSlot::DecalAlbedo,
                        );
                        resources.check_texture_slot(
                            effective.normal_texture_id,
                            TextureSlot::DecalNormalMap,
                        );
                        resources.check_texture_slot(
                            effective.roughness_texture_id,
                            TextureSlot::DecalRoughness,
                        );
                        resources.check_texture_slot(
                            effective.metallic_texture_id,
                            TextureSlot::DecalMetallic,
                        );
                        resources.check_texture_slot(
                            effective.emissive_texture_id,
                            TextureSlot::DecalEmissive,
                        );
                        let gpu = resources.upload_decal_item(device, &effective);
                        let deps = crate::resources::resource_deps::ResourceDeps::textures([
                            Some(effective.texture_id),
                            effective.normal_texture_id,
                            effective.roughness_texture_id,
                            effective.metallic_texture_id,
                            effective.emissive_texture_id,
                        ]);
                        decal_gpu_data.push(gpu.clone());
                        e.insert((gpu, deps));
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
}
