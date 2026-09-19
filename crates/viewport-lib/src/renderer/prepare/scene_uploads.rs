//! Scivis per-frame GPU upload passes called from `prepare_scene_internal`.
//!
//! Associated functions (no `self`): the caller holds a long-lived
//! `&mut self.resources` borrow across the whole scene prepare, so each takes
//! `resources` plus the disjoint `self` fields it fills.

use super::*;

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
}
