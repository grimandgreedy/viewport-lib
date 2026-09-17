//! CPU ray-cast pick: find the nearest item or sub-element under the cursor.

use super::*;

impl ViewportRenderer {
    // -----------------------------------------------------------------------
    // Unified CPU pick : renderer.pick()
    // -----------------------------------------------------------------------

    /// Pick the nearest item or sub-element under `click_pos`.
    ///
    /// Dispatches across all item types retained from the last `prepare()` call.
    /// The `mask` controls which item types and sub-element levels participate.
    ///
    /// Returns `None` if nothing matching the mask is under the cursor.
    ///
    /// # Arguments
    /// * `click_pos`     - cursor position in viewport pixels (top-left origin)
    /// * `viewport_size` - viewport width x height in pixels
    /// * `view_proj`     - combined view x projection matrix from the last frame
    /// * `mask`          - which item types and sub-element levels to include
    ///
    /// # Example
    /// ```rust,ignore
    /// if let Some(hit) = renderer.pick(cursor, vp_size, view_proj, PickMask::FACE) {
    ///     println!("hit face {:?} on object {}", hit.sub_object, hit.id);
    /// }
    /// ```
    pub fn pick(
        &self,
        click_pos: glam::Vec2,
        viewport_size: glam::Vec2,
        view_proj: glam::Mat4,
        mask: PickMask,
    ) -> Option<PickHit> {
        use crate::interaction::query::picking::{pick_transparent_volume_mesh_cpu, screen_to_ray};
        use parry3d::math::{Pose, Vector};
        use parry3d::query::{Ray, RayCast};

        if !self.cpu_pick_cache_enabled {
            warn_pick_cache_disabled();
            return None;
        }

        if viewport_size.x <= 0.0 || viewport_size.y <= 0.0 {
            return None;
        }

        let view_proj_inv = view_proj.inverse();
        let (ray_origin, ray_dir) = screen_to_ray(click_pos, viewport_size, view_proj_inv);

        let wants_face = mask.intersects(PickMask::FACE);
        let wants_vertex = mask.intersects(PickMask::VERTEX);
        let wants_cell = mask.intersects(PickMask::CELL);
        let wants_object = mask.intersects(PickMask::OBJECT);
        let wants_mesh_sub = wants_face || wants_vertex || mask.intersects(PickMask::EDGE);

        // (toi, hit) -- nearest hit so far across all types.
        let mut best: Option<(f32, PickHit)> = None;

        let mut consider = |toi: f32, hit: PickHit| {
            if best.as_ref().map_or(true, |(bt, _)| toi < *bt) {
                best = Some((toi, hit));
            }
        };

        // Build lookup for opaque volume mesh face_to_cell maps (used in section 1
        // to convert surface Face hits to Cell hits).
        let vm_cell_map: std::collections::HashMap<u64, &[u32]> = self
            .pick_volume_mesh_items
            .iter()
            .filter(|item| item.settings.pick_id != PickId::NONE && !item.face_to_cell.is_empty())
            .map(|item| (item.settings.pick_id.0, item.face_to_cell.as_slice()))
            .collect();

        // 1. Surface mesh picks (FACE, VERTEX, EDGE, CELL, or OBJECT fallback).
        if wants_mesh_sub || wants_cell || wants_object {
            // Broad phase: only the items the ray actually pierces. The per-item body
            // below is unchanged, so the result matches a full scan: items the ray
            // misses never produce a hit anyway.
            let candidates = self.pierced_surface_items(ray_origin, ray_dir);
            for &item_index in &candidates {
                let item = &self.pick_scene_items[item_index];
                if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                    continue;
                };
                let (Some(positions), Some(indices)) = (&mesh.cpu_positions, &mesh.cpu_indices)
                else {
                    continue;
                };
                let Some(trimesh) = mesh.cached_pick_trimesh() else {
                    continue;
                };

                let model = glam::Mat4::from_cols_array_2d(&item.model);
                // Cast the ray in the mesh's local space instead of baking the
                // model matrix into a fresh world-space vertex Vec every click:
                // `trimesh` is cached per mesh_id (see `cached_pick_trimesh`), so
                // this serves every instance of a shared mesh. `transform_vector3`
                // (not `transform_point3`) applies only the linear part, so
                // `local_dir` is not renormalized and `toi` comes out identical to
                // the world-space parametrization.
                let inv_model = model.inverse();
                let local_origin = inv_model.transform_point3(ray_origin);
                let local_dir = inv_model.transform_vector3(ray_dir);
                let ray = Ray::new(
                    Vector::new(local_origin.x, local_origin.y, local_origin.z),
                    Vector::new(local_dir.x, local_dir.y, local_dir.z),
                );

                {
                    // Vertices are in mesh-local space: use identity pose.
                    let identity = Pose::identity();
                    let Some(intersection) =
                        trimesh.cast_ray_and_get_normal(&identity, &ray, f32::MAX, true)
                    else {
                        continue;
                    };
                    let toi = intersection.time_of_impact;
                    let world_pos = ray_origin + ray_dir * toi;
                    // Transform the local-space normal back to world space with
                    // the inverse-transpose of the model's linear part, so
                    // non-uniform scale does not distort it.
                    let normal_matrix = glam::Mat3::from_mat4(model).inverse().transpose();
                    let local_normal = glam::Vec3::new(
                        intersection.normal.x,
                        intersection.normal.y,
                        intersection.normal.z,
                    );
                    let normal = normal_matrix.mul_vec3(local_normal).normalize();

                    let feature_sub = SubObjectRef::from_feature_id(intersection.feature);

                    let sub_object = if wants_face {
                        feature_sub
                    } else if wants_cell {
                        // Convert surface Face hit to originating cell index.
                        if let Some(f2c) = vm_cell_map.get(&item.settings.pick_id.0) {
                            match feature_sub {
                                Some(SubObjectRef::Face(face_raw)) => {
                                    let n_tri = indices.len() / 3;
                                    let face = if (face_raw as usize) >= n_tri {
                                        face_raw as usize - n_tri
                                    } else {
                                        face_raw as usize
                                    };
                                    f2c.get(face).map(|&ci| SubObjectRef::Cell(ci))
                                }
                                other => other,
                            }
                        } else if wants_vertex {
                            // No cell map for this item; try vertex picking instead.
                            // Fall through to the vertex branch below by
                            // re-evaluating with the vertex logic inline.
                            match feature_sub {
                                Some(SubObjectRef::Face(face_raw)) => {
                                    let n_tri = indices.len() / 3;
                                    let face = if (face_raw as usize) >= n_tri {
                                        face_raw as usize - n_tri
                                    } else {
                                        face_raw as usize
                                    };
                                    if face * 3 + 2 < indices.len() {
                                        let vis = [
                                            indices[face * 3] as usize,
                                            indices[face * 3 + 1] as usize,
                                            indices[face * 3 + 2] as usize,
                                        ];
                                        let (best_vi, _) = vis
                                            .iter()
                                            .map(|&i| {
                                                let p = model.transform_point3(glam::Vec3::from(
                                                    positions[i],
                                                ));
                                                (i, p.distance(world_pos))
                                            })
                                            .fold((vis[0], f32::MAX), |acc, (i, d)| {
                                                if d < acc.1 { (i, d) } else { acc }
                                            });
                                        Some(SubObjectRef::Vertex(best_vi as u32))
                                    } else {
                                        None
                                    }
                                }
                                other => other,
                            }
                        } else {
                            // No cell map and vertex not wanted; no sub-element.
                            None
                        }
                    } else if wants_vertex {
                        // Convert face hit to nearest triangle corner.
                        match feature_sub {
                            Some(SubObjectRef::Face(face_raw)) => {
                                let n_tri = indices.len() / 3;
                                let face = if (face_raw as usize) >= n_tri {
                                    face_raw as usize - n_tri
                                } else {
                                    face_raw as usize
                                };
                                if face * 3 + 2 < indices.len() {
                                    let vis = [
                                        indices[face * 3] as usize,
                                        indices[face * 3 + 1] as usize,
                                        indices[face * 3 + 2] as usize,
                                    ];
                                    let (best_vi, _) = vis
                                        .iter()
                                        .map(|&i| {
                                            let p = model
                                                .transform_point3(glam::Vec3::from(positions[i]));
                                            (i, p.distance(world_pos))
                                        })
                                        .fold((vis[0], f32::MAX), |acc, (i, d)| {
                                            if d < acc.1 { (i, d) } else { acc }
                                        });
                                    Some(SubObjectRef::Vertex(best_vi as u32))
                                } else {
                                    None
                                }
                            }
                            other => other,
                        }
                    } else {
                        // Object-only: no sub-element.
                        None
                    };

                    // Only emit the hit if we produced a meaningful sub-element
                    // or the caller explicitly asked for object-level hits.
                    // Without this guard, an EDGE-only mask runs the ray-trimesh
                    // intersection (because wants_mesh_sub is true) but falls through
                    // to sub_object=None, producing a spurious object-level hit.
                    if sub_object.is_some() || wants_object {
                        #[allow(deprecated)]
                        let hit = PickHit {
                            id: item.settings.pick_id.0,
                            sub_object,
                            world_pos,
                            normal,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        };
                        consider(toi, hit);
                    }
                }
            }
        }

        // 2. Opaque volume mesh cell picks are handled in section 1 above via
        // vm_cell_map (face_to_cell conversion on surface Face hits).

        // 2c. Scatter-volume object picks. Ray-vs-shape intersection only;
        // there is no sub-object level for participating media
        if wants_object {
            for item in &self.pick_scatter_volume_items {
                if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                    continue;
                }
                if let Some((t_enter, _)) = crate::scene::scatter_volume::ray_intersect(
                    &item.volume.shape,
                    ray_origin,
                    ray_dir,
                ) {
                    let world_pos = ray_origin + ray_dir * t_enter;
                    let normal = (world_pos
                        - match item.volume.shape {
                            crate::scene::scatter_volume::ScatterShape::Box(b) => {
                                (b.min + b.max) * 0.5
                            }
                            crate::scene::scatter_volume::ScatterShape::Sphere {
                                center, ..
                            } => glam::Vec3::from(center),
                        })
                    .try_normalize()
                    .unwrap_or(glam::Vec3::Z);
                    consider(
                        t_enter,
                        PickHit::object_hit(item.settings.pick_id.0, world_pos, normal),
                    );
                }
            }
        }

        // 2b. Interior-inclusive cell picks for volume meshes rendering
        //     transparently. Items rendering as opaque are handled in section 1
        //     above via vm_cell_map (face_to_cell on the boundary surface).
        if wants_cell || wants_object {
            for item in &self.pick_volume_mesh_items {
                if item.settings.pick_id == PickId::NONE || item.transparency.is_none() {
                    continue;
                }
                let Some(data) = item.volume_mesh_data.as_deref() else {
                    continue;
                };
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                if let Some(mut hit) = pick_transparent_volume_mesh_cpu(
                    ray_origin,
                    ray_dir,
                    item.settings.pick_id.0,
                    model,
                    data,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if !wants_cell {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }
        }

        // 6. Instance picks (INSTANCE or OBJECT fallback) for glyphs, tensor glyphs, sprites.
        let wants_instance = mask.intersects(PickMask::INSTANCE);
        if wants_instance || wants_object {}

        // 13. Decal picks (OBJECT only): ray versus the decal projection box.
        // A decal is the unit box [-0.5, 0.5]^3 mapped to world by `transform`.
        // The box front face typically hugs the receiver surface, so a decal
        // that straddles a surface wins over that surface by `toi`, letting a
        // click select the decal itself. A decal whose box floats in empty
        // space is still pickable wherever the ray passes through the volume.
        if wants_object {
            for item in &self.pick_decal_items {
                if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.transform);
                if model.determinant().abs() < 1e-12 {
                    continue;
                }
                let inv = model.inverse();
                let local_origin = inv.transform_point3(ray_origin);
                let local_dir = inv.transform_vector3(ray_dir);
                if let Some(toi) = ray_unit_box_toi(local_origin, local_dir) {
                    let world_pos = ray_origin + ray_dir * toi;
                    #[allow(deprecated)]
                    consider(
                        toi,
                        PickHit {
                            id: item.settings.pick_id.0,
                            sub_object: None,
                            world_pos,
                            normal: -ray_dir.normalize_or_zero(),
                            scalar_value: None,
                            sub_object_world_pos: None,
                        },
                    );
                }
            }
        }

        // Consult registered item-type plugins after the built-in pickers.
        // Each plugin returns its own closest hit; the router compares
        // by world-space ray t against the running best.
        if !self.item_type_plugins.is_empty() {
            let plugin_ray = crate::plugin_api::PickRay {
                origin: ray_origin,
                direction: ray_dir,
            };
            let plugin_ctx = crate::plugin_api::PickContext {
                click_pos,
                viewport_size,
                view_proj,
                mask,
                meshes: crate::resources::MeshGeometry::new(&self.resources),
            };
            for plugin in self.item_type_plugins.values() {
                if let Some((t, hit)) = plugin.pick(&plugin_ray, &plugin_ctx) {
                    consider(t, hit);
                }
            }
        }

        best.map(|(_, hit)| hit)
    }
}

// ---------------------------------------------------------------------------
// Broad-phase BVH over the surface pick items
// ---------------------------------------------------------------------------

use crate::resources::mesh::mesh_store::MeshStore;
use spatial_query::{
    Aabb as SqAabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray as SqRay,
};

/// One pickable surface leaf: an index into `pick_scene_items` and its world AABB.
struct PickEntry {
    item_index: usize,
    world_aabb: SqAabb<3>,
}

/// A spatial BVH over the pickable surface items, plus the revs it was built for.
/// Stored behind the renderer's `pick_bvh` mutex.
pub(crate) struct PickSceneBvh {
    bvh: Option<Bvh<3>>,
    entries: Vec<PickEntry>,
    identity_rev: u64,
    transform_rev: u64,
}

/// `QueryGeometry` over the surface pick items. `test_ray` casts each mesh's cached
/// parry `TriMesh` in mesh-local space, the same narrow phase the pick body uses, so
/// a leaf is "pierced" exactly when the body would find an intersection.
struct SurfaceGeom<'a> {
    entries: &'a [PickEntry],
    items: &'a [SceneRenderItem],
    mesh_store: &'a MeshStore,
}

impl QueryGeometry<3> for SurfaceGeom<'_> {
    type Id = u32;
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        self.entries.len()
    }

    fn id(&self, leaf: usize) -> u32 {
        self.entries[leaf].item_index as u32
    }

    fn world_aabb(&self, leaf: usize) -> SqAabb<3> {
        self.entries[leaf].world_aabb
    }

    fn test_ray(&self, leaf: usize, ray: &SqRay<3>, max_toi: f32) -> Option<LeafHit<3>> {
        let item = &self.items[self.entries[leaf].item_index];
        let mesh = self.mesh_store.get(item.mesh_id)?;
        let trimesh = mesh.cached_pick_trimesh()?;
        let model = glam::Mat4::from_cols_array_2d(&item.model);
        let inv_model = model.inverse();
        let local_origin = inv_model.transform_point3(glam::Vec3::new(
            ray.origin[0],
            ray.origin[1],
            ray.origin[2],
        ));
        let local_dir =
            inv_model.transform_vector3(glam::Vec3::new(ray.dir[0], ray.dir[1], ray.dir[2]));
        let pray = parry3d::query::Ray::new(
            parry3d::math::Vector::new(local_origin.x, local_origin.y, local_origin.z),
            parry3d::math::Vector::new(local_dir.x, local_dir.y, local_dir.z),
        );
        use parry3d::query::RayCast;
        let toi = trimesh.cast_ray(&parry3d::math::Pose::identity(), &pray, max_toi, true)?;
        if toi < 0.0 || toi > max_toi {
            None
        } else {
            Some(LeafHit::new(toi, Point::ZERO))
        }
    }
}

impl ViewportRenderer {
    /// Refresh the pickable-surface revs. Called from `cache_pick_items` each frame.
    /// `identity_rev` captures which items are pickable (id + mesh + whether the mesh
    /// carries CPU geometry); `transform_rev` folds the model matrices in too, so a
    /// pure move changes only the transform rev.
    pub(crate) fn update_pick_bvh_revs(&mut self) {
        use std::hash::{Hash, Hasher};
        let mut id_hasher = std::collections::hash_map::DefaultHasher::new();
        let mut tf_hasher = std::collections::hash_map::DefaultHasher::new();
        for item in &self.pick_scene_items {
            if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                continue;
            }
            let pickable = self
                .resources
                .mesh_store
                .get(item.mesh_id)
                .map(|m| m.cpu_positions.is_some() && m.cpu_indices.is_some())
                .unwrap_or(false);
            item.settings.pick_id.0.hash(&mut id_hasher);
            item.mesh_id.index().hash(&mut id_hasher);
            pickable.hash(&mut id_hasher);
            item.settings.pick_id.0.hash(&mut tf_hasher);
            for row in &item.model {
                for v in row {
                    v.to_bits().hash(&mut tf_hasher);
                }
            }
        }
        let identity = id_hasher.finish();
        self.pick_bvh_identity_rev = identity;
        self.pick_bvh_transform_rev = identity ^ tf_hasher.finish();
    }

    /// Item indices the ray pierces, culled by the surface BVH. Lazily rebuilds the
    /// BVH on an identity change (items added, removed, or toggled), refits it on a
    /// pure move, and reuses it otherwise.
    fn pierced_surface_items(&self, ray_origin: glam::Vec3, ray_dir: glam::Vec3) -> Vec<usize> {
        self.query_pick_bvh(Vec::new(), |bvh, geom| {
            let ray = SqRay::new_unnormalized(
                Point([ray_origin.x, ray_origin.y, ray_origin.z]),
                Point([ray_dir.x, ray_dir.y, ray_dir.z]),
            );
            bvh.raycast_all(geom, &ray, f32::MAX, &QueryFilter::default())
                .into_iter()
                .map(|h| geom.entries[h.leaf].item_index)
                .collect()
        })
    }

    /// Item indices whose world AABB falls in the selection rectangle's frustum,
    /// culled by the surface BVH. The frustum's world AABB is clamped to the scene
    /// bound so deep objects (which the linear scan also selects, since it does not
    /// clip by depth) are kept. The per-item body then refines by projecting
    /// geometry, so the result matches a full scan.
    pub(super) fn rect_candidate_items(
        &self,
        rect_min: glam::Vec2,
        rect_max: glam::Vec2,
        viewport_size: glam::Vec2,
        view_proj: glam::Mat4,
    ) -> Vec<usize> {
        self.query_pick_bvh(Vec::new(), |bvh, geom| {
            // Scene bound: every leaf lives inside this.
            let mut smin = glam::Vec3::splat(f32::INFINITY);
            let mut smax = glam::Vec3::splat(f32::NEG_INFINITY);
            for e in geom.entries {
                smin = smin.min(glam::Vec3::new(
                    e.world_aabb.min[0],
                    e.world_aabb.min[1],
                    e.world_aabb.min[2],
                ));
                smax = smax.max(glam::Vec3::new(
                    e.world_aabb.max[0],
                    e.world_aabb.max[1],
                    e.world_aabb.max[2],
                ));
            }
            let diag = (smax - smin).length().max(1.0);

            let inv = view_proj.inverse();
            let unproject = |sx: f32, sy: f32, ndc_z: f32| -> Option<glam::Vec3> {
                let ndc_x = sx / viewport_size.x * 2.0 - 1.0;
                let ndc_y = 1.0 - sy / viewport_size.y * 2.0;
                let clip = inv * glam::Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
                if clip.w.abs() < 1e-9 {
                    None
                } else {
                    Some(clip.truncate() / clip.w)
                }
            };

            // World AABB of the rect frustum: near corners plus each corner ray
            // extended past the scene, then clamped to the scene bound.
            let corners = [
                (rect_min.x, rect_min.y),
                (rect_max.x, rect_min.y),
                (rect_min.x, rect_max.y),
                (rect_max.x, rect_max.y),
            ];
            let mut lo = glam::Vec3::splat(f32::INFINITY);
            let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
            let mut any = false;
            for (sx, sy) in corners {
                let Some(near) = unproject(sx, sy, 0.0) else {
                    continue;
                };
                lo = lo.min(near);
                hi = hi.max(near);
                any = true;
                if let Some(far) = unproject(sx, sy, 1.0) {
                    let dir = (far - near).normalize_or_zero();
                    let ext = near + dir * (diag * 4.0);
                    lo = lo.min(far).min(ext);
                    hi = hi.max(far).max(ext);
                }
            }
            if !any {
                return Vec::new();
            }
            // Tighten to the scene bound.
            lo = lo.max(smin);
            hi = hi.min(smax);
            if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
                return Vec::new();
            }

            let region = SqAabb::new(Point([lo.x, lo.y, lo.z]), Point([hi.x, hi.y, hi.z]));
            bvh.overlap(geom, &region, &QueryFilter::default())
                .into_iter()
                .map(|o| geom.entries[o.leaf].item_index)
                .collect()
        })
    }

    /// Ensure the surface BVH matches the current pick items (rebuild on an identity
    /// change, refit on a pure move, reuse otherwise), then run `query` against it.
    /// Returns `default` when there are no pickable surface items.
    fn query_pick_bvh<R>(&self, default: R, query: impl FnOnce(&Bvh<3>, &SurfaceGeom) -> R) -> R {
        let mut guard = self.pick_bvh.lock().unwrap();
        let stale = guard
            .as_ref()
            .is_none_or(|b| b.identity_rev != self.pick_bvh_identity_rev);
        if stale {
            *guard = Some(self.build_pick_bvh());
        } else if let Some(b) = guard.as_mut() {
            if b.transform_rev != self.pick_bvh_transform_rev {
                self.refit_pick_bvh(b);
            }
        }

        let b = guard.as_ref().expect("pick BVH populated above");
        let Some(bvh) = &b.bvh else {
            return default;
        };
        if b.entries.is_empty() {
            return default;
        }
        let geom = SurfaceGeom {
            entries: &b.entries,
            items: &self.pick_scene_items,
            mesh_store: &self.resources.mesh_store,
        };
        query(bvh, &geom)
    }

    /// Collect a `PickEntry` for every pickable surface item (world AABB from the
    /// mesh's local AABB times its model). Skips hidden, un-pickable, and CPU-less
    /// meshes, mirroring the pick body's guards.
    fn collect_pick_entries(&self) -> Vec<PickEntry> {
        let mut entries = Vec::new();
        for (i, item) in self.pick_scene_items.iter().enumerate() {
            if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                continue;
            }
            let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                continue;
            };
            if mesh.cpu_positions.is_none() || mesh.cpu_indices.is_none() {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            let world = mesh.aabb.transformed(&model);
            entries.push(PickEntry {
                item_index: i,
                world_aabb: SqAabb::new(
                    Point([world.min.x, world.min.y, world.min.z]),
                    Point([world.max.x, world.max.y, world.max.z]),
                ),
            });
        }
        entries
    }

    /// Build a fresh surface BVH for the current pickable items.
    fn build_pick_bvh(&self) -> PickSceneBvh {
        let entries = self.collect_pick_entries();
        let bvh = if entries.is_empty() {
            None
        } else {
            let geom = SurfaceGeom {
                entries: &entries,
                items: &self.pick_scene_items,
                mesh_store: &self.resources.mesh_store,
            };
            Some(Bvh::build(&geom))
        };
        PickSceneBvh {
            bvh,
            entries,
            identity_rev: self.pick_bvh_identity_rev,
            transform_rev: self.pick_bvh_transform_rev,
        }
    }

    /// Refit the surface BVH in place: recompute each leaf's world AABB from the
    /// current model, then propagate bounds up the existing tree. Valid because the
    /// identity rev is unchanged, so the item set and order match the build.
    fn refit_pick_bvh(&self, b: &mut PickSceneBvh) {
        for entry in &mut b.entries {
            let item = &self.pick_scene_items[entry.item_index];
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            if let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) {
                let world = mesh.aabb.transformed(&model);
                entry.world_aabb = SqAabb::new(
                    Point([world.min.x, world.min.y, world.min.z]),
                    Point([world.max.x, world.max.y, world.max.z]),
                );
            }
        }
        if let Some(bvh) = &mut b.bvh {
            let geom = SurfaceGeom {
                entries: &b.entries,
                items: &self.pick_scene_items,
                mesh_store: &self.resources.mesh_store,
            };
            bvh.refit(&geom);
        }
        b.transform_rev = self.pick_bvh_transform_rev;
    }
}
