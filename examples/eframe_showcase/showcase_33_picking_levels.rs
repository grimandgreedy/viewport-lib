//! Showcase 33: Picking Levels
//!
//! Demonstrates the picking hierarchy across four levels:
//!   Object -> Face -> Vertex -> Point
//!
//! Scene: two cubes and two hemispheres.  A 30-point grid cloud floats above.
//!
//! Controls:
//!   - Left-click : select at the active level
//!   - Shift + click : add / toggle selection
//!   - Left-drag : rubber-band box selection
//!   - Wireframe toggle in the side panel (on by default)

use std::collections::HashMap;

use eframe::egui;
use viewport_lib::{Material, MeshData, MeshId, NodeId, SubObjectRef, VolumeData, ViewportRenderer};

use crate::App;

// ---------------------------------------------------------------------------
// Pick level
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PlPickLevel {
    #[default]
    Object,
    Face,
    Vertex,
    Point,
    Voxel,
}

// ---------------------------------------------------------------------------
// Hit info (last single-click result displayed in the panel)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct PlHitInfo {
    pub object_name: String,
    pub world_pos: glam::Vec3,
    pub normal: glam::Vec3,
    pub sub_object: Option<SubObjectRef>,
    pub scalar_value: Option<f32>,
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Generate a hemisphere (dome + flat base disc).
/// Dome opens upward, flat base at y = 0.
fn hemisphere(radius: f32, n_lat: usize, n_lon: usize) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();

    // Top pole
    positions.push([0.0, radius, 0.0]);
    normals.push([0.0, 1.0, 0.0]);

    // Body rings: phi from PI/(n_lat*2) to PI/2
    for lat in 1..=n_lat {
        let phi = std::f32::consts::FRAC_PI_2 * lat as f32 / n_lat as f32;
        let sp = phi.sin();
        let cp = phi.cos();
        for lon in 0..n_lon {
            let theta = 2.0 * std::f32::consts::PI * lon as f32 / n_lon as f32;
            let nx = sp * theta.cos();
            let ny = cp;
            let nz = sp * theta.sin();
            positions.push([nx * radius, ny * radius, nz * radius]);
            normals.push([nx, ny, nz]);
        }
    }
    // Last ring is the equator (y ≈ 0).

    // Base disc: separate downward-normal vertices so normals are crisp.
    let disc_center = positions.len() as u32;
    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, -1.0, 0.0]);

    let disc_rim = positions.len() as u32;
    for lon in 0..n_lon {
        let theta = 2.0 * std::f32::consts::PI * lon as f32 / n_lon as f32;
        positions.push([radius * theta.cos(), 0.0, radius * theta.sin()]);
        normals.push([0.0, -1.0, 0.0]);
    }

    let mut indices: Vec<u32> = Vec::new();

    // Dome top cap
    for lon in 0..n_lon as u32 {
        indices.extend([0, 1 + (lon + 1) % n_lon as u32, 1 + lon]);
    }
    // Dome body quads
    for lat in 0..(n_lat as u32 - 1) {
        let rs = 1 + lat * n_lon as u32;
        for lon in 0..n_lon as u32 {
            let v0 = rs + lon;
            let v1 = rs + (lon + 1) % n_lon as u32;
            let v2 = rs + n_lon as u32 + lon;
            let v3 = rs + n_lon as u32 + (lon + 1) % n_lon as u32;
            indices.extend([v0, v2, v1, v1, v2, v3]);
        }
    }
    // Base disc (fan, CW when viewed from below so normal faces -Y)
    for lon in 0..n_lon as u32 {
        let v1 = disc_rim + lon;
        let v2 = disc_rim + (lon + 1) % n_lon as u32;
        indices.extend([disc_center, v2, v1]);
    }

    (positions, normals, indices)
}

/// Build a unit box MeshData (replicates box geometry with per-face normals).
fn unit_box_mesh_data() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    #[rustfmt::skip]
    let positions: Vec<[f32; 3]> = vec![
        // Front  (+Z)
        [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
        // Back   (-Z)
        [ 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0,  1.0, -1.0], [ 1.0,  1.0, -1.0],
        // Top    (+Y)
        [-1.0,  1.0,  1.0], [ 1.0,  1.0,  1.0], [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0],
        // Bottom (-Y)
        [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0], [ 1.0, -1.0,  1.0], [-1.0, -1.0,  1.0],
        // Right  (+X)
        [ 1.0, -1.0,  1.0], [ 1.0, -1.0, -1.0], [ 1.0,  1.0, -1.0], [ 1.0,  1.0,  1.0],
        // Left   (-X)
        [-1.0, -1.0, -1.0], [-1.0, -1.0,  1.0], [-1.0,  1.0,  1.0], [-1.0,  1.0, -1.0],
    ];
    #[rustfmt::skip]
    let normals: Vec<[f32; 3]> = vec![
        [ 0.0,  0.0,  1.0], [ 0.0,  0.0,  1.0], [ 0.0,  0.0,  1.0], [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0], [ 0.0,  0.0, -1.0], [ 0.0,  0.0, -1.0], [ 0.0,  0.0, -1.0],
        [ 0.0,  1.0,  0.0], [ 0.0,  1.0,  0.0], [ 0.0,  1.0,  0.0], [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0], [ 0.0, -1.0,  0.0], [ 0.0, -1.0,  0.0], [ 0.0, -1.0,  0.0],
        [ 1.0,  0.0,  0.0], [ 1.0,  0.0,  0.0], [ 1.0,  0.0,  0.0], [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0], [-1.0,  0.0,  0.0], [-1.0,  0.0,  0.0], [-1.0,  0.0,  0.0],
    ];
    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
         0,  1,  2,  0,  2,  3,
         4,  5,  6,  4,  6,  7,
         8,  9, 10,  8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    ];
    (positions, normals, indices)
}

// ---------------------------------------------------------------------------
// Build scene
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_pl_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.pl_scene = viewport_lib::scene::Scene::new();
        self.pl_selection = viewport_lib::Selection::new();
        self.pl_sub_selection = viewport_lib::SubSelection::new();
        self.pl_node_names.clear();
        self.pl_mesh_lookup.clear();
        self.pl_last_hit = None;
        self.pl_hit_marker = None;

        // --- Cube mesh ---
        let (cp, cn, ci) = unit_box_mesh_data();
        let mut cube_mesh = MeshData::default();
        cube_mesh.positions = cp.clone();
        cube_mesh.normals = cn;
        cube_mesh.indices = ci.clone();
        let cube_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &cube_mesh)
            .expect("pl cube mesh");
        self.pl_cube_mesh_id = cube_id;
        self.pl_mesh_lookup.insert(cube_id.index() as u64, (cp, ci));

        // --- Hemisphere mesh ---
        let (hp, hn, hi) = hemisphere(1.5, 8, 20);
        let mut hemi_mesh = MeshData::default();
        hemi_mesh.positions = hp.clone();
        hemi_mesh.normals = hn;
        hemi_mesh.indices = hi.clone();
        let hemi_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &hemi_mesh)
            .expect("pl hemi mesh");
        self.pl_hemi_mesh_id = hemi_id;
        self.pl_mesh_lookup.insert(hemi_id.index() as u64, (hp, hi));

        // --- Scene objects: Cube A, Hemi A, Hemi B, Cube B ---
        let configs: &[(&str, MeshId, glam::Vec3, [f32; 3])] = &[
            ("Cube A",   cube_id, glam::vec3(-4.5, 0.0,  0.0), [0.35, 0.55, 0.95]),
            ("Hemi A",   hemi_id, glam::vec3(-1.5, 0.0,  0.0), [0.35, 0.80, 0.50]),
            ("Hemi B",   hemi_id, glam::vec3( 1.5, 0.0,  0.0), [0.95, 0.60, 0.30]),
            ("Cube B",   cube_id, glam::vec3( 4.5, 0.0,  0.0), [0.75, 0.40, 0.90]),
        ];

        for (name, mesh_id, pos, color) in configs {
            let transform = glam::Mat4::from_translation(*pos);
            let mat = Material::from_color(*color);
            let node_id = self.pl_scene.add_named(name, Some(*mesh_id), transform, mat);
            self.pl_node_names.push((node_id, name.to_string()));
        }

        // --- Point cloud: 6x5 grid above the scene ---
        let mut pc: Vec<[f32; 3]> = Vec::new();
        for row in 0..5_i32 {
            for col in 0..6_i32 {
                pc.push([
                    (col - 2) as f32 * 1.6,
                    3.5 + (row - 2) as f32 * 0.7,
                    0.0,
                ]);
            }
        }
        self.pl_pc_positions = pc;

        // --- Volume: 16x16x16 sphere-shaped scalar field ---
        let dims: [u32; 3] = [16, 16, 16];
        let n = (dims[0] * dims[1] * dims[2]) as usize;
        let mut vol_data = vec![0.0_f32; n];
        let cx = 7.5_f32;
        let cy = 7.5_f32;
        let cz = 7.5_f32;
        let radius = 7.5_f32;
        for iz in 0..dims[2] {
            for iy in 0..dims[1] {
                for ix in 0..dims[0] {
                    let flat = (ix + iy * dims[0] + iz * dims[0] * dims[1]) as usize;
                    let dx = ix as f32 + 0.5 - cx;
                    let dy = iy as f32 + 0.5 - cy;
                    let dz = iz as f32 + 0.5 - cz;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    vol_data[flat] = (1.0 - dist / radius).max(0.0);
                }
            }
        }
        let vol_id = renderer
            .resources_mut()
            .upload_volume(&self.device, &self.queue, &vol_data, dims);
        self.pl_volume_id = Some(vol_id);
        self.pl_volume_data = Some(VolumeData {
            data: vol_data,
            dims,
            origin: [0.0, 0.0, 0.0],
            spacing: [0.25, 0.25, 0.25],
        });

        self.pl_built = true;
    }
}

// ---------------------------------------------------------------------------
// Click handler
// ---------------------------------------------------------------------------

impl App {
    /// Handle a left-click pick.
    ///
    /// `shift` : when true, add/toggle instead of replace.
    pub(crate) fn handle_pl_click(&mut self, pos: glam::Vec2, w: f32, h: f32, shift: bool) {
        let vp_size = glam::Vec2::new(w, h);
        let view_proj = self.camera.view_proj_matrix();
        let vp_inv = view_proj.inverse();
        let (ray_origin, ray_dir) =
            viewport_lib::picking::screen_to_ray(pos, vp_size, vp_inv);

        let mesh_lookup = self.pl_pick_mesh_lookup();

        match self.pl_level {
            PlPickLevel::Object => {
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin, ray_dir, &self.pl_scene, &mesh_lookup,
                );
                if let Some(hit) = hit {
                    if shift {
                        self.pl_selection.toggle(hit.id);
                    } else {
                        self.pl_selection.select_one(hit.id);
                    }
                    self.pl_sub_selection.clear();
                    self.pl_last_hit = Some(PlHitInfo {
                        object_name: self.pl_node_name(hit.id),
                        world_pos: hit.world_pos,
                        normal: hit.normal,
                        sub_object: None,
                        scalar_value: None,
                    });
                    self.pl_hit_marker = Some(hit.world_pos);
                } else if !shift {
                    self.pl_selection.clear();
                    self.pl_sub_selection.clear();
                    self.pl_last_hit = None;
                    self.pl_hit_marker = None;
                }
            }

            PlPickLevel::Face => {
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin, ray_dir, &self.pl_scene, &mesh_lookup,
                );
                if let Some(hit) = hit {
                    if let Some(sub) = hit.sub_object {
                        if shift {
                            self.pl_sub_selection.toggle(hit.id, sub);
                        } else {
                            self.pl_selection.clear();
                            self.pl_sub_selection.select_one(hit.id, sub);
                        }
                    }
                    self.pl_last_hit = Some(PlHitInfo {
                        object_name: self.pl_node_name(hit.id),
                        world_pos: hit.world_pos,
                        normal: hit.normal,
                        sub_object: hit.sub_object,
                        scalar_value: None,
                    });
                    self.pl_hit_marker = Some(hit.world_pos);
                } else if !shift {
                    self.pl_selection.clear();
                    self.pl_sub_selection.clear();
                    self.pl_last_hit = None;
                    self.pl_hit_marker = None;
                }
            }

            PlPickLevel::Vertex => {
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin, ray_dir, &self.pl_scene, &mesh_lookup,
                );
                if let Some(hit) = hit {
                    let vertex_sub = self.pl_nearest_vertex(&hit);
                    if let Some(sub) = vertex_sub {
                        if shift {
                            self.pl_sub_selection.toggle(hit.id, sub);
                        } else {
                            self.pl_selection.clear();
                            self.pl_sub_selection.select_one(hit.id, sub);
                        }
                    }
                    let marker = vertex_sub
                        .and_then(|s| self.pl_vertex_world_pos(hit.id, s))
                        .unwrap_or(hit.world_pos);
                    self.pl_last_hit = Some(PlHitInfo {
                        object_name: self.pl_node_name(hit.id),
                        world_pos: marker,
                        normal: hit.normal,
                        sub_object: vertex_sub,
                        scalar_value: None,
                    });
                    self.pl_hit_marker = Some(marker);
                } else if !shift {
                    self.pl_selection.clear();
                    self.pl_sub_selection.clear();
                    self.pl_last_hit = None;
                    self.pl_hit_marker = None;
                }
            }

            PlPickLevel::Voxel => {
                // Build the VolumeItem matching what the renderer submits.
                let hit = self.pl_volume_id.zip(self.pl_volume_data.as_ref()).and_then(|(vol_id, vol_data)| {
                    let mut item = viewport_lib::VolumeItem::default();
                    item.volume_id = vol_id;
                    item.model = glam::Mat4::from_translation(glam::vec3(-2.0, -1.0, -6.0))
                        .to_cols_array_2d();
                    item.bbox_min = [0.0, 0.0, 0.0];
                    item.bbox_max = [4.0, 4.0, 4.0];
                    item.scalar_range = (0.0, 1.0);
                    item.threshold_min = 0.15;
                    item.threshold_max = 1.0;
                    viewport_lib::pick_volume_cpu(ray_origin, ray_dir, 2, &item, vol_data)
                });

                if let Some(hit) = hit {
                    let sub = hit.sub_object.unwrap();
                    if shift {
                        self.pl_sub_selection.toggle(2, sub);
                    } else {
                        self.pl_selection.clear();
                        self.pl_sub_selection.select_one(2, sub);
                    }
                    self.pl_last_hit = Some(PlHitInfo {
                        object_name: "Volume".to_string(),
                        world_pos: hit.world_pos,
                        normal: hit.normal,
                        sub_object: hit.sub_object,
                        scalar_value: hit.scalar_value,
                    });
                    self.pl_hit_marker = Some(hit.world_pos);
                } else if !shift {
                    self.pl_selection.clear();
                    self.pl_sub_selection.clear();
                    self.pl_last_hit = None;
                    self.pl_hit_marker = None;
                }
            }

            PlPickLevel::Point => {
                let half = 12.0_f32;
                let rect_min = pos - glam::Vec2::splat(half);
                let rect_max = pos + glam::Vec2::splat(half);

                let mut pc_item = viewport_lib::PointCloudItem::default();
                pc_item.positions = self.pl_pc_positions.clone();
                pc_item.id = 1;

                let result = viewport_lib::picking::pick_rect(
                    rect_min, rect_max,
                    &[],
                    &HashMap::new(),
                    std::slice::from_ref(&pc_item),
                    view_proj,
                    vp_size,
                );

                let nearest = result.hits.get(&1).and_then(|subs| {
                    subs.iter()
                        .filter_map(|s| if let SubObjectRef::Point(i) = s { Some(*i) } else { None })
                        .min_by_key(|&i| {
                            let p = glam::Vec3::from(self.pl_pc_positions[i as usize]);
                            let clip = view_proj * p.extend(1.0);
                            let ndc = if clip.w > 0.0 {
                                glam::Vec2::new(clip.x / clip.w, -clip.y / clip.w)
                            } else {
                                glam::Vec2::ZERO
                            };
                            let screen = (ndc * 0.5 + glam::Vec2::splat(0.5)) * vp_size;
                            ((screen - pos).length() * 1000.0) as u32
                        })
                });

                if let Some(pt_idx) = nearest {
                    let sub = SubObjectRef::Point(pt_idx);
                    if shift {
                        self.pl_sub_selection.toggle(1, sub);
                    } else {
                        self.pl_selection.clear();
                        self.pl_sub_selection.select_one(1, sub);
                    }
                    let world_pos = glam::Vec3::from(self.pl_pc_positions[pt_idx as usize]);
                    self.pl_last_hit = Some(PlHitInfo {
                        object_name: "Point Cloud".to_string(),
                        world_pos,
                        normal: glam::Vec3::Y,
                        sub_object: Some(sub),
                        scalar_value: None,
                    });
                    self.pl_hit_marker = Some(world_pos);
                } else if !shift {
                    self.pl_selection.clear();
                    self.pl_sub_selection.clear();
                    self.pl_last_hit = None;
                    self.pl_hit_marker = None;
                }
            }
        }
    }

    /// Handle a rubber-band box selection.
    pub(crate) fn handle_pl_box_select(
        &mut self,
        rect_min: glam::Vec2,
        rect_max: glam::Vec2,
        w: f32,
        h: f32,
        shift: bool,
    ) {
        let vp_size = glam::Vec2::new(w, h);
        let view_proj = self.camera.view_proj_matrix();

        // Normalise rect so min < max
        let r_min = glam::Vec2::new(rect_min.x.min(rect_max.x), rect_min.y.min(rect_max.y));
        let r_max = glam::Vec2::new(rect_min.x.max(rect_max.x), rect_min.y.max(rect_max.y));

        if !shift {
            self.pl_selection.clear();
            self.pl_sub_selection.clear();
        }

        match self.pl_level {
            PlPickLevel::Object => {
                // Use box_select (object-level by position).
                let objects: Vec<&dyn viewport_lib::traits::ViewportObject> =
                    self.pl_scene.nodes().map(|n| n as &dyn viewport_lib::traits::ViewportObject).collect();
                let hits = viewport_lib::picking::box_select(
                    r_min, r_max, &objects, view_proj, vp_size,
                );
                for id in hits {
                    self.pl_selection.add(id);
                }
            }

            PlPickLevel::Face => {
                let mesh_lookup_usize = self.pl_pick_mesh_lookup_usize();
                let scene_items = self.pl_scene.collect_render_items(&viewport_lib::Selection::new());
                let result = viewport_lib::picking::pick_rect(
                    r_min, r_max,
                    &scene_items,
                    &mesh_lookup_usize,
                    &[],
                    view_proj,
                    vp_size,
                );
                for (obj_id, subs) in &result.hits {
                    for &sub in subs {
                        if sub.is_face() {
                            self.pl_sub_selection.add(*obj_id, sub);
                        }
                    }
                }
            }

            PlPickLevel::Vertex => {
                // Collect all vertices that project into the rect.
                for node in self.pl_scene.nodes() {
                    let node_id = viewport_lib::traits::ViewportObject::id(node);
                    let mesh_key = viewport_lib::traits::ViewportObject::mesh_id(node);
                    let model = viewport_lib::traits::ViewportObject::model_matrix(node);
                    if let Some(key) = mesh_key {
                        if let Some((positions, _)) = self.pl_mesh_lookup.get(&key) {
                            for (vi, pos) in positions.iter().enumerate() {
                                let world = model.transform_point3(glam::Vec3::from(*pos));
                                let clip = view_proj * world.extend(1.0);
                                if clip.w <= 0.0 {
                                    continue;
                                }
                                let ndc = glam::Vec2::new(clip.x / clip.w, -clip.y / clip.w);
                                let screen = (ndc * 0.5 + glam::Vec2::splat(0.5)) * vp_size;
                                if screen.x >= r_min.x && screen.x <= r_max.x
                                    && screen.y >= r_min.y && screen.y <= r_max.y
                                {
                                    self.pl_sub_selection.add(node_id, SubObjectRef::Vertex(vi as u32));
                                }
                            }
                        }
                    }
                }
            }

            PlPickLevel::Point => {
                // Project point cloud positions into the rect.
                for (i, pos) in self.pl_pc_positions.iter().enumerate() {
                    let world = glam::Vec3::from(*pos);
                    let clip = view_proj * world.extend(1.0);
                    if clip.w <= 0.0 {
                        continue;
                    }
                    let ndc = glam::Vec2::new(clip.x / clip.w, -clip.y / clip.w);
                    let screen = (ndc * 0.5 + glam::Vec2::splat(0.5)) * vp_size;
                    if screen.x >= r_min.x && screen.x <= r_max.x
                        && screen.y >= r_min.y && screen.y <= r_max.y
                    {
                        self.pl_sub_selection.add(1, SubObjectRef::Point(i as u32));
                    }
                }
            }

            PlPickLevel::Voxel => {
                // Project each above-threshold voxel center to screen space and
                // collect those whose projection falls inside the selection rect.
                if let (Some(vol_id), Some(vol_data)) =
                    (self.pl_volume_id, self.pl_volume_data.as_ref())
                {
                    let _ = vol_id; // id carried by VolumeItem, not needed here
                    let model = glam::Mat4::from_translation(glam::vec3(-2.0, -1.0, -6.0));
                    let bbox_min = glam::Vec3::ZERO;
                    let bbox_max = glam::Vec3::splat(4.0);
                    let [nx, ny, nz] = vol_data.dims;
                    let cell = (bbox_max - bbox_min)
                        / glam::Vec3::new(nx as f32, ny as f32, nz as f32);
                    let threshold_min = 0.15_f32;
                    let threshold_max = 1.0_f32;

                    for iz in 0..nz {
                        for iy in 0..ny {
                            for ix in 0..nx {
                                let flat = ix + iy * nx + iz * nx * ny;
                                let scalar = vol_data.data[flat as usize];
                                if scalar.is_nan()
                                    || scalar < threshold_min
                                    || scalar > threshold_max
                                {
                                    continue;
                                }
                                let local_center = bbox_min
                                    + cell * glam::Vec3::new(
                                        ix as f32 + 0.5,
                                        iy as f32 + 0.5,
                                        iz as f32 + 0.5,
                                    );
                                let world = model.transform_point3(local_center);
                                let clip = view_proj * world.extend(1.0);
                                if clip.w <= 0.0 {
                                    continue;
                                }
                                let ndc = glam::Vec2::new(
                                    clip.x / clip.w,
                                    -clip.y / clip.w,
                                );
                                let screen =
                                    (ndc * 0.5 + glam::Vec2::splat(0.5)) * vp_size;
                                if screen.x >= r_min.x
                                    && screen.x <= r_max.x
                                    && screen.y >= r_min.y
                                    && screen.y <= r_max.y
                                {
                                    self.pl_sub_selection
                                        .add(2, SubObjectRef::Voxel(flat));
                                }
                            }
                        }
                    }
                }
            }
        }

        self.pl_last_hit = None;
        self.pl_hit_marker = None;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl App {
    /// Mesh lookup map for `pick_scene_nodes_cpu` (key = u64 mesh id).
    pub(crate) fn pl_pick_mesh_lookup(&self) -> HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)> {
        self.pl_mesh_lookup.clone()
    }

    /// Mesh lookup map for `pick_rect` (key = usize mesh index).
    fn pl_pick_mesh_lookup_usize(&self) -> HashMap<usize, (Vec<[f32; 3]>, Vec<u32>)> {
        self.pl_mesh_lookup
            .iter()
            .map(|(&k, v)| (k as usize, v.clone()))
            .collect()
    }

    /// Display name for a node ID.
    pub(crate) fn pl_node_name(&self, id: NodeId) -> String {
        self.pl_node_names
            .iter()
            .find(|(nid, _)| *nid == id)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| format!("id={id}"))
    }

    /// World-space model matrix for a node ID.
    pub(crate) fn pl_node_model_matrix(&self, id: NodeId) -> glam::Mat4 {
        for node in self.pl_scene.nodes() {
            if viewport_lib::traits::ViewportObject::id(node) == id {
                return viewport_lib::traits::ViewportObject::model_matrix(node);
            }
        }
        glam::Mat4::IDENTITY
    }

    /// Mesh key (u64) for a node ID.
    fn pl_node_mesh_key(&self, id: NodeId) -> Option<u64> {
        for node in self.pl_scene.nodes() {
            if viewport_lib::traits::ViewportObject::id(node) == id {
                return viewport_lib::traits::ViewportObject::mesh_id(node);
            }
        }
        None
    }

    /// Compute nearest vertex sub-object for a face-level ray hit.
    pub(crate) fn pl_nearest_vertex(&self, hit: &viewport_lib::picking::PickHit) -> Option<SubObjectRef> {
        let face_raw = match hit.sub_object {
            Some(SubObjectRef::Face(i)) => i as usize,
            _ => return None,
        };
        let key = self.pl_node_mesh_key(hit.id)?;
        let (positions, indices) = self.pl_mesh_lookup.get(&key)?;
        let n_tri = indices.len() / 3;
        if n_tri == 0 {
            return None;
        }
        let face = if face_raw >= n_tri { face_raw - n_tri } else { face_raw };
        if face * 3 + 2 >= indices.len() {
            return None;
        }
        let vi = [
            indices[face * 3] as usize,
            indices[face * 3 + 1] as usize,
            indices[face * 3 + 2] as usize,
        ];
        let model = self.pl_node_model_matrix(hit.id);
        let (best_vi, _) = vi
            .iter()
            .map(|&i| {
                let p = model.transform_point3(glam::Vec3::from(positions[i]));
                (i, p.distance(hit.world_pos))
            })
            .fold((vi[0], f32::MAX), |best, (i, d)| {
                if d < best.1 { (i, d) } else { best }
            });
        Some(SubObjectRef::Vertex(best_vi as u32))
    }

    /// World-space position for a Vertex sub-object.
    pub(crate) fn pl_vertex_world_pos(&self, node_id: NodeId, sub: SubObjectRef) -> Option<glam::Vec3> {
        if let SubObjectRef::Vertex(vi) = sub {
            let key = self.pl_node_mesh_key(node_id)?;
            let (positions, _) = self.pl_mesh_lookup.get(&key)?;
            let pos = positions.get(vi as usize)?;
            let model = self.pl_node_model_matrix(node_id);
            Some(model.transform_point3(glam::Vec3::from(*pos)))
        } else {
            None
        }
    }

}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn controls_pick_levels(&mut self, ui: &mut egui::Ui) {
        ui.label("Pick Level:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.pl_level, PlPickLevel::Object, "Object");
            ui.radio_value(&mut self.pl_level, PlPickLevel::Face,   "Face");
            ui.radio_value(&mut self.pl_level, PlPickLevel::Vertex, "Vertex");
            ui.radio_value(&mut self.pl_level, PlPickLevel::Point,  "Point");
            ui.radio_value(&mut self.pl_level, PlPickLevel::Voxel,  "Voxel");
        });

        ui.separator();
        ui.checkbox(&mut self.pl_wireframe, "Wireframe overlay");

        if ui.button("Clear selection").clicked() {
            self.pl_selection.clear();
            self.pl_sub_selection.clear();
            self.pl_last_hit = None;
            self.pl_hit_marker = None;
        }

        ui.separator();
        ui.label(egui::RichText::new("Controls").strong());
        ui.label("Click: select at level");
        ui.label("Shift+click: add / toggle");
        ui.label("Drag: box select");

        ui.separator();

        if let Some(hit) = &self.pl_last_hit {
            ui.label(egui::RichText::new("Last Hit").strong());
            ui.label(format!("Object: {}", hit.object_name));
            match hit.sub_object {
                None                        => { ui.label("Level: Object"); }
                Some(SubObjectRef::Face(i)) => { ui.label(format!("Level: Face #{i}")); }
                Some(SubObjectRef::Vertex(i))=>{ ui.label(format!("Level: Vertex #{i}")); }
                Some(SubObjectRef::Edge(i)) => { ui.label(format!("Level: Edge #{i}")); }
                Some(SubObjectRef::Point(i))=> { ui.label(format!("Level: Point #{i}")); }
                Some(SubObjectRef::Voxel(i))=> { ui.label(format!("Level: Voxel #{i}")); }
                Some(_)                     => { ui.label("Level: (unknown)"); }
            }
            let p = hit.world_pos;
            ui.label(format!("Pos: ({:.2}, {:.2}, {:.2})", p.x, p.y, p.z));
            let n = hit.normal;
            ui.label(format!("Normal: ({:.2}, {:.2}, {:.2})", n.x, n.y, n.z));
            if let Some(sv) = hit.scalar_value {
                ui.label(format!("Scalar: {sv:.3}"));
            }
        } else {
            let sel_count = self.pl_selection.len()
                + self.pl_sub_selection.iter().count();
            if sel_count > 0 {
                ui.label(format!("{sel_count} item(s) selected"));
            } else {
                ui.label("Click or drag to select.");
            }
        }
    }
}
