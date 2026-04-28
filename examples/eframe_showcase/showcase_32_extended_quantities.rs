//! Showcase 32: Extended Quantity Coverage (Phase 15)
//!
//! Three sub-modes demonstrating the Phase 15 additions:
//!
//! **A — Edge/Halfedge/Corner Scalars**: Three spheres coloured by the Z coordinate
//! of each edge midpoint (Edge) or corner (Halfedge/Corner).  Edge is vertex-averaged
//! (smooth gradient); Halfedge and Corner are flat per triangle corner.
//!
//! **B — Volume Mesh Vectors**: Hex-sphere boundary surface with radial arrows placed
//! at every vertex (blue) and every cell centroid (orange).
//!
//! **C — Point Cloud Radius + Transparency**: Fibonacci sphere where per-point radius
//! is large near the equator and per-point transparency follows a sinusoidal longitude
//! pattern, producing transparent stripes.

use crate::App;
use eframe::egui;
use viewport_lib::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColormap, ColormapId, GlyphItem,
    PointCloudItem, SceneRenderItem, TET_SENTINEL, ViewportRenderer, VolumeMeshData,
    volume_mesh_cell_vectors_to_glyphs, volume_mesh_vertex_vectors_to_glyphs,
};

// ---------------------------------------------------------------------------
// Sub-mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EqSubMode {
    EdgeCornerScalars,
    VolumeMeshVectors,
    PointCloudRadiusTransparency,
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn edge_scalars(positions: &[[f32; 3]], indices: &[u32]) -> Vec<f32> {
    let n_tris = indices.len() / 3;
    let mut out = Vec::with_capacity(3 * n_tris);
    for t in 0..n_tris {
        for k in 0..3 {
            let i0 = indices[t * 3 + k] as usize;
            let i1 = indices[t * 3 + (k + 1) % 3] as usize;
            out.push((positions[i0][2] + positions[i1][2]) * 0.5);
        }
    }
    out
}

/// Per-corner scalars that cycle (0.0, 0.5, 1.0) across the three corners of
/// every triangle.  Within each triangle the value interpolates, but neighbouring
/// triangles restart at the same pattern, making the triangulation clearly visible
/// and looking different from the smooth Edge gradient.
fn corner_index_scalars(indices: &[u32]) -> Vec<f32> {
    let n_tris = indices.len() / 3;
    let mut out = Vec::with_capacity(3 * n_tris);
    for _ in 0..n_tris {
        out.push(0.0);
        out.push(0.5);
        out.push(1.0);
    }
    out
}

fn make_hex_sphere_volume_mesh() -> VolumeMeshData {
    let n = 5usize; // 5x5x5 grid – 125 vertices, 64 cells
    let r = 2.0f32;
    let h = 1.0f32;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut idx_map = std::collections::HashMap::new();
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let lx = -h + (ix as f32) / (n as f32 - 1.0) * 2.0 * h;
                let ly = -h + (iy as f32) / (n as f32 - 1.0) * 2.0 * h;
                let lz = -h + (iz as f32) / (n as f32 - 1.0) * 2.0 * h;
                let len = (lx * lx + ly * ly + lz * lz).sqrt().max(1e-9);
                idx_map.insert((ix, iy, iz), positions.len() as u32);
                positions.push([lx / len * r, ly / len * r, lz / len * r]);
            }
        }
    }
    let nc = n - 1;
    let mut cells: Vec<[u32; 8]> = Vec::new();
    for iz in 0..nc {
        for iy in 0..nc {
            for ix in 0..nc {
                let v = |dx, dy, dz| *idx_map.get(&(ix + dx, iy + dy, iz + dz)).unwrap();
                cells.push([
                    v(0,0,0), v(1,0,0), v(1,1,0), v(0,1,0),
                    v(0,0,1), v(1,0,1), v(1,1,1), v(0,1,1),
                ]);
            }
        }
    }
    let mut data = VolumeMeshData::default();
    data.positions = positions;
    data.cells = cells;
    data
}

fn vertex_radial_vectors(positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
    positions.iter().map(|&p| {
        let len = (p[0]*p[0]+p[1]*p[1]+p[2]*p[2]).sqrt().max(1e-9);
        [p[0]/len, p[1]/len, p[2]/len]
    }).collect()
}

fn cell_radial_vectors(data: &VolumeMeshData) -> Vec<[f32; 3]> {
    data.cells.iter().map(|cell| {
        let valid: Vec<usize> = cell.iter()
            .filter(|&&i| i != TET_SENTINEL && (i as usize) < data.positions.len())
            .map(|&i| i as usize).collect();
        if valid.is_empty() { return [0.0, 1.0, 0.0]; }
        let inv = 1.0 / valid.len() as f32;
        let cx: f32 = valid.iter().map(|&i| data.positions[i][0]).sum::<f32>() * inv;
        let cy: f32 = valid.iter().map(|&i| data.positions[i][1]).sum::<f32>() * inv;
        let cz: f32 = valid.iter().map(|&i| data.positions[i][2]).sum::<f32>() * inv;
        let len = (cx*cx+cy*cy+cz*cz).sqrt().max(1e-9);
        [cx/len, cy/len, cz/len]
    }).collect()
}

fn make_pc_data(n: usize) -> (Vec<[f32; 3]>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let r = 2.0f32;
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    let mut pos = Vec::with_capacity(n);
    let mut sc = Vec::with_capacity(n);
    let mut rad = Vec::with_capacity(n);
    let mut tr = Vec::with_capacity(n);
    for i in 0..n {
        let y = 1.0 - (i as f32) / (n as f32 - 1.0) * 2.0;
        let ry = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * i as f32;
        pos.push([theta.cos() * ry * r, y * r, theta.sin() * ry * r]);
        sc.push((y + 1.0) * 0.5);
        rad.push(3.0 + 27.0 * (1.0 - y * y)); // 3px at poles, 30px at equator
        let t_norm = theta.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
        tr.push((t_norm * 8.0 * std::f32::consts::PI).sin() * 0.5 + 0.5);
    }
    (pos, sc, rad, tr)
}

// ---------------------------------------------------------------------------
// App impl
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_eq_scene(&mut self, renderer: &mut ViewportRenderer) {
        let sphere = viewport_lib::primitives::sphere(2.0, 48, 24);
        let ev = edge_scalars(&sphere.positions, &sphere.indices);
        let cv = corner_index_scalars(&sphere.indices);

        let mut em = viewport_lib::primitives::sphere(2.0, 48, 24);
        em.attributes.insert("edge_z".into(), AttributeData::Edge(ev));
        self.eq_edge_mesh_ids[0] = renderer.resources_mut()
            .upload_mesh_data(&self.device, &em).expect("edge mesh");

        let mut hm = viewport_lib::primitives::sphere(2.0, 48, 24);
        hm.attributes.insert("halfedge_z".into(), AttributeData::Halfedge(cv.clone()));
        self.eq_edge_mesh_ids[1] = renderer.resources_mut()
            .upload_mesh_data(&self.device, &hm).expect("halfedge mesh");

        let mut cm = viewport_lib::primitives::sphere(2.0, 48, 24);
        cm.attributes.insert("corner_z".into(), AttributeData::Corner(cv));
        self.eq_edge_mesh_ids[2] = renderer.resources_mut()
            .upload_mesh_data(&self.device, &cm).expect("corner mesh");

        let vm_data = make_hex_sphere_volume_mesh();
        self.eq_vm_mesh_id = renderer.resources_mut()
            .upload_volume_mesh_data(&self.device, &vm_data).expect("vm mesh");
        self.eq_vm_data = vm_data;

        let (p, s, r, t) = make_pc_data(5_000);
        self.eq_pc_positions = p;
        self.eq_pc_scalars = s;
        self.eq_pc_radii = r;
        self.eq_pc_transp = t;

        // Opaque background sphere slightly smaller than the point-cloud shell (r=2.0).
        // Placed in the middle so back-facing points are occluded.
        let bg_sphere = viewport_lib::primitives::sphere(1.75, 32, 16);
        self.eq_pc_bg_mesh_id = renderer.resources_mut()
            .upload_mesh_data(&self.device, &bg_sphere).expect("pc bg sphere");

        self.eq_built = true;
    }

    pub(crate) fn controls_eq(&mut self, ui: &mut egui::Ui) {
        ui.label("Sub-mode:");
        ui.horizontal_wrapped(|ui| {
            for (label, mode) in [
                ("Edge/Corner Scalars", EqSubMode::EdgeCornerScalars),
                ("Volume Vectors", EqSubMode::VolumeMeshVectors),
                ("PC Radius+Alpha", EqSubMode::PointCloudRadiusTransparency),
            ] {
                if ui.radio(self.eq_sub_mode == mode, label).clicked() {
                    self.eq_sub_mode = mode;
                }
            }
        });
        ui.separator();
        match self.eq_sub_mode {
            EqSubMode::EdgeCornerScalars => {
                ui.label("Left: Edge — smooth gradient (Z midpoints averaged to vertices)");
                ui.label("Centre/Right: Halfedge & Corner — repeating 3-colour triangle pattern");
                ui.label("(corner 0=purple, 1=teal, 2=yellow; discontinuous across edges)");
                ui.separator();
                ui.label("Colormap:");
                for cm in [BuiltinColormap::Viridis, BuiltinColormap::Plasma, BuiltinColormap::Coolwarm] {
                    if ui.radio(self.eq_colormap == cm, format!("{cm:?}")).clicked() {
                        self.eq_colormap = cm;
                    }
                }
            }
            EqSubMode::VolumeMeshVectors => {
                ui.label("Blue arrows: per-vertex radial vectors.");
                ui.label("Orange arrows: per-cell centroid radial vectors.");
            }
            EqSubMode::PointCloudRadiusTransparency => {
                ui.label("Radius: large at equator, small at poles.");
                ui.label("Transparency: sinusoidal longitude stripes.");
            }
        }
    }

    pub(crate) fn eq_scene_items(
        &self,
    ) -> (Vec<SceneRenderItem>, Vec<GlyphItem>, Vec<PointCloudItem>) {
        let mut scene_items: Vec<SceneRenderItem> = Vec::new();
        let mut glyph_items: Vec<GlyphItem> = Vec::new();
        let mut pc_items: Vec<PointCloudItem> = Vec::new();

        match self.eq_sub_mode {
            EqSubMode::EdgeCornerScalars => {
                let cm_id = Some(ColormapId(self.eq_colormap as usize));
                let offsets = [-5.0f32, 0.0, 5.0];
                let names = ["edge_z", "halfedge_z", "corner_z"];
                let kinds = [AttributeKind::Edge, AttributeKind::Halfedge, AttributeKind::Corner];
                for i in 0..3 {
                    let mut item = SceneRenderItem::default();
                    item.mesh_id = self.eq_edge_mesh_ids[i];
                    item.model = glam::Mat4::from_translation(glam::vec3(offsets[i], 0.0, 0.0))
                        .to_cols_array_2d();
                    item.active_attribute = Some(AttributeRef {
                        name: names[i].into(),
                        kind: kinds[i],
                    });
                    item.colormap_id = cm_id;
                    item.visible = true;
                    scene_items.push(item);
                }
            }
            EqSubMode::VolumeMeshVectors => {
                let mut item = SceneRenderItem::default();
                item.mesh_id = self.eq_vm_mesh_id;
                item.visible = true;
                scene_items.push(item);
                // Vertex vectors: blue (Viridis at 0.15).
                let vv = vertex_radial_vectors(&self.eq_vm_data.positions);
                let mut vg = volume_mesh_vertex_vectors_to_glyphs(&self.eq_vm_data.positions, &vv, 0.4);
                vg.scalars = vec![0.15; vg.positions.len()];
                vg.scalar_range = Some((0.0, 1.0));
                vg.colormap_id = Some(ColormapId(BuiltinColormap::Viridis as usize));
                glyph_items.push(vg);
                // Cell vectors: orange (Plasma at 0.65).  Use Plasma to guarantee a
                // warm hue clearly distinct from the blue vertex arrows.
                let cv = cell_radial_vectors(&self.eq_vm_data);
                let mut cg = volume_mesh_cell_vectors_to_glyphs(&self.eq_vm_data, &cv, 1.5);
                cg.scalars = vec![0.65; cg.positions.len()];
                cg.scalar_range = Some((0.0, 1.0));
                cg.colormap_id = Some(ColormapId(BuiltinColormap::Plasma as usize));
                glyph_items.push(cg);
            }
            EqSubMode::PointCloudRadiusTransparency => {
                // Opaque sphere behind the point cloud to occlude back-facing points.
                let mut bg = SceneRenderItem::default();
                bg.mesh_id = self.eq_pc_bg_mesh_id;
                bg.visible = true;
                scene_items.push(bg);

                let mut pc = PointCloudItem::default();
                pc.positions = self.eq_pc_positions.clone();
                pc.scalars = self.eq_pc_scalars.clone();
                pc.radii = self.eq_pc_radii.clone();
                pc.transparencies = self.eq_pc_transp.clone();
                pc_items.push(pc);
            }
        }

        (scene_items, glyph_items, pc_items)
    }
}
