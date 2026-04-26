//! Showcase 25: On-Surface Vector Quantities
//!
//! Demonstrates the three surface-vector APIs from `viewport_lib::quantities`:
//!
//! - **Vertex intrinsic vectors** : a tangential vortex field on a sphere, where
//!   each vertex carries a 2D `(u, v)` vector in its tangent frame.
//! - **Face intrinsic vectors** : a face-level flow field on a torus, where each
//!   triangle carries a 2D vector in its per-face tangent frame.
//! - **Edge one-forms** : a diverging source field on a plane, reconstructed from
//!   per-edge scalar values via Whitney form interpolation.
//!
//! All three modes produce a [`GlyphItem`] (arrows) submitted to
//! `SceneFrame::glyphs` each frame. No new GPU pipeline is needed.

use crate::App;
use eframe::egui;
use viewport_lib::{
    GlyphItem, LightingSettings, MeshData, SceneRenderItem,
    quantities::{edge_one_form_to_glyphs, face_intrinsic_to_glyphs, vertex_intrinsic_to_glyphs},
};

// ---------------------------------------------------------------------------
// Sub-mode
// ---------------------------------------------------------------------------

/// Which quantity type is currently displayed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SvMode {
    /// Vertex-indexed intrinsic vectors (vortex field on sphere).
    VertexIntrinsic,
    /// Face-indexed intrinsic vectors (flow field on torus).
    FaceIntrinsic,
    /// Edge one-forms, Whitney-reconstructed (source field on plane).
    EdgeOneForm,
}

// ---------------------------------------------------------------------------
// App state (fields stored in the main App struct, declared here)
// ---------------------------------------------------------------------------
//
// The main App struct holds:
//   sv_built:        bool
//   sv_mode:         SvMode
//   sv_scale:        f32
//   sv_scene:        Scene
//   sv_mesh_index:   [usize; 3]   : [sphere, torus, plane]
//   sv_positions:    [Vec<[f32;3]>; 3]
//   sv_normals:      [Vec<[f32;3]>; 3]
//   sv_tangents:     [Option<Vec<[f32;4]>>; 3]
//   sv_indices:      [Vec<u32>; 3]
//   sv_vertex_vecs:  Vec<[f32; 2]>     : mode 0 data
//   sv_face_vecs:    Vec<[f32; 2]>     : mode 1 data
//   sv_edge_vals:    Vec<f32>          : mode 2 data

impl App {
    // -------------------------------------------------------------------------
    // One-time build
    // -------------------------------------------------------------------------

    /// Upload render meshes (once) and generate initial glyph data.
    pub(crate) fn build_sv_scene(&mut self, renderer: &mut viewport_lib::ViewportRenderer) {
        use viewport_lib::geometry::primitives;

        // Upload render meshes at a fixed resolution (independent of glyph density).
        let sphere = primitives::sphere(1.0, 48, 24);
        let sphere_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("sv sphere");

        let torus = make_torus(1.2, 0.4, 48, 24);
        let torus_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &torus)
            .expect("sv torus");

        let (plane, _) = make_plane_with_source_one_form(20);
        let plane_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &plane)
            .expect("sv plane");

        self.sv_mesh_index = [sphere_idx, torus_idx, plane_idx];
        self.sv_built = true;

        // Generate glyph data at the current density.
        self.rebuild_sv_glyph_data();
    }

    /// Regenerate glyph source data (positions, normals, indices, quantity
    /// vectors) at the current `sv_density`. Called on first build and
    /// whenever the density slider changes.
    pub(crate) fn rebuild_sv_glyph_data(&mut self) {
        use viewport_lib::geometry::primitives;

        let d = self.sv_density;

        // Scale segment counts by density.
        let sphere_lon = (48.0 * d).round().max(6.0) as u32;
        let sphere_lat = (24.0 * d).round().max(3.0) as u32;
        let torus_major = (48.0 * d).round().max(6.0) as usize;
        let torus_minor = (24.0 * d).round().max(3.0) as usize;
        let plane_n = (20.0 * d).round().max(3.0) as usize;

        // --- Sphere (vertex intrinsic) ---
        let sphere = primitives::sphere(1.0, sphere_lon, sphere_lat);
        let vertex_vecs = make_sphere_vortex_intrinsic(&sphere.positions, &sphere.normals);

        // --- Torus (face intrinsic) ---
        let torus = make_torus(1.2, 0.4, torus_major, torus_minor);
        let face_vecs = make_torus_face_vectors(&torus, 1.2);

        // --- Plane (edge one-form) ---
        let (plane, edge_vals) = make_plane_with_source_one_form(plane_n);

        self.sv_positions = [
            sphere.positions.clone(),
            torus.positions.clone(),
            plane.positions.clone(),
        ];
        self.sv_normals = [
            sphere.normals.clone(),
            torus.normals.clone(),
            plane.normals.clone(),
        ];
        self.sv_tangents = [None, None, None];
        self.sv_indices = [
            sphere.indices.clone(),
            torus.indices.clone(),
            plane.indices.clone(),
        ];
        self.sv_vertex_vecs = vertex_vecs;
        self.sv_face_vecs = face_vecs;
        self.sv_edge_vals = edge_vals;
        self.sv_glyph_density = d;
    }

    // -------------------------------------------------------------------------
    // Controls
    // -------------------------------------------------------------------------

    pub(crate) fn controls_surface_vectors(&mut self, ui: &mut egui::Ui) {
        ui.label("Quantity type:");
        for (mode, label) in [
            (SvMode::VertexIntrinsic, "Vertex intrinsic (sphere)"),
            (SvMode::FaceIntrinsic, "Face intrinsic (torus)"),
            (SvMode::EdgeOneForm, "Edge one-form (plane)"),
        ] {
            if ui.radio(self.sv_mode == mode, label).clicked() {
                self.sv_mode = mode;
            }
        }

        ui.separator();
        ui.label("Arrow scale:");
        ui.add(egui::Slider::new(&mut self.sv_scale, 0.01..=1.0).step_by(0.01));

        let count = self.sv_vector_count();
        ui.label("Density:");
        let density_changed = ui
            .add(
                egui::Slider::new(&mut self.sv_density, 0.1..=2.0)
                    .step_by(0.1)
                    .suffix(format!(" ({count} vectors)")),
            )
            .changed();
        if density_changed {
            self.rebuild_sv_glyph_data();
        }

        ui.separator();
        match self.sv_mode {
            SvMode::VertexIntrinsic => {
                ui.label("Vortex tangent field.");
                ui.label("Each vertex carries a 2D (u,v) vector in its tangent frame, forming a pure-rotation flow around the sphere's poles.");
            }
            SvMode::FaceIntrinsic => {
                ui.label("Poloidal flow field.");
                ui.label("Each triangle carries a 2D (u,v) vector in its face tangent frame. The vectors follow the poloidal direction of the torus.");
            }
            SvMode::EdgeOneForm => {
                ui.label("Diverging source field.");
                ui.label("Each directed edge carries a scalar one-form value. Whitney reconstruction recovers a vector field pointing outward from the origin.");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Per-frame helpers
    // -------------------------------------------------------------------------

    /// Mesh surface render item for the active sub-mode.
    pub(crate) fn sv_surface_item(&self) -> SceneRenderItem {
        let mesh_id = match self.sv_mode {
            SvMode::VertexIntrinsic => self.sv_mesh_index[0],
            SvMode::FaceIntrinsic => self.sv_mesh_index[1],
            SvMode::EdgeOneForm => self.sv_mesh_index[2],
        };
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item
    }

    /// Total number of vectors in the full glyph set for the active mode.
    fn sv_total_count(&self) -> usize {
        match self.sv_mode {
            SvMode::VertexIntrinsic => self.sv_vertex_vecs.len(),
            SvMode::FaceIntrinsic => self.sv_face_vecs.len(),
            SvMode::EdgeOneForm => self.sv_indices[2].len() / 3,
        }
    }

    /// Number of vectors that will be shown at the current density.
    pub(crate) fn sv_vector_count(&self) -> usize {
        let total = self.sv_total_count();
        (total as f32 * self.sv_density).ceil().max(1.0) as usize
    }

    /// Build the [`GlyphItem`] for the active sub-mode.
    pub(crate) fn sv_glyph_item(&self) -> GlyphItem {
        match self.sv_mode {
            SvMode::VertexIntrinsic => vertex_intrinsic_to_glyphs(
                &self.sv_positions[0],
                &self.sv_normals[0],
                self.sv_tangents[0].as_deref(),
                &self.sv_vertex_vecs,
                self.sv_scale,
            ),
            SvMode::FaceIntrinsic => face_intrinsic_to_glyphs(
                &self.sv_positions[1],
                &self.sv_normals[1],
                &self.sv_indices[1],
                &self.sv_face_vecs,
                self.sv_scale,
            ),
            SvMode::EdgeOneForm => edge_one_form_to_glyphs(
                &self.sv_positions[2],
                &self.sv_indices[2],
                &self.sv_edge_vals,
                self.sv_scale,
            ),
        }
    }

    /// Standard lighting for Showcase 25.
    pub(crate) fn sv_lighting() -> LightingSettings {
        LightingSettings {
            hemisphere_intensity: 0.5,
            sky_color: [1.0, 1.0, 1.0],
            ground_color: [1.0, 1.0, 1.0],
            ..LightingSettings::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Procedural geometry helpers
// ---------------------------------------------------------------------------

/// Compute per-vertex intrinsic vectors for a vortex field rotating around Z.
///
/// The desired 3D world vector at each vertex is `n × Z` (azimuthal direction).
/// We project it into the Gram-Schmidt tangent frame that the quantities API
/// will use internally (when `tangents` is `None`), so the encoded `(u, v)`
/// round-trips correctly through `vertex_intrinsic_to_glyphs`.
fn make_sphere_vortex_intrinsic(_positions: &[[f32; 3]], normals: &[[f32; 3]]) -> Vec<[f32; 2]> {
    use viewport_lib::quantities::tangent_frames::gram_schmidt_tangent;

    let up = glam::Vec3::Z;
    normals
        .iter()
        .map(|&n| {
            let normal = glam::Vec3::from(n);
            // Azimuthal direction (vortex around Z): normal × Z, projected onto the
            // tangent plane and normalised.
            let az = normal.cross(up);
            let az = (az - az.dot(normal) * normal).normalize_or_zero();

            // Get the same Gram-Schmidt frame the API will use internally.
            let (t, b) = gram_schmidt_tangent(n);
            let tv = glam::Vec3::from(t);
            let bv = glam::Vec3::from(b);

            let u = az.dot(tv);
            let v = az.dot(bv);
            // Leave poles (where the azimuthal vector is degenerate) as zero.
            let len = (u * u + v * v).sqrt();
            if len > 1e-4 {
                [u / len, v / len]
            } else {
                [0.0, 0.0]
            }
        })
        .collect()
}

/// Generate a torus with `major_segs` longitude segments and `minor_segs` latitude
/// (tube) segments. `major_r` is the major radius, `minor_r` the tube radius.
fn make_torus(major_r: f32, minor_r: f32, major_segs: usize, minor_segs: usize) -> MeshData {
    use std::f32::consts::TAU;

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=major_segs {
        let phi = TAU * i as f32 / major_segs as f32;
        let (sp, cp) = phi.sin_cos();
        let center = glam::Vec3::new(cp * major_r, sp * major_r, 0.0);
        // Radial direction (points outward from the torus axis in the XY plane).
        let radial = glam::Vec3::new(cp, sp, 0.0);

        for j in 0..=minor_segs {
            let theta = TAU * j as f32 / minor_segs as f32;
            let (st, ct) = theta.sin_cos();
            // Local normal in the plane of the tube cross-section.
            let local_n = radial * ct + glam::Vec3::Z * st;
            let pos = center + local_n * minor_r;
            positions.push(pos.to_array());
            normals.push(local_n.to_array());
        }
    }

    let stride = minor_segs + 1;
    for i in 0..major_segs {
        for j in 0..minor_segs {
            let a = (i * stride + j) as u32;
            let b = (i * stride + j + 1) as u32;
            let c = ((i + 1) * stride + j + 1) as u32;
            let d = ((i + 1) * stride + j) as u32;
            indices.extend_from_slice(&[a, b, d]);
            indices.extend_from_slice(&[b, c, d]);
        }
    }

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}

/// Compute per-face intrinsic vectors for the torus: flow in the poloidal
/// (tube-circle) direction at each triangle's centroid.
fn make_torus_face_vectors(torus: &MeshData, major_r: f32) -> Vec<[f32; 2]> {
    use glam::Vec3;
    use viewport_lib::quantities::compute_face_tangent_frames;

    let num_tris = torus.indices.len() / 3;
    let frames = compute_face_tangent_frames(&torus.positions, &torus.indices);
    let mut vecs = Vec::with_capacity(num_tris);

    for tri in 0..num_tris {
        let i0 = torus.indices[3 * tri] as usize;
        let i1 = torus.indices[3 * tri + 1] as usize;
        let i2 = torus.indices[3 * tri + 2] as usize;
        let centroid = (Vec3::from(torus.positions[i0])
            + Vec3::from(torus.positions[i1])
            + Vec3::from(torus.positions[i2]))
            / 3.0;

        // Poloidal direction: tangent to the circle of major radius at this point.
        // For a torus with axis Z and major circle in XY:
        //   radial direction = (cx, cy, 0) / major_r  (from tube centre projected to XY)
        //   poloidal tangent = (-ry, rx, 0)             (90° rotation of radial in XY)
        let tube_center_xy = Vec3::new(centroid.x, centroid.y, 0.0);
        let poloidal = if tube_center_xy.length_squared() > 1e-6 {
            let radial = tube_center_xy.normalize() * major_r;
            Vec3::new(-radial.y / major_r, radial.x / major_r, 0.0)
        } else {
            Vec3::X
        };

        // Project poloidal into the face tangent frame.
        let (t, b) = frames[tri];
        let tg = Vec3::from(t);
        let bg = Vec3::from(b);
        let u = poloidal.dot(tg);
        let v = poloidal.dot(bg);
        // Normalise so all arrows have equal length.
        let len = (u * u + v * v).sqrt();
        if len > 1e-6 {
            vecs.push([u / len, v / len]);
        } else {
            vecs.push([0.0, 0.0]);
        }
    }

    vecs
}

/// Generate an N×N quad grid plane in the XY plane (Z=0) and a one-form that
/// encodes a diverging source field centred at the origin.
///
/// Returns `(mesh, edge_values)` where `edge_values` has 3 entries per triangle
/// in the order `(v0->v1, v1->v2, v2->v0)`.
fn make_plane_with_source_one_form(n: usize) -> (MeshData, Vec<f32>) {
    use glam::Vec3;

    // Fixed world-space extent: plane always spans [-10, 10] on each axis.
    let extent = 10.0_f32;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Grid of (n+1)×(n+1) vertices in the XY plane.
    for row in 0..=(n as i32) {
        for col in 0..=(n as i32) {
            let x = (col as f32 / n as f32 - 0.5) * 2.0 * extent;
            let y = (row as f32 / n as f32 - 0.5) * 2.0 * extent;
            positions.push([x, y, 0.0_f32]);
            normals.push([0.0_f32, 0.0, 1.0]);
        }
    }

    let stride = (n + 1) as u32;
    for row in 0..(n as u32) {
        for col in 0..(n as u32) {
            let a = row * stride + col;
            let b = row * stride + col + 1;
            let c = (row + 1) * stride + col + 1;
            let d = (row + 1) * stride + col;
            // Two triangles per quad: (a,b,d) and (b,c,d)
            indices.extend_from_slice(&[a, b, d]);
            indices.extend_from_slice(&[b, c, d]);
        }
    }

    // One-form: the integral of the radial field F = (x, y, 0) over each edge is
    //   w(i->j) = ∫ F·dl = 0.5 * (F(pi) + F(pj)) · (pj - pi)   (midpoint rule)
    let num_tris = indices.len() / 3;
    let mut edge_values = Vec::with_capacity(3 * num_tris);

    for tri in 0..num_tris {
        let i0 = indices[3 * tri] as usize;
        let i1 = indices[3 * tri + 1] as usize;
        let i2 = indices[3 * tri + 2] as usize;
        let p0 = Vec3::from(positions[i0]);
        let p1 = Vec3::from(positions[i1]);
        let p2 = Vec3::from(positions[i2]);

        // Edge integrals for F = (x, y, 0).
        let w01 = edge_line_integral(p0, p1);
        let w12 = edge_line_integral(p1, p2);
        let w20 = edge_line_integral(p2, p0);

        edge_values.push(w01);
        edge_values.push(w12);
        edge_values.push(w20);
    }

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;

    (mesh, edge_values)
}

/// Line integral of the radial field F=(x,y,0) along the straight edge (p->q):
/// ∫₀¹ F(p + t(q-p))·(q-p) dt = 0.5 * (F(p)+F(q))·(q-p)
fn edge_line_integral(p: glam::Vec3, q: glam::Vec3) -> f32 {
    let fp = glam::Vec3::new(p.x, p.y, 0.0);
    let fq = glam::Vec3::new(q.x, q.y, 0.0);
    let dq = q - p;
    0.5 * (fp + fq).dot(dq)
}
