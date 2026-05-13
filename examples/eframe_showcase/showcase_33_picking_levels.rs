//! Showcase 33: Picking Levels
//!
//! Demonstrates the unified picking API (renderer.pick / renderer.pick_rect)
//! and per-type picking across multiple levels.
//!
//! Scene: two cubes and two hemispheres; an 80-point grid cloud; a scalar
//! volume; a 3x3 Gaussian splat grid; a hex capsule mesh; a hex cylinder
//! transparent volume mesh; spiral streamline polylines; arrow and tensor
//! glyph sets; and sprites.
//!
//! Controls:
//!   - Left-click : select at the active level
//!   - Shift + click : add / toggle selection
//!   - Left-drag : rubber-band box selection
//!   - Wireframe toggle in the side panel (on by default)

use std::collections::HashMap;

use eframe::egui;
use viewport_lib::{
    GaussianSplatData, GaussianSplatId,
    Material, MeshId, NodeId, PickMask, PickRectResult, ProjectedTetId,
    ShDegree, SubObjectRef, VolumeData, VolumeMeshData, ViewportRenderer,
};

use crate::App;

// ---------------------------------------------------------------------------
// Pick level (per-type section)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PlPickLevel {
    #[default]
    Object,
    Face,
    Vertex,
    Point,
    Voxel,
    /// Gaussian splat pick (returns SubObjectRef::Point).
    Splat,
    /// Transparent volume mesh cell pick (returns SubObjectRef::Cell).
    Tvm,
}

// ---------------------------------------------------------------------------
// Unified mask options (top section)
// ---------------------------------------------------------------------------

/// Which PickMask to use in the unified section.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PlUnifiedMask {
    #[default]
    Object,
    Face,
    PointLike,
    EdgeLike,
    FaceAndPointLike,
}

impl PlUnifiedMask {
    pub(crate) fn to_pick_mask(self) -> PickMask {
        match self {
            Self::Object           => PickMask::OBJECT,
            Self::Face             => PickMask::FACE,
            Self::PointLike        => PickMask::POINT_LIKE,
            Self::EdgeLike         => PickMask::EDGE_LIKE,
            Self::FaceAndPointLike => PickMask::FACE | PickMask::POINT_LIKE,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Object           => "Object",
            Self::Face             => "Face",
            Self::PointLike        => "Point-like",
            Self::EdgeLike         => "Edge-like",
            Self::FaceAndPointLike => "Face + Point-like",
        }
    }
}

// ---------------------------------------------------------------------------
// Hit info (last single-click result displayed in the panel)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct PlHitInfo {
    pub object_name: String,
    pub object_type: &'static str,
    pub world_pos: glam::Vec3,
    pub normal: glam::Vec3,
    pub sub_object: Option<SubObjectRef>,
    pub scalar_value: Option<f32>,
}


// ---------------------------------------------------------------------------
// Splat and TVM data helpers
// ---------------------------------------------------------------------------

/// Build `GaussianSplatData` for the picking demo: N small spherical splats,
/// colored from teal to orange across the set.
fn make_pl_splat_data(positions: &[[f32; 3]]) -> GaussianSplatData {
    const SH0_C: f32 = 0.28209479177;
    let n = positions.len();
    let sh_coefficients = (0..n)
        .flat_map(|i| {
            let t = i as f32 / (n as f32 - 1.0).max(1.0);
            let r = t;
            let g = 0.5_f32;
            let b = 1.0 - t;
            [(r - 0.5) / SH0_C, (g - 0.5) / SH0_C, (b - 0.5) / SH0_C]
        })
        .collect();
    GaussianSplatData {
        positions: positions.to_vec(),
        scales: vec![[0.2, 0.2, 0.2]; n],
        rotations: vec![[1.0, 0.0, 0.0, 0.0]; n],
        opacities: vec![0.8; n],
        sh_coefficients,
        sh_degree: ShDegree::Zero,
    }
}

/// Build a hex mesh of a capsule (pill) shape.
///
/// The pill is centered at (0, -3.5, 0) and elongated along world Y.
/// Uses a 5x5 cross-section grid with an elliptic square-to-disk mapping,
/// giving 128 hex cells across 8 axial layers.
fn make_pl_tvm_data() -> VolumeMeshData {
    const N: usize = 5;   // cross-section grid side: (N-1)^2 cells per slice
    const NZ: usize = 9;  // axial vertex count: NZ-1 cell layers

    let radius = 0.7_f32;
    let cyl_half = 1.2_f32;
    let center = [4.0, 0.0, -4.0];
    let z_extent = cyl_half + radius;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(N * N * NZ);

    for k in 0..NZ {
        let t = k as f32 / (NZ - 1) as f32;
        let axial = -z_extent + 2.0 * z_extent * t;

        // Cross-sectional radius: cylinder in the middle, spherical caps at the ends.
        let r = if axial.abs() <= cyl_half {
            radius
        } else {
            let dz = axial.abs() - cyl_half;
            (radius * radius - dz * dz).max(0.0).sqrt()
        };

        for j in 0..N {
            for i in 0..N {
                let u = 2.0 * i as f32 / (N - 1) as f32 - 1.0;
                let v = 2.0 * j as f32 / (N - 1) as f32 - 1.0;
                // Elliptic disk mapping: corners of [-1,1]^2 map to the unit circle.
                let dx = r * u * (1.0 - v * v / 2.0).sqrt();
                let dz = r * v * (1.0 - u * u / 2.0).sqrt();
                positions.push([center[0] + dx, center[1] + axial, center[2] + dz]);
            }
        }
    }

    let vi = |i: usize, j: usize, k: usize| -> u32 { (i + j * N + k * N * N) as u32 };
    let n_lat = N - 1;
    let n_axial = NZ - 1;
    let total = n_lat * n_lat * n_axial;
    let mut cells: Vec<[u32; 8]> = Vec::with_capacity(total);
    let mut scalars: Vec<f32> = Vec::with_capacity(total);

    for k in 0..n_axial {
        for j in 0..n_lat {
            for i in 0..n_lat {
                cells.push([
                    vi(i,   j,   k  ), vi(i+1, j,   k  ),
                    vi(i+1, j+1, k  ), vi(i,   j+1, k  ),
                    vi(i,   j,   k+1), vi(i+1, j,   k+1),
                    vi(i+1, j+1, k+1), vi(i,   j+1, k+1),
                ]);
                let idx = i + j * n_lat + k * n_lat * n_lat;
                scalars.push(idx as f32 / (total - 1).max(1) as f32);
            }
        }
    }

    let mut data = VolumeMeshData::default();
    data.positions = positions;
    data.cells = cells;
    data.cell_scalars.insert("scalar".to_string(), scalars);
    data
}

/// Build a hex cylinder: 3 rings of 6 hex cells each (18 cells total),
/// centered at (-7, 0, 0) and elongated along Y.
///
/// Used in the transparent-volume-mesh picking demonstration in showcase 33.
fn make_pl_tvm_tet_data() -> VolumeMeshData {
    const CX: f32 = -7.0;
    const CZ: f32 = 0.0;
    const N_SECTORS: usize = 6;
    const N_LAYERS: usize = 3;
    const R_OUTER: f32 = 0.9;
    const HALF_H: f32 = 1.2;

    // Vertex rings: for each axial layer we have an inner ring (collapsed to
    // center) and an outer ring.  With N_SECTORS=6 and N_LAYERS=3 we get
    // (N_LAYERS+1) slices, each with 1 center + N_SECTORS outer = 7 verts.
    let n_slices = N_LAYERS + 1;
    let verts_per_slice = 1 + N_SECTORS;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n_slices * verts_per_slice);

    for sl in 0..n_slices {
        let y = -HALF_H + 2.0 * HALF_H * sl as f32 / N_LAYERS as f32;
        // Center vertex.
        positions.push([CX, y, CZ]);
        // Outer ring.
        for s in 0..N_SECTORS {
            let angle = std::f32::consts::TAU * s as f32 / N_SECTORS as f32;
            positions.push([
                CX + R_OUTER * angle.cos(),
                y,
                CZ + R_OUTER * angle.sin(),
            ]);
        }
    }

    let vi = |slice: usize, local: usize| -> u32 {
        (slice * verts_per_slice + local) as u32
    };

    let total = N_SECTORS * N_LAYERS;
    let mut cells: Vec<[u32; 8]> = Vec::with_capacity(total);
    let mut scalars: Vec<f32> = Vec::with_capacity(total);

    for layer in 0..N_LAYERS {
        let bot = layer;
        let top = layer + 1;
        for s in 0..N_SECTORS {
            let s_next = (s + 1) % N_SECTORS;
            // Each "wedge" from center to outer edge forms a hex cell:
            // bottom face: center_bot, outer_s_bot, outer_s+1_bot, center_bot (degenerate quad)
            // top face:    center_top, outer_s_top, outer_s+1_top, center_top
            // We use a proper hex with the center vertex duplicated to fill the
            // 8-vertex slot (wedge-like hex).
            cells.push([
                vi(bot, 0),         vi(bot, 1 + s),     vi(bot, 1 + s_next), vi(bot, 0),
                vi(top, 0),         vi(top, 1 + s),     vi(top, 1 + s_next), vi(top, 0),
            ]);
            let t = (layer as f32 + 0.5) / N_LAYERS as f32;
            let u = (s as f32 + 0.5) / N_SECTORS as f32;
            scalars.push(t * 0.6 + u * 0.4);
        }
    }

    let mut data = VolumeMeshData::default();
    data.positions = positions;
    data.cells = cells;
    data.cell_scalars.insert("scalar".to_string(), scalars);
    data
}

// ---------------------------------------------------------------------------
// Build scene
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct PlState {
    pub scene:       viewport_lib::scene::Scene,
    pub selection:   viewport_lib::Selection,
    pub sub_selection: viewport_lib::SubSelection,
    pub built:       bool,
    /// Which section is active: true = unified API, false = per-type reference.
    pub unified_mode: bool,
    /// Active mask for the unified section.
    pub unified_mask: PlUnifiedMask,
    pub level:       PlPickLevel,
    pub cube_mesh_id: MeshId,
    pub hemi_mesh_id: MeshId,
    pub mesh_lookup: std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    pub wireframe:   bool,
    pub shift_held:  bool,
    pub drag_start:  Option<glam::Vec2>,
    pub node_names:  Vec<(NodeId, String)>,
    pub last_hit:    Option<PlHitInfo>,
    pub hit_marker:  Option<glam::Vec3>,
    pub show_hit_marker: bool,
    pub pc_positions: Vec<[f32; 3]>,
    pub volume_id:   Option<viewport_lib::VolumeId>,
    pub volume_data: Option<viewport_lib::VolumeData>,
    /// Local-space positions of the Gaussian splat grid.
    pub splat_positions: Vec<[f32; 3]>,
    /// GPU handle for the uploaded Gaussian splat data.
    pub splat_id:    Option<GaussianSplatId>,
    /// Whether the point cloud is selected at the object level.
    pub pc_selected: bool,
    /// Whether the TVM capsule (pick_id=11) is selected at the object level.
    pub tvm_selected: bool,
    /// Whether the hex cylinder (pick_id=12) is selected at the object level.
    pub tvm_tet_selected: bool,
    /// Whether the Gaussian splat object is selected at the object level.
    pub splat_selected: bool,
    /// Whether the volume is selected at the object level.
    pub vol_selected: bool,
    /// Opaque mesh handle for TVM rendering (boundary surface).
    pub tvm_mesh_id: Option<MeshId>,
    /// face_to_cell mapping from upload_volume_mesh_data for the capsule (pick_id=11).
    pub tvm_face_to_cell: Vec<u32>,
    /// CPU-side volume mesh data (kept for per-type cell picking and highlighting).
    pub tvm_data:    Option<VolumeMeshData>,
    /// Projected-tet handle for the transparent tet mesh (pick_id=12).
    pub tvm_tet_id:   Option<ProjectedTetId>,
    /// Opaque boundary mesh for the hex cylinder so it renders on the LDR path.
    pub tvm_tet_mesh_id: Option<MeshId>,
    /// CPU-side tet mesh data for unified CELL picking and sub-selection highlighting.
    pub tvm_tet_data: Option<std::sync::Arc<VolumeMeshData>>,
    /// Positions for the multi-strip polyline (pick_id=30).
    pub polyline_positions:    Vec<[f32; 3]>,
    /// Strip lengths for the multi-strip polyline.
    pub polyline_strip_lengths: Vec<u32>,
    /// Positions for the arrow glyph set (pick_id=31).
    pub arrow_glyph_positions: Vec<[f32; 3]>,
    /// Positions for the tensor glyph set (pick_id=32).
    pub tensor_glyph_positions: Vec<[f32; 3]>,
    /// Per-instance eigenvalues for tensor glyphs.
    pub tensor_glyph_eigenvalues: Vec<[f32; 3]>,
    /// Per-instance eigenvector bases for tensor glyphs.
    pub tensor_glyph_eigenvectors: Vec<[[f32; 3]; 3]>,
    /// Positions for the sprite set (pick_id=33).
    pub sprite_positions:       Vec<[f32; 3]>,
    /// Per-instance sizes for the sprite arc.
    pub sprite_sizes:           Vec<f32>,
    /// Per-instance colors for the sprite arc.
    pub sprite_colors:          Vec<[f32; 4]>,
    /// Positions for the noughts-and-crosses sprite set (pick_id=34).
    pub xo_sprite_positions:    Vec<[f32; 3]>,
    /// Per-instance sizes for the noughts-and-crosses set.
    pub xo_sprite_sizes:        Vec<f32>,
    /// Per-instance colors for the noughts-and-crosses set.
    pub xo_sprite_colors:       Vec<[f32; 4]>,
    /// Object-level selection flags for new item types.
    pub polyline_selected:      bool,
    pub arrow_glyph_selected:   bool,
    pub tensor_glyph_selected:  bool,
    pub sprite_selected:        bool,
    pub xo_sprite_selected:     bool,
}

impl Default for PlState {
    fn default() -> Self {
        Self {
            scene:            viewport_lib::scene::Scene::new(),
            selection:        viewport_lib::Selection::new(),
            sub_selection:    viewport_lib::SubSelection::new(),
            built:            false,
            unified_mode:     true,
            unified_mask:     PlUnifiedMask::default(),
            level:            PlPickLevel::default(),
            cube_mesh_id:     MeshId::from_index(0),
            hemi_mesh_id:     MeshId::from_index(0),
            mesh_lookup:      std::collections::HashMap::new(),
            wireframe:        false,
            shift_held:       false,
            drag_start:       None,
            node_names:       Vec::new(),
            last_hit:         None,
            hit_marker:       None,
            show_hit_marker:  true,
            pc_positions:     Vec::new(),
            volume_id:        None,
            volume_data:      None,
            splat_positions:  Vec::new(),
            splat_id:         None,
            pc_selected:      false,
            tvm_selected:     false,
            tvm_tet_selected: false,
            splat_selected:   false,
            vol_selected:     false,
            tvm_mesh_id:           None,
            tvm_face_to_cell:      Vec::new(),
            tvm_data:              None,
            tvm_tet_id:            None,
            tvm_tet_mesh_id:       None,
            tvm_tet_data:          None,
            polyline_positions:    Vec::new(),
            polyline_strip_lengths: Vec::new(),
            arrow_glyph_positions: Vec::new(),
            tensor_glyph_positions: Vec::new(),
            tensor_glyph_eigenvalues: Vec::new(),
            tensor_glyph_eigenvectors: Vec::new(),
            sprite_positions:       Vec::new(),
            sprite_sizes:           Vec::new(),
            sprite_colors:          Vec::new(),
            xo_sprite_positions:    Vec::new(),
            xo_sprite_sizes:        Vec::new(),
            xo_sprite_colors:       Vec::new(),
            polyline_selected:      false,
            arrow_glyph_selected:   false,
            tensor_glyph_selected:  false,
            sprite_selected:        false,
            xo_sprite_selected:     false,
        }
    }
}

impl PlState {
    /// Clear all selection state: scene-graph selection, sub-selection, and
    /// every per-item boolean flag.
    pub(crate) fn clear_selection(&mut self) {
        self.selection.clear();
        self.sub_selection.clear();
        self.pc_selected = false;
        self.tvm_selected = false;
        self.tvm_tet_selected = false;
        self.splat_selected = false;
        self.vol_selected = false;
        self.polyline_selected = false;
        self.arrow_glyph_selected = false;
        self.tensor_glyph_selected = false;
        self.sprite_selected = false;
        self.xo_sprite_selected = false;
        self.last_hit = None;
        self.hit_marker = None;
    }

    /// Set the boolean flag for a non-mesh item.  Returns false if the ID
    /// is not a flag item (i.e. it is a mesh node).
    fn set_flag(&mut self, id: NodeId, selected: bool) -> bool {
        match id {
            100 => { self.pc_selected = selected; true }
            10  => { self.splat_selected = selected; true }
            11  => { self.tvm_selected = selected; true }
            12  => { self.tvm_tet_selected = selected; true }
            20  => { self.vol_selected = selected; true }
            30  => { self.polyline_selected = selected; true }
            31  => { self.arrow_glyph_selected = selected; true }
            32  => { self.tensor_glyph_selected = selected; true }
            33  => { self.sprite_selected = selected; true }
            34  => { self.xo_sprite_selected = selected; true }
            _   => false,
        }
    }

    /// Toggle the boolean flag for a non-mesh item.
    fn toggle_flag(&mut self, id: NodeId) -> bool {
        match id {
            100 => { self.pc_selected = !self.pc_selected; true }
            10  => { self.splat_selected = !self.splat_selected; true }
            11  => { self.tvm_selected = !self.tvm_selected; true }
            12  => { self.tvm_tet_selected = !self.tvm_tet_selected; true }
            20  => { self.vol_selected = !self.vol_selected; true }
            30  => { self.polyline_selected = !self.polyline_selected; true }
            31  => { self.arrow_glyph_selected = !self.arrow_glyph_selected; true }
            32  => { self.tensor_glyph_selected = !self.tensor_glyph_selected; true }
            33  => { self.sprite_selected = !self.sprite_selected; true }
            34  => { self.xo_sprite_selected = !self.xo_sprite_selected; true }
            _   => false,
        }
    }

    /// Select a single object (clear everything else first).
    pub(crate) fn select_object(&mut self, id: NodeId) {
        self.clear_selection();
        if !self.set_flag(id, true) {
            self.selection.add(id);
        }
    }

    /// Toggle an object's selection without clearing others.
    pub(crate) fn toggle_object(&mut self, id: NodeId) {
        if !self.toggle_flag(id) {
            self.selection.toggle(id);
        }
    }
}

impl App {
    pub(crate) fn build_pl_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.pl_state.scene = viewport_lib::scene::Scene::new();
        self.pl_state.node_names.clear();
        self.pl_state.mesh_lookup.clear();
        self.pl_state.clear_selection();

        // --- Cube mesh ---
        let cube_mesh = viewport_lib::primitives::cube(2.0);
        let cube_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &cube_mesh)
            .expect("pl cube mesh");
        self.pl_state.cube_mesh_id = cube_id;
        self.pl_state.mesh_lookup.insert(
            cube_id.index() as u64,
            (cube_mesh.positions.clone(), cube_mesh.indices.clone()),
        );

        // --- Hemisphere mesh ---
        let hemi_mesh = viewport_lib::primitives::hemisphere(1.5, 20, 8);
        let hemi_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &hemi_mesh)
            .expect("pl hemi mesh");
        self.pl_state.hemi_mesh_id = hemi_id;
        self.pl_state.mesh_lookup.insert(
            hemi_id.index() as u64,
            (hemi_mesh.positions.clone(), hemi_mesh.indices.clone()),
        );

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
            let node_id = self.pl_state.scene.add_named(name, Some(*mesh_id), transform, mat);
            self.pl_state.node_names.push((node_id, name.to_string()));
        }

        // --- Point cloud: 10x8 grid above the scene ---
        let mut pc: Vec<[f32; 3]> = Vec::new();
        for row in 0..8_i32 {
            for col in 0..10_i32 {
                pc.push([
                    (col as f32 - 4.5) * 0.9,
                    (row as f32 - 3.5) * 0.55,
                    3.5,
                ]);
            }
        }
        self.pl_state.pc_positions = pc;

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
        self.pl_state.volume_id = Some(vol_id);
        self.pl_state.volume_data = Some(VolumeData {
            data: vol_data,
            dims,
            origin: [0.0, 0.0, 0.0],
            spacing: [0.25, 0.25, 0.25],
        });

        // --- Gaussian splats: 3x3 grid in local space, rendered at y=5 ---
        let mut splat_positions: Vec<[f32; 3]> = Vec::with_capacity(9);
        for row in -1..=1_i32 {
            for col in -1..=1_i32 {
                splat_positions.push([
                    col as f32 * 0.8,
                    0.0,
                    -3.5 + row as f32 * 0.8
                ]);
            }
        }
        let splat_data = make_pl_splat_data(&splat_positions);
        let splat_id = renderer.upload_gaussian_splats(&self.device, &self.queue, &splat_data);
        self.pl_state.splat_positions = splat_positions;
        self.pl_state.splat_id = Some(splat_id);

        // --- Volume mesh: capsule hex mesh, rendered as opaque boundary surface ---
        let tvm_data = make_pl_tvm_data();
        if let Ok((mesh_id, face_to_cell)) = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &tvm_data)
        {
            self.pl_state.tvm_mesh_id = Some(mesh_id);
            self.pl_state.tvm_face_to_cell = face_to_cell;
        }
        self.pl_state.tvm_data = Some(tvm_data);

        // --- Hex cylinder at x=-7 (pick_id=12) ---
        // Upload as both a projected-tet mesh (for cell picking) and an opaque
        // boundary surface (for LDR rendering without the HDR/OIT path).
        renderer
            .resources_mut()
            .ensure_colormaps_initialized(&self.device, &self.queue);
        let tet_data = make_pl_tvm_tet_data();
        let tvm_tet_id = renderer
            .resources_mut()
            .upload_projected_tet_mesh(&self.device, &tet_data, "scalar", viewport_lib::ColormapId(0))
            .ok()
            .map(|(id, _, _)| id);
        let tvm_tet_mesh_id = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &tet_data)
            .ok()
            .map(|(id, _)| id);
        self.pl_state.tvm_tet_id       = tvm_tet_id;
        self.pl_state.tvm_tet_mesh_id  = tvm_tet_mesh_id;
        self.pl_state.tvm_tet_data = Some(std::sync::Arc::new(tet_data));

        // --- Polyline: 3 spiral streamlines in front of sprites (pick_id=30) ---
        let mut pl_pos: Vec<[f32; 3]> = Vec::new();
        let mut pl_lens: Vec<u32> = Vec::new();
        let n_per_strip = 40_u32;
        for strip in 0..3_i32 {
            let y_off = (strip - 1) as f32 * 1.8;
            let radius = 1.2 + strip as f32 * 0.3;
            for i in 0..n_per_strip {
                let t = i as f32 / (n_per_strip - 1) as f32;
                let angle = t * std::f32::consts::TAU * 1.5;
                pl_pos.push([
                    radius * angle.cos(),
                    y_off + t * 3.0 - 1.5,
                    7.5 + radius * angle.sin(),
                ]);
            }
            pl_lens.push(n_per_strip);
        }
        self.pl_state.polyline_positions    = pl_pos;
        self.pl_state.polyline_strip_lengths = pl_lens;

        // --- Arrow glyphs: 5 arrows at y=-4 pointing up (pick_id=31) ---
        self.pl_state.arrow_glyph_positions = (0..5_i32)
            .map(|i| [(i - 2) as f32 * 2.0, -4.0, 0.0])
            .collect();

        // --- Tensor glyphs: 4 ellipsoids at x=8 (pick_id=32) ---
        self.pl_state.tensor_glyph_positions = vec![
            [8.0, -1.5, -1.5],
            [8.0,  1.5, -1.5],
            [8.0, -1.5,  1.5],
            [8.0,  1.5,  1.5],
        ];
        // Varied eigenvalues: sphere-ish, cigar, disk, mixed.
        self.pl_state.tensor_glyph_eigenvalues = vec![
            [0.8, 0.7, 0.6],
            [1.2, 0.3, 0.3],
            [0.3, 0.3, 1.2],
            [1.0, 0.6, 0.2],
        ];
        // Rotated eigenvector bases to show directional variation.
        let id_basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rot45 = {
            let c = std::f32::consts::FRAC_1_SQRT_2;
            [[c, c, 0.0], [-c, c, 0.0], [0.0, 0.0, 1.0]]
        };
        let rot30 = {
            let c = 0.866_f32;
            let s = 0.5_f32;
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
        };
        let rot60 = {
            let c = 0.5_f32;
            let s = 0.866_f32;
            [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
        };
        self.pl_state.tensor_glyph_eigenvectors = vec![id_basis, rot45, rot30, rot60];

        // --- Sprites: 6 in a row at z=6 (pick_id=33) ---
        // Sprites arranged in an arc with graduated sizes and colors so the
        // group reads as a connected sequence rather than isolated dots.
        {
            let n = 8;
            let arc_center = [0.0_f32, 0.0, 6.0];
            let arc_radius = 4.0_f32;
            let arc_start = std::f32::consts::FRAC_PI_4;
            let arc_sweep = std::f32::consts::PI * 0.5;
            let mut positions = Vec::with_capacity(n);
            let mut sizes = Vec::with_capacity(n);
            let mut colors = Vec::with_capacity(n);
            for i in 0..n {
                let t = i as f32 / (n - 1) as f32;
                let angle = arc_start + arc_sweep * t;
                positions.push([
                    arc_center[0] + arc_radius * angle.cos(),
                    arc_center[1],
                    arc_center[2] + arc_radius * angle.sin(),
                ]);
                // Small to large along the arc.
                sizes.push(14.0 + 22.0 * t);
                // Yellow -> orange gradient.
                colors.push([1.0, 0.95 - 0.5 * t, 0.15 + 0.1 * t, 1.0]);
            }
            self.pl_state.sprite_positions = positions;
            self.pl_state.sprite_sizes = sizes;
            self.pl_state.sprite_colors = colors;
        }

        // --- Noughts and crosses sprites: alternating + and O at z=-6 (pick_id=34) ---
        // A second sprite group with different shapes to verify outline matching.
        {
            let n = 6;
            let x_start = -5.0_f32;
            let x_step = 2.0_f32;
            let mut positions = Vec::with_capacity(n);
            let mut sizes = Vec::with_capacity(n);
            let mut colors = Vec::with_capacity(n);
            for i in 0..n {
                positions.push([x_start + x_step * i as f32, 0.0, -6.0]);
                // Alternate between small and large.
                sizes.push(if i % 2 == 0 { 20.0 } else { 40.0 });
                // Alternate cyan and magenta.
                if i % 2 == 0 {
                    colors.push([0.2, 0.9, 0.9, 1.0]);
                } else {
                    colors.push([0.9, 0.2, 0.9, 1.0]);
                }
            }
            self.pl_state.xo_sprite_positions = positions;
            self.pl_state.xo_sprite_sizes = sizes;
            self.pl_state.xo_sprite_colors = colors;
        }

        self.pl_state.built = true;
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

        // Helper: record a hit and update hit_marker / last_hit.
        macro_rules! record_hit {
            ($name:expr, $type:expr, $pos:expr, $normal:expr, $sub:expr, $scalar:expr) => {{
                self.pl_state.last_hit = Some(PlHitInfo {
                    object_name: $name,
                    object_type: $type,
                    world_pos: $pos,
                    normal: $normal,
                    sub_object: $sub,
                    scalar_value: $scalar,
                });
                self.pl_state.hit_marker = Some($pos);
            }};
        }

        // Helper: apply sub-object selection with shift logic.
        macro_rules! select_sub {
            ($id:expr, $sub:expr) => {
                if shift {
                    self.pl_state.sub_selection.toggle($id, $sub);
                } else {
                    self.pl_state.clear_selection();
                    self.pl_state.sub_selection.add($id, $sub);
                }
            };
        }

        match self.pl_state.level {
            PlPickLevel::Object => {
                let mesh_hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin, ray_dir, &self.pl_state.scene, &mesh_lookup,
                );
                // Fallback: also try a ray-AABB test against the volume box.
                let vol_hit = self.pl_state.volume_id.is_some() && {
                    let vol_model = glam::Mat4::from_translation(glam::vec3(-2.0, -1.0, -6.0));
                    let inv = vol_model.inverse();
                    let lo = inv.transform_point3(ray_origin);
                    let ld = inv.transform_vector3(ray_dir);
                    let inv_d = glam::Vec3::ONE / ld;
                    let t1 = (glam::Vec3::ZERO - lo) * inv_d;
                    let t2 = (glam::Vec3::splat(4.0) - lo) * inv_d;
                    let tmin = t1.min(t2).max_element();
                    let tmax = t1.max(t2).min_element();
                    tmax >= tmin.max(0.0)
                };
                if let Some(hit) = mesh_hit {
                    if shift {
                        self.pl_state.selection.toggle(hit.id);
                    } else {
                        self.pl_state.select_object(hit.id);
                    }
                    record_hit!(
                        self.pl_node_name(hit.id), "Mesh",
                        hit.world_pos, hit.normal, None, None
                    );
                } else if vol_hit {
                    if shift {
                        self.pl_state.vol_selected = !self.pl_state.vol_selected;
                    } else {
                        self.pl_state.select_object(20);
                    }
                    self.pl_state.last_hit = Some(PlHitInfo {
                        object_name: "Volume".to_string(),
                        object_type: "Volume",
                        world_pos: ray_origin + ray_dir * 5.0,
                        normal: glam::Vec3::ZERO,
                        sub_object: None,
                        scalar_value: None,
                    });
                    self.pl_state.hit_marker = None;
                } else if !shift {
                    self.pl_state.clear_selection();
                }
            }

            PlPickLevel::Face => {
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin, ray_dir, &self.pl_state.scene, &mesh_lookup,
                );
                if let Some(hit) = hit {
                    if let Some(sub) = hit.sub_object {
                        select_sub!(hit.id, sub);
                    }
                    record_hit!(
                        self.pl_node_name(hit.id), "Mesh",
                        hit.world_pos, hit.normal, hit.sub_object, None
                    );
                } else if !shift {
                    self.pl_state.clear_selection();
                }
            }

            PlPickLevel::Vertex => {
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin, ray_dir, &self.pl_state.scene, &mesh_lookup,
                );
                if let Some(hit) = hit {
                    let vertex_sub = self.pl_node_mesh_key(hit.id).and_then(|key| {
                        let (positions, indices) = self.pl_state.mesh_lookup.get(&key)?;
                        let model = self.pl_node_model_matrix(hit.id);
                        viewport_lib::nearest_vertex_on_hit(&hit, positions, indices, model)
                    });
                    if let Some(sub) = vertex_sub {
                        select_sub!(hit.id, sub);
                    }
                    let marker = vertex_sub
                        .and_then(|s| self.pl_vertex_world_pos(hit.id, s))
                        .unwrap_or(hit.world_pos);
                    record_hit!(
                        self.pl_node_name(hit.id), "Mesh",
                        marker, hit.normal, vertex_sub, None
                    );
                } else if !shift {
                    self.pl_state.clear_selection();
                }
            }

            PlPickLevel::Voxel => {
                let hit = self.pl_state.volume_id.zip(self.pl_state.volume_data.as_ref()).and_then(|(vol_id, vol_data)| {
                    let mut item = viewport_lib::VolumeItem::default();
                    item.volume_id = vol_id;
                    item.model = glam::Mat4::from_translation(glam::vec3(-2.0, -1.0, -6.0))
                        .to_cols_array_2d();
                    item.bbox_min = [0.0, 0.0, 0.0];
                    item.bbox_max = [4.0, 4.0, 4.0];
                    item.scalar_range = (0.0, 1.0);
                    item.threshold_min = 0.15;
                    item.threshold_max = 1.0;
                    viewport_lib::pick_volume_cpu(ray_origin, ray_dir, 20, &item, vol_data)
                });

                if let Some(hit) = hit {
                    let sub = hit.sub_object.unwrap();
                    select_sub!(20, sub);
                    record_hit!(
                        "Volume".to_string(), "Volume",
                        hit.world_pos, hit.normal, hit.sub_object, hit.scalar_value
                    );
                } else if !shift {
                    self.pl_state.clear_selection();
                }
            }

            PlPickLevel::Point => {
                let mut pc_item = viewport_lib::PointCloudItem::default();
                pc_item.positions = self.pl_state.pc_positions.clone();
                pc_item.id = 1;
                let hit = viewport_lib::pick_point_cloud_cpu(pos, 1, &pc_item, view_proj, vp_size, 20.0);
                if let Some(hit) = hit {
                    let sub = hit.sub_object.unwrap();
                    select_sub!(1, sub);
                    record_hit!(
                        "Point Cloud".to_string(), "Point Cloud",
                        hit.world_pos, glam::Vec3::Y, Some(sub), None
                    );
                } else if !shift {
                    self.pl_state.clear_selection();
                }
            }

            PlPickLevel::Splat => {
                let splat_model = pl_splat_model();
                let hit = viewport_lib::pick_gaussian_splat_cpu(
                    pos, 10, &self.pl_state.splat_positions,
                    splat_model, view_proj, vp_size, 24.0,
                );
                if let Some(hit) = hit {
                    let sub = hit.sub_object.unwrap();
                    select_sub!(10, sub);
                    record_hit!(
                        "Gaussian Splats".to_string(), "Gaussian Splat",
                        hit.world_pos, glam::Vec3::Y, Some(sub), None
                    );
                } else if !shift {
                    self.pl_state.clear_selection();
                }
            }

            PlPickLevel::Tvm => {
                let hit = self.pl_state.tvm_data.as_ref().and_then(|data| {
                    viewport_lib::pick_transparent_volume_mesh_cpu(
                        ray_origin, ray_dir, 11, glam::Mat4::IDENTITY, data,
                    )
                });
                if let Some(hit) = hit {
                    let sub = hit.sub_object.unwrap();
                    select_sub!(11, sub);
                    record_hit!(
                        "Transparent Volume Mesh".to_string(), "Volume Mesh",
                        hit.world_pos, hit.normal, Some(sub), None
                    );
                } else if !shift {
                    self.pl_state.clear_selection();
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
            self.pl_state.clear_selection();
        }

        match self.pl_state.level {
            PlPickLevel::Object => {
                // Use box_select (object-level by position).
                let objects: Vec<&dyn viewport_lib::traits::ViewportObject> =
                    self.pl_state.scene.nodes().map(|n| n as &dyn viewport_lib::traits::ViewportObject).collect();
                let hits = viewport_lib::picking::box_select(
                    r_min, r_max, &objects, view_proj, vp_size,
                );
                for id in hits {
                    self.pl_state.selection.add(id);
                }
            }

            PlPickLevel::Face => {
                let mesh_lookup_usize = self.pl_pick_mesh_lookup_usize();
                let scene_items = self.pl_state.scene.collect_render_items(&viewport_lib::Selection::new());
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
                            self.pl_state.sub_selection.add(*obj_id, sub);
                        }
                    }
                }
            }

            PlPickLevel::Vertex => {
                // Collect all vertices that project into the rect.
                for node in self.pl_state.scene.nodes() {
                    let node_id = viewport_lib::traits::ViewportObject::id(node);
                    let mesh_key = viewport_lib::traits::ViewportObject::mesh_id(node);
                    let model = viewport_lib::traits::ViewportObject::model_matrix(node);
                    if let Some(key) = mesh_key {
                        if let Some((positions, _)) = self.pl_state.mesh_lookup.get(&key) {
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
                                    self.pl_state.sub_selection.add(node_id, SubObjectRef::Vertex(vi as u32));
                                }
                            }
                        }
                    }
                }
            }

            PlPickLevel::Point => {
                // Project point cloud positions into the rect.
                for (i, pos) in self.pl_state.pc_positions.iter().enumerate() {
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
                        self.pl_state.sub_selection.add(1, SubObjectRef::Point(i as u32));
                    }
                }
            }

            PlPickLevel::Voxel => {
                if let (Some(vol_id), Some(vol_data)) =
                    (self.pl_state.volume_id, self.pl_state.volume_data.as_ref())
                {
                    let mut item = viewport_lib::VolumeItem::default();
                    item.volume_id = vol_id;
                    item.model = glam::Mat4::from_translation(glam::vec3(-2.0, -1.0, -6.0))
                        .to_cols_array_2d();
                    item.bbox_min = [0.0, 0.0, 0.0];
                    item.bbox_max = [4.0, 4.0, 4.0];
                    item.threshold_min = 0.15;
                    item.threshold_max = 1.0;
                    let result = viewport_lib::pick_volume_rect(
                        r_min, r_max, 20, &item, vol_data, view_proj, vp_size,
                    );
                    for (_, subs) in &result.hits {
                        for &sub in subs {
                            self.pl_state.sub_selection.add(20, sub);
                        }
                    }
                }
            }

            PlPickLevel::Splat => {
                let splat_model = pl_splat_model();
                let result = viewport_lib::pick_gaussian_splat_rect(
                    r_min, r_max, 10, &self.pl_state.splat_positions,
                    splat_model, view_proj, vp_size,
                );
                for (_, subs) in &result.hits {
                    for &sub in subs {
                        self.pl_state.sub_selection.add(10, sub);
                    }
                }
            }

            PlPickLevel::Tvm => {
                if let Some(data) = self.pl_state.tvm_data.as_ref() {
                    let result = viewport_lib::pick_transparent_volume_mesh_rect(
                        r_min, r_max, 11, glam::Mat4::IDENTITY, data, view_proj, vp_size,
                    );
                    for (_, subs) in &result.hits {
                        for &sub in subs {
                            self.pl_state.sub_selection.add(11, sub);
                        }
                    }
                }
            }
        }

        self.pl_state.last_hit = None;
        self.pl_state.hit_marker = None;
    }
}

// ---------------------------------------------------------------------------
// Unified click / box-select handlers
// ---------------------------------------------------------------------------

impl App {
    /// Handle a left-click using renderer.pick() with the active PickMask.
    pub(crate) fn handle_pl_unified_click(
        &mut self,
        pos: glam::Vec2,
        vp_size: glam::Vec2,
        view_proj: glam::Mat4,
        shift: bool,
        renderer: &ViewportRenderer,
    ) {
        let mask = self.pl_state.unified_mask.to_pick_mask();
        let Some(hit) = renderer.pick(pos, vp_size, view_proj, mask) else {
            if !shift {
                self.pl_state.clear_selection();
            }
            return;
        };

        match hit.sub_object {
            None => {
                if shift {
                    self.pl_state.toggle_object(hit.id);
                } else {
                    self.pl_state.select_object(hit.id);
                }
            }
            Some(sub) => {
                if shift {
                    self.pl_state.sub_selection.toggle(hit.id, sub);
                } else {
                    self.pl_state.clear_selection();
                    self.pl_state.sub_selection.add(hit.id, sub);
                }
            }
        }

        let is_mesh_node = Self::pl_item_label(hit.id).is_none();
        let object_type = match hit.sub_object {
            None => {
                if is_mesh_node { "Mesh" } else { Self::pl_item_label(hit.id).unwrap_or("Object") }
            }
            Some(SubObjectRef::Face(_))     => "Mesh",
            Some(SubObjectRef::Vertex(_))   => "Mesh",
            Some(SubObjectRef::Point(_))    => "Point Cloud",
            Some(SubObjectRef::Splat(_))    => "Gaussian Splat",
            Some(SubObjectRef::Instance(_)) => "Glyph",
            Some(SubObjectRef::Segment(_))  => "Polyline",
            Some(SubObjectRef::Strip(_))    => "Polyline",
            Some(_)                         => "Object",
        };

        self.pl_state.last_hit = Some(PlHitInfo {
            object_name: self.pl_node_name(hit.id),
            object_type,
            world_pos: hit.world_pos,
            normal: hit.normal,
            sub_object: hit.sub_object,
            scalar_value: hit.scalar_value,
        });
        self.pl_state.hit_marker = Some(hit.world_pos);
    }

    /// Handle a rubber-band box selection using renderer.pick_rect() with the
    /// active PickMask.
    pub(crate) fn handle_pl_unified_box_select(
        &mut self,
        rect_min: glam::Vec2,
        rect_max: glam::Vec2,
        vp_size: glam::Vec2,
        view_proj: glam::Mat4,
        shift: bool,
        renderer: &ViewportRenderer,
    ) {
        let r_min = glam::Vec2::new(rect_min.x.min(rect_max.x), rect_min.y.min(rect_max.y));
        let r_max = glam::Vec2::new(rect_min.x.max(rect_max.x), rect_min.y.max(rect_max.y));

        if !shift {
            self.pl_state.clear_selection();
        }

        let mask = self.pl_state.unified_mask.to_pick_mask();
        let result: PickRectResult = renderer.pick_rect(r_min, r_max, vp_size, view_proj, mask);

        for id in &result.objects {
            if !self.pl_state.set_flag(*id, true) {
                self.pl_state.selection.add(*id);
            }
        }
        for (id, sub) in &result.elements {
            self.pl_state.sub_selection.add(*id, *sub);
        }

        self.pl_state.last_hit = None;
        self.pl_state.hit_marker = None;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// World transform used for the Gaussian splat grid in showcase 33.
///
/// Both the renderer submission and the pick functions must use the same model.
pub(crate) fn pl_splat_model() -> glam::Mat4 {
    glam::Mat4::from_translation(glam::vec3(-4.0, 0.0, -0.75))
}

impl App {
    /// Mesh lookup map for `pick_scene_nodes_cpu` (key = u64 mesh id).
    pub(crate) fn pl_pick_mesh_lookup(&self) -> HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)> {
        self.pl_state.mesh_lookup.clone()
    }

    /// Mesh lookup map for `pick_rect` (key = usize mesh index).
    fn pl_pick_mesh_lookup_usize(&self) -> HashMap<usize, (Vec<[f32; 3]>, Vec<u32>)> {
        self.pl_state.mesh_lookup
            .iter()
            .map(|(&k, v)| (k as usize, v.clone()))
            .collect()
    }

    /// Display name for a node ID.
    pub(crate) fn pl_node_name(&self, id: NodeId) -> String {
        if let Some(label) = Self::pl_item_label(id) {
            return label.to_string();
        }
        self.pl_state.node_names
            .iter()
            .find(|(nid, _)| *nid == id)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| format!("id={id}"))
    }


    /// Human-readable label for the non-mesh pick IDs used in showcase 33.
    fn pl_item_label(id: NodeId) -> Option<&'static str> {
        match id {
            100 => Some("Point Cloud"),
            20  => Some("Volume"),
            10  => Some("Gaussian Splats"),
            11  => Some("Volume Mesh"),
            12  => Some("TVM Tets"),
            30  => Some("Polyline"),
            31  => Some("Arrow Glyphs"),
            32  => Some("Tensor Glyphs"),
            33  => Some("Sprites"),
            _   => None,
        }
    }

    /// World-space model matrix for a node ID.
    pub(crate) fn pl_node_model_matrix(&self, id: NodeId) -> glam::Mat4 {
        for node in self.pl_state.scene.nodes() {
            if viewport_lib::traits::ViewportObject::id(node) == id {
                return viewport_lib::traits::ViewportObject::model_matrix(node);
            }
        }
        glam::Mat4::IDENTITY
    }

    /// Mesh key (u64) for a node ID.
    fn pl_node_mesh_key(&self, id: NodeId) -> Option<u64> {
        for node in self.pl_state.scene.nodes() {
            if viewport_lib::traits::ViewportObject::id(node) == id {
                return viewport_lib::traits::ViewportObject::mesh_id(node);
            }
        }
        None
    }

    /// World-space position for a Vertex sub-object.
    pub(crate) fn pl_vertex_world_pos(&self, node_id: NodeId, sub: SubObjectRef) -> Option<glam::Vec3> {
        if let SubObjectRef::Vertex(vi) = sub {
            let key = self.pl_node_mesh_key(node_id)?;
            let (positions, _) = self.pl_state.mesh_lookup.get(&key)?;
            let pos = positions.get(vi as usize)?;
            let model = self.pl_node_model_matrix(node_id);
            Some(model.transform_point3(glam::Vec3::from(*pos)))
        } else {
            None
        }
    }

}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_pick_levels(app: &mut App, ui: &mut egui::Ui) {
    // -----------------------------------------------------------------------
    // Section toggle
    // -----------------------------------------------------------------------
    ui.horizontal(|ui| {
        ui.selectable_value(&mut app.pl_state.unified_mode, true,  "Unified API");
        ui.selectable_value(&mut app.pl_state.unified_mode, false, "Per-type reference");
    });
    ui.separator();

    if app.pl_state.unified_mode {
        // -------------------------------------------------------------------
        // Unified API section
        // -------------------------------------------------------------------
        ui.label(egui::RichText::new("renderer.pick() / pick_rect()").strong());
        ui.label("Mask:");
        ui.horizontal_wrapped(|ui| {
            for mask in [
                PlUnifiedMask::Object,
                PlUnifiedMask::Face,
                PlUnifiedMask::PointLike,
                PlUnifiedMask::EdgeLike,
                PlUnifiedMask::FaceAndPointLike,
            ] {
                ui.radio_value(&mut app.pl_state.unified_mask, mask, mask.label());
            }
        });
    } else {
        // -------------------------------------------------------------------
        // Per-type reference section
        // -------------------------------------------------------------------
        ui.label(egui::RichText::new("Per-type picking").strong());
        ui.label("Pick Level:");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut app.pl_state.level, PlPickLevel::Object, "Object");
            ui.radio_value(&mut app.pl_state.level, PlPickLevel::Face,   "Face");
            ui.radio_value(&mut app.pl_state.level, PlPickLevel::Vertex, "Vertex");
            ui.radio_value(&mut app.pl_state.level, PlPickLevel::Point,  "Point");
            ui.radio_value(&mut app.pl_state.level, PlPickLevel::Voxel,  "Voxel");
            ui.radio_value(&mut app.pl_state.level, PlPickLevel::Splat,  "Splat");
            ui.radio_value(&mut app.pl_state.level, PlPickLevel::Tvm,    "TVM");
        });
    }

    ui.separator();
    ui.checkbox(&mut app.pl_state.wireframe, "Wireframe overlay");
    ui.checkbox(&mut app.pl_state.show_hit_marker, "Show hit marker");

    if ui.button("Clear selection").clicked() {
        app.pl_state.clear_selection();
    }

    ui.separator();
    ui.label(egui::RichText::new("Controls").strong());
    ui.label("Click: select at level");
    ui.label("Shift+click: add / toggle");
    ui.label("Drag: box select");

    ui.separator();

    if let Some(hit) = &app.pl_state.last_hit {
        ui.label(egui::RichText::new("Last Hit").strong());
        ui.label(format!("Object: {}", hit.object_name));
        ui.label(format!("Type: {}", hit.object_type));
        match hit.sub_object {
            None                             => { ui.label("Level: Object"); }
            Some(SubObjectRef::Face(i))      => { ui.label(format!("Level: Face #{i}")); }
            Some(SubObjectRef::Vertex(i))    => { ui.label(format!("Level: Vertex #{i}")); }
            Some(SubObjectRef::Edge(i))      => { ui.label(format!("Level: Edge #{i}")); }
            Some(SubObjectRef::Point(i))     => { ui.label(format!("Level: Point #{i}")); }
            Some(SubObjectRef::Voxel(i))     => { ui.label(format!("Level: Voxel #{i}")); }
            Some(SubObjectRef::Cell(i))      => { ui.label(format!("Level: Cell #{i}")); }
            Some(SubObjectRef::Splat(i))     => { ui.label(format!("Level: Splat #{i}")); }
            Some(SubObjectRef::Instance(i))  => { ui.label(format!("Level: Instance #{i}")); }
            Some(SubObjectRef::Segment(i))   => { ui.label(format!("Level: Segment #{i}")); }
            Some(SubObjectRef::Strip(i))     => { ui.label(format!("Level: Strip #{i}")); }
            Some(_)                          => { ui.label("Level: (unknown)"); }
        }
        let p = hit.world_pos;
        ui.label(format!("Pos: ({:.2}, {:.2}, {:.2})", p.x, p.y, p.z));
        let n = hit.normal;
        ui.label(format!("Normal: ({:.2}, {:.2}, {:.2})", n.x, n.y, n.z));
        if let Some(sv) = hit.scalar_value {
            ui.label(format!("Scalar: {sv:.3}"));
        }
    } else {
        let obj_count = app.pl_state.selection.len();
        let mut faces     = 0u32;
        let mut verts     = 0u32;
        let mut points    = 0u32;
        let mut splats    = 0u32;
        let mut voxels    = 0u32;
        let mut cells     = 0u32;
        let mut instances = 0u32;
        let mut segments  = 0u32;
        let mut strips    = 0u32;
        for (_, sub) in app.pl_state.sub_selection.iter() {
            match sub {
                SubObjectRef::Face(_)     => faces     += 1,
                SubObjectRef::Vertex(_)   => verts     += 1,
                SubObjectRef::Point(_)    => points    += 1,
                SubObjectRef::Splat(_)    => splats    += 1,
                SubObjectRef::Voxel(_)    => voxels    += 1,
                SubObjectRef::Cell(_)     => cells     += 1,
                SubObjectRef::Instance(_) => instances += 1,
                SubObjectRef::Segment(_)  => segments  += 1,
                SubObjectRef::Strip(_)    => strips    += 1,
                _ => {}
            }
        }
        let sub_total = faces + verts + points + splats + voxels + cells
            + instances + segments + strips;
        if obj_count > 0 || sub_total > 0 {
            ui.label(egui::RichText::new("Selection").strong());
            if obj_count  > 0 { ui.label(format!("Objects: {obj_count}")); }
            if faces      > 0 { ui.label(format!("Faces: {faces}")); }
            if verts      > 0 { ui.label(format!("Vertices: {verts}")); }
            if points     > 0 { ui.label(format!("Points: {points}")); }
            if splats     > 0 { ui.label(format!("Splats: {splats}")); }
            if voxels     > 0 { ui.label(format!("Voxels: {voxels}")); }
            if cells      > 0 { ui.label(format!("Cells: {cells}")); }
            if instances  > 0 { ui.label(format!("Instances: {instances}")); }
            if segments   > 0 { ui.label(format!("Segments: {segments}")); }
            if strips     > 0 { ui.label(format!("Strips: {strips}")); }
        } else {
            ui.label("Click or drag to select.");
        }
    }
}
