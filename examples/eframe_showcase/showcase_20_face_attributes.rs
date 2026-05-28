//! Showcase 20: Per-Face Attributes
//!
//! Demonstrates the two per-face rendering modes:
//!
//!   Left   : `AttributeKind::Vertex`    smooth, Gouraud-interpolated scalar
//!   Centre : `AttributeKind::Face`      flat per-triangle scalar, same data
//!   Right  : `AttributeKind::FaceColour` direct per-face RGBA, no colourmap
//!
//! The left and centre objects carry the same scalar value (face-centroid Z
//! normalised 0->1).  The visual difference:crisp flat facets on Face vs a
//! smooth gradient on Vertex:is the whole point of the showcase.
//!
//! The right object uses `FaceColour` with a hue-cycled rainbow to demonstrate
//! that RGBA colours are applied directly without going through a colourmap.
//! Use the opacity slider to push it into the OIT (order-independent
//! transparency) path and verify correct blending.

use crate::App;
use eframe::egui;
use viewport_lib::{
    AttributeData, BuiltinColourmap, Material, MeshId, NodeId, ViewportRenderer, scene::Scene,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct FaceAttrState {
    pub scene: Scene,
    pub built: bool,
    pub mesh_indices: [MeshId; 3],
    pub node_ids: [NodeId; 3],
    pub colourmap: BuiltinColourmap,
    pub opacity: f32,
}

impl Default for FaceAttrState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            built: false,
            mesh_indices: [MeshId::from_index(0); 3],
            node_ids: [0; 3],
            colourmap: BuiltinColourmap::Viridis,
            opacity: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build Showcase 20: Per-Face Attributes demo.
    pub(crate) fn build_face_attr_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.face_state.scene = Scene::new();

        // Low-poly sphere so flat facets are clearly visible.
        // Build a reference copy to derive scalar data, then make three
        // independent meshes (MeshData is not Clone).
        let ref_sphere = viewport_lib::primitives::sphere(2.0, 48, 24);
        let n_tris = ref_sphere.indices.len() / 3;

        // ---- Scalar data: raw Z coordinates (not pre-normalised) ----
        //
        // We use the raw vertex Z values (-radius .. +radius) and let the
        // renderer auto-range from the stored attribute_ranges at draw time.
        // This is the same pattern as showcase_12 and avoids any mismatch
        // between the pre-normalised data and an explicit scalar_range override.
        //
        // Both Vertex and Face objects use the same underlying Z values, so
        // the renderer maps them to identical colourmap extents automatically.
        let vertex_scalars: Vec<f32> = ref_sphere.positions.iter().map(|p| p[2]).collect();

        let face_scalars: Vec<f32> = (0..n_tris)
            .map(|fi| {
                let i0 = ref_sphere.indices[fi * 3] as usize;
                let i1 = ref_sphere.indices[fi * 3 + 1] as usize;
                let i2 = ref_sphere.indices[fi * 3 + 2] as usize;
                (ref_sphere.positions[i0][2]
                    + ref_sphere.positions[i1][2]
                    + ref_sphere.positions[i2][2])
                    / 3.0
            })
            .collect();

        // ---- Rainbow colours for FaceColour object ----
        // Hue is derived from face centroid Z so every triangle in the same
        // latitude band maps to the same hue, producing closed colour rings.
        // This mirrors the scalar data on the other two spheres and makes the
        // contrast clear: same Z-based data, but colours are applied as direct
        // RGBA rather than through a colourmap.
        let face_colours: Vec<[f32; 4]> = (0..n_tris)
            .map(|fi| {
                let i0 = ref_sphere.indices[fi * 3] as usize;
                let i1 = ref_sphere.indices[fi * 3 + 1] as usize;
                let i2 = ref_sphere.indices[fi * 3 + 2] as usize;
                // Use (max+min)/2 so both triangles of each quad share the same
                // Z midpoint and therefore the same hue, producing closed rings.
                // Centroid Z differs between the two triangle halves of a quad
                // (one has 2 top verts, the other 2 bottom), which causes an
                // interlocked two-colour pattern.
                let z0 = ref_sphere.positions[i0][2];
                let z1 = ref_sphere.positions[i1][2];
                let z2 = ref_sphere.positions[i2][2];
                let z_mid = (z0.max(z1).max(z2) + z0.min(z1).min(z2)) / 2.0;
                // Map z_mid (-radius..+radius) to hue (0..360).
                let t = (z_mid + 2.0) / 4.0;
                let hue = t * 360.0;
                let [r, g, b] = hsv_to_rgb(hue, 0.80, 0.95);
                [r, g, b, 1.0]
            })
            .collect();

        let grey_mat = {
            let mut m = Material::from_colour([0.8, 0.8, 0.8]);
            m.roughness = 0.5;
            m
        };

        // ---- Mesh 0: Vertex attribute (interpolated) ----
        let mut mesh0 = viewport_lib::primitives::sphere(2.0, 48, 24);
        mesh0
            .attributes
            .insert("scalar".to_string(), AttributeData::Vertex(vertex_scalars));
        let idx0 = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &mesh0)
            .expect("face attr mesh 0");
        let node0 = self.face_state.scene.add_named(
            "Vertex (interpolated)",
            Some(idx0),
            glam::Mat4::from_translation(glam::Vec3::new(-5.0, 0.0, 0.0)),
            grey_mat,
        );

        // ---- Mesh 1: Face attribute (flat per-triangle) ----
        let mut mesh1 = viewport_lib::primitives::sphere(2.0, 48, 24);
        mesh1
            .attributes
            .insert("scalar".to_string(), AttributeData::Face(face_scalars));
        let idx1 = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &mesh1)
            .expect("face attr mesh 1");
        let node1 = self.face_state.scene.add_named(
            "Face (flat)",
            Some(idx1),
            glam::Mat4::IDENTITY,
            grey_mat,
        );

        // ---- Mesh 2: FaceColour attribute (direct RGBA, no colourmap) ----
        let mut mesh2 = viewport_lib::primitives::sphere(2.0, 48, 24);
        mesh2
            .attributes
            .insert("colour".to_string(), AttributeData::FaceColour(face_colours));
        let idx2 = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &mesh2)
            .expect("face attr mesh 2");
        let node2 = self.face_state.scene.add_named(
            "FaceColour (direct RGBA)",
            Some(idx2),
            glam::Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0)),
            {
                let mut m = Material::from_colour([1.0, 1.0, 1.0]);
                m.roughness = 0.5;
                m
            },
        );

        self.face_state.mesh_indices = [idx0, idx1, idx2];
        self.face_state.node_ids = [node0, node1, node2];
        self.face_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_face_attr(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Three spheres : same geometry, three attribute kinds:");
    ui.add_space(2.0);
    ui.label("  Left   : Vertex  (Gouraud-interpolated)");
    ui.label("  Centre : Face    (flat per-triangle)");
    ui.label("  Right  : FaceColour (direct RGBA, no colourmap)");

    ui.separator();
    ui.label("Colourmap  (Vertex + Face objects):");
    egui::ComboBox::from_id_salt("face_attr_colourmap")
        .selected_text(format!("{:?}", app.face_state.colourmap))
        .show_ui(ui, |ui| {
            for cm in [
                BuiltinColourmap::Viridis,
                BuiltinColourmap::Plasma,
                BuiltinColourmap::Magma,
                BuiltinColourmap::Inferno,
                BuiltinColourmap::Turbo,
                BuiltinColourmap::Greyscale,
                BuiltinColourmap::Coolwarm,
                BuiltinColourmap::RdBu,
                BuiltinColourmap::Rainbow,
                BuiltinColourmap::Jet,
            ] {
                ui.selectable_value(&mut app.face_state.colourmap, cm, format!("{cm:?}"));
            }
        });

    ui.separator();
    ui.label("FaceColour opacity  (tests OIT path < 1.0):");
    ui.add(egui::Slider::new(&mut app.face_state.opacity, 0.05_f32..=1.0).step_by(0.05));

    ui.separator();
    ui.weak(
        "Vertex and Face use the same scalar\n\
         (face-centroid Z).  The flat-shading\n\
         difference is visible along triangle edges.\n\
         The uniform-coloured rings at the poles are\n\
         expected: all UV-sphere pole-cap triangles\n\
         share the same centroid Z, so they map to\n\
         a single colour.  The Vertex sphere has no\n\
         ring because Gouraud interpolation reaches\n\
         the pole vertex directly.",
    );
}

// ---------------------------------------------------------------------------
// Geometry helper
// ---------------------------------------------------------------------------

/// Convert HSV (h in 0..360, s/v in 0..1) to linear RGB in 0..1.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let h6 = h / 60.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match h6 as u32 {
        0 => (c, x, 0.0_f32),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r1 + m, g1 + m, b1 + m]
}
