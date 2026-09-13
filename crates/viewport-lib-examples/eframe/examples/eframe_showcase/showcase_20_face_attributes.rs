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
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{
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
            mesh_indices: [MeshId::INVALID; 3],
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
        let ref_sphere = vpl::primitives::sphere(2.0, 48, 24);
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
        let mut mesh0 = vpl::primitives::sphere(2.0, 48, 24);
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
        let mut mesh1 = vpl::primitives::sphere(2.0, 48, 24);
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
        let mut mesh2 = vpl::primitives::sphere(2.0, 48, 24);
        mesh2.attributes.insert(
            "colour".to_string(),
            AttributeData::FaceColour(face_colours),
        );
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

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.face_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_face_attr_scene(renderer);
    app.camera = vpl::Camera {
        center: glam::Vec3::ZERO,
        distance: 16.0,
        orientation: glam::Quat::from_rotation_z(0.5)
            * glam::Quat::from_rotation_x(1.1),
        ..vpl::Camera::default()
    };
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. `_out` carries
/// the few extra frame settings a showcase can set alongside its items.
pub(crate) fn scene(
    app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    _out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    let (items, bg_colour, lighting, scene_gen, sel_gen) = {
        let mut items = app
            .face_state
            .scene
            .collect_render_items(&vpl::Selection::new());
        let colourmap_id = vpl::ColourmapId(app.face_state.colourmap as usize);

        // Node 0: Vertex attribute (interpolated)
        // scalar_range left as None : renderer auto-detects from attribute_ranges.
        if let Some(item) = items
            .iter_mut()
            .find(|i| i.settings.pick_id == vpl::PickId(app.face_state.node_ids[0]))
        {
            item.active_attribute = Some(vpl::AttributeRef {
                name: "scalar".to_string(),
                kind: vpl::AttributeKind::Vertex,
            });
            item.colourmap_id = Some(colourmap_id);
        }

        // Node 1: Face attribute (flat per-triangle)
        // scalar_range left as None : renderer auto-detects from attribute_ranges.
        if let Some(item) = items
            .iter_mut()
            .find(|i| i.settings.pick_id == vpl::PickId(app.face_state.node_ids[1]))
        {
            item.active_attribute = Some(vpl::AttributeRef {
                name: "scalar".to_string(),
                kind: vpl::AttributeKind::Face,
            });
            item.colourmap_id = Some(colourmap_id);
        }

        // Node 2: FaceColour attribute (direct RGBA, no colourmap)
        if let Some(item) = items
            .iter_mut()
            .find(|i| i.settings.pick_id == vpl::PickId(app.face_state.node_ids[2]))
        {
            item.active_attribute = Some(vpl::AttributeRef {
                name: "colour".to_string(),
                kind: vpl::AttributeKind::FaceColour,
            });
            item.settings.opacity = app.face_state.opacity;
        }

        let sg = app.face_state.scene.version();
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.hemisphere_intensity = 0.4;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        (items, None, lighting, sg, 0)
    };
    crate::SceneContents {
        items,
        bg_colour,
        lighting,
        scene_gen,
        sel_gen,
    }
}

// ---------------------------------------------------------------------------
// Per-frame frame-data tweaks
// ---------------------------------------------------------------------------

/// Fold this showcase's own contributions into the assembled frame: extra
/// render items, overlays, and effect settings that are re-submitted every
/// frame rather than baked into the scene.


// ---------------------------------------------------------------------------
// Viewport overlay and per-frame tick
// ---------------------------------------------------------------------------

/// Draw this showcase's own egui overlay on top of the rendered viewport:
/// selection rectangles, mode readouts, and in-scene labels.


/// Advance this showcase's animation and ask for another frame. Runs after the
/// viewport has been drawn, so it only affects the next frame.


/// Route a viewport click for this showcase. The host calls this for a plain
/// click that no gizmo or widget has already consumed; `pos` is in viewport
/// pixels.


/// Handle drag gestures this showcase owns, before the camera controller runs.


/// Advance this showcase's own camera animation or object motion for the frame.


/// Update this showcase's interactive widgets for the frame.


/// Flush any per-frame GPU writes this showcase has queued.


/// Cache gizmo placement for next frame's hit-testing.


/// Take over the whole viewport for this frame. Returning false leaves the
/// host's normal single-viewport path in charge.
pub(crate) fn viewport_override(
    _app: &mut crate::App,
    _ui: &mut crate::eframe::egui::Ui,
    _cx: &crate::ViewportCtx,
) -> bool {
    false
}

/// Drive the orbit controller for this showcase. Returning false leaves the
/// host to run the usual suppress-or-apply path.
pub(crate) fn drive_camera(_app: &mut crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

/// Whether the orbit controller should resolve without moving the camera this
/// frame. This showcase never suppresses it.
pub(crate) fn suppress_orbit(_app: &crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScFaceAttributes;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScFaceAttributes = ScFaceAttributes;

impl crate::Showcase for ScFaceAttributes {
    fn needs_build(&self, app: &crate::App) -> bool {
        needs_build(app)
    }
    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        build(app, renderer)
    }
    fn scene(&self, app: &mut crate::App, frame: &crate::eframe::Frame, out: &mut crate::SceneOverrides) -> crate::SceneContents {
        scene(app, frame, out)
    }
    fn viewport_override(&self, app: &mut crate::App, ui: &mut crate::eframe::egui::Ui, cx: &crate::ViewportCtx) -> bool {
        viewport_override(app, ui, cx)
    }
    fn drive_camera(&self, app: &mut crate::App, cx: &crate::ViewportCtx) -> bool {
        drive_camera(app, cx)
    }
    fn suppress_orbit(&self, app: &crate::App, cx: &crate::ViewportCtx) -> bool {
        suppress_orbit(app, cx)
    }
    fn controls(&self, app: &mut crate::App, ui: &mut crate::eframe::egui::Ui, _frame: &crate::eframe::Frame) {
        controls_face_attr(app, ui)
    }
}
