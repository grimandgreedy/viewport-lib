//! Showcase 40: GPU Vertex Warp
//!
//! Demonstrates `warp_attribute` and `warp_scale` on `SceneRenderItem`.
//!
//! A per-vertex displacement field is uploaded once alongside each mesh as an
//! `AttributeData::VertexVector` attribute. The vertex shader reads the field
//! from a storage buffer and adds `warp_scale * displacement` to each vertex
//! position in local space. No CPU re-upload is needed to animate the effect.
//!
//! Three meshes are shown side by side with different displacement patterns:
//!   - Left: a subdivided plane with a sinusoidal height field (standing waves).
//!   - Centre: a sphere with a 4-lobe azimuthal displacement (flower mode).
//!   - Right: a sphere with a Y2,0 spherical harmonic displacement (elongation mode).
//!
//! The warp_scale slider drives all three simultaneously.

use crate::App;
use eframe::egui;
use viewport_lib::{
    AttributeData, BackfacePolicy, LightKind, LightSource, LightingSettings, MeshId,
    SceneRenderItem, ViewportRenderer, primitives,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct VertexWarpState {
    pub built: bool,
    pub mesh_ids: [MeshId; 3],
    pub scale: f32,
}

impl Default for VertexWarpState {
    fn default() -> Self {
        Self {
            built: false,
            mesh_ids: [MeshId::from_index(0); 3],
            scale: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

pub(crate) fn build_warp_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    // Left: wavy plane.
    let plane_id = {
        let mut mesh = primitives::grid_plane(5.0, 5.0, 48, 48);
        let disp: Vec<[f32; 3]> = mesh
            .positions
            .iter()
            .map(|&[x, _, z]| {
                let freq = std::f32::consts::TAU / 5.0;
                let h = (freq * x).sin() * (freq * z).cos();
                [0.0, h, 0.0]
            })
            .collect();
        mesh.attributes
            .insert("warp".to_string(), AttributeData::VertexVector(disp));
        renderer
            .resources_mut()
            .upload_mesh_data(&app.device, &mesh)
            .expect("warp plane")
    };

    // Centre: sphere with 4 azimuthal lobes.
    let sphere_4lobe_id = {
        let mut mesh = primitives::sphere(1.5, 64, 32);
        let disp: Vec<[f32; 3]> = mesh
            .positions
            .iter()
            .zip(mesh.normals.iter())
            .map(|(&[x, y, _z], &nrm)| {
                let phi = y.atan2(x); // azimuth around Z axis
                let amp = (4.0 * phi).sin();
                [nrm[0] * amp, nrm[1] * amp, nrm[2] * amp]
            })
            .collect();
        mesh.attributes
            .insert("warp".to_string(), AttributeData::VertexVector(disp));
        renderer
            .resources_mut()
            .upload_mesh_data(&app.device, &mesh)
            .expect("warp sphere 4lobe")
    };

    // Right: sphere with Y2,0 elongation mode.
    let sphere_y20_id = {
        let mut mesh = primitives::sphere(1.5, 64, 32);
        let disp: Vec<[f32; 3]> = mesh
            .positions
            .iter()
            .zip(mesh.normals.iter())
            .map(|(&[x, y, z], &nrm)| {
                let r2 = x * x + y * y + z * z;
                // Y2,0 harmonic: (3z^2 - r^2) / r^2, normalised so peak is 1.
                let amp = (3.0 * z * z - r2) / r2;
                [nrm[0] * amp, nrm[1] * amp, nrm[2] * amp]
            })
            .collect();
        mesh.attributes
            .insert("warp".to_string(), AttributeData::VertexVector(disp));
        renderer
            .resources_mut()
            .upload_mesh_data(&app.device, &mesh)
            .expect("warp sphere y20")
    };

    app.warp_state.mesh_ids = [plane_id, sphere_4lobe_id, sphere_y20_id];
    app.warp_state.built = true;
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_warp(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Warp scale:");
    ui.add(egui::Slider::new(&mut app.warp_state.scale, -1.5..=1.5).step_by(0.01));

    ui.separator();

    ui.label("Left: wavy plane (sinusoidal height field)");
    ui.label("Centre: sphere, 4-lobe azimuthal mode");
    ui.label("Right: sphere, Y2,0 elongation mode");
    ui.separator();
    ui.label(
        "Displacement vectors are baked into each mesh once at load time.\n\
              The vertex shader scales them by warp_scale each frame with no CPU re-upload.",
    );
}

// ---------------------------------------------------------------------------
// Frame data
// ---------------------------------------------------------------------------

pub(crate) fn warp_scene_items(app: &App) -> Vec<SceneRenderItem> {
    if !app.warp_state.built {
        return vec![];
    }

    let [plane_id, lobe_id, y20_id] = app.warp_state.mesh_ids;

    // Colours: coral, steel blue, sage green.
    let colours: [[f32; 3]; 3] = [[0.85, 0.40, 0.25], [0.30, 0.55, 0.85], [0.35, 0.72, 0.42]];
    let offsets: [f32; 3] = [-4.5, 0.0, 4.5];
    let ids = [plane_id, lobe_id, y20_id];

    ids.iter()
        .zip(colours.iter())
        .zip(offsets.iter())
        .map(|((&mesh_id, &colour), &tx)| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(tx, 0.0, 0.0)).to_cols_array_2d();
            item.material.backface_policy = BackfacePolicy::Identical;
            item.material.base_colour = colour;
            item.material.specular = 0.15;
            item.warp_attribute = Some("warp".to_string());
            item.warp_scale = app.warp_state.scale;
            item
        })
        .collect()
}

pub(crate) fn warp_lighting() -> LightingSettings {
    // Two opposing soft lights so all sides of the mesh are visible regardless
    // of deformation direction, with a neutral hemisphere fill.
    {
        let mut _t = LightingSettings::default();
        _t.lights = vec![
            {
                let mut _t = LightSource::default();
                _t.kind = LightKind::Directional {
                    direction: [0.3, 0.8, 0.5],
                };
                _t.colour = [1.0, 1.0, 1.0];
                _t.intensity = 0.7;
                _t
            },
            {
                let mut _t = LightSource::default();
                _t.kind = LightKind::Directional {
                    direction: [-0.3, -0.5, -0.5],
                };
                _t.colour = [0.8, 0.85, 1.0];
                _t.intensity = 0.3;
                _t
            },
        ];
        _t.shadows_enabled = false;
        _t.hemisphere_intensity = 0.35;
        _t.sky_colour = [0.9, 0.92, 1.0];
        _t.ground_colour = [0.5, 0.5, 0.55];
        _t
    }
}
