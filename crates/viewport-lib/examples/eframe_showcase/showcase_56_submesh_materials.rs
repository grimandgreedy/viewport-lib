//! Showcase 56: Submesh Materials
//!
//! One mesh, several materials. A small rocket is assembled into a single
//! vertex/index buffer from five primitive parts, each triangle tagged with a
//! material id, and `MeshData::sort_triangles_into_submeshes` groups the
//! triangles into contiguous ranges. At draw time the item carries one
//! material per range via `SceneRenderItem::submesh_materials`, so the whole
//! rocket is one upload and one scene item:
//!
//!   - body: brushed metal (PBR metallic, live slider)
//!   - nose + tail: red plastic
//!   - three fins: checkerboard albedo texture (three parts, one material id,
//!     merged into a single range by the sort)
//!   - canopy: tinted glass (alpha-blend albedo, drawn through the
//!     transparent pass while the other ranges stay opaque)
//!
//! Toggling "Per-range materials" off clears `submesh_materials` and the same
//! mesh falls back to the item's single material, which is exactly what a
//! consumer that never sets the field gets.

use crate::App;
use eframe::egui;
use viewport_lib as vpl;
use vpl::{
    AlphaMode, LightKind, LightSource, LightingSettings, Material, MeshData, MeshId,
    SceneRenderItem, TextureId, ViewportRenderer, primitives,
};

// Material ids used when tagging triangles. Sparse on purpose (no id 2) to
// show that `sort_triangles_into_submeshes` returns the id order for lining
// up `submesh_materials`.
const MAT_METAL: u32 = 0;
const MAT_PLASTIC: u32 = 1;
const MAT_CHECKER: u32 = 3;
const MAT_GLASS: u32 = 4;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct SubmeshState {
    pub built: bool,
    pub mesh_id: MeshId,
    /// Distinct material ids in range order, as returned by
    /// `sort_triangles_into_submeshes`.
    pub range_ids: Vec<u32>,
    pub checker_tex: Option<TextureId>,
    pub glass_tex: Option<TextureId>,

    // Controls.
    pub per_range: bool,
    pub metallic: f32,
    pub spin: bool,
    pub angle: f32,
}

impl Default for SubmeshState {
    fn default() -> Self {
        Self {
            built: false,
            mesh_id: MeshId::INVALID,
            range_ids: Vec::new(),
            checker_tex: None,
            glass_tex: None,
            per_range: true,
            metallic: 0.9,
            spin: true,
            angle: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// Append `part` into `merged`, offsetting indices and tagging every added
/// triangle with `mat_id`. `transform` is applied to positions; its rotation
/// part is applied to normals (parts only use rotation + translation here).
fn merge_part(
    merged: &mut MeshData,
    tri_ids: &mut Vec<u32>,
    part: &MeshData,
    transform: glam::Affine3A,
    mat_id: u32,
) {
    let base = merged.positions.len() as u32;
    for p in &part.positions {
        merged
            .positions
            .push(transform.transform_point3((*p).into()).into());
    }
    for n in &part.normals {
        merged
            .normals
            .push(transform.transform_vector3((*n).into()).normalize().into());
    }
    let uvs = merged.uvs.get_or_insert_with(Vec::new);
    for i in 0..part.positions.len() {
        uvs.push(
            part.uvs
                .as_ref()
                .and_then(|u| u.get(i))
                .copied()
                .unwrap_or([0.0, 0.0]),
        );
    }
    for idx in &part.indices {
        merged.indices.push(base + idx);
    }
    tri_ids.extend(std::iter::repeat(mat_id).take(part.indices.len() / 3));
}

pub(crate) fn build_submesh_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    let mut rocket = MeshData::default();
    let mut tri_ids: Vec<u32> = Vec::new();

    let at = |x: f32, y: f32, z: f32| glam::Affine3A::from_translation(glam::Vec3::new(x, y, z));

    // Body: a cylinder along +Z.
    merge_part(
        &mut rocket,
        &mut tri_ids,
        &primitives::cylinder(1.2, 4.0, 48),
        at(0.0, 0.0, 0.0),
        MAT_METAL,
    );
    // Nose cone on top, tail ring below: both plastic.
    merge_part(
        &mut rocket,
        &mut tri_ids,
        &primitives::cone(1.2, 1.8, 48),
        at(0.0, 0.0, 2.9),
        MAT_PLASTIC,
    );
    merge_part(
        &mut rocket,
        &mut tri_ids,
        &primitives::cylinder(1.35, 0.5, 48),
        at(0.0, 0.0, -2.1),
        MAT_PLASTIC,
    );
    // Three fins around the tail. Three separate parts sharing one material
    // id: the sort merges them into a single range.
    for k in 0..3 {
        let angle = k as f32 * std::f32::consts::TAU / 3.0;
        let transform = glam::Affine3A::from_rotation_z(angle)
            * glam::Affine3A::from_translation(glam::Vec3::new(1.6, 0.0, -1.7));
        merge_part(
            &mut rocket,
            &mut tri_ids,
            &primitives::cuboid_unwrapped(1.4, 0.15, 1.6),
            transform,
            MAT_CHECKER,
        );
    }
    // Canopy: a glass sphere half-sunk into the body.
    merge_part(
        &mut rocket,
        &mut tri_ids,
        &primitives::sphere(0.7, 32, 16),
        at(0.0, -1.1, 0.9),
        MAT_GLASS,
    );

    let range_ids = rocket
        .sort_triangles_into_submeshes(&tri_ids)
        .expect("one material id per triangle");

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &rocket)
        .expect("rocket mesh");

    // Checkerboard albedo for the fins.
    let checker_tex = {
        let n = 64usize;
        let mut rgba = Vec::with_capacity(n * n * 4);
        for y in 0..n {
            for x in 0..n {
                let on = ((x / 8) + (y / 8)) % 2 == 0;
                let v = if on { 235u8 } else { 40u8 };
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        renderer
            .resources_mut()
            .upload_texture(&app.device, &app.queue, n as u32, n as u32, &rgba)
            .ok()
    };
    // 1x1 translucent cyan albedo: alpha-blend materials read their alpha
    // from the sampled base colour, so this is what makes the canopy glass.
    let glass_tex = renderer
        .resources_mut()
        .upload_texture(&app.device, &app.queue, 1, 1, &[170, 230, 255, 90])
        .ok();

    app.submesh_state = SubmeshState {
        built: true,
        mesh_id,
        range_ids,
        checker_tex,
        glass_tex,
        ..SubmeshState::default()
    };
}

// ---------------------------------------------------------------------------
// Frame data
// ---------------------------------------------------------------------------

fn material_for(state: &SubmeshState, id: u32) -> Material {
    let mut m = Material::default();
    match id {
        MAT_METAL => {
            m.base_colour = [0.75, 0.77, 0.8];
            m.metallic = state.metallic;
            m.roughness = 0.35;
        }
        MAT_PLASTIC => {
            m.base_colour = [0.82, 0.15, 0.12];
            m.metallic = 0.0;
            m.roughness = 0.5;
        }
        MAT_CHECKER => {
            m.base_colour = [1.0, 1.0, 1.0];
            m.texture_id = state.checker_tex;
            m.roughness = 0.7;
        }
        MAT_GLASS => {
            m.base_colour = [1.0, 1.0, 1.0];
            m.texture_id = state.glass_tex;
            m.alpha_mode = AlphaMode::Blend;
            m.roughness = 0.1;
            m.specular = 0.9;
        }
        _ => {}
    }
    m
}

pub(crate) fn submesh_scene_items(app: &App) -> Vec<SceneRenderItem> {
    let s = &app.submesh_state;
    if !s.built {
        return vec![];
    }
    let mut item = SceneRenderItem::default();
    item.mesh_id = s.mesh_id;
    item.model = glam::Mat4::from_rotation_z(s.angle).to_cols_array_2d();
    // The single-material fallback look, and the whole look when per-range
    // materials are toggled off.
    item.material.base_colour = [0.6, 0.6, 0.65];
    item.material.roughness = 0.5;
    if s.per_range {
        item.submesh_materials = Some(s.range_ids.iter().map(|&id| material_for(s, id)).collect());
    }
    vec![item]
}

pub(crate) fn submesh_lighting() -> LightingSettings {
    let mut t = LightingSettings::default();
    t.lights = vec![
        {
            let mut l = LightSource::default();
            l.kind = LightKind::Directional {
                direction: [0.45, 0.35, -0.8],
            };
            l.intensity = 1.0;
            l
        },
        {
            let mut l = LightSource::default();
            l.kind = LightKind::Directional {
                direction: [-0.5, -0.4, -0.3],
            };
            l.colour = [0.85, 0.9, 1.0];
            l.intensity = 0.35;
            l
        },
    ];
    t.shadows.enabled = false;
    t.hemisphere_intensity = 0.35;
    t
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_submesh(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.submesh_state;
    ui.checkbox(&mut s.per_range, "Per-range materials");
    ui.add(egui::Slider::new(&mut s.metallic, 0.0..=1.0).text("body metallic"));
    ui.checkbox(&mut s.spin, "Spin");
    ui.separator();
    ui.label("One mesh, one item, four material ranges:");
    ui.label("metal body, plastic nose and tail, three checker fins");
    ui.label("(one shared range), and an alpha-blend glass canopy.");
    ui.label("Triangles were tagged per part and grouped with");
    ui.label("MeshData::sort_triangles_into_submeshes.");
}
