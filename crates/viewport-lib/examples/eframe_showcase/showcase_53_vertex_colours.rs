//! Showcase 53: Vertex Colours & Painting
//!
//! Three meshes in a row exercise the per-vertex colour input to the standard
//! PBR mesh path, in the three ways a consumer uses it:
//!
//!   - Left (static): colours baked into the mesh at upload via
//!     `MeshData::vertex_colours`. This is the glTF `COLOR_0` / baked-AO case:
//!     the colour is multiplied into the base colour before lighting, so the
//!     mesh still shades with full PBR, materials, and shadows.
//!   - Centre (painted): a plain mesh that starts light grey and is painted
//!     interactively. Turn on "Paint mode", then drag on the mesh: each stroke
//!     writes only the touched vertices with `update_vertex_colours`, an
//!     in-place GPU write, not a whole-mesh re-upload.
//!   - Right (animated): a grid whose vertex colours are recomputed every frame
//!     as a travelling wave, again through `update_vertex_colours`, to show the
//!     same in-place path driving per-frame animation.
//!
//! The interactive brush ray-casts against a CPU copy of the centre mesh (a
//! plain Moller-Trumbore triangle test) to find the hit point, then blends the
//! brush colour into every vertex within the brush radius.

use crate::App;
use eframe::egui;
use viewport_lib::{
    BackfacePolicy, LightKind, LightSource, LightingSettings, MeshId, SceneRenderItem,
    ViewportRenderer, primitives,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct VertexColourState {
    pub built: bool,

    // Left: static baked colours.
    pub static_id: MeshId,

    // Centre: interactive paint target.
    pub paint_id: MeshId,
    pub paint_positions: Vec<[f32; 3]>,
    pub paint_indices: Vec<u32>,
    pub paint_colours: Vec<[f32; 4]>,
    pub paint_origin: glam::Vec3,

    // Right: per-frame animated colours.
    pub anim_id: MeshId,
    pub anim_positions: Vec<[f32; 3]>,
    pub anim_colours: Vec<[f32; 4]>,
    pub anim_origin: glam::Vec3,
    pub anim_t: f32,

    // Controls.
    pub paint_mode: bool,
    pub animate: bool,
    pub brush_colour: [f32; 3],
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub clear_requested: bool,
}

impl Default for VertexColourState {
    fn default() -> Self {
        Self {
            built: false,
            static_id: MeshId::INVALID,
            paint_id: MeshId::INVALID,
            paint_positions: Vec::new(),
            paint_indices: Vec::new(),
            paint_colours: Vec::new(),
            paint_origin: glam::Vec3::ZERO,
            anim_id: MeshId::INVALID,
            anim_positions: Vec::new(),
            anim_colours: Vec::new(),
            anim_origin: glam::Vec3::ZERO,
            anim_t: 0.0,
            paint_mode: true,
            animate: true,
            brush_colour: [0.85, 0.25, 0.30],
            brush_radius: 0.9,
            brush_strength: 0.6,
            clear_requested: false,
        }
    }
}

const UNPAINTED: [f32; 4] = [0.8, 0.8, 0.8, 1.0];

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

pub(crate) fn build_vertex_colour_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    let static_origin = glam::Vec3::new(-5.5, 0.0, 0.0);
    let paint_origin = glam::Vec3::new(0.0, 0.0, 0.0);
    let anim_origin = glam::Vec3::new(5.5, 0.0, 0.0);

    // Left: a sphere with colours baked in. Z-up, so colour by height (z) plus a
    // darkened underside to read like cheap baked ambient occlusion.
    let static_id = {
        let mut mesh = primitives::sphere(2.0, 64, 32);
        let colours: Vec<[f32; 4]> = mesh
            .positions
            .iter()
            .map(|&[_, _, z]| {
                let t = (z / 2.0 * 0.5 + 0.5).clamp(0.0, 1.0);
                // Warm top fading to cool base, darkened toward the bottom.
                let ao = (0.45 + 0.55 * t).clamp(0.0, 1.0);
                let r = (0.95 * t + 0.15) * ao;
                let g = (0.55 * t + 0.25) * ao;
                let b = (0.30 * (1.0 - t) + 0.35) * ao;
                [r, g, b, 1.0]
            })
            .collect();
        mesh.vertex_colours = Some(colours);
        renderer
            .resources_mut()
            .upload_mesh_data(&app.device, &mesh)
            .expect("static vertex-colour sphere")
    };

    // Centre: an icosphere painted interactively. Uploaded plain (all vertices
    // start at UNPAINTED); colours are pushed in place as the user paints.
    let (paint_id, paint_positions, paint_indices) = {
        let mut mesh = primitives::icosphere(2.0, 4);
        let colours = vec![UNPAINTED; mesh.positions.len()];
        mesh.vertex_colours = Some(colours);
        let positions = mesh.positions.clone();
        let indices = mesh.indices.clone();
        let id = renderer
            .resources_mut()
            .upload_mesh_data(&app.device, &mesh)
            .expect("paintable icosphere");
        (id, positions, indices)
    };

    // Right: a subdivided plane whose colours animate each frame.
    let (anim_id, anim_positions) = {
        let mut mesh = primitives::grid_plane(4.0, 4.0, 64, 64);
        let colours = vec![UNPAINTED; mesh.positions.len()];
        mesh.vertex_colours = Some(colours);
        let positions = mesh.positions.clone();
        let id = renderer
            .resources_mut()
            .upload_mesh_data(&app.device, &mesh)
            .expect("animated colour plane");
        (id, positions)
    };

    let n_paint = paint_positions.len();
    let n_anim = anim_positions.len();

    // `static_origin` is captured into the item-build path via a fixed constant;
    // keep the sphere's origin in sync there.
    let _ = static_origin;

    app.vcol_state = VertexColourState {
        built: true,
        static_id,
        paint_id,
        paint_positions,
        paint_indices,
        paint_colours: vec![UNPAINTED; n_paint],
        paint_origin,
        anim_id,
        anim_positions,
        anim_colours: vec![UNPAINTED; n_anim],
        anim_origin,
        anim_t: 0.0,
        ..VertexColourState::default()
    };
}

// ---------------------------------------------------------------------------
// Per-frame drivers (called with renderer + queue in scope)
// ---------------------------------------------------------------------------

/// Recompute the animated grid's colours as a travelling wave and push the whole
/// buffer in place. Demonstrates `update_vertex_colours` driving animation.
pub(crate) fn vcol_animate(
    state: &mut VertexColourState,
    renderer: &mut ViewportRenderer,
    queue: &eframe::wgpu::Queue,
    dt: f32,
) {
    state.anim_t += dt;
    let t = state.anim_t;
    for (i, &[x, y, _z]) in state.anim_positions.iter().enumerate() {
        let phase = 1.4 * x + 1.1 * y + t * 2.2;
        // Cosine palette: smooth, saturated, cycles through the spectrum.
        let r = 0.5 + 0.5 * (phase).cos();
        let g = 0.5 + 0.5 * (phase + 2.094).cos();
        let b = 0.5 + 0.5 * (phase + 4.188).cos();
        state.anim_colours[i] = [r, g, b, 1.0];
    }
    let _ = renderer.resources_mut().update_vertex_colours(
        queue,
        state.anim_id,
        0,
        &state.anim_colours,
    );
}

/// Ray-cast the cursor against the centre mesh and paint every vertex within the
/// brush radius, writing only the touched vertices in place.
pub(crate) fn vcol_paint(
    state: &mut VertexColourState,
    renderer: &mut ViewportRenderer,
    queue: &eframe::wgpu::Queue,
    cursor: glam::Vec2,
    w: f32,
    h: f32,
    view_proj: glam::Mat4,
) {
    let vp_inv = view_proj.inverse();
    let (ro, rd) = viewport_lib::picking::screen_to_ray(cursor, glam::Vec2::new(w, h), vp_inv);
    // Model is a pure translation, so local space is just the ray shifted by the
    // mesh origin.
    let lo = ro - state.paint_origin;
    let Some(hit) = raycast_mesh(&state.paint_positions, &state.paint_indices, lo, rd) else {
        return;
    };

    let radius = state.brush_radius.max(1e-3);
    let strength = state.brush_strength.clamp(0.0, 1.0);
    let target = [
        state.brush_colour[0],
        state.brush_colour[1],
        state.brush_colour[2],
        1.0,
    ];

    for (v, p) in state.paint_positions.iter().enumerate() {
        let d = (glam::Vec3::from(*p) - hit).length();
        if d > radius {
            continue;
        }
        // Smooth falloff toward the brush edge, scaled by strength.
        let falloff = 1.0 - d / radius;
        let wgt = falloff * falloff * strength;
        let cur = state.paint_colours[v];
        let blended = [
            cur[0] + (target[0] - cur[0]) * wgt,
            cur[1] + (target[1] - cur[1]) * wgt,
            cur[2] + (target[2] - cur[2]) * wgt,
            1.0,
        ];
        state.paint_colours[v] = blended;
        let _ =
            renderer
                .resources_mut()
                .update_vertex_colours(queue, state.paint_id, v, &[blended]);
    }
}

/// Reset the painted mesh back to the unpainted grey and flush.
pub(crate) fn vcol_clear_paint(
    state: &mut VertexColourState,
    renderer: &mut ViewportRenderer,
    queue: &eframe::wgpu::Queue,
) {
    for c in state.paint_colours.iter_mut() {
        *c = UNPAINTED;
    }
    let _ = renderer.resources_mut().update_vertex_colours(
        queue,
        state.paint_id,
        0,
        &state.paint_colours,
    );
}

/// Nearest positive-t ray/triangle hit against a CPU triangle list, returning
/// the hit point in the mesh's local space. Plain Moller-Trumbore.
fn raycast_mesh(
    positions: &[[f32; 3]],
    indices: &[u32],
    origin: glam::Vec3,
    dir: glam::Vec3,
) -> Option<glam::Vec3> {
    let dir = dir.normalize_or_zero();
    if dir == glam::Vec3::ZERO {
        return None;
    }
    let mut best_t = f32::INFINITY;
    let mut best_hit = None;
    for tri in indices.chunks_exact(3) {
        let a = glam::Vec3::from(positions[tri[0] as usize]);
        let b = glam::Vec3::from(positions[tri[1] as usize]);
        let c = glam::Vec3::from(positions[tri[2] as usize]);
        let e1 = b - a;
        let e2 = c - a;
        let pvec = dir.cross(e2);
        let det = e1.dot(pvec);
        if det.abs() < 1e-7 {
            continue;
        }
        let inv_det = 1.0 / det;
        let tvec = origin - a;
        let u = tvec.dot(pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let qvec = tvec.cross(e1);
        let v = dir.dot(qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = e2.dot(qvec) * inv_det;
        if t > 1e-4 && t < best_t {
            best_t = t;
            best_hit = Some(origin + dir * t);
        }
    }
    best_hit
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_vertex_colour(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.vcol_state;

    ui.checkbox(&mut s.paint_mode, "Paint mode (drag on the centre mesh)");
    ui.label("With paint mode on, left-drag paints; orbit is suppressed.");
    ui.separator();

    ui.label("Brush colour:");
    ui.color_edit_button_rgb(&mut s.brush_colour);
    ui.add(egui::Slider::new(&mut s.brush_radius, 0.1..=2.0).text("radius"));
    ui.add(egui::Slider::new(&mut s.brush_strength, 0.02..=1.0).text("strength"));
    s.clear_requested |= ui.button("Clear painted colours").clicked();

    ui.separator();
    ui.checkbox(&mut s.animate, "Animate right-hand grid");

    ui.separator();
    ui.label("Left: colours baked at upload (MeshData::vertex_colours).");
    ui.label("Centre: painted in place (update_vertex_colours).");
    ui.label("Right: per-frame colour wave (update_vertex_colours).");
    ui.label("All three keep full PBR lighting: colour multiplies base colour.");
}

// ---------------------------------------------------------------------------
// Frame data
// ---------------------------------------------------------------------------

pub(crate) fn vcol_scene_items(app: &App) -> Vec<SceneRenderItem> {
    let s = &app.vcol_state;
    if !s.built {
        return vec![];
    }

    let make = |mesh_id: MeshId, origin: glam::Vec3| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model = glam::Mat4::from_translation(origin).to_cols_array_2d();
        // White base colour so the per-vertex colour shows unmodified; PBR
        // shading still applies on top.
        item.material.base_colour = [1.0, 1.0, 1.0];
        item.material.specular = 0.2;
        item.material.backface_policy = BackfacePolicy::Identical;
        item
    };

    vec![
        make(s.static_id, glam::Vec3::new(-5.5, 0.0, 0.0)),
        make(s.paint_id, s.paint_origin),
        make(s.anim_id, s.anim_origin),
    ]
}

pub(crate) fn vcol_lighting() -> LightingSettings {
    let mut t = LightingSettings::default();
    t.lights = vec![
        {
            let mut l = LightSource::default();
            l.kind = LightKind::Directional {
                direction: [0.4, 0.5, -0.75],
            };
            l.colour = [1.0, 1.0, 1.0];
            l.intensity = 0.8;
            l
        },
        {
            let mut l = LightSource::default();
            l.kind = LightKind::Directional {
                direction: [-0.5, -0.3, -0.4],
            };
            l.colour = [0.85, 0.9, 1.0];
            l.intensity = 0.3;
            l
        },
    ];
    t.shadows.enabled = false;
    t.hemisphere_intensity = 0.4;
    t.sky_colour = [0.95, 0.96, 1.0];
    t.ground_colour = [0.5, 0.5, 0.55];
    t
}
