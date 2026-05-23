//! Showcase 48: Screen-Space Decals
//!
//! Demonstrates the screen-space decal system (D1 + D2).
//!
//! Scene: a concrete floor and a wall meeting at a corner. Click anywhere on
//! the geometry to stamp a decal at the hit point. The decal is oriented with
//! its projection axis along the surface normal so it lies flat on the surface
//! it hits.
//!
//! D2 extension: each decal can optionally use a normal map (bullet-hole
//! crater). A per-decal blend-strength slider controls how strongly the normal
//! map perturbs the surface shading. When the normal map toggle is off the
//! decals are simple alpha-blended stickers (D1 behaviour).
//!
//! Controls:
//! - Click on the floor or wall to stamp a decal.
//! - Decal list: shows placed decals with a delete button per entry.
//! - Size / Depth sliders: control the footprint and projection depth.
//! - Normal map toggle: switches between D1 (sticker) and D2 (crater).
//! - Normal blend strength slider: scales the D2 shading effect.
//! - Blend mode selector: Replace (alpha blend) or Multiply.
//! - Clear All: removes all placed decals.

use eframe::egui;
use viewport_lib::{
    DecalBlendMode, DecalItem, Material, MeshId, SceneRenderItem,
    scene::{Scene, material::BackfacePolicy},
    selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------

/// Circular decal albedo: soft white disc on a transparent background.
fn make_disc_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let c = size as f32 * 0.5;
    let r = c * 0.9;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let d = (dx * dx + dy * dy).sqrt();
            let alpha = ((r - d) / (r * 0.15)).clamp(0.0, 1.0);
            let idx = ((y * size + x) * 4) as usize;
            buf[idx]     = 230;
            buf[idx + 1] = 220;
            buf[idx + 2] = 210;
            buf[idx + 3] = (alpha * 220.0) as u8;
        }
    }
    buf
}

/// Bullet-hole crater normal map: encodes an inward dome shape.
/// Centre normal points inward (-Z in tangent space), rim normals tilt outward.
fn make_crater_normal_map(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let c = size as f32 * 0.5;
    let r = c * 0.85;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let d = (dx * dx + dy * dy).sqrt();
            let t = (d / r).clamp(0.0, 1.0);
            // At the centre: normal points straight in (-Z -> (0,0,-1) in tangent space,
            // encoded as (0.5, 0.5, 0.0) in the map). At the rim: flat surface (0,0,1).
            let nz = t;        // 0 at centre, 1 at rim
            let scale = (1.0 - nz * nz).sqrt(); // length of the XY deflection
            let (nx, ny) = if d > 0.001 {
                (dx / d * scale * 0.6, dy / d * scale * 0.6)
            } else {
                (0.0, 0.0)
            };
            // Re-normalise.
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
            let (nx, ny, nz) = (nx / len, ny / len, nz / len);
            let idx = ((y * size + x) * 4) as usize;
            buf[idx]     = ((nx * 0.5 + 0.5) * 255.0) as u8;
            buf[idx + 1] = ((ny * 0.5 + 0.5) * 255.0) as u8;
            buf[idx + 2] = ((nz * 0.5 + 0.5) * 255.0) as u8;
            buf[idx + 3] = 255;
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Flat quad in the XY plane, centred at the origin, half-extent `e`.
/// Vertices are in Z-up space; face normal is +Z.
fn flat_quad(ex: f32, ey: f32) -> viewport_lib::MeshData {
    let mut mesh = viewport_lib::MeshData::default();
    mesh.positions = vec![
        [-ex, -ey, 0.0],
        [ ex, -ey, 0.0],
        [ ex,  ey, 0.0],
        [-ex,  ey, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.uvs = Some(vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]);
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh
}

// ---------------------------------------------------------------------------
// Decal transform builder
// ---------------------------------------------------------------------------

/// Build a model matrix for a decal placed at `hit` on a surface with `normal`.
/// `size` controls the footprint; `depth` controls how far the box extends
/// along the projection axis.
fn decal_transform(hit: glam::Vec3, normal: glam::Vec3, size: f32, depth: f32) -> [[f32; 4]; 4] {
    let n = normal.normalize();
    let ref_up = if n.abs_diff_eq(glam::Vec3::Y, 0.9) {
        glam::Vec3::Z
    } else {
        glam::Vec3::Y
    };
    let tangent = ref_up.cross(n).normalize();
    let bitangent = n.cross(tangent).normalize();
    glam::Mat4::from_cols(
        (tangent   * size).extend(0.0),
        (bitangent * size).extend(0.0),
        (n         * depth).extend(0.0),
        hit.extend(1.0),
    ).to_cols_array_2d()
}

// ---------------------------------------------------------------------------
// Per-placed decal record
// ---------------------------------------------------------------------------

pub(crate) struct PlacedDecal {
    id: u64,
    label: String,
    pub(crate) hit: glam::Vec3,
    pub(crate) normal: glam::Vec3,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct Decal48State {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub floor_mesh: Option<MeshId>,
    pub wall_mesh:  Option<MeshId>,
    pub albedo_tex: Option<u64>,
    pub normal_tex: Option<u64>,
    pub decals: Vec<PlacedDecal>,
    pub next_id: u64,
    pub decal_size: f32,
    pub decal_depth: f32,
    pub use_normal_map: bool,
    pub normal_blend: f32,
    pub blend_mode: DecalBlendMode,
    pub alpha: f32,
}

impl Default for Decal48State {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            floor_mesh: None,
            wall_mesh:  None,
            albedo_tex: None,
            normal_tex: None,
            decals: Vec::new(),
            next_id: 1,
            decal_size: 1.2,
            decal_depth: 1.5,
            use_normal_map: true,
            normal_blend: 0.8,
            blend_mode: DecalBlendMode::Replace,
            alpha: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

pub(crate) fn build_decal48_scene(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    app.decal48_state.scene = Scene::new();

    let res = renderer.resources_mut();

    // Upload textures.
    let disc = make_disc_texture(128);
    let albedo_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &disc)
        .expect("decal albedo upload");

    let crater = make_crater_normal_map(128);
    let normal_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &crater)
        .expect("decal normal map upload");

    app.decal48_state.albedo_tex = Some(albedo_id);
    app.decal48_state.normal_tex = Some(normal_id);

    // Floor: 8x8 quad in XY plane at z=0.
    let floor_data = flat_quad(4.0, 4.0);
    let floor_id = res
        .upload_mesh_data(&app.device, &floor_data)
        .expect("floor mesh upload");
    app.decal48_state.floor_mesh = Some(floor_id);

    // Wall: 8x4 quad in the XZ plane (vertical, Z-up) at y=4.
    // Normal = -Y (faces the -Y direction, toward the viewer at y < 4).
    // Vertices listed so that front face (CCW from -Y side) winds correctly.
    // Z runs bottom (z=0) to top (z=4), matching the Z-up floor seam at z=0.
    let wall_data = {
        let mut mesh = viewport_lib::MeshData::default();
        mesh.positions = vec![
            [-4.0_f32, 0.0, 0.0], // 0: left  bottom
            [ 4.0,     0.0, 0.0], // 1: right bottom
            [ 4.0,     0.0, 4.0], // 2: right top
            [-4.0,     0.0, 4.0], // 3: left  top
        ];
        mesh.normals = vec![[0.0_f32, -1.0, 0.0]; 4];
        mesh.uvs = Some(vec![
            [0.0_f32, 0.0], // left  bottom
            [1.0,     0.0], // right bottom
            [1.0,     1.0], // right top
            [0.0,     1.0], // left  top
        ]);
        // CCW from the -Y (front) side: 0,1,2 and 0,2,3.
        mesh.indices = vec![0u32, 1, 2, 0, 2, 3];
        mesh
    };
    let wall_id = res
        .upload_mesh_data(&app.device, &wall_data)
        .expect("wall mesh upload");
    app.decal48_state.wall_mesh = Some(wall_id);

    let scene = &mut app.decal48_state.scene;

    // Floor node: placed at z=0. DoubleSided so it stays visible from any angle.
    let mut floor_mat = Material::from_colour([0.55, 0.52, 0.48]);
    floor_mat.backface_policy = BackfacePolicy::Identical;
    scene.add(Some(floor_id), glam::Mat4::IDENTITY, floor_mat);

    // Wall node: translated to y=4. DoubleSided for the same reason.
    let mut wall_mat = Material::from_colour([0.50, 0.48, 0.45]);
    wall_mat.backface_policy = BackfacePolicy::Identical;
    scene.add(
        Some(wall_id),
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 4.0, 0.0)),
        wall_mat,
    );

    app.decal48_state.built = true;
}

// ---------------------------------------------------------------------------
// Click handling
// ---------------------------------------------------------------------------

/// Test `ray` against the floor (z=0, XY plane) and wall (y=4, XZ plane).
/// Returns (hit_point, surface_normal) for the closer positive intersection.
pub(crate) fn decal48_ray_hit(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
) -> Option<(glam::Vec3, glam::Vec3)> {
    // Floor: z = 0, normal = +Z.
    let t_floor = if ray_dir.z.abs() > 1e-6 {
        let t = -ray_origin.z / ray_dir.z;
        if t > 0.001 {
            let hit = ray_origin + ray_dir * t;
            // Within the floor quad [-4, 4]^2.
            if hit.x.abs() <= 4.0 && hit.y.abs() <= 4.0 { Some((t, hit, glam::Vec3::Z)) }
            else { None }
        } else { None }
    } else { None };

    // Wall: y = 4, normal = -Y (faces the -Y direction toward viewer).
    let t_wall = if ray_dir.y.abs() > 1e-6 {
        let t = (4.0 - ray_origin.y) / ray_dir.y;
        if t > 0.001 {
            let hit = ray_origin + ray_dir * t;
            // Within the wall quad: x in [-4, 4], z in [0, 4].
            if hit.x.abs() <= 4.0 && hit.z >= 0.0 && hit.z <= 4.0 {
                Some((t, hit, glam::Vec3::new(0.0, -1.0, 0.0)))
            } else { None }
        } else { None }
    } else { None };

    match (t_floor, t_wall) {
        (Some((tf, hf, nf)), Some((tw, hw, nw))) => {
            if tf < tw { Some((hf, nf)) } else { Some((hw, nw)) }
        }
        (Some((_, h, n)), None) | (None, Some((_, h, n))) => Some((h, n)),
        (None, None) => None,
    }
}

/// Place a decal at the clicked viewport position.
pub(crate) fn decal48_place(
    app: &mut App,
    cursor: glam::Vec2,
    vp_size: glam::Vec2,
) {
    if !app.decal48_state.built { return; }
    let vp_inv = app.camera.view_proj_matrix().inverse();
    let (ro, rd) = viewport_lib::picking::screen_to_ray(cursor, vp_size, vp_inv);
    if let Some((hit, normal)) = decal48_ray_hit(ro, rd) {
        let id = app.decal48_state.next_id;
        app.decal48_state.next_id += 1;
        let surface = if normal.z > 0.5 { "floor" } else { "wall" };
        app.decal48_state.decals.push(PlacedDecal {
            id,
            label: format!("#{id} ({surface})"),
            hit,
            normal,
        });
    }
}

// ---------------------------------------------------------------------------
// Scene items
// ---------------------------------------------------------------------------

pub(crate) fn decal48_scene_items(app: &mut App) -> Vec<SceneRenderItem> {
    app.decal48_state
        .scene
        .collect_render_items(&app.decal48_state.selection)
}

/// Push placed decals into `fd.scene.decals`.
pub(crate) fn submit_decal48_items(app: &App, fd: &mut viewport_lib::FrameData) {
    let st = &app.decal48_state;
    let Some(albedo) = st.albedo_tex else { return };
    for placed in &st.decals {
        let transform = decal_transform(placed.hit, placed.normal, st.decal_size, st.decal_depth);
        let mut item = DecalItem::default();
        item.transform            = transform;
        item.texture_id           = albedo;
        item.blend_mode           = st.blend_mode;
        item.alpha                = st.alpha;
        item.normal_texture_id    = if st.use_normal_map { st.normal_tex } else { None };
        item.normal_blend_strength = st.normal_blend;
        fd.scene.decals.push(item);
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_decal48(app: &mut App, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label("Click on the floor or wall to stamp a decal.");
        ui.add_space(6.0);

        ui.separator();
        ui.label("Decal settings:");
        ui.add(
            egui::Slider::new(&mut app.decal48_state.decal_size, 0.3..=3.0)
                .text("Size"),
        );
        ui.add(
            egui::Slider::new(&mut app.decal48_state.decal_depth, 0.3..=4.0)
                .text("Proj. depth"),
        );
        ui.small("Projection depth: box extent along hit normal. Affects curved")
            .on_hover_text("On flat surfaces all fragments have local.z = 0, so this has no visible effect. Increase it when a decal spans a curved or angled surface where depth variation within the footprint exists.");
        ui.small("surfaces; no visible effect on flat geometry.");
        ui.add(
            egui::Slider::new(&mut app.decal48_state.alpha, 0.1..=1.0)
                .text("Alpha"),
        );

        ui.add_space(4.0);
        ui.label("Blend mode:");
        ui.horizontal(|ui| {
            let cur = app.decal48_state.blend_mode;
            if ui
                .selectable_label(cur == DecalBlendMode::Replace, "Replace")
                .clicked()
            {
                app.decal48_state.blend_mode = DecalBlendMode::Replace;
            }
            if ui
                .selectable_label(cur == DecalBlendMode::Multiply, "Multiply")
                .clicked()
            {
                app.decal48_state.blend_mode = DecalBlendMode::Multiply;
            }
        });

        ui.add_space(6.0);
        ui.separator();
        ui.label("D2 - Normal map:");
        ui.checkbox(&mut app.decal48_state.use_normal_map, "Enable crater normal map");
        ui.add(
            egui::Slider::new(&mut app.decal48_state.normal_blend, 0.0..=1.0)
                .text("Normal blend"),
        );
        ui.small("When enabled, decals use a crater normal map to perturb");
        ui.small("shading: the centre appears sunken, the rim raised.");

        ui.add_space(6.0);
        ui.separator();
        ui.label("Placed decals:");

        let mut remove_id: Option<u64> = None;
        for placed in &app.decal48_state.decals {
            ui.horizontal(|ui| {
                ui.label(&placed.label);
                if ui.small_button("x").clicked() {
                    remove_id = Some(placed.id);
                }
            });
        }
        if let Some(id) = remove_id {
            app.decal48_state.decals.retain(|d| d.id != id);
        }

        if app.decal48_state.decals.is_empty() {
            ui.small("(none)");
        }

        ui.add_space(4.0);
        if ui.button("Clear all").clicked() {
            app.decal48_state.decals.clear();
        }

        ui.add_space(6.0);
        ui.separator();
        ui.label("What this shows:");
        ui.small("- D1: screen-space projection from scene depth.");
        ui.small("  Decals span the floor/wall corner seamlessly.");
        ui.small("- D2: tangent-space normal map perturbs shading.");
        ui.small("  N.V ratio modulates colour: centre dark, rim bright.");
        ui.small("- Replace: alpha-blended over the opaque pass.");
        ui.small("- Multiply: dst.rgb * src.rgb (darkening blend).");
    });
}
