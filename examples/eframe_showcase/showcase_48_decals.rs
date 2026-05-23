//! Showcase 48: Screen-Space Decals
//!
//! Demonstrates the full decal pipeline (D1 through D4).
//!
//! Scene: a white upright wall in the XZ plane. Click anywhere on it to stamp
//! a decal at the hit point.
//!
//! D1: screen-space projection from the opaque-pass depth buffer.
//! D2: optional crater normal map that perturbs surface shading.
//! D3: roughness and metallic controls; sort_key ordering demo.
//! D4: lifetime-managed fading decals and a UV-scroll animation demo.

use eframe::egui;
use viewport_lib::{
    DecalAnimation, DecalBlendMode, DecalHandle, DecalItem, Material, MeshId, SceneRenderItem,
    scene::Scene,
    selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------

/// Circular decal albedo: medium-gray disc on a transparent background.
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
            buf[idx]     = 120;
            buf[idx + 1] = 115;
            buf[idx + 2] = 110;
            buf[idx + 3] = (alpha * 220.0) as u8;
        }
    }
    buf
}

/// Bullet-hole crater normal map: encodes an inward dome shape.
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
            let nz = t;
            let scale = (1.0 - nz * nz).sqrt();
            let (nx, ny) = if d > 0.001 {
                (dx / d * scale * 0.6, dy / d * scale * 0.6)
            } else {
                (0.0, 0.0)
            };
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

/// D3: Glossy "wet" disc - slightly darker base with a brighter specular look.
fn make_wet_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let c = size as f32 * 0.5;
    let r = c * 0.85;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let d = (dx * dx + dy * dy).sqrt();
            let alpha = ((r - d) / (r * 0.1)).clamp(0.0, 1.0);
            let idx = ((y * size + x) * 4) as usize;
            // Slightly blue-tinted for a water look.
            buf[idx]     = 180;
            buf[idx + 1] = 195;
            buf[idx + 2] = 210;
            buf[idx + 3] = (alpha * 200.0) as u8;
        }
    }
    buf
}

/// D4: Diagonal stripe pattern for UV-scroll animation demo (alternating blue/red lines).
/// A narrow transparent gap at each stripe boundary keeps bilinear filtering from
/// blending the two colours into a fuzzy edge; instead it blends toward transparent.
fn make_stripe_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    // Gap half-width as a fraction of one stripe period.  At size=64 and 4 cycles
    // this is 64/4 = 16 px per period, so gap = 16 * 0.04 ≈ 0.6 px -- just enough
    // to give bilinear a transparent target at the boundary.
    const GAP: f32 = 0.04;
    for y in 0..size {
        for x in 0..size {
            let t = ((x + y) as f32 / size as f32 * 4.0).fract();
            let idx = ((y * size + x) * 4) as usize;
            let near_edge = t < GAP || (t > 0.5 - GAP && t < 0.5 + GAP) || t > 1.0 - GAP;
            if near_edge {
                buf[idx + 3] = 0; // transparent gap
            } else if t < 0.5 {
                buf[idx]     = 40;
                buf[idx + 1] = 80;
                buf[idx + 2] = 220;
                buf[idx + 3] = 255;
            } else {
                buf[idx]     = 220;
                buf[idx + 1] = 50;
                buf[idx + 2] = 40;
                buf[idx + 3] = 255;
            }
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Decal transform builder
// ---------------------------------------------------------------------------

/// Build a model matrix for a decal placed at `hit` on a surface with `normal`.
fn decal_transform(hit: glam::Vec3, normal: glam::Vec3, size: f32, depth: f32) -> [[f32; 4]; 4] {
    let n = normal.normalize();
    // Avoid degenerate cross product when n is nearly parallel to Y (either +Y or -Y).
    let ref_up = if n.abs().abs_diff_eq(glam::Vec3::Y, 0.1) { glam::Vec3::Z } else { glam::Vec3::Y };
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
// Per-placed decal record (D1-D3, manually managed)
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
    pub wall_mesh:   Option<MeshId>,
    pub albedo_tex:  Option<u64>,
    pub normal_tex:  Option<u64>,
    pub wet_tex:     Option<u64>,   // D3
    pub stripe_tex:  Option<u64>,   // D4

    // D1/D2/D3: manually placed decals
    pub decals: Vec<PlacedDecal>,
    pub next_id: u64,
    pub decal_size: f32,
    pub decal_depth: f32,
    pub use_normal_map: bool,
    pub normal_blend: f32,
    pub blend_mode: DecalBlendMode,
    pub alpha: f32,

    // D3
    pub decal_roughness: f32,
    pub decal_metallic: f32,
    pub show_wet_patch: bool,

    // D4
    pub fading_mode: bool,
    pub fade_lifetime: f32,
    pub fade_out: f32,
    pub show_scroll: bool,
    pub scroll_handle: Option<DecalHandle>,
}

impl Default for Decal48State {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            wall_mesh:   None,
            albedo_tex:  None,
            normal_tex:  None,
            wet_tex:     None,
            stripe_tex:  None,
            decals: Vec::new(),
            next_id: 1,
            decal_size: 0.15,
            decal_depth: 1.5,
            use_normal_map: true,
            normal_blend: 0.8,
            blend_mode: DecalBlendMode::Replace,
            alpha: 0.9,
            decal_roughness: 0.25,
            decal_metallic: 0.5,
            show_wet_patch: false,
            fading_mode: true,
            fade_lifetime: 4.0,
            fade_out: 1.5,
            show_scroll: false,
            scroll_handle: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

pub(crate) fn build_decal48_scene(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    app.decal48_state.scene = Scene::new();

    let res = renderer.resources_mut();

    let albedo_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_disc_texture(128))
        .expect("decal albedo upload");
    let normal_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_crater_normal_map(128))
        .expect("decal normal map upload");
    let wet_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_wet_texture(128))
        .expect("wet texture upload");
    let stripe_id = res
        .upload_texture(&app.device, &app.queue, 64, 64, &make_stripe_texture(64))
        .expect("stripe texture upload");

    app.decal48_state.albedo_tex = Some(albedo_id);
    app.decal48_state.normal_tex = Some(normal_id);
    app.decal48_state.wet_tex    = Some(wet_id);
    app.decal48_state.stripe_tex = Some(stripe_id);

    // Wall: cuboid 8x0.3x4 (X wide, Y thin, Z tall), centered then translated so
    // the front face (+Y normal) sits at y=0 and Z spans 0..4.
    let wall_data = viewport_lib::primitives::cuboid(8.0, 0.3, 4.0);
    let wall_id = res
        .upload_mesh_data(&app.device, &wall_data)
        .expect("wall mesh upload");
    app.decal48_state.wall_mesh = Some(wall_id);

    let scene = &mut app.decal48_state.scene;
    let wall_mat = Material::from_colour([0.95, 0.95, 0.95]);
    scene.add(
        Some(wall_id),
        glam::Mat4::from_translation(glam::Vec3::new(0.0, -0.15, 2.0)),
        wall_mat,
    );

    app.decal48_state.built = true;
}

// ---------------------------------------------------------------------------
// Click handling
// ---------------------------------------------------------------------------

/// Test `ray` against the wall box. Picks the face the ray enters from.
/// Wall is a cuboid 8x0.3x4 translated to (0,-0.15,2): top face at y=0, bottom at y=-0.3.
pub(crate) fn decal48_ray_hit(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
) -> Option<(glam::Vec3, glam::Vec3)> {
    if ray_dir.y.abs() < 1e-6 { return None; }
    // Choose which horizontal face to test based on which side the camera is on.
    let (face_y, normal) = if ray_origin.y >= -0.15 {
        (0.0_f32, glam::Vec3::Y)
    } else {
        (-0.3_f32, -glam::Vec3::Y)
    };
    let t = (face_y - ray_origin.y) / ray_dir.y;
    if t < 0.001 { return None; }
    let hit = ray_origin + ray_dir * t;
    if hit.x.abs() <= 4.0 && hit.z >= 0.0 && hit.z <= 4.0 {
        Some((hit, normal))
    } else {
        None
    }
}

/// Place a decal at the clicked viewport position.
pub(crate) fn decal48_place(app: &mut App, cursor: glam::Vec2, vp_size: glam::Vec2) {
    if !app.decal48_state.built { return; }
    let vp_inv = app.camera.view_proj_matrix().inverse();
    let (ro, rd) = viewport_lib::picking::screen_to_ray(cursor, vp_size, vp_inv);
    let Some((hit, normal)) = decal48_ray_hit(ro, rd) else { return };

    let st = &mut app.decal48_state;

    if st.fading_mode {
        // D4: add a temporary decal via the scene API.
        let transform = decal_transform(hit, normal, st.decal_size, st.decal_depth);
        let Some(albedo) = st.albedo_tex else { return };
        let mut item = DecalItem::default();
        item.transform             = transform;
        item.texture_id            = albedo;
        item.blend_mode            = st.blend_mode;
        item.alpha                 = st.alpha;
        item.normal_texture_id     = if st.use_normal_map { st.normal_tex } else { None };
        item.normal_blend_strength = st.normal_blend;
        item.roughness             = st.decal_roughness;
        item.metallic              = st.decal_metallic;
        item.sort_key              = 0;
        let lt = st.fade_lifetime;
        st.scene.add_decal_with_lifetime(item, lt, st.fade_out);
    } else {
        // D1-D3: permanent, manually tracked.
        let id = st.next_id;
        st.next_id += 1;
        st.decals.push(PlacedDecal {
            id,
            label: format!("#{id}"),
            hit,
            normal,
        });
    }
}

// ---------------------------------------------------------------------------
// Per-frame update (D4)
// ---------------------------------------------------------------------------

/// Advance live decal ages and sync the scroll/wet-patch state with the scene.
pub(crate) fn update_decal48(app: &mut App, dt: f32) {
    if !app.decal48_state.built { return; }

    app.decal48_state.scene.update_decals(dt);

    let st = &mut app.decal48_state;

    // Scroll animation: add or remove as the toggle changes.
    let want_scroll = st.show_scroll;
    match (want_scroll, st.scroll_handle.is_some()) {
        (true, false) => {
            if let Some(stripe) = st.stripe_tex {
                let transform = decal_transform(
                    glam::Vec3::new(0.0, 0.0, 2.0),
                    glam::Vec3::Y,
                    3.0,
                    1.5,
                );
                let mut item = DecalItem::default();
                item.transform  = transform;
                item.texture_id = stripe;
                item.alpha      = 1.0;
                item.sort_key   = -10;  // render below all other decals
                let handle = st.scene.add_decal_animated(
                    item,
                    DecalAnimation::UvScroll { vx: 0.2, vy: 0.1 },
                    None,
                );
                st.scroll_handle = Some(handle);
            }
        }
        (false, true) => {
            if let Some(h) = st.scroll_handle.take() {
                st.scene.remove_decal(h);
            }
        }
        _ => {}
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

/// Push all active decals into `fd.scene.decals`.
pub(crate) fn submit_decal48_items(app: &App, fd: &mut viewport_lib::FrameData) {
    let st = &app.decal48_state;
    let Some(albedo) = st.albedo_tex else { return };

    // D3: wet patch -- higher sort_key renders on top of placed decals.
    if st.show_wet_patch {
        if let Some(wet) = st.wet_tex {
            let transform = decal_transform(
                glam::Vec3::new(-1.5, 0.0, 1.5),
                glam::Vec3::Y,
                2.0,
                1.5,
            );
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = wet;
            item.roughness  = 0.05;
            item.metallic   = 0.0;
            item.alpha      = 0.85;
            item.sort_key   = 10;  // render above bullet holes
            fd.scene.decals.push(item);
        }
    }

    // D1-D3: manually placed permanent decals.
    for placed in &st.decals {
        let transform = decal_transform(placed.hit, placed.normal, st.decal_size, st.decal_depth);
        let mut item = DecalItem::default();
        item.transform             = transform;
        item.texture_id            = albedo;
        item.blend_mode            = st.blend_mode;
        item.alpha                 = st.alpha;
        item.normal_texture_id     = if st.use_normal_map { st.normal_tex } else { None };
        item.normal_blend_strength = st.normal_blend;
        item.roughness             = st.decal_roughness;
        item.metallic              = st.decal_metallic;
        item.sort_key              = 0;
        fd.scene.decals.push(item);
    }

    // D4: live decals (fading + animation) from the scene.
    fd.scene.decals.extend(st.scene.collect_decal_items());
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_decal48(app: &mut App, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        let fading = app.decal48_state.fading_mode;
        if fading {
            ui.label("Click to stamp a fading decal (auto-removes).");
        } else {
            ui.label("Click on the wall to stamp a permanent decal.");
        }
        ui.add_space(6.0);

        ui.separator();
        ui.label("Decal settings:");
        ui.add(egui::Slider::new(&mut app.decal48_state.decal_size, 0.01..=3.0).text("Size"));
        ui.add(
            egui::Slider::new(&mut app.decal48_state.decal_depth, 0.3..=4.0)
                .text("Proj. depth")
                .show_value(true),
        );
        ui.add(egui::Slider::new(&mut app.decal48_state.alpha, 0.1..=1.0).text("Alpha"));

        ui.add_space(4.0);
        ui.label("Blend mode:");
        ui.horizontal(|ui| {
            let cur = app.decal48_state.blend_mode;
            if ui.selectable_label(cur == DecalBlendMode::Replace,  "Replace").clicked()  {
                app.decal48_state.blend_mode = DecalBlendMode::Replace;
            }
            if ui.selectable_label(cur == DecalBlendMode::Multiply, "Multiply").clicked() {
                app.decal48_state.blend_mode = DecalBlendMode::Multiply;
            }
        });

        ui.add_space(6.0);
        ui.separator();
        ui.label("D2 - Normal map:");
        ui.checkbox(&mut app.decal48_state.use_normal_map, "Crater normal map");
        ui.add(
            egui::Slider::new(&mut app.decal48_state.normal_blend, 0.0..=1.0)
                .text("Normal blend"),
        );

        ui.add_space(6.0);
        ui.separator();
        ui.label("D3 - Roughness / metallic:");
        ui.add(
            egui::Slider::new(&mut app.decal48_state.decal_roughness, 0.0..=1.0)
                .text("Roughness"),
        );
        ui.add(
            egui::Slider::new(&mut app.decal48_state.decal_metallic, 0.0..=1.0)
                .text("Metallic"),
        );
        ui.small("Low roughness adds a tight specular highlight (N.V proxy).");
        ui.add_space(4.0);
        ui.checkbox(&mut app.decal48_state.show_wet_patch, "Show wet patch (sort above)");
        ui.small("Wet patch: roughness 0.05, sort_key +10 (renders above bullets).");

        ui.add_space(6.0);
        ui.separator();
        ui.label("D4 - Lifetime / animation:");
        ui.checkbox(&mut app.decal48_state.fading_mode, "Fading mode (auto-remove)");
        if app.decal48_state.fading_mode {
            ui.add(
                egui::Slider::new(&mut app.decal48_state.fade_lifetime, 2.0..=20.0)
                    .text("Lifetime (s)"),
            );
            let max_fade = app.decal48_state.fade_lifetime;
            ui.add(
                egui::Slider::new(&mut app.decal48_state.fade_out, 0.1..=max_fade)
                    .text("Fade-out (s)"),
            );
        }
        ui.checkbox(&mut app.decal48_state.show_scroll, "UV scroll animation");
        ui.small("Diagonal stripe decal scrolling across the wall.");

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
        if app.decal48_state.decals.is_empty() && !app.decal48_state.fading_mode {
            ui.small("(none)");
        }
        let live_count = app.decal48_state.scene.collect_decal_items().len();
        if live_count > 0 {
            ui.small(format!("{live_count} live (fading/animated) decals active."));
        }

        ui.add_space(4.0);
        if ui.button("Clear all permanent").clicked() {
            app.decal48_state.decals.clear();
        }

        ui.add_space(6.0);
        ui.separator();
        ui.label("What this shows:");
        ui.small("D1: screen-space projection from scene depth.");
        ui.small("D2: crater normal map perturbs shading (N.V ratio).");
        ui.small("D3: roughness/metallic specular; sort_key ordering.");
        ui.small("D4: lifetime fading; UV-scroll animation via LiveDecal.");
    });
}
