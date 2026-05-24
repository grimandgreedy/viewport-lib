//! Showcase 48: Screen-Space Decals
//!
//! Demonstrates the full decal pipeline (D1 through D5).
//!
//! Scene: a vertical wall (XZ plane, +Y normal) meets a ground floor (XY
//! plane, +Z normal). Click any visible surface to stamp a gunshot decal.
//! Static footprints and blood splatters are pre-placed on the ground.
//!
//! D1: screen-space projection from the opaque-pass depth buffer.
//! D2: optional crater normal map that perturbs surface shading.
//! D3: roughness and metallic controls; wet-patch sort_key demo.
//! D4: lifetime-managed fading decals and a UV-scroll animation demo.
//! D5: receiver masking -- orange box on the floor blocks decal projection.
//! D6: emissive channel -- glowing rune and spark-impact decals.
//! D7: soft projection-box edges -- edge_fade slider on/off comparison.

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

/// Circular gunshot decal albedo: medium-gray disc on transparent background.
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

/// D3: Glossy "wet" disc for the standing-water patch demo.
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
            buf[idx]     = 180;
            buf[idx + 1] = 195;
            buf[idx + 2] = 210;
            buf[idx + 3] = (alpha * 200.0) as u8;
        }
    }
    buf
}

/// D4: Diagonal stripe pattern for UV-scroll animation demo.
fn make_stripe_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    const GAP: f32 = 0.04;
    for y in 0..size {
        for x in 0..size {
            let t = ((x + y) as f32 / size as f32 * 4.0).fract();
            let idx = ((y * size + x) * 4) as usize;
            let near_edge = t < GAP || (t > 0.5 - GAP && t < 0.5 + GAP) || t > 1.0 - GAP;
            if near_edge {
                buf[idx + 3] = 0;
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

/// Single boot-print silhouette viewed from above.
/// In texture space the sole runs left-to-right, heel at left, toe at right.
fn make_footprint_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let s = size as f32;
    for y in 0..size {
        for x in 0..size {
            let tx = x as f32 / s - 0.5;
            let ty = y as f32 / s - 0.5;
            let ax = 0.40_f32;
            let ay_base = 0.20_f32;
            let taper = 1.0 - tx.max(0.0) * 0.55;
            let ay = ay_base * taper;
            let d = (tx / ax).powi(2) + (ty / ay).powi(2);
            let alpha = ((1.0 - d) / 0.08).clamp(0.0, 1.0);
            if alpha > 0.0 {
                let idx = ((y * size + x) * 4) as usize;
                buf[idx]     = 55;
                buf[idx + 1] = 32;
                buf[idx + 2] = 12;
                buf[idx + 3] = (alpha * 210.0) as u8;
            }
        }
    }
    buf
}

/// Blood splatter: central pool plus smaller satellite drops.
fn make_blood_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let s = size as f32;
    let drops: &[(f32, f32, f32)] = &[
        (0.50, 0.50, 0.36),
        (0.22, 0.28, 0.09),
        (0.76, 0.32, 0.07),
        (0.62, 0.76, 0.08),
        (0.30, 0.72, 0.06),
        (0.82, 0.65, 0.05),
    ];
    for y in 0..size {
        for x in 0..size {
            let tx = x as f32 / s;
            let ty = y as f32 / s;
            let mut alpha = 0.0_f32;
            for &(cx, cy, r) in drops {
                let dx = tx - cx;
                let dy = ty - cy;
                let d = (dx * dx + dy * dy).sqrt();
                alpha = alpha.max(((r - d) / (r * 0.15)).clamp(0.0, 1.0));
            }
            if alpha > 0.0 {
                let idx = ((y * size + x) * 4) as usize;
                buf[idx]     = 175;
                buf[idx + 1] = 12;
                buf[idx + 2] = 12;
                buf[idx + 3] = (alpha * 230.0) as u8;
            }
        }
    }
    buf
}

/// D6: Glowing rune -- a five-pointed star outline on a transparent background.
fn make_rune_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let s = size as f32;
    let c = s * 0.5;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let d = (dx * dx + dy * dy).sqrt();
            let angle = dy.atan2(dx);
            // Star: 5 outer points, 5 inner indents.
            let spoke_angle = (angle * 5.0 / 2.0).cos().abs();
            let outer = c * 0.45;
            let inner = c * 0.20;
            let star_r = inner + (outer - inner) * spoke_angle;
            let edge_w = c * 0.07;
            let rim_dist = (d - star_r).abs();
            let alpha = ((edge_w - rim_dist) / edge_w).clamp(0.0, 1.0);
            if alpha > 0.0 {
                let idx = ((y * size + x) * 4) as usize;
                buf[idx]     = 255;
                buf[idx + 1] = 200;
                buf[idx + 2] = 80;
                buf[idx + 3] = (alpha * 255.0) as u8;
            }
        }
    }
    buf
}

/// D6: Spark-impact -- a bright central burst with 8 radiating streaks.
fn make_spark_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let s = size as f32;
    let c = s * 0.5;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let d = (dx * dx + dy * dy).sqrt() / c;
            let angle = dy.atan2(dx);
            let spoke = (angle * 8.0).cos().abs().powf(6.0);
            let radial = (1.0 - d).max(0.0).powf(1.5);
            let alpha = (radial * 0.5 + spoke * radial * 0.7).clamp(0.0, 1.0);
            if alpha > 0.005 {
                let idx = ((y * size + x) * 4) as usize;
                let g = (0.7 + 0.3 * radial).clamp(0.0, 1.0);
                let b = (0.2 * radial).clamp(0.0, 1.0);
                buf[idx]     = 255;
                buf[idx + 1] = (g * 255.0) as u8;
                buf[idx + 2] = (b * 255.0) as u8;
                buf[idx + 3] = (alpha * 255.0) as u8;
            }
        }
    }
    buf
}

/// D8: Checkerboard -- contrasting tiles that show UV stretching vs. tri-planar wrapping clearly.
fn make_checker_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![255u8; (size * size * 4) as usize];
    let tiles = 8u32;
    for y in 0..size {
        for x in 0..size {
            let tx = (x * tiles / size) % 2;
            let ty = (y * tiles / size) % 2;
            let dark = (tx + ty) % 2 == 0;
            let idx = ((y * size + x) * 4) as usize;
            if dark {
                buf[idx]     = 40;
                buf[idx + 1] = 40;
                buf[idx + 2] = 140;
            } else {
                buf[idx]     = 210;
                buf[idx + 1] = 200;
                buf[idx + 2] = 230;
            }
            buf[idx + 3] = 220;
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Decal transform builder
// ---------------------------------------------------------------------------

/// Build a model matrix for a decal placed at `hit` on a surface with `normal`.
fn decal_transform(hit: glam::Vec3, normal: glam::Vec3, size: f32, depth: f32) -> [[f32; 4]; 4] {
    decal_transform_yaw(hit, normal, size, depth, 0.0)
}

/// Like `decal_transform` but rotates the UV frame by `yaw` radians around the surface normal.
/// Useful for oriented decals such as footprints: yaw = PI/2 points the toe toward +Y on a Z-up ground.
fn decal_transform_yaw(hit: glam::Vec3, normal: glam::Vec3, size: f32, depth: f32, yaw: f32) -> [[f32; 4]; 4] {
    let n = normal.normalize();
    let ref_up = if n.abs().abs_diff_eq(glam::Vec3::Y, 0.1) { glam::Vec3::Z } else { glam::Vec3::Y };
    let t0 = ref_up.cross(n).normalize();
    let b0 = n.cross(t0).normalize();
    let (s, c) = yaw.sin_cos();
    let tangent   = c * t0 + s * b0;
    let bitangent = -s * t0 + c * b0;
    glam::Mat4::from_cols(
        (tangent   * size).extend(0.0),
        (bitangent * size).extend(0.0),
        (n         * depth).extend(0.0),
        hit.extend(1.0),
    ).to_cols_array_2d()
}

// ---------------------------------------------------------------------------
// Per-placed decal record (manually placed gunshots)
// ---------------------------------------------------------------------------

pub(crate) struct PlacedDecal {
    id: u64,
    label: String,
    pub(crate) hit:    glam::Vec3,
    pub(crate) normal: glam::Vec3,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct Decal48State {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,

    pub wall_mesh:      Option<MeshId>,
    pub ground_mesh:    Option<MeshId>,

    pub albedo_tex:     Option<u64>,
    pub normal_tex:     Option<u64>,
    pub wet_tex:        Option<u64>,
    pub stripe_tex:     Option<u64>,
    pub footprint_tex:  Option<u64>,
    pub blood_tex:      Option<u64>,

    // D1/D2/D3: manually placed gunshot decals
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

    // D5
    pub wall_obstacle_node: Option<viewport_lib::interaction::selection::NodeId>,
    pub show_obstacle: bool,

    // D6
    pub rune_tex:   Option<u64>,
    pub spark_tex:  Option<u64>,
    pub show_rune:  bool,
    pub rune_emissive: f32,
    pub show_spark: bool,
    pub spark_emissive: f32,

    // D7
    pub edge_fade: f32,
    pub apply_edge_fade: bool,

    // D8
    pub checker_tex: Option<u64>,
    pub show_corner_decal: bool,
    pub use_tri_planar: bool,
    pub tri_blend_sharpness: f32,
}

impl Default for Decal48State {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            wall_mesh:     None,
            ground_mesh:   None,
            albedo_tex:    None,
            normal_tex:    None,
            wet_tex:       None,
            stripe_tex:    None,
            footprint_tex: None,
            blood_tex:     None,
            decals: Vec::new(),
            next_id: 1,
            decal_size: 0.25,
            decal_depth: 1.5,
            use_normal_map: true,
            normal_blend: 0.8,
            blend_mode: DecalBlendMode::Replace,
            alpha: 0.9,
            decal_roughness: 0.25,
            decal_metallic: 0.5,
            show_wet_patch: false,
            fading_mode: true,
            fade_lifetime: 6.0,
            fade_out: 2.0,
            show_scroll: false,
            scroll_handle: None,
            wall_obstacle_node: None,
            show_obstacle: true,
            rune_tex:       None,
            spark_tex:      None,
            show_rune:      false,
            rune_emissive:  2.0,
            show_spark:     false,
            spark_emissive: 3.0,
            edge_fade:      0.2,
            apply_edge_fade: false,
            checker_tex:         None,
            show_corner_decal:   false,
            use_tri_planar:      false,
            tri_blend_sharpness: 4.0,
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
    let footprint_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_footprint_texture(128))
        .expect("footprint texture upload");
    let blood_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_blood_texture(128))
        .expect("blood texture upload");
    let rune_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_rune_texture(128))
        .expect("rune texture upload");
    let spark_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_spark_texture(128))
        .expect("spark texture upload");

    app.decal48_state.albedo_tex    = Some(albedo_id);
    app.decal48_state.normal_tex    = Some(normal_id);
    app.decal48_state.wet_tex       = Some(wet_id);
    app.decal48_state.stripe_tex    = Some(stripe_id);
    app.decal48_state.footprint_tex = Some(footprint_id);
    app.decal48_state.blood_tex     = Some(blood_id);
    app.decal48_state.rune_tex      = Some(rune_id);
    app.decal48_state.spark_tex     = Some(spark_id);

    let checker_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_checker_texture(128))
        .expect("checker texture upload");
    app.decal48_state.checker_tex = Some(checker_id);

    // Vertical wall: 6 wide (X), 0.2 thick (Y), 4 tall (Z).
    // Centered at (0, -0.1, 2): front face (+Y normal) at y = 0,
    // spans x in [-3, 3], z in [0, 4].
    let wall_data = viewport_lib::primitives::cuboid(6.0, 0.2, 4.0);
    let wall_id = res
        .upload_mesh_data(&app.device, &wall_data)
        .expect("wall mesh upload");
    app.decal48_state.wall_mesh = Some(wall_id);

    // Ground floor: 8 wide (X), 6 deep (Y), 0.2 thick (Z).
    // Centered at (0, 3, -0.1): top face (+Z normal) at z = 0,
    // spans x in [-4, 4], y in [0, 6].
    let ground_data = viewport_lib::primitives::cuboid(8.0, 6.0, 0.2);
    let ground_id = res
        .upload_mesh_data(&app.device, &ground_data)
        .expect("ground mesh upload");
    app.decal48_state.ground_mesh = Some(ground_id);

    // D5: non-receiver box mounted on the wall face.
    // cuboid(0.6, 0.25, 0.6): sticks 0.25 out from the wall, center y = 0.125.
    let wall_obstacle_data = viewport_lib::primitives::cuboid(0.6, 0.25, 0.6);
    let wall_obstacle_id = res
        .upload_mesh_data(&app.device, &wall_obstacle_data)
        .expect("wall obstacle mesh upload");

    let scene = &mut app.decal48_state.scene;

    let wall_mat = Material::from_colour([0.92, 0.92, 0.92]);
    scene.add(
        Some(wall_id),
        glam::Mat4::from_translation(glam::Vec3::new(0.0, -0.1, 2.0)),
        wall_mat,
    );

    let ground_mat = Material::from_colour([0.78, 0.76, 0.72]);
    scene.add(
        Some(ground_id),
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 3.0, -0.1)),
        ground_mat,
    );

    // Wall obstacle: mounted on the wall face at left-center.
    // center y = 0.125 puts its back face flush against y = 0.
    let wall_obstacle_node = scene.add(
        Some(wall_obstacle_id),
        glam::Mat4::from_translation(glam::Vec3::new(-1.5, 0.125, 2.0)),
        Material::from_colour([1.0, 0.45, 0.1]),
    );
    scene.set_receives_decals(wall_obstacle_node, false);
    app.decal48_state.wall_obstacle_node = Some(wall_obstacle_node);

    app.decal48_state.built = true;
}

// ---------------------------------------------------------------------------
// Click handling
// ---------------------------------------------------------------------------

/// Test `ray` against the wall face and the ground floor.
/// Returns (hit_point, surface_normal), choosing the closest hit.
///
/// Wall:   +Y normal at y = 0, bounds x in [-3, 3], z in [0, 4].
/// Ground: +Z normal at z = 0, bounds x in [-4, 4], y in [0, 6].
fn decal48_ray_hit(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
) -> Option<(glam::Vec3, glam::Vec3)> {
    let mut best_t = f32::INFINITY;
    let mut best: Option<(glam::Vec3, glam::Vec3)> = None;

    // Test vertical wall (y = 0 plane, camera must be on +y side).
    if ray_dir.y.abs() > 1e-6 {
        let t = -ray_origin.y / ray_dir.y;
        if t > 0.001 && t < best_t {
            let hit = ray_origin + ray_dir * t;
            // Obstacle footprint exclusion: box at (-1.5, 0.125, 2.0), half-extents (0.3, *, 0.3).
            let in_obstacle = hit.x >= -1.8 && hit.x <= -1.2
                           && hit.z >= 1.7  && hit.z <= 2.3;
            if hit.x.abs() <= 3.0 && hit.z >= 0.0 && hit.z <= 4.0 && !in_obstacle {
                best_t = t;
                best = Some((hit, glam::Vec3::Y));
            }
        }
    }

    // Test ground floor (z = 0 plane, camera must be above).
    if ray_dir.z.abs() > 1e-6 {
        let t = -ray_origin.z / ray_dir.z;
        if t > 0.001 && t < best_t {
            let hit = ray_origin + ray_dir * t;
            if hit.x.abs() <= 4.0 && hit.y >= 0.0 && hit.y <= 6.0 {
                best = Some((hit, glam::Vec3::Z));
            }
        }
    }

    best
}

/// Place a gunshot decal at the clicked viewport position.
pub(crate) fn decal48_place(app: &mut App, cursor: glam::Vec2, vp_size: glam::Vec2) {
    if !app.decal48_state.built { return; }
    let vp_inv = app.camera.view_proj_matrix().inverse();
    let (ro, rd) = viewport_lib::picking::screen_to_ray(cursor, vp_size, vp_inv);
    let Some((hit, normal)) = decal48_ray_hit(ro, rd) else { return };

    let st = &mut app.decal48_state;
    let Some(tex) = st.albedo_tex else { return };
    let transform = decal_transform(hit, normal, st.decal_size, st.decal_depth);

    if st.fading_mode {
        let mut item = DecalItem::default();
        item.transform             = transform;
        item.texture_id            = tex;
        item.blend_mode            = st.blend_mode;
        item.alpha                 = st.alpha;
        item.normal_texture_id     = if st.use_normal_map { st.normal_tex } else { None };
        item.normal_blend_strength = st.normal_blend;
        item.roughness             = st.decal_roughness;
        item.metallic              = st.decal_metallic;
        let lt = st.fade_lifetime;
        st.scene.add_decal_with_lifetime(item, lt, st.fade_out);
    } else {
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

pub(crate) fn update_decal48(app: &mut App, dt: f32) {
    if !app.decal48_state.built { return; }

    app.decal48_state.scene.update_decals(dt);

    // D5: sync obstacle visibility.
    let show = app.decal48_state.show_obstacle;
    if let Some(node) = app.decal48_state.wall_obstacle_node {
        app.decal48_state.scene.set_visible(node, show);
    }

    let st = &mut app.decal48_state;

    // Scroll animation: covers the left side of the ground floor.
    let want_scroll = st.show_scroll;
    match (want_scroll, st.scroll_handle.is_some()) {
        (true, false) => {
            if let Some(stripe) = st.stripe_tex {
                // Ground: normal = +Z, center on left half at z = 0.
                let transform = decal_transform(
                    glam::Vec3::new(-1.0, 3.0, 0.0),
                    glam::Vec3::Z,
                    3.0,
                    1.5,
                );
                let mut item = DecalItem::default();
                item.transform  = transform;
                item.texture_id = stripe;
                item.alpha      = 1.0;
                item.sort_key   = -10;
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

    // D3: wet patch on the left side of the ground floor.
    if st.show_wet_patch {
        if let Some(wet) = st.wet_tex {
            // Ground: normal = +Z, placed at z = 0.
            let transform = decal_transform(
                glam::Vec3::new(-1.5, 3.0, 0.0),
                glam::Vec3::Z,
                1.5,
                1.0,
            );
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = wet;
            item.roughness  = 0.05;
            item.metallic   = 0.0;
            item.alpha      = 0.85;
            item.sort_key   = 10;
            fd.scene.decals.push(item);
        }
    }

    // Static footprints: trail across the left half of the ground (x < 0).
    // Alternating right/left foot, walking roughly along +Y with slight wandering.
    // yaw = PI/2 rotates the texture so the toe points toward +Y.
    if let Some(fp) = st.footprint_tex {
        use std::f32::consts::PI;
        // (x, y, yaw_offset_from_PI_2)  -- right foot: x closer to 0; left: more negative x
        let steps: &[(f32, f32, f32)] = &[
            (-1.85, 0.45,  0.04),   // R
            (-2.40, 0.90, -0.05),   // L
            (-1.90, 1.40,  0.06),   // R
            (-2.45, 1.90, -0.04),   // L
            (-1.95, 2.40,  0.03),   // R
            (-2.50, 2.90, -0.06),   // L
            (-2.00, 3.40,  0.05),   // R
            (-2.55, 3.90, -0.03),   // L
            (-2.05, 4.35,  0.04),   // R
            (-2.60, 4.80, -0.05),   // L
            (-2.10, 5.25,  0.03),   // R
            (-2.65, 5.60, -0.04),   // L
        ];
        for &(x, y, yaw_delta) in steps {
            let pos = glam::Vec3::new(x, y, 0.0);
            let yaw = PI / 2.0 + yaw_delta;
            let transform = decal_transform_yaw(pos, glam::Vec3::Z, 0.35, 0.5, yaw);
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = fp;
            item.roughness  = 0.9;
            item.metallic   = 0.0;
            item.alpha      = 0.88;
            fd.scene.decals.push(item);
        }
    }

    // Static blood splatters: pre-placed on the right side of the ground (x >= 0).
    if let Some(bl) = st.blood_tex {
        let splatters: &[(glam::Vec3, f32)] = &[
            (glam::Vec3::new(2.0, 2.5, 0.0), 0.9),
            (glam::Vec3::new(1.2, 4.0, 0.0), 0.55),
            (glam::Vec3::new(2.8, 3.5, 0.0), 0.4),
        ];
        for &(pos, size) in splatters {
            let transform = decal_transform(pos, glam::Vec3::Z, size, 0.5);
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = bl;
            item.roughness  = 0.6;
            item.metallic   = 0.0;
            item.alpha      = 0.9;
            fd.scene.decals.push(item);
        }
    }

    // D1-D3: permanently placed gunshot decals.
    for placed in &st.decals {
        let transform = decal_transform(placed.hit, placed.normal, st.decal_size, st.decal_depth);
        let Some(tex) = st.albedo_tex else { continue };
        let mut item = DecalItem::default();
        item.transform             = transform;
        item.texture_id            = tex;
        item.blend_mode            = st.blend_mode;
        item.alpha                 = st.alpha;
        item.normal_texture_id     = if st.use_normal_map { st.normal_tex } else { None };
        item.normal_blend_strength = st.normal_blend;
        item.roughness             = st.decal_roughness;
        item.metallic              = st.decal_metallic;
        // D7: apply edge fade to placed decals when the toggle is on.
        item.edge_fade = if st.apply_edge_fade { st.edge_fade } else { 0.0 };
        fd.scene.decals.push(item);
    }

    // D4: live decals (fading + animation) from the scene.
    fd.scene.decals.extend(st.scene.collect_decal_items());

    // D6: glowing rune on the wall, center-right.
    if st.show_rune {
        if let Some(rune) = st.rune_tex {
            let transform = decal_transform(
                glam::Vec3::new(1.5, 0.0, 2.0),
                glam::Vec3::Y,
                0.5,
                1.0,
            );
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = rune;
            item.alpha      = 1.0;
            item.emissive   = st.rune_emissive;
            item.edge_fade  = 0.1;
            fd.scene.decals.push(item);
        }
    }

    // D8: corner-spanning checkerboard decal at the wall/floor junction.
    // Planar mode stretches visibly across the 90-degree corner; tri-planar wraps cleanly.
    if st.show_corner_decal {
        if let Some(checker) = st.checker_tex {
            // Place the decal centred on the corner junction (x=1, y=0, z=0),
            // oriented along +Y as projection axis so it faces the camera from the wall.
            // Scale it large enough (2.0) to cover both wall and floor across the corner.
            let transform = decal_transform(
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::Y,
                2.0,
                2.0,
            );
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = checker;
            item.alpha      = 0.9;
            item.edge_fade  = 0.05;
            item.projection = if st.use_tri_planar {
                viewport_lib::DecalProjection::TriPlanar { blend_sharpness: st.tri_blend_sharpness }
            } else {
                viewport_lib::DecalProjection::Planar
            };
            fd.scene.decals.push(item);
        }
    }

    // D6: spark-impact on the ground, center-right.
    if st.show_spark {
        if let Some(spark) = st.spark_tex {
            let transform = decal_transform(
                glam::Vec3::new(2.5, 1.5, 0.0),
                glam::Vec3::Z,
                0.4,
                0.8,
            );
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = spark;
            item.alpha      = 1.0;
            item.emissive   = st.spark_emissive;
            fd.scene.decals.push(item);
        }
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_decal48(app: &mut App, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        if app.decal48_state.fading_mode {
            ui.label("Click the wall or ground to stamp a gunshot decal (auto-fade).");
        } else {
            ui.label("Click the wall or ground to stamp a permanent gunshot decal.");
        }
        ui.add_space(6.0);

        ui.separator();
        ui.label("Decal settings:");
        ui.add(egui::Slider::new(&mut app.decal48_state.decal_size, 0.05..=2.0).text("Size"));
        ui.add(
            egui::Slider::new(&mut app.decal48_state.decal_depth, 0.3..=4.0)
                .text("Proj. depth"),
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
        ui.label("Normal map:");
        ui.checkbox(&mut app.decal48_state.use_normal_map, "Crater normal map");
        ui.add(
            egui::Slider::new(&mut app.decal48_state.normal_blend, 0.0..=1.0)
                .text("Normal blend"),
        );

        ui.add_space(6.0);
        ui.separator();
        ui.label("Roughness / metallic:");
        ui.add(
            egui::Slider::new(&mut app.decal48_state.decal_roughness, 0.0..=1.0)
                .text("Roughness"),
        );
        ui.add(
            egui::Slider::new(&mut app.decal48_state.decal_metallic, 0.0..=1.0)
                .text("Metallic"),
        );
        ui.add_space(4.0);
        ui.checkbox(&mut app.decal48_state.show_wet_patch, "Show wet patch (ground left)");
        ui.small("Wet patch: roughness 0.05, sort above footprints.");

        ui.add_space(6.0);
        ui.separator();
        ui.label("Lifetime / animation:");
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
        ui.small("Diagonal stripe decal scrolling across the ground.");

        ui.add_space(6.0);
        ui.separator();
        ui.label("Placed gunshots:");
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
        ui.label("Receiver masking:");
        ui.checkbox(&mut app.decal48_state.show_obstacle, "Show non-receiver obstacles");
        ui.small("Orange boxes (wall + ground): receives_decals = false.");

        ui.add_space(6.0);
        ui.separator();
        ui.label("Emissive:");
        ui.small("Emissive adds self-illumination on top of the blend result.");
        ui.small("Values above 1.0 bloom under tone-mapping (post_process required).");
        ui.add_space(2.0);
        ui.checkbox(&mut app.decal48_state.show_rune, "Glowing rune (wall right)");
        if app.decal48_state.show_rune {
            ui.add(
                egui::Slider::new(&mut app.decal48_state.rune_emissive, 0.0..=8.0)
                    .text("Rune emissive"),
            );
        }
        ui.checkbox(&mut app.decal48_state.show_spark, "Spark impact (ground right)");
        if app.decal48_state.show_spark {
            ui.add(
                egui::Slider::new(&mut app.decal48_state.spark_emissive, 0.0..=8.0)
                    .text("Spark emissive"),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.label("Soft edges:");
        ui.small("Edge fade smooths the rectangular boundary of the projection box.");
        ui.small("Toggle on placed gunshots to compare hard vs. soft edges.");
        ui.add_space(2.0);
        ui.checkbox(&mut app.decal48_state.apply_edge_fade, "Apply edge fade to gunshots");
        if app.decal48_state.apply_edge_fade {
            ui.add(
                egui::Slider::new(&mut app.decal48_state.edge_fade, 0.0..=0.5)
                    .text("Edge fade"),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.label("Tri-planar projection:");
        ui.small("Planar stretches across the wall/floor corner; tri-planar wraps cleanly.");
        ui.add_space(2.0);
        ui.checkbox(&mut app.decal48_state.show_corner_decal, "Show corner decal (checkerboard)");
        if app.decal48_state.show_corner_decal {
            ui.checkbox(&mut app.decal48_state.use_tri_planar, "Tri-planar (off = planar)");
            if app.decal48_state.use_tri_planar {
                ui.add(
                    egui::Slider::new(&mut app.decal48_state.tri_blend_sharpness, 1.0..=16.0)
                        .text("Blend sharpness"),
                );
                ui.small("Higher = sharper face transitions.");
            }
        }

        ui.add_space(6.0);
        ui.separator();
        ui.label("Surface guide:");
        ui.small("Wall (y = 0):        gunshot craters, glowing rune.");
        ui.small("Ground left  (x<0):  muddy footprints (static).");
        ui.small("Ground right (x>=0): blood splatter, spark impact.");
    });
}
