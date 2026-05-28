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
//! D8: tri-planar projection -- corner-spanning checkerboard decal.
//! D9: cylindrical projection -- label wrapped around a column.

use eframe::egui;
use viewport_lib::{
    BuiltinMatcap, CylindricalFacing, DecalAnimation, DecalBlendMode, DecalHandle, DecalItem,
    DecalProjection, Material, MeshId, SceneRenderItem,
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

/// D4: Diagonal black/white stripe pattern for UV-scroll animation demo.
fn make_stripe_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    const GAP: f32 = 0.03;
    for y in 0..size {
        for x in 0..size {
            let t = ((x + y) as f32 / size as f32 * 4.0).fract();
            let idx = ((y * size + x) * 4) as usize;
            let near_edge = t < GAP || (t > 0.5 - GAP && t < 0.5 + GAP) || t > 1.0 - GAP;
            if near_edge {
                buf[idx + 3] = 0;
            } else if t < 0.5 {
                buf[idx]     = 10;
                buf[idx + 1] = 10;
                buf[idx + 2] = 10;
                buf[idx + 3] = 230;
            } else {
                buf[idx]     = 245;
                buf[idx + 1] = 245;
                buf[idx + 2] = 245;
                buf[idx + 3] = 230;
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

/// D10: Fire overlay -- tall flame shape with orange/yellow core; designed for additive blending.
fn make_fire_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let s = size as f32;
    for y in 0..size {
        for x in 0..size {
            // tx in [0,1] left to right; ty in [0,1] top to bottom.
            let tx = x as f32 / s;
            let ty = y as f32 / s;

            // Heat: 1.0 at the bottom (fire base), 0.0 at the top (flame tip).
            let heat = 1.0 - ty;

            // Flame silhouette: widest at the base, tapering to a point at the top.
            // half_w shrinks as heat drops (moving toward the tip).
            let half_w = 0.48 * heat.powf(0.35);
            let dx = (tx - 0.5).abs();
            if dx >= half_w || heat < 0.02 {
                continue;
            }

            // Edge fade: 1.0 at center, 0.0 at silhouette edge.
            let edge = 1.0 - (dx / half_w).min(1.0);

            // Turbulence from two overlapping sine waves, breaks up the smooth shape.
            let turb = ((tx * 8.7 + ty * 12.3).sin() * 0.5 + 0.5)
                * ((ty * 9.1 - tx * 5.6).cos() * 0.5 + 0.5);

            // Core brightness: strong at the base and center, fades at tip and edges.
            let core = (heat.powf(0.5) * edge * (0.65 + 0.35 * turb)).clamp(0.0, 1.0);

            if core < 0.02 {
                continue;
            }

            let idx = ((y * size + x) * 4) as usize;
            // Color: white-yellow core at high intensity, deep orange-red at edges.
            buf[idx]     = 255;
            buf[idx + 1] = (core * core * 240.0).min(255.0) as u8;
            buf[idx + 2] = ((core - 0.6).max(0.0) / 0.4 * 180.0).min(255.0) as u8;
            buf[idx + 3] = (core * 255.0).min(255.0) as u8;
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

/// D9: Cylindrical label -- a horizontal warning band with repeating chevrons.
fn make_label_texture(size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (size * size * 4) as usize];
    let w = size as f32;
    let h = size as f32;
    for y in 0..size {
        for x in 0..size {
            let tx = x as f32 / w;
            let ty = y as f32 / h;
            // Yellow border bands at top and bottom.
            let border = 0.12_f32;
            let in_border = ty < border || ty > 1.0 - border;
            // Chevron pattern in the middle band: diagonal stripes.
            let chevron = ((tx * 8.0 + ty * 4.0).fract() > 0.5) != ((tx * 8.0 - ty * 4.0).fract() > 0.5);
            let idx = ((y * size + x) * 4) as usize;
            if in_border {
                buf[idx]     = 255;
                buf[idx + 1] = 200;
                buf[idx + 2] = 0;
                buf[idx + 3] = 230;
            } else if chevron {
                buf[idx]     = 30;
                buf[idx + 1] = 30;
                buf[idx + 2] = 30;
                buf[idx + 3] = 220;
            } else {
                buf[idx]     = 220;
                buf[idx + 1] = 220;
                buf[idx + 2] = 220;
                buf[idx + 3] = 200;
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
    pub(crate) hit:       glam::Vec3,
    pub(crate) normal:    glam::Vec3,
    pub(crate) on_column: bool,
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
    pub wall_cpu_mesh:   Option<(Vec<[f32; 3]>, Vec<u32>)>,
    pub ground_cpu_mesh: Option<(Vec<[f32; 3]>, Vec<u32>)>,
    pub column_cpu_mesh: Option<(Vec<[f32; 3]>, Vec<u32>)>,

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

    // D9
    pub column_mesh:   Option<MeshId>,
    pub column_node:   Option<viewport_lib::interaction::selection::NodeId>,
    pub label_tex:     Option<u64>,
    pub cyl_facing:    CylindricalFacing,

    // D10
    pub fire_tex:       Option<u64>,
    pub show_fire:      bool,
    pub fire_alpha:     f32,
}

impl Default for Decal48State {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            wall_mesh:       None,
            ground_mesh:     None,
            wall_cpu_mesh:   None,
            ground_cpu_mesh: None,
            column_cpu_mesh: None,
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
            scroll_handle: None,
            wall_obstacle_node: None,
            show_obstacle: true,
            rune_tex:       None,
            spark_tex:      None,
            show_rune:      true,
            rune_emissive:  2.0,
            show_spark:     true,
            spark_emissive: 3.0,
            edge_fade:      0.2,
            apply_edge_fade: false,
            checker_tex:         None,
            show_corner_decal:   true,
            use_tri_planar:      true,
            tri_blend_sharpness: 4.0,
            column_mesh:         None,
            column_node:         None,
            label_tex:           None,
            cyl_facing:          CylindricalFacing::Outward,
            fire_tex:            None,
            show_fire:           false,
            fire_alpha:          0.6,
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

    let label_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_label_texture(128))
        .expect("label texture upload");
    app.decal48_state.label_tex = Some(label_id);

    let fire_id = res
        .upload_texture(&app.device, &app.queue, 128, 128, &make_fire_texture(128))
        .expect("fire texture upload");
    app.decal48_state.fire_tex = Some(fire_id);

    // Vertical wall: 6 wide (X), 0.2 thick (Y), 4 tall (Z).
    let wall_data = viewport_lib::primitives::cuboid(6.0, 0.2, 4.0);
    app.decal48_state.wall_cpu_mesh = Some((wall_data.positions.clone(), wall_data.indices.clone()));
    let wall_id = res
        .upload_mesh_data(&app.device, &wall_data)
        .expect("wall mesh upload");
    app.decal48_state.wall_mesh = Some(wall_id);

    // Ground floor: 8 wide (X), 6 deep (Y), 0.2 thick (Z).
    let ground_data = viewport_lib::primitives::cuboid(8.0, 6.0, 0.2);
    app.decal48_state.ground_cpu_mesh = Some((ground_data.positions.clone(), ground_data.indices.clone()));
    let ground_id = res
        .upload_mesh_data(&app.device, &ground_data)
        .expect("ground mesh upload");
    app.decal48_state.ground_mesh = Some(ground_id);

    // D9: column (cylinder) standing on the ground, right side.
    // radius=0.3, height=3.0, center at (2.5, 0.5, 1.5).
    res.ensure_matcaps_initialized(&app.device, &app.queue);
    let column_data = viewport_lib::primitives::cylinder(0.3, 3.0, 24);
    app.decal48_state.column_cpu_mesh = Some((column_data.positions.clone(), column_data.indices.clone()));
    let column_id = res
        .upload_mesh_data(&app.device, &column_data)
        .expect("column mesh upload");
    app.decal48_state.column_mesh = Some(column_id);

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

    // D9: column standing on the ground, right side. Wax matcap with black base colour.
    let column_mat = {
        let mut m = Material::from_colour([0.0, 0.0, 0.0]);
        m.shading_model = viewport_lib::ShadingModel::Matcap(res.builtin_matcap_id(BuiltinMatcap::Wax));
        m
    };
    let column_node = scene.add(
        Some(column_id),
        glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.5, 1.5)),
        column_mat,
    );
    app.decal48_state.column_node = Some(column_node);

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

/// Build the mesh lookup table for CPU picking from stored mesh data.
fn decal48_mesh_lookup(
    st: &Decal48State,
) -> std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)> {
    let mut map = std::collections::HashMap::new();
    if let (Some(id), Some(data)) = (st.wall_mesh,   &st.wall_cpu_mesh)   { map.insert(id.index() as u64, data.clone()); }
    if let (Some(id), Some(data)) = (st.ground_mesh, &st.ground_cpu_mesh) { map.insert(id.index() as u64, data.clone()); }
    if let (Some(id), Some(data)) = (st.column_mesh, &st.column_cpu_mesh) { map.insert(id.index() as u64, data.clone()); }
    map
}

/// Place a gunshot decal at the clicked viewport position using CPU ray-casting.
pub(crate) fn decal48_place(app: &mut App, cursor: glam::Vec2, vp_size: glam::Vec2) {
    if !app.decal48_state.built { return; }
    let vp_inv = app.camera.view_proj_matrix().inverse();
    let (ro, rd) = viewport_lib::picking::screen_to_ray(cursor, vp_size, vp_inv);

    let mesh_lookup = decal48_mesh_lookup(&app.decal48_state);
    let Some(pick) = viewport_lib::picking::pick_scene_nodes_cpu(
        ro, rd, &app.decal48_state.scene, &mesh_lookup,
    ) else { return };

    let hit    = pick.world_pos;
    let normal = pick.normal;

    // Detect whether the hit was on the column by comparing the picked node ID.
    let column_node_id = app.decal48_state.column_node.unwrap_or(u64::MAX);
    let on_column = pick.id == column_node_id;

    let st = &mut app.decal48_state;
    let Some(tex) = st.albedo_tex else { return };

    // Build the decal transform. For column hits, local Z = world Z (cylinder axis)
    // and XY is fixed at the column's radial scale; depth controls height coverage.
    let (transform, projection) = if on_column {
        let t = glam::Mat4::from_cols(
            glam::Vec4::new(0.65, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.65, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, st.decal_depth, 0.0),
            glam::Vec4::new(2.5, 0.5, hit.z, 1.0),
        ).to_cols_array_2d();
        (t, DecalProjection::Cylindrical { facing: CylindricalFacing::Outward })
    } else {
        (decal_transform(hit, normal, st.decal_size, st.decal_depth), DecalProjection::Planar)
    };

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
        item.projection            = projection;
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
            on_column,
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

    // Scroll animation: always-on black/white stripe across the full width of the wall top.
    // Wall: 6 wide (X), 4 tall (Z), face at y=0, normal +Y.
    // Top 1/6th of wall height = 0.67 tall, centered at z = 3.67.
    // tangent = -X, bitangent = +Z (derived from n=+Y, ref_up=+Z).
    if st.scroll_handle.is_none() {
        if let Some(stripe) = st.stripe_tex {
            let transform = glam::Mat4::from_cols(
                glam::Vec4::new(-6.0, 0.0, 0.0, 0.0),
                glam::Vec4::new(0.0, 0.0, 0.67, 0.0),
                glam::Vec4::new(0.0, 0.3, 0.0, 0.0),
                glam::Vec4::new(0.0, 0.0, 3.67, 1.0),
            ).to_cols_array_2d();
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = stripe;
            item.alpha      = 0.85;
            item.sort_key   = -10;
            let handle = st.scene.add_decal_animated(
                item,
                DecalAnimation::UvScroll { vx: 0.15, vy: 0.0 },
                None,
            );
            st.scroll_handle = Some(handle);
        }
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
        let Some(tex) = st.albedo_tex else { continue };
        let (transform, projection) = if placed.on_column {
            let t = glam::Mat4::from_cols(
                glam::Vec4::new(0.65, 0.0, 0.0, 0.0),
                glam::Vec4::new(0.0, 0.65, 0.0, 0.0),
                glam::Vec4::new(0.0, 0.0, st.decal_depth, 0.0),
                glam::Vec4::new(2.5, 0.5, placed.hit.z, 1.0),
            ).to_cols_array_2d();
            (t, DecalProjection::Cylindrical { facing: CylindricalFacing::Outward })
        } else {
            (decal_transform(placed.hit, placed.normal, st.decal_size, st.decal_depth), DecalProjection::Planar)
        };
        let mut item = DecalItem::default();
        item.transform             = transform;
        item.texture_id            = tex;
        item.blend_mode            = st.blend_mode;
        item.alpha                 = st.alpha;
        item.normal_texture_id     = if st.use_normal_map { st.normal_tex } else { None };
        item.normal_blend_strength = st.normal_blend;
        item.roughness             = st.decal_roughness;
        item.metallic              = st.decal_metallic;
        item.projection            = projection;
        item.edge_fade             = if st.apply_edge_fade { st.edge_fade } else { 0.0 };
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

    // D9: cylindrical label decal wrapped around the column.
    // The decal's local Z axis = world Z (column axis). Scale XY just outside
    // the column radius (0.3 -> 0.65 world units = 0.5 in local), Z covers the
    // middle section of the column (world z = [0.5, 2.5]).
    if let Some(label) = st.label_tex {
        let transform = glam::Mat4::from_cols(
            glam::Vec4::new(0.65, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.65, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, 2.0, 0.0),
            glam::Vec4::new(2.5, 0.5, 1.5, 1.0),
        ).to_cols_array_2d();
        let mut item = DecalItem::default();
        item.transform  = transform;
        item.texture_id = label;
        item.alpha      = 0.95;
        item.edge_fade  = 0.05;
        item.projection = DecalProjection::Cylindrical { facing: st.cyl_facing };
        fd.scene.decals.push(item);
    }

    // D6: spark-impact on the wall, alongside the rune.
    if st.show_spark {
        if let Some(spark) = st.spark_tex {
            let transform = decal_transform(
                glam::Vec3::new(2.2, 0.0, 2.0),
                glam::Vec3::Y,
                0.4,
                1.0,
            );
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = spark;
            item.alpha      = 1.0;
            item.emissive   = st.spark_emissive;
            fd.scene.decals.push(item);
        }
    }

    // D10: fire overlay on the wall, additive blend.
    // Non-square: 1.2 wide, 2.4 tall (flame shape). Wall +Y normal -> tangent=-X, bitangent=+Z.
    if st.show_fire {
        if let Some(fire) = st.fire_tex {
            let transform = glam::Mat4::from_cols(
                glam::Vec4::new(-1.2, 0.0, 0.0, 0.0),
                glam::Vec4::new(0.0, 0.0, 2.4, 0.0),
                glam::Vec4::new(0.0, 0.4, 0.0, 0.0),
                glam::Vec4::new(-1.5, 0.0, 1.2, 1.0),
            ).to_cols_array_2d();
            let mut item = DecalItem::default();
            item.transform  = transform;
            item.texture_id = fire;
            item.blend_mode = DecalBlendMode::Additive;
            item.alpha      = st.fire_alpha;
            item.emissive   = 1.5;
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
            if ui.selectable_label(cur == DecalBlendMode::Additive, "Additive").clicked() {
                app.decal48_state.blend_mode = DecalBlendMode::Additive;
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
        ui.small("Black/white stripe decal scrolls along the top strip of the wall.");

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
        ui.label("Cylindrical projection:");
        ui.small("Wraps the decal around the column using angle/Z UV coordinates.");
        ui.small("The facing check rejects surfaces that point the wrong way.");
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            let cur = app.decal48_state.cyl_facing;
            if ui.selectable_label(cur == CylindricalFacing::Outward, "Outward (column exterior)").clicked() {
                app.decal48_state.cyl_facing = CylindricalFacing::Outward;
            }
            if ui.selectable_label(cur == CylindricalFacing::Inward, "Inward (tube interior)").clicked() {
                app.decal48_state.cyl_facing = CylindricalFacing::Inward;
            }
        });

        ui.add_space(6.0);
        ui.separator();
        ui.label("Additive blend (D10):");
        ui.small("Additive decals brighten the receiver instead of replacing its colour.");
        ui.small("Stack multiple additive decals to accumulate brightness.");
        ui.add_space(2.0);
        ui.checkbox(&mut app.decal48_state.show_fire, "Fire overlay (wall left, additive)");
        if app.decal48_state.show_fire {
            ui.add(
                egui::Slider::new(&mut app.decal48_state.fire_alpha, 0.0..=1.0)
                    .text("Fire alpha"),
            );
        }

        ui.add_space(6.0);
        ui.separator();
        ui.label("Surface guide:");
        ui.small("Wall (y = 0):       gunshot craters, glowing rune, spark impact, fire overlay.");
        ui.small("Ground left  (x<0): muddy footprints (static).");
        ui.small("Ground right (x>=0): blood splatter.");
    });
}
