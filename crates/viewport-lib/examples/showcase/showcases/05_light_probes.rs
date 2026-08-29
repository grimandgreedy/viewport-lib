//! Indirect lighting, three ways, switchable by the top chip:
//!
//! - **Light probes (SH):** three coloured SH probes (warm, green, cool) light a
//!   row of white spheres by position, so the row reads as a colour gradient.
//!   Toggle probes off and every sphere falls back to flat sky ambient.
//! - **Env zones:** three world-space environment zones (warm, cool, green) each
//!   hold their own IBL environment. A row of polished metal spheres crosses the
//!   zones and reflects whichever environment covers it, cross-fading in the
//!   overlaps. Clear the zones and every sphere reflects the single default
//!   environment instead.
//! - **Lightmaps:** an upright panel carries a baked lightmap texture, sampled
//!   by a second UV set. No runtime light produces the coloured light pools and
//!   dark corners on it: they are baked in. The toggle blends the same panel
//!   Off / Replace / Add / AmbientOcclusion.
//! - **Baked GI:** a small Cornell-style box (grey floor, red and blue walls, a
//!   few blocks) whose floor lightmap is path-traced at runtime. The floor is
//!   rasterised into its lightmap UVs to get a world point per texel, then a GI
//!   hemisphere is shot from each; the atlas ends up with contact shadows, wall
//!   colour bleed, and corner darkening. Toggle it against the flat fallback.
//!
//! Probes and zones are the "indirect diffuse comes from somewhere per-object /
//! per-fragment" story; the lightmap modes are the "indirect light is
//! precomputed into a texture over a static surface" story : the last one bakes
//! that texture live instead of painting it.

use eframe::egui;
use glam::{Mat3, Mat4, Vec2, Vec3};
use viewport_lib::bake::{TexelGeometry, rasterize_texel_gbuffer};
use viewport_lib::raytrace::{
    RtLight, RtMaterial, RtScene, RtSettings, TexelSurfaces, bake_lightmap,
};
use viewport_lib::resources::{
    LightProbe, LightProbeSet, LightmapData, LightmapMode, SHCoefficients, TextureId,
};
use viewport_lib::{
    Aabb, BackfacePolicy, EnvironmentMapId, EnvironmentSettings, EnvironmentZone,
    IndirectLightSource, LightKind, LightSource, Material, MeshId, NodeId, primitives,
};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

// ---------------------------------------------------------------------------
// Light-probe mode
// ---------------------------------------------------------------------------

/// The three probe colours and their x positions along the row.
const PROBES: [([f32; 3], f32); 3] = [
    ([1.0, 0.45, 0.12], -4.0), // warm
    ([0.20, 0.80, 0.30], 0.0), // green
    ([0.20, 0.40, 1.00], 4.0), // cool
];

const PROBE_ROW_Z: f32 = 1.0;
const N_SPHERES: usize = 9;

/// Build a probe whose SH radiates one flat colour in every direction.
fn flat_sh(colour: [f32; 3]) -> SHCoefficients {
    const INV_Y00: f32 = 1.0 / 0.282095;
    let mut sh = SHCoefficients::default();
    sh.r[0] = colour[0] * INV_Y00;
    sh.g[0] = colour[1] * INV_Y00;
    sh.b[0] = colour[2] * INV_Y00;
    sh
}

/// Build the probe field from the currently enabled probes.
fn build_probe_set(enabled: &[bool]) -> LightProbeSet {
    let mut probes = Vec::new();
    for (i, &(colour, x)) in PROBES.iter().enumerate() {
        if enabled.get(i).copied().unwrap_or(false) {
            probes.push(LightProbe {
                position: [x, 0.0, PROBE_ROW_Z],
                sh: flat_sh(colour),
            });
        }
    }
    LightProbeSet::new(probes)
}

// ---------------------------------------------------------------------------
// Environment-zone mode
// ---------------------------------------------------------------------------

/// One zoned environment: a name, the reflected sky/ground colours, and the x
/// position of its zone along the row.
struct EnvDef {
    name: &'static str,
    sky: [f32; 3],
    ground: [f32; 3],
    x: f32,
}

const ENVS: [EnvDef; 3] = [
    EnvDef {
        name: "Warm",
        sky: [1.0, 0.55, 0.2],
        ground: [0.35, 0.12, 0.05],
        x: -5.0,
    },
    EnvDef {
        name: "Cool",
        sky: [0.3, 0.6, 1.0],
        ground: [0.05, 0.14, 0.4],
        x: 0.0,
    },
    EnvDef {
        name: "Green",
        sky: [0.55, 0.9, 0.4],
        ground: [0.08, 0.28, 0.12],
        x: 5.0,
    },
];

/// The default (layer 0) environment, used as the skybox and wherever no zone
/// covers a fragment.
const DEFAULT_SKY: [f32; 3] = [0.55, 0.6, 0.68];
const DEFAULT_GROUND: [f32; 3] = [0.14, 0.14, 0.17];

const ZONE_ROW_Z: f32 = 1.1;
const N_METAL: usize = 13;
const ROW_HALF_X: f32 = 8.0;
const ZONE_HALF_X: f32 = 1.8;
const ZONE_FADE: f32 = 1.7;

/// A vertical-gradient equirect panorama (sky at the +Z pole, ground at -Z).
fn equirect_gradient(sky: [f32; 3], ground: [f32; 3], w: u32, h: u32) -> Vec<f32> {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let v = y as f32 / (h - 1).max(1) as f32;
        let t = v * v * (3.0 - 2.0 * v); // soft horizon
        for _x in 0..w {
            px.push(sky[0] + (ground[0] - sky[0]) * t);
            px.push(sky[1] + (ground[1] - sky[1]) * t);
            px.push(sky[2] + (ground[2] - sky[2]) * t);
            px.push(1.0);
        }
    }
    px
}

// ---------------------------------------------------------------------------
// Lightmap mode
// ---------------------------------------------------------------------------

const LIGHTMAP_TEX: u32 = 256;
/// Toggle labels: index 0 is off, 1..=3 map to the three blend modes.
const LIGHTMAP_MODES: [&str; 4] = ["Off", "Replace", "Add", "AO"];

/// Paint a baked-radiance lightmap: soft coloured light pools over a dark base,
/// darkened toward the edges. This is the kind of indirect light a bake captures
/// and a single realtime light cannot cheaply reproduce.
fn bake_radiance(w: u32, h: u32) -> Vec<u8> {
    // (centre uv, radius, colour).
    let pools: [([f32; 2], f32, [f32; 3]); 3] = [
        ([0.28, 0.32], 0.34, [1.0, 0.55, 0.2]), // warm
        ([0.72, 0.68], 0.32, [0.25, 0.5, 1.0]), // cool
        ([0.5, 0.5], 0.24, [0.3, 0.9, 0.4]),    // green
    ];
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        let v = y as f32 / (h - 1).max(1) as f32;
        for x in 0..w {
            let u = x as f32 / (w - 1).max(1) as f32;
            let mut c = [0.04f32, 0.04, 0.05];
            for (ctr, r, col) in pools.iter() {
                let d = ((u - ctr[0]).powi(2) + (v - ctr[1]).powi(2)).sqrt();
                let f = (1.0 - d / r).max(0.0);
                let f = f * f;
                c[0] += col[0] * f;
                c[1] += col[1] * f;
                c[2] += col[2] * f;
            }
            // Baked edge darkening (contact / corner occlusion).
            let edge = 1.0 - (2.0 * u - 1.0).abs().max((2.0 * v - 1.0).abs());
            let vign = 0.3 + 0.7 * edge.clamp(0.0, 1.0);
            let i = ((y * w + x) * 4) as usize;
            px[i] = ((c[0] * vign).clamp(0.0, 1.0) * 255.0) as u8;
            px[i + 1] = ((c[1] * vign).clamp(0.0, 1.0) * 255.0) as u8;
            px[i + 2] = ((c[2] * vign).clamp(0.0, 1.0) * 255.0) as u8;
            px[i + 3] = 255;
        }
    }
    px
}

/// Paint a grayscale ambient-occlusion lightmap: bright where light reaches,
/// dark pools where geometry would trap it. The shader reads the red channel as
/// the occlusion factor.
fn bake_ao(w: u32, h: u32) -> Vec<u8> {
    let occ: [([f32; 2], f32); 3] = [
        ([0.3, 0.32], 0.3),
        ([0.68, 0.66], 0.32),
        ([0.5, 0.85], 0.22),
    ];
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        let v = y as f32 / (h - 1).max(1) as f32;
        for x in 0..w {
            let u = x as f32 / (w - 1).max(1) as f32;
            let mut ao = 1.0f32;
            for (ctr, r) in occ.iter() {
                let d = ((u - ctr[0]).powi(2) + (v - ctr[1]).powi(2)).sqrt();
                let f = (1.0 - d / r).max(0.0);
                ao -= 0.85 * f * f;
            }
            let g = (ao.clamp(0.05, 1.0) * 255.0) as u8;
            let i = ((y * w + x) * 4) as usize;
            px[i] = g;
            px[i + 1] = g;
            px[i + 2] = g;
            px[i + 3] = 255;
        }
    }
    px
}

// ---------------------------------------------------------------------------
// Baked GI mode (path traced)
// ---------------------------------------------------------------------------
//
// A miniature Cornell-style box: a grey floor between a red wall and a blue wall,
// with a few blocks. The floor's lighting is baked at runtime with the path
// tracer : rasterise the floor into its lightmap atlas to get a world position
// per texel, then shoot the GI hemisphere from each. The result carries soft
// contact shadows under the blocks, colour bleed from the walls, and corner
// darkening, none of which a single realtime light reproduces. The walls and
// blocks are lit flat for context; only the floor is baked.

const GI_ATLAS: u32 = 256;
const GI_FLOOR_W: f32 = 16.0;
const GI_FLOOR_D: f32 = 12.0;
const GI_WALL_H: f32 = 6.0;
const GI_FLOOR_ALBEDO: [f32; 3] = [0.80, 0.80, 0.80];
const GI_RED: [f32; 3] = [0.85, 0.12, 0.10];
const GI_BLUE: [f32; 3] = [0.12, 0.28, 0.90];
const GI_BLOCK_ALBEDO: [f32; 3] = [0.80, 0.80, 0.82];

/// Local mesh geometry kept around so the bake can transform it to world space
/// (the display side uploads the same primitives and places them with the same
/// transforms).
#[derive(Clone, Default)]
struct Geo {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    idx: Vec<u32>,
}

/// One piece of the baked-GI scene: which local geometry, where, and its albedo.
struct GiPiece {
    kind: GiKind,
    xf: Mat4,
    albedo: [f32; 3],
}

#[derive(Clone, Copy, PartialEq)]
enum GiKind {
    Floor,
    Wall,
    Block,
}

/// The scene layout, shared by the display build and the bake so the two always
/// match. The floor sits at the origin; the walls stand on its short edges; the
/// blocks rest on it.
fn gi_layout() -> Vec<GiPiece> {
    let hw = GI_FLOOR_W * 0.5;
    let half_pi = std::f32::consts::FRAC_PI_2;
    let block = |s: f32, x: f32, y: f32| {
        Mat4::from_translation(Vec3::new(x, y, 0.5 * s)) * Mat4::from_scale(Vec3::splat(s))
    };
    vec![
        GiPiece {
            kind: GiKind::Floor,
            xf: Mat4::IDENTITY,
            albedo: GI_FLOOR_ALBEDO,
        },
        // Left wall (red): stand the plane up and face it +X.
        GiPiece {
            kind: GiKind::Wall,
            xf: Mat4::from_translation(Vec3::new(-hw, 0.0, GI_WALL_H * 0.5))
                * Mat4::from_rotation_y(half_pi),
            albedo: GI_RED,
        },
        // Right wall (blue): face it -X.
        GiPiece {
            kind: GiKind::Wall,
            xf: Mat4::from_translation(Vec3::new(hw, 0.0, GI_WALL_H * 0.5))
                * Mat4::from_rotation_y(-half_pi),
            albedo: GI_BLUE,
        },
        GiPiece {
            kind: GiKind::Block,
            xf: block(3.0, -3.0, -1.0),
            albedo: GI_BLOCK_ALBEDO,
        },
        GiPiece {
            kind: GiKind::Block,
            xf: block(2.0, 2.6, 2.2),
            albedo: GI_BLOCK_ALBEDO,
        },
        GiPiece {
            kind: GiKind::Block,
            xf: block(1.4, 0.6, -3.2),
            albedo: GI_BLOCK_ALBEDO,
        },
    ]
}

/// Transform local geometry by `xf` and add it to the trace scene as an opaque
/// diffuse surface tinted `albedo`.
fn add_world_mesh(scene: &mut RtScene, geo: &Geo, xf: Mat4, albedo: [f32; 3]) {
    let nrm_mat = Mat3::from_mat4(xf);
    let positions: Vec<Vec3> = geo
        .pos
        .iter()
        .map(|p| xf.transform_point3(Vec3::from_array(*p)))
        .collect();
    let normals: Vec<Vec3> = geo
        .nrm
        .iter()
        .map(|n| (nrm_mat * Vec3::from_array(*n)).normalize_or_zero())
        .collect();
    scene.add_mesh(
        &positions,
        &geo.idx,
        Some(&normals),
        RtMaterial {
            base_colour: albedo,
            roughness: 0.9,
            ..RtMaterial::default()
        },
    );
}

/// Pack a baked irradiance atlas into an sRGB RGBA8 lightmap texture. Incident
/// irradiance is turned into diffuse radiosity (`albedo * E / pi`), tonemapped,
/// and gamma-encoded; empty texels go black.
fn irradiance_to_texture(img: &viewport_lib::raytrace::RtImage, albedo: [f32; 3]) -> Vec<u8> {
    const EXPOSURE: f32 = 1.25;
    let inv_pi = 1.0 / std::f32::consts::PI;
    let mut out = vec![0u8; (img.width * img.height * 4) as usize];
    for (i, px) in img.rgba.chunks_exact(4).enumerate() {
        if px[3] <= 0.5 {
            out[i * 4 + 3] = 255;
            continue;
        }
        for c in 0..3 {
            let lin = px[c] * albedo[c] * inv_pi * EXPOSURE;
            let tone = lin / (1.0 + lin);
            out[i * 4 + c] = (tone.powf(1.0 / 2.2).clamp(0.0, 1.0) * 255.0) as u8;
        }
        out[i * 4 + 3] = 255;
    }
    out
}

// ---------------------------------------------------------------------------

/// A small emissive marker cube tinted `colour`, dimmed when `on` is false.
fn marker_material(colour: [f32; 3], on: bool) -> Material {
    let base = if on { 0.18 } else { 0.03 };
    let mut m = Material::pbr(
        [colour[0] * base, colour[1] * base, colour[2] * base],
        0.0,
        0.85,
    );
    m.emissive = if on {
        colour
    } else {
        [colour[0] * 0.1, colour[1] * 0.1, colour[2] * 0.1]
    };
    m
}

pub struct IndirectLightingShowcase {
    /// 0 = light probes, 1 = environment zones, 2 = lightmaps, 3 = baked GI.
    mode: usize,
    shown: Option<usize>,
    /// Nodes for the active mode, torn down on a mode switch.
    nodes: Vec<NodeId>,
    /// Shared meshes, uploaded once in `setup`.
    sphere: Option<MeshId>,
    cube: Option<MeshId>,

    // Probe mode.
    probe_spheres: Vec<NodeId>,
    probe_markers: Vec<NodeId>,
    probe_enabled: Vec<bool>,
    use_probes: bool,
    applied_probes: Option<(bool, Vec<bool>)>,

    // Zone mode.
    env_ids: Vec<EnvironmentMapId>,
    zone_markers: Vec<NodeId>,
    zone_enabled: Vec<bool>,
    use_zones: bool,
    applied_zones: Option<(bool, Vec<bool>)>,

    // Lightmap mode.
    lightmap_panel: Option<MeshId>,
    lightmap_uv1: Vec<Vec2>,
    radiance_tex: Option<TextureId>,
    ao_tex: Option<TextureId>,
    /// 0 = off, 1 = Replace, 2 = Add, 3 = AmbientOcclusion.
    lightmap_sel: usize,
    applied_lightmap: Option<usize>,

    // Baked GI mode. Meshes and local geometry are uploaded once in `setup`; the
    // atlas is path-traced the first time the mode is opened and cached.
    gi_floor_mesh: Option<MeshId>,
    gi_wall_mesh: Option<MeshId>,
    gi_block_mesh: Option<MeshId>,
    gi_floor: Geo,
    gi_wall: Geo,
    gi_block: Geo,
    gi_floor_uv1: Vec<Vec2>,
    gi_baked_tex: Option<TextureId>,
    gi_on: bool,
    gi_applied: Option<bool>,
}

impl IndirectLightingShowcase {
    pub fn new() -> Self {
        Self {
            mode: 0,
            shown: None,
            nodes: Vec::new(),
            sphere: None,
            cube: None,
            probe_spheres: Vec::new(),
            probe_markers: Vec::new(),
            probe_enabled: vec![true; PROBES.len()],
            use_probes: true,
            applied_probes: None,
            env_ids: Vec::new(),
            zone_markers: Vec::new(),
            zone_enabled: vec![true; ENVS.len()],
            use_zones: true,
            applied_zones: None,
            lightmap_panel: None,
            lightmap_uv1: Vec::new(),
            radiance_tex: None,
            ao_tex: None,
            lightmap_sel: 1,
            applied_lightmap: None,
            gi_floor_mesh: None,
            gi_wall_mesh: None,
            gi_block_mesh: None,
            gi_floor: Geo::default(),
            gi_wall: Geo::default(),
            gi_block: Geo::default(),
            gi_floor_uv1: Vec::new(),
            gi_baked_tex: None,
            gi_on: true,
            gi_applied: None,
        }
    }

    /// Path-trace the floor's lightmap: rasterise the floor into its atlas to get
    /// a world position and normal per texel, build the trace scene from the
    /// shared layout, bake incident irradiance from those texels, and upload the
    /// result as a lightmap texture. Cached in `gi_baked_tex` after the first run.
    fn bake_gi(&mut self, ctx: &mut ShowcaseCtx) {
        let device = ctx.device;
        let queue = ctx.queue;

        // Texel G-buffer for the floor (identity transform: local == world).
        let gbuf = rasterize_texel_gbuffer(
            device,
            queue,
            &TexelGeometry {
                positions: &self.gi_floor.pos,
                normals: &self.gi_floor.nrm,
                uv1: &self
                    .gi_floor_uv1
                    .iter()
                    .map(|u| [u.x, u.y])
                    .collect::<Vec<_>>(),
                indices: &self.gi_floor.idx,
                model: Mat4::IDENTITY,
            },
            GI_ATLAS,
            GI_ATLAS,
        );

        // Trace scene: the same geometry the layout places, plus a key light and
        // a soft sky. The coloured walls are what bleed onto the floor.
        let mut scene = RtScene::new();
        scene.set_sky([0.35, 0.4, 0.5], [0.08, 0.08, 0.10]);
        for piece in gi_layout() {
            let geo = match piece.kind {
                GiKind::Floor => &self.gi_floor,
                GiKind::Wall => &self.gi_wall,
                GiKind::Block => &self.gi_block,
            };
            add_world_mesh(&mut scene, geo, piece.xf, piece.albedo);
        }
        scene.add_light(RtLight::Directional {
            direction: Vec3::new(0.35, -0.2, 1.0).normalize().to_array(),
            colour: [3.2, 3.1, 2.9],
        });

        let img = bake_lightmap(
            device,
            queue,
            &scene,
            &TexelSurfaces {
                width: gbuf.width,
                height: gbuf.height,
                world_pos: &gbuf.world_pos,
                world_normal: &gbuf.world_normal,
            },
            &RtSettings {
                samples: 160,
                max_bounces: 4,
                denoise: false,
                seed: 0,
            },
        );

        let texels = irradiance_to_texture(&img, GI_FLOOR_ALBEDO);
        let tex = ctx
            .session
            .resources_mut()
            .upload_texture(device, queue, img.width, img.height, &texels)
            .unwrap();
        self.gi_baked_tex = Some(tex);
    }

    /// Tear down the current scene nodes and build the active mode's geometry,
    /// leaving the renderer indirect-light state to be (re)applied in `update`.
    fn rebuild(&mut self, session: &mut viewport_lib::ViewportInstance) {
        if !self.nodes.is_empty() {
            let ids = std::mem::take(&mut self.nodes);
            session.scene_mut().remove_many(&ids);
        }
        self.probe_spheres.clear();
        self.probe_markers.clear();
        self.zone_markers.clear();

        let sphere = self.sphere.unwrap();
        let cube = self.cube.unwrap();

        // Restore the standard key light for the probe/zone/lightmap modes; the
        // baked-GI mode overrides it below so its floor reads as purely baked.
        {
            let mut key = LightSource::default();
            key.kind = LightKind::Directional {
                direction: [0.3, 0.2, 1.0],
            };
            key.colour = [1.0, 1.0, 1.0];
            key.intensity = 0.32;
            key.cast_shadows = false;
            let l = &mut session.effects_mut().lighting;
            l.lights = vec![key];
            l.hemisphere_intensity = 0.3;
            l.sky_colour = [0.7, 0.78, 0.9];
            l.ground_colour = [0.3, 0.3, 0.3];
        }

        if self.mode == 0 {
            // Probe mode: hemisphere-ambient fallback (no global IBL), clear any
            // zones, and light white spheres from the SH field.
            session.effects_mut().environment = None;
            session.renderer_mut().clear_environment_zones();

            for &(colour, x) in PROBES.iter() {
                let id = session.scene_mut().add(
                    Some(cube),
                    Mat4::from_translation(Vec3::new(x, 0.0, PROBE_ROW_Z + 1.6)),
                    marker_material(colour, true),
                );
                self.probe_markers.push(id);
                self.nodes.push(id);
            }
            let x0 = PROBES[0].1;
            let x1 = PROBES[PROBES.len() - 1].1;
            for i in 0..N_SPHERES {
                let t = i as f32 / (N_SPHERES - 1) as f32;
                let x = x0 + (x1 - x0) * t;
                let id = session.scene_mut().add(
                    Some(sphere),
                    Mat4::from_translation(Vec3::new(x, 0.0, PROBE_ROW_Z)),
                    Material::pbr([1.0, 1.0, 1.0], 0.0, 0.85),
                );
                self.probe_spheres.push(id);
                self.nodes.push(id);
            }
        } else if self.mode == 1 {
            // Zone mode: enable the default environment (+ skybox), clear probes,
            // and reflect the zoned environments off polished metal spheres.
            session.effects_mut().environment = Some(EnvironmentSettings {
                intensity: 1.0,
                rotation: 0.0,
                show_skybox: true,
            });
            session
                .renderer_mut()
                .set_light_probes(LightProbeSet::new(vec![]));

            for i in 0..N_METAL {
                let t = i as f32 / (N_METAL - 1) as f32;
                let x = -ROW_HALF_X + 2.0 * ROW_HALF_X * t;
                let id = session.scene_mut().add(
                    Some(sphere),
                    Mat4::from_translation(Vec3::new(x, 0.0, ZONE_ROW_Z)),
                    Material::pbr([1.0, 1.0, 1.0], 1.0, 0.12),
                );
                self.nodes.push(id);
            }
            for e in ENVS.iter() {
                let id = session.scene_mut().add(
                    Some(cube),
                    Mat4::from_translation(Vec3::new(e.x, 0.0, ZONE_ROW_Z + 2.2)),
                    marker_material(e.sky, true),
                );
                self.zone_markers.push(id);
                self.nodes.push(id);
            }
        } else if self.mode == 2 {
            // Lightmap mode: no IBL, no zones, no probes. An upright panel carries
            // the baked lightmap; a modest hemisphere ambient stays so Add mode
            // has something to add onto. The blend mode is applied in `update`.
            session.effects_mut().environment = None;
            session.renderer_mut().clear_environment_zones();
            session
                .renderer_mut()
                .set_light_probes(LightProbeSet::new(vec![]));

            let panel = self.lightmap_panel.unwrap();
            let mut mat = Material::pbr([1.0, 1.0, 1.0], 0.0, 0.9);
            // The panel is single-sided geometry; show it whichever way it faces.
            mat.backface_policy = BackfacePolicy::Identical;
            // Stand the XY panel upright, its face toward -Y (the camera).
            let xf = Mat4::from_translation(Vec3::new(0.0, 3.0, 2.6))
                * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
            let id = session.scene_mut().add(Some(panel), xf, mat);
            self.nodes.push(id);
        } else {
            // Baked GI mode: no IBL, no zones, no probes. Runtime lights are off
            // so the floor shows only its path-traced lightmap; the walls and
            // blocks get a dim hemisphere ambient for context. The atlas is baked
            // and applied in `update`.
            session.effects_mut().environment = None;
            session.renderer_mut().clear_environment_zones();
            session
                .renderer_mut()
                .set_light_probes(LightProbeSet::new(vec![]));
            {
                let l = &mut session.effects_mut().lighting;
                l.lights = Vec::new();
                l.hemisphere_intensity = 0.32;
                l.sky_colour = [0.5, 0.54, 0.62];
                l.ground_colour = [0.16, 0.16, 0.18];
            }

            for piece in gi_layout() {
                let mesh = match piece.kind {
                    GiKind::Floor => self.gi_floor_mesh.unwrap(),
                    GiKind::Wall => self.gi_wall_mesh.unwrap(),
                    GiKind::Block => self.gi_block_mesh.unwrap(),
                };
                // The floor's display albedo is white; its baked lightmap already
                // carries the surface colour. Walls and blocks show their albedo.
                let base = if piece.kind == GiKind::Floor {
                    [1.0, 1.0, 1.0]
                } else {
                    piece.albedo
                };
                let mut mat = Material::pbr(base, 0.0, 0.9);
                mat.backface_policy = BackfacePolicy::Identical;
                let id = session.scene_mut().add(Some(mesh), piece.xf, mat);
                self.nodes.push(id);
            }
        }

        // Force the per-mode indirect-light state to re-apply this frame.
        self.applied_probes = None;
        self.applied_zones = None;
        self.applied_lightmap = None;
        self.gi_applied = None;
    }

    /// The active environment zones for the zone-mode toggles.
    fn zones(&self) -> Vec<EnvironmentZone> {
        if !self.use_zones {
            return Vec::new();
        }
        ENVS.iter()
            .enumerate()
            .filter(|(i, _)| self.zone_enabled[*i])
            .map(|(i, e)| EnvironmentZone {
                bounds: Aabb {
                    min: Vec3::new(e.x - ZONE_HALF_X, -8.0, ZONE_ROW_Z - 8.0),
                    max: Vec3::new(e.x + ZONE_HALF_X, 8.0, ZONE_ROW_Z + 8.0),
                },
                environment: self.env_ids[i],
                fade_distance: ZONE_FADE,
                // Distant gradient environments, not local captures: no parallax.
                parallax: false,
            })
            .collect()
    }
}

impl Showcase for IndirectLightingShowcase {
    fn name(&self) -> &str {
        "Indirect lighting"
    }

    fn setup(&mut self, ctx: &mut SetupCtx) {
        self.sphere = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &primitives::sphere(0.6, 48, 24))
                .unwrap(),
        );
        self.cube = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &primitives::cube(0.38))
                .unwrap(),
        );

        // Lightmap panel: a UV-mapped quad plus its two baked textures. UV1
        // reuses the quad's own UVs, so each corner samples a texture corner.
        let panel = primitives::plane(14.0, 8.0);
        self.lightmap_uv1 = panel
            .uvs
            .as_ref()
            .map(|uvs| uvs.iter().map(|u| Vec2::new(u[0], u[1])).collect())
            .unwrap_or_default();
        self.lightmap_panel = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &panel)
                .unwrap(),
        );
        self.radiance_tex = Some(
            ctx.session
                .resources_mut()
                .upload_texture(
                    ctx.device,
                    ctx.queue,
                    LIGHTMAP_TEX,
                    LIGHTMAP_TEX,
                    &bake_radiance(LIGHTMAP_TEX, LIGHTMAP_TEX),
                )
                .unwrap(),
        );
        self.ao_tex = Some(
            ctx.session
                .resources_mut()
                .upload_texture(
                    ctx.device,
                    ctx.queue,
                    LIGHTMAP_TEX,
                    LIGHTMAP_TEX,
                    &bake_ao(LIGHTMAP_TEX, LIGHTMAP_TEX),
                )
                .unwrap(),
        );

        // Baked-GI meshes: a floor, a wall (reused for both walls), and a unit
        // block. The local geometry is kept so the bake can transform it to world
        // space to match what the display places.
        let floor = primitives::plane(GI_FLOOR_W, GI_FLOOR_D);
        self.gi_floor = Geo {
            pos: floor.positions.clone(),
            nrm: floor.normals.clone(),
            idx: floor.indices.clone(),
        };
        self.gi_floor_uv1 = floor
            .uvs
            .as_ref()
            .map(|uvs| uvs.iter().map(|u| Vec2::new(u[0], u[1])).collect())
            .unwrap_or_default();
        self.gi_floor_mesh = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &floor)
                .unwrap(),
        );
        let wall = primitives::plane(GI_WALL_H, GI_FLOOR_D);
        self.gi_wall = Geo {
            pos: wall.positions.clone(),
            nrm: wall.normals.clone(),
            idx: wall.indices.clone(),
        };
        self.gi_wall_mesh = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &wall)
                .unwrap(),
        );
        let blk = primitives::cube(1.0);
        self.gi_block = Geo {
            pos: blk.positions.clone(),
            nrm: blk.normals.clone(),
            idx: blk.indices.clone(),
        };
        self.gi_block_mesh = Some(
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &blk)
                .unwrap(),
        );

        // Environments for zone mode: default (layer 0) + one per zone. Uploaded
        // once here; they persist on the resources across mode switches.
        let default_px = equirect_gradient(DEFAULT_SKY, DEFAULT_GROUND, 64, 32);
        ctx.session
            .renderer_mut()
            .upload_environment_map(ctx.device, ctx.queue, &default_px, 64, 32)
            .unwrap();
        self.env_ids.clear();
        for e in ENVS.iter() {
            let px = equirect_gradient(e.sky, e.ground, 64, 32);
            let id = ctx
                .session
                .renderer_mut()
                .upload_environment(ctx.device, ctx.queue, &px, 64, 32)
                .unwrap();
            self.env_ids.push(id);
        }

        // A dim key light gives the spheres shape; the indirect term (probe SH or
        // reflected environment) carries the colour, so keep direct light low.
        let mut key = LightSource::default();
        key.kind = LightKind::Directional {
            direction: [0.3, 0.2, 1.0],
        };
        key.colour = [1.0, 1.0, 1.0];
        key.intensity = 0.32;
        key.cast_shadows = false;
        let l = &mut ctx.session.effects_mut().lighting;
        l.lights = vec![key];
        l.hemisphere_intensity = 0.3;
        l.sky_colour = [0.7, 0.78, 0.9];
        l.ground_colour = [0.3, 0.3, 0.3];

        ctx.session.viewport_frame_mut().show_grid = true;
        ctx.session.camera_mut().distance = 22.0;
        ctx.session.camera_mut().orientation = glam::Quat::from_rotation_x(0.55);

        self.rebuild(ctx.session);
        self.shown = Some(self.mode);
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        if self.shown != Some(self.mode) {
            self.rebuild(ctx.session);
            self.shown = Some(self.mode);
        }

        if self.mode == 0 {
            let state = (self.use_probes, self.probe_enabled.clone());
            if self.applied_probes.as_ref() != Some(&state) {
                let set = if self.use_probes {
                    build_probe_set(&self.probe_enabled)
                } else {
                    LightProbeSet::new(vec![])
                };
                ctx.session.renderer_mut().set_light_probes(set);
                let source = if self.use_probes {
                    IndirectLightSource::LightProbe
                } else {
                    IndirectLightSource::GlobalIbl
                };
                for &id in &self.probe_spheres {
                    ctx.session.scene_mut().set_indirect_light(id, source);
                }
                for (i, &id) in self.probe_markers.iter().enumerate() {
                    let on = self.use_probes && self.probe_enabled[i];
                    ctx.session
                        .scene_mut()
                        .set_material(id, marker_material(PROBES[i].0, on));
                }
                self.applied_probes = Some(state);
            }
        } else if self.mode == 1 {
            let state = (self.use_zones, self.zone_enabled.clone());
            if self.applied_zones.as_ref() != Some(&state) {
                let zones = self.zones();
                ctx.session
                    .renderer_mut()
                    .set_environment_zones(ctx.queue, &zones);
                for (i, &id) in self.zone_markers.iter().enumerate() {
                    let on = self.use_zones && self.zone_enabled[i];
                    ctx.session
                        .scene_mut()
                        .set_material(id, marker_material(ENVS[i].sky, on));
                }
                self.applied_zones = Some(state);
            }
        } else if self.mode == 2 {
            if self.applied_lightmap != Some(self.lightmap_sel) {
                // Re-register the panel's lightmap for the selected blend mode
                // (Replace/Add sample the radiance texture, AO the occlusion one).
                let panel = self.lightmap_panel.unwrap();
                let res = ctx.session.resources_mut();
                match self.lightmap_sel {
                    0 => {
                        let _ = res.clear_lightmap(panel);
                    }
                    1 | 2 => {
                        let mode = if self.lightmap_sel == 1 {
                            LightmapMode::Replace
                        } else {
                            LightmapMode::Add
                        };
                        let _ = res.set_lightmap(
                            ctx.device,
                            panel,
                            &self.lightmap_uv1,
                            LightmapData::NonDirectional {
                                radiance: self.radiance_tex.unwrap(),
                            },
                            mode,
                        );
                    }
                    _ => {
                        let _ = res.set_lightmap(
                            ctx.device,
                            panel,
                            &self.lightmap_uv1,
                            LightmapData::AmbientOcclusion {
                                occlusion: self.ao_tex.unwrap(),
                            },
                            LightmapMode::AmbientOcclusion,
                        );
                    }
                }
                self.applied_lightmap = Some(self.lightmap_sel);
            }
        } else {
            // Baked GI: path-trace the floor's atlas the first time in, then
            // apply or clear the lightmap to match the toggle.
            if self.gi_baked_tex.is_none() {
                self.bake_gi(ctx);
            }
            if self.gi_applied != Some(self.gi_on) {
                let panel = self.gi_floor_mesh.unwrap();
                let res = ctx.session.resources_mut();
                if self.gi_on {
                    let _ = res.set_lightmap(
                        ctx.device,
                        panel,
                        &self.gi_floor_uv1,
                        LightmapData::NonDirectional {
                            radiance: self.gi_baked_tex.unwrap(),
                        },
                        LightmapMode::Replace,
                    );
                } else {
                    let _ = res.clear_lightmap(panel);
                }
                self.gi_applied = Some(self.gi_on);
            }
        }

        ctx.drive_camera();
    }

    fn description(&self) -> &str {
        match self.mode {
            0 => {
                "Light probes: three coloured SH probes light a row of white spheres \
                 by position, so the row reads as a gradient. Toggle probes off for \
                 the flat ambient fallback."
            }
            1 => {
                "Env zones: three world-space zones each hold their own environment. \
                 Polished metal spheres reflect whichever zone covers them and \
                 cross-fade in the overlaps; clear the zones for the default \
                 environment everywhere."
            }
            2 => {
                "Lightmaps: the panel's coloured light pools and dark corners are \
                 baked into a texture, sampled by a second UV set. No runtime light \
                 makes them. Switch the blend between Off, Replace, Add, and AO."
            }
            _ => {
                "Baked GI: the floor's lighting is path-traced at runtime. Contact \
                 shadows under the blocks, colour bleed from the red and blue walls, \
                 and corner darkening are all baked into its lightmap; toggle it to \
                 see the flat fallback."
            }
        }
    }

    fn top_overlay(&mut self, ui: &mut egui::Ui) {
        if let Some(i) = crate::ui::segmented(
            ui,
            self.mode,
            &["Light probes", "Env zones", "Lightmaps", "Baked GI"],
        ) {
            self.mode = i;
        }
    }

    fn has_controls(&self) -> bool {
        true
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        if self.mode == 3 {
            ui.heading("Baked GI (path traced)");
            ui.add_space(4.0);
            ui.checkbox(&mut self.gi_on, "Show baked lightmap");
            ui.add_space(8.0);
            ui.label(if self.gi_on {
                "On: the floor shows its path-traced lightmap: soft contact shadows, \
                 colour bleed from the walls, and darkened corners."
            } else {
                "Off: the floor falls back to flat ambient. All the baked GI is gone."
            });
            ui.add_space(6.0);
            ui.label(
                "The atlas is baked with the compute path tracer the first time this \
                 tab opens: the floor is rasterised into its lightmap UVs, then a GI \
                 hemisphere is shot from every texel. Walls and blocks are lit flat \
                 for context; only the floor is baked.",
            );
            return;
        }
        if self.mode == 2 {
            ui.heading("Lightmaps");
            ui.add_space(4.0);
            ui.label("Blend mode:");
            ui.add_space(2.0);
            if let Some(i) = crate::ui::segmented(ui, self.lightmap_sel, &LIGHTMAP_MODES) {
                self.lightmap_sel = i;
            }
            ui.add_space(8.0);
            ui.label(match self.lightmap_sel {
                0 => "Off: the panel falls back to plain ambient. The baked light is gone.",
                1 => "Replace: the baked radiance stands in for the panel's indirect diffuse.",
                2 => "Add: the baked radiance is added on top of the ambient term.",
                _ => "AmbientOcclusion: the baked map's red channel darkens the ambient term.",
            });
            ui.add_space(6.0);
            ui.label(
                "The lightmap texture is sampled by a per-vertex UV1 set. Static \
                 surfaces carry precomputed indirect light no runtime light \
                 reproduces.",
            );
            return;
        }
        if self.mode == 0 {
            ui.heading("Light probes");
            ui.add_space(4.0);
            ui.checkbox(&mut self.use_probes, "Sample light probes");
            ui.add_space(8.0);
            ui.label("Active probes:");
            ui.add_space(2.0);
            let names = ["Warm", "Green", "Cool"];
            for i in 0..PROBES.len() {
                swatch_checkbox(ui, PROBES[i].0, names[i], &mut self.probe_enabled[i]);
            }
            ui.add_space(6.0);
            ui.label(
                "Each sphere takes indirect diffuse from the blended SH of the \
                 active probes at its position.",
            );
        } else {
            ui.heading("Environment zones");
            ui.add_space(4.0);
            ui.checkbox(&mut self.use_zones, "Select environment by zone");
            ui.add_space(8.0);
            ui.add_enabled_ui(self.use_zones, |ui| {
                ui.label("Active zones:");
                ui.add_space(2.0);
                for (i, e) in ENVS.iter().enumerate() {
                    swatch_checkbox(ui, e.sky, e.name, &mut self.zone_enabled[i]);
                }
            });
            ui.add_space(6.0);
            ui.label(
                "Fragments inside a zone reflect that environment; overlaps blend \
                 by influence, and everywhere else uses the default environment.",
            );
        }
    }
}

/// A colour swatch followed by a labelled checkbox.
fn swatch_checkbox(ui: &mut egui::Ui, colour: [f32; 3], label: &str, on: &mut bool) {
    ui.horizontal(|ui| {
        let swatch = egui::Color32::from_rgb(
            (colour[0] * 255.0) as u8,
            (colour[1] * 255.0) as u8,
            (colour[2] * 255.0) as u8,
        );
        let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, swatch);
        ui.checkbox(on, label);
    });
}
