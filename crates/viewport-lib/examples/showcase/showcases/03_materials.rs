//! Materials & shading (Set A): a grid of spheres, one row per family, showing
//! the "what kind of material" axes side by side. Rows: shading models, custom
//! WGSL plugins, textures, a PBR metallic/roughness sweep, and matcaps.
//!
//! Also a stress-test of `ViewportInstance`: materials, lights, texture/matcap
//! uploads, and `MaterialPlugin` registration all route through `scene_mut()`,
//! `resources_mut()`, and `effects_mut()`.

use std::f32::consts::TAU;
use viewport_lib as vpl;

use eframe::egui;
use glam::{Mat4, Vec3};
use vpl::{
    AlphaMode, BackfacePattern, BackfacePolicy, BuiltinMatcap, ClipObject, ItemSettings, LightKind,
    LightSource, MatcapId, Material, MeshData, MeshId, NodeId, ParamVis, ParamVisMode,
    PatternConfig, ShadingModel, TextureId, ViewportInstance, primitives,
};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

#[allow(dead_code)]
#[path = "../../plugins/surface_detail_plugin.rs"]
mod surface_detail;
#[allow(dead_code)]
#[path = "../../plugins/toon_plugin.rs"]
mod toon_plugin;

use surface_detail::{DetailLayerPlugin, DissolvePlugin, ParallaxPlugin};
use toon_plugin::{RimPlugin, ToonPlugin};

/// Ball radius and grid spacing.
const BALL: f32 = 0.65;
const SPACING: f32 = 1.8;

/// One grid cell: a mesh, its material, and per-node appearance (for opacity).
#[derive(Clone, Copy)]
struct Spec {
    mesh: MeshId,
    material: Material,
    settings: ItemSettings,
}

impl Spec {
    fn new(mesh: MeshId, material: Material) -> Self {
        Self {
            mesh,
            material,
            settings: ItemSettings::default(),
        }
    }
    fn with_opacity(mut self, opacity: f32) -> Self {
        self.settings.opacity = opacity;
        self
    }
}

pub struct MaterialsShowcase {
    /// `sets[0]` = Set A (kinds), `sets[1]` = Set B (surface); each is rows of
    /// specimens. Built once in setup; the chip swaps which set is in the scene.
    sets: Vec<Vec<Vec<Spec>>>,
    active: usize,
    shown: Option<usize>,
    nodes: Vec<NodeId>,
}

impl MaterialsShowcase {
    pub fn new() -> Self {
        Self {
            sets: Vec::new(),
            active: 0,
            shown: None,
            nodes: Vec::new(),
        }
    }

    /// Clear the current grid and lay out the active set, one row per family.
    fn rebuild(&mut self, scene: &mut vpl::Scene) {
        if !self.nodes.is_empty() {
            let ids = std::mem::take(&mut self.nodes);
            scene.remove_many(&ids);
        }
        let rows = &self.sets[self.active];
        let n_rows = rows.len();
        for (row, specs) in rows.iter().enumerate() {
            let y = ((n_rows - 1) as f32 * 0.5 - row as f32) * SPACING;
            let n = specs.len();
            for (i, spec) in specs.iter().enumerate() {
                let x = (i as f32 - (n - 1) as f32 * 0.5) * SPACING;
                let id = scene.add(
                    Some(spec.mesh),
                    Mat4::from_translation(Vec3::new(x, y, BALL))
                        * Mat4::from_scale(Vec3::splat(BALL)),
                    spec.material,
                );
                scene.set_appearance(id, spec.settings);
                self.nodes.push(id);
            }
        }
    }

    /// In Set B, clip the backface row (the last, front-most row) so its
    /// camera-facing half is cut away and the interior back faces are visible.
    /// The plane sits at that row's equator; other rows are further back in Y
    /// and stay inside the preserved half-space.
    fn apply_clip(&self, session: &mut ViewportInstance) {
        let effects = session.effects_mut();
        if self.active == 1 {
            let rows = self.sets[self.active].len() as f32;
            // y of the last row (see rebuild): -((rows - 1) / 2) * SPACING.
            let row_y = -((rows - 1.0) * 0.5) * SPACING;
            let plane = ClipObject::plane([0.0, 1.0, 0.0], -row_y);
            effects.clip.objects = vec![plane];
            effects.clip.cap_fill_enabled = false;
        } else {
            effects.clip.objects.clear();
        }
    }
}

impl Showcase for MaterialsShowcase {
    fn name(&self) -> &str {
        "Materials & shading"
    }

    fn setup(&mut self, ctx: &mut SetupCtx) {
        let sphere_data = primitives::sphere(1.0, 48, 24);
        let sphere = ctx
            .session
            .resources_mut()
            .upload_mesh_data(ctx.device, &sphere_data)
            .unwrap();

        // Light rig: warm key, cool fill, hemisphere ambient. LightSource is
        // non-exhaustive, so build via Default and mutate.
        // LDR render (no tonemapping), so keep intensities near 1.0 or the
        // materials clip to white.
        let mut key = LightSource::default();
        key.kind = LightKind::Directional {
            direction: [0.5, 0.35, 1.2],
        };
        key.colour = [1.0, 0.97, 0.92];
        key.intensity = 1.4;
        let mut fill = LightSource::default();
        fill.kind = LightKind::Directional {
            direction: [-0.6, -0.4, 0.5],
        };
        fill.colour = [0.55, 0.65, 0.9];
        fill.intensity = 0.4;
        fill.cast_shadows = false;
        let l = &mut ctx.session.effects_mut().lighting;
        l.lights = vec![key, fill];
        l.hemisphere_intensity = 0.25;
        l.sky_colour = [0.9, 0.95, 1.0];
        l.ground_colour = [0.24, 0.22, 0.20];
        l.shadows.enabled = true;

        // Set A (kinds): shading models, custom WGSL, textures, PBR, matcaps.
        let textures = build_textures(ctx);
        let matcaps = build_matcaps(ctx);
        let custom = build_custom_materials(ctx);
        let set_a = vec![
            to_specs(sphere, shading_family(&matcaps)),
            to_specs(sphere, custom),
            to_specs(sphere, texture_family(&textures)),
            to_specs(sphere, pbr_family()),
            to_specs(sphere, matcap_family(&matcaps)),
        ];

        // Set B (surface): UV param, surface maps, transparency, vertex
        // colours, backface policy.
        let maps = build_surface_maps(ctx);
        let alpha_tex = upload(ctx, alpha_checker_texture());
        let vc_meshes = build_vertex_colour_meshes(ctx, &sphere_data);
        let set_b = vec![
            to_specs(sphere, uv_param_family()),
            to_specs(sphere, surface_maps_family(&maps)),
            transparency_family(sphere, alpha_tex),
            vertex_colour_family(&vc_meshes),
            to_specs(sphere, backface_family()),
        ];

        self.sets = vec![set_a, set_b];
        self.rebuild(ctx.session.scene_mut());
        self.apply_clip(ctx.session);
        self.shown = Some(self.active);

        // No grid: material balls read best against the plain background.
        // Elevated three-quarter view (identity orientation is straight down).
        let cam = ctx.session.camera_mut();
        cam.distance = 24.0;
        cam.orientation = glam::Quat::from_rotation_x(0.85);
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        if self.shown != Some(self.active) {
            self.rebuild(ctx.session.scene_mut());
            self.apply_clip(ctx.session);
            self.shown = Some(self.active);
        }
        ctx.drive_camera();
    }

    fn description(&self) -> &str {
        if self.active == 0 {
            "Material kinds, one family per row (back to front):\n\
             - Shading models\n\
             - Custom WGSL plugins\n\
             - Textures\n\
             - PBR metallic / roughness sweep\n\
             - Matcaps"
        } else {
            "Surface behaviour, one family per row (back to front):\n\
             - UV parameterization\n\
             - Surface maps (normal / AO / ORM / emissive)\n\
             - Transparency\n\
             - Vertex colours\n\
             - Backface policy"
        }
    }

    fn top_overlay(&mut self, ui: &mut egui::Ui) {
        if let Some(i) = crate::ui::segmented(ui, self.active, &["Kinds", "Surface"]) {
            self.active = i;
        }
    }
}

/// Wrap a family's materials as specimens on a single mesh.
fn to_specs(mesh: MeshId, mats: Vec<Material>) -> Vec<Spec> {
    mats.into_iter().map(|m| Spec::new(mesh, m)).collect()
}

// ---------------------------------------------------------------------------
// Resource builders (run once in setup)
// ---------------------------------------------------------------------------

fn build_textures(ctx: &mut SetupCtx) -> Vec<TextureId> {
    let mut out = Vec::new();
    for (w, h, rgba) in [
        checker_texture(),
        wheel_texture(),
        gradient_texture(),
        stripes_texture(),
    ] {
        out.push(
            ctx.session
                .resources_mut()
                .upload_texture(ctx.device, ctx.queue, w, h, &rgba)
                .unwrap(),
        );
    }
    out
}

fn build_matcaps(ctx: &mut SetupCtx) -> Vec<(MatcapId, &'static str)> {
    let res = ctx.session.resources_mut();
    res.ensure_matcaps_initialized(ctx.device, ctx.queue);
    let mut out: Vec<(MatcapId, &'static str)> = Vec::new();
    for (bm, name) in [
        (BuiltinMatcap::Clay, "Clay"),
        (BuiltinMatcap::Wax, "Wax"),
        (BuiltinMatcap::Candy, "Candy"),
        (BuiltinMatcap::Ceramic, "Ceramic"),
        (BuiltinMatcap::Jade, "Jade"),
        (BuiltinMatcap::Mud, "Mud"),
        (BuiltinMatcap::Normal, "Normal"),
        (BuiltinMatcap::Flat, "Flat"),
    ] {
        out.push((res.builtin_matcap_id(bm), name));
    }
    let (rgba, _) = matcap_texture();
    let custom = res
        .upload_matcap(ctx.device, ctx.queue, &rgba, false)
        .unwrap();
    out.push((custom, "Custom"));
    out
}

/// The custom-WGSL row. Registers each `MaterialPlugin`, uploads the textures
/// the detail and parallax plugins sample, and returns one material per variant.
fn build_custom_materials(ctx: &mut SetupCtx) -> Vec<Material> {
    // Plugin-owned textures (uploaded before the long-lived resources borrow).
    let detail_tex = upload(ctx, detail_texture());
    let height_tex = upload(ctx, brick_height_texture());
    let brick_tex = upload(ctx, brick_albedo_texture());

    let device = ctx.device;
    let res = ctx.session.resources_mut();
    let toon = res.register_material_plugin(device, &ToonPlugin).unwrap();
    let rim = res.register_material_plugin(device, &RimPlugin).unwrap();
    let detail = res
        .register_material_plugin(device, &DetailLayerPlugin)
        .unwrap();
    let dissolve = res
        .register_material_plugin(device, &DissolvePlugin)
        .unwrap();
    let parallax = res
        .register_material_plugin(device, &ParallaxPlugin)
        .unwrap();

    let toon_a = res
        .create_material_plugin_variant(
            device,
            toon,
            &toon_plugin::toon_params(4.0, 0.2, 1.0, [0.9, 0.5, 0.3]),
            &[],
        )
        .unwrap();
    let toon_b = res
        .create_material_plugin_variant(
            device,
            toon,
            &toon_plugin::toon_params(3.0, 0.25, 1.0, [0.3, 0.6, 0.95]),
            &[],
        )
        .unwrap();
    // attr_mask = 0 applies the detail everywhere (no per-vertex mask needed);
    // the bound texture is the detail albedo the shader multiplies in.
    let detail_v = res
        .create_material_plugin_variant(
            device,
            detail,
            &surface_detail::detail_params(6.0, 0.7, 0.0),
            &[detail_tex],
        )
        .unwrap();
    let dissolve_v = res
        .create_material_plugin_variant(
            device,
            dissolve,
            &surface_detail::dissolve_params(0.5, 0.08, 6.0, [2.0, 1.0, 0.3]),
            &[],
        )
        .unwrap();
    // Parallax owns its textures: slot 0 = height (R), slot 1 = albedo.
    let parallax_v = res
        .create_material_plugin_variant(
            device,
            parallax,
            &surface_detail::parallax_params(0.06, 3.0),
            &[height_tex, brick_tex],
        )
        .unwrap();

    // Plugin hooks add on top of the built-in PBR direct + ambient terms, so
    // the base is a PBR material (matching the eframe_showcase reference).
    let plugin_mat = |id, colour: [f32; 3]| {
        let mut m = Material::pbr(colour, 0.1, 0.55);
        m.shading_plugin = Some(id);
        m
    };
    // Dissolve needs a Mask material to actually cut fragments (opaque only glows).
    let mut dissolve_mat = plugin_mat(dissolve_v, [0.55, 0.35, 0.25]);
    dissolve_mat.alpha_mode = AlphaMode::Mask(0.5);

    vec![
        plugin_mat(toon_a, [0.75, 0.3, 0.3]),
        plugin_mat(toon_b, [0.75, 0.3, 0.3]),
        // Dark base so the blue rim glow reads (rim colour is the plugin's
        // default params: [0.2, 0.5, 1.0], power 3).
        plugin_mat(rim, [0.25, 0.25, 0.3]),
        dissolve_mat,
        plugin_mat(detail_v, [0.4, 0.5, 0.35]),
        plugin_mat(parallax_v, [0.8, 0.8, 0.8]),
    ]
}

/// Upload a `(w, h, rgba)` texture through the session.
fn upload(ctx: &mut SetupCtx, tex: (u32, u32, Vec<u8>)) -> TextureId {
    let (w, h, rgba) = tex;
    ctx.session
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, w, h, &rgba)
        .unwrap()
}

// ---------------------------------------------------------------------------
// Family material lists
// ---------------------------------------------------------------------------

fn shading_family(matcaps: &[(MatcapId, &'static str)]) -> Vec<Material> {
    let jade = matcaps
        .iter()
        .find(|(_, n)| *n == "Jade")
        .map(|(id, _)| *id)
        .unwrap();
    let mut matcap = Material::from_colour([0.8, 0.8, 0.8]);
    matcap.shading_model = ShadingModel::Matcap(jade);
    vec![
        Material::from_colour([0.62, 0.14, 0.12]), // Phong (default model)
        Material::pbr([0.25, 0.5, 0.85], 0.0, 0.35),
        Material::pbr([0.72, 0.48, 0.06], 1.0, 0.25),
        matcap,
        Material::flat([0.5, 0.78, 0.45]),
    ]
}

fn texture_family(tex: &[TextureId]) -> Vec<Material> {
    let checker = tex[0];
    let base = |t| {
        let mut m = Material::pbr([1.0, 1.0, 1.0], 0.0, 0.55);
        m.texture_id = Some(t);
        m
    };
    let mut tiled = base(checker);
    tiled.uv_scale = [3.0, 3.0];
    let mut offset = base(checker);
    offset.uv_offset = [0.5, 0.0];
    vec![
        base(checker),
        base(tex[1]),
        base(tex[2]),
        base(tex[3]),
        tiled,
        offset,
    ]
}

fn pbr_family() -> Vec<Material> {
    // Gold-tinted so the metallic row reads as gold, not silver.
    let base = [0.95, 0.73, 0.34];
    let mut out = Vec::new();
    for &metallic in &[0.0, 1.0] {
        for &roughness in &[0.1, 0.37, 0.63, 0.9] {
            out.push(Material::pbr(base, metallic, roughness));
        }
    }
    out
}

fn matcap_family(matcaps: &[(MatcapId, &'static str)]) -> Vec<Material> {
    let tints = [
        [0.9, 0.5, 0.4],
        [0.5, 0.9, 0.6],
        [0.5, 0.6, 0.95],
        [0.9, 0.85, 0.4],
        [0.8, 0.8, 0.85],
        [0.85, 0.5, 0.85],
        [0.6, 0.85, 0.9],
        [0.95, 0.7, 0.5],
        [0.7, 0.9, 0.7],
        [0.9, 0.6, 0.9],
    ];
    matcaps
        .iter()
        .enumerate()
        .map(|(i, (id, _))| {
            let mut m = Material::from_colour(tints[i % tints.len()]);
            m.shading_model = ShadingModel::Matcap(*id);
            m
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Set B families
// ---------------------------------------------------------------------------

fn uv_param_family() -> Vec<Material> {
    [
        ParamVisMode::Checker,
        ParamVisMode::Grid,
        ParamVisMode::LocalChecker,
        ParamVisMode::LocalRadial,
    ]
    .into_iter()
    .map(|mode| {
        let mut m = Material::pbr([0.7, 0.7, 0.75], 0.0, 0.5);
        m.param_vis = Some(ParamVis { mode, scale: 8.0 });
        m
    })
    .collect()
}

struct SurfaceMaps {
    normal: TextureId,
    ao: TextureId,
    orm: TextureId,
}

fn build_surface_maps(ctx: &mut SetupCtx) -> SurfaceMaps {
    let (w, h, rgba) = bump_normal_texture();
    let normal = ctx
        .session
        .resources_mut()
        .upload_normal_map(ctx.device, ctx.queue, w, h, &rgba)
        .unwrap();
    let ao = upload(ctx, bump_ao_texture());
    let orm = upload(ctx, orm_texture());
    SurfaceMaps { normal, ao, orm }
}

fn surface_maps_family(m: &SurfaceMaps) -> Vec<Material> {
    let pbr = || Material::pbr([0.75, 0.72, 0.68], 0.0, 0.5);
    let plain = pbr();
    let mut normal = pbr();
    normal.normal_map_id = Some(m.normal);
    let mut normal_ao = normal;
    normal_ao.ao_map_id = Some(m.ao);
    let mut ao = pbr();
    ao.ao_map_id = Some(m.ao);
    let mut orm = pbr();
    orm.metallic_roughness_texture_id = Some(m.orm);
    orm.metallic = 1.0;
    let mut emissive = pbr();
    emissive.emissive = [1.2, 0.5, 0.15];
    vec![plain, normal, normal_ao, ao, orm, emissive]
}

fn transparency_family(mesh: MeshId, alpha_tex: TextureId) -> Vec<Spec> {
    let blend = |o: f32| {
        let mut m = Material::pbr([0.4, 0.6, 0.9], 0.0, 0.35);
        m.alpha_mode = AlphaMode::Blend;
        Spec::new(mesh, m).with_opacity(o)
    };
    let mut mask = Material::pbr([1.0, 1.0, 1.0], 0.0, 0.5);
    mask.texture_id = Some(alpha_tex);
    mask.alpha_mode = AlphaMode::Mask(0.5);
    vec![
        blend(0.25),
        blend(0.45),
        blend(0.65),
        blend(0.85),
        Spec::new(mesh, mask),
    ]
}

fn build_vertex_colour_meshes(ctx: &mut SetupCtx, base: &MeshData) -> Vec<MeshId> {
    let colourings: [fn([f32; 3]) -> [f32; 4]; 4] =
        [vc_rainbow, vc_gradient, vc_two_tone, vc_latitude];
    colourings
        .iter()
        .map(|f| {
            let mut mesh = base.clone();
            mesh.vertex_colours = Some(base.positions.iter().map(|p| f(*p)).collect());
            ctx.session
                .resources_mut()
                .upload_mesh_data(ctx.device, &mesh)
                .unwrap()
        })
        .collect()
}

fn vertex_colour_family(meshes: &[MeshId]) -> Vec<Spec> {
    // White base so the baked per-vertex colour shows through PBR shading.
    meshes
        .iter()
        .map(|&mesh| Spec::new(mesh, Material::pbr([1.0, 1.0, 1.0], 0.0, 0.5)))
        .collect()
}

fn vc_rainbow(p: [f32; 3]) -> [f32; 4] {
    let hue = (p[1].atan2(p[0]) / TAU + 0.5).fract();
    let c = hsv_to_rgb(hue, 0.85, 1.0);
    [c[0], c[1], c[2], 1.0]
}
fn vc_gradient(p: [f32; 3]) -> [f32; 4] {
    let t = (p[2] * 0.5 + 0.5).clamp(0.0, 1.0);
    [t, 0.25, 1.0 - t, 1.0]
}
fn vc_two_tone(p: [f32; 3]) -> [f32; 4] {
    if p[2] >= 0.0 {
        [0.95, 0.75, 0.2, 1.0]
    } else {
        [0.2, 0.5, 0.85, 1.0]
    }
}
fn vc_latitude(p: [f32; 3]) -> [f32; 4] {
    let t = p[2].abs().clamp(0.0, 1.0);
    let c = hsv_to_rgb(0.33 * (1.0 - t), 0.7, 1.0);
    [c[0], c[1], c[2], 1.0]
}

fn backface_family() -> Vec<Material> {
    let base = |policy: BackfacePolicy| {
        let mut m = Material::pbr([0.72, 0.42, 0.30], 0.0, 0.5);
        m.backface_policy = policy;
        m
    };
    let pat = |pattern: BackfacePattern| {
        let mut cfg = PatternConfig::default();
        cfg.pattern = pattern;
        base(BackfacePolicy::Pattern(cfg))
    };
    vec![
        base(BackfacePolicy::Cull),
        base(BackfacePolicy::Identical),
        base(BackfacePolicy::DifferentColour([0.2, 0.6, 0.9])),
        base(BackfacePolicy::Tint(0.4)),
        pat(BackfacePattern::Checker),
        pat(BackfacePattern::Hatching),
        pat(BackfacePattern::Crosshatch),
        pat(BackfacePattern::Stripes),
    ]
}

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------

fn checker_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let cells = 8u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let on = ((x * cells / size) + (y * cells / size)) % 2 == 0;
            let v = if on { 230 } else { 40 };
            px.extend_from_slice(&[v, v, v, 255]);
        }
    }
    (size, size, px)
}

fn stripes_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for _y in 0..size {
        for x in 0..size {
            let on = (x / 12) % 2 == 0;
            if on {
                px.extend_from_slice(&[230, 120, 40, 255]);
            } else {
                px.extend_from_slice(&[30, 30, 40, 255]);
            }
        }
    }
    (size, size, px)
}

fn gradient_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for _y in 0..size {
        for x in 0..size {
            let h = x as f32 / (size - 1) as f32;
            let [r, g, b] = hsv_to_rgb(h, 0.7, 1.0);
            px.extend_from_slice(&[(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]);
        }
    }
    (size, size, px)
}

fn wheel_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let nx = (x as f32 / (size - 1) as f32) * 2.0 - 1.0;
            let ny = (y as f32 / (size - 1) as f32) * 2.0 - 1.0;
            let hue = (ny.atan2(nx) / TAU + 0.5).fract();
            let sat = (nx * nx + ny * ny).sqrt().min(1.0);
            let [r, g, b] = hsv_to_rgb(hue, sat, 1.0);
            px.extend_from_slice(&[(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]);
        }
    }
    (size, size, px)
}

/// Fine grey checker used as the detail-layer albedo (high-frequency so it
/// reads as surface detail when tiled).
fn detail_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let cells = 16u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let on = ((x * cells / size) + (y * cells / size)) % 2 == 0;
            let v = if on { 200 } else { 90 };
            px.extend_from_slice(&[v, v, v, 255]);
        }
    }
    (size, size, px)
}

/// A brick height field (R channel): bricks raised, mortar recessed. Slot 0 of
/// the parallax plugin.
fn brick_height_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let (r, _, _) = brick_pattern(size);
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for &h in &r {
        let v = (h * 255.0) as u8;
        px.extend_from_slice(&[v, v, v, 255]);
    }
    (size, size, px)
}

/// A brick albedo (red bricks, grey mortar). Slot 1 of the parallax plugin.
fn brick_albedo_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let (r, _, _) = brick_pattern(size);
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for &h in &r {
        let c = if h > 0.5 {
            [0.62, 0.24, 0.18]
        } else {
            [0.55, 0.53, 0.5]
        };
        px.extend_from_slice(&[
            (c[0] * 255.0) as u8,
            (c[1] * 255.0) as u8,
            (c[2] * 255.0) as u8,
            255,
        ]);
    }
    (size, size, px)
}

/// Shared brick mask: 1.0 on a brick, 0.0 in the mortar lines, with a running
/// half-brick offset every other row. Returns the mask flattened row-major.
fn brick_pattern(size: u32) -> (Vec<f32>, u32, u32) {
    let mut mask = Vec::with_capacity((size * size) as usize);
    let brick_h = size / 4;
    let brick_w = size / 2;
    let mortar = size / 32;
    for y in 0..size {
        let row = y / brick_h;
        let offset = if row % 2 == 0 { 0 } else { brick_w / 2 };
        for x in 0..size {
            let ly = y % brick_h;
            let lx = (x + offset) % brick_w;
            let on = ly > mortar && lx > mortar;
            mask.push(if on { 1.0 } else { 0.0 });
        }
    }
    (mask, size, size)
}

/// Egg-carton normal map (tangent space) for the surface-maps row.
fn bump_normal_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let h = |x: f32, y: f32| (x * TAU * 4.0).sin() * (y * TAU * 4.0).sin() * 0.5 + 0.5;
    let e = 1.0 / size as f32;
    let strength = 2.0;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let fx = x as f32 / size as f32;
            let fy = y as f32 / size as f32;
            let dx = (h(fx + e, fy) - h(fx - e, fy)) * strength;
            let dy = (h(fx, fy + e) - h(fx, fy - e)) * strength;
            let n = Vec3::new(-dx, -dy, 1.0).normalize();
            px.extend_from_slice(&[
                ((n.x * 0.5 + 0.5) * 255.0) as u8,
                ((n.y * 0.5 + 0.5) * 255.0) as u8,
                ((n.z * 0.5 + 0.5) * 255.0) as u8,
                255,
            ]);
        }
    }
    (size, size, px)
}

/// AO map matching the egg-carton bumps (valleys darker).
fn bump_ao_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let fx = x as f32 / size as f32;
            let fy = y as f32 / size as f32;
            let h = (fx * TAU * 4.0).sin() * (fy * TAU * 4.0).sin() * 0.5 + 0.5;
            let ao = (0.55 + 0.45 * h).clamp(0.0, 1.0);
            let v = (ao * 255.0) as u8;
            px.extend_from_slice(&[v, v, v, 255]);
        }
    }
    (size, size, px)
}

/// Packed ORM texture: G = roughness ramp, B = metallic (constant metal).
fn orm_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for _y in 0..size {
        for x in 0..size {
            let rough = (0.1 + 0.85 * x as f32 / (size - 1) as f32).clamp(0.0, 1.0);
            px.extend_from_slice(&[255, (rough * 255.0) as u8, 230, 255]);
        }
    }
    (size, size, px)
}

/// Checker with transparent cells, for the alpha-mask cutout specimen.
fn alpha_checker_texture() -> (u32, u32, Vec<u8>) {
    let size = 128u32;
    let cells = 6u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let on = ((x * cells / size) + (y * cells / size)) % 2 == 0;
            let a = if on { 255 } else { 0 };
            px.extend_from_slice(&[210, 210, 220, a]);
        }
    }
    (size, size, px)
}

/// A 256x256 procedural matcap: a shaded ball, diffuse + specular over an HSV
/// tint, with a dark edge.
fn matcap_texture() -> (Vec<u8>, (u32, u32)) {
    let size = 256u32;
    let mut px = vec![0u8; (size * size * 4) as usize];
    let light = Vec3::new(0.4, 0.5, 0.75).normalize();
    for y in 0..size {
        for x in 0..size {
            let nx = (x as f32 / (size - 1) as f32) * 2.0 - 1.0;
            let ny = -((y as f32 / (size - 1) as f32) * 2.0 - 1.0);
            let r2 = nx * nx + ny * ny;
            let i = ((y * size + x) * 4) as usize;
            if r2 > 1.0 {
                px[i..i + 4].copy_from_slice(&[18, 18, 24, 255]);
                continue;
            }
            let nz = (1.0 - r2).sqrt();
            let n = Vec3::new(nx, ny, nz);
            let diff = n.dot(light).max(0.0);
            let spec = n.dot(light).max(0.0).powf(32.0);
            let base = hsv_to_rgb(0.58, 0.5, 1.0);
            let c = [
                (base[0] * (0.2 + 0.8 * diff) + spec).min(1.0),
                (base[1] * (0.2 + 0.8 * diff) + spec).min(1.0),
                (base[2] * (0.2 + 0.8 * diff) + spec).min(1.0),
            ];
            px[i..i + 4].copy_from_slice(&[
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
                255,
            ]);
        }
    }
    (px, (size, size))
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor() as u32 % 6;
    let f = h * 6.0 - (h * 6.0).floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}
