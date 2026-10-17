//! Material scenes that draw each material on both mesh paths.
//!
//! Every material appears twice: one copy carries a per-material sampler,
//! which keeps it on the per-object path, and the other is batched onto the
//! instanced path. The two should look the same, so a parity break shows up as
//! a difference inside one reference image as well as against the reference.

use glam::{Mat4, Vec3};
use viewport_lib::{
    AlphaMode, BackfacePolicy, Colour, LightKind, LightSource, LightingSettings, Material,
    MeshData, Projection, SamplerKey, SceneRenderItem, ShadingModel, TextureSlot, UvTransform,
    primitives,
};

use super::{
    BuildCtx, BuiltScene, NamedCamera, NamedScene, ground, item_model, orbit_camera, rigs,
    textures, upload,
};

/// Distance from a pair's centre to each copy: per-object on the left,
/// instanced on the right.
const HALF_PAIR: f32 = 0.85;

/// Spacing between one material's pair and the next.
const PAIR_SPACING: f32 = 3.4;

/// The centre of the `i`th of four pairs.
fn pair_x(i: usize) -> f32 {
    (i as f32 - 1.5) * PAIR_SPACING
}

/// Keep an item on the per-object path. A per-material sampler does that
/// without changing how the default sampler would draw it.
fn per_object(mut material: Material) -> Material {
    material = material.with_sampler(TextureSlot::Albedo, SamplerKey::default());
    material
}

/// Push one copy of `material` per path, at `x`, with `model` placing the mesh
/// relative to that spot.
fn both_paths(
    items: &mut Vec<SceneRenderItem>,
    mesh: viewport_lib::MeshId,
    x: f32,
    model: Mat4,
    material: Material,
    opacity: f32,
) {
    for (dx, mat) in [
        (-HALF_PAIR, per_object(material.clone())),
        (HALF_PAIR, material),
    ] {
        let mut it = item_model(
            mesh,
            Mat4::from_translation(Vec3::new(x + dx, 0.0, 0.0)) * model,
            mat,
        );
        it.settings.opacity = opacity;
        items.push(it);
    }
}

/// An upright card in the XZ plane, facing -Y, standing on z = 0. Vertex
/// alpha ramps from 0 on the left edge to 1 on the right.
fn card(half_width: f32, height: f32) -> MeshData {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-half_width, 0.0, 0.0],
        [half_width, 0.0, 0.0],
        [half_width, 0.0, height],
        [-half_width, 0.0, height],
    ];
    mesh.normals = vec![[0.0, -1.0, 0.0]; 4];
    mesh.uvs = Some(vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
    mesh.vertex_colours = Some(vec![
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
    ]);
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh
}

fn upload_texture(
    ctx: &mut BuildCtx<'_>,
    data: viewport_lib::TextureData,
) -> viewport_lib::TextureId {
    ctx.renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, data)
        .expect("texture upload")
}

/// A checker whose dark cells are fully transparent: a cut-out pattern.
fn holes(size: u32, cells: u32) -> viewport_lib::TextureData {
    let t = textures::checker(size, cells, [235, 210, 120], [0, 0, 0]);
    let mut rgba = t.rgba;
    for px in rgba.chunks_exact_mut(4) {
        if px[0] == 0 {
            px[3] = 0;
        }
    }
    viewport_lib::TextureData::srgb(t.width, t.height, rgba)
}

/// Alpha handling on both paths: a texture mask, a vertex-colour mask, a
/// masked card faded below full opacity, and a premultiplied pane, all casting
/// onto a ground slab under an angled sun so the shadow cutouts show too.
// Not in the catalogue: a per-object caster loses its shadow when a textured
// masked per-object caster shares the frame, so its reference would record
// that as correct. Register it (camera: `orbit_camera(Vec3::new(0.0, 1.0,
// 0.4), 12.5, 0.0, 0.95)`) once both copies cast.
#[allow(dead_code)]
fn build_materials_alpha_shadows(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let g = ground(ctx, 0.0);
    let card_mesh = upload(ctx, &card(0.7, 1.6));
    let holes_tex = upload_texture(ctx, holes(256, 6));

    let mut items = vec![g];
    let upright = Mat4::IDENTITY;
    let two_sided = |mut m: Material| {
        m.backface_policy = BackfacePolicy::Identical;
        m
    };

    // Texture-alpha mask.
    let mut tex_mask = two_sided(Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.6));
    tex_mask.texture_id = Some(holes_tex);
    tex_mask.alpha_mode = AlphaMode::Mask(0.5);
    both_paths(
        &mut items,
        card_mesh,
        pair_x(0),
        upright,
        tex_mask.clone(),
        1.0,
    );

    // Vertex-colour mask: the left half of the card is cut away.
    let mut vertex_mask = two_sided(Material::pbr(Colour::linear_rgb(0.3, 0.6, 0.9), 0.0, 0.5));
    vertex_mask.alpha_mode = AlphaMode::Mask(0.5);
    both_paths(&mut items, card_mesh, pair_x(1), upright, vertex_mask, 1.0);

    // The texture mask again below full opacity: drawn through the
    // transparent pass, still cut out, and casting no shadow.
    both_paths(&mut items, card_mesh, pair_x(2), upright, tex_mask, 0.8);

    // A premultiplied pane: its colour already carries the alpha.
    let mut premult = two_sided(Material::pbr(Colour::linear_rgb(0.45, 0.2, 0.05), 0.0, 0.3));
    premult.alpha_mode = AlphaMode::BlendPremultiplied;
    both_paths(&mut items, card_mesh, pair_x(3), upright, premult, 0.5);

    // A sun from the front right, so each card's shadow falls on the ground
    // behind it, in view.
    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [0.5, -0.9, 1.0],
    };
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![sun];
    lighting.hemisphere_intensity = 0.1;

    BuiltScene {
        items,
        lighting,
        background: None,
        ..Default::default()
    }
}

/// Push one item at `pos`, kept on the per-object path when `per_object` is
/// set and left to batch onto the instanced path otherwise.
fn place(
    items: &mut Vec<SceneRenderItem>,
    mesh: viewport_lib::MeshId,
    model: Mat4,
    material: Material,
    opacity: f32,
    per_object_path: bool,
) {
    let material = if per_object_path {
        per_object(material)
    } else {
        material
    };
    let mut it = item_model(mesh, model, material);
    it.settings.opacity = opacity;
    items.push(it);
}

/// Grid placement: column `c` of `columns`, row `r` counted from the top. Rows
/// come in per-object/instanced pairs, with a wider gap between pairs.
fn grid_pos(c: usize, columns: usize, r: usize, col_step: f32, row_step: f32) -> Vec3 {
    let x = (c as f32 - (columns as f32 - 1.0) * 0.5) * col_step;
    let z = -(r as f32 * row_step + (r / 2) as f32 * row_step * 0.3);
    Vec3::new(x, 0.0, z)
}

/// `mesh` with vertex colours taken from its normals, so every face gets a
/// different tint.
fn tinted(mut mesh: MeshData) -> MeshData {
    mesh.vertex_colours = Some(
        mesh.normals
            .iter()
            .map(|n| [0.5 + 0.5 * n[0], 0.5 + 0.5 * n[1], 0.5 + 0.5 * n[2], 1.0])
            .collect(),
    );
    mesh
}

/// Texture slots crossed with shading models. Columns: untextured, albedo,
/// normal with AO, metallic-roughness, emissive, all five slots, and vertex
/// colours. Rows: Phong, PBR and flat, each drawn once per path (per-object
/// above, instanced below). Every slot carries its own `UvTransform`, so a slot
/// that read another slot's UV draws a visibly different pattern.
fn build_materials_textures(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let sphere_data = primitives::sphere(0.42, 24, 12);
    let sphere = upload(ctx, &sphere_data);
    let tinted_sphere = upload(ctx, &tinted(sphere_data));

    // glTF ORM packing: G is roughness, B is metallic. Polished metal cells
    // against rough dielectric ones.
    let mr = textures::checker(256, 4, [0, 30, 255], [0, 230, 0]);
    let mr_tex = upload_texture(
        ctx,
        viewport_lib::TextureData::linear(mr.width, mr.height, mr.rgba),
    );
    let stripes = textures::checker(256, 8, [255, 140, 40], [0, 0, 0]);
    let emissive_tex = upload_texture(
        ctx,
        viewport_lib::TextureData::srgb(stripes.width, stripes.height, stripes.rgba),
    );
    let nm = textures::normal_bumps(256, 6);
    let normal_tex = upload_texture(
        ctx,
        viewport_lib::TextureData::normal_map(nm.width, nm.height, nm.rgba),
    );
    let ao = textures::value_noise(256);
    let ao_tex = upload_texture(
        ctx,
        viewport_lib::TextureData::linear(ao.width, ao.height, ao.rgba),
    );
    let albedo = textures::checker(256, 8, [230, 230, 230], [60, 60, 80]);
    let albedo_tex = upload_texture(
        ctx,
        viewport_lib::TextureData::srgb(albedo.width, albedo.height, albedo.rgba),
    );

    let transform = |scale: f32, rotation: f32| UvTransform {
        scale: [scale, scale],
        rotation,
        ..UvTransform::default()
    };
    let with_albedo = |m: Material| {
        let mut m = m.with_texture(TextureSlot::Albedo, albedo_tex);
        m.uv_rotation = 0.5;
        m
    };
    let with_normal_ao = |m: Material| {
        m.with_texture(TextureSlot::Normal, normal_tex)
            .with_texture(TextureSlot::Ao, ao_tex)
            .with_texture_transform(TextureSlot::Normal, transform(2.0, 0.4))
            .with_texture_transform(TextureSlot::Ao, transform(0.5, 0.0))
    };
    let with_mr = |m: Material| {
        m.with_texture(TextureSlot::MetallicRoughness, mr_tex)
            .with_texture_transform(TextureSlot::MetallicRoughness, transform(2.0, 0.6))
    };
    let with_emissive = |mut m: Material| {
        m.emissive = Colour::linear_rgb(1.0, 1.0, 1.0);
        m.emissive_strength = 1.5;
        m.with_texture(TextureSlot::Emissive, emissive_tex)
            .with_texture_transform(TextureSlot::Emissive, transform(1.5, 0.8))
    };
    let columns: [&dyn Fn(Material) -> Material; 7] = [
        &|m| m,
        &with_albedo,
        &with_normal_ao,
        &with_mr,
        &with_emissive,
        &|m| with_emissive(with_mr(with_normal_ao(with_albedo(m)))),
        &|m| m,
    ];

    let base = Colour::linear_rgb(0.8, 0.78, 0.74);
    let mut phong = Material::from_colour(base);
    phong.shading_model = ShadingModel::Phong;
    let shading = [phong, Material::pbr(base, 1.0, 1.0), Material::flat(base)];

    let mut items = Vec::new();
    for (s, model) in shading.iter().enumerate() {
        for (c, column) in columns.iter().enumerate() {
            let mesh = if c == columns.len() - 1 {
                tinted_sphere
            } else {
                sphere
            };
            for path in 0..2 {
                let pos = grid_pos(c, columns.len(), s * 2 + path, 1.15, 0.95);
                place(
                    &mut items,
                    mesh,
                    Mat4::from_translation(pos),
                    column(model.clone()),
                    1.0,
                    path == 0,
                );
            }
        }
    }

    BuiltScene {
        items,
        lighting: {
            let mut l = rigs::three_point();
            for light in &mut l.lights {
                light.cast_shadows = false;
            }
            l
        },
        background: None,
        ..Default::default()
    }
}

/// Alpha modes crossed with alpha sources, on upright cards in front of a
/// banded wall. Columns: no alpha, opacity 0.6, a vertex-alpha ramp, a
/// texture with transparent cells, the texture and the ramp together, the
/// texture at opacity 0.6, then the back face of the textured ramp card under
/// `Identical` and under `Cull` (which draws nothing). Rows: opaque, mask,
/// blend and premultiplied blend, each drawn once per path (per-object above,
/// instanced below).
fn build_materials_alpha(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    const COL_STEP: f32 = 1.05;
    let ramp_card = upload(ctx, &card(0.42, 0.6));
    let mut solid = card(0.42, 0.6);
    solid.vertex_colours = None;
    let solid_card = upload(ctx, &solid);
    let holes_tex = upload_texture(ctx, holes(256, 4));

    // Two-colour vertical bands, one pair per column, split down the middle of
    // each card: every card has a light and a dark half behind it, and every
    // row of cards has the same backdrop.
    let bands = textures::checker(64, 2, [150, 140, 125], [90, 90, 100]);
    let row = bands.rgba[..64 * 4].to_vec();
    let wall_tex = upload_texture(ctx, viewport_lib::TextureData::srgb(64, 64, row.repeat(64)));
    let wall_half_width = 6.5;
    let wall_mesh = upload(ctx, &card(wall_half_width, 9.6));
    let mut wall = Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0))
        .with_texture(TextureSlot::Albedo, wall_tex);
    // One band pair per column step, with a band edge on each card centre
    // (x = 0.5 * COL_STEP). The wall's u runs 0 -> 1 across its width.
    let repeats = 2.0 * wall_half_width / COL_STEP;
    let phase = (wall_half_width + 0.5 * COL_STEP) / COL_STEP;
    wall.uv_scale = [repeats, 1.0];
    wall.uv_offset = [phase.ceil() - phase, 0.0];
    let mut items = vec![item_model(
        wall_mesh,
        Mat4::from_translation(Vec3::new(0.0, 1.5, -7.6)),
        wall,
    )];

    let modes = [
        AlphaMode::Opaque,
        AlphaMode::Mask(0.5),
        AlphaMode::Blend,
        AlphaMode::BlendPremultiplied,
    ];
    let back = Mat4::from_rotation_z(std::f32::consts::PI);
    // (mesh, textured, opacity, model, backface policy)
    let columns = [
        (
            solid_card,
            false,
            1.0,
            Mat4::IDENTITY,
            BackfacePolicy::Identical,
        ),
        (
            solid_card,
            false,
            0.6,
            Mat4::IDENTITY,
            BackfacePolicy::Identical,
        ),
        (
            ramp_card,
            false,
            1.0,
            Mat4::IDENTITY,
            BackfacePolicy::Identical,
        ),
        (
            solid_card,
            true,
            1.0,
            Mat4::IDENTITY,
            BackfacePolicy::Identical,
        ),
        (
            ramp_card,
            true,
            1.0,
            Mat4::IDENTITY,
            BackfacePolicy::Identical,
        ),
        (
            solid_card,
            true,
            0.6,
            Mat4::IDENTITY,
            BackfacePolicy::Identical,
        ),
        (ramp_card, true, 1.0, back, BackfacePolicy::Identical),
        (ramp_card, true, 1.0, back, BackfacePolicy::Cull),
    ];

    for (m, mode) in modes.iter().enumerate() {
        for (c, (mesh, textured, opacity, turn, policy)) in columns.iter().enumerate() {
            let mut material = Material::pbr(Colour::linear_rgb(0.3, 0.6, 0.9), 0.0, 0.5);
            material.alpha_mode = *mode;
            material.backface_policy = *policy;
            if *textured {
                material = material.with_texture(TextureSlot::Albedo, holes_tex);
            }
            for path in 0..2 {
                let pos = grid_pos(c, columns.len(), m * 2 + path, COL_STEP, 0.72);
                place(
                    &mut items,
                    *mesh,
                    Mat4::from_translation(pos) * *turn,
                    material.clone(),
                    *opacity,
                    path == 0,
                );
            }
        }
    }

    // A light from the camera side; no shadows, so only the surfaces are
    // compared.
    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [0.3, -1.0, 0.6],
    };
    sun.cast_shadows = false;
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![sun];
    lighting.hemisphere_intensity = 0.4;

    BuiltScene {
        items,
        lighting,
        background: None,
        ..Default::default()
    }
}

/// Straight on from -Y, looking at the middle of a grid that hangs below z = 0.
/// Orthographic, so under directional light every cell shades the same
/// wherever it sits and the per-object and instanced rows compare exactly.
fn grid_camera(centre_z: f32, distance: f32) -> Vec<NamedCamera> {
    let mut camera = orbit_camera(Vec3::new(0.0, 0.0, centre_z), distance, 0.0, 1.5708);
    camera.projection = Projection::Orthographic;
    camera.aspect = 4.0 / 3.0;
    vec![NamedCamera {
        name: "front",
        camera,
    }]
}

/// The material scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "materials_textures",
            cameras: grid_camera(-2.66, 8.6),
            build: build_materials_textures,
        },
        NamedScene {
            name: "materials_alpha",
            cameras: grid_camera(-2.55, 10.5),
            build: build_materials_alpha,
        },
    ]
}
