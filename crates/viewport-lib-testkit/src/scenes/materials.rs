//! Material scenes that put each material on both mesh paths side by side.
//!
//! Every material appears twice, side by side: the left copy carries a
//! per-material sampler, which keeps it on the per-object path, and the right
//! copy is batched onto the instanced path. The two should look the same, so a parity break shows up as
//! a difference inside one reference image as well as against the reference.

use glam::{Mat4, Vec3};
use viewport_lib::{
    AlphaMode, BackfacePolicy, Colour, LightKind, LightSource, LightingSettings, Material,
    MeshData, SamplerKey, SceneRenderItem, TextureSlot, UvTransform, primitives,
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
fn build_materials_alpha(ctx: &mut BuildCtx<'_>) -> BuiltScene {
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

/// The texture slots on both paths, each with its own `UvTransform` so a slot
/// that read another slot's UV would draw a visibly different pattern:
/// metallic-roughness, emissive, normal with AO, and a rotated albedo.
fn build_materials_slots(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let sphere = upload(ctx, &primitives::sphere(0.8, 48, 24));

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

    let rotated = |scale: f32, rotation: f32| UvTransform {
        scale: [scale, scale],
        rotation,
        ..UvTransform::default()
    };
    let at = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.8));
    let mut items = Vec::new();

    let mut metal = Material::pbr(Colour::linear_rgb(0.9, 0.85, 0.75), 1.0, 1.0);
    metal.metallic_roughness_texture_id = Some(mr_tex);
    metal = metal.with_texture_transform(TextureSlot::MetallicRoughness, rotated(2.0, 0.6));
    both_paths(&mut items, sphere, pair_x(0), at, metal, 1.0);

    let mut glow = Material::pbr(Colour::linear_rgb(0.1, 0.1, 0.12), 0.0, 0.6);
    glow.emissive = Colour::linear_rgb(1.0, 1.0, 1.0);
    glow.emissive_strength = 4.0;
    glow.emissive_texture_id = Some(emissive_tex);
    glow = glow.with_texture_transform(TextureSlot::Emissive, rotated(1.5, 0.8));
    both_paths(&mut items, sphere, pair_x(1), at, glow, 1.0);

    let mut bumpy = Material::pbr(Colour::linear_rgb(0.7, 0.72, 0.78), 0.1, 0.45);
    bumpy.normal_map_id = Some(normal_tex);
    bumpy.ao_map_id = Some(ao_tex);
    bumpy = bumpy
        .with_texture_transform(TextureSlot::Normal, rotated(2.0, 0.4))
        .with_texture_transform(TextureSlot::Ao, rotated(0.5, 0.0));
    both_paths(&mut items, sphere, pair_x(2), at, bumpy, 1.0);

    let mut checker = Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.6);
    checker.texture_id = Some(albedo_tex);
    checker = checker.with_uv_rotation(0.5);
    both_paths(&mut items, sphere, pair_x(3), at, checker, 1.0);

    BuiltScene {
        items,
        lighting: rigs::three_point(),
        background: None,
        ..Default::default()
    }
}

/// Looking along +Y from above the front, so the four pairs read left to
/// right with the ground behind them in view.
fn front_camera(distance: f32) -> Vec<NamedCamera> {
    vec![NamedCamera {
        name: "front",
        camera: orbit_camera(Vec3::new(0.0, 0.5, 0.8), distance, 0.0, 1.2),
    }]
}

/// The material scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "materials_slots",
        cameras: front_camera(12.5),
        build: build_materials_slots,
    }]
}
