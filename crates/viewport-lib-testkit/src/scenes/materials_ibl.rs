//! Spheres lit only by an environment map.
//!
//! The environment is built in code: a sky gradient over a darker ground, with
//! a bright sun and a ring of dark posts on the horizon, so a smooth sphere
//! shows a sharp reflection and a rough one shows how far the lobe spreads.
//! No analytic lights and no hemisphere ambient, so every pixel is the
//! environment's diffuse and specular terms. The layered-lobe grids (clear
//! coat, sheen, anisotropy) use the same environment.

use glam::Vec3;
use viewport_lib::{
    Colour, EnvironmentLighting, EnvironmentOptions, LightingSettings, Material, TextureData,
    TextureSlot, primitives,
};

use super::materials::{grid_camera, upload_texture};
use super::{BuildCtx, BuiltScene, NamedScene, item, textures, upload};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;

/// The procedural environment as linear RGBA f32 texels, equirect with +Z up.
fn environment_pixels() -> Vec<f32> {
    let sun = Vec3::new(0.6, -0.5, 0.62).normalize();
    let mut px = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        let elevation = (0.5 - (y as f32 + 0.5) / HEIGHT as f32) * std::f32::consts::PI;
        for x in 0..WIDTH {
            let azimuth = ((x as f32 + 0.5) / WIDTH as f32 - 0.5) * std::f32::consts::TAU;
            let dir = Vec3::new(
                elevation.cos() * azimuth.cos(),
                elevation.cos() * azimuth.sin(),
                elevation.sin(),
            );
            let mut rgb = if elevation >= 0.0 {
                let t = elevation / std::f32::consts::FRAC_PI_2;
                Vec3::new(0.9, 0.95, 1.0).lerp(Vec3::new(0.25, 0.45, 0.9), t.sqrt())
            } else {
                Vec3::new(0.3, 0.25, 0.2)
            };
            // Twelve dark posts just above the horizon.
            if (0.0..0.35).contains(&elevation) && (x * 12 / WIDTH) % 2 == 0 && x % 21 < 4 {
                rgb = Vec3::splat(0.05);
            }
            if dir.dot(sun) > 0.995 {
                rgb = Vec3::new(40.0, 36.0, 30.0);
            }
            px.extend_from_slice(&[rgb.x, rgb.y, rgb.z, 1.0]);
        }
    }
    px
}

/// Upload the procedural environment.
fn upload_environment(ctx: &mut BuildCtx<'_>) -> viewport_lib::EnvironmentMapId {
    ctx.renderer
        .upload_environment(
            ctx.device,
            ctx.queue,
            TextureData::hdr(WIDTH, HEIGHT, environment_pixels()),
            EnvironmentOptions::default(),
        )
        .expect("environment upload")
}

/// A scene of `items` lit only by the procedural environment.
fn environment_scene(
    items: Vec<viewport_lib::SceneRenderItem>,
    env: viewport_lib::EnvironmentMapId,
) -> BuiltScene {
    let mut lighting = LightingSettings::default();
    lighting.lights.clear();
    lighting.hemisphere_intensity = 0.0;
    BuiltScene {
        items,
        lighting,
        environment: Some(EnvironmentLighting::new(env)),
        ..Default::default()
    }
}

/// Grid position of a sphere: five columns, rows downward.
fn at(row: usize, col: usize) -> Vec3 {
    Vec3::new(col as f32 - 2.0, 0.0, -(row as f32))
}

/// Rows from the top: metallic 0, 0.5 and 1 across roughness 0 to 1; dark
/// dielectrics at reflectance 0, 0.25, 0.5, 0.75 and 1; and an AO-mapped row.
fn build_materials_ibl(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let env = upload_environment(ctx);
    let sphere = upload(ctx, &primitives::sphere(0.42, 48, 24));
    let ao = textures::checker(256, 6, [255, 255, 255], [70, 70, 70]);
    let ao_tex = upload_texture(ctx, TextureData::linear(ao.width, ao.height, ao.rgba));

    let base = Colour::linear_rgb(0.85, 0.82, 0.78);
    let mut items = Vec::new();
    for (row, metallic) in [0.0, 0.5, 1.0].into_iter().enumerate() {
        for col in 0..5 {
            let roughness = col as f32 / 4.0;
            let material = Material::pbr(base, metallic, roughness);
            items.push(item(ctx, sphere, at(row, col), material));
        }
    }
    for col in 0..5 {
        let mut material = Material::pbr(Colour::linear_rgb(0.04, 0.04, 0.05), 0.0, 0.3);
        material.reflectance = col as f32 / 4.0;
        items.push(item(ctx, sphere, at(3, col), material));
    }
    for (col, (metallic, roughness)) in
        [(0.0, 0.5), (0.0, 1.0), (0.5, 0.5), (1.0, 0.25), (1.0, 0.75)]
            .into_iter()
            .enumerate()
    {
        let material =
            Material::pbr(base, metallic, roughness).with_texture(TextureSlot::Ao, ao_tex);
        items.push(item(ctx, sphere, at(4, col), material));
    }
    environment_scene(items, env)
}

/// Clear coat over a red dielectric. Rows from the top: coat 0.25, 0.5 and
/// 1 at coat roughness 0, then coat 1 at coat roughness 0.3 and 0.7. Columns:
/// base roughness 0 to 1.
fn build_lobes_clearcoat(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let env = upload_environment(ctx);
    let sphere = upload(ctx, &primitives::sphere(0.42, 48, 24));
    let base = Colour::linear_rgb(0.6, 0.06, 0.05);
    let mut items = Vec::new();
    for (row, (coat, coat_roughness)) in
        [(0.25, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.3), (1.0, 0.7)]
            .into_iter()
            .enumerate()
    {
        for col in 0..5 {
            let material =
                Material::pbr(base, 0.0, col as f32 / 4.0).with_clearcoat(coat, coat_roughness);
            items.push(item(ctx, sphere, at(row, col), material));
        }
    }
    environment_scene(items, env)
}

/// Sheen over a dark cloth base. Rows from the top: white, red, blue and
/// gold sheen, then white sheen over a mid-grey base. Columns: sheen
/// roughness 0 to 1.
fn build_lobes_sheen(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let env = upload_environment(ctx);
    let sphere = upload(ctx, &primitives::sphere(0.42, 48, 24));
    let dark = Colour::linear_rgb(0.04, 0.04, 0.05);
    let grey = Colour::linear_rgb(0.4, 0.4, 0.4);
    let mut items = Vec::new();
    for (row, (base, sheen)) in [
        (dark, Colour::linear_rgb(1.0, 1.0, 1.0)),
        (dark, Colour::linear_rgb(0.9, 0.1, 0.1)),
        (dark, Colour::linear_rgb(0.2, 0.3, 1.0)),
        (dark, Colour::linear_rgb(1.0, 0.75, 0.3)),
        (grey, Colour::linear_rgb(1.0, 1.0, 1.0)),
    ]
    .into_iter()
    .enumerate()
    {
        for col in 0..5 {
            let material =
                Material::pbr(base, 0.0, 0.8).with_sheen(sheen, (col as f32 / 4.0).max(0.05));
            items.push(item(ctx, sphere, at(row, col), material));
        }
    }
    environment_scene(items, env)
}

/// Anisotropic brushed metal. Rows from the top: anisotropy -1, -0.5, 0.5
/// and 1 on a metal at roughness 0.4, then 0.8 on a dielectric at roughness
/// 0.3. Columns: rotation 0 to a quarter turn. The sphere's tangent runs
/// around its axis.
fn build_lobes_anisotropy(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let env = upload_environment(ctx);
    let sphere = upload(ctx, &primitives::sphere(0.42, 48, 24));
    let metal = Colour::linear_rgb(0.95, 0.9, 0.85);
    let mut items = Vec::new();
    for (row, (anisotropy, metallic, roughness, base)) in [
        (-1.0, 1.0, 0.4, metal),
        (-0.5, 1.0, 0.4, metal),
        (0.5, 1.0, 0.4, metal),
        (1.0, 1.0, 0.4, metal),
        (0.8, 0.0, 0.3, Colour::linear_rgb(0.05, 0.2, 0.5)),
    ]
    .into_iter()
    .enumerate()
    {
        for col in 0..5 {
            let rotation = col as f32 * std::f32::consts::FRAC_PI_8;
            let material =
                Material::pbr(base, metallic, roughness).with_anisotropy(anisotropy, rotation);
            items.push(item(ctx, sphere, at(row, col), material));
        }
    }
    environment_scene(items, env)
}

/// Glass spheres over a checker floor. From the left: a thin wall (no
/// bending), a solid sphere (thickness its diameter), the same at roughness
/// 0.35, and a solid sphere absorbing toward blue. The floor shows through
/// each, refracted by the solid ones and blurred by the rough one.
fn build_transmission_checker(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let env = upload_environment(ctx);
    let checker = textures::checker(384, 12, [235, 235, 235], [30, 30, 40]);
    let checker = upload_texture(
        ctx,
        TextureData::srgb(checker.width, checker.height, checker.rgba),
    );
    let floor = upload(ctx, &primitives::plane(6.0, 6.0));
    let sphere = upload(ctx, &primitives::sphere(0.5, 64, 32));
    let floor_material = Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.8)
        .with_texture(TextureSlot::Albedo, checker);
    let mut items = vec![item(ctx, floor, Vec3::new(0.0, 0.0, -0.6), floor_material)];
    let glass = |roughness: f32, thickness: f32| {
        Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, roughness)
            .with_transmission(1.0, thickness)
            .with_ior(1.5)
    };
    let spheres = [
        glass(0.0, 0.0),
        glass(0.0, 1.0),
        glass(0.35, 1.0),
        glass(0.0, 1.0).with_attenuation(Colour::linear_rgb(0.2, 0.5, 1.0), 0.6),
    ];
    for (i, material) in spheres.into_iter().enumerate() {
        let x = i as f32 * 1.5 - 2.25;
        items.push(item(ctx, sphere, Vec3::new(x, 0.0, 0.0), material));
    }
    environment_scene(items, env)
}

/// The environment-lit material scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "materials_ibl",
            cameras: grid_camera(-2.0, 7.0),
            build: build_materials_ibl,
        },
        NamedScene {
            name: "lobes_clearcoat",
            cameras: grid_camera(-2.0, 7.0),
            build: build_lobes_clearcoat,
        },
        NamedScene {
            name: "lobes_sheen",
            cameras: grid_camera(-2.0, 7.0),
            build: build_lobes_sheen,
        },
        NamedScene {
            name: "lobes_anisotropy",
            cameras: grid_camera(-2.0, 7.0),
            build: build_lobes_anisotropy,
        },
        NamedScene {
            name: "transmission_checker",
            cameras: vec![super::NamedCamera {
                name: "iso",
                camera: super::orbit_camera(Vec3::new(0.0, 0.0, -0.3), 5.0, 0.0, 0.9),
            }],
            build: build_transmission_checker,
        },
    ]
}
