//! Light transport scenes: where exposure reaches and where it does not.
//!
//! Each scene puts one kind of radiance source beside a reference under a
//! named exposure: emitters in nits beside an unlit card and a lit 18% card,
//! emitters across the emissive exposure weight,
//! bloom from an emitter at physical and at unit scale, a textured PBR model
//! under image-based lighting with and without an environment zone, a
//! captured reflection probe at two intensities, a `recolour` hook surface, a
//! scalar colour map, and the default lighting with no environment.

use glam::{Mat4, Quat, Vec3};
use viewport_lib::{
    Aabb, AttributeData, AttributeKind, AttributeRef, Colour, ColourmapId, EnvironmentIntensity,
    EnvironmentLighting, EnvironmentOptions, EnvironmentZone, ExposureSettings, LightingSettings,
    Material, MeshData, PostProcessSettings, SceneRenderItem, TextureData, primitives,
};
use viewport_lib::{ColourSource, SizeSource};
use viewport_lib_plugins::item_types::point_cloud::PointCloudItem;

use super::materials::upload_texture;
use super::materials_ibl::upload_environment;
use super::{
    BuildCtx, BuiltScene, NamedCamera, NamedScene, frame_for, item_model, orbit_camera, textures,
    upload,
};
use crate::fixtures::material_plugin::RecolourTintMaterialPlugin;

const CARD: f32 = 0.9;
const COL_STEP: f32 = 1.2;

/// A diffuse card: no specular, so it reflects `albedo * E / pi`.
fn lambert(albedo: f32) -> Material {
    let mut m = Material::from_colour(Colour::linear_rgb(albedo, albedo, albedo));
    m.specular = 0.0;
    m
}

/// A black surface emitting `nits` of white.
fn emitter(nits: f32) -> Material {
    let mut m = lambert(0.0);
    m.emissive = Colour::WHITE;
    m.emissive_strength = nits;
    m
}

/// Daylight with its shadows off, so the charts read flat.
fn daylight() -> LightingSettings {
    let mut lighting = LightingSettings::daylight();
    for light in &mut lighting.lights {
        light.cast_shadows = false;
    }
    lighting
}

fn sunny16() -> ExposureSettings {
    ExposureSettings::physical(16.0, 1.0 / 125.0, 100.0)
}

fn ground_item(ctx: &mut BuildCtx<'_>, albedo: f32) -> SceneRenderItem {
    let ground = upload(ctx, &primitives::plane(200.0, 200.0));
    item_model(ctx, ground, Mat4::IDENTITY, lambert(albedo))
}

/// Straight down, orthographic, framing a row of cards.
fn top_camera() -> Vec<NamedCamera> {
    let mut camera = orbit_camera(Vec3::ZERO, 6.0, 0.0, 0.0);
    camera.set_orthographic(true);
    camera.aspect = 4.0 / 3.0;
    vec![NamedCamera {
        name: "top",
        camera,
    }]
}

/// A three-quarter perspective view of a small set.
fn iso_camera(distance: f32) -> Vec<NamedCamera> {
    let mut camera = orbit_camera(Vec3::new(0.0, 0.0, 0.4), distance, 0.5, 1.05);
    camera.aspect = 4.0 / 3.0;
    vec![NamedCamera {
        name: "iso",
        camera,
    }]
}

// --- emitters ---------------------------------------------------------------

/// Left to right: a 50 nit emitter, a 2 000 nit emitter, an unlit orange card
/// and a Lambert 18% card, on a mid-grey ground under full daylight.
fn emitter_chart(ctx: &mut BuildCtx<'_>, exposure: ExposureSettings) -> BuiltScene {
    let card = upload(ctx, &primitives::plane(CARD, CARD));
    let mut items = vec![ground_item(ctx, 0.3)];
    let x = |i: usize| (i as f32 - 1.5) * COL_STEP;
    let at = |i: usize| Mat4::from_translation(Vec3::new(x(i), 0.0, 0.01));
    items.push(item_model(ctx, card, at(0), emitter(50.0)));
    items.push(item_model(ctx, card, at(1), emitter(2_000.0)));
    let mut unlit = item_model(
        ctx,
        card,
        at(2),
        Material::from_colour(Colour::linear_rgb(0.9, 0.45, 0.1)),
    );
    unlit.settings.unlit = true;
    items.push(unlit);
    items.push(item_model(ctx, card, at(3), lambert(0.18)));
    BuiltScene {
        items,
        lighting: daylight(),
        exposure: Some(exposure),
        ..Default::default()
    }
}

fn build_emitter_ev0(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    emitter_chart(ctx, ExposureSettings::manual(0.0))
}

fn build_emitter_sunny16(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    emitter_chart(ctx, sunny16())
}

/// Left to right: three 2 000 nit emitters at emissive exposure weights 1,
/// 0.5 and 0, and a Lambert 18% card, on a mid-grey ground in daylight under
/// a sunny-16 camera.
fn build_emitter_weights(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let card = upload(ctx, &primitives::plane(CARD, CARD));
    let mut items = vec![ground_item(ctx, 0.3)];
    let x = |i: usize| (i as f32 - 1.5) * COL_STEP;
    let at = |i: usize| Mat4::from_translation(Vec3::new(x(i), 0.0, 0.01));
    for (i, weight) in [1.0, 0.5, 0.0].into_iter().enumerate() {
        let m = emitter(2_000.0).with_emissive_exposure_weight(weight);
        items.push(item_model(ctx, card, at(i), m));
    }
    items.push(item_model(ctx, card, at(3), lambert(0.18)));
    BuiltScene {
        items,
        lighting: daylight(),
        exposure: Some(sunny16()),
        ..Default::default()
    }
}

// --- bloom ------------------------------------------------------------------

/// An emitting sphere of `nits` between a lit cube and a lit sphere on a
/// ground, with bloom on.
fn bloom_scene(
    ctx: &mut BuildCtx<'_>,
    nits: f32,
    lighting: LightingSettings,
    exposure: ExposureSettings,
) -> BuiltScene {
    let sphere = upload(ctx, &primitives::sphere(0.45, 32, 16));
    let cube = upload(ctx, &primitives::cube(0.8));
    let lit = Material::pbr(Colour::linear_rgb(0.7, 0.68, 0.64), 0.0, 0.5);
    let items = vec![
        ground_item(ctx, 0.3),
        item_model(
            ctx,
            sphere,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.45)),
            emitter(nits),
        ),
        item_model(
            ctx,
            cube,
            Mat4::from_translation(Vec3::new(-1.4, 0.3, 0.4)),
            lit,
        ),
        item_model(
            ctx,
            sphere,
            Mat4::from_translation(Vec3::new(1.4, -0.3, 0.45)),
            lit,
        ),
    ];
    let mut post = PostProcessSettings::default();
    post.bloom.enabled = true;
    BuiltScene {
        items,
        lighting,
        exposure: Some(exposure),
        post_process: Some(post),
        ..Default::default()
    }
}

fn build_bloom_physical(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    bloom_scene(ctx, 2_000.0, daylight(), sunny16())
}

/// The same set at unit scale: a 2 nit emitter under the default 1 lux key.
fn build_bloom_unit(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut lighting = LightingSettings::default();
    for light in &mut lighting.lights {
        light.cast_shadows = false;
    }
    bloom_scene(ctx, 2.0, lighting, ExposureSettings::default())
}

// --- textured model under IBL -----------------------------------------------

/// A sphere and a cube carrying every glTF texture slot: base colour,
/// tangent-space normal, metallic-roughness, occlusion and emissive.
fn textured_model(ctx: &mut BuildCtx<'_>) -> Vec<SceneRenderItem> {
    let albedo = textures::checker(256, 4, [200, 170, 120], [90, 95, 110]);
    let albedo = upload_texture(
        ctx,
        TextureData::srgb(albedo.width, albedo.height, albedo.rgba),
    );
    let normal = textures::normal_bumps(256, 4);
    let normal = upload_texture(
        ctx,
        TextureData::normal_map(normal.width, normal.height, normal.rgba),
    );
    // A smooth gradient: occlusion (R), roughness (G) and metallic (B) each
    // sweep across the surface.
    let ramp = textures::gradient(256);
    let orm = upload_texture(ctx, TextureData::linear(ramp.width, ramp.height, ramp.rgba));
    let glow = textures::checker(256, 4, [255, 120, 30], [0, 0, 0]);
    let glow = upload_texture(ctx, TextureData::srgb(glow.width, glow.height, glow.rgba));

    let mut m = Material::pbr(Colour::WHITE, 1.0, 1.0);
    m.texture_id = Some(albedo);
    m.normal_map_id = Some(normal);
    m.metallic_roughness_texture_id = Some(orm);
    m.ao_map_id = Some(orm);
    m.emissive_texture_id = Some(glow);
    m.emissive = Colour::WHITE;
    m.emissive_strength = 0.6;

    let sphere = upload(ctx, &primitives::sphere(0.7, 64, 32));
    let cube = upload(ctx, &primitives::cube(0.9));
    vec![
        item_model(
            ctx,
            sphere,
            Mat4::from_translation(Vec3::new(-0.6, 0.0, 0.7)),
            m,
        ),
        item_model(
            ctx,
            cube,
            Mat4::from_rotation_translation(Quat::from_rotation_z(0.6), Vec3::new(0.95, 0.4, 0.45)),
            m,
        ),
    ]
}

/// A warm, dim interior: a uniform amber upper half over a dark floor.
fn interior_environment(ctx: &mut BuildCtx<'_>) -> viewport_lib::EnvironmentMapId {
    const W: u32 = 64;
    const H: u32 = 32;
    let mut px = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        let rgb = if y < H / 2 {
            [0.9, 0.55, 0.25]
        } else {
            [0.08, 0.06, 0.05]
        };
        for _ in 0..W {
            px.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
        }
    }
    ctx.renderer
        .upload_environment(
            ctx.device,
            ctx.queue,
            TextureData::hdr(W, H, px),
            EnvironmentOptions::default(),
        )
        .expect("environment upload")
}

/// Image-based lighting only: no analytic lights and no hemisphere fill.
fn ibl_lighting() -> LightingSettings {
    let mut lighting = LightingSettings::default();
    lighting.lights.clear();
    lighting.hemisphere_intensity = 0.0;
    lighting
}

fn build_helmet_ibl(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let env = upload_environment(ctx);
    let items = textured_model(ctx);
    BuiltScene {
        items,
        lighting: ibl_lighting(),
        environment: Some(EnvironmentLighting::new(env)),
        ..Default::default()
    }
}

/// The same model with an interior zone over its right half: the cube sits
/// inside the zone, the sphere outside it, with a fade band between.
fn build_helmet_zone(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let env = upload_environment(ctx);
    let interior = interior_environment(ctx);
    let items = textured_model(ctx);
    let zone = EnvironmentZone::new(Vec3::new(1.2, 0.0, 1.0), Vec3::new(0.9, 2.0, 1.5), interior)
        .with_fade_distance(0.4);
    ctx.renderer.set_environment_zones(ctx.queue, &[zone]);
    BuiltScene {
        items,
        lighting: ibl_lighting(),
        environment: Some(EnvironmentLighting::new(env)),
        ..Default::default()
    }
}

// --- reflection probe -------------------------------------------------------

/// A mirror sphere between a red and a blue box on a ground, under the
/// procedural environment and a unit key light. A reflection probe captured
/// above the sphere, with a box around the set, lights it again at
/// `intensity`.
fn build_probe(ctx: &mut BuildCtx<'_>, intensity: f32) -> BuiltScene {
    let env = upload_environment(ctx);
    let sphere = upload(ctx, &primitives::sphere(0.5, 48, 24));
    let cube = upload(ctx, &primitives::cube(0.7));
    let items = vec![
        ground_item(ctx, 0.5),
        item_model(
            ctx,
            sphere,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5)),
            Material::pbr(Colour::linear_rgb(0.95, 0.95, 0.95), 1.0, 0.08),
        ),
        item_model(
            ctx,
            cube,
            Mat4::from_translation(Vec3::new(-1.2, 0.0, 0.35)),
            Material::pbr(Colour::linear_rgb(0.8, 0.1, 0.08), 0.0, 0.6),
        ),
        item_model(
            ctx,
            cube,
            Mat4::from_translation(Vec3::new(1.2, 0.0, 0.35)),
            Material::pbr(Colour::linear_rgb(0.08, 0.15, 0.8), 0.0, 0.6),
        ),
    ];
    let mut lighting = LightingSettings::default();
    for light in &mut lighting.lights {
        light.cast_shadows = false;
    }
    let scene = BuiltScene {
        items,
        lighting,
        environment: Some(EnvironmentLighting::new(env)),
        ..Default::default()
    };

    let bounds = Aabb {
        min: Vec3::new(-2.0, -2.0, 0.0),
        max: Vec3::new(2.0, 2.0, 2.4),
    };
    let camera = &iso_camera(5.0)[0].camera;
    let mut frame = frame_for(&scene, camera, [400.0, 300.0]);
    // The probe sees the environment as its sky; the view itself keeps the
    // flat test background.
    frame.viewport.environment_background = viewport_lib::EnvironmentBackground::default();
    let zones = ctx
        .renderer
        .capture_reflection_probes(ctx.device, ctx.queue, &mut frame, &[(bounds, 0.5)], 64, 64)
        .expect("probe capture");
    let zones: Vec<EnvironmentZone> = zones
        .into_iter()
        .map(|zone| zone.with_intensity(EnvironmentIntensity::Multiplier(intensity)))
        .collect();
    ctx.renderer.set_environment_zones(ctx.queue, &zones);
    scene
}

fn build_probe_x1(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    build_probe(ctx, 1.0)
}

fn build_probe_x4(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    build_probe(ctx, 4.0)
}

// --- recolour hook -----------------------------------------------------------

/// Two spheres in daylight under a physical camera: the stock material on the
/// left, the same material through a `recolour` hook that tints it cyan on the
/// right.
fn build_recolour_physical(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let plugin = RecolourTintMaterialPlugin::new("light_transport_recolour", [0.3, 0.9, 1.0]);
    let plugin_id = ctx
        .renderer
        .resources_mut()
        .register_material_plugin(ctx.device, &plugin)
        .expect("register recolour plugin");
    let sphere = upload(ctx, &primitives::sphere(0.6, 48, 24));
    let base = Material::pbr(Colour::linear_rgb(0.7, 0.7, 0.7), 0.0, 0.5);
    let items = vec![
        ground_item(ctx, 0.3),
        item_model(
            ctx,
            sphere,
            Mat4::from_translation(Vec3::new(-0.8, 0.0, 0.6)),
            base,
        ),
        item_model(
            ctx,
            sphere,
            Mat4::from_translation(Vec3::new(0.8, 0.0, 0.6)),
            base.with_plugin(plugin_id),
        ),
    ];
    BuiltScene {
        items,
        lighting: daylight(),
        exposure: Some(sunny16()),
        ..Default::default()
    }
}

// --- colour map -------------------------------------------------------------

/// A plane carrying a per-vertex scalar that ramps along X.
fn ramp_plane() -> MeshData {
    let mut mesh = primitives::plane(3.0, 0.7);
    let values = mesh.positions.iter().map(|p| p[0] / 3.0 + 0.5).collect();
    mesh.attributes
        .insert("ramp".into(), AttributeData::Vertex(values));
    mesh
}

/// A colour-mapped plane at `y`, lit or unlit.
fn mapped_plane(
    ctx: &mut BuildCtx<'_>,
    mesh: viewport_lib::MeshId,
    y: f32,
    unlit: bool,
) -> SceneRenderItem {
    let mut it = item_model(
        ctx,
        mesh,
        Mat4::from_translation(Vec3::new(-0.4, y, 0.01)),
        lambert(0.5),
    );
    it.active_attribute = Some(AttributeRef {
        name: "ramp".into(),
        kind: AttributeKind::Vertex,
    });
    it.colourmap_id = Some(ColourmapId(0));
    it.scalar_range = Some((0.0, 1.0));
    it.settings.unlit = unlit;
    it
}

/// Top to bottom: a lit colour-mapped plane, the same plane unlit, and a
/// colour-mapped point cloud, beside a lit 18% grey sphere, in daylight under
/// a physical camera.
fn build_colourmap_physical(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let plane = upload(ctx, &ramp_plane());
    let sphere = upload(ctx, &primitives::sphere(0.45, 32, 16));
    let items = vec![
        ground_item(ctx, 0.3),
        mapped_plane(ctx, plane, 1.0, false),
        mapped_plane(ctx, plane, 0.0, true),
        item_model(
            ctx,
            sphere,
            Mat4::from_translation(Vec3::new(1.9, 0.0, 0.45)),
            lambert(0.18),
        ),
    ];

    let mut cloud = PointCloudItem::default();
    let mut values = Vec::new();
    for i in 0..12 {
        for j in 0..2 {
            cloud
                .positions
                .push([-1.9 + i as f32 * (3.0 / 11.0), -0.85 - j as f32 * 0.3, 0.1]);
            values.push(i as f32 / 11.0);
        }
    }
    cloud.colour = ColourSource::Scalar {
        values,
        range: Some((0.0, 1.0)),
        colourmap: Some(ColourmapId(0)),
    };
    cloud.size = SizeSource::Uniform(10.0);

    BuiltScene {
        items,
        point_clouds: vec![cloud],
        lighting: daylight(),
        exposure: Some(sunny16()),
        ..Default::default()
    }
}

// --- no environment ---------------------------------------------------------

/// Spheres crossing roughness (columns) with metallic (rows) on a ground,
/// under the default lighting with no environment.
fn build_no_environment(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let sphere = upload(ctx, &primitives::sphere(0.35, 40, 20));
    let mut items = vec![ground_item(ctx, 0.5)];
    for (row, metallic) in [0.0, 1.0].into_iter().enumerate() {
        for col in 0..4 {
            let roughness = 0.1 + col as f32 * 0.3;
            let at = Vec3::new((col as f32 - 1.5) * 0.9, (row as f32 - 0.5) * 0.9, 0.35);
            items.push(item_model(
                ctx,
                sphere,
                Mat4::from_translation(at),
                Material::pbr(Colour::linear_rgb(0.8, 0.6, 0.4), metallic, roughness),
            ));
        }
    }
    BuiltScene {
        items,
        lighting: LightingSettings::default(),
        ..Default::default()
    }
}

/// The light transport scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "light_transport_emitter_ev0",
            cameras: top_camera(),
            build: build_emitter_ev0,
        },
        NamedScene {
            name: "light_transport_emitter_sunny16",
            cameras: top_camera(),
            build: build_emitter_sunny16,
        },
        NamedScene {
            name: "light_transport_emitter_weights",
            cameras: top_camera(),
            build: build_emitter_weights,
        },
        NamedScene {
            name: "light_transport_bloom_physical",
            cameras: iso_camera(5.0),
            build: build_bloom_physical,
        },
        NamedScene {
            name: "light_transport_bloom_unit",
            cameras: iso_camera(5.0),
            build: build_bloom_unit,
        },
        NamedScene {
            name: "light_transport_helmet_ibl",
            cameras: iso_camera(4.0),
            build: build_helmet_ibl,
        },
        NamedScene {
            name: "light_transport_helmet_zone",
            cameras: iso_camera(4.0),
            build: build_helmet_zone,
        },
        NamedScene {
            name: "light_transport_probe_x1",
            cameras: iso_camera(5.0),
            build: build_probe_x1,
        },
        NamedScene {
            name: "light_transport_probe_x4",
            cameras: iso_camera(5.0),
            build: build_probe_x4,
        },
        NamedScene {
            name: "light_transport_recolour_physical",
            cameras: iso_camera(4.0),
            build: build_recolour_physical,
        },
        NamedScene {
            name: "light_transport_colourmap_physical",
            cameras: top_camera(),
            build: build_colourmap_physical,
        },
        NamedScene {
            name: "light_transport_no_environment",
            cameras: iso_camera(4.5),
            build: build_no_environment,
        },
    ]
}
