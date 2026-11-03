//! A normal-mapped tube bent at a skinned joint.
//!
//! The tube runs along Z and is skinned to two joints that meet at its middle,
//! with the weights blended over a band around the joint. The upper joint is
//! bent 90 degrees about X, so the upper half lies along +Y. The normal map's
//! bumps must shade the same on both halves: a tangent frame left in the bind
//! pose tilts them the wrong way on the bent half. Beside it, a normal-mapped
//! sphere squashed by a non-uniform model scale checks the model normal
//! transform.
//!
//! Building it registers the skinning deformer, which recomposes the mesh
//! shaders for every scene built after it on the same renderer, so it is kept
//! last in the catalogue.

use glam::{Mat4, Vec3};
use viewport_lib::plugins::skinning::{SkinWeights, SkinningPlugin};
use viewport_lib::{
    Colour, LightKind, LightSource, LightingSettings, Material, MeshData, primitives,
};

use super::{BuildCtx, BuiltScene, NamedScene, item_model, standard_cameras, textures, upload};

const RADIUS: f32 = 0.45;
const HALF_LENGTH: f32 = 1.5;
/// Half the height of the band around the joint over which the two joints'
/// weights blend.
const BLEND: f32 = 0.35;

/// The tube's rings, denser around the joint where it bends.
fn ring_heights() -> Vec<f32> {
    let mut zs = Vec::new();
    let mut z = -HALF_LENGTH;
    while z < HALF_LENGTH - 1e-4 {
        zs.push(z);
        z += if z.abs() < BLEND + 0.1 { 0.05 } else { 0.25 };
    }
    zs.push(HALF_LENGTH);
    zs
}

/// An open tube along Z with radial normals, circumferential tangents, UVs, and
/// skin weights blending joint 0 (below) into joint 1 (above) across the band.
fn tube() -> (MeshData, SkinWeights) {
    let sectors = 48u32;
    let zs = ring_heights();
    let mut mesh = MeshData::default();
    let mut tangents = Vec::new();
    let mut uvs = Vec::new();
    let mut weights = SkinWeights {
        joint_indices: Vec::new(),
        joint_weights: Vec::new(),
    };
    for &z in &zs {
        let w1 = ((z + BLEND) / (2.0 * BLEND)).clamp(0.0, 1.0);
        for s in 0..=sectors {
            let a = std::f32::consts::TAU * s as f32 / sectors as f32;
            let (sin, cos) = a.sin_cos();
            mesh.positions.push([RADIUS * cos, RADIUS * sin, z]);
            mesh.normals.push([cos, sin, 0.0]);
            tangents.push([-sin, cos, 0.0, 1.0]);
            uvs.push([
                s as f32 / sectors as f32,
                (z + HALF_LENGTH) / (2.0 * HALF_LENGTH),
            ]);
            weights.joint_indices.push([0, 1, 0, 0]);
            weights.joint_weights.push([1.0 - w1, w1, 0.0, 0.0]);
        }
    }
    let row = sectors + 1;
    for r in 0..zs.len() as u32 - 1 {
        for s in 0..sectors {
            let a = r * row + s;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            mesh.indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    mesh.tangents = Some(tangents);
    mesh.uvs = Some(uvs);
    (mesh, weights)
}

fn build_skinned_normalmap(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let skin = SkinningPlugin::install(ctx.renderer.resources_mut(), ctx.device)
        .expect("register the skinning deformer");

    let bumps = textures::normal_bumps(256, 6);
    let map = ctx
        .renderer
        .resources_mut()
        .upload_texture(
            ctx.device,
            ctx.queue,
            viewport_lib::TextureData::normal_map(bumps.width, bumps.height, bumps.rgba.to_vec()),
        )
        .expect("normal map upload");
    let normal_mapped = |colour: Colour| {
        let mut m = Material::pbr(colour, 0.0, 0.45);
        m.normal_map_id = Some(map);
        m
    };

    let (mesh, weights) = tube();
    let tube_mesh = upload(ctx, &mesh);
    skin.attach_weights(
        ctx.renderer.resources_mut(),
        ctx.device,
        tube_mesh,
        &weights,
    );
    // Joint 0 stays in the bind pose; joint 1 sits at the origin and bends the
    // upper half 90 degrees about X, toward +Y.
    let palette = [
        Mat4::IDENTITY,
        Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
    ];
    let attached = skin.attach_palette(
        ctx.renderer.resources_mut(),
        ctx.device,
        ctx.queue,
        tube_mesh,
        0,
        &palette,
    );
    assert!(attached, "the tube has skin weights");
    let mut tube_item = item_model(
        ctx,
        tube_mesh,
        Mat4::IDENTITY,
        normal_mapped(Colour::linear_rgb(0.75, 0.7, 0.65)),
    );
    tube_item.deform_instance = Some(0);

    let sphere = upload(ctx, &primitives::sphere(0.8, 64, 32));
    let squashed = item_model(
        ctx,
        sphere,
        Mat4::from_translation(Vec3::new(2.3, -0.9, -0.5))
            * Mat4::from_rotation_z(0.6)
            * Mat4::from_scale(Vec3::new(1.5, 0.8, 0.6)),
        normal_mapped(Colour::linear_rgb(0.6, 0.65, 0.75)),
    );

    let mut fill = LightSource::default();
    fill.kind = LightKind::Directional {
        direction: [-0.8, -0.5, 0.3],
    };
    fill.intensity = 0.5;
    fill.cast_shadows = false;
    let mut lighting = LightingSettings::default();
    lighting.lights.push(fill);

    BuiltScene {
        items: vec![tube_item, squashed],
        lighting,
        ..Default::default()
    }
}

/// The skinned scene.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "skinned_normalmap",
        cameras: standard_cameras(Vec3::new(1.0, 0.0, -0.2), 5.8),
        build: build_skinned_normalmap,
    }]
}
