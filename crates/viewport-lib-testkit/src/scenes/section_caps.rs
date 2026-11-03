//! Section caps: the filled cross-sections clip planes leave through solids.

use glam::Vec3;
use viewport_lib::{ClipObject, ClipShape, Colour, Material, MeshData};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, item, orbit_camera, rigs, upload};

/// A closed hollow tube along +Z from `z = 0` to `height`, with flat-shaded
/// walls and end rings. Vertices sit half a sector off the axes so no section
/// through the axis passes exactly through one.
pub fn hollow_tube(outer: f32, inner: f32, height: f32, sectors: u32) -> MeshData {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let step = std::f32::consts::TAU / sectors as f32;
    let at = |r: f32, k: u32, z: f32| {
        let a = (k as f32 + 0.5) * step;
        [r * a.cos(), r * a.sin(), z]
    };
    let mut quad = |p: [[f32; 3]; 4], n: [f32; 3]| {
        let base = positions.len() as u32;
        positions.extend_from_slice(&p);
        normals.extend_from_slice(&[n; 4]);
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };
    for k in 0..sectors {
        let mid = (k as f32 + 1.0) * step;
        let radial = [mid.cos(), mid.sin(), 0.0];
        let inward = [-radial[0], -radial[1], 0.0];
        quad(
            [
                at(outer, k, 0.0),
                at(outer, k + 1, 0.0),
                at(outer, k + 1, height),
                at(outer, k, height),
            ],
            radial,
        );
        quad(
            [
                at(inner, k + 1, 0.0),
                at(inner, k, 0.0),
                at(inner, k, height),
                at(inner, k + 1, height),
            ],
            inward,
        );
        quad(
            [
                at(outer, k, height),
                at(outer, k + 1, height),
                at(inner, k + 1, height),
                at(inner, k, height),
            ],
            [0.0, 0.0, 1.0],
        );
        quad(
            [
                at(outer, k + 1, 0.0),
                at(outer, k, 0.0),
                at(inner, k, 0.0),
                at(inner, k + 1, 0.0),
            ],
            [0.0, 0.0, -1.0],
        );
    }
    MeshData::new(positions, normals, indices)
}

/// A hollow tube opened by two crossing planes, each cap stopping at the other
/// plane and a box clip volume trimming both. The ground and a cube are
/// `ignore_clip`, so they stay whole and get no cap; a plane that does not
/// clip geometry leaves no cut and no cap.
fn build_section_caps(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut ground = super::ground(ctx, 0.0);
    ground.settings.ignore_clip = true;
    let tube = upload(ctx, &hollow_tube(1.0, 0.6, 2.0, 48));
    let cube = upload(ctx, &viewport_lib::primitives::cube(0.8));
    let tube = item(
        ctx,
        tube,
        Vec3::ZERO,
        Material::pbr(Colour::linear_rgb(0.8, 0.45, 0.2), 0.0, 0.6),
    );
    let mut cube = item(
        ctx,
        cube,
        Vec3::new(0.0, 2.0, 1.4),
        Material::pbr(Colour::linear_rgb(0.3, 0.6, 0.35), 0.0, 0.6),
    );
    cube.settings.ignore_clip = true;

    // Keep x <= 0, with the cap in its own colour.
    let mut lengthwise = ClipObject::plane([-1.0, 0.0, 0.0], 0.0);
    if let ClipShape::Plane { cap_colour, .. } = &mut lengthwise.shape {
        *cap_colour = Some([0.2, 0.35, 0.8, 1.0]);
    }
    // Keep z <= 1.4.
    let across = ClipObject::plane([0.0, 0.0, -1.0], 1.4);
    // Keep y >= -0.7 (and everything else in reach).
    let trim = ClipObject::box_shape(
        [0.0, 2.3, 1.0],
        [4.0, 3.0, 4.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    );
    // Would keep y <= 0.3 only, but does not clip.
    let mut inert = ClipObject::plane([0.0, -1.0, 0.0], 0.3);
    inert.clip_geometry = false;

    BuiltScene {
        items: vec![ground, tube, cube],
        lighting: rigs::three_point(),
        clip: Some((vec![inert, lengthwise, across, trim], true)),
        ..Default::default()
    }
}

/// The section cap scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "section_caps",
        cameras: vec![NamedCamera {
            name: "cut",
            camera: orbit_camera(Vec3::new(0.0, 0.3, 0.9), 6.5, 1.1, 1.0),
        }],
        build: build_section_caps,
    }]
}
