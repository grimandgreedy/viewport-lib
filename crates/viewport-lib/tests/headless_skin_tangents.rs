//! GPU skinning moves the tangent frame with the surface.
//!
//! A normal-mapped tube skinned rigidly to one joint must shade like the same
//! tube moved by the model matrix instead: the joint matrix and the model
//! matrix take different paths through the vertex shader (the skinning deformer
//! in object space, then the model transform), but a normal map reads the same
//! tangent frame from both.

use viewport_lib::Colour;
use viewport_lib::plugins::skinning::{SkinWeights, SkinningPlugin};
use viewport_lib::wgpu;

mod common;
use common::*;

const SIZE: u32 = 96;

/// Largest per-channel difference allowed between the two paths over the lit
/// region: both run the same f32 maths in a different order.
const TOLERANCE: u8 = 2;

/// An open tube along Z from -1 to 1 with radial normals, circumferential
/// tangents and UVs.
fn tube() -> MeshData {
    let sectors = 32u32;
    let rings = 8u32;
    let radius = 0.4f32;
    let mut mesh = MeshData::default();
    let mut tangents = Vec::new();
    let mut uvs = Vec::new();
    for r in 0..=rings {
        let z = -1.0 + 2.0 * r as f32 / rings as f32;
        for s in 0..=sectors {
            let a = std::f32::consts::TAU * s as f32 / sectors as f32;
            let (sin, cos) = a.sin_cos();
            mesh.positions.push([radius * cos, radius * sin, z]);
            mesh.normals.push([cos, sin, 0.0]);
            tangents.push([-sin, cos, 0.0, 1.0]);
            uvs.push([s as f32 / sectors as f32, r as f32 / rings as f32]);
        }
    }
    let row = sectors + 1;
    for r in 0..rings {
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
    mesh
}

struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: ViewportRenderer,
    skin: SkinningPlugin,
    skinned: MeshId,
    plain: MeshId,
    material: MaterialId,
    flat_material: MaterialId,
}

fn fixture() -> Option<Fixture> {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return None;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let skin = match SkinningPlugin::install(renderer.resources_mut(), &device) {
        Ok(skin) => skin,
        Err(err) => {
            eprintln!("skipping: this device cannot register the skinning deformer: {err}");
            return None;
        }
    };
    let mesh = tube();
    let skinned = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();
    let plain = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();
    let weights = SkinWeights {
        joint_indices: vec![[0, 0, 0, 0]; mesh.positions.len()],
        joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; mesh.positions.len()],
    };
    skin.attach_weights(renderer.resources_mut(), &device, skinned, &weights);

    // Every texel tilts the normal well toward the tangent, so a tangent frame
    // left in the bind pose shades visibly differently.
    let enc = |x: f32, y: f32, z: f32| {
        [
            ((x * 0.5 + 0.5) * 255.0).round() as u8,
            ((y * 0.5 + 0.5) * 255.0).round() as u8,
            ((z * 0.5 + 0.5) * 255.0).round() as u8,
            255,
        ]
    };
    let mut normal_map = |n: [f32; 3]| {
        let texels: Vec<u8> = std::iter::repeat_n(enc(n[0], n[1], n[2]), 4)
            .flatten()
            .collect();
        renderer
            .resources_mut()
            .upload_texture(
                &device,
                &queue,
                viewport_lib::TextureData::normal_map(2, 2, texels),
            )
            .unwrap()
    };
    let tilted = normal_map([0.6, 0.0, 0.8]);
    let flat = normal_map([0.0, 0.0, 1.0]);
    let mut material_with = |nm| {
        let mut mat = Material::default();
        mat.base_colour = Colour::linear_rgb(0.8, 0.8, 0.8);
        mat.normal_map_id = Some(nm);
        mat.normal_strength = 1.0;
        upload_material(&mut renderer, mat)
    };
    let material = material_with(tilted);
    let flat_material = material_with(flat);
    Some(Fixture {
        device,
        queue,
        renderer,
        skin,
        skinned,
        plain,
        material,
        flat_material,
    })
}

fn frame(items: Vec<SceneRenderItem>) -> FrameData {
    let cam = Camera::orbit(glam::DVec3::ZERO, 4.0, Camera::default().orientation());
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.4, -0.6, 0.7],
    };
    light.intensity = 3.0;
    // The shadow fit reads the skinned mesh's bind-pose bounds, so the two
    // paths would fit different cascades; shading alone is under test.
    light.cast_shadows = false;
    frame.effects.lighting.lights = vec![light];
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    frame
}

/// Renders the tube skinned by `joint` at an identity model, then the same
/// tube unskinned with `joint` as its model, and returns both images and an
/// empty frame for the background.
fn render_pair(fx: &mut Fixture, joint: glam::Mat4, material: MaterialId) -> [Vec<u8>; 3] {
    assert!(fx.skin.attach_palette(
        fx.renderer.resources_mut(),
        &fx.device,
        &fx.queue,
        fx.skinned,
        0,
        &[joint],
    ));
    let mut skinned = SceneRenderItem::default();
    skinned.mesh_id = fx.skinned;
    skinned.model = glam::Mat4::IDENTITY.into();
    skinned.material = material;
    skinned.deform_instance = Some(0);
    let mut plain = SceneRenderItem::default();
    plain.mesh_id = fx.plain;
    plain.model = joint.into();
    plain.material = material;

    let mut out = Vec::new();
    for items in [vec![skinned], vec![plain], vec![]] {
        let f = frame(items);
        out.push(
            fx.renderer
                .render_offscreen(&fx.device, &fx.queue, &f, SIZE, SIZE),
        );
    }
    [out.remove(0), out.remove(0), out.remove(0)]
}

/// The largest per-channel difference between `a` and `b` over the pixels
/// where either differs from `background`, and how many such pixels there are.
fn lit_region_delta(a: &[u8], b: &[u8], background: &[u8]) -> (u8, usize) {
    let mut max = 0u8;
    let mut count = 0usize;
    for ((pa, pb), pg) in a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .zip(background.chunks_exact(4))
    {
        if pa == pg && pb == pg {
            continue;
        }
        count += 1;
        for c in 0..3 {
            max = max.max(pa[c].abs_diff(pb[c]));
        }
    }
    (max, count)
}

fn assert_paths_match(label: &str, joint: glam::Mat4) {
    let Some(mut fx) = fixture() else {
        return;
    };
    let material = fx.material;
    let [skinned, plain, background] = render_pair(&mut fx, joint, material);
    let (delta, count) = lit_region_delta(&skinned, &plain, &background);
    println!("{label}: max channel delta {delta} over {count} lit pixels");
    assert!(count > 400, "{label}: the tube covers only {count} pixels");
    assert!(
        delta <= TOLERANCE,
        "{label}: skinned and model-transformed tubes differ by {delta} levels"
    );

    // The comparison means something only if the normal map moves the image.
    let flat_material = fx.flat_material;
    let [_, flat, _] = render_pair(&mut fx, joint, flat_material);
    let (map_delta, _) = lit_region_delta(&plain, &flat, &background);
    assert!(
        map_delta > 20,
        "{label}: the normal map changes the image by only {map_delta} levels"
    );
}

/// A joint rotated 90 degrees about X carries the whole tube: the normal map
/// must shade as on the tube rotated by its model matrix.
#[test]
fn rigid_skin_rotation_matches_the_model_rotation() {
    assert_paths_match(
        "rotation",
        glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
    );
}

/// A joint scaled (2, 1, 1), then tilted so the scale is not along the view,
/// must shade as the same matrix on the model: both take normals through the
/// cofactor and tangents through the matrix.
#[test]
fn rigid_skin_non_uniform_scale_matches_the_model_scale() {
    assert_paths_match(
        "scale",
        glam::Mat4::from_rotation_x(0.7) * glam::Mat4::from_scale(glam::Vec3::new(2.0, 1.0, 1.0)),
    );
}
