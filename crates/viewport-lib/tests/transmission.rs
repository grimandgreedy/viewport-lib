//! A glass sphere in front of an emissive backdrop, under a uniform
//! environment of a different radiance: the glass passes on the backdrop and
//! reflects a little of the environment, and the rasteriser's transmission
//! pass and the path tracer's dielectric lobe agree at the centre of the
//! sphere.
//!
//! Every reading is the mean radiance over a small disc at the centre of a
//! unit sphere, where N faces V. The rasteriser's output goes through the
//! default Khronos Neutral tone map, which subtracts 0.04 between linear 0.08
//! and 0.76; the scenes keep the disc inside that band and add the offset
//! back.

use viewport_lib::Colour;
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{EnvironmentBackground, EnvironmentLighting, EnvironmentOptions, TextureData};

const SIZE: u32 = 64;
/// Radius of the centre disc, in pixels.
const DISC: i32 = 4;
/// Radiance of the environment.
const RADIANCE: f32 = 0.5;
/// Radiance of the backdrop behind the sphere.
const BACKDROP: f32 = 0.3;
/// Share of the light smooth glass of IOR 1.5 reflects at normal incidence.
const REFLECTED: f32 = 0.04;

/// Allowed difference from [`expected`], in linear radiance: the specular
/// lobe's energy compensation and the 8-bit readback each move the reading
/// by a few thousandths.
const SLACK: f32 = 0.006;

/// What the centre of the sphere reads when the glass passes `transmitted`
/// of the backdrop: the reflected environment plus the rest.
fn expected(transmitted: f32) -> f32 {
    REFLECTED * RADIANCE + (1.0 - REFLECTED) * transmitted * BACKDROP
}

fn sphere() -> MeshData {
    let mut m = viewport_lib::primitives::sphere(1.0, 96, 48);
    m.uvs = None;
    m
}

fn camera() -> Camera {
    let mut cam = Camera::default();
    cam.aspect = 1.0;
    cam
}

/// A square facing the camera, three units behind the sphere, large enough
/// to fill the view behind it.
fn backdrop() -> (MeshData, glam::Mat4) {
    let to_eye = camera().eye_position().as_vec3().normalize();
    let model = glam::Mat4::from_rotation_translation(
        glam::Quat::from_rotation_arc(glam::Vec3::Z, to_eye),
        -to_eye * 3.0,
    );
    (viewport_lib::primitives::plane(40.0, 40.0), model)
}

/// An unlit emitter of `BACKDROP`: black, with no specular.
fn backdrop_material() -> Material {
    let mut m = Material::pbr(Colour::linear_rgb(0.0, 0.0, 0.0), 0.0, 1.0);
    m.reflectance = 0.0;
    m.specular_factor = 0.0;
    m.emissive = Colour::linear_rgb(BACKDROP, BACKDROP, BACKDROP);
    m.backface_policy = viewport_lib::BackfacePolicy::Identical;
    m
}

/// Mean of the three channels over the centre disc of an sRGB frame.
fn centre_mean(px: &[u8]) -> f32 {
    let decode = |v: u8| {
        let c = v as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let c = SIZE as i32 / 2;
    let mut sum = 0.0;
    let mut n = 0;
    for y in c - DISC..=c + DISC {
        for x in c - DISC..=c + DISC {
            if (x - c) * (x - c) + (y - c) * (y - c) <= DISC * DISC {
                let i = ((y as u32 * SIZE + x as u32) * 4) as usize;
                sum += (decode(px[i]) + decode(px[i + 1]) + decode(px[i + 2])) / 3.0;
                n += 1;
            }
        }
    }
    sum / n as f32
}

/// Mean linear radiance over the centre disc of the rasterised sphere.
fn raster_centre(device: &wgpu::Device, queue: &wgpu::Queue, material: Material) -> f32 {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let px = [RADIANCE, RADIANCE, RADIANCE, 1.0].repeat(8 * 4);
    let env = renderer
        .upload_environment(
            device,
            queue,
            TextureData::hdr(8, 4, px),
            EnvironmentOptions::default(),
        )
        .unwrap();
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &sphere())
        .unwrap();
    let (wall, wall_model) = backdrop();
    let wall = renderer
        .resources_mut()
        .upload_mesh_data(device, &wall)
        .unwrap();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&camera());
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame.viewport.environment_background = EnvironmentBackground::colour();
    frame.effects.environment = Some(EnvironmentLighting::new(env));
    frame.effects.lighting.lights = vec![];
    frame.effects.lighting.hemisphere_intensity = 0.0;
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::IDENTITY.into();
    item.material = upload_material(&mut renderer, material);
    let mut wall_item = SceneRenderItem::default();
    wall_item.mesh_id = wall;
    wall_item.model = wall_model.into();
    wall_item.material = upload_material(&mut renderer, backdrop_material());
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item, wall_item].into());
    let px = renderer.render_offscreen(device, queue, &frame, SIZE, SIZE);
    let mean = centre_mean(&px);
    assert!(
        (0.04..0.72).contains(&mean),
        "outside the tone map's offset band: {mean}"
    );
    mean + 0.04
}

/// Clear glass whose transmission tint and thickness are set by the caller.
fn glass(thickness: f32) -> Material {
    Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.0)
        .with_transmission(1.0, thickness)
        .with_ior(1.5)
}

/// Clear glass passes on the backdrop behind it, read from the opaque scene
/// copy: the light it does not reflect it transmits, thin or thick.
#[test]
fn clear_glass_passes_the_light_behind_it() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let want = expected(1.0);
    for thickness in [0.0, 2.0] {
        let mean = raster_centre(&device, &queue, glass(thickness));
        eprintln!("clear glass thickness {thickness}: {mean} (expected {want})");
        assert!(
            (mean - want).abs() <= SLACK,
            "thickness {thickness}: {mean} against {want}"
        );
    }
}

/// Absorption follows Beer-Lambert over the thickness: 0.6 of the light left
/// after one attenuation distance, 0.36 after two.
#[test]
fn absorbing_glass_follows_beer_lambert() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let grey = Colour::linear_rgb(0.6, 0.6, 0.6);
    for (distance, transmitted) in [(2.0, 0.6), (1.0, 0.36)] {
        let mean = raster_centre(&device, &queue, glass(2.0).with_attenuation(grey, distance));
        let want = expected(transmitted);
        eprintln!("absorbing glass, distance {distance}: {mean} (expected {want})");
        assert!(
            (mean - want).abs() <= SLACK,
            "distance {distance}: {mean} against {want}"
        );
    }
}

/// On a device with two bind groups there is no transmission pass: the
/// glass draws with the opaque scene, transmitting the environment instead
/// of the backdrop, and is not dropped.
#[test]
fn glass_without_the_pass_transmits_the_environment() {
    let Some((device, queue)) = headless_device_limited_bind_groups() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mean = raster_centre(&device, &queue, glass(2.0));
    eprintln!("two bind groups: {mean} (expected {RADIANCE})");
    assert!((mean - RADIANCE).abs() <= SLACK, "{mean}");
}

#[cfg(feature = "raytrace")]
mod parity {
    use super::*;
    use viewport_lib::raytrace::{RtCamera, RtMaterial, RtScene, RtSettings, Tracer};

    /// Samples per pixel for the tracer.
    const SAMPLES: u32 = 1024;
    /// Allowed difference between the two renderers, relative to the raster
    /// reading. The rasteriser takes one refraction and no internal
    /// reflections; the tracer follows every bounce.
    const TOLERANCE: f32 = 0.06;

    /// Mean linear radiance over the centre disc of the traced sphere.
    fn traced_centre(material: RtMaterial) -> f32 {
        let (device, queue) = device_queue().expect("device");
        let to_vec3 = |m: &MeshData, model: glam::Mat4| {
            let p: Vec<glam::Vec3> = m
                .positions
                .iter()
                .map(|&p| model.transform_point3(p.into()))
                .collect();
            let n: Vec<glam::Vec3> = m
                .normals
                .iter()
                .map(|&n| model.transform_vector3(n.into()).normalize())
                .collect();
            (p, n)
        };
        let mut scene = RtScene::new();
        scene.set_sky([0.0; 3], [0.0; 3]);
        scene.set_environment(&[RADIANCE, RADIANCE, RADIANCE, 1.0].repeat(8 * 4), 8, 4);
        let ball = sphere();
        let (p, n) = to_vec3(&ball, glam::Mat4::IDENTITY);
        scene.add_mesh(&p, &ball.indices, Some(&n), material);
        let (wall, model) = backdrop();
        let (p, n) = to_vec3(&wall, model);
        scene.add_mesh(
            &p,
            &wall.indices,
            Some(&n),
            RtMaterial {
                base_colour: Colour::linear_rgb(0.0, 0.0, 0.0),
                emissive: Colour::linear_rgb(BACKDROP, BACKDROP, BACKDROP),
                reflectance: 0.0,
                specular_factor: 0.0,
                ..RtMaterial::default()
            },
        );
        let cam = camera();
        let rt_cam = RtCamera {
            inv_view_proj: cam.view_proj_matrix().inverse(),
            position: cam.eye_position().as_vec3(),
            width: SIZE,
            height: SIZE,
        };
        let mut tracer = Tracer::new(&device, &queue, &scene);
        let img = tracer.trace(
            &device,
            &queue,
            &rt_cam,
            &RtSettings {
                samples: SAMPLES,
                max_bounces: 8,
                denoise: false,
                seed: 7,
            },
        );
        let c = SIZE as i32 / 2;
        let mut sum = 0.0;
        let mut n = 0;
        for y in c - DISC..=c + DISC {
            for x in c - DISC..=c + DISC {
                if (x - c) * (x - c) + (y - c) * (y - c) <= DISC * DISC {
                    let i = ((y as u32 * SIZE + x as u32) * 4) as usize;
                    sum += (img.rgba[i] + img.rgba[i + 1] + img.rgba[i + 2]) / 3.0;
                    n += 1;
                }
            }
        }
        sum / n as f32
    }

    fn rt_glass(attenuation: Colour, distance: f32) -> RtMaterial {
        RtMaterial {
            base_colour: Colour::linear_rgb(1.0, 1.0, 1.0),
            roughness: 0.0,
            transmission: 1.0,
            ior: 1.5,
            attenuation_colour: attenuation,
            attenuation_distance: distance,
            ..RtMaterial::default()
        }
    }

    /// Smooth clear glass, thin-walled in the rasteriser and solid in the
    /// tracer: at the centre both see the backdrop through it, so the
    /// bending does not change the reading.
    #[test]
    fn clear_glass_matches_the_tracer() {
        let Some((device, queue)) = device_queue() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let raster = raster_centre(&device, &queue, glass(0.0));
        let traced = traced_centre(rt_glass(Colour::linear_rgb(1.0, 1.0, 1.0), f32::MAX));
        eprintln!("clear glass: raster {raster}, traced {traced}");
        assert!(
            (raster - traced).abs() <= TOLERANCE * raster,
            "raster {raster} and tracer {traced} differ by more than {}%",
            TOLERANCE * 100.0
        );
    }

    /// Absorbing glass at the centre, where the tracer's path through the
    /// sphere is its diameter, the rasteriser's thickness.
    #[test]
    fn absorbing_glass_matches_the_tracer() {
        let Some((device, queue)) = device_queue() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let grey = Colour::linear_rgb(0.6, 0.6, 0.6);
        let raster = raster_centre(&device, &queue, glass(2.0).with_attenuation(grey, 2.0));
        let traced = traced_centre(rt_glass(grey, 2.0));
        eprintln!("absorbing glass: raster {raster}, traced {traced}");
        assert!(
            (raster - traced).abs() <= TOLERANCE * raster,
            "raster {raster} and tracer {traced} differ by more than {}%",
            TOLERANCE * 100.0
        );
    }
}
