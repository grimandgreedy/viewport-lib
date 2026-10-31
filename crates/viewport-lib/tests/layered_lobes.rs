//! Clear coat, sheen and anisotropy under a uniform environment: the sheen
//! keeps a white surface at or below the light it receives, and the
//! rasteriser and the path tracer agree on each lobe.
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

/// The sphere both renderers draw: no UVs, so the rasteriser builds its
/// tangent frame from the normal as the tracer does.
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

/// Mean linear radiance (the mean of the three channels) over the centre disc
/// of the rasterised sphere, lit only by a uniform environment of `radiance`.
fn raster_centre(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material: Material,
    radiance: f32,
) -> f32 {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let px = [radiance, radiance, radiance, 1.0].repeat(8 * 4);
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
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    let px = renderer.render_offscreen(device, queue, &frame, SIZE, SIZE);

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
    let mean = sum / n as f32;
    assert!(
        (0.04..0.72).contains(&mean),
        "outside the tone map's offset band: {mean}"
    );
    mean + 0.04
}

/// White furnace for the sheen: a white diffuse base under a white sheen at
/// every sheen roughness reflects at most the radiance it receives. The
/// sheen dims the base by the light it takes, so the sum stays at or below
/// what the base alone reflects.
#[test]
fn sheen_over_a_white_base_stays_within_the_furnace() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Half radiance keeps the disc inside the tone map's linear band.
    let radiance = 0.5;
    let base = Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.5);
    let plain = raster_centre(&device, &queue, base, radiance) / radiance;
    eprintln!("furnace base alone: {plain}");
    for roughness in [0.1, 0.3, 0.6, 1.0] {
        let sheen = base.with_sheen(Colour::linear_rgb(1.0, 1.0, 1.0), roughness);
        let mean = raster_centre(&device, &queue, sheen, radiance) / radiance;
        eprintln!("furnace sheen roughness {roughness}: {mean}");
        assert!(
            mean <= 1.01,
            "sheen roughness {roughness}: reflects {mean} of the furnace's radiance"
        );
    }
}

#[cfg(feature = "raytrace")]
mod parity {
    use super::*;
    use viewport_lib::raytrace::{RtCamera, RtMaterial, RtScene, RtSettings, Tracer};

    /// Samples per pixel for the tracer. The centre disc averages about 50
    /// pixels, so its mean settles to well under a percent.
    const SAMPLES: u32 = 1024;
    /// Allowed difference between the two renderers, relative to the raster
    /// reading. The rasteriser's split-sum environment term and its
    /// prefiltered lobes differ from the tracer's integral by a few percent
    /// on their own.
    const TOLERANCE: f32 = 0.06;

    /// Mean linear radiance over the centre disc of the traced sphere.
    fn traced_centre(material: RtMaterial, radiance: f32) -> f32 {
        let (device, queue) = device_queue().expect("device");
        let mesh = sphere();
        let positions: Vec<glam::Vec3> = mesh.positions.iter().map(|&p| p.into()).collect();
        let normals: Vec<glam::Vec3> = mesh.normals.iter().map(|&n| n.into()).collect();
        let mut scene = RtScene::new();
        scene.set_sky([0.0; 3], [0.0; 3]);
        scene.set_environment(&[radiance, radiance, radiance, 1.0].repeat(8 * 4), 8, 4);
        scene.add_mesh(&positions, &mesh.indices, Some(&normals), material);
        let cam = camera();
        let rt_cam = RtCamera {
            inv_view_proj: cam.view_proj_matrix().inverse(),
            position: cam.eye().as_vec3(),
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
                max_bounces: 2,
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

    fn check(name: &str, material: Material, rt: RtMaterial, radiance: f32) {
        let Some((device, queue)) = device_queue() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let raster = raster_centre(&device, &queue, material, radiance);
        let traced = traced_centre(rt, radiance);
        eprintln!("{name}: raster {raster}, traced {traced}");
        assert!(
            (raster - traced).abs() <= TOLERANCE * raster,
            "{name}: raster {raster} and tracer {traced} differ by more than {}%",
            TOLERANCE * 100.0
        );
    }

    /// A clear coat over a black base with no specular of its own, so the
    /// reading is the coat lobe alone.
    #[test]
    fn clearcoat_matches_the_tracer() {
        let mut m =
            Material::pbr(Colour::linear_rgb(0.0, 0.0, 0.0), 0.0, 0.5).with_clearcoat(1.0, 0.2);
        m.reflectance = 0.0;
        m.specular_factor = 0.0;
        let rt = RtMaterial {
            base_colour: Colour::linear_rgb(0.0, 0.0, 0.0),
            roughness: 0.5,
            reflectance: 0.0,
            specular_factor: 0.0,
            clearcoat: 1.0,
            clearcoat_roughness: 0.2,
            ..RtMaterial::default()
        };
        check("clear coat", m, rt, 8.0);
    }

    /// A sheen over a black base with no specular of its own: the reading is
    /// the sheen lobe's albedo times the radiance.
    #[test]
    fn sheen_matches_the_tracer() {
        let mut m = Material::pbr(Colour::linear_rgb(0.0, 0.0, 0.0), 0.0, 0.5)
            .with_sheen(Colour::linear_rgb(1.0, 1.0, 1.0), 1.0);
        m.reflectance = 0.0;
        m.specular_factor = 0.0;
        let rt = RtMaterial {
            base_colour: Colour::linear_rgb(0.0, 0.0, 0.0),
            roughness: 0.5,
            reflectance: 0.0,
            specular_factor: 0.0,
            sheen_colour: Colour::linear_rgb(1.0, 1.0, 1.0),
            sheen_roughness: 1.0,
            ..RtMaterial::default()
        };
        check("sheen", m, rt, 4.0);
    }

    /// An anisotropic white metal. The rasteriser's environment term is the
    /// isotropic split sum with a bent reflection, so it keeps the isotropic
    /// albedo where the tracer loses a little more as the lobe stretches;
    /// at this anisotropy the two stay inside the tolerance.
    #[test]
    fn anisotropy_matches_the_tracer() {
        let m =
            Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 1.0, 0.5).with_anisotropy(0.6, 0.3);
        let rt = RtMaterial {
            base_colour: Colour::linear_rgb(1.0, 1.0, 1.0),
            metallic: 1.0,
            roughness: 0.5,
            anisotropy: 0.6,
            anisotropy_rotation: 0.3,
            ..RtMaterial::default()
        };
        check("anisotropy", m, rt, 0.5);
    }
}
