//! Headless path-tracer reference render.
//!
//! Traces a small Z-up scene (a ground plane and three spheres: diffuse, clear
//! glass, and metal) with the offscreen path tracer and writes tone-mapped PPM
//! files so the result can be inspected without a window. It renders the same
//! low-sample frame with the denoiser off and on, plus a higher-sample
//! reference, so the transmission lobe and the denoiser are both visible.
//!
//! Usage:
//!   cargo run --release --example raytrace-reference --features raytrace
//!
//! Writes rt_noisy.ppm (16 spp), rt_denoised.ppm (16 spp + denoise), and
//! rt_reference.ppm (256 spp) to the current directory.

use glam::{Mat4, Vec3};
use viewport_lib::primitives;
use viewport_lib::raytrace::{
    RtCamera, RtLight, RtMaterial, RtScene, RtSettings, pick_backend, trace,
};

const W: u32 = 640;
const H: u32 = 400;

/// Add a `MeshData` primitive to the scene, translated to `origin`.
fn add_primitive(
    scene: &mut RtScene,
    mesh: &viewport_lib::MeshData,
    origin: Vec3,
    material: RtMaterial,
) {
    let positions: Vec<Vec3> = mesh
        .positions
        .iter()
        .map(|p| Vec3::from(*p) + origin)
        .collect();
    let normals: Vec<Vec3> = mesh.normals.iter().map(|n| Vec3::from(*n)).collect();
    scene.add_mesh(&positions, &mesh.indices, Some(&normals), material);
}

/// A procedural equirect HDR environment: a Z-up sky gradient with a bright sun
/// disc aligned to `sun_dir`, in the projection the tracer samples (longitude
/// around +Z, latitude with +Z at the top).
fn procedural_env(w: u32, h: u32, sun_dir: Vec3) -> Vec<f32> {
    use std::f32::consts::PI;
    let sun = sun_dir.normalize();
    let ground = Vec3::new(0.20, 0.18, 0.16);
    let horizon = Vec3::new(0.72, 0.78, 0.88);
    let zenith = Vec3::new(0.18, 0.40, 0.90);
    let mut px = vec![0.0f32; (w * h * 4) as usize];
    for y in 0..h {
        let v = (y as f32 + 0.5) / h as f32;
        let theta = (0.5 - v) * PI;
        let (st, ct) = theta.sin_cos();
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let phi = (u - 0.5) * 2.0 * PI;
            let dir = Vec3::new(ct * phi.cos(), ct * phi.sin(), st);
            let mut c = if dir.z >= 0.0 {
                horizon.lerp(zenith, dir.z.powf(0.6))
            } else {
                ground.lerp(horizon, (dir.z + 1.0).clamp(0.0, 1.0))
            };
            let d = dir.dot(sun).clamp(-1.0, 1.0);
            if d > 0.9995 {
                c = Vec3::splat(10.0);
            } else {
                c += Vec3::new(1.0, 0.85, 0.6) * d.max(0.0).powf(64.0) * 3.0;
            }
            let i = ((y * w + x) * 4) as usize;
            px[i] = c.x;
            px[i + 1] = c.y;
            px[i + 2] = c.z;
            px[i + 3] = 1.0;
        }
    }
    px
}

/// Reinhard tone map + gamma, HDR linear RGBA f32 -> 8-bit PPM (P6) bytes.
fn to_ppm(rgba: &[f32], w: u32, h: u32) -> Vec<u8> {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for px in rgba.chunks_exact(4) {
        for c in &px[..3] {
            let mapped = c / (1.0 + c); // Reinhard
            let gamma = mapped.powf(1.0 / 2.2);
            out.push((gamma.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    out
}

fn main() {
    let instance = viewport_lib::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &viewport_lib::gpu::RequestAdapterOptions {
            power_preference: viewport_lib::gpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        },
    ))
    .expect("no GPU adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&viewport_lib::gpu::DeviceDescriptor::default()))
            .expect("no device");

    println!("traversal backend: {:?}", pick_backend(&device));

    // Z-up scene: a ground slab with three spheres resting on it, lit by a
    // procedural equirect environment (image-based lighting) plus a key light.
    let mut scene = RtScene::new();
    let sun_dir = Vec3::new(0.3, -0.4, 0.85);
    scene.set_environment(&procedural_env(512, 256, sun_dir), 512, 256);

    let ground = primitives::cuboid(24.0, 24.0, 0.4);
    add_primitive(
        &mut scene,
        &ground,
        Vec3::new(0.0, 0.0, -0.2),
        RtMaterial {
            base_colour: [0.6, 0.6, 0.62],
            roughness: 0.9,
            ..RtMaterial::default()
        },
    );

    let sphere = primitives::sphere(1.0, 48, 24);
    // Diffuse red.
    add_primitive(
        &mut scene,
        &sphere,
        Vec3::new(-2.6, 0.0, 1.0),
        RtMaterial {
            base_colour: [0.75, 0.12, 0.10],
            roughness: 0.6,
            ..RtMaterial::default()
        },
    );
    // Clear glass (transmission lobe).
    add_primitive(
        &mut scene,
        &sphere,
        Vec3::new(0.0, 0.0, 1.0),
        RtMaterial {
            base_colour: [0.95, 0.98, 1.0],
            roughness: 0.05,
            transmission: 1.0,
            ior: 1.5,
            ..RtMaterial::default()
        },
    );
    // Polished metal.
    add_primitive(
        &mut scene,
        &sphere,
        Vec3::new(2.6, 0.0, 1.0),
        RtMaterial {
            base_colour: [0.95, 0.85, 0.55],
            metallic: 1.0,
            roughness: 0.15,
            ..RtMaterial::default()
        },
    );

    scene.add_light(RtLight::Directional {
        direction: [0.3, -0.4, 0.85],
        colour: [3.0, 2.9, 2.7],
    });

    println!("triangles: {}", scene.triangle_count());

    let eye = Vec3::new(0.0, -8.5, 3.6);
    let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.0, 1.0), Vec3::Z);
    let proj = Mat4::perspective_rh(42f32.to_radians(), W as f32 / H as f32, 0.1, 100.0);
    let camera = RtCamera {
        inv_view_proj: (proj * view).inverse(),
        position: eye,
        width: W,
        height: H,
    };

    let render = |samples: u32, denoise: bool, name: &str| {
        let img = trace(
            &device,
            &queue,
            &scene,
            &camera,
            &RtSettings {
                samples,
                max_bounces: 8,
                denoise,
                seed: 0,
            },
        );
        let path = format!("{name}.ppm");
        std::fs::write(&path, to_ppm(&img.rgba, img.width, img.height)).expect("write ppm");
        println!("wrote {path} ({samples} spp, denoise={denoise})");
    };

    render(16, false, "rt_noisy");
    render(16, true, "rt_denoised");
    render(256, false, "rt_reference");
}
