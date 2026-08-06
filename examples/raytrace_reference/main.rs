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

    // Z-up scene: a ground slab with three spheres resting on it.
    let mut scene = RtScene::new();
    scene.set_sky([0.55, 0.72, 1.0], [0.25, 0.26, 0.28]);

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
