//! Path tracer functional tests.
//!
//! These exercise the tracer end to end on a real GPU device (headless), so
//! they validate the BVH build, traversal, intersection, shading, and readback
//! together, not just that it compiles. Run with:
//!   cargo test --features raytrace --test raytrace

#![cfg(feature = "raytrace")]

use glam::{Mat4, Vec3};
use viewport_lib::raytrace::{RtCamera, RtLight, RtMaterial, RtScene, RtSettings, trace};

/// Create a headless device/queue, or skip the test if no adapter is available.
fn device_queue() -> Option<(viewport_lib::gpu::Device, viewport_lib::gpu::Queue)> {
    let instance = viewport_lib::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &viewport_lib::gpu::RequestAdapterOptions {
            power_preference: viewport_lib::gpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        },
    ))
    .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&viewport_lib::gpu::DeviceDescriptor::default()))
            .ok()?;
    Some((device, queue))
}

/// A camera looking down -Y at the origin (Z-up), covering roughly [-1,1] in x/z.
fn camera(width: u32, height: u32) -> RtCamera {
    let eye = Vec3::new(0.0, -4.0, 0.0);
    let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Z);
    let aspect = width as f32 / height as f32;
    let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.1, 100.0);
    RtCamera {
        inv_view_proj: (proj * view).inverse(),
        position: eye,
        width,
        height,
    }
}

/// A single quad on the z=0 plane (Z-up), facing +Z, spanning [-s, s] in x and z.
fn add_quad(scene: &mut RtScene, s: f32, material: RtMaterial) {
    let p = [
        Vec3::new(-s, 0.0, -s),
        Vec3::new(s, 0.0, -s),
        Vec3::new(s, 0.0, s),
        Vec3::new(-s, 0.0, s),
    ];
    let n = [Vec3::new(0.0, -1.0, 0.0); 4]; // faces toward the camera (-Y)
    let idx = [0u32, 1, 2, 0, 2, 3];
    scene.add_mesh(&p, &idx, Some(&n), material);
}

fn mean_luma(img: &viewport_lib::raytrace::RtImage) -> f32 {
    let mut sum = 0.0f64;
    let px = (img.width * img.height) as usize;
    for i in 0..px {
        let r = img.rgba[i * 4] as f64;
        let g = img.rgba[i * 4 + 1] as f64;
        let b = img.rgba[i * 4 + 2] as f64;
        sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    (sum / px as f64) as f32
}

/// Every returned value must be finite (no NaN/Inf from bad PDFs or traversal).
#[test]
fn output_is_finite() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    add_quad(&mut scene, 1.5, RtMaterial::default());
    scene.add_light(RtLight::Directional {
        direction: [0.0, -0.4, 1.0],
        colour: [3.0, 3.0, 3.0],
    });
    let cam = camera(64, 64);
    let img = trace(
        &device,
        &queue,
        &scene,
        &cam,
        &RtSettings {
            samples: 16,
            max_bounces: 4,
        },
    );
    assert_eq!(img.rgba.len(), 64 * 64 * 4);
    assert!(
        img.rgba.iter().all(|v| v.is_finite()),
        "tracer produced non-finite values"
    );
}

/// An empty scene traces to a black image without touching the GPU path.
#[test]
fn empty_scene_is_black() {
    let Some((device, queue)) = device_queue() else {
        return;
    };
    let scene = RtScene::new();
    let cam = camera(16, 16);
    let img = trace(&device, &queue, &scene, &cam, &RtSettings::default());
    assert!(img.rgba.iter().all(|&v| v == 0.0));
}

/// A lit diffuse quad must come out brighter than the same quad unlit, and the
/// unlit-but-emissive quad must be brighter still. This exercises the two paths
/// that deposit radiance (NEE to the analytic light, and emissive-on-hit).
#[test]
fn lit_and_emissive_deposit_radiance() {
    let Some((device, queue)) = device_queue() else {
        return;
    };
    let cam = camera(48, 48);
    let settings = RtSettings {
        samples: 24,
        max_bounces: 3,
    };
    let black_sky = |s: &mut RtScene| s.set_sky([0.0; 3], [0.0; 3]);

    // Unlit: dark quad, no lights, black sky -> near black.
    let mut unlit = RtScene::new();
    black_sky(&mut unlit);
    add_quad(
        &mut unlit,
        1.5,
        RtMaterial {
            base_colour: [0.8, 0.8, 0.8],
            ..RtMaterial::default()
        },
    );
    let unlit_img = trace(&device, &queue, &unlit, &cam, &settings);

    // Lit: same quad under a directional light.
    let mut lit = RtScene::new();
    black_sky(&mut lit);
    add_quad(
        &mut lit,
        1.5,
        RtMaterial {
            base_colour: [0.8, 0.8, 0.8],
            ..RtMaterial::default()
        },
    );
    lit.add_light(RtLight::Directional {
        direction: [0.0, -0.3, 1.0],
        colour: [4.0, 4.0, 4.0],
    });
    let lit_img = trace(&device, &queue, &lit, &cam, &settings);

    // Emissive: bright quad, no lights.
    let mut emis = RtScene::new();
    black_sky(&mut emis);
    add_quad(
        &mut emis,
        1.5,
        RtMaterial {
            base_colour: [0.0; 3],
            emissive: [5.0, 5.0, 5.0],
            ..RtMaterial::default()
        },
    );
    let emis_img = trace(&device, &queue, &emis, &cam, &settings);

    let unlit_l = mean_luma(&unlit_img);
    let lit_l = mean_luma(&lit_img);
    let emis_l = mean_luma(&emis_img);

    assert!(
        lit_l > unlit_l + 0.05,
        "lit quad ({lit_l}) should be brighter than unlit ({unlit_l})"
    );
    assert!(
        emis_l > lit_l,
        "emissive quad ({emis_l}) should be brightest; lit was {lit_l}"
    );
}
