//! Path tracer functional tests.
//!
//! These exercise the tracer end to end on a real GPU device (headless), so
//! they validate the BVH build, traversal, intersection, shading, and readback
//! together, not just that it compiles. Run with:
//!   cargo test --features raytrace --test raytrace

#![cfg(feature = "raytrace")]

use glam::{Mat4, Vec3};
use viewport_lib::raytrace::{
    RtBackend, RtCamera, RtLight, RtMaterial, RtScene, RtSettings, Tracer, pick_backend, trace,
};

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
            denoise: false,
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
        denoise: false,
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

/// Mean squared luminance difference between horizontally adjacent pixels : a
/// simple measure of per-pixel noise that a denoiser should reduce.
fn neighbour_variance(img: &viewport_lib::raytrace::RtImage) -> f32 {
    let w = img.width as usize;
    let h = img.height as usize;
    let luma = |i: usize| -> f64 {
        0.2126 * img.rgba[i * 4] as f64
            + 0.7152 * img.rgba[i * 4 + 1] as f64
            + 0.0722 * img.rgba[i * 4 + 2] as f64
    };
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
    for y in 0..h {
        for x in 0..w - 1 {
            let a = luma(y * w + x);
            let b = luma(y * w + x + 1);
            sum += (a - b) * (a - b);
            n += 1.0;
        }
    }
    (sum / n) as f32
}

/// A clear (fully transmissive) quad in front of the sky must let the sky
/// through, coming out much brighter than the same quad made opaque and black.
/// Exercises the refraction lobe end to end.
#[test]
fn transmission_lets_the_background_through() {
    let Some((device, queue)) = device_queue() else {
        return;
    };
    let cam = camera(48, 48);
    let settings = RtSettings {
        samples: 32,
        max_bounces: 4,
        denoise: false,
    };

    // Clear glass with no bending (ior = 1) so rays pass straight to the sky.
    let mut clear = RtScene::new();
    add_quad(
        &mut clear,
        4.0,
        RtMaterial {
            base_colour: [1.0, 1.0, 1.0],
            roughness: 0.05,
            transmission: 1.0,
            ior: 1.0,
            ..RtMaterial::default()
        },
    );
    let clear_l = mean_luma(&trace(&device, &queue, &clear, &cam, &settings));

    // Opaque black quad, no lights: blocks the sky behind it.
    let mut opaque = RtScene::new();
    add_quad(
        &mut opaque,
        4.0,
        RtMaterial {
            base_colour: [0.0, 0.0, 0.0],
            ..RtMaterial::default()
        },
    );
    let opaque_l = mean_luma(&trace(&device, &queue, &opaque, &cam, &settings));

    assert!(
        clear_l > opaque_l + 0.1,
        "clear quad ({clear_l}) should pass the sky through; opaque was {opaque_l}"
    );
}

/// The denoiser must lower per-pixel noise on a low-sample image while keeping
/// the overall brightness roughly the same. A rough metal quad lit only by the
/// sky gradient produces GGX-sampling noise at a few samples per pixel.
#[test]
fn denoise_reduces_noise_and_preserves_mean() {
    let Some((device, queue)) = device_queue() else {
        return;
    };
    let cam = camera(64, 64);
    let mut scene = RtScene::new();
    add_quad(
        &mut scene,
        4.0,
        RtMaterial {
            base_colour: [0.9, 0.9, 0.9],
            metallic: 1.0,
            roughness: 0.6,
            ..RtMaterial::default()
        },
    );

    let raw = trace(
        &device,
        &queue,
        &scene,
        &cam,
        &RtSettings {
            samples: 4,
            max_bounces: 3,
            denoise: false,
        },
    );
    let denoised = trace(
        &device,
        &queue,
        &scene,
        &cam,
        &RtSettings {
            samples: 4,
            max_bounces: 3,
            denoise: true,
        },
    );

    let raw_var = neighbour_variance(&raw);
    let den_var = neighbour_variance(&denoised);
    let raw_mean = mean_luma(&raw);
    let den_mean = mean_luma(&denoised);

    assert!(
        den_var < raw_var * 0.8,
        "denoise should cut neighbour variance: raw {raw_var}, denoised {den_var}"
    );
    assert!(
        (den_mean - raw_mean).abs() < 0.1 * raw_mean + 0.02,
        "denoise should preserve mean luma: raw {raw_mean}, denoised {den_mean}"
    );
}

/// A persistent `Tracer` reused across two cameras and a resolution change must
/// keep producing correct images : the second camera sees the lit quad, and a
/// smaller re-trace returns the smaller buffer without stale data. Guards the
/// pipeline/scene caching and the size-dependent buffer reallocation.
#[test]
fn tracer_reuse_across_cameras_and_sizes() {
    let Some((device, queue)) = device_queue() else {
        return;
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.0; 3], [0.0; 3]);
    add_quad(
        &mut scene,
        1.5,
        RtMaterial {
            base_colour: [0.8, 0.8, 0.8],
            ..RtMaterial::default()
        },
    );
    scene.add_light(RtLight::Directional {
        direction: [0.0, -0.3, 1.0],
        colour: [4.0, 4.0, 4.0],
    });

    let settings = RtSettings {
        samples: 16,
        max_bounces: 3,
        denoise: false,
    };

    let mut tracer = Tracer::new(&device, &scene);

    // First trace at 48x48.
    let a = tracer.trace(&device, &queue, &camera(48, 48), &settings);
    assert_eq!(a.rgba.len(), 48 * 48 * 4);
    assert!(mean_luma(&a) > 0.01, "lit quad should not be black");

    // Re-trace at a different resolution: buffers reallocate, output resizes.
    let b = tracer.trace(&device, &queue, &camera(80, 32), &settings);
    assert_eq!(b.rgba.len(), 80 * 32 * 4);
    assert!(b.rgba.iter().all(|v| v.is_finite()));
    assert!(
        mean_luma(&b) > 0.01,
        "reused tracer should still light the quad"
    );
}

/// Without the `raytrace-hardware` feature the tracer always reports the
/// portable compute backend, regardless of adapter capabilities.
#[test]
fn pick_backend_defaults_to_software() {
    let Some((device, _queue)) = device_queue() else {
        return;
    };
    let backend = pick_backend(&device);
    if cfg!(feature = "raytrace-hardware") {
        // Either is valid depending on the adapter; just ensure it runs.
        assert!(matches!(backend, RtBackend::Software | RtBackend::Hardware));
    } else {
        assert_eq!(backend, RtBackend::Software);
    }
}
