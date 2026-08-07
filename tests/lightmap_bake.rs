//! Lightmap GI-solve tests.
//!
//! These drive the texel ray-gen bake ([`bake_lightmap`]) on authored per-texel
//! surfaces (no unwrap needed) so the solve is validated on its own: direct
//! irradiance follows the cosine law, occluders cast shadows, indirect light
//! carries colour, and empty texels stay black. Run with:
//!   cargo test --features raytrace --test lightmap_bake
//!
//! Real GPU device (headless); skips if no adapter is available.

#![cfg(feature = "raytrace")]

use glam::Vec3;
use viewport_lib::raytrace::{
    RtImage, RtLight, RtMaterial, RtScene, RtSettings, TexelSurfaces, bake_lightmap,
    bake_lightmap_directional,
};

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

/// A floor quad on z=0 facing up (+Z), spanning [-s, s] in x and y.
fn add_floor(scene: &mut RtScene, s: f32, material: RtMaterial) {
    let p = [
        Vec3::new(-s, -s, 0.0),
        Vec3::new(s, -s, 0.0),
        Vec3::new(s, s, 0.0),
        Vec3::new(-s, s, 0.0),
    ];
    let n = [Vec3::Z; 4];
    let idx = [0u32, 1, 2, 0, 2, 3];
    scene.add_mesh(&p, &idx, Some(&n), material);
}

/// A `w` x `h` atlas whose every texel is the same covered surface point.
fn uniform_surfaces(
    w: u32,
    h: u32,
    pos: [f32; 3],
    normal: [f32; 3],
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let n = (w * h) as usize;
    let world_pos = vec![[pos[0], pos[1], pos[2], 1.0]; n];
    let world_normal = vec![[normal[0], normal[1], normal[2], 0.0]; n];
    (world_pos, world_normal)
}

/// Mean rgb over covered texels (alpha > 0.5).
fn mean_covered_rgb(img: &RtImage) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    let mut n = 0u64;
    let px = (img.width * img.height) as usize;
    for i in 0..px {
        if img.rgba[i * 4 + 3] > 0.5 {
            sum[0] += img.rgba[i * 4] as f64;
            sum[1] += img.rgba[i * 4 + 1] as f64;
            sum[2] += img.rgba[i * 4 + 2] as f64;
            n += 1;
        }
    }
    if n == 0 {
        return [0.0; 3];
    }
    [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64]
}

fn settings(samples: u32) -> RtSettings {
    RtSettings {
        samples,
        max_bounces: 4,
        denoise: false,
    }
}

/// A texel facing a directional light head-on receives irradiance equal to the
/// light radiance times the cosine (1 here), with a black sky killing indirect.
#[test]
fn direct_irradiance_follows_the_cosine_law() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.0; 3], [0.0; 3]);
    add_floor(&mut scene, 5.0, RtMaterial::default());
    scene.add_light(RtLight::Directional {
        direction: [0.0, 0.0, 1.0], // straight up, toward the light
        colour: [2.0, 2.0, 2.0],
    });

    let (pos, nrm) = uniform_surfaces(32, 32, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surf = TexelSurfaces {
        width: 32,
        height: 32,
        world_pos: &pos,
        world_normal: &nrm,
    };
    let img = bake_lightmap(&device, &queue, &scene, &surf, &settings(128));
    let m = mean_covered_rgb(&img);
    for c in m {
        assert!(
            (c - 2.0).abs() < 0.15,
            "expected ~2.0 irradiance, got {m:?}"
        );
    }
}

/// An occluder between the texel and the light drops direct irradiance to near
/// zero (only faint bounce off the shadowed occluder underside remains).
#[test]
fn occluder_casts_a_shadow() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // Black floor and occluder so the test isolates visibility: no surface
    // bounces light into the shadow, so a shadowed texel goes to ~0.
    let black = RtMaterial {
        base_colour: [0.0, 0.0, 0.0],
        ..RtMaterial::default()
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.0; 3], [0.0; 3]);
    add_floor(&mut scene, 5.0, black);
    scene.add_light(RtLight::Directional {
        direction: [0.0, 0.0, 1.0],
        colour: [2.0, 2.0, 2.0],
    });
    // A ceiling quad at z=1 over the texel, blocking the light.
    let p = [
        Vec3::new(-2.0, -2.0, 1.0),
        Vec3::new(2.0, -2.0, 1.0),
        Vec3::new(2.0, 2.0, 1.0),
        Vec3::new(-2.0, 2.0, 1.0),
    ];
    let n = [Vec3::Z; 4];
    scene.add_mesh(&p, &[0u32, 1, 2, 0, 2, 3], Some(&n), black);

    let (pos, nrm) = uniform_surfaces(32, 32, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surf = TexelSurfaces {
        width: 32,
        height: 32,
        world_pos: &pos,
        world_normal: &nrm,
    };
    let img = bake_lightmap(&device, &queue, &scene, &surf, &settings(128));
    let m = mean_covered_rgb(&img);
    for c in m {
        assert!(c < 0.15, "shadowed texel should be dark, got {m:?}");
    }
}

/// A red emissive panel above the texel bleeds red into the bake via the
/// hemisphere integral (indirect light carries colour). No analytic light, black
/// sky : the only illumination is the panel.
#[test]
fn indirect_light_carries_colour() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.0; 3], [0.0; 3]);
    add_floor(&mut scene, 5.0, RtMaterial::default());
    // A red emissive panel at z=1, over the texel.
    let p = [
        Vec3::new(-2.0, -2.0, 1.0),
        Vec3::new(2.0, -2.0, 1.0),
        Vec3::new(2.0, 2.0, 1.0),
        Vec3::new(-2.0, 2.0, 1.0),
    ];
    let n = [Vec3::Z; 4];
    let emissive = RtMaterial {
        emissive: [5.0, 0.0, 0.0],
        ..RtMaterial::default()
    };
    scene.add_mesh(&p, &[0u32, 1, 2, 0, 2, 3], Some(&n), emissive);

    let (pos, nrm) = uniform_surfaces(32, 32, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surf = TexelSurfaces {
        width: 32,
        height: 32,
        world_pos: &pos,
        world_normal: &nrm,
    };
    let img = bake_lightmap(&device, &queue, &scene, &surf, &settings(256));
    let m = mean_covered_rgb(&img);
    assert!(m[0] > 0.05, "expected red indirect, got {m:?}");
    assert!(
        m[0] > m[2] * 5.0 + 0.01,
        "indirect should be red-dominant, got {m:?}"
    );
}

/// Empty texels (coverage <= 0) are left black with zero alpha; covered texels
/// carry alpha 1.
#[test]
fn empty_texels_stay_black() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    add_floor(&mut scene, 5.0, RtMaterial::default());
    scene.add_light(RtLight::Directional {
        direction: [0.0, 0.0, 1.0],
        colour: [2.0, 2.0, 2.0],
    });

    // Left half covered, right half empty.
    let (w, h) = (8u32, 4u32);
    let mut world_pos = vec![[0.0, 0.0, 0.0, 0.0]; (w * h) as usize];
    let world_normal = vec![[0.0, 0.0, 1.0, 0.0]; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            if x < w / 2 {
                world_pos[(y * w + x) as usize] = [0.0, 0.0, 0.0, 1.0];
            }
        }
    }
    let surf = TexelSurfaces {
        width: w,
        height: h,
        world_pos: &world_pos,
        world_normal: &world_normal,
    };
    let img = bake_lightmap(&device, &queue, &scene, &surf, &settings(32));

    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let a = img.rgba[i * 4 + 3];
            if x < w / 2 {
                assert!(a > 0.5, "covered texel ({x},{y}) should have alpha 1");
                assert!(
                    img.rgba[i * 4] > 0.1,
                    "covered texel ({x},{y}) should be lit"
                );
            } else {
                assert_eq!(a, 0.0, "empty texel ({x},{y}) should have alpha 0");
                assert_eq!(
                    img.rgba[i * 4],
                    0.0,
                    "empty texel ({x},{y}) should be black"
                );
            }
        }
    }
}

/// The directional bake's dominant-direction atlas points back toward a single
/// light: a texel lit only from straight up reports a direction along +Z.
#[test]
fn directional_bake_points_at_the_light() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.0; 3], [0.0; 3]);
    add_floor(&mut scene, 5.0, RtMaterial::default());
    scene.add_light(RtLight::Directional {
        direction: [0.0, 0.0, 1.0],
        colour: [2.0, 2.0, 2.0],
    });

    let (pos, nrm) = uniform_surfaces(32, 32, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surf = TexelSurfaces {
        width: 32,
        height: 32,
        world_pos: &pos,
        world_normal: &nrm,
    };
    let bake = bake_lightmap_directional(&device, &queue, &scene, &surf, &settings(128));
    assert_eq!(bake.irradiance.len(), 32 * 32 * 4);
    assert_eq!(bake.direction.len(), 32 * 32 * 4);

    // Average the weighted-direction vectors over covered texels and normalise.
    let mut d = [0.0f64; 3];
    let mut n = 0u64;
    for px in bake.direction.chunks_exact(4) {
        if px[3] > 0.5 {
            d[0] += px[0] as f64;
            d[1] += px[1] as f64;
            d[2] += px[2] as f64;
            n += 1;
        }
    }
    assert!(n > 0);
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    assert!(len > 1e-6, "expected a non-zero dominant direction");
    let dz = d[2] / len;
    assert!(
        dz > 0.9,
        "dominant direction should point toward +Z, got {d:?}"
    );
}

/// A scene with no geometry bakes a black atlas of the requested size.
#[test]
fn empty_scene_bakes_black() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let scene = RtScene::new();
    let (pos, nrm) = uniform_surfaces(16, 16, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surf = TexelSurfaces {
        width: 16,
        height: 16,
        world_pos: &pos,
        world_normal: &nrm,
    };
    let img = bake_lightmap(&device, &queue, &scene, &surf, &settings(16));
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert!(img.rgba.iter().all(|&c| c == 0.0));
}
