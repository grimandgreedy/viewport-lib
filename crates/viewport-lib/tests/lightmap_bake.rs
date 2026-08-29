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
    bake_lightmap_directional, bake_shadowmask,
};

mod common;
use common::device_queue;

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

/// Coefficient of variation (std / mean) of luminance over covered texels. When
/// every texel shades the same surface point, this is the per-estimate noise.
fn coeff_of_variation(img: &RtImage) -> f64 {
    let mut vals = Vec::new();
    let px = (img.width * img.height) as usize;
    for i in 0..px {
        if img.rgba[i * 4 + 3] > 0.5 {
            let l = 0.2126 * img.rgba[i * 4] as f64
                + 0.7152 * img.rgba[i * 4 + 1] as f64
                + 0.0722 * img.rgba[i * 4 + 2] as f64;
            vals.push(l);
        }
    }
    if vals.is_empty() {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    if mean <= 1e-9 {
        return 0.0;
    }
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
    var.sqrt() / mean
}

fn settings(samples: u32) -> RtSettings {
    RtSettings {
        samples,
        max_bounces: 4,
        denoise: false,
        seed: 0,
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

/// A small bright emissive panel is found by area-light next-event estimation, not
/// just chance BSDF bounces, so it lights a diffuse floor with low noise even at a
/// few samples - and the low-sample mean matches the converged mean (unbiased). The
/// panel is the only light (black sky, no analytic lights), so this isolates the
/// emissive NEE path.
#[test]
fn emissive_panel_lights_the_floor_with_low_noise() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.0; 3], [0.0; 3]);
    add_floor(&mut scene, 10.0, RtMaterial::default());
    // A 1x1 bright emissive panel two units above the origin. Small and distant
    // enough that cosine-sampled bounces rarely find it: BSDF-only would be very
    // noisy, area-light NEE is not.
    let s = 0.5;
    let z = 2.0;
    let panel = [
        Vec3::new(-s, -s, z),
        Vec3::new(s, -s, z),
        Vec3::new(s, s, z),
        Vec3::new(-s, s, z),
    ];
    scene.add_mesh(
        &panel,
        &[0u32, 1, 2, 0, 2, 3],
        Some(&[-Vec3::Z; 4]),
        RtMaterial {
            emissive: [40.0, 40.0, 40.0],
            ..RtMaterial::default()
        },
    );

    let (pos, nrm) = uniform_surfaces(32, 32, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surf = TexelSurfaces {
        width: 32,
        height: 32,
        world_pos: &pos,
        world_normal: &nrm,
    };
    let low = bake_lightmap(&device, &queue, &scene, &surf, &settings(16));
    let hi = bake_lightmap(&device, &queue, &scene, &surf, &settings(256));

    let ml = mean_covered_rgb(&low);
    let mh = mean_covered_rgb(&hi);
    assert!(ml[0] > 0.02, "floor should be lit by the panel, got {ml:?}");
    // Unbiased: the few-sample mean agrees with the converged mean.
    for c in 0..3 {
        let rel = (ml[c] - mh[c]).abs() / mh[c].max(1e-6);
        assert!(
            rel < 0.15,
            "emissive NEE should be unbiased: 16-spp {ml:?} vs 256-spp {mh:?}"
        );
    }
    // Low noise: every texel is the same point, so the spread across texels is the
    // per-estimate noise. NEE keeps it small at 16 spp.
    let cov = coeff_of_variation(&low);
    assert!(
        cov < 0.3,
        "emissive NEE should be low-noise at 16 spp, got CoV = {cov}"
    );
}

/// The shadowmask bake writes per-light static-occluder visibility into the RGBA
/// channels: a texel shadowed from light 0 reads ~0 in the red channel, an
/// unoccluded texel reads ~1, and channels without a light stay lit (1).
#[test]
fn shadowmask_bakes_per_light_visibility() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.0; 3], [0.0; 3]);
    add_floor(&mut scene, 20.0, RtMaterial::default());
    // An occluder 2 units above the origin, blocking the straight-up light there.
    let occ = [
        Vec3::new(-1.5, -1.5, 2.0),
        Vec3::new(1.5, -1.5, 2.0),
        Vec3::new(1.5, 1.5, 2.0),
        Vec3::new(-1.5, 1.5, 2.0),
    ];
    scene.add_mesh(
        &occ,
        &[0u32, 1, 2, 0, 2, 3],
        Some(&[Vec3::Z; 4]),
        RtMaterial::default(),
    );
    scene.add_light(RtLight::Directional {
        direction: [0.0, 0.0, 1.0], // straight up, toward the light
        colour: [1.0, 1.0, 1.0],
    });

    let bake_at = |p: [f32; 3]| {
        let (pos, nrm) = uniform_surfaces(16, 16, p, [0.0, 0.0, 1.0]);
        let surf = TexelSurfaces {
            width: 16,
            height: 16,
            world_pos: &pos,
            world_normal: &nrm,
        };
        bake_shadowmask(&device, &queue, &scene, &surf, &settings(64))
    };
    let channel_mean = |img: &RtImage, c: usize| {
        let mut s = 0.0f64;
        let n = (img.width * img.height) as usize;
        for px in img.rgba.chunks_exact(4) {
            s += px[c] as f64;
        }
        (s / n as f64) as f32
    };

    let shadowed = bake_at([0.0, 0.0, 0.0]); // under the occluder
    let lit = bake_at([6.0, 0.0, 0.0]); // in the open, past the occluder

    assert!(
        channel_mean(&shadowed, 0) < 0.1,
        "occluded texel should read shadowed in light 0's channel, got {}",
        channel_mean(&shadowed, 0)
    );
    assert!(
        channel_mean(&lit, 0) > 0.9,
        "open texel should read lit in light 0's channel, got {}",
        channel_mean(&lit, 0)
    );
    // Only one light in the scene, so channel 1 (no light) stays fully lit.
    assert!(
        (channel_mean(&lit, 1) - 1.0).abs() < 0.01,
        "an unmapped light channel should stay lit (1), got {}",
        channel_mean(&lit, 1)
    );
}

/// The bake is deterministic: baking the same scene and surfaces twice with the
/// same settings produces a bit-for-bit identical atlas. This is the contract the
/// lightmapper's regression tests rely on - a change that perturbs the output is a
/// real change to diff, not RNG drift between runs. The scene mixes a sky, a
/// direct light, and an occluder so the stochastic parts (shadow rays, cosine-
/// sampled indirect, environment) are all exercised.
#[test]
fn bake_is_bit_for_bit_reproducible() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.1, 0.12, 0.16], [0.2, 0.24, 0.32]);
    add_floor(&mut scene, 5.0, RtMaterial::default());
    scene.add_light(RtLight::Directional {
        direction: [0.3, 0.2, 0.9],
        colour: [2.0, 1.9, 1.7],
    });
    // A small occluder above the texels, so shadow rays actually miss/hit.
    let occ = [
        Vec3::new(-1.0, -1.0, 1.5),
        Vec3::new(1.0, -1.0, 1.5),
        Vec3::new(1.0, 1.0, 1.5),
        Vec3::new(-1.0, 1.0, 1.5),
    ];
    scene.add_mesh(
        &occ,
        &[0u32, 1, 2, 0, 2, 3],
        Some(&[Vec3::Z; 4]),
        RtMaterial::default(),
    );

    let (pos, nrm) = uniform_surfaces(48, 48, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surf = TexelSurfaces {
        width: 48,
        height: 48,
        world_pos: &pos,
        world_normal: &nrm,
    };
    let a = bake_lightmap(&device, &queue, &scene, &surf, &settings(64));
    let b = bake_lightmap(&device, &queue, &scene, &surf, &settings(64));
    assert_eq!(
        a.rgba, b.rgba,
        "two bakes of the same scene must be bit-for-bit identical"
    );
}

/// The seed knob draws an independent sample stream: two different seeds give
/// different per-texel noise (so the determinism above is a fixed stream, not a
/// stuck RNG) but converge to the same signal (both are unbiased estimators).
#[test]
fn different_seeds_vary_the_noise_but_not_the_signal() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut scene = RtScene::new();
    scene.set_sky([0.2, 0.24, 0.3], [0.4, 0.44, 0.52]);
    add_floor(&mut scene, 5.0, RtMaterial::default());
    scene.add_light(RtLight::Directional {
        direction: [0.2, 0.1, 0.95],
        colour: [1.5, 1.5, 1.5],
    });

    let (pos, nrm) = uniform_surfaces(32, 32, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surf = TexelSurfaces {
        width: 32,
        height: 32,
        world_pos: &pos,
        world_normal: &nrm,
    };

    let s0 = RtSettings {
        samples: 32,
        max_bounces: 4,
        denoise: false,
        seed: 0,
    };
    let s1 = RtSettings { seed: 1, ..s0 };
    let a = bake_lightmap(&device, &queue, &scene, &surf, &s0);
    let b = bake_lightmap(&device, &queue, &scene, &surf, &s1);

    // Different stream: the two atlases must not be identical.
    assert_ne!(a.rgba, b.rgba, "a different seed should change the noise");

    // Same signal: the means agree within Monte-Carlo error at this sample count.
    let ma = mean_covered_rgb(&a);
    let mb = mean_covered_rgb(&b);
    for c in 0..3 {
        assert!(
            (ma[c] - mb[c]).abs() < 0.05,
            "seeds should converge to the same mean, got {ma:?} vs {mb:?}"
        );
    }
}
