//! The rasteriser and the path tracer agree on one lit scene under the same
//! exposure.
//!
//! A grey card (a floor plane) with a sphere on it, under one directional
//! light, no image-based lighting, and a white sky and ground at a fill of
//! 0.55. The tracer's sky on ray miss is the same white at 0.55 nits, so both
//! renderers see the same no-environment fill. The raster frame draws on the
//! HDR path through Reinhard, which is inverted per pixel to recover the
//! exposed radiance before the operator.
//!
//! Each reading is the mean radiance over a floor region chosen in world
//! space: the lit region, in direct light well away from the sphere, and the
//! shadowed region, inside the sphere's umbra.
//!
//! Measured on Metal (M4), tracer at 512 samples and 8 bounces:
//!
//! | Exposure | Region | Raster | Tracer | Tracer / raster |
//! | --- | --- | --- | --- | --- |
//! | Passthrough | lit | 0.70026 | 0.70766 | 1.0106 |
//! | Passthrough | shadow | 0.27482 | 0.20880 | 0.7598 |
//! | EV 12 | lit | 1.42445e-4 | 1.43974e-4 | 1.0107 |
//! | EV 12 | shadow | 5.58527e-5 | 4.24811e-5 | 0.7606 |
//!
//! The lit card is direct light plus the fill, and the two agree to about a
//! percent: the tracer adds light bounced off the sphere and loses the part
//! of the sky the sphere hides. The shadowed floor does not agree and is not
//! meant to: the raster fill is the full 0.55 times the albedo with no
//! occlusion, while the tracer's floor sits under the sphere, which hides
//! much of its sky. The shadow is gated on its measured ratio only, so a
//! change to either fill shows. Under EV 12 each render is its Passthrough
//! render times the multiplier: exactly for the tracer, which draws the same
//! sample paths, and to half-float precision for the raster target.

use glam::{Vec2, Vec3};
use viewport_lib::raytrace::{RtCamera, RtLight, RtMaterial, RtScene, RtSettings, Tracer};
use viewport_lib::{
    Camera, CameraFrame, Colour, ExposureSettings, FrameData, LightSource, LightingSettings, Lux,
    Material, MeshData, MeshItem, PipelineMode, SceneFrame, ToneMapping,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 240;
const H: u32 = 180;
const FILL: f32 = 0.55;
const ALBEDO: f32 = 0.5;
const SPHERE_CENTRE: Vec3 = Vec3::new(0.0, 0.0, 1.0);
const LIGHT_DIR: [f32; 3] = [0.5, 0.3, 1.0];
const LIGHT_LUX: f32 = std::f32::consts::PI;

fn grey() -> Material {
    Material::pbr(Colour::linear_rgb(ALBEDO, ALBEDO, ALBEDO), 0.0, 1.0)
}

fn grey_rt() -> RtMaterial {
    RtMaterial {
        base_colour: Colour::linear_rgb(ALBEDO, ALBEDO, ALBEDO),
        metallic: 0.0,
        roughness: 1.0,
        ..RtMaterial::default()
    }
}

fn floor_mesh() -> MeshData {
    viewport_lib::geometry::primitives::plane(40.0, 40.0)
}

fn sphere_mesh() -> MeshData {
    viewport_lib::geometry::primitives::sphere(1.0, 96, 48)
}

fn camera() -> Camera {
    let mut cam = orbit_camera(Vec3::new(0.0, 0.0, 0.5), 9.0, 0.3, 0.7);
    cam.aspect = W as f32 / H as f32;
    cam
}

/// Which floor region a pixel's centre ray lands in.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Region {
    Lit,
    Shadow,
    Other,
}

/// Classify every pixel by where its centre ray meets the floor.
fn regions(cam: &Camera) -> Vec<Region> {
    let l = Vec3::from(LIGHT_DIR).normalize();
    let mut out = Vec::with_capacity((W * H) as usize);
    for py in 0..H {
        for px in 0..W {
            let ndc = Vec2::new(
                (px as f32 + 0.5) / W as f32 * 2.0 - 1.0,
                1.0 - (py as f32 + 0.5) / H as f32 * 2.0,
            );
            let (o, d) = cam.screen_ray(ndc);
            out.push(classify(o, d, l));
        }
    }
    out
}

fn classify(o: Vec3, d: Vec3, l: Vec3) -> Region {
    if d.z >= 0.0 {
        return Region::Other;
    }
    let t = -o.z / d.z;
    // Skip rays that meet the sphere before the floor.
    let oc = o - SPHERE_CENTRE;
    let b = oc.dot(d);
    let c = oc.length_squared() - 1.0;
    if b * b - c > 0.0 && -b - (b * b - c).sqrt() < t {
        return Region::Other;
    }
    let p = o + d * t;
    // Distance from the sphere centre to the shadow ray from `p`.
    let to_c = SPHERE_CENTRE - p;
    let along = to_c.dot(l);
    let miss = if along > 0.0 {
        (to_c - l * along).length()
    } else {
        f32::INFINITY
    };
    let r = Vec2::new(p.x, p.y).length();
    if miss < 0.6 {
        Region::Shadow
    } else if miss > 1.8 && (3.0..5.0).contains(&r) {
        Region::Lit
    } else {
        Region::Other
    }
}

/// Mean of the three colour channels over the pixels in `region`, with the
/// pixel count.
fn region_mean(px: &[[f32; 4]], regions: &[Region], region: Region) -> (f32, usize) {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (p, r) in px.iter().zip(regions) {
        if *r == region {
            sum += f64::from(p[0] + p[1] + p[2]) / 3.0;
            n += 1;
        }
    }
    ((sum / n.max(1) as f64) as f32, n)
}

/// The raster frame at `exposure`, read back as exposed radiance.
fn raster(h: &mut Harness, exposure: ExposureSettings, generation: u64) -> Vec<[f32; 4]> {
    let cam = camera();
    let mut items = Vec::new();
    for (mesh, model) in [
        (floor_mesh(), glam::Mat4::IDENTITY),
        (sphere_mesh(), glam::Mat4::from_translation(SPHERE_CENTRE)),
    ] {
        let mut item = MeshItem::default();
        item.mesh_id = h
            .renderer
            .resources_mut()
            .upload_mesh_data(&h.device, &mesh)
            .expect("mesh upload");
        item.material = h
            .renderer
            .resources_mut()
            .upload_material(grey())
            .expect("material upload");
        item.model = model.into();
        items.push(item);
    }
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&cam, [W as f32, H as f32]),
        SceneFrame::from_surface_items(items).with_generation(generation),
    );
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![LightSource::directional_lux(LIGHT_DIR, Lux(LIGHT_LUX))];
    lighting.sky_colour = Colour::WHITE;
    lighting.ground_colour = Colour::WHITE;
    lighting.hemisphere_intensity = FILL;
    frame.effects.lighting = lighting;
    frame.effects.display.mode = PipelineMode::Hdr;
    frame.effects.display.operator = ToneMapping::Reinhard;
    frame.effects.display.exposure = exposure;
    frame.viewport.background_colour = Some(Colour::BLACK);
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    let px = h.render_float(&frame, W, H);
    let inv = |y: f32| y / (1.0 - y);
    px.into_iter()
        .map(|p| [inv(p[0]), inv(p[1]), inv(p[2]), p[3]])
        .collect()
}

/// The traced frame at the exposure multiplier `k`.
fn traced(h: &Harness, k: f32) -> Vec<[f32; 4]> {
    let cam = camera();
    let mut scene = RtScene::new();
    scene.set_sky([FILL; 3], [FILL; 3]);
    for (mesh, offset) in [(floor_mesh(), Vec3::ZERO), (sphere_mesh(), SPHERE_CENTRE)] {
        let positions: Vec<Vec3> = mesh
            .positions
            .iter()
            .map(|&p| Vec3::from(p) + offset)
            .collect();
        let normals: Vec<Vec3> = mesh.normals.iter().map(|&n| n.into()).collect();
        scene.add_mesh(&positions, &mesh.indices, Some(&normals), grey_rt());
    }
    scene.add_light(RtLight::Directional {
        direction: LIGHT_DIR,
        colour: Colour::linear_rgb(LIGHT_LUX, LIGHT_LUX, LIGHT_LUX),
    });
    let mut tracer = Tracer::new(&h.device, &h.queue, &scene);
    tracer.set_exposure(k);
    let img = tracer.trace(
        &h.device,
        &h.queue,
        &RtCamera {
            inv_view_proj: cam.view_proj_matrix().inverse(),
            position: cam.eye().as_vec3(),
            width: W,
            height: H,
        },
        &RtSettings {
            samples: 512,
            max_bounces: 8,
            denoise: false,
            seed: 1,
        },
    );
    img.rgba
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

/// The linear multiplier of a manual EV100, as `ExposureSettings` resolves it.
fn multiplier(ev: f32) -> f32 {
    1.0 / (1.2 * 2f32.powf(ev))
}

/// Region means `(lit, shadow)` for the raster and the tracer at one exposure.
struct Reading {
    raster: (f32, f32),
    tracer: (f32, f32),
}

/// Read both renders: the raster under `exposure`, the tracer at its
/// multiplier `k`.
fn read(h: &mut Harness, exposure: ExposureSettings, k: f32, generation: u64) -> Reading {
    let regions = regions(&camera());
    let r = raster(h, exposure, generation);
    let t = traced(h, k);
    let (rl, nl) = region_mean(&r, &regions, Region::Lit);
    let (rs, ns) = region_mean(&r, &regions, Region::Shadow);
    let (tl, _) = region_mean(&t, &regions, Region::Lit);
    let (ts, _) = region_mean(&t, &regions, Region::Shadow);
    assert!(
        nl > 5000 && ns > 150,
        "regions too small: lit {nl}, shadow {ns}"
    );
    eprintln!(
        "{exposure:?}: lit {nl} px raster {rl} tracer {tl} ({}), shadow {ns} px raster {rs} tracer {ts} ({})",
        tl / rl,
        ts / rs
    );
    Reading {
        raster: (rl, rs),
        tracer: (tl, ts),
    }
}

/// Lit: the tracer within this fraction of the raster reading. Measured 1.1%;
/// the rest is room for another adapter's shadow filtering and sampling.
const LIT_TOLERANCE: f32 = 0.03;
/// Shadow: the tracer over the raster, measured 0.76, stays in this band.
/// Sample noise over the region is well under a percent, so a move out of it
/// is a change to the raster fill or the tracer's sky.
const SHADOW_RATIO: std::ops::Range<f32> = 0.70..0.83;

/// Under `Passthrough` the lit card agrees and the shadow keeps its measured
/// ratio.
#[test]
fn raster_and_tracer_agree_on_a_lit_card() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let r = read(&mut h, ExposureSettings::passthrough(), 1.0, 1);
    let lit = r.tracer.0 / r.raster.0;
    assert!(
        (lit - 1.0).abs() <= LIT_TOLERANCE,
        "lit card: tracer / raster is {lit}"
    );
    let shadow = r.tracer.1 / r.raster.1;
    assert!(
        SHADOW_RATIO.contains(&shadow),
        "shadow: tracer / raster is {shadow}, outside {SHADOW_RATIO:?}"
    );
}

/// Under `Manual { ev: 12 }` both renders are the `Passthrough` renders
/// scaled by the same multiplier, and the lit card still agrees.
#[test]
fn raster_and_tracer_scale_by_the_same_exposure() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let k = multiplier(12.0);
    let base = read(&mut h, ExposureSettings::passthrough(), 1.0, 1);
    let ev12 = read(&mut h, ExposureSettings::manual(12.0), k, 2);
    let checks = [
        ("raster lit", base.raster.0, ev12.raster.0, 0.005),
        ("raster shadow", base.raster.1, ev12.raster.1, 0.005),
        ("tracer lit", base.tracer.0, ev12.tracer.0, 1e-4),
        ("tracer shadow", base.tracer.1, ev12.tracer.1, 1e-4),
    ];
    for (label, unexposed, exposed, tol) in checks {
        let scale = exposed / unexposed;
        assert!(
            (scale / k - 1.0).abs() <= tol,
            "{label}: scaled by {scale}, expected {k}"
        );
    }
    let lit = ev12.tracer.0 / ev12.raster.0;
    assert!(
        (lit - 1.0).abs() <= LIT_TOLERANCE,
        "lit card at ev 12: tracer / raster is {lit}"
    );
}
