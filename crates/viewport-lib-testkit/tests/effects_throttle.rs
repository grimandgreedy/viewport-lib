//! A throttled frame must composite as if the throttled effects were off.
//!
//! Under load the degradation ladder skips the SSAO, contact shadow, bloom and
//! depth of field producers. Their per-viewport targets then still hold the
//! last frame that ran them, so a composite that keeps reading them shows a
//! stale image: with depth of field on, the whole scene freezes on the frame
//! before the throttle while the camera keeps moving.

use viewport_lib::Colour;
use viewport_lib::{
    CameraFrame, FrameData, Material, MaterialId, MeshId, PerformancePolicy, SceneFrame,
    SceneRenderItem,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

const W: u32 = 160;
const H: u32 = 120;

/// Spheres spread along the view so depth of field blurs some and not others,
/// in a bright material so bloom has something to spread.
fn frame(mesh: MeshId, material: MaterialId, yaw: f32, effects: bool) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 8.0, yaw, 1.1);
    let items = [-3.0f32, 0.0, 3.0]
        .iter()
        .map(|&x| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh;
            item.material = material;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(x, x * 0.5, 0.0)).into();
            item
        })
        .collect();
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(items),
    );
    fd.viewport.show_axes_indicator = false;
    let pp = &mut fd.effects.post_process;
    pp.dof.enabled = effects;
    pp.dof.focal_distance = 8.0;
    pp.dof.focal_range = 0.5;
    pp.dof.max_blur_radius = 6.0;
    pp.bloom.enabled = effects;
    pp.bloom.threshold = 0.2;
    pp.bloom.intensity = 0.5;
    fd
}

fn throttling_policy() -> PerformancePolicy {
    PerformancePolicy {
        // No frame can meet this budget, so every frame misses and the tier
        // climbs one step per frame until effects throttle.
        target_fps: Some(1.0e9),
        allow_dynamic_resolution: false,
        allow_effect_throttling: true,
        ..PerformancePolicy::default()
    }
}

#[test]
fn a_throttled_frame_composites_without_the_throttled_effects() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mesh = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::stress_sphere(1.0, 3).into())
        .expect("mesh upload");
    let material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.9, 0.8, 0.6)))
        .expect("material upload");

    let scene_a = frame(mesh, material, 0.3, true);
    let scene_b = frame(mesh, material, 1.4, true);
    let scene_b_plain = frame(mesh, material, 1.4, false);

    // Scene A with the effects running, until the throttle engages. The last
    // unthrottled frame is the image a stale composite would keep showing.
    h.renderer.set_performance_policy(throttling_policy());
    let mut a_with_effects = None;
    let mut throttled = false;
    for _ in 0..16 {
        let pixels = h.render(&scene_a, W, H);
        if h.stats().effects_throttled {
            throttled = true;
            break;
        }
        a_with_effects = Some(pixels);
    }
    assert!(throttled, "test premise: the effects throttle must engage");
    let a_with_effects = a_with_effects.expect("test premise: one frame before the throttle");

    // Scene B, camera moved, effects requested but throttled.
    let b_throttled = h.render(&scene_b, W, H);
    assert!(
        h.stats().effects_throttled,
        "test premise: scene B must render throttled"
    );

    // Scene B with the effects off, once the ladder has stepped back down.
    h.renderer
        .set_performance_policy(PerformancePolicy::default());
    let mut b_plain = None;
    for _ in 0..16 {
        let pixels = h.render(&scene_b_plain, W, H);
        if !h.stats().effects_throttled {
            b_plain = Some(pixels);
            break;
        }
    }
    let b_plain = b_plain.expect("test premise: the throttle must release");

    assert_ne!(
        a_with_effects, b_plain,
        "test premise: the two scenes must differ"
    );
    // The bypassed effects bind their neutral placeholders, so the composite
    // reduces to exactly the effects-off frame.
    let differing = b_throttled
        .iter()
        .zip(&b_plain)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "a throttled frame must equal the frame with the throttled effects off"
    );
    assert_ne!(
        b_throttled, a_with_effects,
        "a throttled frame must not show the last unthrottled depth of field image"
    );
}
