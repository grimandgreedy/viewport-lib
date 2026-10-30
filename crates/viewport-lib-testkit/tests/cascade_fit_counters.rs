//! The directional cascades cover the view depths where the scene's casters
//! can throw a shadow, taken from the mesh instance bounds. These pin when
//! the caster and scene bounds are rebuilt, and that the range follows the
//! scene rather than the camera's orbit radius.

use glam::{Quat, Vec3};
use viewport_lib::Colour;
use viewport_lib::wgpu;
use viewport_lib::{
    Camera, CameraFrame, CascadeSplit, FrameData, FrameStats, LightKind, LightSource,
    LightingSettings, Material, SceneFrame, SceneRenderItem,
};
use viewport_lib_testkit::device::DeviceProfile;
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 160;
const H: u32 = 120;

/// A 30 m ground slab and a cube on it, uploaded once.
fn scene(h: &mut Harness) -> Vec<SceneRenderItem> {
    let resources = h.renderer.resources_mut();
    let ground_mesh = resources
        .upload_mesh_data(
            &h.device,
            &viewport_lib::primitives::cuboid(30.0, 30.0, 0.2),
        )
        .expect("ground");
    let cube_mesh = resources
        .upload_mesh_data(&h.device, &viewport_lib::primitives::cube(1.0))
        .expect("cube");
    let material = resources
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material");
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = ground_mesh;
    ground.model = glam::Mat4::from_translation(Vec3::new(0.0, 0.0, -0.1)).to_cols_array_2d();
    ground.material = material;
    let mut cube = SceneRenderItem::default();
    cube.mesh_id = cube_mesh;
    cube.model = glam::Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5)).to_cols_array_2d();
    cube.material = material;
    vec![ground, cube]
}

fn frame(items: &[SceneRenderItem], camera: &Camera, generation: u64) -> FrameData {
    let mut fd = FrameData::new(
        CameraFrame::from_camera(camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(items.to_vec()).with_generation(generation),
    );
    // The default lighting: one casting directional light.
    fd.effects.lighting = LightingSettings::default();
    fd
}

fn render(h: &mut Harness, fd: &FrameData) -> FrameStats {
    let _ = h.render(fd, W, H);
    h.stats()
}

#[test]
fn bounds_rebuild_only_when_the_instance_bounds_change() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut items = scene(&mut h);
    let camera = orbit_camera(Vec3::ZERO, 12.0, 0.4, 0.5);

    let s = render(&mut h, &frame(&items, &camera, 1));
    assert_eq!(s.shadow_cascade_passes, 4);
    assert_eq!(
        s.shadow_bounds_rebuilds, 1,
        "the first frame builds the bounds"
    );

    let s = render(&mut h, &frame(&items, &camera, 1));
    assert_eq!(s.shadow_bounds_rebuilds, 0, "a repeat frame reuses them");

    // A new generation with the same items repacks the instances but leaves
    // their bounds as they were.
    let s = render(&mut h, &frame(&items, &camera, 2));
    assert_eq!(s.shadow_bounds_rebuilds, 0, "unchanged bounds are kept");

    // Moving the cube keeps the batch structure, so the instance buffer is
    // patched in place, and the bounds still have to follow it.
    items[1].model = glam::Mat4::from_translation(Vec3::new(3.0, 2.0, 0.5)).to_cols_array_2d();
    let s = render(&mut h, &frame(&items, &camera, 3));
    assert_eq!(
        s.shadow_bounds_rebuilds, 1,
        "a moved node rebuilds the bounds"
    );

    let s = render(&mut h, &frame(&items, &camera, 3));
    assert_eq!(s.shadow_bounds_rebuilds, 0);
}

#[test]
fn no_casting_directional_light_builds_no_bounds() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let camera = orbit_camera(Vec3::ZERO, 12.0, 0.4, 0.5);
    let mut fd = frame(&items, &camera, 1);
    fd.effects.lighting.lights.clear();
    let s = render(&mut h, &fd);
    assert_eq!(s.shadow_cascade_passes, 0);
    assert_eq!(s.shadow_bounds_rebuilds, 0);
}

#[test]
fn a_first_person_camera_fits_the_cascades_to_the_scene() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    // Standing at the -Y edge of the ground at eye height, looking along +Y
    // across it. The orbit radius is a token value with nothing to say about
    // how far the scene reaches.
    let eye = Vec3::new(0.0, -14.0, 1.7);
    let camera = Camera {
        center: eye + Vec3::Y * 0.01,
        distance: 0.01,
        orientation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        ..Camera::default()
    };
    let _ = render(&mut h, &frame(&items, &camera, 1));
    let splits = h.renderer.shadow_debug_stats().cascade_splits;
    // The ground runs about 29 m ahead of the eye; the camera far plane is
    // 1000 m away.
    let far = splits[3];
    assert!(
        (28.0..=31.0).contains(&far),
        "the last cascade ends where the ground does, not at {far} ({splits:?})"
    );
    assert!(splits.windows(2).all(|w| w[0] < w[1]), "{splits:?}");
}

#[test]
fn a_shadow_distance_ends_the_last_cascade_there() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let camera = orbit_camera(Vec3::ZERO, 12.0, 0.4, 0.5);
    let mut fd = frame(&items, &camera, 1);
    fd.effects.lighting.shadows.shadow_distance = Some(9.0);
    let _ = render(&mut h, &fd);
    let splits = h.renderer.shadow_debug_stats().cascade_splits;
    assert_eq!(splits[3], 9.0, "{splits:?}");
    assert!(splits.windows(2).all(|w| w[0] < w[1]), "{splits:?}");

    // Past the camera's far plane it stops there.
    let (_, camera_far) = fd.camera.render_camera.depth_range();
    fd.effects.lighting.shadows.shadow_distance = Some(camera_far * 10.0);
    let _ = render(&mut h, &fd);
    let splits = h.renderer.shadow_debug_stats().cascade_splits;
    assert_eq!(splits[3], camera_far, "{splits:?}");
}

#[test]
fn fraction_splits_divide_the_shadow_distance() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let camera = orbit_camera(Vec3::ZERO, 12.0, 0.4, 0.5);
    let mut fd = frame(&items, &camera, 1);
    fd.effects.lighting.shadows.shadow_distance = Some(40.0);
    fd.effects.lighting.shadows.cascade_split = CascadeSplit::Fractions([0.125, 0.25, 0.5]);
    let _ = render(&mut h, &fd);
    let s = h.renderer.shadow_debug_stats().cascade_splits;
    // The splits are fractions of near..40; the near end is the fitted one.
    let near = 40.0 - (s[1] - s[0]) / 0.125;
    assert!(near > 0.0 && near < s[0], "{s:?}");
    for (split, f) in s.iter().zip([0.125, 0.25, 0.5, 1.0]) {
        let want = near + (40.0 - near) * f;
        assert!((split - want).abs() < 1e-3, "{s:?}");
    }
}

/// A 60 m pillar on a wide ground, lit from 45 degrees, seen from 10 m
/// straight above a patch of ground the pillar's top shadows, 50 m from it.
/// Nothing of the pillar is in view or anywhere near the view slice along
/// the light, so its shadow is there only if each cascade's depth range
/// reaches the pillar (no depth clamp) or the shadow pass clamps it into
/// the range (depth clamp).
fn tall_caster_probe(h: &mut Harness) -> (f32, f32) {
    let resources = h.renderer.resources_mut();
    let ground_mesh = resources
        .upload_mesh_data(
            &h.device,
            &viewport_lib::primitives::cuboid(200.0, 200.0, 0.2),
        )
        .expect("ground");
    let pillar_mesh = resources
        .upload_mesh_data(&h.device, &viewport_lib::primitives::cuboid(1.0, 1.0, 60.0))
        .expect("pillar");
    let material = resources
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material");
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = ground_mesh;
    ground.model = glam::Mat4::from_translation(Vec3::new(0.0, 40.0, -0.1)).to_cols_array_2d();
    ground.material = material;
    let mut pillar = SceneRenderItem::default();
    pillar.mesh_id = pillar_mesh;
    pillar.model = glam::Mat4::from_translation(Vec3::new(0.0, 0.0, 30.0)).to_cols_array_2d();
    pillar.material = material;
    let items = vec![ground, pillar];

    let camera = orbit_camera(Vec3::new(0.0, 50.0, 0.0), 10.0, 0.0, 0.01);
    let mut fd = frame(&items, &camera, 1);
    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [0.0, -1.0, 1.0],
    };
    fd.effects.lighting.lights = vec![sun];
    let lit = h.render(&fd, W, H);
    fd.effects.lighting.shadows.enabled = false;
    let unshadowed = h.render(&fd, W, H);
    let centre = |img: &[u8]| {
        let i = ((H / 2) * W + W / 2) as usize * 4;
        img[i..i + 3].iter().map(|&c| c as f32).sum::<f32>()
    };
    (centre(&lit), centre(&unshadowed))
}

#[test]
fn a_tall_caster_outside_the_view_slice_still_shadows_with_depth_clamp() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if !h.renderer.capabilities().depth_clip_control {
        eprintln!("skipping: no DEPTH_CLIP_CONTROL");
        return;
    }
    let (shadowed, open) = tall_caster_probe(&mut h);
    assert!(shadowed < open * 0.7, "{shadowed} vs {open} unshadowed");
}

#[test]
fn a_tall_caster_outside_the_view_slice_still_shadows_without_depth_clamp() {
    let mut profile = DeviceProfile::harness();
    profile
        .optional_features
        .remove(wgpu::Features::DEPTH_CLIP_CONTROL);
    let Some(mut h) = Harness::with_profile(&profile) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(!h.renderer.capabilities().depth_clip_control);
    let (shadowed, open) = tall_caster_probe(&mut h);
    assert!(shadowed < open * 0.7, "{shadowed} vs {open} unshadowed");
}
