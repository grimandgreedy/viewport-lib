//! The directional cascades cover the view depths where the scene's casters
//! can throw a shadow, taken from the mesh instance bounds. These pin when
//! the caster and scene bounds are rebuilt, and that the range follows the
//! scene rather than the camera's orbit radius.

use glam::{Quat, Vec3};
use viewport_lib::Colour;
use viewport_lib::{
    Camera, CameraFrame, FrameData, FrameStats, LightingSettings, Material, SceneFrame,
    SceneRenderItem,
};
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
