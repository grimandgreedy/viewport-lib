//! `ItemSettings::draw_priority` and `blend_order` decide which blended mesh
//! draws on top on the LDR (Direct) path, where blended draws are sorted
//! rather than resolved through OIT.

use viewport_lib::Colour;
use viewport_lib::{
    CameraFrame, FrameData, Material, NodeId, PipelineMode, Scene, SceneFrame, Selection,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 64;
const H: u32 = 64;

/// Two blended, unlit 4x4 quads in the XY plane, red and blue, the blue one
/// raised by `blue_lift` towards the camera above.
fn scene(h: &mut Harness, blue_lift: f32) -> (Scene, NodeId, NodeId) {
    let quad = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &viewport_lib::primitives::plane(4.0, 4.0))
        .expect("upload quad");
    let mut material = |c: Colour| {
        h.renderer
            .resources_mut()
            .upload_material(Material::from_colour(c))
            .expect("material upload")
    };
    let red_mat = material(Colour::linear_rgb(1.0, 0.0, 0.0));
    let blue_mat = material(Colour::linear_rgb(0.0, 0.0, 1.0));

    let mut scene = Scene::new();
    let red = scene.add(Some(quad), glam::Mat4::IDENTITY, red_mat);
    let blue = scene.add(
        Some(quad),
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, blue_lift)),
        blue_mat,
    );
    for id in [red, blue] {
        let mut s = *scene.node(id).unwrap().appearance();
        s.unlit = true;
        s.opacity = 0.6;
        scene.set_appearance(id, s);
    }
    (scene, red, blue)
}

/// Render the scene from straight above (Z-up) and return the centre pixel.
fn centre_pixel(h: &mut Harness, scene: &mut Scene) -> [u8; 4] {
    let camera = orbit_camera(glam::Vec3::ZERO, 10.0, 0.0, 0.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_scene(scene, &Selection::new()),
    );
    fd.effects.display.mode = PipelineMode::Direct;
    fd.viewport.show_axes_indicator = false;
    fd.viewport.show_grid = false;
    let _ = h.render(&fd, W, H);
    let px = h.render(&fd, W, H);
    let i = ((H / 2 * W + W / 2) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn red_on_top(px: [u8; 4]) -> bool {
    px[0] > px[2]
}

fn blue_on_top(px: [u8; 4]) -> bool {
    px[2] > px[0]
}

#[test]
fn draw_priority_orders_blended_quads_at_equal_depth() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut scene, red, blue) = scene(&mut h, 0.0);

    // Red draws last, so it blends over blue.
    scene.set_draw_priority(red, 1);
    scene.set_draw_priority(blue, 0);
    let px = centre_pixel(&mut h, &mut scene);
    assert!(red_on_top(px), "red has the higher priority: {px:?}");

    // Swapped on the retained scene: the setter bumps the generation, so
    // the batches are rebuilt and blue now draws over red.
    scene.set_draw_priority(red, 0);
    scene.set_draw_priority(blue, 1);
    let px = centre_pixel(&mut h, &mut scene);
    assert!(blue_on_top(px), "blue has the higher priority: {px:?}");
}

#[test]
fn blend_order_draws_far_over_near() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Blue is nearer the camera.
    let (mut scene, red, blue) = scene(&mut h, 1.0);

    // By depth, the near quad draws last.
    let px = centre_pixel(&mut h, &mut scene);
    assert!(
        blue_on_top(px),
        "the near quad is on top by default: {px:?}"
    );

    // A fixed order puts the far quad last.
    scene.set_blend_order(blue, Some(0));
    scene.set_blend_order(red, Some(1));
    let px = centre_pixel(&mut h, &mut scene);
    assert!(
        red_on_top(px),
        "blend order puts the far quad on top: {px:?}"
    );
}
