//! A `Camera2d` with a caller depth range draws layers deep behind its plane.
//!
//! The automatic planes reach about `max(1000, 3 * distance)` from the eye, so
//! a camera close to its plane clips a layer far behind it. `depth_range`
//! sets the planes the written projection uses instead.

use glam::Vec2;
use viewport_lib::{Camera2d, CameraFrame, Colour, FrameData, Plane2d, SceneFrame};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};
use viewport_lib_testkit::Harness;

const W: u32 = 64;
const H: u32 = 48;

/// One opaque square filling the middle of the view, `depth` units along the
/// plane's normal (negative is behind the plane, away from the eye).
fn square(depth: f32) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[0.0, 0.0, depth]];
    item.colours = vec![Colour::linear(0.9, 0.5, 0.1, 1.0)];
    item.default_size = 20.0;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item
}

fn frame(camera: &Camera2d, depth: f32, generation: u64) -> FrameData {
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![square(depth)];
    scene.generation = generation;
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera.camera(), [W as f32, H as f32]),
        scene,
    );
    frame.viewport.show_axes_indicator = false;
    frame.viewport.show_grid = false;
    frame.viewport.background_colour = Some(Colour::srgb(0.1, 0.1, 0.1, 1.0));
    frame
}

/// Whether the square shows at the centre of the view.
fn draws(h: &mut Harness, camera: &Camera2d, depth: f32, generation: u64) -> bool {
    let frame = frame(camera, depth, generation);
    h.render(&frame, W, H);
    let img = h.render(&frame, W, H);
    let i = (((H / 2) * W + W / 2) * 4) as usize;
    let centre = &img[i..i + 3];
    // The square is bright orange; the background is a dark grey.
    centre[0] > 150 && centre[0] > centre[2] + 60
}

fn camera(distance: f32, depth_range: Option<[f32; 2]>) -> Camera2d {
    let mut camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32));
    camera.distance = distance;
    camera.depth_range = depth_range;
    camera
}

#[test]
fn a_caller_depth_range_draws_a_layer_far_behind_the_plane() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // 2000 units from the plane, with planes from 100 to 4000: a layer 1500
    // behind the plane draws, and one 1950 in front of it (50 from the eye)
    // falls before the near plane.
    let far_eye = camera(2000.0, Some([100.0, 4000.0]));
    assert!(draws(&mut h, &far_eye, -1500.0, 10));
    assert!(draws(&mut h, &far_eye, 0.0, 12));
    assert!(!draws(&mut h, &far_eye, 1950.0, 14));

    // 100 units from the plane, the automatic far plane is 1000 from the eye
    // and clips the same layer; a caller range reaching 2000 draws it.
    assert!(!draws(&mut h, &camera(100.0, None), -1500.0, 16));
    assert!(draws(
        &mut h,
        &camera(100.0, Some([1.0, 2000.0])),
        -1500.0,
        18
    ));
}
