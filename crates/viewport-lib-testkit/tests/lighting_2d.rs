//! The 2D light layer: it encodes its passes only when the frame has point
//! lights, and an occluder darkens what lies behind it from the light to the
//! ambient level.

use glam::{DVec3, Vec2};
use viewport_lib::{
    Camera2d, CameraFrame, Candela, Colour, FrameData, LightSource, Plane2d, SceneFrame,
};
use viewport_lib_plugins::item_types::lighting_2d::{
    Lighting2dItem, Lighting2dPlugin, Lighting2dStats, Occluder2d, TYPE_NAME,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};
use viewport_lib_testkit::Harness;

const W: u32 = 160;
const H: u32 = 100;

/// A white floor sprite under the light layer, lit by `lights`.
fn frame(item: Lighting2dItem, lights: Vec<LightSource>) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    let mut floor = SpriteItem::default();
    floor.positions = vec![[0.0, 0.0, 0.0]];
    floor.sizes = vec![[W as f32, H as f32]];
    floor.size_mode = SpriteSizeMode::WorldSpace;
    floor.default_colour = Colour::linear_rgb(0.5, 0.5, 0.5);
    floor.depth_write = true;
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![floor];
    *scene.items_mut::<Lighting2dItem>() = vec![item];
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    );
    frame.effects.lighting.lights = lights;
    frame.viewport.show_axes_indicator = false;
    frame
}

fn torch(x: f64) -> LightSource {
    LightSource::point_candela(DVec3::new(x, 0.0, 5.0), Candela(1.0), 120.0, 0.0)
}

fn stats(harness: &Harness) -> Lighting2dStats {
    harness
        .renderer
        .item_type_plugin::<Lighting2dPlugin>(TYPE_NAME)
        .expect("registered")
        .last_stats()
}

fn luma(pixels: &[u8], x: f32, y: f32) -> u32 {
    let px = (W as f32 / 2.0 + x) as usize;
    let py = (H as f32 / 2.0 - y) as usize;
    let i = (py * W as usize + px) * 4;
    pixels[i] as u32 + pixels[i + 1] as u32 + pixels[i + 2] as u32
}

#[test]
fn the_layer_encodes_passes_only_when_there_are_lights() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // The frame's default light is directional: nothing to project.
    let unlit = harness.render(&frame(Lighting2dItem::default(), Vec::new()), W, H);
    assert_eq!(stats(&harness).passes, 0);
    assert_eq!(stats(&harness).lights, 0);

    let lights = vec![torch(-40.0), torch(0.0), torch(40.0)];
    let _ = harness.render(&frame(Lighting2dItem::default(), lights), W, H);
    assert_eq!(
        stats(&harness),
        Lighting2dStats {
            lights: 3,
            edges: 0,
            passes: 2
        }
    );

    // And back: no lights, no passes, and the scene draws as it did.
    let again = harness.render(&frame(Lighting2dItem::default(), Vec::new()), W, H);
    assert_eq!(stats(&harness).passes, 0);
    assert_eq!(unlit, again);
}

#[test]
fn an_occluder_shadows_what_lies_behind_it() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut item = Lighting2dItem::default();
    item.occluders = vec![Occluder2d::Rect {
        min: [-25.0, -5.0],
        max: [-15.0, 5.0],
    }];
    let pixels = harness.render(&frame(item, vec![torch(-60.0)]), W, H);
    assert_eq!(stats(&harness).edges, 4);
    let behind = luma(&pixels, 10.0, 0.0);
    let beside = luma(&pixels, 10.0, 20.0);
    let in_front = luma(&pixels, -40.0, 0.0);
    // Further into the shadow, where the unshadowed light would be dimmer:
    // the same ambient level, so no light reaches either.
    let deeper = luma(&pixels, 40.0, 0.0);
    assert!(
        behind.abs_diff(deeper) <= 3,
        "behind {behind}, deeper {deeper}"
    );
    assert!(beside > behind + 60, "behind {behind}, beside {beside}");
    assert!(in_front > beside, "beside {beside}, in front {in_front}");
}
