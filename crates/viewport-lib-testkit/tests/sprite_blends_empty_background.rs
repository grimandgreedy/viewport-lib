//! Screen and multiply sprites over the empty background.
//!
//! Where nothing else drew, the HDR path composites the background under the
//! scene target by the target's alpha, so a blended sprite has to write its
//! coverage there to show. The LDR path draws over the cleared background
//! directly. Each blend must change the background under it on both paths.
//!
//! Over the empty background the two paths differ: the HDR path blends the
//! sprite against the cleared target and composites the background under the
//! result by coverage, so screen does not only lighten there and multiply
//! darkens by coverage rather than by colour.

use glam::Vec2;
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, PipelineMode, Plane2d, SceneFrame, SpriteBlend,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};
use viewport_lib_testkit::Harness;

const W: u32 = 160;
const H: u32 = 120;

fn frame(sprites: Vec<SpriteItem>, mode: PipelineMode, generation: u64) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = sprites;
    scene.generation = generation;
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    );
    frame.effects.display.mode = mode;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.show_grid = false;
    frame.viewport.background_colour = Some(Colour::srgb(0.5, 0.6, 0.7, 1.0));
    frame
}

/// Two overlapping squares with `blend`, centred at x = -10 and x = 10, that
/// do not write depth.
fn pair(blend: SpriteBlend) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
    item.colours = vec![
        Colour::linear(0.6, 0.2, 0.1, 0.6),
        Colour::linear(0.1, 0.2, 0.6, 0.6),
    ];
    item.default_size = 30.0;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.depth_write = false;
    item.blend = blend;
    item
}

/// The frame drawn twice, so the second draw has every pipeline built.
fn draw(h: &mut Harness, frame: &FrameData) -> Vec<u8> {
    h.render(frame, W, H);
    h.render(frame, W, H)
}

fn pixel(img: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [img[i], img[i + 1], img[i + 2], img[i + 3]]
}

#[test]
fn screen_and_multiply_draw_over_the_empty_background() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cx, cy) = (W / 2, H / 2);
    // Inside one sprite only, inside both, inside the other only.
    let at = [(cx - 20, cy), (cx, cy), (cx + 20, cy)];
    let mut generation = 10;
    for blend in [SpriteBlend::Screen, SpriteBlend::Multiply] {
        for mode in [PipelineMode::Hdr, PipelineMode::Direct] {
            generation += 2;
            let base = draw(&mut h, &frame(Vec::new(), mode, generation));
            let img = draw(&mut h, &frame(vec![pair(blend)], mode, generation + 1));
            for (x, y) in at {
                let (b, i) = (pixel(&base, x, y), pixel(&img, x, y));
                assert_ne!(i, b, "{blend:?} on {mode:?}: nothing drawn at ({x}, {y})");
                // On the LDR path the background is in the target, so screen
                // only lightens it and multiply only darkens it. The HDR path
                // composites the background afterwards, by coverage alone.
                if mode == PipelineMode::Hdr {
                    continue;
                }
                let sum = |p: [u8; 4]| p[..3].iter().map(|&c| c as i32).sum::<i32>();
                let lighter = sum(i) > sum(b);
                assert_eq!(
                    lighter,
                    blend == SpriteBlend::Screen,
                    "{blend:?} on {mode:?} at ({x}, {y}): {i:?} over {b:?}"
                );
            }
        }
    }
}
