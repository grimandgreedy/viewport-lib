//! Blended sprites on the LDR path.
//!
//! The HDR path draws blended sprites in its read-only-depth pass (soft,
//! additive, screen and multiply batches) and through OIT (alpha and
//! premultiplied batches). The LDR path has neither pass, so it draws them
//! after its opaque sprites. Each blend here must draw on the LDR path what it
//! draws on the HDR path, over an opaque backdrop so the blends that keep the
//! destination's alpha have something to blend with on both paths.

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
    frame
}

/// An opaque grey square behind the blended pair, writing depth.
fn backdrop() -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[0.0, 0.0, -5.0]];
    item.colours = vec![Colour::linear(0.3, 0.3, 0.3, 1.0)];
    item.default_size = 100.0;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.depth_write = true;
    item
}

/// Two overlapping squares with `blend`, centred at x = -10 and x = 10.
fn pair(blend: SpriteBlend, soft: bool) -> SpriteItem {
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
    if soft {
        item.soft_particle_distance = Some(0.5);
    }
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
fn blended_sprites_draw_on_the_ldr_path() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // (name, blend, soft fade, drawn through OIT on the HDR path)
    let cases = [
        ("alpha", SpriteBlend::AlphaBlend, false, true),
        ("premultiplied", SpriteBlend::Premultiplied, false, true),
        ("additive", SpriteBlend::Additive, false, false),
        ("screen", SpriteBlend::Screen, false, false),
        ("multiply", SpriteBlend::Multiply, false, false),
        ("soft alpha", SpriteBlend::AlphaBlend, true, false),
    ];
    let (cx, cy) = (W / 2, H / 2);
    // Inside one sprite only, inside both, inside the other only.
    let left = (cx - 20, cy);
    let both = (cx, cy);
    let right = (cx + 20, cy);
    let mut generation = 10;
    for (name, blend, soft, oit) in cases {
        let mut images = Vec::new();
        for mode in [PipelineMode::Hdr, PipelineMode::Direct] {
            generation += 2;
            let base = draw(&mut h, &frame(vec![backdrop()], mode, generation));
            let img = draw(
                &mut h,
                &frame(vec![backdrop(), pair(blend, soft)], mode, generation + 1),
            );
            for (x, y) in [left, both, right] {
                assert_ne!(
                    pixel(&img, x, y),
                    pixel(&base, x, y),
                    "{name} on {mode:?}: nothing drawn at ({x}, {y})"
                );
            }
            images.push((base, img));
        }
        // The paths tone map differently, so each is compared by what the
        // sprites change from its own backdrop. Order-independent transparency
        // approximates where the two overlap, so only the single-sprite pixels
        // are compared for it.
        let at: &[(u32, u32)] = if oit {
            &[left, right]
        } else {
            &[left, both, right]
        };
        let change = |(base, img): &(Vec<u8>, Vec<u8>), x: u32, y: u32| -> [i32; 3] {
            let (b, i) = (pixel(base, x, y), pixel(img, x, y));
            [0, 1, 2].map(|c| i[c] as i32 - b[c] as i32)
        };
        for &(x, y) in at {
            let (a, b) = (change(&images[0], x, y), change(&images[1], x, y));
            let close = a.iter().zip(&b).all(|(p, q)| (p - q).abs() <= 16);
            assert!(close, "{name} at ({x}, {y}): HDR changes {a:?}, LDR {b:?}");
        }
    }
}
