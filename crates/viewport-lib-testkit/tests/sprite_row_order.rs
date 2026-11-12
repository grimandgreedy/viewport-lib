//! Sprite row order: a texture uploaded top row first, as image loaders
//! return it, draws upright, and every path that reads a sprite's texture
//! (draw modes, flips, the pick alpha test and alpha masks) reads the same
//! rows.
//!
//! One world unit is one pixel, with the world origin in the middle of the
//! view, so the checks read pixel positions straight off the world layout.

use glam::Vec2;
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, MaskTest, PickBackend, PickId, PickMask, Plane2d,
    SceneFrame, TextureData, TextureFilter, TextureId,
};
use viewport_lib_plugins::item_types::sprite::{
    MaskSource, SpriteDrawMode, SpriteFlip, SpriteItem, SpriteMaskItem, SpriteOrder,
    SpriteSizeMode, TileMode,
};
use viewport_lib_testkit::Harness;

const W: u32 = 240;
const H: u32 = 160;

fn frame(sprites: Vec<SpriteItem>, masks: Vec<SpriteMaskItem>) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = sprites;
    *scene.items_mut::<SpriteMaskItem>() = masks;
    FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    )
}

/// The RGB of the pixel at world `(x, y)` of a `W` by `H` render.
fn pixel(pixels: &[u8], x: f32, y: f32) -> [u8; 3] {
    let col = (W as f32 * 0.5 + x).floor() as usize;
    let row = (H as f32 * 0.5 - y).floor() as usize;
    let i = (row * W as usize + col) * 4;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn is_colour(p: [u8; 3], want: [bool; 3]) -> bool {
    p.iter()
        .zip(want)
        .all(|(&c, on)| if on { c > 180 } else { c < 80 })
}

const RED: [bool; 3] = [true, false, false];
const GREEN: [bool; 3] = [false, true, false];
const BLUE: [bool; 3] = [false, false, true];

const RED_TEXEL: [u8; 4] = [255, 0, 0, 255];
const GREEN_TEXEL: [u8; 4] = [0, 255, 0, 255];
const BLUE_TEXEL: [u8; 4] = [0, 0, 255, 255];

/// Upload a `w` by `h` texture, `texel(x, y)` with `y = 0` the first row
/// uploaded.
fn upload_texture(
    harness: &mut Harness,
    w: u32,
    h: u32,
    texel: impl Fn(u32, u32) -> [u8; 4],
) -> TextureId {
    let pixels: Vec<u8> = (0..w * h).flat_map(|i| texel(i % w, i / w)).collect();
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    harness
        .renderer
        .resources_mut()
        .upload_texture(&device, &queue, TextureData::srgb(w, h, pixels))
        .expect("texture upload")
}

/// A world-space, nearest-sampled sprite of `texture`.
fn textured(at: [f32; 3], size: [f32; 2], texture: TextureId) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![at];
    item.sizes = vec![size];
    item.texture_id = Some(texture);
    item.filter = TextureFilter::Nearest;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item
}

#[test]
fn the_first_row_draws_at_the_top_and_flips_and_rects_follow() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Four rows from the top: red, green, blue, blue.
    let tex = upload_texture(&mut harness, 1, 4, |_, y| match y {
        0 => RED_TEXEL,
        1 => GREEN_TEXEL,
        _ => BLUE_TEXEL,
    });
    let plain = textured([0.0, 0.0, 0.0], [40.0, 40.0], tex);
    let mut flipped = textured([45.0, 0.0, 0.0], [40.0, 40.0], tex);
    flipped.flips = vec![SpriteFlip::VERTICAL];
    // The top half of the image: red over green.
    let mut rect = textured([90.0, 0.0, 0.0], [40.0, 40.0], tex);
    rect.uv_rects = vec![[0.0, 0.0, 1.0, 0.5]];
    let pixels = harness.render(&frame(vec![plain, flipped, rect], vec![]), W, H);

    // Each quad spans y -20..20 in rows of 10.
    assert!(is_colour(pixel(&pixels, 0.0, 15.0), RED), "row 0 on top");
    assert!(is_colour(pixel(&pixels, 0.0, 5.0), GREEN));
    assert!(
        is_colour(pixel(&pixels, 0.0, -15.0), BLUE),
        "last row below"
    );
    assert!(
        is_colour(pixel(&pixels, 45.0, 15.0), BLUE),
        "a vertical flip puts the last row on top"
    );
    assert!(is_colour(pixel(&pixels, 45.0, -15.0), RED));
    assert!(
        is_colour(pixel(&pixels, 90.0, 10.0), RED),
        "v0 is the rect's top edge"
    );
    assert!(is_colour(pixel(&pixels, 90.0, -10.0), GREEN));
}

#[test]
fn sliced_borders_and_tiles_follow_the_image_rows() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // 32 x 32: a 12-row red band on top, a 2-row blue band at the bottom,
    // green between.
    let bands = upload_texture(&mut harness, 32, 32, |_, y| {
        if y < 12 {
            RED_TEXEL
        } else if y >= 30 {
            BLUE_TEXEL
        } else {
            GREEN_TEXEL
        }
    });
    let mut sliced = textured([0.0, 0.0, 0.0], [40.0, 80.0], bands);
    sliced.draw_mode = SpriteDrawMode::sliced([0.0, 2.0, 0.0, 12.0]);
    // 16 x 16: top half red, bottom half blue.
    let halves = upload_texture(&mut harness, 16, 16, |_, y| {
        if y < 8 { RED_TEXEL } else { BLUE_TEXEL }
    });
    let mut tiled = textured([60.0, 0.0, 0.0], [16.0, 40.0], halves);
    tiled.draw_mode = SpriteDrawMode::Tiled {
        tile_size: [16.0, 16.0],
        mode: TileMode::Continuous,
    };
    let pixels = harness.render(&frame(vec![sliced, tiled], vec![]), W, H);

    // The sliced quad spans y -40..40: the top border is 12 units, the
    // bottom one 2.
    assert!(
        is_colour(pixel(&pixels, 0.0, 40.0 - 11.5), RED),
        "top border, 12 px"
    );
    assert!(is_colour(pixel(&pixels, 0.0, 40.0 - 12.5), GREEN));
    assert!(
        is_colour(pixel(&pixels, 0.0, -40.0 + 1.5), BLUE),
        "bottom border, 2 px"
    );
    assert!(is_colour(pixel(&pixels, 0.0, -40.0 + 2.5), GREEN));

    // The tiled quad spans y -20..20, tiles running down from the top edge
    // and the last cut at 8 units: its red top half.
    for (down, want) in [
        (4.0, RED),
        (12.0, BLUE),
        (20.0, RED),
        (28.0, BLUE),
        (36.0, RED),
    ] {
        assert!(
            is_colour(pixel(&pixels, 60.0, 20.0 - down), want),
            "tiled at {down} from the top"
        );
    }
}

#[test]
fn the_pick_alpha_test_and_alpha_masks_read_the_drawn_rows() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    // Opaque top row, transparent bottom row.
    let top_half = upload_texture(&mut harness, 1, 2, |_, y| {
        if y == 0 { [255, 255, 255, 255] } else { [0; 4] }
    });

    // Picks: the transparent half falls through.
    let mut picked = textured([0.0, 0.0, 0.0], [40.0, 40.0], top_half);
    picked.settings.pick_id = PickId(31);
    picked.pick_alpha_threshold = Some(0.5);
    // An alpha mask from the same texture over a solid red sprite: only the
    // top half shows.
    let mut masked = SpriteItem::default();
    masked.positions = vec![[60.0, 0.0, 0.0]];
    masked.sizes = vec![[40.0, 40.0]];
    masked.default_colour = Colour::linear_rgb(1.0, 0.0, 0.0);
    masked.size_mode = SpriteSizeMode::WorldSpace;
    masked.order = SpriteOrder::Painter {
        layer: 1,
        depth_test: false,
    };
    masked.mask = MaskTest::VisibleInside(0);
    let mask = SpriteMaskItem::new(
        MaskSource::Alpha {
            sprite: textured([60.0, 0.0, 0.0], [40.0, 40.0], top_half),
            threshold: 0.5,
        },
        0,
        1..=1,
    );
    let f = frame(vec![picked, masked], vec![mask]);
    let pixels = harness.render(&f, W, H);

    assert!(
        is_colour(pixel(&pixels, 60.0, 10.0), RED),
        "the mask's opaque top half shows"
    );
    assert!(
        !is_colour(pixel(&pixels, 60.0, -10.0), RED),
        "its transparent bottom half cuts"
    );

    let pick = |harness: &mut Harness, x: f32, y: f32| {
        harness.renderer.pick_object(
            PickBackend::Gpu,
            Vec2::new(W as f32 * 0.5 + x, H as f32 * 0.5 - y),
            &f,
            &device,
            &queue,
            PickMask::OBJECT,
        )
    };
    assert_eq!(
        pick(&mut harness, 0.0, 10.0).map(|h| h.id),
        Some(31),
        "the opaque top half picks"
    );
    assert!(
        pick(&mut harness, 0.0, -10.0).is_none(),
        "the transparent bottom half does not"
    );
}
