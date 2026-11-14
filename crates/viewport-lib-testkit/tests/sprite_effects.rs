//! Sprite effects: a frame without them draws through the plain pipelines, a
//! flash reaches one sprite of a batch, the effects channel writes one sprite
//! of an uploaded set, and a sprite material draws through its own hook.

use glam::Vec2;
use viewport_lib::plugin_api::{Uploads, Writes};
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, MaterialPlugin, Plane2d, SceneFrame, ViewportError,
};
use viewport_lib_plugins::item_types::sprite::channels::{Effects, SpriteEffect};
use viewport_lib_plugins::item_types::sprite::{
    SpriteEffects, SpriteItem, SpriteMaterial, SpriteMaterials, SpritePlugin, SpriteSetRefItem,
    SpriteSizeMode, TYPE_NAME,
};
use viewport_lib_testkit::Harness;

const W: u32 = 120;
const H: u32 = 80;

fn frame(sprites: Vec<SpriteItem>, refs: Vec<SpriteSetRefItem>) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = sprites;
    *scene.items_mut::<SpriteSetRefItem>() = refs;
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    );
    frame.viewport.background_colour = Some(Colour::BLACK);
    frame.viewport.show_axes_indicator = false;
    frame
}

/// Two dark red squares, at x = -30 and x = 30.
fn pair() -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[-30.0, 0.0, 0.0], [30.0, 0.0, 0.0]];
    item.default_colour = Colour::linear_rgb(0.3, 0.02, 0.02);
    item.default_size = 30.0;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item
}

/// The pixel at world `(x, y)`.
fn pixel(pixels: &[u8], x: f32, y: f32) -> [u8; 3] {
    let px = (W as f32 / 2.0 + x) as usize;
    let py = (H as f32 / 2.0 - y) as usize;
    let i = (py * W as usize + px) * 4;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn sprite_plugin(harness: &Harness) -> &SpritePlugin {
    harness
        .renderer
        .item_type_plugin::<SpritePlugin>(TYPE_NAME)
        .expect("registered")
}

#[test]
fn a_frame_without_effects_draws_through_the_plain_pipelines() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let plain = harness.render(&frame(vec![pair()], vec![]), W, H);
    assert_eq!(sprite_plugin(&harness).effect_batch_count(), 0);

    // The second sprite flashes fully; the first is left alone.
    let mut flashed = pair();
    let mut fx = SpriteEffects::default();
    fx.flash = vec![0.0, 1.0];
    flashed.effects = Some(fx);
    let lit = harness.render(&frame(vec![flashed], vec![]), W, H);
    assert_eq!(sprite_plugin(&harness).effect_batch_count(), 1);

    assert_eq!(pixel(&lit, -30.0, 0.0), pixel(&plain, -30.0, 0.0));
    let white = pixel(&lit, 30.0, 0.0);
    assert!(white.iter().all(|&c| c > 200), "flashed sprite: {white:?}");
}

#[test]
fn the_effects_channel_writes_one_sprite_of_a_set() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let mut item = pair();
    item.effects = Some(SpriteEffects::default());
    let id = harness.renderer.upload(&device, &queue, &item).unwrap();
    // The frame carries an empty inline collection ahead of the reference,
    // which must not hide the reference from the read-only-depth pass.
    let before = harness.render(&frame(vec![], vec![SpriteSetRefItem::new(id)]), W, H);
    assert!(pixel(&before, -30.0, 0.0)[0] > 100, "the set draws");
    assert!(pixel(&before, -30.0, 0.0)[1] < 100);

    harness
        .renderer
        .write_range(
            Effects,
            &queue,
            id,
            0,
            &[SpriteEffect {
                flash: 1.0,
                ..SpriteEffect::default()
            }],
        )
        .unwrap();
    let after = harness.render(&frame(vec![], vec![SpriteSetRefItem::new(id)]), W, H);
    assert!(pixel(&after, -30.0, 0.0).iter().all(|&c| c > 200));
    assert_eq!(pixel(&after, 30.0, 0.0), pixel(&before, 30.0, 0.0));

    // A set uploaded without effects has no effects channel.
    let plain = harness.renderer.upload(&device, &queue, &pair()).unwrap();
    let err = harness
        .renderer
        .write_range(Effects, &queue, plain, 0, &[SpriteEffect::default()])
        .unwrap_err();
    assert!(
        matches!(err, ViewportError::ChannelNotPresent { .. }),
        "{err:?}"
    );
}

/// Paints every sprite in its `tint` parameter, keeping the texel's alpha.
struct Solid;

impl MaterialPlugin for Solid {
    fn name(&self) -> &str {
        "solid"
    }

    fn wgsl_body(&self) -> String {
        "fn shade_sprite(f: SpriteFragment) -> vec4<f32> {\n    \
             return vec4<f32>(params.tint, f.texel.a * f.colour.a);\n}\n"
            .to_string()
    }

    fn param_layout(&self) -> Vec<viewport_lib::ParamDesc> {
        vec![viewport_lib::ParamDesc::vec3("tint", [1.0, 0.0, 1.0])]
    }
}

/// A body with no `shade_sprite`.
struct Empty;

impl MaterialPlugin for Empty {
    fn name(&self) -> &str {
        "empty"
    }

    fn wgsl_body(&self) -> String {
        "fn helper() -> f32 { return 1.0; }".to_string()
    }
}

#[test]
fn a_sprite_material_draws_through_its_hook() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let id = harness
        .renderer
        .register_sprite_material(&device, &Solid)
        .expect("registers");
    assert_eq!(harness.renderer.sprite_material_id("solid"), Some(id));
    assert!(
        harness
            .renderer
            .register_sprite_material(&device, &Empty)
            .is_err()
    );

    let mut item = pair();
    let mut fx = SpriteEffects::default();
    fx.material = Some(SpriteMaterial::new(id).with_param("tint", [0.0, 1.0, 0.0]));
    item.effects = Some(fx);
    let pixels = harness.render(&frame(vec![item], vec![]), W, H);
    let [r, g, b] = pixel(&pixels, 30.0, 0.0);
    assert!(
        r < 60 && g > 200 && b < 60,
        "green from the hook: {:?}",
        [r, g, b]
    );
}
