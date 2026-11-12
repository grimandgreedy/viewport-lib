//! Sprite and tile deformers: bodies a plugin registers once that move the
//! quads of the batches and layers that opt in, read their parameters from a
//! mesh deformer's slot, run in priority order before a material's own vertex
//! hook, reach uploaded sets through their references, and move the pick with
//! the quad.
//!
//! One world unit is one pixel, with the world origin in the middle of the
//! view. The harness draws an axes indicator in the bottom-left corner, so
//! the content keeps clear of it.

use glam::Vec2;
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, MaterialPlugin, PickBackend, PickId, PickMask,
    Plane2d, SceneFrame,
};
use viewport_lib_plugins::item_types::sprite::{
    SpriteDeformerDesc, SpriteDeformerSet, SpriteDeformers, SpriteEffects, SpriteItem,
    SpriteMaterial, SpriteMaterials, SpriteSetRefItem, SpriteSizeMode,
};
use viewport_lib_plugins::item_types::tilemap::{Tile, Tilemap, TilemapItem, Tileset};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::fixtures::{ConstantOffsetDeformer, constant_offset_deformer_named};

const W: u32 = 240;
const H: u32 = 160;

fn frame(scene: SceneFrame) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    )
}

fn sprites(items: Vec<SpriteItem>) -> FrameData {
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = items;
    frame(scene)
}

/// Whether the pixel at world `(x, y)` of a `W` by `H` render is red.
fn red(pixels: &[u8], x: f32, y: f32) -> bool {
    let col = (W as f32 * 0.5 + x).floor() as usize;
    let row = (H as f32 * 0.5 - y).floor() as usize;
    let i = (row * W as usize + col) * 4;
    pixels[i] > 180 && pixels[i + 1] < 80 && pixels[i + 2] < 80
}

const RED: Colour = Colour::linear(1.0, 0.0, 0.0, 1.0);

/// A red world-space square of side 40 at `(x, y)`.
fn square(x: f32, y: f32) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[x, y, 0.0]];
    item.default_size = 40.0;
    item.default_colour = RED;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item
}

/// Leans the top edge right by the `x` of its mesh slot's first parameter.
const LEAN: &str = "fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32> {\n    \
     let shift = deform_header.slot_params[slot * 4u].x;\n    \
     return v.world + v.right * shift * v.local_uv.y;\n}\n";

/// A mesh deformer slot whose first parameter is `(dx, 0, 0)`, for the sprite
/// deformers to read. `None` when the device cannot register one.
fn mesh_slot(harness: &mut Harness, name: &str, dx: f32) -> Option<viewport_lib::DeformerId> {
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = match harness
        .renderer
        .resources_mut()
        .register_deformer(&device, constant_offset_deformer_named(name.to_string()))
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a deformer: {err}");
            return None;
        }
    };
    ConstantOffsetDeformer::set_offset(
        harness.renderer.resources_mut(),
        &queue,
        id.slot(),
        glam::Vec3::new(dx, 0.0, 0.0),
    );
    Some(id)
}

#[test]
fn a_deformer_moves_only_the_batches_that_opt_in_by_its_slot_params() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some(slot) = mesh_slot(&mut harness, "wind_slot", 30.0) else {
        return;
    };
    let device = harness.device.clone();
    let id = harness
        .renderer
        .register_sprite_deformer(
            &device,
            &SpriteDeformerDesc::new("lean", LEAN).with_params(slot),
        )
        .expect("registers");
    assert_eq!(harness.renderer.sprite_deformer_id("lean"), Some(id));
    let mut swaying = square(-50.0, 40.0);
    swaying.deformers = id.into();
    let still = square(50.0, 40.0);
    let pixels = harness.render(&sprites(vec![swaying, still]), W, H);
    // The swaying square spans -70..-30 and 20..60; its top edge moves to
    // -40..0.
    assert!(red(&pixels, -5.0, 58.0), "the top edge moved right");
    assert!(!red(&pixels, -5.0, 22.0), "the bottom edge did not");
    assert!(
        red(&pixels, 66.0, 58.0),
        "the batch that did not opt in stays"
    );
    assert!(!red(&pixels, 75.0, 58.0), "and does not lean");

    // One write to the slot moves it again.
    ConstantOffsetDeformer::set_offset(
        harness.renderer.resources_mut(),
        &harness.queue.clone(),
        slot.slot(),
        glam::Vec3::ZERO,
    );
    let mut swaying = square(-50.0, 40.0);
    swaying.deformers = id.into();
    let pixels = harness.render(&sprites(vec![swaying]), W, H);
    assert!(!red(&pixels, -5.0, 58.0), "a zero parameter stands it up");
    assert!(red(&pixels, -66.0, 58.0), "upright");
}

/// Lifts every corner by 20.
struct Lift;

impl MaterialPlugin for Lift {
    fn name(&self) -> &str {
        "lift"
    }

    fn wgsl_body(&self) -> String {
        "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
             return v.world + v.up * 20.0;\n}\n"
            .to_string()
    }
}

#[test]
fn deformers_run_in_priority_order_before_a_host_material() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    // `double` scales x about the origin, `shift` moves 20 right: shift then
    // double lands a corner at 2x + 40, double then shift at 2x + 20.
    let double = harness
        .renderer
        .register_sprite_deformer(
            &device,
            &SpriteDeformerDesc::new(
                "double",
                "fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32> {\n    \
                 return vec3<f32>(v.world.x * 2.0, v.world.y, v.world.z);\n}\n",
            )
            .with_priority(1),
        )
        .expect("registers");
    let shift = harness
        .renderer
        .register_sprite_deformer(
            &device,
            &SpriteDeformerDesc::new(
                "shift",
                "fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32> {\n    \
                 return v.world + vec3<f32>(20.0, 0.0, 0.0);\n}\n",
            )
            .with_priority(0),
        )
        .expect("registers");
    let lift = harness
        .renderer
        .register_sprite_material(&device, &Lift)
        .expect("a material registers after the deformers");
    // A 20-unit square at (10, 0): x 0..20, so 40..80 after both deformers,
    // and y -10..10, so 10..30 after the material.
    let mut item = square(10.0, 0.0);
    item.default_size = 20.0;
    item.deformers = SpriteDeformerSet::NONE.with(double).with(shift);
    let mut fx = SpriteEffects::default();
    fx.material = Some(SpriteMaterial::new(lift));
    item.effects = Some(fx);
    item.settings.pick_id = PickId(9);
    let f = sprites(vec![item]);
    let pixels = harness.render(&f, W, H);
    assert!(red(&pixels, 45.0, 25.0), "shifted, doubled and lifted");
    assert!(red(&pixels, 75.0, 15.0), "to its far corner");
    assert!(!red(&pixels, 35.0, 25.0), "not doubled first");
    assert!(!red(&pixels, 45.0, 5.0), "lifted by the material");

    let queue = harness.queue.clone();
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
        pick(&mut harness, 60.0, 20.0).map(|h| h.id),
        Some(9),
        "the moved quad picks"
    );
    assert!(
        pick(&mut harness, 10.0, 0.0).is_none(),
        "where it was does not"
    );
}

#[test]
fn an_uploaded_set_takes_deformers_from_its_reference() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some(slot) = mesh_slot(&mut harness, "set_slot", 30.0) else {
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .register_sprite_deformer(
            &device,
            &SpriteDeformerDesc::new("lean", LEAN).with_params(slot),
        )
        .expect("registers");
    let set = harness
        .renderer
        .upload(&device, &queue, &square(-50.0, 40.0))
        .expect("upload");
    let render = |harness: &mut Harness, deformers| {
        let mut r = SpriteSetRefItem::new(set);
        r.deformers = deformers;
        let mut scene = SceneFrame::default();
        scene.items_mut::<SpriteSetRefItem>().push(r);
        harness.render(&frame(scene), W, H)
    };
    let pixels = render(&mut harness, None);
    assert!(!red(&pixels, -5.0, 58.0), "uploaded without the deformer");
    let pixels = render(&mut harness, Some(id.into()));
    assert!(red(&pixels, -5.0, 58.0), "the reference turned it on");
    let pixels = render(&mut harness, None);
    assert!(red(&pixels, -5.0, 58.0), "and the set keeps it");
    let pixels = render(&mut harness, Some(SpriteDeformerSet::NONE));
    assert!(!red(&pixels, -5.0, 58.0), "until a reference turns it off");
}

#[test]
fn a_tile_layer_runs_the_deformers_it_opts_into() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some(slot) = mesh_slot(&mut harness, "tile_slot", 30.0) else {
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .register_sprite_deformer(
            &device,
            &SpriteDeformerDesc::new("lean", LEAN).with_params(slot),
        )
        .expect("registers");
    let mut map = Tilemap::new(Tileset::new(None, [8, 8], 1), [1, 1]);
    map.set(0, 0, Tile::new(0));
    let map_id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    for (deformers, moved) in [(SpriteDeformerSet::NONE, false), (id.into(), true)] {
        let mut layer = TilemapItem::new(map_id);
        layer.origin = [0.0, -20.0];
        layer.cell_size = [40.0, 40.0];
        layer.tint = RED;
        layer.deformers = deformers;
        let mut scene = SceneFrame::default();
        scene.items_mut::<TilemapItem>().push(layer);
        let pixels = harness.render(&frame(scene), W, H);
        // The cell spans 0..40; its top edge leans to 30..70.
        assert_eq!(red(&pixels, 60.0, 18.0), moved, "top edge, moved: {moved}");
        assert!(red(&pixels, 35.0, -18.0), "the bottom edge stays");
    }
}

#[test]
fn a_body_without_deform_sprite_is_refused() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let err = harness.renderer.register_sprite_deformer(
        &device,
        &SpriteDeformerDesc::new("nothing", "fn helper() -> f32 { return 1.0; }"),
    );
    assert!(err.is_err(), "no deform_sprite");
    let err = harness.renderer.register_sprite_deformer(
        &device,
        &SpriteDeformerDesc::new(
            "unbound",
            "fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32> {\n    \
             return v.world + deform_header.slot_params[0].xyz;\n}\n",
        ),
    );
    assert!(err.is_err(), "reads the header without naming a slot");
    // A failed registration leaves nothing behind.
    let mut item = square(-50.0, 40.0);
    item.deformers = SpriteDeformerSet::NONE;
    let pixels = harness.render(&sprites(vec![item]), W, H);
    assert!(red(&pixels, -50.0, 40.0), "sprites still draw");
}
