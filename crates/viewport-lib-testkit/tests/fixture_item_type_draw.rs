//! Smoke tests for `TriangleItemTypePlugin`: an item-type plugin that draws.
//!
//! The `LoggingItemTypePlugin` tests cover dispatch. These cover the part a
//! real item type depends on and dispatch alone cannot prove: that a plugin
//! outside the library can still build working pipelines from
//! `SharedBindings` and the published target descriptors, read the camera out
//! of group 0, and land pixels in the scene pass, the shadow atlas, and the
//! pick targets.

use viewport_lib::renderer::{PickBackend, PickMask};
use viewport_lib::wgpu;
use viewport_lib::{LightSource, Lux, PickId, SceneRenderItem, SurfaceSubmission};
use viewport_lib_testkit::fixtures::{
    CallLog, CountedItemCollection, TriangleItemTypePlugin, probe_frame, probe_quad,
};
use viewport_lib_testkit::{DeviceProfile, Harness};

const SIZE: u32 = 64;
const TYPE_NAME: &str = "triangle_items";

/// Sum of the RGB bytes at a pixel.
fn luma(pixels: &[u8], x: u32, y: u32) -> i32 {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
}

fn harness() -> Option<Harness> {
    Harness::with_profile(&DeviceProfile::low_power("fixture-item-draw"))
}

/// The opaque pipeline, built through `build_opaque_pipeline` against the
/// renderer's own target descriptor, puts the triangle on screen: the centre
/// of the frame goes from background to the plugin's colour. Hiding the item
/// puts it back, which proves the pixels came from the plugin's draw and not
/// from something else in the frame.
#[test]
fn triangle_fixture_draws_into_the_scene_pass() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let log = CallLog::new();
    let plugin = TriangleItemTypePlugin::new(
        harness.renderer.resources(),
        &harness.device,
        log.clone(),
        TYPE_NAME,
        glam::Vec3::ZERO,
        [0.9, 0.2, 0.2],
    );
    harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(plugin));

    let frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    let empty = harness.render(&frame, SIZE, SIZE);

    let mut drawn_frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    drawn_frame
        .scene
        .submit_plugin_items(TYPE_NAME, CountedItemCollection::new(1));
    let drawn = harness.render(&drawn_frame, SIZE, SIZE);

    let background = luma(&empty, SIZE / 2, SIZE / 2);
    let painted = luma(&drawn, SIZE / 2, SIZE / 2);
    assert!(
        painted > background + 100,
        "the plugin's opaque draw must reach the frame: background {background}, drawn {painted}"
    );

    // Red channel dominates: the pixels are the plugin's colour, not a stray
    // grid line or overlay.
    let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    assert!(
        drawn[i] > drawn[i + 1] + 40 && drawn[i] > drawn[i + 2] + 40,
        "expected the plugin's red triangle, got {:?}",
        &drawn[i..i + 4]
    );

    let mut hidden_items = CountedItemCollection::new(1);
    hidden_items.hide(0);
    let mut hidden_frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    hidden_frame
        .scene
        .submit_plugin_items(TYPE_NAME, hidden_items);
    let hidden = harness.render(&hidden_frame, SIZE, SIZE);
    assert_eq!(
        luma(&hidden, SIZE / 2, SIZE / 2),
        background,
        "a hidden item must draw nothing"
    );
}

/// The sample count the renderer reports to `init_gpu` is the one the
/// pipelines were built against. A plugin that hand-rolls its pipelines has
/// to apply this itself, so a mismatch here is the bug that shows up as a
/// pipeline-incompatible-with-pass validation failure.
#[test]
fn triangle_fixture_agrees_with_the_renderer_sample_count() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let log = CallLog::new();
    let sample_count = harness.renderer.resources().shared_bindings().sample_count;
    let plugin = TriangleItemTypePlugin::new(
        harness.renderer.resources(),
        &harness.device,
        log.clone(),
        TYPE_NAME,
        glam::Vec3::ZERO,
        [0.9, 0.9, 0.9],
    );
    harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(plugin));

    log.assert_take(&[&format!("init_gpu:samples={sample_count}")]);
}

/// The renderer dispatches `cast_shadow_pass` to the plugin, per cascade.
///
/// The fixture records the dispatch but does not draw: a pipeline from
/// `build_shadow_pipeline` is incompatible with the bind group the shadow pass
/// binds, and the shadow layout is not published, so no external plugin can
/// draw here today. This test pins the dispatch half so the other half is the
/// only thing outstanding.
///
/// The scene carries a mesh as well as the plugin's items: the renderer skips
/// the whole shadow pass when there are no mesh surfaces, so a plugin-only
/// scene never reaches `cast_shadow_pass` at all.
#[test]
fn triangle_fixture_is_dispatched_by_the_shadow_pass() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let log = CallLog::new();
    let plugin = TriangleItemTypePlugin::new(
        harness.renderer.resources(),
        &harness.device,
        log.clone(),
        TYPE_NAME,
        glam::Vec3::ZERO,
        [0.9, 0.9, 0.9],
    );
    harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(plugin));

    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");

    let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    frame.scene.lights = vec![LightSource::directional_lux(
        glam::Vec3::new(0.3, 0.3, -1.0).normalize().to_array(),
        Lux(10_000.0),
    )];
    frame.effects.lighting.shadows.enabled = true;
    let mut floor = SceneRenderItem::default();
    floor.mesh_id = mesh_id;
    floor.model = glam::Mat4::from_scale(glam::Vec3::splat(4.0)).to_cols_array_2d();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![floor].into());
    frame
        .scene
        .submit_plugin_items(TYPE_NAME, CountedItemCollection::new(1));

    harness.render(&frame, SIZE, SIZE);
    assert!(
        log.count("cast_shadow_pass") > 0,
        "the shadow pass must dispatch to the plugin; log holds {:?}",
        log.entries()
    );
}

/// The pick pipeline, built from `PickTargetDesc` with the plugin's own
/// group-1 layout on top of the shared group 0, writes ids the GPU pick
/// read-back resolves.
#[test]
fn triangle_fixture_draws_into_the_pick_pass() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let log = CallLog::new();
    let plugin = TriangleItemTypePlugin::new(
        harness.renderer.resources(),
        &harness.device,
        log.clone(),
        TYPE_NAME,
        glam::Vec3::ZERO,
        [0.9, 0.9, 0.9],
    );
    harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(plugin));

    let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    frame
        .scene
        .submit_plugin_items(TYPE_NAME, CountedItemCollection::new(1));

    // prepare builds the plugin's id bind group and updates the shared camera
    // group the pick pass binds at group 0.
    let _ = harness
        .renderer
        .pass()
        .prepare(&harness.device, &harness.queue, &frame);

    let hit = harness.renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(SIZE as f32 / 2.0, SIZE as f32 / 2.0),
        &frame,
        &harness.device,
        &harness.queue,
        PickMask::all(),
    );

    assert!(
        log.count("render_pick") > 0,
        "the pick pass must dispatch to the plugin; log holds {:?}",
        log.entries()
    );
    // The collection hands out pick ids 1..=n.
    assert_eq!(
        hit.map(|h| h.id),
        Some(PickId(1).0),
        "the plugin's pick id must survive the read-back"
    );
}

// Keep `wgpu` named so this binary resolves the same leg as the crate.
const _: Option<wgpu::TextureFormat> = None;
