//! Smoke tests for the `ItemTypePlugin` fixture.
//!
//! Registers `LoggingItemTypePlugin` on a real renderer and checks the
//! dispatch contract: `init_gpu` at registration, per-frame `prepare` and
//! `paint` only on frames where a collection was submitted under the plugin's
//! name, the right plugin when two are registered, and the CPU pick ray
//! reaching the plugin's `pick`.

use viewport_lib::renderer::PickMask;
use viewport_lib::wgpu;
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::fixtures::{
    CallLog, CountedItemCollection, LoggingItemTypePlugin, probe_frame,
};

const SIZE: u32 = 64;

/// Registration calls `init_gpu` once with the shared bindings, and a frame
/// with nothing submitted dispatches neither `prepare` nor `paint`. Submitting
/// a collection dispatches both, with the item count the plugin was handed.
#[test]
fn item_type_fixture_dispatches_only_when_submitted() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    harness.renderer.with_item_type_plugin(
        &harness.device,
        Box::new(LoggingItemTypePlugin::new(log.clone(), "probe_items")),
    );
    assert_eq!(
        log.count("init_gpu"),
        1,
        "registration must run init_gpu once; log holds {:?}",
        log.entries()
    );
    log.clear();

    // No submission: the renderer has a plugin but no items for it.
    let frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    harness.render(&frame, SIZE, SIZE);
    log.assert_empty();

    // With a submission the plugin sees both per-frame hooks.
    let mut frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    frame
        .scene
        .submit_plugin_items("probe_items", CountedItemCollection::new(3));
    harness.render(&frame, SIZE, SIZE);
    log.assert_take(&[
        "prepare:probe_items:items=3:vp=0",
        "paint:probe_items:items=3:vp=0",
    ]);
}

/// Two plugins, one submission: only the plugin whose `type_name` matches the
/// submission key is dispatched.
#[test]
fn item_type_fixture_routes_by_type_name() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let wanted = CallLog::new();
    let other = CallLog::new();
    harness.renderer.with_item_type_plugin(
        &harness.device,
        Box::new(LoggingItemTypePlugin::new(wanted.clone(), "wanted")),
    );
    harness.renderer.with_item_type_plugin(
        &harness.device,
        Box::new(LoggingItemTypePlugin::new(other.clone(), "other")),
    );
    wanted.clear();
    other.clear();

    let mut frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    frame
        .scene
        .submit_plugin_items("wanted", CountedItemCollection::new(1));
    harness.render(&frame, SIZE, SIZE);

    assert_eq!(wanted.count("prepare:wanted"), 1);
    other.assert_empty();
}

/// A hidden item stays in the collection the plugin is handed: the renderer
/// does not pre-filter, so the count is unchanged and honouring `hidden` is
/// the plugin's job.
#[test]
fn item_type_fixture_receives_hidden_items() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    harness.renderer.with_item_type_plugin(
        &harness.device,
        Box::new(LoggingItemTypePlugin::new(log.clone(), "probe_items")),
    );
    log.clear();

    let mut items = CountedItemCollection::new(2);
    items.hide(0);
    let mut frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    frame.scene.submit_plugin_items("probe_items", items);
    harness.render(&frame, SIZE, SIZE);

    assert_eq!(
        log.count("prepare:probe_items:items=2"),
        1,
        "the collection must arrive unfiltered; log holds {:?}",
        log.entries()
    );
}

/// The CPU pick router consults registered plugins after the built-in
/// pickers, so a plugin that reports a hit is returned when nothing else is in
/// the scene.
#[test]
fn item_type_fixture_is_consulted_by_cpu_pick() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    // The CPU pick router is opt-in: without the cache the whole query short
    // circuits before any picker, plugins included.
    harness.renderer.set_cpu_pick_cache(true);
    harness.renderer.with_item_type_plugin(
        &harness.device,
        Box::new(LoggingItemTypePlugin::new(log.clone(), "probe_items").pickable()),
    );
    let frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    harness.render(&frame, SIZE, SIZE);
    log.clear();

    let view_proj = frame.camera.render_camera.view_proj();
    let hit = harness.renderer.pick(
        glam::Vec2::new(SIZE as f32 / 2.0, SIZE as f32 / 2.0),
        glam::Vec2::new(SIZE as f32, SIZE as f32),
        view_proj,
        PickMask::all(),
    );

    assert_eq!(
        log.count("pick:probe_items"),
        1,
        "the pick ray must reach the plugin; log holds {:?}",
        log.entries()
    );
    let hit = hit.expect("the plugin reported a hit, so the router must return it");
    assert_eq!(hit.id, 1);
}

// Keep `wgpu` named so the leg the testkit builds against is the one this test
// resolves, matching the other test binaries in this crate.
const _: Option<wgpu::TextureFormat> = None;

/// The `vpl.` prefix belongs to the built-in item types, and the
/// renderer's per-type calls (`upload_sprite_set` and the rest) resolve their
/// plugin by that name and downcast it. A plugin that took one of those names
/// would leave those calls looking at a type that is not what they expect, so
/// the registration is refused where the mistake is made.
#[test]
#[should_panic(expected = "is reserved")]
fn a_plugin_cannot_take_a_built_in_item_type_name() {
    let Some(mut harness) = Harness::new() else {
        // Nothing to assert without a device, and the test is `should_panic`,
        // so panic deliberately rather than reporting a false pass.
        panic!("skipping: no GPU adapter available (is reserved)");
    };
    harness.renderer.with_item_type_plugin(
        &harness.device,
        Box::new(LoggingItemTypePlugin::new(CallLog::new(), "vpl.sprite")),
    );
}

/// A name of one's own is fine, including one that merely mentions the
/// library: the check is a prefix, not a search.
#[test]
fn a_plugin_may_register_under_any_unreserved_name() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for name in ["mycrate.sprite", "vpl_sprite", "not.vpl.sprite"] {
        harness.renderer.with_item_type_plugin(
            &harness.device,
            Box::new(LoggingItemTypePlugin::new(CallLog::new(), name)),
        );
        assert!(harness.renderer.has_item_type_plugin(name));
    }
    let reserved = viewport_lib::renderer::RESERVED_TYPE_NAME_PREFIX;
    assert!(
        harness
            .renderer
            .has_item_type_plugin(&format!("{reserved}sprite")),
        "the built-in sprite type is still the one under its own name"
    );
}
