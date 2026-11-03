//! Smoke tests for the `RuntimePlugin` fixtures.
//!
//! Drives `LoggingRuntimePlugin` through a real `ViewportRuntime` to check the
//! dispatch contract still holds from outside the library: every plugin's
//! `submit` runs in priority order before the step loop, `collect` runs after
//! it, typed events and camera commands reach the caller through
//! `RuntimeOutput`, and the bus clears between frames.

use viewport_lib::interaction::select::selection::Selection;
use viewport_lib::runtime::plugin::phase;
use viewport_lib::runtime::{CameraCommand, RuntimeFrameContext, RuntimeOutput, ViewportRuntime};
use viewport_lib::scene::scene::Scene;
use viewport_lib_testkit::fixtures::{CallLog, LoggingRuntimePlugin, ProbeEvent};

fn step(runtime: &mut ViewportRuntime, scene: &mut Scene) -> RuntimeOutput {
    let mut selection = Selection::new();
    let mut frame = RuntimeFrameContext::default();
    frame.dt = 1.0 / 60.0;
    runtime.step(scene, &mut selection, &frame)
}

/// Two plugins in different bands: the runtime runs every `submit` in priority
/// order, then the step loop, then every `collect`. Registration order is the
/// reverse of priority order here, so an assertion on the log cannot pass by
/// accident.
#[test]
fn runtime_fixtures_dispatch_in_band_order() {
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new()
        .with_plugin(LoggingRuntimePlugin::new(
            log.clone(),
            "late",
            phase::WRITEBACK,
        ))
        .with_plugin(LoggingRuntimePlugin::new(
            log.clone(),
            "early",
            phase::ANIMATE,
        ));

    let mut scene = Scene::new();
    step(&mut runtime, &mut scene);

    log.assert_take(&[
        "early:submit",
        "late:submit",
        "early:step",
        "late:step",
        "early:collect",
        "late:collect",
    ]);
}

/// The event a fixture emits from `step` is delivered as its own type, and the
/// camera command it emits alongside lands on the same bus. Both are per-frame:
/// the next step starts from an empty bus.
#[test]
fn runtime_fixture_emits_event_and_command() {
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new().with_plugin(LoggingRuntimePlugin::new(
        log.clone(),
        "emitter",
        phase::ANIMATE,
    ));
    let mut scene = Scene::new();

    let output = step(&mut runtime, &mut scene);
    assert_eq!(output.events.count::<ProbeEvent>(), 1);
    assert_eq!(
        output.events.read::<ProbeEvent>().next().unwrap(),
        &ProbeEvent { label: "emitter" }
    );
    assert_eq!(output.events.count::<CameraCommand>(), 1);

    let next = step(&mut runtime, &mut scene);
    assert_eq!(
        next.events.count::<ProbeEvent>(),
        1,
        "each frame emits exactly one event, so the bus must not accumulate"
    );
}

/// Adding a node to the scene between steps fires `on_event` before the step
/// loop, and removing it fires the counterpart.
#[test]
fn runtime_fixture_receives_lifecycle_events() {
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new().with_plugin(LoggingRuntimePlugin::new(
        log.clone(),
        "watcher",
        phase::ANIMATE,
    ));

    // First step establishes the baseline node set; no diff is reported for it.
    let mut scene = Scene::new();
    step(&mut runtime, &mut scene);
    log.clear();

    let node = scene.add(
        None,
        glam::Mat4::IDENTITY,
        viewport_lib::MaterialId::DEFAULT,
    );
    step(&mut runtime, &mut scene);
    assert_eq!(
        log.count("watcher:on_event:node_added"),
        1,
        "adding a node must reach on_event; log holds {:?}",
        log.entries()
    );
    log.assert_before("watcher:on_event:node_added", "watcher:step");

    log.clear();
    scene.remove(node);
    step(&mut runtime, &mut scene);
    assert_eq!(
        log.count("watcher:on_event:node_removed"),
        1,
        "removing a node must reach on_event; log holds {:?}",
        log.entries()
    );
}

/// A removed plugin receives no further calls, the one left keeps running, and
/// the removed plugin's timing entry is dropped straight away.
#[test]
fn runtime_fixture_removed_plugin_is_not_called() {
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new();
    let gone = runtime.add_plugin(LoggingRuntimePlugin::new(
        log.clone(),
        "gone",
        phase::ANIMATE,
    ));
    let kept = runtime.add_plugin(LoggingRuntimePlugin::new(
        log.clone(),
        "kept",
        phase::ANIMATE,
    ));
    assert_ne!(gone, kept);

    let mut scene = Scene::new();
    step(&mut runtime, &mut scene);
    assert_eq!(log.count("gone:step"), 1);
    assert!(runtime.last_stats().step_ms.contains_key("gone"));
    log.clear();

    let removed = runtime.remove_plugin(gone).expect("the id is registered");
    assert_eq!(removed.type_name(), "gone");
    assert!(!runtime.last_stats().step_ms.contains_key("gone"));
    assert_eq!(runtime.plugin_ids().collect::<Vec<_>>(), vec![kept]);

    step(&mut runtime, &mut scene);
    log.assert_take(&["kept:submit", "kept:step", "kept:collect"]);
}

/// Removing the middle of three same-priority plugins keeps the other two in
/// registration order.
#[test]
fn runtime_fixture_removal_keeps_order() {
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new();
    let a = runtime.add_plugin(LoggingRuntimePlugin::new(log.clone(), "a", phase::ANIMATE));
    let b = runtime.add_plugin(LoggingRuntimePlugin::new(log.clone(), "b", phase::ANIMATE));
    let c = runtime.add_plugin(LoggingRuntimePlugin::new(log.clone(), "c", phase::ANIMATE));

    let mut scene = Scene::new();
    step(&mut runtime, &mut scene);
    log.clear();

    assert!(runtime.remove_plugin(b).is_some());
    assert_eq!(runtime.plugin_ids().collect::<Vec<_>>(), vec![a, c]);
    step(&mut runtime, &mut scene);
    log.assert_take(&[
        "a:submit",
        "c:submit",
        "a:step",
        "c:step",
        "a:collect",
        "c:collect",
    ]);
}

/// Removing an id twice returns `None` the second time rather than panicking
/// or taking another plugin.
#[test]
fn runtime_fixture_stale_id_returns_none() {
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new();
    let first = runtime.add_plugin(LoggingRuntimePlugin::new(
        log.clone(),
        "first",
        phase::ANIMATE,
    ));
    let _second = runtime.add_plugin(LoggingRuntimePlugin::new(
        log.clone(),
        "second",
        phase::ANIMATE,
    ));

    assert!(runtime.remove_plugin(first).is_some());
    assert!(runtime.remove_plugin(first).is_none());
    assert_eq!(runtime.plugin_count(), 1);
}
