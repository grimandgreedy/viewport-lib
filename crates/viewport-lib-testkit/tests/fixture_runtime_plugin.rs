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
        viewport_lib::Material::default(),
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
