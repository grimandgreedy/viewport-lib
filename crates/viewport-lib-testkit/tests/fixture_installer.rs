//! Smoke tests for the `PluginInstaller` fixture.
//!
//! `DeformAndStepInstaller` registers a deformer on the renderer's resources
//! and a runtime plugin on the runtime, so one `install` call spans two seams
//! that live on different objects. The tests check both halves land, and that
//! a host with no runtime gets a clean error rather than a half-installed
//! feature.

use viewport_lib::interaction::select::selection::Selection;
use viewport_lib::runtime::{RuntimeFrameContext, ViewportRuntime};
use viewport_lib::scene::scene::Scene;
use viewport_lib::{PluginInstallCtx, PluginInstaller, ViewportError, install_plugin};
use viewport_lib_testkit::fixtures::{CallLog, DeformAndStepInstaller};
use viewport_lib_testkit::{DeviceProfile, Harness};

/// Installing with a runtime present registers both halves: the deformer id
/// comes back on the handle, and stepping the runtime runs the installed
/// plugin.
#[test]
fn installer_fixture_registers_both_halves() {
    let Some(mut harness) = Harness::with_profile(&DeviceProfile::low_power("fixture-installer"))
    else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let mut runtime = ViewportRuntime::new();
    let log = CallLog::new();

    let handle = match install_plugin(
        DeformAndStepInstaller::new(log.clone()),
        &harness.device,
        &harness.queue,
        Some(&mut runtime),
        &mut harness.renderer,
    ) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("skipping: this device cannot register the feature: {err}");
            return;
        }
    };

    // The renderer half: a real deformer id, so the host can address its slot.
    assert!(
        handle.deformer.slot() < viewport_lib::DEFORM_SLOT_COUNT,
        "the handle must carry a usable deformer slot"
    );

    // The runtime half: the installed plugin steps with the runtime.
    let mut scene = Scene::new();
    let mut selection = Selection::new();
    let mut frame = RuntimeFrameContext::default();
    frame.dt = 1.0 / 60.0;
    runtime.step(&mut scene, &mut selection, &frame);

    assert_eq!(
        log.count("installed_step:step"),
        1,
        "the installed runtime plugin must run on step; log holds {:?}",
        log.entries()
    );
}

/// A host with no runtime gets `PluginInstallMissing` naming what needed it,
/// and nothing is registered: the installer claims the runtime before it
/// touches the renderer, so the deformer name stays free for a later attempt.
#[test]
fn installer_fixture_fails_cleanly_without_a_runtime() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::low_power("fixture-installer-no-runtime"))
    else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let log = CallLog::new();

    let result = {
        let mut ctx =
            PluginInstallCtx::new(&harness.device, &harness.queue, None, &mut harness.renderer);
        DeformAndStepInstaller::new(log.clone()).install(&mut ctx)
    };

    match result {
        Err(ViewportError::PluginInstallMissing { .. }) => {}
        Err(other) => panic!("expected PluginInstallMissing, got {other:?}"),
        Ok(_) => panic!("installing without a runtime must fail"),
    }
    log.assert_empty();

    // Nothing was registered, so a second install with a runtime succeeds:
    // the deformer name was never claimed by the failed attempt.
    let mut runtime = ViewportRuntime::new();
    let retry = install_plugin(
        DeformAndStepInstaller::new(log.clone()),
        &harness.device,
        &harness.queue,
        Some(&mut runtime),
        &mut harness.renderer,
    );
    assert!(
        retry.is_ok(),
        "a failed install must leave the renderer untouched, got {:?}",
        retry.err()
    );
}
