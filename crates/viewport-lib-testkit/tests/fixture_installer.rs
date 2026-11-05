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

/// Install, render, uninstall, render: after `uninstall` the renderer's
/// registries and the runtime's plugin list read as they did before the
/// install, the removed runtime plugin is not stepped, the frame still
/// renders, and the feature installs again under the same names.
#[test]
fn installer_fixture_uninstall_restores_the_registries() {
    use viewport_lib::{Colour, Material, SceneRenderItem, SurfaceSubmission, uninstall_plugin};
    use viewport_lib_testkit::fixtures::{probe_frame, probe_quad};

    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::low_power("fixture-installer-uninstall"))
    else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let counts = |harness: &Harness| {
        let res = harness.renderer.resources();
        (
            res.registered_deformer_count(),
            res.registered_shading_hook_count(),
            res.material_plugin_stats().len(),
        )
    };
    let never_installed = counts(&harness);

    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");
    let mut frame = probe_frame(64, [0.0, 0.0, 0.0, 1.0]);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = harness
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.9, 0.9, 0.9)))
        .expect("material upload");
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let mut runtime = ViewportRuntime::new();
    let runtime_never_installed = runtime.plugin_count();
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
    let deformer = handle.deformer;
    assert_ne!(
        counts(&harness),
        never_installed,
        "install registered a piece"
    );
    assert_eq!(runtime.plugin_count(), runtime_never_installed + 1);
    let installed = harness.render(&frame, 64, 64);

    uninstall_plugin::<DeformAndStepInstaller>(
        handle,
        &harness.device,
        &harness.queue,
        Some(&mut runtime),
        &mut harness.renderer,
    )
    .expect("uninstall");
    assert_eq!(
        counts(&harness),
        never_installed,
        "uninstall must leave the registries as they were before the install"
    );
    assert_eq!(
        runtime.plugin_count(),
        runtime_never_installed,
        "uninstall must remove the runtime plugin"
    );
    let mut scene = Scene::new();
    let mut selection = Selection::new();
    let mut step_frame = RuntimeFrameContext::default();
    step_frame.dt = 1.0 / 60.0;
    log.clear();
    runtime.step(&mut scene, &mut selection, &step_frame);
    assert!(
        log.entries().is_empty(),
        "the uninstalled runtime plugin was stepped: {:?}",
        log.entries()
    );
    assert!(
        matches!(
            harness
                .renderer
                .resources_mut()
                .remove_deformer(&harness.device, deformer),
            Err(ViewportError::StaleHandle { .. })
        ),
        "the uninstalled deformer's id must no longer resolve"
    );
    let uninstalled = harness.render(&frame, 64, 64);
    assert_eq!(
        installed, uninstalled,
        "the constant-offset deformer moves nothing until a mesh attaches to it, so the \
         frame renders the same with and without the feature"
    );

    // The names are free again, so the feature installs a second time.
    let again = install_plugin(
        DeformAndStepInstaller::new(log.clone()),
        &harness.device,
        &harness.queue,
        Some(&mut runtime),
        &mut harness.renderer,
    );
    assert!(again.is_ok(), "reinstall failed: {:?}", again.err());
}

/// After the renderer is replaced and the runtime kept, `reattach_plugin`
/// registers the deformer on the new renderer and updates the handle, and the
/// runtime still holds one copy of the feature's runtime plugin.
#[test]
fn installer_fixture_reattaches_to_a_new_renderer() {
    use viewport_lib::{ViewportRenderer, reattach_plugin, uninstall_plugin};
    use viewport_lib_testkit::fixtures::ConstantOffsetDeformer;

    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::low_power("fixture-installer-reattach"))
    else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let mut runtime = ViewportRuntime::new();
    let log = CallLog::new();
    let mut handle = match install_plugin(
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
    let plugins = runtime.plugin_ids().collect::<Vec<_>>();

    harness.renderer = ViewportRenderer::new(&harness.device, Harness::DEFAULT_TARGET_FORMAT);
    assert_eq!(
        harness
            .renderer
            .resources()
            .deformer_id_by_name(ConstantOffsetDeformer::NAME),
        None
    );
    reattach_plugin::<DeformAndStepInstaller>(
        &mut handle,
        &harness.device,
        &harness.queue,
        Some(&mut runtime),
        &mut harness.renderer,
    )
    .expect("reattach");
    assert_eq!(
        harness
            .renderer
            .resources()
            .deformer_id_by_name(ConstantOffsetDeformer::NAME),
        Some(handle.deformer)
    );
    assert_eq!(runtime.plugin_ids().collect::<Vec<_>>(), plugins);

    let mut scene = Scene::new();
    let mut selection = Selection::new();
    let mut step_frame = RuntimeFrameContext::default();
    step_frame.dt = 1.0 / 60.0;
    log.clear();
    runtime.step(&mut scene, &mut selection, &step_frame);
    assert_eq!(
        log.count("installed_step:step"),
        1,
        "one runtime plugin stepped: {:?}",
        log.entries()
    );

    uninstall_plugin::<DeformAndStepInstaller>(
        handle,
        &harness.device,
        &harness.queue,
        Some(&mut runtime),
        &mut harness.renderer,
    )
    .expect("uninstall after reattach");
    assert_eq!(runtime.plugin_count(), 0);
}

/// Upload the probe quad on `instance`, move it with the fixture's
/// constant-offset deformer and render it.
fn render_offset_quad(
    instance: &mut viewport_lib::ViewportInstance,
    device: &viewport_lib::wgpu::Device,
    queue: &viewport_lib::wgpu::Queue,
    deformer: Option<viewport_lib::DeformerId>,
) -> Vec<u8> {
    use viewport_lib::{Colour, Material, SceneRenderItem, SurfaceSubmission};
    use viewport_lib_testkit::fixtures::{ConstantOffsetDeformer, probe_frame, probe_quad};

    let resources = instance.resources_mut();
    let mesh_id = resources
        .upload_mesh_data(device, &probe_quad())
        .expect("upload probe quad");
    if let Some(deformer) = deformer {
        ConstantOffsetDeformer::attach(
            resources,
            device,
            mesh_id,
            deformer.slot(),
            probe_quad().positions.len(),
        );
        ConstantOffsetDeformer::set_offset(
            resources,
            queue,
            deformer.slot(),
            glam::Vec3::new(0.25, 0.0, 0.0),
        );
    }
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = resources
        .upload_material(Material::from_colour(Colour::linear_rgb(0.9, 0.9, 0.9)))
        .expect("material upload");
    item.settings.unlit = true;
    let mut frame = probe_frame(64, [0.0, 0.0, 0.0, 1.0]);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    instance
        .renderer_mut()
        .render_offscreen(device, queue, &frame, 64, 64)
}

/// Through a `ViewportInstance`: install, `rebuild`, `reattach_plugin`. The
/// deformed mesh draws as before the rebuild and the runtime still holds one
/// copy of the feature's runtime plugin.
#[test]
fn installer_fixture_reattaches_after_instance_rebuild() {
    use viewport_lib::{RendererConfig, ViewportInstance};

    let Some(harness) =
        Harness::with_profile(&DeviceProfile::low_power("fixture-installer-rebuild"))
    else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let (device, queue) = (harness.device.clone(), harness.queue.clone());
    let mut instance = ViewportInstance::try_with_config(
        &device,
        &RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT),
    )
    .expect("instance");
    instance.set_runtime(ViewportRuntime::new());
    let log = CallLog::new();
    let mut handle =
        match instance.install_plugin(DeformAndStepInstaller::new(log.clone()), &device, &queue) {
            Ok(handle) => handle,
            Err(err) => {
                eprintln!("skipping: this device cannot register the feature: {err}");
                return;
            }
        };
    let plugins = instance
        .runtime_mut()
        .map(|r| r.plugin_ids().collect::<Vec<_>>());
    let before = render_offset_quad(&mut instance, &device, &queue, Some(handle.deformer));
    let undeformed = render_offset_quad(&mut instance, &device, &queue, None);
    assert_ne!(
        before, undeformed,
        "test premise: the deformer moves the quad"
    );

    instance.rebuild(&device).expect("rebuild");
    instance
        .reattach_plugin::<DeformAndStepInstaller>(&mut handle, &device, &queue)
        .expect("reattach");
    assert_eq!(
        instance
            .runtime_mut()
            .map(|r| r.plugin_ids().collect::<Vec<_>>()),
        plugins,
        "reattach adds no runtime plugin"
    );
    let after = render_offset_quad(&mut instance, &device, &queue, Some(handle.deformer));
    assert!(
        before == after,
        "the deformed mesh draws as before the rebuild"
    );

    instance
        .uninstall_plugin::<DeformAndStepInstaller>(handle, &device, &queue)
        .expect("uninstall after reattach");
    assert_eq!(instance.runtime_mut().map(|r| r.plugin_count()), Some(0));
}
