//! Device loss: the watcher's callback path, the renderer's lost state, and
//! recovery on a new device.
//!
//! The loss we can trigger deterministically is an explicit `device.destroy()`.
//! A real loss (driver reset, watchdog kill) takes the same callback with
//! `DeviceLostReason::Unknown`, which is what flips `is_lost()`; the renderer
//! treats both the same.

use std::path::Path;

use viewport_lib::wgpu;
use viewport_lib::{
    DeviceLostWatcher, NoticeDetail, RendererConfig, ViewportError, ViewportInstance,
    ViewportRenderer, primitives,
};
use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::{Harness, catalogue, frame_for, headless_device};

const W: u32 = 400;
const H: u32 = 300;
const SCENE: &str = "primitives_trio";

#[test]
fn destroy_reports_through_watcher() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let watcher = DeviceLostWatcher::install(&device);
    assert!(!watcher.is_lost());
    assert!(!watcher.is_lost_or_destroyed());
    assert!(watcher.info().is_none());

    device.destroy();
    // The callback is delivered on maintain, not at the destroy call.
    let _ = device.poll(wgpu::PollType::Poll);

    let info = watcher
        .info()
        .expect("device-lost callback did not fire after destroy + poll");
    assert_eq!(info.reason, wgpu::DeviceLostReason::Destroyed);
    // An explicit destroy is not a loss to `is_lost`, but the renderer's test
    // counts it.
    assert!(!watcher.is_lost());
    assert!(watcher.is_lost_or_destroyed());
}

/// Render `SCENE` and return the settled second frame.
fn render_scene(harness: &mut Harness) -> RgbaImage {
    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == SCENE)
        .expect("catalogue scene");
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let _ = harness.render(&frame, W, H);
    let pixels = harness.render(&frame, W, H);
    RgbaImage::from_raw(W, H, pixels).expect("frame -> image")
}

/// After `device.destroy()` the renderer is lost for good: frames are empty,
/// uploads fail, one notice is reported. A renderer built on a new device then
/// draws the catalogue scene exactly as its golden reference.
#[test]
fn lost_renderer_is_inert_and_a_new_device_matches_the_golden() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(harness.backend_dir_name());

    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == SCENE)
        .expect("catalogue scene");
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &primitives::cube(1.0))
        .expect("upload on a live device");
    let _ = harness.render(&frame, W, H);
    let _ = harness.renderer.take_notices();
    assert!(!harness.renderer.is_device_lost());
    assert!(!harness.renderer.last_frame_stats().device_lost);

    // A target made while the device was live, for the paint below.
    let target = harness.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lost_paint_target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: Harness::DEFAULT_TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    harness.device.destroy();
    let _ = harness.device.poll(wgpu::PollType::Poll);

    let renderer = &mut harness.renderer;
    assert!(renderer.is_device_lost());
    assert!(renderer.device_lost_watcher().is_lost_or_destroyed());
    let info = renderer.device_lost_info().expect("lost info");
    assert_eq!(info.reason, wgpu::DeviceLostReason::Destroyed);

    let prepared = renderer
        .pass()
        .prepare(&harness.device, &harness.queue, &frame);
    assert!(
        prepared.is_empty(),
        "prepare on a lost device returned work"
    );
    {
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lost_paint"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        renderer.pass_view().paint(&mut pass, &frame);
    }
    let _ = renderer
        .owned()
        .render(&harness.device, &harness.queue, &view, &frame);
    assert!(
        renderer
            .render_offscreen(&harness.device, &harness.queue, &frame, W, H)
            .is_empty(),
        "a readback on a lost device returned pixels"
    );

    let upload = renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &primitives::cube(1.0));
    assert!(
        matches!(upload, Err(ViewportError::DeviceLost { .. })),
        "upload on a lost device returned {upload:?}"
    );

    let stats = renderer.last_frame_stats();
    assert!(stats.device_lost);
    assert_eq!(stats.frames_skipped_device_lost, 3);

    let notices = renderer.take_notices();
    let lost: Vec<_> = notices
        .iter()
        .filter(|n| matches!(n.detail, NoticeDetail::DeviceLost { .. }))
        .collect();
    assert_eq!(lost.len(), 1, "device-lost notices: {notices:?}");
    assert_eq!(lost[0].count, 1);
    drop(harness);

    // Recovery is a new renderer on a new device.
    let Some(mut rebuilt) = Harness::new() else {
        panic!("no adapter for the rebuilt device");
    };
    assert!(!rebuilt.renderer.is_device_lost());
    let actual = render_scene(&mut rebuilt);
    if !dir.join(format!("{SCENE}.png")).exists() {
        eprintln!(
            "no {SCENE} reference in {}: rendered on the new device, not compared",
            dir.display()
        );
        return;
    }
    let tol = golden::load_tolerances(&dir.join("tolerances.txt"))
        .get(SCENE)
        .copied()
        .unwrap_or(Tolerance::EXACT);
    match golden::check(&dir, SCENE, &actual, tol, GoldenMode::Compare) {
        Outcome::Match => {}
        other => panic!("{SCENE} on the rebuilt device: {other:?}"),
    }
}

/// A renderer built on a device that is already gone refuses to build.
#[test]
fn building_on_a_destroyed_device_returns_device_lost() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let watcher = DeviceLostWatcher::install(&device);
    device.destroy();
    let _ = device.poll(wgpu::PollType::Poll);
    let config =
        RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT).with_device_lost_watcher(watcher);
    assert!(matches!(
        ViewportRenderer::try_with_config(&device, &config),
        Err(ViewportError::DeviceLost { .. })
    ));
}

/// Renderers sharing one watcher both see the loss, where each installing its
/// own would leave only the last one hearing it.
#[test]
fn a_shared_watcher_reaches_every_renderer() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let watcher = DeviceLostWatcher::install(&device);
    let config = RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT)
        .with_device_lost_watcher(watcher.clone());
    let a = ViewportRenderer::try_with_config(&device, &config).expect("renderer a");
    let b = ViewportRenderer::try_with_config(&device, &config).expect("renderer b");
    device.destroy();
    let _ = device.poll(wgpu::PollType::Poll);
    assert!(a.is_device_lost());
    assert!(b.is_device_lost());
}

/// `ViewportInstance::rebuild` moves an instance to a new device, keeps its
/// scene and camera, and names the handles the host has to upload again.
#[test]
fn instance_rebuild_keeps_the_scene_and_lists_stale_handles() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut instance = ViewportInstance::try_with_config(
        &device,
        &RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT),
    )
    .expect("instance");
    let material = instance
        .resources_mut()
        .upload_material(viewport_lib::Material::default())
        .expect("material");
    let (node, mesh) = instance
        .upload_and_add(
            &device,
            &primitives::cube(1.0),
            glam::Mat4::IDENTITY,
            material,
        )
        .expect("mesh");
    instance.camera_mut().set_orbit_distance(7.5);
    device.destroy();
    let _ = device.poll(wgpu::PollType::Poll);
    assert!(instance.renderer_mut().is_device_lost());

    let Some((new_device, _new_queue)) = headless_device() else {
        panic!("no adapter for the rebuilt device");
    };
    let stale = instance.rebuild(&new_device).expect("rebuild");
    assert!(!instance.renderer_mut().is_device_lost());
    assert_eq!(stale.meshes(), &[mesh]);
    assert_eq!(stale.materials(), &[material]);
    assert!(stale.textures().is_empty());
    assert!(instance.scene().node(node).is_some());
    assert_eq!(instance.camera().focus_distance(), 7.5);

    // The host uploads again and rebinds.
    let new_mesh = instance
        .resources_mut()
        .upload_mesh_data(&new_device, &primitives::cube(1.0))
        .expect("upload on the new device");
    instance.scene_mut().set_mesh(node, Some(new_mesh));
}

/// Everything the stale-plugin test registers, in one place so the first
/// registration and the one after `rebuild` match.
const STALE_SIZE: u32 = 64;
const STALE_DEFORMER: &str = "stale_test_offset";
const STALE_PLUGIN: &str = "stale_test_flat";
const STALE_HOOK: &str = "stale_test_hook";
const STALE_ITEM_TYPE: &str = "stale_test.items";
const STALE_STAGE: &str = "stale_test_stage";
const STALE_STAGE_ORDER: i32 = 100;

fn stale_hook_desc() -> viewport_lib::ShadingHookDesc {
    viewport_lib::ShadingHookDesc {
        name: STALE_HOOK.into(),
        wgsl_body: "\
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return vec3<f32>(0.0);
}
"
        .to_string(),
        needs_back_hemisphere: false,
        texture_count: 0,
        reads_vertex_attribute: false,
        params: Vec::new(),
    }
}

/// Upload the probe quad, deform it with the offset deformer `deformer`, give
/// it a material shaded by `plugin`, and render it.
fn render_stale_scene(
    instance: &mut ViewportInstance,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    deformer: Option<viewport_lib::DeformerId>,
    plugin: Option<viewport_lib::MaterialPluginId>,
) -> Vec<u8> {
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
    let mut material =
        viewport_lib::Material::from_colour(viewport_lib::Colour::linear_rgb(0.8, 0.1, 0.1));
    material.shading_plugin = plugin;
    let mut item = viewport_lib::MeshItem::default();
    item.mesh_id = mesh_id;
    item.material = resources.upload_material(material).expect("material");
    let mut frame = probe_frame(STALE_SIZE, [0.0, 0.0, 0.0, 1.0]);
    frame.scene.surfaces = viewport_lib::SurfaceSubmission::Flat(vec![item].into());
    instance
        .renderer_mut()
        .render_offscreen(device, queue, &frame, STALE_SIZE, STALE_SIZE)
}

/// `rebuild` lists every plugin the host registered on the old renderer, by
/// the id and name it was registered under, and registering each again from
/// the lists draws the frame as before: the mesh deformed and its material
/// shaded by the plugin.
#[test]
fn instance_rebuild_lists_the_plugins_to_register_again() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use viewport_lib::PostEffectSlot;
    use viewport_lib_testkit::DeviceProfile;
    use viewport_lib_testkit::fixtures::{
        CallLog, FlatColourMaterialPlugin, LoggingItemTypePlugin, LoggingPostEffectProducer,
        PassthroughPostEffectStage, constant_offset_deformer_named,
    };

    let Some(harness) = Harness::with_profile(&DeviceProfile::low_power("stale-plugins")) else {
        eprintln!("skipping: no GPU adapter with recommended limits");
        return;
    };
    let (device, queue) = (harness.device.clone(), harness.queue.clone());
    let mut instance = ViewportInstance::try_with_config(
        &device,
        &RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT),
    )
    .expect("instance");
    let log = CallLog::new();
    let disabled = || Arc::new(AtomicBool::new(false));
    let plugin = FlatColourMaterialPlugin::new(STALE_PLUGIN, [0.0, 0.9, 0.0]);

    // Register one of each kind, the way a host does. Post effects stay
    // disabled so they leave the image alone.
    let register = |instance: &mut ViewportInstance| {
        let resources = instance.resources_mut();
        let deformer = resources
            .register_deformer(&device, constant_offset_deformer_named(STALE_DEFORMER))
            .ok()?;
        let plugin_id = resources.register_material_plugin(&device, &plugin).ok()?;
        let hook = resources
            .register_shading_hook(&device, stale_hook_desc())
            .ok()?;
        let item_type = instance
            .with_item_type_plugin(
                &device,
                Box::new(LoggingItemTypePlugin::new(log.clone(), STALE_ITEM_TYPE)),
            )
            .ok()?;
        let renderer = instance.renderer_mut();
        let producer = renderer
            .add_post_effect_producer(Box::new(LoggingPostEffectProducer::new(
                log.clone(),
                PostEffectSlot::Bloom,
                disabled(),
            )))
            .ok()?;
        let stage = renderer
            .add_post_effect_stage(
                Box::new(PassthroughPostEffectStage::new(
                    &device,
                    log.clone(),
                    STALE_STAGE,
                    0.5,
                    disabled(),
                )),
                STALE_STAGE_ORDER,
            )
            .ok()?;
        Some((deformer, plugin_id, hook, item_type, producer, stage))
    };

    let Some((deformer, plugin_id, hook, item_type, producer, stage)) = register(&mut instance)
    else {
        eprintln!("skipping: this device cannot register every plugin kind");
        return;
    };
    let before = render_stale_scene(
        &mut instance,
        &device,
        &queue,
        Some(deformer),
        Some(plugin_id),
    );

    let stale = instance.rebuild(&device).expect("rebuild");
    assert_eq!(stale.deformers().len(), 1, "{:?}", stale.deformers());
    assert_eq!(stale.deformers()[0].id, deformer);
    assert_eq!(stale.deformers()[0].name, STALE_DEFORMER);
    assert_eq!(stale.material_plugins().len(), 1);
    assert_eq!(stale.material_plugins()[0].id, plugin_id);
    assert_eq!(stale.material_plugins()[0].name, STALE_PLUGIN);
    assert_eq!(
        stale.shading_hooks().len(),
        1,
        "{:?}",
        stale.shading_hooks()
    );
    assert_eq!(stale.shading_hooks()[0].id, hook);
    assert_eq!(stale.shading_hooks()[0].name, STALE_HOOK);
    assert_eq!(stale.item_types().len(), 1, "{:?}", stale.item_types());
    assert_eq!(stale.item_types()[0].id, item_type);
    assert_eq!(stale.item_types()[0].type_name, STALE_ITEM_TYPE);
    assert_eq!(stale.post_effect_producers().len(), 1);
    assert_eq!(stale.post_effect_producers()[0].id, producer);
    assert_eq!(
        stale.post_effect_producers()[0].type_name,
        "logging_post_effect_producer"
    );
    assert_eq!(stale.post_effect_producers()[0].slot, PostEffectSlot::Bloom);
    assert_eq!(stale.post_effect_stages().len(), 1);
    assert_eq!(stale.post_effect_stages()[0].id, stage);
    assert_eq!(stale.post_effect_stages()[0].type_name, STALE_STAGE);
    assert_eq!(stale.post_effect_stages()[0].order, STALE_STAGE_ORDER);

    // Nothing is registered on the new renderer, so the same scene draws
    // undeformed and with built-in shading.
    let lost = render_stale_scene(&mut instance, &device, &queue, None, None);
    assert_ne!(before, lost, "test premise: the plugins change the image");

    // Register again from the lists, mapping old ids to new ones.
    let mut new_deformer = None;
    for old in stale.deformers() {
        assert_eq!(old.name, STALE_DEFORMER);
        new_deformer = Some(
            instance
                .resources_mut()
                .register_deformer(&device, constant_offset_deformer_named(old.name.clone()))
                .expect("register the deformer again"),
        );
    }
    let mut plugin_ids = std::collections::HashMap::new();
    for old in stale.material_plugins() {
        assert_eq!(old.name, STALE_PLUGIN);
        let new = instance
            .resources_mut()
            .register_material_plugin(&device, &plugin)
            .expect("register the material plugin again");
        plugin_ids.insert(old.id, new);
    }
    let after = render_stale_scene(
        &mut instance,
        &device,
        &queue,
        new_deformer,
        plugin_ids.get(&plugin_id).copied(),
    );
    assert!(
        before == after,
        "registering again from the lists draws as before"
    );
}
