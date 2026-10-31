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
