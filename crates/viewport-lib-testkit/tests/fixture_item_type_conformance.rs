//! `ConformanceItemTypePlugin` drives everything an item type does, from a crate that
//! depends on `viewport-lib` as an ordinary dependency.
//!
//! The fixture compiling at all is most of what this checks: it is written
//! against the public API and nothing else, so a route that stops being public
//! breaks the build here rather than in someone's repository. These tests are
//! the other half, checking that the reachable API also works from outside:
//! that an off-thread upload completes, that the draw reaches the frame, that a
//! pick comes back, and that a freed or replaced texture is revalidated rather
//! than pinned.

use viewport_lib::renderer::{PickBackend, PickMask};
use viewport_lib::resources::{TextureData, TextureId, UploadStatus};
use viewport_lib::{FrameData, ItemSettings, PickId, ViewportRenderer};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::fixtures::{
    ConformanceItemTypePlugin, ConformanceItems, QuadId, probe_frame,
};

const SIZE: u32 = 96;

/// A harness with the conformance type registered.
fn harness() -> Option<Harness> {
    let mut harness = Harness::new()?;
    let plugin = ConformanceItemTypePlugin::new(harness.renderer.resources(), &harness.device);
    harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(plugin));
    Some(harness)
}

/// The top view, looking down at the world XY plane where the quads stand,
/// with the grid off so nothing else is coplanar with them.
fn frame() -> FrameData {
    probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0])
}

fn solid_texture(harness: &mut Harness, colour: [u8; 4]) -> TextureId {
    let pixels: Vec<u8> = std::iter::repeat_n(colour, 4).flatten().collect();
    harness
        .renderer
        .resources_mut()
        .upload_texture(
            &harness.device,
            &harness.queue,
            TextureData::srgb(2, 2, pixels),
        )
        .expect("a 2x2 texture uploads")
}

fn plugin_mut(renderer: &mut ViewportRenderer) -> &mut ConformanceItemTypePlugin {
    renderer
        .item_type_plugin_mut::<ConformanceItemTypePlugin>(ConformanceItemTypePlugin::TYPE_NAME)
        .expect("registered under its own name")
}

/// Upload one quad through the accessor that lends out the plugin, the job
/// runner and the shared arenas at once.
fn upload(harness: &mut Harness, texture: Option<TextureId>) -> QuadId {
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let host = harness
        .renderer
        .item_type_plugin_host::<ConformanceItemTypePlugin>(ConformanceItemTypePlugin::TYPE_NAME)
        .expect("registered under its own name");
    host.plugin.upload(
        &device,
        &queue,
        host.resources,
        glam::Vec3::ZERO,
        [1.0, 1.0, 1.0],
        texture,
    )
}

fn luma(pixels: &[u8], x: u32, y: u32) -> i32 {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
}

fn submit(frame: &mut FrameData, quad: QuadId, settings: ItemSettings) {
    let mut items = ConformanceItems::default();
    items.push(quad, settings);
    frame
        .scene
        .submit_plugin_items(ConformanceItemTypePlugin::TYPE_NAME, items);
}

/// The whole round trip: upload into a plugin-owned store from outside the
/// library, submit an item naming the handle, and land pixels.
#[test]
fn an_external_item_type_stores_content_and_draws_it() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let tex = solid_texture(&mut harness, [220, 40, 40, 255]);
    let quad = upload(&mut harness, Some(tex));

    assert!(
        harness
            .renderer
            .item_type_plugin::<ConformanceItemTypePlugin>(ConformanceItemTypePlugin::TYPE_NAME)
            .is_some_and(|p| p.contains(quad)),
        "the handle resolves in the plugin's own store"
    );
    assert!(
        harness.renderer.resident_bytes().plugin_bytes > 0,
        "content a plugin holds reaches the renderer's working-set figure"
    );

    let empty = harness.render(&frame(), SIZE, SIZE);
    let mut drawn_frame = frame();
    submit(&mut drawn_frame, quad, ItemSettings::default());
    let drawn = harness.render(&drawn_frame, SIZE, SIZE);

    let background = luma(&empty, SIZE / 2, SIZE / 2);
    let painted = luma(&drawn, SIZE / 2, SIZE / 2);
    assert!(
        painted > background + 60,
        "the plugin's opaque draw must reach the frame: background {background}, drawn {painted}"
    );

    // Hiding the item puts the pixel back, so what changed was this draw.
    let mut hidden_frame = frame();
    let mut hidden = ItemSettings::default();
    hidden.hidden = true;
    submit(&mut hidden_frame, quad, hidden);
    let hidden_px = harness.render(&hidden_frame, SIZE, SIZE);
    assert_eq!(luma(&hidden_px, SIZE / 2, SIZE / 2), background);
}

/// The same store filled through the job runner instead. `Jobs::new` is not
/// public, so `item_type_plugin_host` is the only way an external type gets a
/// runner at upload time; if that stops being true this test stops compiling.
#[test]
fn an_external_item_type_uploads_off_the_frame_thread() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let tex = solid_texture(&mut harness, [40, 220, 40, 255]);
    let empty = harness.render(&frame(), SIZE, SIZE);

    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let host = harness
        .renderer
        .item_type_plugin_host::<ConformanceItemTypePlugin>(ConformanceItemTypePlugin::TYPE_NAME)
        .expect("registered");
    let job = host.plugin.begin_upload(
        &host.jobs,
        &device,
        &queue,
        host.resources,
        glam::Vec3::ZERO,
        [1.0, 1.0, 1.0],
        Some(tex),
    );

    let mut collected = None;
    for _ in 0..200 {
        harness
            .renderer
            .resources_mut()
            .process_uploads(&device, &queue);
        match harness.renderer.upload_status(job) {
            UploadStatus::Ready => {
                let host = harness
                    .renderer
                    .item_type_plugin_host::<ConformanceItemTypePlugin>(
                        ConformanceItemTypePlugin::TYPE_NAME,
                    )
                    .expect("registered");
                collected = host
                    .plugin
                    .take_upload(&host.jobs, &device, host.resources, job);
                break;
            }
            UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
            UploadStatus::Unknown => panic!("job id disappeared"),
            UploadStatus::Pending { .. } => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }

    let quad = collected.expect("the finished job yields a handle");
    let mut f = frame();
    submit(&mut f, quad, ItemSettings::default());
    let drawn = harness.render(&f, SIZE, SIZE);
    assert!(
        luma(&drawn, SIZE / 2, SIZE / 2) > luma(&empty, SIZE / 2, SIZE / 2) + 60,
        "a quad built on a worker thread draws like any other"
    );
}

/// A GPU pick returns the plugin's item, which means the pick pipeline built
/// from the published descriptors wrote into the shared id target.
#[test]
fn an_external_item_type_answers_a_gpu_pick() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let quad = upload(&mut harness, None);

    let mut f = frame();
    let mut settings = ItemSettings::default();
    settings.pick_id = PickId(77);
    submit(&mut f, quad, settings);
    let _ = harness.render(&f, SIZE, SIZE);

    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let hit = harness.renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(SIZE as f32 / 2.0, SIZE as f32 / 2.0),
        &f,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(
        hit.map(|h| h.id),
        Some(77),
        "the plugin's pick pipeline must write its id into the shared pick target"
    );
}

/// Wireframe mode and selection both reach the plugin, and it answers with
/// polylines the library draws through its own line substrate: a plugin needs
/// no pipeline of its own for either.
#[test]
fn an_external_item_type_contributes_wireframe_and_selection_lines() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let quad = upload(&mut harness, None);

    let plain = {
        let mut f = frame();
        submit(&mut f, quad, ItemSettings::default());
        harness.render(&f, SIZE, SIZE)
    };

    let mut selected_frame = frame();
    let mut selected = ItemSettings::default();
    selected.selected = true;
    submit(&mut selected_frame, quad, selected);
    let outlined = harness.render(&selected_frame, SIZE, SIZE);
    assert_ne!(
        plain, outlined,
        "a selected item's bounds wireframe has to reach the frame"
    );

    // And the frame-wide toggle reaches it too, without the item saying anything.
    let mut wire_frame = frame();
    submit(&mut wire_frame, quad, ItemSettings::default());
    wire_frame.viewport.wireframe_mode = true;
    let wire = harness.render(&wire_frame, SIZE, SIZE);
    assert_ne!(
        plain, wire,
        "ctx.wireframe_mode has to reach the plugin's wireframe hook"
    );
}

/// The point of the whole fixture. A stored entry bakes a texture view into a
/// bind group and keeps it; freeing the texture behind it must reach that bind
/// group, or the freed texture is pinned by the plugin and still sampled.
#[test]
fn a_freed_texture_is_revalidated_out_of_a_stored_bind_group() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let tex = solid_texture(&mut harness, [220, 40, 40, 255]);
    let quad = upload(&mut harness, Some(tex));

    let mut f = frame();
    submit(&mut f, quad, ItemSettings::default());
    let before = harness.render(&f, SIZE, SIZE);
    assert_eq!(
        plugin_mut(&mut harness.renderer).texture_of(quad),
        Some(tex),
        "the entry holds the host's texture to start with"
    );
    let rebinds_before = plugin_mut(&mut harness.renderer).rebind_count();

    assert!(harness.renderer.resources_mut().free_texture(tex));

    let mut after_frame = frame();
    submit(&mut after_frame, quad, ItemSettings::default());
    let after = harness.render(&after_frame, SIZE, SIZE);

    let plugin = plugin_mut(&mut harness.renderer);
    assert!(
        plugin.rebind_count() > rebinds_before,
        "the free has to be noticed: the plugin never polled its ResourceGate"
    );
    assert_eq!(
        plugin.texture_of(quad),
        None,
        "the dead id must be forgotten, not kept for a later lookup"
    );
    assert!(
        plugin.contains(quad),
        "the entry is rebound, not discarded: the host still holds its handle"
    );
    assert_ne!(
        before, after,
        "dropping to the fallback view has to change what is drawn"
    );
}

/// A replace behind a live id is the half no per-entry liveness check can see:
/// the id still resolves, so only the view epoch says anything happened.
#[test]
fn a_replaced_texture_reaches_a_stored_bind_group() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let tex = solid_texture(&mut harness, [220, 40, 40, 255]);
    let quad = upload(&mut harness, Some(tex));

    let mut f = frame();
    submit(&mut f, quad, ItemSettings::default());
    let before = harness.render(&f, SIZE, SIZE);

    let pixels: Vec<u8> = std::iter::repeat_n([40u8, 40, 220, 255], 4)
        .flatten()
        .collect();
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    harness
        .renderer
        .resources_mut()
        .replace_texture(&device, &queue, tex, TextureData::srgb(2, 2, pixels))
        .expect("the replacement uploads");

    let mut after_frame = frame();
    submit(&mut after_frame, quad, ItemSettings::default());
    let after = harness.render(&after_frame, SIZE, SIZE);

    assert_eq!(
        plugin_mut(&mut harness.renderer).texture_of(quad),
        Some(tex),
        "a replace does not invalidate the id, only the view behind it"
    );
    assert_ne!(
        before, after,
        "the new pixels have to reach a bind group built before the swap"
    );
}

/// A handle to a freed slot must not resolve once the slot is reused. This is
/// the generational-id property, and it is the plugin's to get right: the
/// library hands out no store.
#[test]
fn a_reused_slot_does_not_resurrect_an_old_handle() {
    let Some(mut harness) = harness() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let first = upload(&mut harness, None);
    assert!(plugin_mut(&mut harness.renderer).free(first));

    let second = upload(&mut harness, None);
    let plugin = plugin_mut(&mut harness.renderer);
    assert!(plugin.contains(second));
    assert!(
        !plugin.contains(first),
        "the freed handle must not resolve through the slot its successor took"
    );
}
