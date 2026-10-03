//! The sprite item type: the batches it holds on the consumer's behalf.
//!
//! Sprite has two handle spaces over one payload, static billboards and entity
//! sprites, so the handles are checked not to be interchangeable.
//!
//! One file per item type, so a type's coverage travels with it.

mod common;
use common::*;
use viewport_lib::plugin_api::Handles;

use viewport_lib::plugin_api::{Uploads, Writes};
use viewport_lib::resources::UploadStatus;
use viewport_lib_item_types::SpriteInstanceUploads;
use viewport_lib_item_types::channels::sprite as sp;
use viewport_lib_item_types::{SpriteItem, SpriteSetId, SpriteSetRefItem};

/// The build log is process-wide, so a test that reads it must not overlap
/// another test building this type's pipelines.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn sample_sprites() -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    item.sizes = vec![8.0; 3];
    item
}

#[test]
fn an_uploaded_sprite_set_resolves_until_it_is_dropped() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer.upload(&device, &queue, &sample_sprites()).unwrap();
    assert!(
        renderer.resident_bytes().plugin_bytes > baseline,
        "an uploaded batch counts toward the plugin working set"
    );
    assert!(
        renderer
            .replace(&device, &queue, id, &sample_sprites())
            .is_ok()
    );

    assert!(renderer.release(id));
    assert!(!renderer.release(id), "a handle drops once");
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

/// The two handle spaces are separate stores, so an id minted in one must not
/// resolve in the other even when the slot index matches.
#[test]
fn the_two_sprite_handle_spaces_do_not_alias() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let set = renderer.upload(&device, &queue, &sample_sprites()).unwrap();
    let instance_set = renderer.upload_sprite_instance_set(&device, &queue, &sample_sprites());

    // Dropping one leaves the other live.
    assert!(renderer.release(set));
    assert!(
        renderer
            .replace_sprite_instance_set(&device, &queue, instance_set, &sample_sprites())
            .is_ok()
    );
    assert!(renderer.release(instance_set));
}

#[test]
fn begin_upload_sprite_set_drains_to_a_handle() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let job = renderer
        .begin_upload(&device, &queue, sample_sprites())
        .unwrap();
    for _ in 0..200 {
        renderer.resources_mut().process_uploads(&device, &queue);
        match renderer.upload_status(job) {
            UploadStatus::Ready => break,
            UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
            UploadStatus::Pending { .. } => std::thread::sleep(std::time::Duration::from_millis(5)),
            UploadStatus::Unknown => panic!("job id disappeared"),
        }
    }

    let id: viewport_lib_item_types::SpriteSetId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(renderer.release(id));
    assert!(matches!(
        Handles::<viewport_lib_item_types::SpriteSetId>::upload_result(&mut renderer, job),
        Err(viewport_lib::error::ViewportError::JobResultMissing { .. })
    ));
}

/// Every billboard in a selected batch goes into the outline mask, not just
/// the ones whose instance record happened to land on a stride boundary. The
/// mask shader reads the instance buffer as a storage array, so a struct that
/// does not match the record the store writes silently outlines every fourth
/// sprite.
///
/// Measured as outlined area rather than by locating each sprite: adjacent
/// outlines merge, so counting rings is unreliable, while the total area scales
/// with how many were drawn.
#[test]
fn a_selected_batch_outlines_every_billboard() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let outlined_pixels = |renderer: &mut ViewportRenderer, count: usize| {
        let mut frame = sub_object_pick_frame();
        frame.camera.render_camera.aspect = 4.0;
        frame.camera.viewport_size = [512.0, 128.0];
        frame.camera.pixels_per_point = 1.0;
        frame.interaction.outline_selected = true;
        frame.interaction.outline_colour = [1.0, 0.0, 0.0, 1.0].into();
        frame.interaction.outline_width_px = 3.0;

        let mut item = SpriteItem::default();
        item.positions = (0..count)
            .map(|i| [(i as f32 - 3.5) * 1.5, 0.0, 0.0])
            .collect();
        item.default_size = 16.0;
        item.default_colour = [0.0, 0.0, 1.0, 1.0].into();
        item.depth_write = true;
        item.settings.pick_id = PickId(1);
        item.settings.selected = true;
        frame.scene.items_mut::<SpriteItem>().push(item);

        let img = renderer.render_offscreen(&device, &queue, &frame, 512, 128);
        img.chunks_exact(4)
            .filter(|px| px[0] > 120 && px[1] < 90 && px[2] < 90)
            .count()
    };

    let mut renderer = renderer_with_item_types(&device);
    let one = outlined_pixels(&mut renderer, 1);
    let eight = outlined_pixels(&mut renderer, 8);
    assert!(one > 0, "the single-sprite case must outline something");
    assert!(
        eight >= one * 6,
        "eight selected sprites outlined {eight} pixels against {one} for one; \
         every sprite in the batch must be outlined, not every fourth"
    );
}

// ---------------------------------------------------------------------------
// Ranged writes
// ---------------------------------------------------------------------------

/// Splitting positions from the rest of the record is the point for sprites: a
/// particle feed that only moves its sprites writes a quarter of the bytes.
#[test]
fn a_reserved_sprite_batch_takes_ranged_writes_per_channel() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut batch = SpriteItem::default();
    batch.positions = vec![[0.0, 0.0, 0.0]; 100];
    let id: SpriteSetId = renderer.upload(&device, &queue, &batch).expect("upload");

    renderer
        .reserve(sp::Positions, &device, &queue, id, 10_000)
        .expect("reserve");
    // Both channels grow together: they share a sprite index, so growing one
    // alone would leave a record write addressing a buffer that cannot hold it.
    assert!(renderer.extent(sp::Positions, id).unwrap().capacity >= 10_000);
    assert!(renderer.extent(sp::Sprites, id).unwrap().capacity >= 10_000);
    assert_eq!(renderer.extent(sp::Positions, id).unwrap().len, 100);

    let moved: Vec<[f32; 3]> = (0..100).map(|i| [i as f32, 1.0, 2.0]).collect();
    renderer
        .write_range(sp::Positions, &queue, id, 100, &moved)
        .expect("position write into the reserve");
    assert_eq!(
        renderer.extent(sp::Positions, id).unwrap().len,
        200,
        "a write past the live count raises it, so the batch draws what arrived"
    );

    let recoloured = vec![
        sp::Sprite {
            colour: viewport_lib::Colour::linear_rgb(1.0, 0.0, 0.0),
            size: 2.0,
            ..Default::default()
        };
        100
    ];
    renderer
        .write_range(sp::Sprites, &queue, id, 100, &recoloured)
        .expect("record write into the reserve");

    let mut frame = sub_object_pick_frame();
    frame
        .scene
        .items_mut::<SpriteSetRefItem>()
        .push(SpriteSetRefItem::new(id));
    let _ = renderer.pass().prepare(&device, &queue, &frame);
    assert!(renderer.release(id));
}

/// Lowering the live count hides sprites on both channels without freeing the
/// allocation.
#[test]
fn set_len_moves_the_sprite_draw_count() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut batch = SpriteItem::default();
    batch.positions = vec![[0.0, 0.0, 0.0]; 50];
    let id: SpriteSetId = renderer.upload(&device, &queue, &batch).expect("upload");

    renderer.set_len(sp::Positions, id, 10).expect("set_len");
    assert_eq!(renderer.extent(sp::Positions, id).unwrap().len, 10);
    assert_eq!(renderer.extent(sp::Sprites, id).unwrap().len, 10);
    assert_eq!(renderer.extent(sp::Positions, id).unwrap().capacity, 50);
    assert!(
        renderer.set_len(sp::Positions, id, 51).is_err(),
        "past the capacity is an error rather than a grow"
    );
}

/// Naming the sprite type in a warm-up builds its pipelines, so the first
/// frame that draws sprites, in either format, compiles none of them: not the
/// colour variants, and not the OIT, refraction, outline or pick pipelines.
#[test]
fn a_warmed_sprite_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default()
            .with_item_type::<viewport_lib_item_types::SpritePlugin>(),
    );
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for hdr in [true, false] {
        let mut frame = sub_object_pick_frame();
        if !hdr {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        frame.interaction.outline_selected = true;
        // Blended (OIT in HDR), depth-writing and selected, and refractive.
        let mut opaque = sample_sprites();
        opaque.depth_write = true;
        opaque.settings.pick_id = PickId(1);
        opaque.settings.selected = true;
        let mut refractive = sample_sprites();
        refractive.refraction_strength = Some(0.5);
        frame
            .scene
            .items_mut::<SpriteItem>()
            .extend([sample_sprites(), opaque, refractive]);
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        let _ = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    }
    let sprite_builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| l.starts_with("sprite") || l.starts_with("module sprite"))
        .collect();
    assert!(
        sprite_builds.is_empty(),
        "the first sprite frames built pipelines after the warm-up: {sprite_builds:?}"
    );
}
