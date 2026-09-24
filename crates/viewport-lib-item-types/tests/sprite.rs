//! The sprite item type: the batches it holds on the consumer's behalf.
//!
//! Sprite has two handle spaces over one payload, static billboards and entity
//! sprites, so the handles are checked not to be interchangeable.
//!
//! One file per item type, so a type's coverage travels with it.

mod common;
use common::*;
use viewport_lib::plugin_api::Handles;

use viewport_lib::plugin_api::Uploads;
use viewport_lib::resources::UploadStatus;
use viewport_lib_item_types::SpriteInstanceUploads;
use viewport_lib_item_types::SpriteItem;

fn sample_sprites() -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    item.sizes = vec![8.0; 3];
    item
}

#[test]
fn an_uploaded_sprite_set_resolves_until_it_is_dropped() {
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
