//! The sprite item type: the batches it holds on the consumer's behalf.
//!
//! Sprite has two handle spaces over one payload, static billboards and entity
//! sprites, so the handles are checked not to be interchangeable.
//!
//! One file per item type, so a type's coverage travels with it.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;
use viewport_lib::plugin_api::Handles;

use viewport_lib::plugin_api::Uploads;
use viewport_lib::renderer::SpriteInstanceUploads;
use viewport_lib::renderer::SpriteItem;
use viewport_lib::resources::UploadStatus;

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
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
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
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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

    let id: viewport_lib::resources::SpriteSetId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(renderer.release(id));
    assert!(matches!(
        Handles::<viewport_lib::resources::SpriteSetId>::upload_result(&mut renderer, job),
        Err(viewport_lib::error::ViewportError::JobResultMissing { .. })
    ));
}
