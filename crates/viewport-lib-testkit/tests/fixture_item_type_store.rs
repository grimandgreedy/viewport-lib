//! An item type outside the crate can hold the content it draws, and a host
//! can upload into it through the published API.
//!
//! The built-in types that own a store upload through
//! `ViewportRenderer::item_type_plugin_host`, which lends out the plugin, the
//! job runner, and read access to the shared content arenas together. Nothing
//! else can: taking the plugin on its own borrows the whole renderer. This
//! drives `StoringItemTypePlugin` through that accessor from a crate that
//! depends on `viewport-lib` as an ordinary dependency, so the route the
//! built-ins use is checked to be one an external item type also has.

use viewport_lib::resources::UploadStatus;
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::fixtures::StoringItemTypePlugin;

/// Reach the plugin and upload into it, reading the shared texture store on
/// the way: the three things an upload path needs, from one accessor.
#[test]
fn a_host_uploads_into_a_plugin_owned_store() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(StoringItemTypePlugin::default()));

    let baseline = harness.renderer.resident_bytes().plugin_bytes;

    let host = harness
        .renderer
        .item_type_plugin_host::<StoringItemTypePlugin>(StoringItemTypePlugin::TYPE_NAME)
        .expect("the fixture is registered under its own name");
    let id = host
        .plugin
        .upload(
            &harness.device,
            &harness.queue,
            host.resources,
            &[7u8; 64],
            None,
        )
        .expect("a valid upload");

    assert!(
        harness
            .renderer
            .item_type_plugin::<StoringItemTypePlugin>(StoringItemTypePlugin::TYPE_NAME)
            .is_some_and(|p| p.contains(id)),
        "the handle resolves in the plugin's own store"
    );
    assert!(
        harness.renderer.resident_bytes().plugin_bytes > baseline,
        "content a plugin holds is visible to the renderer's working-set figure"
    );

    // And freeing gives the bytes back, so an eviction policy can act on it.
    harness
        .renderer
        .item_type_plugin_mut::<StoringItemTypePlugin>(StoringItemTypePlugin::TYPE_NAME)
        .expect("registered")
        .free(id);
    assert_eq!(harness.renderer.resident_bytes().plugin_bytes, baseline);
}

/// The same store, filled through the job runner instead: submit from the
/// host, poll, and collect. `take_upload_result` is where the handle is
/// minted, because that is the call holding `&mut` on the plugin.
#[test]
fn a_host_uploads_into_a_plugin_owned_store_off_thread() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(StoringItemTypePlugin::default()));

    let host = harness
        .renderer
        .item_type_plugin_host::<StoringItemTypePlugin>(StoringItemTypePlugin::TYPE_NAME)
        .expect("registered");
    let job = host
        .plugin
        .begin_upload(&host.jobs, &harness.device, &harness.queue, vec![3u8; 128]);

    let mut collected = None;
    for _ in 0..200 {
        harness
            .renderer
            .resources_mut()
            .process_uploads(&harness.device, &harness.queue);
        match harness.renderer.resources().upload_status(job) {
            UploadStatus::Ready => {
                let host = harness
                    .renderer
                    .item_type_plugin_host::<StoringItemTypePlugin>(
                        StoringItemTypePlugin::TYPE_NAME,
                    )
                    .expect("registered");
                collected = host.plugin.take_upload_result(&host.jobs, job);
                break;
            }
            UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
            UploadStatus::Unknown => panic!("job id disappeared"),
            UploadStatus::Pending { .. } => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }

    let id = collected.expect("the finished job yields a handle");
    assert!(
        harness
            .renderer
            .item_type_plugin::<StoringItemTypePlugin>(StoringItemTypePlugin::TYPE_NAME)
            .is_some_and(|p| p.contains(id)),
        "the collected buffer is in the plugin's store"
    );
}

/// A worker that fails reports through `UploadStatus::Failed` rather than
/// leaving a result that never arrives, which is what `try_submit_cpu` adds
/// over `submit_cpu`.
#[test]
fn a_failing_upload_job_surfaces_its_error() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(StoringItemTypePlugin::default()));

    let host = harness
        .renderer
        .item_type_plugin_host::<StoringItemTypePlugin>(StoringItemTypePlugin::TYPE_NAME)
        .expect("registered");
    // An empty payload is what this fixture rejects on the worker thread.
    let job = host
        .plugin
        .begin_upload(&host.jobs, &harness.device, &harness.queue, Vec::new());

    for _ in 0..200 {
        harness
            .renderer
            .resources_mut()
            .process_uploads(&harness.device, &harness.queue);
        match harness.renderer.resources().upload_status(job) {
            UploadStatus::Failed(_) => return,
            UploadStatus::Ready => panic!("a rejected payload must not report Ready"),
            UploadStatus::Unknown => panic!("job id disappeared"),
            UploadStatus::Pending { .. } => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }
    panic!("the failing job never resolved");
}
