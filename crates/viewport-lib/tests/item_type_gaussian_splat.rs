//! The Gaussian splat item type holds the sets a consumer uploads to it.
//!
//! The upload calls are on `ViewportRenderer` because the sets live with the
//! item type that draws them, which `DeviceResources` cannot reach. These pin
//! handle lifetime, the async upload pair, and the byte accounting that goes
//! with owning content.
//!
//! One file per item type, so a type's coverage travels with it.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::error::ViewportError;
use viewport_lib::renderer::GaussianSplatData;
use viewport_lib::resources::UploadStatus;

fn sample_splats(n: usize) -> GaussianSplatData {
    let mut data = GaussianSplatData::default();
    data.positions = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
    data.scales = vec![[0.1, 0.1, 0.1]; n];
    data.rotations = vec![[0.0, 0.0, 0.0, 1.0]; n];
    data.opacities = vec![0.5; n];
    data
}

#[test]
fn a_stale_splat_handle_does_not_alias_after_slot_reuse() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let id1 = renderer
        .upload_gaussian_splat(&device, &queue, &sample_splats(8))
        .expect("upload a splat set");
    renderer.free_gaussian_splat(id1);

    // The next upload reuses the freed slot at a new generation.
    let id2 = renderer
        .upload_gaussian_splat(&device, &queue, &sample_splats(4))
        .expect("upload a second splat set");
    assert_ne!(id1, id2, "the reused slot must carry a new generation");

    // The live handle resolves; the stale one does not, so it cannot overwrite
    // the set now occupying its slot.
    renderer
        .replace_gaussian_splat(&device, &queue, id2, &sample_splats(2))
        .expect("replace on a live handle succeeds");
    assert!(matches!(
        renderer.replace_gaussian_splat(&device, &queue, id1, &sample_splats(2)),
        Err(ViewportError::StaleHandle { .. })
    ));
}

#[test]
fn splat_bytes_are_reported_and_reclaimed() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer
        .upload_gaussian_splat(&device, &queue, &sample_splats(8))
        .expect("upload a splat set");
    let after_upload = renderer.resident_bytes().plugin_bytes;
    assert!(
        after_upload > baseline,
        "an uploaded set must count toward the plugin working set"
    );

    // Replacing with a smaller set keeps the handle and shrinks the charge.
    renderer
        .replace_gaussian_splat(&device, &queue, id, &sample_splats(2))
        .expect("replace on a live handle succeeds");
    let after_replace = renderer.resident_bytes().plugin_bytes;
    assert!(
        after_replace < after_upload,
        "replacing with fewer splats must reduce resident bytes"
    );

    renderer.free_gaussian_splat(id);
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_gaussian_splat_validates_before_submitting() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let err = renderer
        .begin_upload_gaussian_splat(&device, &queue, GaussianSplatData::default())
        .expect_err("an empty splat list is rejected");
    assert!(matches!(
        err,
        ViewportError::InvalidGaussianSplatData { .. }
    ));
}

#[test]
fn begin_upload_gaussian_splat_drains_to_a_handle() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let job = renderer
        .begin_upload_gaussian_splat(&device, &queue, sample_splats(8))
        .expect("job submitted");
    for _ in 0..200 {
        renderer.resources_mut().process_uploads(&device, &queue);
        match renderer.resources().upload_status(job) {
            UploadStatus::Ready => break,
            UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
            UploadStatus::Pending { .. } => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            UploadStatus::Unknown => panic!("job id disappeared"),
        }
    }

    let id = renderer
        .upload_result_gaussian_splat(job)
        .expect("the finished job yields a handle");
    assert!(
        renderer.resident_bytes().plugin_bytes > 0,
        "the set is in the store once its handle is taken"
    );

    // The result is taken once; a second take has nothing to hand back.
    assert!(matches!(
        renderer.upload_result_gaussian_splat(job),
        Err(ViewportError::JobResultMissing { .. })
    ));
    renderer.free_gaussian_splat(id);
}
