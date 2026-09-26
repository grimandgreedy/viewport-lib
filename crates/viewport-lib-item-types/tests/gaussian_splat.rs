//! The Gaussian splat item type holds the sets a consumer uploads to it.
//!
//! The upload calls are on `ViewportRenderer` because the sets live with the
//! item type that draws them, which `DeviceResources` cannot reach. These pin
//! handle lifetime, the async upload pair, and the byte accounting that goes
//! with owning content.
//!
//! One file per item type, so a type's coverage travels with it.

use viewport_lib::plugin_api::Handles;
use viewport_lib::plugin_api::{Uploads, Writes};
use viewport_lib_item_types::channels::gaussian_splat as gs;
use viewport_lib_item_types::*;

mod common;
use common::*;

use viewport_lib::error::ViewportError;
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
    let mut renderer = renderer_with_item_types(&device);

    let id1 = renderer
        .upload(&device, &queue, &sample_splats(8))
        .expect("upload a splat set");
    renderer.release(id1);

    // The next upload reuses the freed slot at a new generation.
    let id2 = renderer
        .upload(&device, &queue, &sample_splats(4))
        .expect("upload a second splat set");
    assert_ne!(id1, id2, "the reused slot must carry a new generation");

    // The live handle resolves; the stale one does not, so it cannot overwrite
    // the set now occupying its slot.
    renderer
        .replace(&device, &queue, id2, &sample_splats(2))
        .expect("replace on a live handle succeeds");
    assert!(matches!(
        renderer.replace(&device, &queue, id1, &sample_splats(2)),
        Err(ViewportError::StaleHandle { .. })
    ));
}

#[test]
fn splat_bytes_are_reported_and_reclaimed() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer
        .upload(&device, &queue, &sample_splats(8))
        .expect("upload a splat set");
    let after_upload = renderer.resident_bytes().plugin_bytes;
    assert!(
        after_upload > baseline,
        "an uploaded set must count toward the plugin working set"
    );

    // Replacing with a smaller set keeps the handle and shrinks the charge.
    renderer
        .replace(&device, &queue, id, &sample_splats(2))
        .expect("replace on a live handle succeeds");
    let after_replace = renderer.resident_bytes().plugin_bytes;
    assert!(
        after_replace < after_upload,
        "replacing with fewer splats must reduce resident bytes"
    );

    renderer.release(id);
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_gaussian_splat_validates_before_submitting() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let err = renderer
        .begin_upload(&device, &queue, GaussianSplatData::default())
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
    let mut renderer = renderer_with_item_types(&device);

    let job = renderer
        .begin_upload(&device, &queue, sample_splats(8))
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

    let id: GaussianSplatId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(
        renderer.resident_bytes().plugin_bytes > 0,
        "the set is in the store once its handle is taken"
    );

    // The result is taken once; a second take has nothing to hand back.
    assert!(matches!(
        Handles::<GaussianSplatId>::upload_result(&mut renderer, job),
        Err(ViewportError::JobResultMissing { .. })
    ));
    renderer.release(id);
}

// ---------------------------------------------------------------------------
// Ranged writes
// ---------------------------------------------------------------------------

/// Splats are the awkward case: three channels pad on the way in, and the SH
/// channel's stride is a runtime property of the set.
#[test]
fn a_reserved_splat_set_takes_ranged_writes_on_every_channel() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let n = 32usize;
    let data = GaussianSplatData {
        positions: vec![[0.0, 0.0, 0.0]; n],
        scales: vec![[0.1, 0.1, 0.1]; n],
        rotations: vec![[0.0, 0.0, 0.0, 1.0]; n],
        opacities: vec![1.0; n],
        sh_coefficients: vec![0.5; n * ShDegree::Zero.coeff_count()],
        sh_degree: ShDegree::Zero,
    };
    let id = renderer.upload(&device, &queue, &data).expect("upload");

    renderer
        .reserve(gs::Positions, &device, &queue, id, 256)
        .expect("reserve");
    let extent = renderer.extent(gs::Positions, id).expect("live handle");
    assert!(extent.capacity >= 256);
    assert_eq!(extent.len, n as u32);

    renderer
        .write_range(gs::Positions, &queue, id, 8, &[[1.0, 2.0, 3.0]; 4])
        .expect("padded position write");
    renderer
        .write_range(gs::Scales, &queue, id, 8, &[[0.2, 0.2, 0.2]; 4])
        .expect("padded scale write");
    renderer
        .write_range(gs::Rotations, &queue, id, 8, &[[0.0, 0.0, 0.0, 1.0]; 4])
        .expect("rotation write");
    renderer
        .write_range(gs::Opacities, &queue, id, 8, &[0.5; 4])
        .expect("opacity write");

    // The SH channel is addressed in splats, so four splats' worth of
    // coefficients is what a four-splat window takes.
    let per_splat = renderer
        .item_type_plugin::<GaussianSplatPlugin>(GAUSSIAN_SPLAT_TYPE_NAME)
        .and_then(|p| p.sh_coefficients_per_splat(id))
        .expect("a live set reports its SH stride");
    assert_eq!(per_splat, 3, "degree zero is three coefficients per splat");
    renderer
        .write_range(
            gs::ShCoefficients,
            &queue,
            id,
            8,
            &vec![0.25; 4 * per_splat as usize],
        )
        .expect("a whole number of splats' worth of coefficients");

    // A partial splat's worth is refused rather than shifting every colour by
    // one coefficient.
    let err = renderer
        .write_range(gs::ShCoefficients, &queue, id, 8, &[0.25, 0.25])
        .expect_err("two coefficients is not a whole splat at stride three");
    assert!(
        matches!(
            err,
            viewport_lib::error::ViewportError::ContentBufferWriteOutOfRange { .. }
        ),
        "{err}"
    );
}

/// A set uploaded with no SH coefficients has no SH channel, and writing one
/// would mean reallocating and rebinding under a call the caller thinks is cheap.
#[test]
fn a_splat_set_without_sh_refuses_an_sh_write() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let data = GaussianSplatData {
        positions: vec![[0.0, 0.0, 0.0]; 4],
        scales: vec![[0.1, 0.1, 0.1]; 4],
        rotations: vec![[0.0, 0.0, 0.0, 1.0]; 4],
        opacities: vec![1.0; 4],
        sh_coefficients: Vec::new(),
        sh_degree: ShDegree::Zero,
    };
    let id = renderer.upload(&device, &queue, &data).expect("upload");

    let err = renderer
        .write_range(gs::ShCoefficients, &queue, id, 0, &[0.5, 0.5, 0.5])
        .expect_err("no coefficients were uploaded, so there is no channel");
    assert!(
        matches!(
            err,
            viewport_lib::error::ViewportError::ChannelNotPresent { .. }
        ),
        "{err}"
    );
    // The channels that are always there still take a write.
    assert!(
        renderer
            .write_range(gs::Positions, &queue, id, 0, &[[1.0, 1.0, 1.0]])
            .is_ok()
    );
}
