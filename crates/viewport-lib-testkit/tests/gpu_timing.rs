//! GPU timestamp readback checks: `gpu_frame_ms` must be refreshed
//! continuously, not latched. Short passes on Metal used to yield stale
//! equal-timestamp samples (the queries were resolved in the same command
//! buffer as the pass), so a sub-ms scene could report one sample over
//! hundreds of frames while `gpu_frame_ms` re-surfaced a stale value.
//! `gpu_sample_generation` distinguishes fresh measurements from
//! carried-over ones.

use viewport_lib::wgpu;
use viewport_lib::{CameraFrame, FrameData, Material, SceneFrame, SceneRenderItem};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

/// Over a run of frames on a tiny (sub-ms) scene, fresh GPU samples must
/// keep arriving: the generation counter advances for a good fraction of
/// frames and the measured time is plausibly small.
#[test]
fn gpu_samples_stay_fresh_on_short_frames() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if !h
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        eprintln!("skipping: TIMESTAMP_QUERY not supported");
        return;
    }

    let data = meshes::heightfield(32, 32, 10.0, 1.0);
    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &data)
        .expect("upload");
    let mut item = SceneRenderItem::default();
    item.mesh_id = id;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    let camera = orbit_camera(glam::Vec3::ZERO, 20.0, 0.6, 1.0);
    let fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(vec![item]),
    );

    let frames = 30;
    let mut generations = Vec::with_capacity(frames);
    for _ in 0..frames {
        let _ = h.render(&fd, 200, 150);
        generations.push(h.stats().gpu_sample_generation);
    }

    let last = *generations.last().unwrap();
    // The readback pipeline delivers roughly one fresh sample every two
    // frames; require at least a third to catch a relapse into latching
    // (which would leave the counter at 0 or 1) without being timing-exact.
    assert!(
        last >= (frames as u64) / 3,
        "expected fresh GPU samples across {frames} frames, got generation {last} \
         (a latched stream stays at 0 or 1): {generations:?}"
    );

    let gpu_ms = h.stats().gpu_frame_ms.expect("gpu_frame_ms populated");
    assert!(
        gpu_ms > 0.0 && gpu_ms < 100.0,
        "tiny scene should measure a small positive GPU time, got {gpu_ms} ms"
    );
}
