//! Cluster light-index overflow must degrade deterministically.
//!
//! Every cluster owns a fixed slice of the light index list; a cluster whose
//! light demand exceeds the slice drops its lowest-priority lights and
//! nothing else. The old shared-allocator scheme let crowded clusters starve
//! other clusters of their entire light lists, with the winners decided by a
//! GPU atomic race that changed frame to frame (visible lighting flicker and
//! 2x cost swings at 200+ scene lights).

use viewport_lib::{
    CameraFrame, FrameData, LightKind, LightSource, LightingSettings, Material, SceneFrame,
    SceneRenderItem,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

/// A frame whose lights all overlap the view region, so per-cluster demand
/// far exceeds the per-cluster index capacity.
fn overflow_frame(h: &mut Harness, lights: usize) -> FrameData {
    let ground_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::heightfield(32, 32, 60.0, 1.0))
        .expect("ground");
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = ground_id;
    ground.material = Material::from_colour([0.7, 0.7, 0.7]);

    // Non-casting point lights stacked over one small area with generous
    // ranges: every cluster covering the ground sees most of them.
    let mut lighting = LightingSettings::default();
    for i in 0..lights {
        let mut s = LightSource::default();
        s.kind = LightKind::Point {
            position: [
                (i % 10) as f32 * 3.0 - 13.5,
                ((i / 10) % 10) as f32 * 3.0 - 13.5,
                6.0 + (i / 100) as f32 * 2.0,
            ],
            range: 40.0,
            radius: 0.0,
        };
        s.intensity = 0.05;
        s.cast_shadows = false;
        lighting.lights.push(s);
    }

    let camera = orbit_camera(glam::Vec3::ZERO, 80.0, 0.6, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(vec![ground]),
    );
    fd.effects.lighting = lighting;
    fd
}

#[test]
fn overflowing_clusters_drop_lights_deterministically() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = overflow_frame(&mut h, 200);
    fd.effects.debug.cluster_stats_request = true;

    // Several renders of the identical frame: the cluster build reruns every
    // frame, so any race in slice assignment shows up as pixel churn.
    let first = h.render(&fd, 200, 150);
    for _ in 0..4 {
        let again = h.render(&fd, 200, 150);
        assert_eq!(
            first, again,
            "saturated cluster frames must render identically every frame"
        );
    }

    let stats = h.renderer.cluster_stats().expect("stats requested");
    assert!(
        !stats.fallback_active,
        "test premise: 200 lights must run the clustered path"
    );
    assert!(
        stats.max_punctual > 64,
        "test premise: per-cluster demand must exceed the slice capacity; got {}",
        stats.max_punctual
    );
    assert!(
        stats.dropped_punctual_slots > 0,
        "overflowing clusters must report dropped lights"
    );
    // Every cluster that wanted lights got some: with fixed slices no cluster
    // can be starved to zero by another, so slots used covers all non-empty
    // cells (each kept at least min(demand, capacity) > 0 lights).
    assert!(
        stats.total_index_slots_used >= stats.non_empty_cells,
        "no populated cluster may end up with an empty light list; {} slots over {} cells",
        stats.total_index_slots_used,
        stats.non_empty_cells
    );
}
