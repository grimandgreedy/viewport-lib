//! Crowded clusters must stay deterministic and drop no light.
//!
//! Every cluster holds a bitmask over the light array, so a cell reached by
//! more lights than the old fixed 64-entry list held still shades with every
//! one of them, renders the same as the unclustered path, and leaves every
//! other cluster alone.

use viewport_lib::Colour;
use viewport_lib::{
    CameraFrame, FrameData, LightKind, LightSource, LightingSettings, Material, MeshItem,
    SceneFrame,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

/// A frame whose lights all overlap the view region, so cells hold far more
/// than 64 lights each.
fn overflow_frame(h: &mut Harness, lights: usize) -> FrameData {
    let ground_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::heightfield(32, 32, 60.0, 1.0))
        .expect("ground");
    let mut ground = MeshItem::default();
    ground.mesh_id = ground_id;
    ground.material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material upload");

    // Non-casting point lights stacked over one small area with generous
    // ranges: every cluster covering the ground sees most of them.
    let mut lighting = LightingSettings::default();
    for i in 0..lights {
        let mut s = LightSource::default();
        s.kind = LightKind::Point {
            position: glam::Vec3::new(
                (i % 10) as f32 * 3.0 - 13.5,
                ((i / 10) % 10) as f32 * 3.0 - 13.5,
                6.0 + (i / 100) as f32 * 2.0,
            )
            .as_dvec3(),
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
fn overflowing_clusters_render_deterministically() {
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
        stats.max_punctual > 128,
        "test premise: some cell must hold more than 128 lights; got {}",
        stats.max_punctual
    );
    assert_eq!(
        stats.dropped_punctual_slots, 0,
        "a cell holds every light that reaches it"
    );
    // Every cluster that wanted lights holds them: no cluster can be starved
    // to zero by another, so the lights held cover all non-empty cells.
    assert!(
        stats.total_index_slots_used >= stats.non_empty_cells,
        "no populated cluster may end up with an empty light list; {} slots over {} cells",
        stats.total_index_slots_used,
        stats.non_empty_cells
    );
}

/// Largest per-channel difference between two RGBA8 frames.
fn max_channel_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// Render `fd` clustered and with the unclustered path forced, and assert the
/// frames match. Summation order can differ between the two paths, so the
/// comparison allows a couple of 8-bit steps.
fn assert_matches_unclustered(h: &mut Harness, mut fd: FrameData, scene: &str) {
    fd.effects.debug.cluster_stats_request = true;
    let clustered = h.render(&fd, 200, 150);
    let stats = h.renderer.cluster_stats().expect("stats requested");
    assert!(
        !stats.fallback_active && stats.max_punctual > 64 && stats.dropped_punctual_slots == 0,
        "test premise ({scene}): the clustered path must run with cells over 64 lights, dropping none"
    );

    fd.effects.debug.force_cluster_fallback = true;
    let reference = h.render(&fd, 200, 150);
    let diff = max_channel_diff(&clustered, &reference);
    const TOLERANCE: u8 = 2;
    assert!(
        diff <= TOLERANCE,
        "{scene}: overflowing clustered frame must match the unclustered reference; max channel diff {diff} > {TOLERANCE}"
    );
}

/// Overflowing cells walk the full light array, so the clustered frame matches
/// the forced unclustered frame.
#[test]
fn overflowing_clusters_match_unclustered_reference() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let fd = overflow_frame(&mut h, 200);
    assert_matches_unclustered(&mut h, fd, "200 dim lights");
    let fd = ranked_last_frame(&mut h, true);
    assert_matches_unclustered(&mut h, fd, "bright light ranked last");
}

/// A frame where 70 dark, high-importance point lights fill the centre cell's
/// slice ahead of one bright, low-importance light, which the host ranks last
/// (light index 70). A slice-only lookup would never see the bright light.
fn ranked_last_frame(h: &mut Harness, with_bright: bool) -> FrameData {
    let ground_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::heightfield(32, 32, 60.0, 0.0))
        .expect("ground");
    let mut ground = MeshItem::default();
    ground.mesh_id = ground_id;
    ground.material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material upload");

    let mut lighting = LightingSettings::default();
    lighting.hemisphere_intensity = 0.0;
    for i in 0..70 {
        let mut s = LightSource::default();
        s.kind = LightKind::Point {
            position: glam::Vec3::new(
                (i % 10) as f32 * 2.0 - 9.0,
                (i / 10) as f32 * 2.0 - 7.0,
                4.0,
            )
            .as_dvec3(),
            range: 40.0,
            radius: 0.0,
        };
        s.intensity = 0.0;
        s.importance = 10.0;
        s.cast_shadows = false;
        lighting.lights.push(s);
    }
    if with_bright {
        let mut s = LightSource::default();
        s.kind = LightKind::Point {
            position: [0.0, 0.0, 3.0].into(),
            range: 20.0,
            radius: 0.0,
        };
        s.intensity = 40.0;
        s.importance = 0.01;
        s.cast_shadows = false;
        lighting.lights.push(s);
    }

    let camera = orbit_camera(glam::Vec3::ZERO, 30.0, 0.6, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(vec![ground]),
    );
    fd.effects.lighting = lighting;
    fd
}

fn luminance_at(pixels: &[u8], width: u32, x: u32, y: u32) -> u32 {
    let i = ((y * width + x) * 4) as usize;
    pixels[i] as u32 + pixels[i + 1] as u32 + pixels[i + 2] as u32
}

/// The light past the slice capacity still reaches the fragments of the
/// overflowing cell. Without it the frame is what a slice-only lookup drew.
#[test]
fn light_past_slice_capacity_still_shades() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = ranked_last_frame(&mut h, true);
    fd.effects.debug.cluster_stats_request = true;
    let lit = h.render(&fd, 200, 150);
    let stats = h.renderer.cluster_stats().expect("stats requested");
    assert!(
        !stats.fallback_active && stats.max_punctual > 64 && stats.dropped_punctual_slots == 0,
        "test premise: 71 lights must run the clustered path with cells over 64 lights"
    );

    let without = ranked_last_frame(&mut h, false);
    let dropped = h.render(&without, 200, 150);
    let probe_lit = luminance_at(&lit, 200, 100, 75);
    let probe_dropped = luminance_at(&dropped, 200, 100, 75);
    assert!(
        probe_lit > probe_dropped + 30,
        "light index 70 must light the probe pixel: {probe_lit} with it, {probe_dropped} without"
    );
}
