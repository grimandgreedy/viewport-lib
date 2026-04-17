//! Integration tests for Phase N extended clip volumes.
//!
//! These tests verify the CPU-side API: enum construction, default values,
//! and uniform struct sizing. No GPU device is required.

use viewport_lib::{ClipVolume, ClipVolumeUniform, renderer::FrameData};

#[test]
fn clip_volume_default_is_none() {
    let v = ClipVolume::default();
    assert!(matches!(v, ClipVolume::None));
}

#[test]
fn clip_volume_uniform_size_is_128() {
    assert_eq!(
        std::mem::size_of::<ClipVolumeUniform>(),
        128,
        "ClipVolumeUniform must be exactly 128 bytes to match the WGSL layout"
    );
}

#[test]
fn frame_data_default_clip_volume_is_none() {
    let frame = FrameData::default();
    assert!(matches!(frame.clip_volume, ClipVolume::None));
}

#[test]
fn clip_volume_variants_construct_and_assign() {
    let mut frame = FrameData::default();

    // Plane variant
    frame.clip_volume = ClipVolume::Plane {
        normal: [0.0, 1.0, 0.0],
        distance: -5.0,
    };
    assert!(matches!(frame.clip_volume, ClipVolume::Plane { .. }));

    // Box variant
    frame.clip_volume = ClipVolume::Box {
        center: [1.0, 2.0, 3.0],
        half_extents: [0.5, 0.5, 0.5],
        orientation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    assert!(matches!(frame.clip_volume, ClipVolume::Box { .. }));

    // Sphere variant
    frame.clip_volume = ClipVolume::Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 2.5,
    };
    assert!(matches!(frame.clip_volume, ClipVolume::Sphere { .. }));

    // None resets to no clip
    frame.clip_volume = ClipVolume::None;
    assert!(matches!(frame.clip_volume, ClipVolume::None));
}

#[test]
fn clip_volume_uniform_from_none_has_type_zero() {
    let u = ClipVolumeUniform::from_clip_volume(&ClipVolume::None);
    assert_eq!(u.volume_type, 0);
}

#[test]
fn clip_volume_uniform_from_plane() {
    let v = ClipVolume::Plane {
        normal: [0.0, 1.0, 0.0],
        distance: 3.0,
    };
    let u = ClipVolumeUniform::from_clip_volume(&v);
    assert_eq!(u.volume_type, 1);
    assert_eq!(u.plane_normal, [0.0, 1.0, 0.0]);
    assert!((u.plane_dist - 3.0).abs() < 1e-6);
}

#[test]
fn clip_volume_uniform_from_box() {
    let v = ClipVolume::Box {
        center: [1.0, 2.0, 3.0],
        half_extents: [0.5, 1.0, 1.5],
        orientation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let u = ClipVolumeUniform::from_clip_volume(&v);
    assert_eq!(u.volume_type, 2);
    assert_eq!(u.box_center, [1.0, 2.0, 3.0]);
    assert_eq!(u.box_half_extents, [0.5, 1.0, 1.5]);
    assert_eq!(u.box_col0, [1.0, 0.0, 0.0]);
}

#[test]
fn clip_volume_uniform_from_sphere() {
    let v = ClipVolume::Sphere {
        center: [5.0, 0.0, -2.0],
        radius: 3.14,
    };
    let u = ClipVolumeUniform::from_clip_volume(&v);
    assert_eq!(u.volume_type, 3);
    assert_eq!(u.sphere_center, [5.0, 0.0, -2.0]);
    assert!((u.sphere_radius - 3.14).abs() < 1e-5);
}
