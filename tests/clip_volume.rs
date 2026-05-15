//! Integration tests for clip volumes / clip shapes.
//!
//! These tests verify the CPU-side API: shape construction, EffectsFrame
//! defaults, and uniform struct sizing. No GPU device is required.

use viewport_lib::{
    CLIP_VOLUME_MAX, ClipObject, ClipShape, ClipVolumeEntry, ClipVolumesUniform,
    renderer::FrameData,
};

#[test]
fn clip_volume_entry_size_is_96() {
    assert_eq!(
        std::mem::size_of::<ClipVolumeEntry>(),
        96,
        "ClipVolumeEntry must be 96 bytes to match the WGSL ClipVolumeEntry struct"
    );
}

#[test]
fn clip_volumes_uniform_size() {
    let expected = 16 + CLIP_VOLUME_MAX * 96;
    assert_eq!(
        std::mem::size_of::<ClipVolumesUniform>(),
        expected,
        "ClipVolumesUniform must be 16 (header) + CLIP_VOLUME_MAX * 96 bytes"
    );
}

#[test]
fn frame_data_default_clip_objects_is_empty() {
    let frame = FrameData::default();
    assert!(frame.effects.clip_objects.is_empty());
}

#[test]
fn clip_objects_construct_and_assign() {
    let mut frame = FrameData::default();

    // Plane variant
    let mut obj = ClipObject::default();
    obj.shape = ClipShape::Plane {
        normal: [0.0, 1.0, 0.0],
        distance: -5.0,
        cap_colour: None,
        display_center: None,
    };
    frame.effects.clip_objects.push(obj);
    assert!(matches!(
        frame.effects.clip_objects[0].shape,
        ClipShape::Plane { .. }
    ));
    frame.effects.clip_objects.clear();

    // Box variant
    let mut obj = ClipObject::default();
    obj.shape = ClipShape::Box {
        center: [1.0, 2.0, 3.0],
        half_extents: [0.5, 0.5, 0.5],
        orientation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    frame.effects.clip_objects.push(obj);
    assert!(matches!(
        frame.effects.clip_objects[0].shape,
        ClipShape::Box { .. }
    ));
    frame.effects.clip_objects.clear();

    // Sphere variant
    let mut obj = ClipObject::default();
    obj.shape = ClipShape::Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 2.5,
    };
    frame.effects.clip_objects.push(obj);
    assert!(matches!(
        frame.effects.clip_objects[0].shape,
        ClipShape::Sphere { .. }
    ));
    frame.effects.clip_objects.clear();

    assert!(frame.effects.clip_objects.is_empty());
}

#[test]
fn clip_volume_entry_from_box() {
    let entry = ClipVolumeEntry::from_box(
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 1.5],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    );
    assert_eq!(entry.volume_type, 2);
    assert_eq!(entry.center, [1.0, 2.0, 3.0]);
    assert_eq!(entry.half_extents, [0.5, 1.0, 1.5]);
    assert_eq!(entry.col0, [1.0, 0.0, 0.0]);
    assert_eq!(entry.col1, [0.0, 1.0, 0.0]);
    assert_eq!(entry.col2, [0.0, 0.0, 1.0]);
}

#[test]
fn clip_volume_entry_from_sphere() {
    let entry = ClipVolumeEntry::from_sphere([5.0, 0.0, -2.0], 3.14);
    assert_eq!(entry.volume_type, 3);
    assert_eq!(entry.center, [5.0, 0.0, -2.0]);
    assert!((entry.radius - 3.14).abs() < 1e-5);
}

#[test]
fn clip_volumes_uniform_packs_multiple() {
    // Verify the zeroed uniform starts with count=0.
    let u: ClipVolumesUniform = bytemuck::Zeroable::zeroed();
    assert_eq!(u.count, 0);
    assert_eq!(u.volumes[0].volume_type, 0);

    // A manually assembled uniform with two entries.
    let mut u: ClipVolumesUniform = bytemuck::Zeroable::zeroed();
    u.volumes[0] = ClipVolumeEntry::from_box(
        [0.0; 3],
        [1.0; 3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    );
    u.volumes[1] = ClipVolumeEntry::from_sphere([0.0; 3], 2.0);
    u.count = 2;
    assert_eq!(u.count, 2);
    assert_eq!(u.volumes[0].volume_type, 2);
    assert_eq!(u.volumes[1].volume_type, 3);
}
