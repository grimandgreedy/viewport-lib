//! Integration tests for clip volumes / clip shapes.
//!
//! These tests verify the CPU-side API: shape construction, EffectsFrame
//! defaults, and ClipVolumeUniform struct sizing. No GPU device is required.

use viewport_lib::{ClipObject, ClipShape, ClipVolumeUniform, renderer::FrameData};

#[test]
fn clip_volume_uniform_size_is_128() {
    assert_eq!(
        std::mem::size_of::<ClipVolumeUniform>(),
        128,
        "ClipVolumeUniform must be exactly 128 bytes to match the WGSL layout"
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
    obj.shape = ClipShape::Plane { normal: [0.0, 1.0, 0.0], distance: -5.0, cap_color: None };
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
    obj.shape = ClipShape::Sphere { center: [0.0, 0.0, 0.0], radius: 2.5 };
    frame.effects.clip_objects.push(obj);
    assert!(matches!(
        frame.effects.clip_objects[0].shape,
        ClipShape::Sphere { .. }
    ));
    frame.effects.clip_objects.clear();

    // Clear means no active clip objects
    assert!(frame.effects.clip_objects.is_empty());
}

#[test]
fn clip_volume_uniform_from_plane() {
    let shape = ClipShape::Plane {
        normal: [0.0, 1.0, 0.0],
        distance: 3.0,
        cap_color: None,
    };
    let u = ClipVolumeUniform::from_clip_shape(&shape);
    assert_eq!(u.volume_type, 1);
    assert_eq!(u.plane_normal, [0.0, 1.0, 0.0]);
    assert!((u.plane_dist - 3.0).abs() < 1e-6);
}

#[test]
fn clip_volume_uniform_from_box() {
    let shape = ClipShape::Box {
        center: [1.0, 2.0, 3.0],
        half_extents: [0.5, 1.0, 1.5],
        orientation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let u = ClipVolumeUniform::from_clip_shape(&shape);
    assert_eq!(u.volume_type, 2);
    assert_eq!(u.box_center, [1.0, 2.0, 3.0]);
    assert_eq!(u.box_half_extents, [0.5, 1.0, 1.5]);
    assert_eq!(u.box_col0, [1.0, 0.0, 0.0]);
}

#[test]
fn clip_volume_uniform_from_sphere() {
    let shape = ClipShape::Sphere {
        center: [5.0, 0.0, -2.0],
        radius: 3.14,
    };
    let u = ClipVolumeUniform::from_clip_shape(&shape);
    assert_eq!(u.volume_type, 3);
    assert_eq!(u.sphere_center, [5.0, 0.0, -2.0]);
    assert!((u.sphere_radius - 3.14).abs() < 1e-5);
}
