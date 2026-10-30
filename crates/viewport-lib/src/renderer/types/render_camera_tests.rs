//! The derived `RenderCamera` accessors against the fields `from_camera` stores.

use super::RenderCamera;
use crate::camera::{Camera, Projection};

const EPS: f32 = 1e-5;

fn close(a: f32, b: f32) -> bool {
    // Relative above 1, absolute below, so far planes in the hundreds compare
    // at the same precision as unit vectors.
    (a - b).abs() <= EPS * b.abs().max(1.0)
}

fn close3(a: glam::Vec3, b: [f32; 3]) -> bool {
    (0..3).all(|i| close(a[i], b[i]))
}

fn known_camera(projection: Projection) -> Camera {
    let mut cam = Camera::default();
    cam.projection = projection;
    cam.center = glam::Vec3::new(1.5, -2.0, 0.75);
    cam.distance = 12.0;
    cam.orientation = glam::Quat::from_rotation_z(-0.4) * glam::Quat::from_rotation_x(0.9);
    cam.fov_y = 0.9;
    cam.aspect = 16.0 / 9.0;
    cam.znear = Some(0.25);
    cam.zfar = 400.0;
    cam
}

fn check_pose(rc: &RenderCamera) {
    assert!(close3(rc.eye_position(), rc.eye_position), "eye");
    assert!(close3(rc.forward(), rc.forward), "forward");
    // q and -q are the same rotation.
    assert!(
        close(rc.orientation().dot(rc.orientation).abs(), 1.0),
        "orientation"
    );
    let (near, far) = rc.depth_range();
    assert!(close(near, rc.near), "near {near} vs {}", rc.near);
    // A perspective far plane lives in how far z_axis.z sits from -1, so it
    // carries only about f32 epsilon * far / near of relative precision.
    let far_eps = if rc.is_orthographic() {
        EPS
    } else {
        EPS.max(4.0 * f32::EPSILON * rc.far / rc.near)
    };
    assert!(
        (far - rc.far).abs() <= far_eps * rc.far,
        "far {far} vs {}",
        rc.far
    );
    assert!(close(rc.aspect(), rc.aspect), "aspect");
    // The basis is orthonormal and right-handed.
    let (r, u, f) = (rc.right(), rc.up(), rc.forward());
    assert!(close(r.cross(u).dot(-f), 1.0), "basis");
}

#[test]
fn perspective_accessors_match_stored_fields() {
    let rc = RenderCamera::from_camera(&known_camera(Projection::Perspective));
    assert!(!rc.is_orthographic());
    check_pose(&rc);
    let fov = rc.vertical_fov().expect("perspective has a fov");
    assert!(close(fov, rc.fov), "fov {fov} vs {}", rc.fov);
    let depth = 7.0;
    let expected = 2.0 * (rc.fov / 2.0).tan() * depth;
    assert!(close(rc.view_height_at(depth), expected), "view height");
}

#[test]
fn orthographic_accessors_match_stored_fields() {
    let cam = known_camera(Projection::Orthographic);
    let rc = RenderCamera::from_camera(&cam);
    assert!(rc.is_orthographic());
    check_pose(&rc);
    assert_eq!(rc.vertical_fov(), None);
    // The orbit camera frames 2 * distance * tan(fov / 2) at every depth.
    let expected = 2.0 * cam.distance * (rc.fov / 2.0).tan();
    for depth in [0.5, 12.0, 300.0] {
        assert!(close(rc.view_height_at(depth), expected), "view height");
    }
}

#[test]
fn accessors_follow_matrices_set_directly() {
    let mut rc = RenderCamera::default();
    let eye = glam::Vec3::new(3.0, 4.0, 5.0);
    rc.view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Z);
    rc.projection = glam::Mat4::perspective_rh(0.6, 2.0, 0.5, 50.0);
    assert!(close3(rc.eye_position(), eye.to_array()));
    assert!(close3(rc.forward(), (-eye).normalize().to_array()));
    let (near, far) = rc.depth_range();
    assert!(close(near, 0.5));
    assert!((far - 50.0).abs() <= 4.0 * f32::EPSILON * 100.0 * 50.0);
    assert!(close(rc.vertical_fov().unwrap(), 0.6));
    assert!(close(rc.aspect(), 2.0));
}

#[test]
fn identity_matrices_report_stored_fields() {
    let rc = RenderCamera::default();
    assert_eq!(rc.eye_position(), glam::Vec3::from(rc.eye_position));
    assert_eq!(rc.forward(), glam::Vec3::from(rc.forward));
    assert_eq!(rc.orientation(), rc.orientation);
    assert_eq!(rc.depth_range(), (rc.near, rc.far));
    assert_eq!(rc.vertical_fov(), Some(rc.fov));
    assert_eq!(rc.aspect(), rc.aspect);
    assert!(!rc.is_orthographic());
}
