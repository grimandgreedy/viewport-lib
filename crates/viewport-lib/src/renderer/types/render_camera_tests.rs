//! `RenderCamera` built from matrices: the derived state against the inputs.

use super::{ForegroundPass, ForegroundProjection, RenderCamera};
use crate::camera::{Camera, Projection};
use crate::resources::CameraUniform;

const EPS: f32 = 1e-5;

fn close(a: f32, b: f32) -> bool {
    // Relative above 1, absolute below, so far planes in the hundreds compare
    // at the same precision as unit vectors.
    (a - b).abs() <= EPS * b.abs().max(1.0)
}

fn close3(a: glam::Vec3, b: glam::Vec3) -> bool {
    (0..3).all(|i| close(a[i], b[i]))
}

/// A perspective far plane lives in how far `z_axis.z` sits from -1, so it
/// carries only about `f32::EPSILON * far / near` of relative precision.
fn far_close(far: f32, expected: f32, near: f32, orthographic: bool) -> bool {
    let eps = if orthographic {
        EPS
    } else {
        EPS.max(4.0 * f32::EPSILON * expected / near)
    };
    (far - expected).abs() <= eps * expected
}

fn known_camera(projection: Projection) -> Camera {
    let mut cam = Camera::default();
    cam.projection = projection;
    cam.center = glam::DVec3::new(1.5, -2.0, 0.75);
    cam.distance = 12.0;
    cam.orientation = glam::Quat::from_rotation_z(-0.4) * glam::Quat::from_rotation_x(0.9);
    cam.fov_y = 0.9;
    cam.aspect = 16.0 / 9.0;
    cam.znear = Some(0.25);
    cam.zfar = 400.0;
    cam
}

/// The default camera, an orbit camera, an orthographic one and a wide one.
fn cameras() -> Vec<Camera> {
    let mut wide = known_camera(Projection::Perspective);
    wide.aspect = 3.2;
    wide.fov_y = 1.4;
    vec![
        Camera::default(),
        known_camera(Projection::Perspective),
        known_camera(Projection::Orthographic),
        wide,
    ]
}

/// The uniform `from_camera` packed when the camera stored its eye, forward
/// and depth range as fields beside the matrices. The range is read from the
/// finite culling projection; far packs as 0 for a perspective projection,
/// which has no far plane.
fn stored_field_uniform(cam: &Camera) -> CameraUniform {
    let (view, proj) = (cam.view_matrix(), cam.proj_matrix());
    let vp = proj * view;
    let eye = cam.eye_position();
    let ortho = cam.projection == Projection::Orthographic;
    let culling = cam.culling_proj_matrix();
    let (a, b) = (culling.z_axis.z, culling.w_axis.z);
    let (near, far) = if ortho {
        (b / a, b / a - 1.0 / a)
    } else {
        (b / a, 0.0)
    };
    CameraUniform {
        view_proj: vp.to_cols_array_2d(),
        eye_pos: eye.as_vec3().to_array(),
        lit_clamp: 1.0,
        forward: (cam.center.as_vec3() - eye.as_vec3()).normalize_or_zero().to_array(),
        _pad1: 0.0,
        inv_view_proj: vp.inverse().to_cols_array_2d(),
        view: view.to_cols_array_2d(),
        proj: proj.to_cols_array_2d(),
        inv_proj: proj.inverse().to_cols_array_2d(),
        viewport: [0.0; 4],
        clip: [near, far, f32::from(u8::from(ortho)), 0.0],
        time: [0.0; 4],
        world_origin: [0.0; 3],
        origin_epoch: 0,
    }
}

#[test]
fn from_camera_uniform_is_unchanged() {
    for cam in cameras() {
        let u = RenderCamera::from_camera(&cam).camera_uniform();
        assert_eq!(
            bytemuck::bytes_of(&u),
            bytemuck::bytes_of(&stored_field_uniform(&cam)),
            "uniform bytes"
        );
    }
}

#[test]
fn from_matrices_matches_from_camera() {
    for cam in cameras() {
        let rc = RenderCamera::from_camera(&cam);
        // A perspective projection has no far plane to read back, so a host
        // building from matrices supplies the range.
        let (near, far) = rc.depth_range();
        let rm = RenderCamera::from_matrices(cam.view_matrix(), cam.proj_matrix())
            .with_depth_range(near, far);
        assert_eq!(rc.view(), rm.view());
        assert_eq!(rc.projection(), rm.projection());
        // Eye and forward read back from the view matrix agree with the
        // camera's own to rounding; every other byte is identical.
        let u = rc.camera_uniform();
        let mut m = rm.camera_uniform();
        assert!(close3(m.eye_pos.into(), u.eye_pos.into()), "eye");
        assert!(close3(m.forward.into(), u.forward.into()), "forward");
        m.eye_pos = u.eye_pos;
        m.forward = u.forward;
        assert_eq!(
            bytemuck::bytes_of(&m),
            bytemuck::bytes_of(&u),
            "uniform bytes"
        );
        assert_eq!(rm.depth_range(), rc.depth_range());
        assert_eq!(rm.aspect(), rc.aspect());
        assert_eq!(rm.vertical_fov(), rc.vertical_fov());
    }
}

#[test]
fn from_camera_round_trips_the_camera() {
    for cam in cameras() {
        let rc = RenderCamera::from_camera(&cam);
        let ortho = cam.projection == Projection::Orthographic;
        assert_eq!(rc.is_orthographic(), ortho);
        assert!(close3(rc.eye_position().as_vec3(), cam.eye_position().as_vec3()), "eye");
        assert!(close3(rc.forward(), cam.forward()), "forward");
        assert!(close(rc.aspect(), cam.aspect), "aspect");
        let (near, far) = rc.depth_range();
        assert!(close(near, cam.effective_znear()), "near {near}");
        assert!(
            far_close(far, cam.effective_zfar(), near, ortho),
            "far {far} vs {}",
            cam.effective_zfar()
        );
        match rc.vertical_fov() {
            Some(fov) => assert!(!ortho && close(fov, cam.fov_y), "fov {fov}"),
            None => assert!(ortho),
        }
        // The camera's own angle, exactly, for both projections.
        assert_eq!(rc.fov_y(), cam.fov_y);
    }
}

#[test]
fn basis_is_orthonormal_and_right_handed() {
    for cam in cameras() {
        let rc = RenderCamera::from_camera(&cam);
        let (r, u, f) = (rc.right(), rc.up(), rc.forward());
        assert!(close(r.cross(u).dot(-f), 1.0), "basis");
        // q and -q are the same rotation.
        assert!(close(rc.orientation().dot(cam.orientation).abs(), 1.0));
    }
}

#[test]
fn perspective_view_height_grows_with_depth() {
    let cam = known_camera(Projection::Perspective);
    let rc = RenderCamera::from_camera(&cam);
    let depth = 7.0;
    let expected = 2.0 * (cam.fov_y / 2.0).tan() * depth;
    assert!(close(rc.view_height_at(depth), expected), "view height");
}

#[test]
fn orthographic_view_height_is_constant() {
    let cam = known_camera(Projection::Orthographic);
    let rc = RenderCamera::from_camera(&cam);
    // The orbit camera frames 2 * distance * tan(fov / 2) at every depth.
    let expected = 2.0 * cam.distance * (cam.fov_y / 2.0).tan();
    for depth in [0.5, 12.0, 300.0] {
        assert!(close(rc.view_height_at(depth), expected), "view height");
    }
}

#[test]
fn setters_refresh_the_derived_state() {
    let mut rc = RenderCamera::default();
    let eye = glam::Vec3::new(3.0, 4.0, 5.0);
    rc.set_view(glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Z));
    rc.set_projection(glam::Mat4::perspective_rh(0.6, 2.0, 0.5, 50.0));
    assert!(close3(rc.eye_position().as_vec3(), eye));
    assert!(close3(rc.forward(), (-eye).normalize()));
    let (near, far) = rc.depth_range();
    assert!(close(near, 0.5));
    assert!(far_close(far, 50.0, 0.5, false));
    assert!(close(rc.vertical_fov().unwrap(), 0.6));
    assert!(close(rc.aspect(), 2.0));

    rc.set_projection(glam::Mat4::orthographic_rh(-4.0, 4.0, -2.0, 2.0, 1.0, 30.0));
    assert!(rc.is_orthographic());
    assert_eq!(rc.vertical_fov(), None);
    let (near, far) = rc.depth_range();
    assert!(close(near, 1.0) && close(far, 30.0));
    assert!(close(rc.aspect(), 2.0));
}

#[test]
fn default_is_a_perspective_camera_down_negative_z() {
    let rc = RenderCamera::default();
    assert!(close3(rc.eye_position().as_vec3(), glam::Vec3::new(0.0, 0.0, 5.0)));
    assert!(close3(rc.forward(), glam::Vec3::NEG_Z));
    assert!(close(
        rc.vertical_fov().unwrap(),
        std::f32::consts::FRAC_PI_4
    ));
    assert!(close(rc.aspect(), 1.333));
    let (near, far) = rc.depth_range();
    assert!(close(near, 0.1));
    assert!(far_close(far, 1000.0, 0.1, false));
}

#[test]
fn depth_range_override_wins_until_the_projection_changes() {
    // Reversed-Z with an infinite far plane: nothing to read back.
    let proj = glam::Mat4::perspective_infinite_reverse_rh(0.8, 1.5, 0.05);
    let rc = RenderCamera::from_matrices(glam::Mat4::IDENTITY, proj).with_depth_range(0.05, 500.0);
    assert_eq!(rc.depth_range(), (0.05, 500.0));
    assert_eq!(rc.projection(), proj);
    // The clip lane packs far as 0 for a projection with no far plane.
    let clip = rc.camera_uniform().clip;
    assert_eq!((clip[0], clip[1]), (0.05, 0.0));

    let mut rc = rc;
    rc.set_projection(glam::Mat4::perspective_rh(0.8, 1.5, 0.2, 80.0));
    let (near, far) = rc.depth_range();
    assert!(close(near, 0.2));
    assert!(far_close(far, 80.0, 0.2, false));
}

#[test]
fn foreground_without_override_is_the_scene_camera() {
    let rc = RenderCamera::from_camera(&known_camera(Projection::Perspective));
    for pass in [None, Some(ForegroundPass::default())] {
        let fg = rc.foreground_camera(pass.as_ref());
        assert_eq!(fg.view(), rc.view());
        assert_eq!(fg.projection(), rc.projection());
        assert_eq!(fg.fov_y(), rc.fov_y());
    }
}

#[test]
fn foreground_override_replaces_the_projection() {
    let pass = ForegroundPass {
        projection: Some(ForegroundProjection {
            fov_y: 1.1,
            near: 0.01,
            far: None,
        }),
    };
    for projection in [Projection::Perspective, Projection::Orthographic] {
        let rc = RenderCamera::from_camera(&known_camera(projection));
        let fg = rc.foreground_camera(Some(&pass));
        // Reversed depth: far before near.
        let expected =
            glam::Mat4::perspective_rh(1.1, rc.aspect(), rc.depth_range().1.max(0.02), 0.01);
        assert_eq!(fg.view(), rc.view());
        assert_eq!(fg.projection(), expected);
        // Always perspective, even over an orthographic scene camera.
        assert!(!fg.is_orthographic());
        assert_eq!(fg.fov_y(), 1.1);
        assert!(close(fg.depth_range().0, 0.01));
    }
}

#[test]
fn screen_ray_through_the_centre_is_the_eye_and_forward() {
    for cam in cameras()
        .into_iter()
        .filter(|c| c.projection == Projection::Perspective)
    {
        let rc = RenderCamera::from_camera(&cam);
        let (origin, dir) = rc.screen_ray(glam::Vec2::ZERO);
        assert!(close3(origin, rc.eye_position().as_vec3()), "origin {origin:?}");
        assert!((dir - rc.forward()).length() < 1e-5, "dir {dir:?}");
    }
}

#[test]
fn orthographic_screen_ray_starts_on_the_near_plane() {
    let rc = RenderCamera::from_camera(&known_camera(Projection::Orthographic));
    let (origin, dir) = rc.screen_ray(glam::Vec2::new(0.3, -0.7));
    assert!((dir - rc.forward()).length() < 1e-5, "dir {dir:?}");
    let depth = (origin - rc.eye_position().as_vec3()).dot(rc.forward());
    assert!((depth - rc.depth_range().0).abs() < 1e-4, "depth {depth}");
}

#[test]
fn screen_ray_matches_the_camera_ray() {
    for cam in cameras() {
        let rc = RenderCamera::from_camera(&cam);
        for ndc in [glam::Vec2::new(0.8, 0.6), glam::Vec2::new(-0.4, -0.9)] {
            let (o, d) = rc.screen_ray(ndc);
            let (co, cd) = cam.screen_ray(ndc);
            assert!((o - co).length() < 1e-4, "origin {o:?} vs {co:?}");
            assert!((d - cd).length() < 1e-5, "dir {d:?} vs {cd:?}");
        }
    }
}

#[test]
fn reversed_projections_read_back_their_range() {
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(0.0, -10.0, 2.0),
        glam::Vec3::ZERO,
        glam::Vec3::Z,
    );
    let finite = glam::Mat4::perspective_rh(0.9, 1.5, 400.0, 0.25);
    let rc = RenderCamera::from_matrices(view, finite);
    let (near, far) = rc.depth_range();
    assert!(
        close(near, 0.25) && far_close(far, 400.0, near, false),
        "{near} {far}"
    );

    let ortho = glam::Mat4::orthographic_rh(-4.0, 4.0, -3.0, 3.0, 400.0, 0.25);
    let rc = RenderCamera::from_matrices(view, ortho);
    let (near, far) = rc.depth_range();
    assert!(close(near, 0.25) && close(far, 400.0), "{near} {far}");

    let infinite = glam::Mat4::perspective_infinite_reverse_rh(0.9, 1.5, 0.25);
    let rc = RenderCamera::from_matrices(view, infinite);
    let (near, far) = rc.depth_range();
    assert!(
        close(near, 0.25) && far_close(far, 2500.0, near, false),
        "{near} {far}"
    );
    assert_eq!(
        rc.camera_uniform().clip[1],
        0.0,
        "no far plane packs far as 0"
    );
    let rc = rc.with_depth_range(0.25, 60.0);
    assert_eq!(rc.depth_range(), (0.25, 60.0));
    // The culling projection takes the range: a point past it is outside.
    let frustum = crate::camera::frustum::Frustum::from_view_proj(&rc.culling_view_proj());
    let inside = crate::scene::aabb::Aabb::from_positions(&[[0.0, 40.0, 0.0]]);
    let beyond = crate::scene::aabb::Aabb::from_positions(&[[0.0, 80.0, 0.0]]);
    assert!(!frustum.cull_aabb(&inside));
    assert!(frustum.cull_aabb(&beyond));
}

#[test]
fn projection_reversed_from_a_standard_one() {
    let standard = glam::Mat4::perspective_rh(0.9, 1.5, 0.25, 400.0);
    let rc = RenderCamera::default().with_projection_reversed(standard);
    let expected = glam::Mat4::perspective_rh(0.9, 1.5, 400.0, 0.25);
    assert!(
        rc.projection().abs_diff_eq(expected, 1e-6),
        "{:?}",
        rc.projection()
    );
    let (near, far) = rc.depth_range();
    assert!(
        close(near, 0.25) && far_close(far, 400.0, near, false),
        "{near} {far}"
    );
    // A point at the near plane lands at depth 1, one at the far plane at 0.
    let p = rc.projection();
    assert!((p.project_point3(glam::Vec3::new(0.0, 0.0, -0.25)).z - 1.0).abs() < 1e-5);
    assert!(p.project_point3(glam::Vec3::new(0.0, 0.0, -400.0)).z.abs() < 1e-5);
}
