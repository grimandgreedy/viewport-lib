use super::*;

fn ortho() -> Camera {
    let mut cam = Camera::default();
    cam.set_orthographic(true);
    cam
}

#[test]
fn orbit_eye_lies_along_the_local_z_axis() {
    let cases = [
        (DVec3::ZERO, 5.0, Quat::IDENTITY),
        (
            DVec3::new(1.5, -2.0, 0.25),
            12.0,
            Quat::from_rotation_z(0.6) * Quat::from_rotation_x(1.1),
        ),
        (
            DVec3::new(1_048_576.0, 3.0, -7.5),
            0.75,
            Quat::from_rotation_z(-2.3) * Quat::from_rotation_x(0.4),
        ),
    ];
    for (c, d, q) in cases {
        let cam = Camera::orbit(c, d, q);
        assert_eq!(cam.eye(), c + (q * (Vec3::Z * d)).as_dvec3());
        assert_eq!(cam.orientation(), q);
        assert_eq!(cam.target(), c, "the focus point is the orbit centre");
        assert_eq!(cam.orbit_target(), CameraTarget::new(c, d, q));
    }
}

#[test]
fn orbit_view_matrix_is_built_from_the_centre() {
    let c = DVec3::new(0.0, 0.0, 0.5);
    let (d, q) = (8.0, Quat::from_rotation_z(0.7) * Quat::from_rotation_x(1.0));
    let centre = c.as_vec3();
    let eye = centre + q * (Vec3::Z * d);
    let expected = Mat4::look_at_rh(eye, centre, q * Vec3::Y);
    assert_eq!(Camera::orbit(c, d, q).view_matrix(), expected);
}

#[test]
fn look_at_faces_the_target() {
    let eye = DVec3::new(3.0, -4.0, 2.0);
    let target = DVec3::new(0.0, 1.0, 0.5);
    let cam = Camera::look_at(eye, target, Vec3::Z);
    let dir = (target - eye).normalize().as_vec3();
    assert!((cam.forward() - dir).length() < 1e-5);
    assert!(cam.right().z.abs() < 1e-5, "no roll with a Z up vector");
    assert!((cam.target() - target).length() < 1e-5);
}

#[test]
fn test_view_matrix_looks_at_centre() {
    let cam = Camera::default();
    let centre_view = cam.view_matrix().transform_point3(cam.target().as_vec3());
    assert!(centre_view.x.abs() < 1e-4, "{centre_view:?}");
    assert!(centre_view.y.abs() < 1e-4, "{centre_view:?}");
    assert!(centre_view.z < 0.0, "{centre_view:?}");
}

#[test]
fn projection_is_reversed_and_culling_is_standard() {
    for cam in [Camera::default(), ortho()] {
        let (n, f) = cam.depth_range();
        let near = Vec3::new(0.0, 0.0, -n);
        let far = Vec3::new(0.0, 0.0, -f);
        let p = cam.proj_matrix();
        assert!((p.project_point3(near).z - 1.0).abs() < 1e-5);
        assert!(p.project_point3(far).z < 1e-3);
        let c = cam.culling_proj_matrix();
        assert!(c.project_point3(near).z.abs() < 1e-5);
        assert!((c.project_point3(far).z - 1.0).abs() < 1e-4);
    }
}

#[test]
fn orthographic_frames_the_same_height_as_perspective() {
    let persp = Camera::default();
    let ortho = ortho();
    let d = persp.focus_distance();
    assert!((ortho.view_height_at(1.0) - persp.view_height_at(d)).abs() < 1e-5);
    assert!((ortho.fov_y() - persp.fov_y()).abs() < 1e-5);
}

#[test]
fn screen_ray_through_the_centre_is_the_eye_and_forward() {
    let cam = Camera::default();
    let (origin, dir) = cam.screen_ray(Vec2::ZERO);
    assert!((origin - cam.eye().as_vec3()).length() < 1e-5);
    assert!((dir - cam.forward()).length() < 1e-5);
}

#[test]
fn screen_ray_lands_on_its_screen_point() {
    for cam in [Camera::default(), ortho()] {
        let ndc = Vec2::new(0.6, -0.3);
        let (origin, dir) = cam.screen_ray(ndc);
        let p = cam
            .culling_view_proj_matrix()
            .project_point3(origin + dir * 3.0);
        assert!((p.truncate() - ndc).length() < 1e-4, "{p:?} for {ndc:?}");
    }
}

#[test]
fn orthographic_screen_ray_starts_on_the_near_plane() {
    let cam = ortho();
    let (origin, dir) = cam.screen_ray(Vec2::new(-0.5, 0.25));
    assert!((dir - cam.forward()).length() < 1e-6);
    let depth = (origin - cam.eye().as_vec3()).dot(cam.forward());
    assert!((depth - cam.depth_range().0).abs() < 1e-4, "{depth}");
}

#[test]
fn test_view_proj_roundtrip() {
    let cam = Camera::default();
    let world_pt = cam
        .view_proj_matrix()
        .inverse()
        .project_point3(Vec3::new(0.0, 0.0, 0.5));
    let eye = cam.eye();
    let to_centre = (cam.target() - eye).as_vec3().normalize();
    let to_pt = (world_pt - eye.as_vec3()).normalize();
    assert!(to_centre.dot(to_pt) > 0.99);
}

#[test]
fn test_centre_on_domain() {
    let mut cam = Camera::default();
    cam.centre_on_domain(10.0, 10.0, 10.0);
    assert!((cam.target() - DVec3::splat(5.0)).length() < 1e-5);
    assert!(cam.focus_distance() > 0.0);
}

#[test]
fn test_fit_sphere_perspective() {
    let cam = Camera::default();
    let (centre, dist) = cam.fit_sphere(Vec3::ZERO, 5.0);
    assert!(centre.length() < 1e-5);
    let expected = 5.0 / (cam.fov_y() / 2.0).tan() * 1.2;
    assert!((dist - expected).abs() < 1e-4, "dist={dist}");
}

#[test]
fn test_fit_sphere_orthographic() {
    let (_, dist) = ortho().fit_sphere(Vec3::ZERO, 5.0);
    assert!((dist - 6.0).abs() < 1e-4, "dist={dist}");
}

#[test]
fn test_fit_aabb_unit_cube() {
    let cam = Camera::default();
    let aabb = crate::scene::aabb::Aabb {
        min: Vec3::splat(-0.5),
        max: Vec3::splat(0.5),
    };
    let (centre, dist) = cam.fit_aabb(&aabb);
    assert!(centre.length() < 1e-5);
    let radius = aabb.half_extents().length();
    let expected = radius / (cam.fov_y() / 2.0).tan() * 1.2;
    assert!((dist - expected).abs() < 1e-4);
}

#[test]
fn test_right_up_orthogonal() {
    let cam = Camera::default();
    assert!(cam.right().dot(cam.up()).abs() < 1e-5);
}

#[test]
fn set_orbit_scales_an_orthographic_height() {
    let mut cam = ortho();
    let h = cam.view_height_at(1.0);
    let t = cam.orbit_target();
    cam.set_orbit(t.centre, t.distance * 2.0, t.orientation);
    assert!((cam.view_height_at(1.0) - 2.0 * h).abs() < 1e-4);
    assert_eq!(cam.focus_distance(), t.distance * 2.0);
}

#[test]
fn test_set_aspect_ratio() {
    let mut cam = Camera::default();
    cam.set_aspect_ratio(800.0, 600.0);
    assert!((cam.aspect - 800.0 / 600.0).abs() < 1e-5);
    cam.set_aspect_ratio(800.0, 0.0);
    assert_eq!(cam.aspect, 1.0);
}

#[test]
fn test_set_fov_y() {
    let mut cam = Camera::default();
    cam.set_fov_y(1.2);
    assert!((cam.fov_y() - 1.2).abs() < 1e-6);
}

#[test]
fn test_set_clip_planes() {
    let mut cam = Camera::default();
    cam.set_clip_planes(0.1, 500.0);
    assert_eq!(cam.depth_range(), (0.1, 500.0));
}

#[test]
fn target_orbit_matches_manual() {
    let mut t = Camera::default().orbit_target();
    let q = t.orientation;
    t.orbit(0.1, 0.2);
    let expected = (Quat::from_rotation_z(-0.1) * q * Quat::from_rotation_x(-0.2)).normalize();
    assert!((t.orientation - expected).length() < 1e-5);
}

#[test]
fn target_pan_world_moves_centre() {
    let mut t = CameraTarget::new(DVec3::ZERO, 5.0, Quat::IDENTITY);
    t.pan_world(1.0, 0.5);
    assert!((t.centre - DVec3::new(-1.0, 0.5, 0.0)).length() < 1e-6);
}

#[test]
fn target_zoom_clamps() {
    let mut t = CameraTarget::new(DVec3::ZERO, 1.0, Quat::IDENTITY);
    t.zoom_by_factor(0.0);
    assert_eq!(t.distance, Camera::MIN_DISTANCE);
    t.distance = 1.0;
    t.zoom_by_factor(2.0e7);
    assert_eq!(t.distance, Camera::MAX_DISTANCE);
    t.distance = 5.0;
    t.zoom_by_delta(2.0);
    assert!((t.distance - 7.0).abs() < 1e-5);
}

#[test]
fn test_frame_sphere_applies_result() {
    let mut cam = Camera::default();
    let (expected_c, expected_d) = cam.fit_sphere(Vec3::new(1.0, 2.0, 3.0), 5.0);
    cam.frame_sphere(Vec3::new(1.0, 2.0, 3.0), 5.0);
    assert!((cam.target() - expected_c).length() < 1e-5);
    assert!((cam.focus_distance() - expected_d).abs() < 1e-5);
}

#[test]
fn test_fit_aabb_target() {
    let cam = Camera::default();
    let aabb = crate::scene::aabb::Aabb {
        min: Vec3::splat(-2.0),
        max: Vec3::splat(2.0),
    };
    let target = cam.fit_aabb_target(&aabb);
    let (expected_c, expected_d) = cam.fit_aabb(&aabb);
    assert!((target.centre - expected_c).length() < 1e-5);
    assert!((target.distance - expected_d).abs() < 1e-5);
    assert_eq!(target.orientation, cam.orientation());
}

/// The frustum of a camera 6 400 km out keeps its target inside and drops
/// points behind it, past the far plane and off to the side, as it does at
/// the world origin.
#[test]
fn frustum_far_from_origin_matches_the_world_origin() {
    for base in [DVec3::ZERO, DVec3::new(6.4e6, 6.4e6, 0.0)] {
        let cam = Camera::orbit(
            base,
            5.0,
            Quat::from_rotation_z(0.3) * Quat::from_rotation_x(0.8),
        );
        let frustum = cam.frustum();
        let at = |p: DVec3| crate::scene::aabb::Aabb::from_positions(&[p.as_vec3().to_array()]);
        let (_, far) = cam.depth_range();
        let eye = cam.eye();
        let forward = cam.forward().as_dvec3();
        assert!(!frustum.cull_aabb(&at(base)), "the target at {base}");
        let behind = eye - forward * 20.0;
        let beyond = eye + forward * (f64::from(far) + 20.0);
        let aside = base + cam.right().as_dvec3() * 50.0;
        for (name, p) in [("behind", behind), ("beyond", beyond), ("aside", aside)] {
            assert!(frustum.cull_aabb(&at(p)), "{name} at {base}");
        }
    }
}
