//! View frustum extraction and AABB culling.
//!
//! Uses the Gribb-Hartmann method to extract six planes from a view-projection
//! matrix, then tests AABBs against them for visibility culling.

use crate::scene::aabb::Aabb;

/// A plane in 3D space: `dot(normal, point) + d = 0`.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// Unit normal of the plane.
    pub normal: glam::Vec3,
    /// Signed distance from origin along `normal`.
    pub d: f32,
}

/// The six planes of a view frustum: left, right, bottom, top, near, far.
#[derive(Debug, Clone)]
pub struct Frustum {
    /// Inward-facing planes in order: left, right, bottom, top, near, far.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Extract a frustum from a combined view-projection matrix using the
    /// Gribb-Hartmann plane extraction method.
    ///
    /// The resulting planes face inward (normals point toward the interior).
    pub fn from_view_proj(vp: &glam::Mat4) -> Self {
        let row0 = vp.row(0);
        let row1 = vp.row(1);
        let row2 = vp.row(2);
        let row3 = vp.row(3);

        let mut planes = [
            // Left:   row3 + row0
            extract_plane(row3 + row0),
            // Right:  row3 - row0
            extract_plane(row3 - row0),
            // Bottom: row3 + row1
            extract_plane(row3 + row1),
            // Top:    row3 - row1
            extract_plane(row3 - row1),
            // Near:   row2         (wgpu depth 0..1, so near = row2 directly)
            extract_plane(row2),
            // Far:    row3 - row2
            extract_plane(row3 - row2),
        ];

        // Normalize all planes.
        for plane in &mut planes {
            let len = plane.normal.length();
            if len > 1e-8 {
                plane.normal /= len;
                plane.d /= len;
            }
        }

        Self { planes }
    }

    /// Extract a world-space frustum from `vp_rel`, a view-projection for
    /// points relative to `eye` (its view carries no translation).
    ///
    /// Far from the world origin an absolute view-projection carries the
    /// eye's distance in every row, and the far plane is the difference of
    /// two nearly equal rows, so [`from_view_proj`](Self::from_view_proj)
    /// loses about `far / near` f32 steps of that distance: kilometres at
    /// 6 400 km. Here the planes are extracted where the numbers are small
    /// and each plane's distance is moved back to world space in f64, then
    /// rounded to f32 once. A world-space frustum built this way is as
    /// precise as an f32 world position at the eye's distance, half a metre
    /// at 6 400 km; test positions relative to the eye for anything finer.
    pub fn from_eye_relative(vp_rel: &glam::Mat4, eye: glam::DVec3) -> Self {
        let mut frustum = Self::from_view_proj(vp_rel);
        if eye != glam::DVec3::ZERO {
            // dot(n, p - eye) + d = 0  ->  dot(n, p) + (d - dot(n, eye)) = 0
            for plane in &mut frustum.planes {
                plane.d = (f64::from(plane.d) - plane.normal.as_dvec3().dot(eye)) as f32;
            }
        }
        frustum
    }

    /// Test whether a sphere should be culled.
    ///
    /// Returns `true` if the sphere is fully outside at least one plane. The
    /// test is conservative: spheres straddling a plane are kept. The CPU
    /// light cull uses this to drop point and spot lights whose entire range
    /// is off-screen before they reach the GPU cluster build.
    pub fn cull_sphere(&self, centre: glam::Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.normal.dot(centre) + plane.d < -radius {
                return true;
            }
        }
        false
    }

    /// Test whether a (right-circular) cone should be culled.
    ///
    /// The cone has its apex at `apex`, points along the unit vector `axis`,
    /// has an outer half-angle of `half_angle` radians, and reaches `range`
    /// units along the axis. The test is the standard "cone vs plane" check:
    /// project the apex onto the plane normal and add the cone's spread at the
    /// far end.
    ///
    /// Conservative: cones partially inside the frustum are kept.
    pub fn cull_cone(
        &self,
        apex: glam::Vec3,
        axis: glam::Vec3,
        half_angle: f32,
        range: f32,
    ) -> bool {
        let sin_a = half_angle.sin();
        let cos_a = half_angle.cos();
        // Far cap radius and centre.
        let far_centre = apex + axis * range * cos_a;
        let far_radius = range * sin_a;
        // The cone is the hull of its apex and far cap, so it is outside a
        // plane only when the apex and the cap's nearest point both are.
        for plane in &self.planes {
            let apex_dist = plane.normal.dot(apex) + plane.d;
            let far_dist = plane.normal.dot(far_centre) + plane.d + far_radius;
            if apex_dist < 0.0 && far_dist < 0.0 {
                return true;
            }
        }
        false
    }

    /// Test whether an AABB should be culled (is fully outside the frustum).
    ///
    /// Returns `true` if the AABB is entirely outside at least one plane
    /// (meaning it should be culled / not drawn).
    pub fn cull_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            // Find the "positive vertex" : the corner of the AABB most in the
            // direction of the plane normal. If even this vertex is behind the
            // plane, the entire AABB is outside.
            let p = glam::Vec3::new(
                if plane.normal.x >= 0.0 {
                    aabb.max.x
                } else {
                    aabb.min.x
                },
                if plane.normal.y >= 0.0 {
                    aabb.max.y
                } else {
                    aabb.min.y
                },
                if plane.normal.z >= 0.0 {
                    aabb.max.z
                } else {
                    aabb.min.z
                },
            );
            if plane.normal.dot(p) + plane.d < 0.0 {
                return true; // Fully outside this plane -> cull.
            }
        }
        false // Inside or intersecting all planes -> visible.
    }
}

fn extract_plane(row: glam::Vec4) -> Plane {
    Plane {
        normal: glam::Vec3::new(row.x, row.y, row.z),
        d: row.w,
    }
}

/// Statistics from a culling pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct CullStats {
    /// Total objects tested for culling.
    pub total: u32,
    /// Objects that passed the frustum test (will be rendered).
    pub visible: u32,
    /// Objects rejected by the frustum test (not rendered).
    pub culled: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Planes extracted relative to an eye 6 400 km out keep a point 5 m in
    /// front of it inside; the absolute matrix puts the far plane behind it.
    #[test]
    fn eye_relative_planes_hold_far_from_origin() {
        let eye = glam::DVec3::new(0.0, 6.4e6, 3.0);
        let target = eye + glam::DVec3::new(0.0, 4.0, -3.0);
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, 1.0, 0.05, 1000.0);
        let rel_view = glam::Mat4::look_to_rh(
            glam::Vec3::ZERO,
            (target - eye).as_vec3().normalize(),
            glam::Vec3::Z,
        );
        let frustum = Frustum::from_eye_relative(&(proj * rel_view), eye);
        let point = |p: glam::DVec3| Aabb::from_positions(&[p.as_vec3().to_array()]);
        assert!(!frustum.cull_aabb(&point(target)));
        let forward = (target - eye).normalize();
        assert!(frustum.cull_aabb(&point(eye - forward * 20.0)));
        assert!(frustum.cull_aabb(&point(eye + forward * 1020.0)));
        // The same frustum from the absolute matrix, for contrast.
        let abs_view = glam::Mat4::look_at_rh(eye.as_vec3(), target.as_vec3(), glam::Vec3::Z);
        let absolute = Frustum::from_view_proj(&(proj * abs_view));
        assert!(absolute.cull_aabb(&point(target)));
    }

    fn test_camera_vp() -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        proj * view
    }

    #[test]
    fn test_frustum_from_perspective() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // All plane normals should be roughly unit length after normalization.
        for plane in &frustum.planes {
            let len = plane.normal.length();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "plane normal not unit length: {len}"
            );
        }
    }

    #[test]
    fn test_cull_aabb_inside() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Box at origin : directly in front of camera at z=5 looking at origin.
        let aabb = Aabb {
            min: glam::Vec3::splat(-0.5),
            max: glam::Vec3::splat(0.5),
        };
        assert!(!frustum.cull_aabb(&aabb), "box at origin should be visible");
    }

    #[test]
    fn test_cull_aabb_behind_camera() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Camera at z=5 looking toward -Z. Box at z=100 is behind the camera.
        let aabb = Aabb {
            min: glam::Vec3::new(-0.5, -0.5, 99.5),
            max: glam::Vec3::new(0.5, 0.5, 100.5),
        };
        assert!(
            frustum.cull_aabb(&aabb),
            "box behind camera should be culled"
        );
    }

    #[test]
    fn test_cull_aabb_far_left() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Box far to the left : should be outside the left frustum plane.
        let aabb = Aabb {
            min: glam::Vec3::new(-1000.0, -0.5, -0.5),
            max: glam::Vec3::new(-999.0, 0.5, 0.5),
        };
        assert!(frustum.cull_aabb(&aabb), "box far left should be culled");
    }

    #[test]
    fn test_cull_sphere_in_view() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        assert!(!frustum.cull_sphere(glam::Vec3::ZERO, 0.5));
    }

    #[test]
    fn test_cull_sphere_behind_camera() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Camera at z=5 looking toward origin (-Z view). Sphere at z=100 is
        // far behind the camera.
        assert!(frustum.cull_sphere(glam::Vec3::new(0.0, 0.0, 100.0), 0.5));
    }

    #[test]
    fn test_cull_sphere_large_radius_keeps() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Sphere centre is behind the camera but radius reaches into view.
        assert!(!frustum.cull_sphere(glam::Vec3::new(0.0, 0.0, 100.0), 200.0));
    }

    #[test]
    fn test_cull_cone_pointing_into_view() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Apex inside the frustum at the origin, axis pointing further into
        // the scene. Should be kept.
        let axis = -glam::Vec3::Z;
        assert!(!frustum.cull_cone(glam::Vec3::ZERO, axis, 0.4, 5.0));
    }

    #[test]
    fn test_cull_cone_off_to_the_side() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Cone planted far to the right, axis pointing further right.
        let axis = glam::Vec3::X;
        assert!(frustum.cull_cone(glam::Vec3::new(1000.0, 0.0, 0.0), axis, 0.1, 1.0));
    }

    #[test]
    fn test_cull_cone_apex_outside_but_cap_covers_view() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Apex well outside the left plane, axis pointing across the view;
        // the wide far cap reaches past the origin, so the cone is visible.
        let axis = glam::Vec3::X;
        assert!(!frustum.cull_cone(glam::Vec3::new(-10.0, 0.0, 0.0), axis, 0.5, 20.0));
    }

    #[test]
    fn test_cull_aabb_straddling_near_plane() {
        let frustum = Frustum::from_view_proj(&test_camera_vp());
        // Large box that straddles the frustum : should NOT be culled.
        let aabb = Aabb {
            min: glam::Vec3::splat(-2.0),
            max: glam::Vec3::splat(2.0),
        };
        assert!(!frustum.cull_aabb(&aabb), "large box should be visible");
    }
}
