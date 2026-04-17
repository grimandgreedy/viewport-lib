/// Projection mode for the viewport camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Projection {
    /// Perspective projection — objects farther away appear smaller.
    Perspective,
    /// Orthographic projection — no foreshortening, parallel lines stay parallel.
    Orthographic,
}

/// Arcball camera for the 3D viewport.
///
/// The camera orbits around a center point. `orientation` is a quaternion that
/// maps camera-space axes to world-space: eye = center + orientation * (Z * distance).
/// Pan translates the center in camera-space. Zoom adjusts the distance.
///
/// Using a quaternion avoids gimbal lock and allows full 360° orbit in any direction.
/// All matrices are computed right-handed (wgpu NDC convention).
#[derive(Clone)]
pub struct Camera {
    /// Projection mode (perspective or orthographic).
    pub projection: Projection,
    /// Orbit target point in world space.
    pub center: glam::Vec3,
    /// Distance from center to eye (zoom).
    pub distance: f32,
    /// Camera orientation as a unit quaternion.
    /// eye = center + orientation * (Vec3::Z * distance).
    /// Identity = looking along -Z from +Z position (standard front view).
    pub orientation: glam::Quat,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Viewport width / height ratio — updated each frame from viewport rect.
    pub aspect: f32,
    /// Near clipping plane distance.
    pub znear: f32,
    /// Far clipping plane distance.
    pub zfar: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            projection: Projection::Perspective,
            center: glam::Vec3::ZERO,
            distance: 5.0,
            // Match old default yaw=0.3, pitch=0.3 via quaternion composition.
            orientation: glam::Quat::from_rotation_y(0.3) * glam::Quat::from_rotation_x(0.3),
            fov_y: std::f32::consts::FRAC_PI_4,
            aspect: 1.5,
            znear: 0.01,
            zfar: 1000.0,
        }
    }
}

impl Camera {
    /// Compute the eye offset from the orbit center in world space.
    fn eye_offset(&self) -> glam::Vec3 {
        self.orientation * (glam::Vec3::Z * self.distance)
    }

    /// Eye (camera) position in world space.
    pub fn eye_position(&self) -> glam::Vec3 {
        self.center + self.eye_offset()
    }

    /// Right-handed view matrix (world → camera space).
    pub fn view_matrix(&self) -> glam::Mat4 {
        let eye = self.eye_position();
        let up = self.orientation * glam::Vec3::Y;
        glam::Mat4::look_at_rh(eye, self.center, up)
    }

    /// Right-handed projection matrix (wgpu depth 0..1).
    ///
    /// In orthographic mode the viewing volume is derived from `distance` and
    /// `fov_y` so that switching between perspective and orthographic at the
    /// same distance produces a similar framing.
    pub fn proj_matrix(&self) -> glam::Mat4 {
        // Ensure the far plane always extends past the orbit center regardless
        // of how far the user has zoomed out. zfar is a minimum; if the camera
        // distance exceeds it the stored value is stale.
        let effective_zfar = self.zfar.max(self.distance * 3.0);
        match self.projection {
            Projection::Perspective => {
                glam::Mat4::perspective_rh(self.fov_y, self.aspect, self.znear, effective_zfar)
            }
            Projection::Orthographic => {
                let half_h = self.distance * (self.fov_y / 2.0).tan();
                let half_w = half_h * self.aspect;
                glam::Mat4::orthographic_rh(
                    -half_w,
                    half_w,
                    -half_h,
                    half_h,
                    self.znear,
                    effective_zfar,
                )
            }
        }
    }

    /// Combined view-projection matrix for use in the camera uniform buffer.
    /// Note: projection * view (column-major order: proj applied after view).
    pub fn view_proj_matrix(&self) -> glam::Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Camera right vector in world space (used for pan).
    pub fn right(&self) -> glam::Vec3 {
        self.orientation * glam::Vec3::X
    }

    /// Camera up vector in world space (used for pan).
    pub fn up(&self) -> glam::Vec3 {
        self.orientation * glam::Vec3::Y
    }

    /// Extract the view frustum from the current view-projection matrix.
    pub fn frustum(&self) -> crate::camera::frustum::Frustum {
        crate::camera::frustum::Frustum::from_view_proj(&self.view_proj_matrix())
    }

    /// Compute `(center, distance)` that would frame a bounding sphere in view,
    /// preserving the current orientation.
    ///
    /// A 1.2× padding factor is applied so the object doesn't touch the edges.
    pub fn fit_sphere(&self, center: glam::Vec3, radius: f32) -> (glam::Vec3, f32) {
        let distance = match self.projection {
            Projection::Perspective => radius / (self.fov_y / 2.0).tan() * 1.2,
            Projection::Orthographic => radius * 1.2,
        };
        (center, distance)
    }

    /// Compute `(center, distance)` that would frame an AABB in view,
    /// preserving the current orientation.
    ///
    /// Converts the AABB to a bounding sphere and delegates to [`fit_sphere`](Self::fit_sphere).
    pub fn fit_aabb(&self, aabb: &crate::scene::aabb::Aabb) -> (glam::Vec3, f32) {
        let center = aabb.center();
        let radius = aabb.half_extents().length();
        self.fit_sphere(center, radius)
    }

    /// Center the camera on a domain of the given extents, adjusting distance
    /// and clipping planes so the entire domain is visible.
    pub fn center_on_domain(&mut self, nx: f32, ny: f32, nz: f32) {
        self.center = glam::Vec3::new(nx / 2.0, ny / 2.0, nz / 2.0);
        let diagonal = (nx * nx + ny * ny + nz * nz).sqrt();
        self.distance = (diagonal / 2.0) / (self.fov_y / 2.0).tan() * 1.2;
        self.znear = (diagonal * 0.0001).max(0.01);
        self.zfar = (diagonal * 10.0).max(1000.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_eye_position() {
        let cam = Camera::default();
        let expected = cam.center + cam.orientation * (glam::Vec3::Z * cam.distance);
        let eye = cam.eye_position();
        assert!(
            (eye - expected).length() < 1e-5,
            "eye={eye:?} expected={expected:?}"
        );
    }

    #[test]
    fn test_view_matrix_looks_at_center() {
        let cam = Camera::default();
        let view = cam.view_matrix();
        // Center in view space should be near the negative Z axis.
        let center_view = view.transform_point3(cam.center);
        // X and Y should be near zero; Z should be negative (in front of camera).
        assert!(
            center_view.x.abs() < 1e-4,
            "center_view.x={}",
            center_view.x
        );
        assert!(
            center_view.y.abs() < 1e-4,
            "center_view.y={}",
            center_view.y
        );
        assert!(
            center_view.z < 0.0,
            "center should be in front of camera, z={}",
            center_view.z
        );
    }

    #[test]
    fn test_view_proj_roundtrip() {
        let cam = Camera::default();
        let vp = cam.view_proj_matrix();
        let vp_inv = vp.inverse();
        // NDC center (0,0,0.5) should unproject to somewhere near the center.
        let world_pt = vp_inv.project_point3(glam::Vec3::new(0.0, 0.0, 0.5));
        // The unprojected point should lie along the camera-to-center ray.
        let eye = cam.eye_position();
        let to_center = (cam.center - eye).normalize();
        let to_pt = (world_pt - eye).normalize();
        let dot = to_center.dot(to_pt);
        assert!(
            dot > 0.99,
            "dot={dot}, point should be along camera-to-center ray"
        );
    }

    #[test]
    fn test_center_on_domain() {
        let mut cam = Camera::default();
        cam.center_on_domain(10.0, 10.0, 10.0);
        assert!((cam.center - glam::Vec3::splat(5.0)).length() < 1e-5);
        assert!(cam.distance > 0.0);
    }

    #[test]
    fn test_fit_sphere_perspective() {
        let cam = Camera::default(); // perspective, fov_y = PI/4
        let (center, dist) = cam.fit_sphere(glam::Vec3::ZERO, 5.0);
        assert!((center - glam::Vec3::ZERO).length() < 1e-5);
        let expected = 5.0 / (cam.fov_y / 2.0).tan() * 1.2;
        assert!(
            (dist - expected).abs() < 1e-4,
            "dist={dist}, expected={expected}"
        );
    }

    #[test]
    fn test_fit_sphere_orthographic() {
        let mut cam = Camera::default();
        cam.projection = Projection::Orthographic;
        let (_, dist) = cam.fit_sphere(glam::Vec3::ZERO, 5.0);
        let expected = 5.0 * 1.2;
        assert!(
            (dist - expected).abs() < 1e-4,
            "dist={dist}, expected={expected}"
        );
    }

    #[test]
    fn test_fit_aabb_unit_cube() {
        let cam = Camera::default();
        let aabb = crate::scene::aabb::Aabb {
            min: glam::Vec3::splat(-0.5),
            max: glam::Vec3::splat(0.5),
        };
        let (center, dist) = cam.fit_aabb(&aabb);
        assert!(center.length() < 1e-5, "center should be origin");
        assert!(dist > 0.0, "distance should be positive");
        // Half-diagonal of unit cube = sqrt(0.75) ≈ 0.866
        let radius = aabb.half_extents().length();
        let expected = radius / (cam.fov_y / 2.0).tan() * 1.2;
        assert!(
            (dist - expected).abs() < 1e-4,
            "dist={dist}, expected={expected}"
        );
    }

    #[test]
    fn test_fit_aabb_preserves_padding() {
        let cam = Camera::default();
        let aabb = crate::scene::aabb::Aabb {
            min: glam::Vec3::splat(-2.0),
            max: glam::Vec3::splat(2.0),
        };
        let (_, dist) = cam.fit_aabb(&aabb);
        // Without padding: radius / tan(fov/2)
        let radius = aabb.half_extents().length();
        let no_pad = radius / (cam.fov_y / 2.0).tan();
        assert!(
            dist > no_pad,
            "padded distance ({dist}) should exceed unpadded ({no_pad})"
        );
    }

    #[test]
    fn test_right_up_orthogonal() {
        let cam = Camera::default();
        let dot = cam.right().dot(cam.up());
        assert!(
            dot.abs() < 1e-5,
            "right and up should be orthogonal, dot={dot}"
        );
    }
}
