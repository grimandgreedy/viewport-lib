/// A snapshot of camera position state (center, distance, orientation).
///
/// Returned by framing helpers such as [`Camera::fit_sphere_target`] and
/// [`Camera::fit_aabb_target`]. Useful for presets and animation targets: pass
/// `target.center` / `target.distance` / `target.orientation` to
/// [`CameraAnimator::fly_to`](crate::camera::animator::CameraAnimator::fly_to).
///
/// The getters/setters on [`Camera`] (`center()`, `distance()`,
/// `orientation()`) are forward-compatible accessors intended for eventual
/// field privatization (Phase 4). Field access (`camera.center` etc.) still
/// works and is not deprecated.
#[derive(Clone, Copy, Debug)]
pub struct CameraTarget {
    /// Orbit target point in world space.
    pub center: glam::Vec3,
    /// Distance from center to eye.
    pub distance: f32,
    /// Camera orientation as a unit quaternion.
    pub orientation: glam::Quat,
}

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
            // Default to a slight top-down view above the x-y plane (Z-up world).
            orientation: glam::Quat::from_rotation_z(0.3) * glam::Quat::from_rotation_x(0.8),
            fov_y: std::f32::consts::FRAC_PI_4,
            aspect: 1.5,
            znear: 0.01,
            zfar: 1000.0,
        }
    }
}

impl Camera {
    /// Minimum allowed camera distance (zoom limit).
    pub const MIN_DISTANCE: f32 = 0.01;

    /// Maximum allowed camera distance (zoom limit).
    pub const MAX_DISTANCE: f32 = 1.0e6;

    // -----------------------------------------------------------------------
    // Getters and setters
    // -----------------------------------------------------------------------

    /// Return the orbit target center in world space.
    ///
    /// Forward-compatible accessor — equivalent to reading `self.center`.
    pub fn center(&self) -> glam::Vec3 {
        self.center
    }

    /// Set the orbit target center in world space.
    pub fn set_center(&mut self, center: glam::Vec3) {
        self.center = center;
    }

    /// Return the camera distance (zoom).
    ///
    /// Forward-compatible accessor — equivalent to reading `self.distance`.
    pub fn distance(&self) -> f32 {
        self.distance
    }

    /// Set the camera distance, clamped to `[MIN_DISTANCE, MAX_DISTANCE]`.
    pub fn set_distance(&mut self, d: f32) {
        self.distance = d.clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE);
    }

    /// Return the camera orientation quaternion.
    ///
    /// Forward-compatible accessor — equivalent to reading `self.orientation`.
    pub fn orientation(&self) -> glam::Quat {
        self.orientation
    }

    /// Set the camera orientation, normalizing the quaternion.
    pub fn set_orientation(&mut self, q: glam::Quat) {
        self.orientation = q.normalize();
    }

    /// Set the vertical field of view in radians.
    pub fn set_fov_y(&mut self, fov_y: f32) {
        self.fov_y = fov_y;
    }

    /// Set the aspect ratio from pixel dimensions.
    ///
    /// If `height` is zero or negative, aspect is set to `1.0`.
    pub fn set_aspect_ratio(&mut self, width: f32, height: f32) {
        self.aspect = if height > 0.0 { width / height } else { 1.0 };
    }

    /// Set the near and far clipping plane distances.
    pub fn set_clip_planes(&mut self, znear: f32, zfar: f32) {
        self.znear = znear;
        self.zfar = zfar;
    }

    // -----------------------------------------------------------------------
    // Operation methods
    // -----------------------------------------------------------------------

    /// Orbit the camera by `yaw` and `pitch` radians.
    ///
    /// Applies `Quat::from_rotation_z(-yaw) * orientation * Quat::from_rotation_x(-pitch)`.
    /// The sign convention matches all examples: positive yaw rotates counter-clockwise
    /// when viewed from above (Z-up), positive pitch tilts up.
    pub fn orbit(&mut self, yaw: f32, pitch: f32) {
        self.orientation = (glam::Quat::from_rotation_z(-yaw)
            * self.orientation
            * glam::Quat::from_rotation_x(-pitch))
        .normalize();
    }

    /// Pan the camera by world-space deltas.
    ///
    /// `right_delta` subtracts from center along the camera right vector;
    /// `up_delta` adds to center along the camera up vector. This sign
    /// convention matches mouse pan: dragging right moves the scene left.
    pub fn pan_world(&mut self, right_delta: f32, up_delta: f32) {
        self.center -= self.right() * right_delta;
        self.center += self.up() * up_delta;
    }

    /// Pan the camera by pixel-space deltas.
    ///
    /// Computes `pan_scale = 2 * distance * tan(fov_y/2) / viewport_height`
    /// then delegates to [`pan_world`](Self::pan_world).
    pub fn pan_pixels(&mut self, delta_pixels: glam::Vec2, viewport_height: f32) {
        let pan_scale = 2.0 * self.distance * (self.fov_y / 2.0).tan()
            / viewport_height.max(1.0);
        self.pan_world(delta_pixels.x * pan_scale, delta_pixels.y * pan_scale);
    }

    /// Zoom by multiplying the distance by `factor`, clamped to `[MIN_DISTANCE, MAX_DISTANCE]`.
    pub fn zoom_by_factor(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE);
    }

    /// Zoom by adding `delta` to the distance, clamped to `[MIN_DISTANCE, MAX_DISTANCE]`.
    pub fn zoom_by_delta(&mut self, delta: f32) {
        self.distance = (self.distance + delta).clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE);
    }

    // -----------------------------------------------------------------------
    // Framing helpers
    // -----------------------------------------------------------------------

    /// Frame the camera to contain a bounding sphere, applying the result directly.
    pub fn frame_sphere(&mut self, center: glam::Vec3, radius: f32) {
        let (c, d) = self.fit_sphere(center, radius);
        self.center = c;
        self.set_distance(d);
    }

    /// Frame the camera to contain an AABB, applying the result directly.
    pub fn frame_aabb(&mut self, aabb: &crate::scene::aabb::Aabb) {
        let (c, d) = self.fit_aabb(aabb);
        self.center = c;
        self.set_distance(d);
    }

    /// Compute a [`CameraTarget`] that would frame a bounding sphere,
    /// preserving the current orientation.
    pub fn fit_sphere_target(&self, center: glam::Vec3, radius: f32) -> CameraTarget {
        let (c, d) = self.fit_sphere(center, radius);
        CameraTarget {
            center: c,
            distance: d,
            orientation: self.orientation,
        }
    }

    /// Compute a [`CameraTarget`] that would frame an AABB,
    /// preserving the current orientation.
    pub fn fit_aabb_target(&self, aabb: &crate::scene::aabb::Aabb) -> CameraTarget {
        let (c, d) = self.fit_aabb(aabb);
        CameraTarget {
            center: c,
            distance: d,
            orientation: self.orientation,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Compute the eye offset from the orbit center in world space.
    fn eye_offset(&self) -> glam::Vec3 {
        self.orientation * (glam::Vec3::Z * self.distance)
    }

    /// Eye (camera) position in world space.
    pub fn eye_position(&self) -> glam::Vec3 {
        self.center + self.eye_offset()
    }

    /// Right-handed view matrix (world -> camera space).
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

    // -----------------------------------------------------------------------
    // Tests for new methods (Task 1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert!((Camera::MIN_DISTANCE - 0.01).abs() < 1e-7);
        assert!((Camera::MAX_DISTANCE - 1.0e6).abs() < 1.0);
    }

    #[test]
    fn test_set_distance_clamps() {
        let mut cam = Camera::default();
        cam.set_distance(-1.0);
        assert!(
            (cam.distance - Camera::MIN_DISTANCE).abs() < 1e-7,
            "negative distance should clamp to MIN_DISTANCE"
        );
        cam.set_distance(2.0e6);
        assert!(
            (cam.distance - Camera::MAX_DISTANCE).abs() < 1.0,
            "too-large distance should clamp to MAX_DISTANCE"
        );
    }

    #[test]
    fn test_set_center_and_getter() {
        let mut cam = Camera::default();
        let target = glam::Vec3::new(1.0, 2.0, 3.0);
        cam.set_center(target);
        assert_eq!(cam.center(), target);
        assert_eq!(cam.center, target);
    }

    #[test]
    fn test_set_distance_getter() {
        let mut cam = Camera::default();
        cam.set_distance(7.5);
        assert!((cam.distance() - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_set_orientation_normalizes() {
        let mut cam = Camera::default();
        // Supply a non-unit quaternion.
        let q = glam::Quat::from_xyzw(0.0, 0.707, 0.0, 0.707) * 2.0;
        cam.set_orientation(q);
        let len = (cam.orientation.x * cam.orientation.x
            + cam.orientation.y * cam.orientation.y
            + cam.orientation.z * cam.orientation.z
            + cam.orientation.w * cam.orientation.w)
            .sqrt();
        assert!((len - 1.0).abs() < 1e-5, "orientation should be normalized, len={len}");
    }

    #[test]
    fn test_set_aspect_ratio_normal() {
        let mut cam = Camera::default();
        cam.set_aspect_ratio(800.0, 600.0);
        let expected = 800.0 / 600.0;
        assert!((cam.aspect - expected).abs() < 1e-5);
    }

    #[test]
    fn test_set_aspect_ratio_zero_height() {
        let mut cam = Camera::default();
        cam.set_aspect_ratio(800.0, 0.0);
        assert!((cam.aspect - 1.0).abs() < 1e-5, "zero height should produce aspect=1.0");
    }

    #[test]
    fn test_set_fov_y() {
        let mut cam = Camera::default();
        cam.set_fov_y(1.2);
        assert!((cam.fov_y - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_set_clip_planes() {
        let mut cam = Camera::default();
        cam.set_clip_planes(0.1, 500.0);
        assert!((cam.znear - 0.1).abs() < 1e-6);
        assert!((cam.zfar - 500.0).abs() < 1e-4);
    }

    #[test]
    fn test_orbit_matches_manual() {
        let mut cam = Camera::default();
        let orig_orientation = cam.orientation;
        let yaw = 0.1_f32;
        let pitch = 0.2_f32;
        let expected = (glam::Quat::from_rotation_z(-yaw)
            * orig_orientation
            * glam::Quat::from_rotation_x(-pitch))
        .normalize();
        cam.orbit(yaw, pitch);
        let diff = (cam.orientation - expected).length();
        assert!(diff < 1e-5, "orbit() result mismatch, diff={diff}");
    }

    #[test]
    fn test_pan_world_moves_center() {
        let mut cam = Camera::default();
        cam.orientation = glam::Quat::IDENTITY;
        let right = cam.right();
        let up = cam.up();
        let orig_center = cam.center;
        cam.pan_world(1.0, 0.5);
        let expected = orig_center - right * 1.0 + up * 0.5;
        assert!(
            (cam.center - expected).length() < 1e-5,
            "pan_world center mismatch"
        );
    }

    #[test]
    fn test_pan_pixels_uses_correct_scale() {
        let mut cam = Camera::default();
        cam.orientation = glam::Quat::IDENTITY;
        cam.distance = 10.0;
        let viewport_h = 600.0_f32;
        let pan_scale = 2.0 * cam.distance * (cam.fov_y / 2.0).tan() / viewport_h;
        let dx = 100.0_f32;
        let dy = 50.0_f32;
        let orig_center = cam.center;
        let right = cam.right();
        let up = cam.up();
        cam.pan_pixels(glam::vec2(dx, dy), viewport_h);
        let expected = orig_center - right * dx * pan_scale + up * dy * pan_scale;
        assert!(
            (cam.center - expected).length() < 1e-4,
            "pan_pixels center mismatch"
        );
    }

    #[test]
    fn test_zoom_by_factor_clamps() {
        let mut cam = Camera::default();
        cam.distance = 1.0;
        cam.zoom_by_factor(0.0);
        assert!(
            (cam.distance - Camera::MIN_DISTANCE).abs() < 1e-7,
            "factor=0 should clamp to MIN_DISTANCE"
        );

        cam.distance = 1.0;
        cam.zoom_by_factor(2.0e7);
        assert!(
            (cam.distance - Camera::MAX_DISTANCE).abs() < 1.0,
            "large factor should clamp to MAX_DISTANCE"
        );
    }

    #[test]
    fn test_zoom_by_delta() {
        let mut cam = Camera::default();
        cam.distance = 5.0;
        cam.zoom_by_delta(2.0);
        assert!((cam.distance - 7.0).abs() < 1e-5);
        cam.zoom_by_delta(-100.0);
        assert!(
            cam.distance >= Camera::MIN_DISTANCE,
            "delta clamped to MIN_DISTANCE"
        );
    }

    #[test]
    fn test_frame_sphere_applies_result() {
        let mut cam = Camera::default();
        let sphere_center = glam::Vec3::new(1.0, 2.0, 3.0);
        let radius = 5.0;
        let (expected_c, expected_d) = cam.fit_sphere(sphere_center, radius);
        cam.frame_sphere(sphere_center, radius);
        assert!((cam.center - expected_c).length() < 1e-5);
        assert!((cam.distance - expected_d.clamp(Camera::MIN_DISTANCE, Camera::MAX_DISTANCE)).abs() < 1e-5);
    }

    #[test]
    fn test_frame_aabb_applies_result() {
        let mut cam = Camera::default();
        let aabb = crate::scene::aabb::Aabb {
            min: glam::Vec3::splat(-1.0),
            max: glam::Vec3::splat(1.0),
        };
        let (expected_c, expected_d) = cam.fit_aabb(&aabb);
        cam.frame_aabb(&aabb);
        assert!((cam.center - expected_c).length() < 1e-5);
        assert!((cam.distance - expected_d.clamp(Camera::MIN_DISTANCE, Camera::MAX_DISTANCE)).abs() < 1e-5);
    }

    #[test]
    fn test_fit_sphere_target() {
        let cam = Camera::default();
        let sphere_center = glam::Vec3::new(0.0, 1.0, 0.0);
        let radius = 2.0;
        let target = cam.fit_sphere_target(sphere_center, radius);
        let (expected_c, expected_d) = cam.fit_sphere(sphere_center, radius);
        assert!((target.center - expected_c).length() < 1e-5);
        assert!((target.distance - expected_d).abs() < 1e-5);
        // Orientation should be preserved.
        let diff = (target.orientation - cam.orientation).length();
        assert!(diff < 1e-5, "fit_sphere_target should preserve orientation");
    }

    #[test]
    fn test_fit_aabb_target() {
        let cam = Camera::default();
        let aabb = crate::scene::aabb::Aabb {
            min: glam::Vec3::splat(-2.0),
            max: glam::Vec3::splat(2.0),
        };
        let target = cam.fit_aabb_target(&aabb);
        let (expected_c, expected_d) = cam.fit_aabb(&aabb);
        assert!((target.center - expected_c).length() < 1e-5);
        assert!((target.distance - expected_d).abs() < 1e-5);
        let diff = (target.orientation - cam.orientation).length();
        assert!(diff < 1e-5, "fit_aabb_target should preserve orientation");
    }
}
