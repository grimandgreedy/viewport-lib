use glam::{DVec3, Mat4, Quat, Vec2, Vec3};

use super::exposure::PhysicalExposure;
use super::pose::Pose;
use super::projection::Projection;

/// An orbit view: the point a camera turns about, its distance from the eye,
/// and the camera orientation.
///
/// Orbit controllers keep one as their state, [`Camera::orbit_target`] reads
/// one off a camera, and the framing helpers
/// ([`Camera::fit_sphere_target`], [`Camera::fit_aabb_target`]) return one for
/// a preset or a fly-to: pass `target.center` / `target.distance` /
/// `target.orientation` to `CameraAnimator::fly_to`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CameraTarget {
    /// Orbit centre in world space, in f64.
    pub center: DVec3,
    /// Distance from the centre to the eye.
    pub distance: f32,
    /// Camera orientation as a unit quaternion.
    pub orientation: Quat,
}

impl CameraTarget {
    /// An orbit view from its parts.
    pub fn new(center: impl Into<DVec3>, distance: f32, orientation: Quat) -> Self {
        Self {
            center: center.into(),
            distance,
            orientation,
        }
    }

    /// The camera pose this view describes.
    pub fn pose(&self) -> Pose {
        Pose::orbit(self.center, self.distance, self.orientation)
    }

    /// Eye position in world space.
    pub fn eye(&self) -> DVec3 {
        self.pose().position
    }

    /// Turn about the centre by `yaw` and `pitch` radians:
    /// `Quat::from_rotation_z(-yaw) * orientation * Quat::from_rotation_x(-pitch)`.
    /// Positive yaw turns counter-clockwise seen from above (Z-up), positive
    /// pitch tilts up.
    pub fn orbit(&mut self, yaw: f32, pitch: f32) {
        self.orientation =
            (Quat::from_rotation_z(-yaw) * self.orientation * Quat::from_rotation_x(-pitch))
                .normalize();
    }

    /// Move the centre by world-space amounts along the view's right and up
    /// axes. `right_delta` moves it left and `up_delta` up, so a drag to the
    /// right moves the scene with the pointer.
    pub fn pan_world(&mut self, right_delta: f32, up_delta: f32) {
        self.center -= (self.orientation * Vec3::X * right_delta).as_dvec3();
        self.center += (self.orientation * Vec3::Y * up_delta).as_dvec3();
    }

    /// Multiply the distance by `factor`, clamped to
    /// `[Camera::MIN_DISTANCE, Camera::MAX_DISTANCE]`.
    pub fn zoom_by_factor(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(Camera::MIN_DISTANCE, Camera::MAX_DISTANCE);
    }

    /// Add `delta` to the distance, clamped to
    /// `[Camera::MIN_DISTANCE, Camera::MAX_DISTANCE]`.
    pub fn zoom_by_delta(&mut self, delta: f32) {
        self.distance = (self.distance + delta).clamp(Camera::MIN_DISTANCE, Camera::MAX_DISTANCE);
    }
}

/// The viewport camera: where it is, how it projects, and how it exposes.
///
/// [`pose`](Self::pose) places the eye in f64 world space and orients it.
/// [`projection`](Self::projection) maps view space to the screen, and
/// [`aspect`](Self::aspect) is the viewport's width over height, set each
/// frame from the viewport size. [`exposure`](Self::exposure) carries the
/// physical camera settings and the focus distance.
///
/// Build an orbit camera with [`Camera::orbit`] or a pointed one with
/// [`Camera::look_at`]. The camera itself does not orbit: orbit controllers
/// keep the centre and distance, and write the camera's pose. The focus
/// distance doubles as the orbit distance, so [`target`](Self::target) is the
/// orbit centre of a camera an orbit controller drives.
///
/// All matrices are right-handed (wgpu NDC convention).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[non_exhaustive]
pub struct Camera {
    /// Eye position and orientation.
    pub pose: Pose,
    /// Projection.
    pub projection: Projection,
    /// Physical exposure settings and focus distance. The renderer reads
    /// exposure from `DisplaySettings::exposure`, not from here; set it
    /// from [`PhysicalExposure::to_exposure_mode`].
    pub exposure: PhysicalExposure,
    /// Viewport width / height ratio, updated each frame from the viewport rect.
    pub aspect: f32,
}

impl Default for Camera {
    /// An orbit camera 5 units from the origin, about 27 degrees above the
    /// x-y plane, with a 45 degree perspective.
    fn default() -> Self {
        // from_rotation_x(1.1) is ~63 deg from the top, ~27 deg above horizontal.
        Self::orbit(
            DVec3::ZERO,
            5.0,
            Quat::from_rotation_z(0.6) * Quat::from_rotation_x(1.1),
        )
    }
}

impl Camera {
    /// Minimum orbit distance the zoom helpers clamp to.
    pub const MIN_DISTANCE: f32 = 0.01;

    /// Maximum orbit distance the zoom helpers clamp to.
    pub const MAX_DISTANCE: f32 = 1.0e6;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// A camera with the given pose, the default perspective projection and
    /// the default exposure.
    pub fn new(pose: Pose) -> Self {
        Self {
            pose,
            projection: Projection::default(),
            exposure: PhysicalExposure::default(),
            aspect: 1.5,
        }
    }

    /// An orbit camera: the eye sits `distance` from `center` along the
    /// orientation's local `+Z` (`eye = center + orientation * (Z * distance)`),
    /// looking at `center`, which is also the focus point.
    pub fn orbit(center: impl Into<DVec3>, distance: f32, orientation: Quat) -> Self {
        let mut camera = Self::new(Pose::orbit(center, distance, orientation));
        camera.exposure.focus_distance = distance;
        camera
    }

    /// A camera at `eye` looking at `target`, with `up` as close to the
    /// screen's up as it can be. `target` is the focus point.
    pub fn look_at(eye: impl Into<DVec3>, target: impl Into<DVec3>, up: Vec3) -> Self {
        let (eye, target) = (eye.into(), target.into());
        let mut camera = Self::new(Pose::look_at(eye, target, up));
        camera.exposure.focus_distance = (target - eye).length() as f32;
        camera
    }

    /// Builder: set the projection.
    pub fn with_projection(mut self, projection: Projection) -> Self {
        self.projection = projection;
        self
    }

    /// Builder: set the aspect ratio (width / height).
    pub fn with_aspect(mut self, aspect: f32) -> Self {
        self.aspect = aspect;
        self
    }

    /// Builder: switch to the orthographic projection that frames the same
    /// height at the focus distance; see [`set_orthographic`](Self::set_orthographic).
    pub fn orthographic(mut self) -> Self {
        self.set_orthographic(true);
        self
    }

    /// Builder: set the vertical field of view; see [`set_fov_y`](Self::set_fov_y).
    pub fn with_fov_y(mut self, fov_y: f32) -> Self {
        self.set_fov_y(fov_y);
        self
    }

    /// Builder: set the exposure.
    pub fn with_exposure(mut self, exposure: PhysicalExposure) -> Self {
        self.exposure = exposure;
        self
    }

    // -----------------------------------------------------------------------
    // Pose
    // -----------------------------------------------------------------------

    /// Eye position in world space, in f64.
    pub fn eye(&self) -> DVec3 {
        self.pose.position
    }

    /// Camera orientation in f32.
    pub fn orientation(&self) -> Quat {
        self.pose.orientation_f32()
    }

    /// The direction the camera looks, in world space.
    pub fn forward(&self) -> Vec3 {
        self.pose.forward()
    }

    /// Camera right vector in world space.
    pub fn right(&self) -> Vec3 {
        self.pose.right()
    }

    /// Camera up vector in world space.
    pub fn up(&self) -> Vec3 {
        self.pose.up()
    }

    /// The point `distance` in front of the eye along the view direction.
    pub fn target_at(&self, distance: f32) -> DVec3 {
        self.pose.target_at(distance)
    }

    /// The focus point: [`focus_distance`](Self::focus_distance) in front of
    /// the eye. For a camera an orbit controller drives, the orbit centre.
    pub fn target(&self) -> DVec3 {
        self.target_at(self.focus_distance())
    }

    /// Distance from the eye to the focus point.
    pub fn focus_distance(&self) -> f32 {
        self.exposure.focus_distance
    }

    /// Set the focus distance without moving the camera.
    pub fn set_focus_distance(&mut self, distance: f32) {
        self.exposure.focus_distance = distance;
    }

    /// Replace the pose.
    pub fn set_pose(&mut self, pose: Pose) {
        self.pose = pose;
    }

    /// The camera's orbit view: its focus point, focus distance and
    /// orientation.
    pub fn orbit_target(&self) -> CameraTarget {
        CameraTarget {
            center: self.target(),
            distance: self.focus_distance(),
            orientation: self.orientation(),
        }
    }

    /// Place the camera on an orbit: the pose of [`Camera::orbit`], with the
    /// focus distance set to `distance`. An orthographic view height scales
    /// with the change in distance, so zooming frames the same way in both
    /// projections.
    pub fn set_orbit(&mut self, center: impl Into<DVec3>, distance: f32, orientation: Quat) {
        let previous = self.focus_distance();
        if let Projection::Orthographic { height, .. } = &mut self.projection {
            if previous > 0.0 && distance != previous {
                *height *= distance / previous;
            }
        }
        self.pose = Pose::orbit(center, distance, orientation);
        self.exposure.focus_distance = distance;
    }

    /// Move the orbit centre to `center`, keeping the distance and
    /// orientation.
    pub fn set_orbit_center(&mut self, center: impl Into<DVec3>) {
        self.set_orbit(center, self.focus_distance(), self.orientation());
    }

    /// Move the eye to `distance` from the orbit centre, keeping the centre
    /// and orientation. The distance is clamped to
    /// `[MIN_DISTANCE, MAX_DISTANCE]`.
    pub fn set_orbit_distance(&mut self, distance: f32) {
        let distance = distance.clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE);
        self.set_orbit(self.target(), distance, self.orientation());
    }

    /// Turn the camera to `orientation` (normalised) about the orbit centre,
    /// keeping the centre and distance.
    pub fn set_orbit_orientation(&mut self, orientation: Quat) {
        self.set_orbit(
            self.target(),
            self.focus_distance(),
            orientation.normalize(),
        );
    }

    /// [`set_orbit`](Self::set_orbit) from a [`CameraTarget`].
    pub fn set_orbit_target(&mut self, target: CameraTarget) {
        self.set_orbit(target.center, target.distance, target.orientation);
    }

    // -----------------------------------------------------------------------
    // Projection
    // -----------------------------------------------------------------------

    /// Vertical angle that frames the view at the focus distance: a
    /// perspective camera's field of view, or the matching angle for an
    /// orthographic one.
    pub fn fov_y(&self) -> f32 {
        self.projection.fov_y(self.focus_distance())
    }

    /// Set the vertical field of view in radians. An orthographic camera
    /// takes the height that angle frames at the focus distance; a matrix
    /// projection is left alone.
    pub fn set_fov_y(&mut self, fov_y: f32) {
        let distance = self.focus_distance();
        match &mut self.projection {
            Projection::Perspective { fov_y: f, .. } => *f = fov_y,
            Projection::Orthographic { height, .. } => {
                *height = 2.0 * (distance * (fov_y / 2.0).tan());
            }
            _ => {}
        }
    }

    /// Whether the projection is orthographic.
    pub fn is_orthographic(&self) -> bool {
        self.projection.is_orthographic()
    }

    /// Switch between perspective and orthographic, keeping the framing at
    /// the focus distance. A matrix projection is left alone.
    pub fn set_orthographic(&mut self, orthographic: bool) {
        let distance = self.focus_distance();
        self.projection = if orthographic {
            self.projection.to_orthographic(distance)
        } else {
            self.projection.to_perspective(distance)
        };
    }

    /// Set the aspect ratio from pixel dimensions. A zero or negative height
    /// gives `1.0`.
    pub fn set_aspect_ratio(&mut self, width: f32, height: f32) {
        self.aspect = if height > 0.0 { width / height } else { 1.0 };
    }

    /// Set the near and far plane distances. On a matrix projection, sets the
    /// culling range.
    pub fn set_clip_planes(&mut self, znear: f32, zfar: f32) {
        match &mut self.projection {
            Projection::Perspective { near, far, .. }
            | Projection::Orthographic { near, far, .. } => {
                *near = Some(znear);
                *far = Some(zfar);
            }
            Projection::Matrix { culling_range, .. } => *culling_range = (znear, zfar),
        }
    }

    /// `(near, far)` distances the camera culls between; see
    /// [`Projection::depth_range`].
    pub fn depth_range(&self) -> (f32, f32) {
        self.projection.depth_range(self.focus_distance())
    }

    /// World-space height of the view at view depth `depth`.
    pub fn view_height_at(&self, depth: f32) -> f32 {
        self.projection.view_height_at(depth)
    }

    // -----------------------------------------------------------------------
    // Matrices
    // -----------------------------------------------------------------------

    /// Right-handed view matrix (world -> camera space), in f32 world
    /// coordinates. Far from the world origin its translation loses
    /// precision; the renderer rebuilds it from the f64 eye.
    pub fn view_matrix(&self) -> Mat4 {
        let orientation = self.orientation();
        let distance = self.focus_distance();
        let up = orientation * Vec3::Y;
        if !(distance > 0.0 && distance.is_finite()) {
            return Mat4::look_to_rh(self.eye().as_vec3(), self.forward(), up);
        }
        // Built looking from the eye at the focus point, both rounded to f32
        // from the f64 focus point, so an orbit camera's matrix depends only
        // on its centre, distance and orientation.
        let center = self.target().as_vec3();
        let eye = center + orientation * (Vec3::Z * distance);
        Mat4::look_at_rh(eye, center, up)
    }

    /// The projection matrix the renderer draws with; see
    /// [`Projection::proj_matrix`].
    pub fn proj_matrix(&self) -> Mat4 {
        self.projection
            .proj_matrix(self.aspect, self.focus_distance())
    }

    /// The finite culling projection; see [`Projection::culling_proj_matrix`].
    pub fn culling_proj_matrix(&self) -> Mat4 {
        self.projection
            .culling_proj_matrix(self.aspect, self.focus_distance())
    }

    /// [`proj_matrix`](Self::proj_matrix) times the view matrix. Unproject
    /// screen points with [`screen_ray`](Self::screen_ray) rather than through
    /// its inverse at depth 0 and 1.
    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// [`culling_proj_matrix`](Self::culling_proj_matrix) times the view matrix.
    pub fn culling_view_proj_matrix(&self) -> Mat4 {
        self.culling_proj_matrix() * self.view_matrix()
    }

    /// The view frustum, from [`culling_view_proj_matrix`](Self::culling_view_proj_matrix).
    pub fn frustum(&self) -> crate::camera::frustum::Frustum {
        crate::camera::frustum::Frustum::from_view_proj(&self.culling_view_proj_matrix())
    }

    /// The world-space ray through a point on screen, as `(origin, direction)`
    /// with a unit direction.
    ///
    /// `ndc` is the point in normalised device coordinates: -1..1 left to
    /// right and bottom to top. A perspective ray starts at the eye; an
    /// orthographic ray starts on the near plane and runs along
    /// [`forward`](Self::forward).
    pub fn screen_ray(&self, ndc: Vec2) -> (Vec3, Vec3) {
        let forward = self.forward();
        let right = self.right();
        let up = self.up();
        let eye = self.eye().as_vec3();
        if self.is_orthographic() {
            let half_h = self.view_height_at(1.0) * 0.5;
            let half_w = half_h * self.aspect;
            let origin = eye
                + right * (ndc.x * half_w)
                + up * (ndc.y * half_h)
                + forward * self.depth_range().0;
            (origin, forward)
        } else {
            let tan_half = self.view_height_at(1.0) * 0.5;
            let dir = forward + right * (ndc.x * tan_half * self.aspect) + up * (ndc.y * tan_half);
            (eye, dir.normalize())
        }
    }

    // -----------------------------------------------------------------------
    // Framing helpers
    // -----------------------------------------------------------------------

    /// Frame a bounding sphere, keeping the orientation.
    pub fn frame_sphere(&mut self, center: impl Into<DVec3>, radius: f32) {
        let (c, d) = self.fit_sphere(center, radius);
        self.set_orbit(
            c,
            d.clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE),
            self.orientation(),
        );
    }

    /// Frame an AABB, keeping the orientation.
    pub fn frame_aabb(&mut self, aabb: &crate::scene::aabb::Aabb) {
        let (c, d) = self.fit_aabb(aabb);
        self.set_orbit(
            c,
            d.clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE),
            self.orientation(),
        );
    }

    /// The [`CameraTarget`] that frames a bounding sphere, keeping the
    /// orientation.
    pub fn fit_sphere_target(&self, center: impl Into<DVec3>, radius: f32) -> CameraTarget {
        let (c, d) = self.fit_sphere(center, radius);
        CameraTarget::new(c, d, self.orientation())
    }

    /// The [`CameraTarget`] that frames an AABB, keeping the orientation.
    pub fn fit_aabb_target(&self, aabb: &crate::scene::aabb::Aabb) -> CameraTarget {
        let (c, d) = self.fit_aabb(aabb);
        CameraTarget::new(c, d, self.orientation())
    }

    /// `(center, distance)` that frames a bounding sphere, keeping the
    /// orientation. Pads by 1.2x so the object does not touch the edges.
    pub fn fit_sphere(&self, center: impl Into<DVec3>, radius: f32) -> (DVec3, f32) {
        let distance = if self.is_orthographic() {
            radius * 1.2
        } else {
            radius / (self.fov_y() / 2.0).tan() * 1.2
        };
        (center.into(), distance)
    }

    /// `(center, distance)` that frames an AABB, through its bounding sphere.
    pub fn fit_aabb(&self, aabb: &crate::scene::aabb::Aabb) -> (DVec3, f32) {
        self.fit_sphere(aabb.center(), aabb.half_extents().length())
    }

    /// Centre the camera on a domain of the given extents, with the distance
    /// and clipping planes set so all of it is visible.
    pub fn center_on_domain(&mut self, nx: f32, ny: f32, nz: f32) {
        let center = Vec3::new(nx, ny, nz).as_dvec3() / 2.0;
        let diagonal = (nx * nx + ny * ny + nz * nz).sqrt();
        let distance = (diagonal / 2.0) / (self.fov_y() / 2.0).tan() * 1.2;
        self.set_orbit(center, distance, self.orientation());
        self.set_clip_planes((diagonal * 0.0001).max(0.01), (diagonal * 10.0).max(1000.0));
    }
}

#[cfg(test)]
#[path = "camera_tests.rs"]
mod tests;
