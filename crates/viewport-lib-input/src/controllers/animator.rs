//! Smooth camera motion with exponential damping and fly-to animations.
//!
//! `CameraAnimator` does NOT own the [`Camera`]. It takes `&mut Camera` in
//! [`update()`](CameraAnimator::update) so the application decides whether to
//! use animation at all.

use viewport_lib_types::camera::{Camera, Projection};

use super::orbit::orbit_in_mode;
use crate::input::NavigationMode;

/// Easing function for fly-to animations.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Easing {
    /// Constant velocity (t).
    Linear,
    /// Decelerates near the end : feels natural for camera snapping.
    EaseOutCubic,
    /// Accelerates then decelerates : smooth for longer transitions.
    EaseInOutCubic,
}

impl Easing {
    /// Evaluate the easing curve at `t` (clamped to 0..1).
    pub fn eval(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseOutCubic => {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            }
            Self::EaseInOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let p = -2.0 * t + 2.0;
                    1.0 - p * p * p / 2.0
                }
            }
        }
    }
}

/// Damping coefficients for camera motion channels.
///
/// Each value is in the range `0.0..1.0` where 0 = no damping (instant) and
/// 0.85 = smooth deceleration. A value is the share of velocity kept per 60 Hz
/// frame; the animator scales it to the real frame time, so both how fast the
/// motion slows and how far it coasts in total are the same at any frame rate.
#[derive(Clone, Debug)]
pub struct CameraDamping {
    /// Orbit (rotation) damping factor.
    pub orbit: f32,
    /// Pan (translation) damping factor.
    pub pan: f32,
    /// Zoom (distance) damping factor.
    pub zoom: f32,
    /// Velocity magnitude below which motion stops.
    pub epsilon: f32,
}

impl Default for CameraDamping {
    fn default() -> Self {
        Self {
            orbit: 0.85,
            pan: 0.85,
            zoom: 0.85,
            epsilon: 0.0001,
        }
    }
}

/// An in-progress fly-to animation.
#[derive(Clone, Debug)]
struct CameraFlight {
    start_center: glam::Vec3,
    start_distance: f32,
    start_orientation: glam::Quat,
    #[allow(dead_code)]
    start_projection: Projection,
    target_center: glam::Vec3,
    target_distance: f32,
    target_orientation: glam::Quat,
    target_projection: Option<Projection>,
    duration: f32,
    elapsed: f32,
    easing: Easing,
}

/// Smooth camera controller with exponential damping and animated transitions.
///
/// Feed raw input deltas via [`apply_orbit`](Self::apply_orbit),
/// [`apply_pan`](Self::apply_pan), [`apply_zoom`](Self::apply_zoom). Then call
/// [`update`](Self::update) once per frame to apply decayed velocities to the
/// camera.
///
/// For animated transitions (view presets, zoom-to-fit), use
/// [`fly_to`](Self::fly_to). Any raw input during a flight cancels it.
///
/// The orbit coast rotates as an arcball unless told otherwise. Driving it from
/// an [`OrbitCameraController`](crate::controllers::OrbitCameraController) in
/// another mode, set the same mode with
/// [`with_navigation_mode`](Self::with_navigation_mode) so the coast turns the
/// camera the way the drag did.
pub struct CameraAnimator {
    damping: CameraDamping,
    navigation_mode: NavigationMode,
    /// (yaw_rate, pitch_rate) in radians.
    orbit_velocity: glam::Vec2,
    /// (right, up) in world units.
    pan_velocity: glam::Vec2,
    /// Distance change rate.
    zoom_velocity: f32,
    /// Active fly-to animation, if any.
    flight: Option<CameraFlight>,
}

impl CameraAnimator {
    /// Create an animator with custom damping.
    pub fn new(damping: CameraDamping) -> Self {
        Self {
            damping,
            navigation_mode: NavigationMode::Arcball,
            orbit_velocity: glam::Vec2::ZERO,
            pan_velocity: glam::Vec2::ZERO,
            zoom_velocity: 0.0,
            flight: None,
        }
    }

    /// Create an animator with sensible default damping values.
    pub fn with_default_damping() -> Self {
        Self::new(CameraDamping::default())
    }

    /// Builder: coast orbit motion as `mode` interprets an orbit drag, matching
    /// [`OrbitCameraController::navigation_mode`](crate::controllers::OrbitCameraController::navigation_mode).
    /// The default is [`NavigationMode::Arcball`].
    pub fn with_navigation_mode(mut self, mode: NavigationMode) -> Self {
        self.navigation_mode = mode;
        self
    }

    /// Set the mode the orbit coast rotates in. See
    /// [`with_navigation_mode`](Self::with_navigation_mode).
    pub fn set_navigation_mode(&mut self, mode: NavigationMode) {
        self.navigation_mode = mode;
    }

    /// The mode the orbit coast rotates in.
    pub fn navigation_mode(&self) -> NavigationMode {
        self.navigation_mode
    }

    /// Feed a raw orbit delta (yaw, pitch) in radians.
    pub fn apply_orbit(&mut self, yaw_delta: f32, pitch_delta: f32) {
        if self.flight.is_some() {
            self.flight = None;
        }
        self.orbit_velocity += glam::Vec2::new(yaw_delta, pitch_delta);
    }

    /// Feed a raw pan delta (right, up) in world units.
    pub fn apply_pan(&mut self, right_delta: f32, up_delta: f32) {
        if self.flight.is_some() {
            self.flight = None;
        }
        self.pan_velocity += glam::Vec2::new(right_delta, up_delta);
    }

    /// Feed a raw zoom delta (distance change).
    pub fn apply_zoom(&mut self, delta: f32) {
        if self.flight.is_some() {
            self.flight = None;
        }
        self.zoom_velocity += delta;
    }

    /// Start an animated fly-to transition with default easing.
    pub fn fly_to(
        &mut self,
        camera: &Camera,
        target_center: glam::Vec3,
        target_distance: f32,
        target_orientation: glam::Quat,
        duration: f32,
    ) {
        self.fly_to_full(
            camera,
            target_center,
            target_distance,
            target_orientation,
            None,
            duration,
            Easing::EaseOutCubic,
        );
    }

    /// Start an animated fly-to transition with custom easing and optional projection change.
    pub fn fly_to_with_easing(
        &mut self,
        camera: &Camera,
        target_center: glam::Vec3,
        target_distance: f32,
        target_orientation: glam::Quat,
        duration: f32,
        easing: Easing,
    ) {
        self.fly_to_full(
            camera,
            target_center,
            target_distance,
            target_orientation,
            None,
            duration,
            easing,
        );
    }

    /// Start a fly-to transition with all options.
    #[allow(clippy::too_many_arguments)]
    pub fn fly_to_full(
        &mut self,
        camera: &Camera,
        target_center: glam::Vec3,
        target_distance: f32,
        target_orientation: glam::Quat,
        target_projection: Option<Projection>,
        duration: f32,
        easing: Easing,
    ) {
        // Zero out velocities : the flight takes over.
        self.orbit_velocity = glam::Vec2::ZERO;
        self.pan_velocity = glam::Vec2::ZERO;
        self.zoom_velocity = 0.0;

        self.flight = Some(CameraFlight {
            start_center: camera.center(),
            start_distance: camera.distance(),
            start_orientation: camera.orientation(),
            start_projection: camera.projection,
            target_center,
            target_distance,
            target_orientation,
            target_projection,
            duration: duration.max(0.001),
            elapsed: 0.0,
            easing,
        });
    }

    /// Cancel any active fly-to animation.
    pub fn cancel_flight(&mut self) {
        self.flight = None;
    }

    /// The camera as it will be when the current flight lands: `camera` with the
    /// flight's target centre, distance, orientation and projection applied, or
    /// `camera` unchanged when no flight is active.
    ///
    /// Use this as the starting point for a move relative to the view, such as
    /// a quarter turn clicked while a flight is still under way, so it composes
    /// with where the camera is heading rather than with a half-interpolated
    /// orientation. Damped orbit, pan and zoom momentum has no fixed endpoint
    /// and is not included.
    pub fn destination(&self, camera: &Camera) -> Camera {
        let mut out = camera.clone();
        if let Some(flight) = &self.flight {
            out.set_center(flight.target_center);
            out.set_distance(flight.target_distance);
            out.set_orientation(flight.target_orientation);
            if let Some(proj) = flight.target_projection {
                out.projection = proj;
            }
        }
        out
    }

    /// Returns `true` if the animator has residual velocity or an active flight.
    pub fn is_animating(&self) -> bool {
        if self.flight.is_some() {
            return true;
        }
        let eps = self.damping.epsilon;
        self.orbit_velocity.length() > eps
            || self.pan_velocity.length() > eps
            || self.zoom_velocity.abs() > eps
    }

    /// Advance the animator by `dt` seconds and apply motion to `camera`.
    ///
    /// Returns `true` if the camera was modified (useful for triggering redraws).
    pub fn update(&mut self, dt: f32, camera: &mut Camera) -> bool {
        // Handle active flight.
        if let Some(ref mut flight) = self.flight {
            flight.elapsed += dt;
            let raw_t = (flight.elapsed / flight.duration).min(1.0);
            let t = flight.easing.eval(raw_t);

            camera.set_center(flight.start_center.lerp(flight.target_center, t));
            camera.set_distance(
                flight.start_distance + (flight.target_distance - flight.start_distance) * t,
            );
            camera.set_orientation(flight.start_orientation.slerp(flight.target_orientation, t));

            // Switch projection at the end of the animation.
            if raw_t >= 1.0 {
                if let Some(proj) = flight.target_projection {
                    camera.projection = proj;
                }
                self.flight = None;
            }
            return true;
        }

        let eps = self.damping.epsilon;
        let mut changed = false;

        // Apply orbit velocity.
        if self.orbit_velocity.length() > eps {
            let (step, decay) = coast_step(self.damping.orbit, dt);
            let turn = self.orbit_velocity * step;
            orbit_in_mode(camera, self.navigation_mode, turn.x, turn.y);
            self.orbit_velocity *= decay;
            if self.orbit_velocity.length() <= eps {
                self.orbit_velocity = glam::Vec2::ZERO;
            }
            changed = true;
        }

        // Apply pan velocity.
        // Note: the animator's pan_velocity.y convention is the opposite sign from
        // pan_world's up_delta (animator subtracts up, pan_world adds up), so negate y.
        if self.pan_velocity.length() > eps {
            let (step, decay) = coast_step(self.damping.pan, dt);
            camera.pan_world(self.pan_velocity.x * step, -self.pan_velocity.y * step);
            self.pan_velocity *= decay;
            if self.pan_velocity.length() <= eps {
                self.pan_velocity = glam::Vec2::ZERO;
            }
            changed = true;
        }

        // Apply zoom velocity.
        if self.zoom_velocity.abs() > eps {
            let (step, decay) = coast_step(self.damping.zoom, dt);
            camera.zoom_by_delta(self.zoom_velocity * step);
            self.zoom_velocity *= decay;
            if self.zoom_velocity.abs() <= eps {
                self.zoom_velocity = 0.0;
            }
            changed = true;
        }

        changed
    }
}

/// How much of a velocity to apply over `dt`, and the factor it decays by.
///
/// The velocity decays continuously, keeping `damping` of itself per 60th of a
/// second, and the step is the motion that decay covers in `dt`, normalised so
/// a 60 Hz frame applies exactly the velocity. The total coast is then
/// `v / (1 - damping)` however the time is sliced.
fn coast_step(damping: f32, dt: f32) -> (f32, f32) {
    let decay = damping.powf(dt * 60.0);
    let step = if damping < 1.0 {
        (1.0 - decay) / (1.0 - damping)
    } else {
        // No decay at all: the velocity is a plain rate per 60 Hz frame.
        dt * 60.0
    };
    (step, decay)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_camera() -> Camera {
        Camera::default()
    }

    #[test]
    fn test_damping_decays_velocity() {
        let mut anim = CameraAnimator::with_default_damping();
        let mut cam = default_camera();
        anim.apply_orbit(0.5, 0.3);
        assert!(anim.is_animating());
        // Run many frames : velocity should decay to zero.
        for _ in 0..300 {
            anim.update(1.0 / 60.0, &mut cam);
        }
        assert!(!anim.is_animating(), "should have settled after 300 frames");
    }

    #[test]
    fn test_zero_damping_passes_through() {
        let damping = CameraDamping {
            orbit: 0.0,
            pan: 0.0,
            zoom: 0.0,
            epsilon: 0.0001,
        };
        let mut anim = CameraAnimator::new(damping);
        let mut cam = default_camera();
        let orig_orientation = cam.orientation;
        anim.apply_orbit(0.1, 0.0);
        anim.update(1.0 / 60.0, &mut cam);
        // With zero damping, velocity decays instantly (0^anything=0).
        assert!(
            !anim.is_animating(),
            "zero damping should settle in one frame"
        );
        // Camera should have moved.
        assert!(
            (cam.orientation.x - orig_orientation.x).abs() > 1e-6
                || (cam.orientation.y - orig_orientation.y).abs() > 1e-6,
            "camera orientation should have changed"
        );
    }

    #[test]
    fn test_fly_to_reaches_target() {
        let mut anim = CameraAnimator::with_default_damping();
        let mut cam = default_camera();
        let target_center = glam::Vec3::new(10.0, 20.0, 30.0);
        let target_dist = 15.0;
        let target_orient = glam::Quat::from_rotation_y(std::f32::consts::PI);
        anim.fly_to(&cam, target_center, target_dist, target_orient, 0.5);

        // Advance well past the duration.
        for _ in 0..120 {
            anim.update(1.0 / 60.0, &mut cam);
        }
        assert!(
            (cam.center - target_center).length() < 1e-4,
            "center should match target: {:?}",
            cam.center
        );
        assert!(
            (cam.distance - target_dist).abs() < 1e-4,
            "distance should match target: {}",
            cam.distance
        );
    }

    #[test]
    fn destination_without_flight_is_the_camera() {
        let anim = CameraAnimator::with_default_damping();
        let cam = default_camera();
        let dest = anim.destination(&cam);
        assert_eq!(dest.center, cam.center);
        assert_eq!(dest.distance, cam.distance);
        assert_eq!(dest.orientation, cam.orientation);
    }

    #[test]
    fn destination_mid_flight_is_where_the_flight_lands() {
        let mut anim = CameraAnimator::with_default_damping();
        let mut cam = default_camera();
        let target_orient = glam::Quat::from_rotation_z(1.2) * glam::Quat::from_rotation_x(0.4);
        anim.fly_to_full(
            &cam,
            glam::Vec3::new(3.0, -2.0, 5.0),
            12.0,
            target_orient,
            Some(Projection::Orthographic),
            0.5,
            Easing::EaseInOutCubic,
        );
        for _ in 0..10 {
            anim.update(1.0 / 60.0, &mut cam);
        }
        assert!(anim.is_animating(), "flight should still be under way");
        let dest = anim.destination(&cam);
        assert!(
            cam.orientation.angle_between(target_orient) > 0.1,
            "the camera should be part-way, not at the target"
        );

        for _ in 0..60 {
            anim.update(1.0 / 60.0, &mut cam);
        }
        assert!(!anim.is_animating());
        assert!((dest.center - cam.center).length() < 1e-4);
        assert!((dest.distance - cam.distance).abs() < 1e-4);
        assert!(dest.orientation.angle_between(cam.orientation) < 1e-3);
        assert_eq!(dest.projection, cam.projection);
        assert_eq!(dest.projection, Projection::Orthographic);
    }

    #[test]
    fn test_fly_to_cancelled_by_input() {
        let mut anim = CameraAnimator::with_default_damping();
        let mut cam = default_camera();
        let target = glam::Vec3::new(100.0, 0.0, 0.0);
        anim.fly_to(&cam, target, 50.0, glam::Quat::IDENTITY, 1.0);

        // Advance a few frames.
        for _ in 0..5 {
            anim.update(1.0 / 60.0, &mut cam);
        }
        // Apply user input : should cancel the flight.
        anim.apply_orbit(0.1, 0.0);
        // The flight should be gone.
        anim.update(1.0 / 60.0, &mut cam);
        // Camera should NOT have reached the target.
        assert!(
            (cam.center - target).length() > 1.0,
            "flight should have been cancelled"
        );
    }

    #[test]
    fn test_is_animating_reflects_state() {
        let mut anim = CameraAnimator::with_default_damping();
        let mut cam = default_camera();
        assert!(!anim.is_animating());

        anim.apply_zoom(1.0);
        assert!(anim.is_animating());

        for _ in 0..300 {
            anim.update(1.0 / 60.0, &mut cam);
        }
        assert!(!anim.is_animating());
    }

    #[test]
    fn test_easing_boundaries() {
        for easing in [Easing::Linear, Easing::EaseOutCubic, Easing::EaseInOutCubic] {
            let v0 = easing.eval(0.0);
            let v1 = easing.eval(1.0);
            assert!(v0.abs() < 1e-6, "{easing:?}: eval(0) = {v0}, expected 0");
            assert!(
                (v1 - 1.0).abs() < 1e-6,
                "{easing:?}: eval(1) = {v1}, expected 1"
            );
        }
    }

    /// Coast a pan flick at a fixed frame rate until it settles, returning how
    /// far the centre moved.
    fn pan_coast(hz: f32) -> f32 {
        let mut anim = CameraAnimator::with_default_damping();
        let mut cam = default_camera();
        let start = cam.center;
        anim.apply_pan(0.1, 0.0);
        for _ in 0..(hz as usize * 10) {
            anim.update(1.0 / hz, &mut cam);
        }
        (cam.center - start).length()
    }

    #[test]
    fn coast_distance_does_not_depend_on_frame_rate() {
        let at_60 = pan_coast(60.0);
        // At 60 Hz the coast is v / (1 - d), as it always was.
        assert!(
            (at_60 - 0.1 / (1.0 - 0.85)).abs() < 1e-3,
            "60 Hz coast={at_60}"
        );
        for hz in [30.0, 120.0, 144.0, 240.0] {
            let coast = pan_coast(hz);
            assert!(
                (coast - at_60).abs() < 1e-3,
                "{hz} Hz coast={coast}, 60 Hz coast={at_60}"
            );
        }
    }

    #[test]
    fn fly_coast_turns_about_the_eye() {
        let mut cam = default_camera();
        cam.orientation = crate::LookAngles::new(0.3, -0.4).orientation();
        let eye = cam.eye_position();
        let mut anim =
            CameraAnimator::with_default_damping().with_navigation_mode(NavigationMode::Fly);
        anim.apply_orbit(0.2, 0.15);
        for _ in 0..600 {
            anim.update(1.0 / 60.0, &mut cam);
        }
        assert!((cam.eye_position() - eye).length() < 1e-3, "the eye moved");
        // The same flick coasted as an arcball swings the eye round the centre.
        let mut arcball_cam = default_camera();
        arcball_cam.orientation = crate::LookAngles::new(0.3, -0.4).orientation();
        let mut arcball = CameraAnimator::with_default_damping();
        arcball.apply_orbit(0.2, 0.15);
        for _ in 0..600 {
            arcball.update(1.0 / 60.0, &mut arcball_cam);
        }
        assert!((arcball_cam.eye_position() - eye).length() > 0.1);
    }

    #[test]
    fn turntable_coast_stops_at_the_pole_limit() {
        let mut cam = default_camera();
        cam.orientation = crate::LookAngles::new(0.0, -1.3).orientation();
        let mut anim =
            CameraAnimator::with_default_damping().with_navigation_mode(NavigationMode::Turntable);
        // A flick hard over the top.
        anim.apply_orbit(0.0, 0.5);
        for _ in 0..600 {
            anim.update(1.0 / 60.0, &mut cam);
        }
        let eye_elevation = (cam.orientation * glam::Vec3::Z).z.asin();
        assert!(
            eye_elevation < 89.0_f32.to_radians() + 1e-3,
            "elevation={eye_elevation}"
        );
    }

    #[test]
    fn arcball_is_the_default_mode() {
        assert_eq!(
            CameraAnimator::with_default_damping().navigation_mode(),
            NavigationMode::Arcball
        );
    }
}
