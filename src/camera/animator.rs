//! Smooth camera motion with exponential damping and fly-to animations.
//!
//! `CameraAnimator` does NOT own the [`Camera`]. It takes `&mut Camera` in
//! [`update()`](crate::camera::animator::CameraAnimator::update) so the application decides whether to
//! use animation at all.

use crate::camera::camera::{Camera, Projection};

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
/// 0.85 = smooth deceleration. Values are frame-rate independent via
/// `damping.powf(dt * 60.0)`.
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
pub struct CameraAnimator {
    damping: CameraDamping,
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
            camera.orbit(self.orbit_velocity.x, self.orbit_velocity.y);
            self.orbit_velocity *= self.damping.orbit.powf(dt * 60.0);
            if self.orbit_velocity.length() <= eps {
                self.orbit_velocity = glam::Vec2::ZERO;
            }
            changed = true;
        }

        // Apply pan velocity.
        // Note: the animator's pan_velocity.y convention is the opposite sign from
        // pan_world's up_delta (animator subtracts up, pan_world adds up), so negate y.
        if self.pan_velocity.length() > eps {
            camera.pan_world(self.pan_velocity.x, -self.pan_velocity.y);
            self.pan_velocity *= self.damping.pan.powf(dt * 60.0);
            if self.pan_velocity.length() <= eps {
                self.pan_velocity = glam::Vec2::ZERO;
            }
            changed = true;
        }

        // Apply zoom velocity.
        if self.zoom_velocity.abs() > eps {
            camera.zoom_by_delta(self.zoom_velocity);
            self.zoom_velocity *= self.damping.zoom.powf(dt * 60.0);
            if self.zoom_velocity.abs() <= eps {
                self.zoom_velocity = 0.0;
            }
            changed = true;
        }

        changed
    }
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
}
