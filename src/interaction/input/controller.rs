//! Orbit/pan/zoom camera controller.
//!
//! [`OrbitCameraController`] wraps [`super::viewport_input::ViewportInput`] and
//! applies resolved orbit / pan / zoom actions directly to a [`crate::Camera`].

use crate::Camera;

use super::action::Action;
use super::action_frame::ActionFrame;
use super::context::ViewportContext;
use super::event::ViewportEvent;
use super::mode::NavigationMode;
use super::preset::{BindingPreset, viewport_all_bindings, viewport_primitives_bindings};
use super::viewport_input::ViewportInput;

/// High-level orbit / pan / zoom camera controller.
///
/// Wraps the lower-level [`ViewportInput`] resolver and applies semantic
/// camera actions to a [`Camera`] in a single `apply_to_camera` call.
///
/// # Integration pattern (winit / single window)
///
/// ```text
/// // --- AppState construction ---
/// controller.begin_frame(ViewportContext { hovered: true, focused: true, viewport_size });
///
/// // --- window_event ---
/// controller.push_event(translated_event);
///
/// // --- RedrawRequested ---
/// controller.apply_to_camera(&mut state.camera);
/// controller.begin_frame(ViewportContext { hovered: true, focused: true, viewport_size });
/// // ... render ...
/// ```
///
/// # Integration pattern (eframe / egui)
///
/// ```text
/// // --- update() ---
/// controller.begin_frame(ViewportContext {
///     hovered: response.hovered(),
///     focused: response.has_focus(),
///     viewport_size: [rect.width(), rect.height()],
/// });
/// // push events from ui.input(|i| { ... })
/// controller.apply_to_camera(&mut self.camera);
/// ```
pub struct OrbitCameraController {
    input: ViewportInput,
    /// How drag input is interpreted by [`apply_to_camera`](Self::apply_to_camera).
    ///
    /// Defaults to [`NavigationMode::Arcball`].
    pub navigation_mode: NavigationMode,
    /// Movement speed for [`NavigationMode::FirstPerson`], in world units per frame.
    ///
    /// Defaults to `0.1`. Scale this to match your scene : a value of `0.1` is
    /// suitable for a scene with objects a few units across.
    pub fly_speed: f32,
    /// Sensitivity for drag-based orbit (radians per pixel).
    pub orbit_sensitivity: f32,
    /// Sensitivity for scroll-based zoom (scale factor per pixel).
    pub zoom_sensitivity: f32,
    /// Sensitivity applied to two-finger trackpad rotation gesture (radians per radian).
    /// Default: `1.0` (gesture angle applied directly to camera yaw).
    /// Set to `0.0` to suppress the gesture entirely.
    pub gesture_sensitivity: f32,
    /// Current viewport size (cached from the last `begin_frame`).
    viewport_size: [f32; 2],
}

impl OrbitCameraController {
    /// Default drag orbit sensitivity: 0.005 radians per pixel.
    pub const DEFAULT_ORBIT_SENSITIVITY: f32 = 0.005;
    /// Default scroll zoom sensitivity: 0.001 scale per pixel.
    pub const DEFAULT_ZOOM_SENSITIVITY: f32 = 0.001;
    /// Default gesture sensitivity: 1.0 (gesture radians applied 1:1 to camera yaw).
    pub const DEFAULT_GESTURE_SENSITIVITY: f32 = 1.0;
    /// Default first-person fly speed: 0.1 world units per frame.
    pub const DEFAULT_FLY_SPEED: f32 = 0.1;

    /// Create a controller from the given binding preset.
    pub fn new(preset: BindingPreset) -> Self {
        let bindings = match preset {
            BindingPreset::ViewportPrimitives => viewport_primitives_bindings(),
            BindingPreset::ViewportAll => viewport_all_bindings(),
        };
        Self {
            input: ViewportInput::new(bindings),
            navigation_mode: NavigationMode::Arcball,
            fly_speed: Self::DEFAULT_FLY_SPEED,
            orbit_sensitivity: Self::DEFAULT_ORBIT_SENSITIVITY,
            zoom_sensitivity: Self::DEFAULT_ZOOM_SENSITIVITY,
            gesture_sensitivity: Self::DEFAULT_GESTURE_SENSITIVITY,
            viewport_size: [1.0, 1.0],
        }
    }

    /// Create a controller with the [`BindingPreset::ViewportPrimitives`] preset.
    ///
    /// This is the canonical control scheme matching `examples/winit_primitives`.
    pub fn viewport_primitives() -> Self {
        Self::new(BindingPreset::ViewportPrimitives)
    }

    /// Create a controller with the [`BindingPreset::ViewportAll`] preset.
    ///
    /// Includes all camera navigation bindings plus keyboard shortcuts for
    /// normal mode, fly mode, and manipulation mode. Use this to replace
    /// [`crate::InputSystem`] entirely.
    pub fn viewport_all() -> Self {
        Self::new(BindingPreset::ViewportAll)
    }

    /// Begin a new frame.
    ///
    /// Resets per-frame accumulators and records viewport context (hover/focus
    /// state and size). Call this at the **end** of each rendered frame : after
    /// `apply_to_camera` : so the accumulator is ready for the next batch of
    /// events.
    ///
    /// Also call once immediately after construction to prime the accumulator.
    pub fn begin_frame(&mut self, ctx: ViewportContext) {
        self.viewport_size = ctx.viewport_size;
        self.input.begin_frame(ctx);
    }

    /// Push a single viewport-scoped event into the accumulator.
    ///
    /// Call this from the host's event handler whenever a relevant native event
    /// arrives, after translating it to a [`ViewportEvent`].
    pub fn push_event(&mut self, event: ViewportEvent) {
        self.input.push_event(event);
    }

    /// Resolve accumulated events into an [`ActionFrame`] without applying any
    /// camera navigation.
    ///
    /// Use this when the caller needs to inspect actions but camera movement
    /// should be suppressed : for example during gizmo manipulation or fly mode
    /// where the camera is driven by other logic.
    pub fn resolve(&self) -> ActionFrame {
        self.input.resolve()
    }

    /// Resolve accumulated events, apply camera navigation, and return the
    /// [`ActionFrame`] for this frame.
    ///
    /// Call this in the render / update step, **before** `begin_frame` for the
    /// next frame.
    ///
    /// The behavior of orbit drag depends on [`Self::navigation_mode`]:
    /// - [`NavigationMode::Arcball`]: unconstrained arcball (default).
    /// - [`NavigationMode::Turntable`]: yaw around world Z, pitch clamped to ±89°.
    /// - [`NavigationMode::Planar`]: pan only, orbit input is ignored.
    /// - [`NavigationMode::FirstPerson`]: mouselook + WASD translation. Requires
    ///   the `ViewportAll` binding preset so that movement keys are resolved.
    pub fn apply_to_camera(&mut self, camera: &mut Camera) -> ActionFrame {
        let frame = self.input.resolve();
        let nav = &frame.navigation;
        let h = self.viewport_size[1];

        match self.navigation_mode {
            NavigationMode::Arcball => {
                if nav.orbit != glam::Vec2::ZERO {
                    camera.orbit(
                        nav.orbit.x * self.orbit_sensitivity,
                        nav.orbit.y * self.orbit_sensitivity,
                    );
                }
                if nav.twist != 0.0 && self.gesture_sensitivity != 0.0 {
                    camera.orbit(nav.twist * self.gesture_sensitivity, 0.0);
                }
                if nav.pan != glam::Vec2::ZERO {
                    camera.pan_pixels(nav.pan, h);
                }
                if nav.zoom != 0.0 {
                    camera.zoom_by_factor(1.0 - nav.zoom * self.zoom_sensitivity);
                }
            }

            NavigationMode::Turntable => {
                if nav.orbit != glam::Vec2::ZERO {
                    let yaw = nav.orbit.x * self.orbit_sensitivity;
                    let pitch = nav.orbit.y * self.orbit_sensitivity;
                    apply_turntable(camera, yaw, pitch);
                }
                if nav.twist != 0.0 && self.gesture_sensitivity != 0.0 {
                    // Gesture twist is yaw-only in turntable mode.
                    apply_turntable(camera, nav.twist * self.gesture_sensitivity, 0.0);
                }
                if nav.pan != glam::Vec2::ZERO {
                    camera.pan_pixels(nav.pan, h);
                }
                if nav.zoom != 0.0 {
                    camera.zoom_by_factor(1.0 - nav.zoom * self.zoom_sensitivity);
                }
            }

            NavigationMode::Planar => {
                // Orbit input is silently ignored; pan and zoom still work.
                if nav.pan != glam::Vec2::ZERO {
                    camera.pan_pixels(nav.pan, h);
                }
                if nav.zoom != 0.0 {
                    camera.zoom_by_factor(1.0 - nav.zoom * self.zoom_sensitivity);
                }
            }

            NavigationMode::FirstPerson => {
                // Mouselook: drag rotates the view while the eye stays fixed.
                if nav.orbit != glam::Vec2::ZERO {
                    let yaw = nav.orbit.x * self.orbit_sensitivity;
                    let pitch = nav.orbit.y * self.orbit_sensitivity;
                    apply_firstperson_look(camera, yaw, pitch);
                }
                if nav.twist != 0.0 && self.gesture_sensitivity != 0.0 {
                    apply_firstperson_look(camera, nav.twist * self.gesture_sensitivity, 0.0);
                }

                // WASD / QE translation.
                let forward = -(camera.orientation * glam::Vec3::Z);
                let right = camera.orientation * glam::Vec3::X;
                let up = camera.orientation * glam::Vec3::Y;
                let speed = self.fly_speed;

                let mut move_delta = glam::Vec3::ZERO;
                if frame.is_active(Action::FlyForward) {
                    move_delta += forward * speed;
                }
                if frame.is_active(Action::FlyBackward) {
                    move_delta -= forward * speed;
                }
                if frame.is_active(Action::FlyRight) {
                    move_delta += right * speed;
                }
                if frame.is_active(Action::FlyLeft) {
                    move_delta -= right * speed;
                }
                if frame.is_active(Action::FlyUp) {
                    move_delta += up * speed;
                }
                if frame.is_active(Action::FlyDown) {
                    move_delta -= up * speed;
                }
                // Translate center (and thus eye) without changing orientation or distance.
                camera.center += move_delta;
            }
        }

        frame
    }
}

// ---------------------------------------------------------------------------
// Navigation mode helpers
// ---------------------------------------------------------------------------

/// Apply a turntable yaw + clamped pitch to the camera.
///
/// Yaw rotates around world Z. Pitch changes elevation in camera-local space,
/// but is blocked when the eye would reach ±89° above/below the XY plane.
fn apply_turntable(camera: &mut Camera, yaw: f32, pitch: f32) {
    // Yaw: pre-multiply by a world-Z rotation (same as the yaw component of Camera::orbit).
    if yaw != 0.0 {
        camera.orientation = (glam::Quat::from_rotation_z(-yaw) * camera.orientation).normalize();
    }

    if pitch != 0.0 {
        // Proposed orientation after applying pitch in camera-local space.
        let proposed = (camera.orientation * glam::Quat::from_rotation_x(-pitch)).normalize();

        // The eye direction is `orientation * Z`. Its Z-component equals
        // sin(elevation_angle), so clamping it to sin(±89°) keeps the camera
        // away from the poles.
        let max_sin_el = 89.0_f32.to_radians().sin(); // ≈ 0.9998
        let eye_z = (proposed * glam::Vec3::Z).z;

        if eye_z.abs() <= max_sin_el {
            camera.orientation = proposed;
        }
        // At the pole limit: silently discard the pitch to avoid jitter.
    }
}

/// Apply mouselook (yaw + pitch) while keeping the camera eye position fixed.
///
/// The orbit center is adjusted so that `eye = center + orientation * Z * distance`
/// remains constant after the rotation.
fn apply_firstperson_look(camera: &mut Camera, yaw: f32, pitch: f32) {
    let eye = camera.eye_position();
    camera.orientation = (glam::Quat::from_rotation_z(-yaw)
        * camera.orientation
        * glam::Quat::from_rotation_x(-pitch))
    .normalize();
    // Re-derive center so the eye does not shift.
    camera.center = eye - camera.orientation * (glam::Vec3::Z * camera.distance);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::input::binding::KeyCode;
    use crate::interaction::input::event::{ButtonState, ScrollUnits, ViewportEvent};

    fn make_ctx() -> ViewportContext {
        ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [800.0, 600.0],
        }
    }

    #[test]
    fn new_defaults() {
        let ctrl = OrbitCameraController::viewport_primitives();
        assert_eq!(ctrl.navigation_mode, NavigationMode::Arcball);
        assert!((ctrl.fly_speed - OrbitCameraController::DEFAULT_FLY_SPEED).abs() < 1e-6);
        assert!(
            (ctrl.orbit_sensitivity - OrbitCameraController::DEFAULT_ORBIT_SENSITIVITY).abs()
                < 1e-6
        );
    }

    #[test]
    fn resolve_no_events_zero_nav() {
        let mut ctrl = OrbitCameraController::viewport_all();
        ctrl.begin_frame(make_ctx());
        let frame = ctrl.resolve();
        assert_eq!(frame.navigation.orbit, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.pan, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.zoom, 0.0);
    }

    #[test]
    fn apply_zoom_changes_distance() {
        let mut ctrl = OrbitCameraController::viewport_primitives();
        ctrl.begin_frame(make_ctx());
        let mut cam = Camera::default();
        let d0 = cam.distance;
        ctrl.push_event(ViewportEvent::Wheel {
            delta: glam::Vec2::new(0.0, 100.0),
            units: ScrollUnits::Pixels,
        });
        ctrl.apply_to_camera(&mut cam);
        assert!(
            (cam.distance - d0).abs() > 1e-4,
            "zoom should change camera distance"
        );
    }

    #[test]
    fn planar_mode_ignores_orbit() {
        let mut ctrl = OrbitCameraController::viewport_primitives();
        ctrl.navigation_mode = NavigationMode::Planar;
        ctrl.begin_frame(make_ctx());
        let mut cam = Camera::default();
        let orient_before = cam.orientation;
        // Simulate a left-drag (orbit gesture in primitives preset)
        ctrl.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(100.0, 100.0),
        });
        ctrl.push_event(ViewportEvent::MouseButton {
            button: crate::interaction::input::binding::MouseButton::Left,
            state: ButtonState::Pressed,
        });
        ctrl.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(200.0, 200.0),
        });
        ctrl.apply_to_camera(&mut cam);
        assert!(
            (cam.orientation.x - orient_before.x).abs() < 1e-6
                && (cam.orientation.y - orient_before.y).abs() < 1e-6
                && (cam.orientation.z - orient_before.z).abs() < 1e-6
                && (cam.orientation.w - orient_before.w).abs() < 1e-6,
            "planar mode should not change orientation"
        );
    }

    #[test]
    fn turntable_pitch_clamped() {
        let mut cam = Camera::default();
        // Try to apply extreme pitch
        for _ in 0..1000 {
            apply_turntable(&mut cam, 0.0, 0.1);
        }
        // Check eye direction Z component stays within sin(89°)
        let eye_z = (cam.orientation * glam::Vec3::Z).z;
        let max_sin = 89.0_f32.to_radians().sin();
        assert!(
            eye_z.abs() <= max_sin + 1e-4,
            "turntable pitch should be clamped: eye_z={eye_z}"
        );
    }

    #[test]
    fn firstperson_look_preserves_eye() {
        let mut cam = Camera::default();
        let eye_before = cam.eye_position();
        apply_firstperson_look(&mut cam, 0.3, 0.2);
        let eye_after = cam.eye_position();
        let diff = (eye_after - eye_before).length();
        assert!(
            diff < 1e-3,
            "firstperson look should preserve eye position, diff={diff}"
        );
    }

    #[test]
    fn firstperson_fly_moves_camera() {
        let mut ctrl = OrbitCameraController::viewport_all();
        ctrl.navigation_mode = NavigationMode::FirstPerson;
        ctrl.fly_speed = 1.0;
        ctrl.begin_frame(make_ctx());
        let mut cam = Camera::default();
        cam.center = glam::Vec3::ZERO;
        cam.orientation = glam::Quat::IDENTITY;
        let center_before = cam.center;
        ctrl.push_event(ViewportEvent::Key {
            key: KeyCode::W,
            state: ButtonState::Pressed,
            repeat: false,
        });
        ctrl.apply_to_camera(&mut cam);
        assert!(
            (cam.center - center_before).length() > 0.5,
            "FlyForward should move camera center"
        );
    }
}
