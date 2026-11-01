//! Orbit/pan/zoom camera controller.
//!
//! [`OrbitCameraController`] wraps [`crate::input::viewport_input::ViewportInput`] and
//! applies resolved orbit / pan / zoom actions directly to a [`crate::Camera`].

use crate::{Camera, LookAngles};
use viewport_lib_types::camera::{CameraTarget, Pose};

use crate::input::action::Action;
use crate::input::action_frame::ActionFrame;
use crate::input::context::ViewportContext;
use crate::input::event::ViewportEvent;
use crate::input::mode::NavigationMode;
use crate::input::preset::{BindingPreset, viewer_bindings, viewport_default_bindings};
use crate::input::viewport_binding::ViewportBinding;
use crate::input::viewport_input::ViewportInput;

use super::movement::wish_from_actions;

/// High-level orbit / pan / zoom camera controller.
///
/// Wraps the lower-level [`ViewportInput`] resolver and applies semantic
/// camera actions to a [`Camera`] in a single `apply_to_camera` call.
///
/// The controller keeps the orbit centre and distance as its own state and
/// writes the camera's pose, with the focus distance set to the orbit
/// distance. When something else moves the camera (a fly-to, a preset, a
/// host writing the pose), the next call reads the pivot back from the
/// camera's focus point, so the orbit continues from the new view.
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
///     focused: response.hovered() && !ui.ctx().wants_keyboard_input(),
///     viewport_size: [rect.width(), rect.height()],
/// });
/// // push each event in ui.input(|i| { ... }) through from_egui_into
/// controller.apply_to_camera(&mut self.camera);
/// ```
pub struct OrbitCameraController {
    input: ViewportInput,
    /// How drag input is interpreted by [`apply_to_camera`](Self::apply_to_camera).
    ///
    /// Defaults to [`NavigationMode::Arcball`].
    pub navigation_mode: NavigationMode,
    /// Movement speed for [`NavigationMode::Fly`], in world units per second.
    /// Moving diagonally is as fast as moving straight.
    ///
    /// Defaults to `6.0`. Scale this to match your scene : `6.0` suits a scene
    /// with objects a few units across.
    pub fly_speed: f32,
    /// Multiplier on [`fly_speed`](Self::fly_speed) while
    /// [`Action::FlySpeedBoost`](crate::input::Action::FlySpeedBoost) is held
    /// (Shift in the default bindings). Defaults to `3.0`.
    pub fly_boost: f32,
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
    /// The orbit centre, distance and orientation the controller turns about.
    pivot: Option<CameraTarget>,
    /// The camera pose and focus distance last written, to tell when something
    /// else has moved the camera and the pivot must be read from it again.
    written: Option<(Pose, f32)>,
}

impl OrbitCameraController {
    /// Default drag orbit sensitivity: 0.005 radians per pixel.
    pub const DEFAULT_ORBIT_SENSITIVITY: f32 = 0.005;
    /// Default scroll zoom sensitivity: 0.001 scale per pixel.
    pub const DEFAULT_ZOOM_SENSITIVITY: f32 = 0.001;
    /// Default gesture sensitivity: 1.0 (gesture radians applied 1:1 to camera yaw).
    pub const DEFAULT_GESTURE_SENSITIVITY: f32 = 1.0;
    /// Default fly speed: 6 world units per second.
    pub const DEFAULT_FLY_SPEED: f32 = 6.0;
    /// Default fly speed multiplier while boosting.
    pub const DEFAULT_FLY_BOOST: f32 = 3.0;

    /// Create a controller that holds no bindings of its own.
    ///
    /// This is the one to use. The host (or a `ViewportInstance`) owns the
    /// [`ViewportInput`] that holds the bindings and resolves the frame; the
    /// controller just applies the resulting [`ActionFrame`] to a camera through
    /// [`apply`](Self::apply). Sensitivities stay on the controller.
    pub fn new_stateless() -> Self {
        Self::from_bindings(Vec::new())
    }

    /// Create a controller from the given binding preset.
    ///
    /// The preset seeds an internal resolver used only by the deprecated
    /// event-driven methods ([`push_event`](Self::push_event) and
    /// [`apply_to_camera`](Self::apply_to_camera)). **A session ignores it**:
    /// `ViewportInstance` resolves with its own bindings and calls
    /// [`apply`](Self::apply), so the preset chosen here has no effect there. Set the
    /// scheme on the session instead, with `ViewportInstance::with_bindings`.
    pub fn new(preset: BindingPreset) -> Self {
        let bindings = match preset {
            BindingPreset::Default => viewport_default_bindings(),
            BindingPreset::Viewer => viewer_bindings(),
            // BindingPreset is non_exhaustive; an unknown preset gets the default set.
            _ => viewport_default_bindings(),
        };
        Self::from_bindings(bindings)
    }

    fn from_bindings(bindings: Vec<ViewportBinding>) -> Self {
        Self {
            input: ViewportInput::new(bindings),
            navigation_mode: NavigationMode::Arcball,
            fly_speed: Self::DEFAULT_FLY_SPEED,
            fly_boost: Self::DEFAULT_FLY_BOOST,
            orbit_sensitivity: Self::DEFAULT_ORBIT_SENSITIVITY,
            zoom_sensitivity: Self::DEFAULT_ZOOM_SENSITIVITY,
            gesture_sensitivity: Self::DEFAULT_GESTURE_SENSITIVITY,
            viewport_size: [1.0, 1.0],
            pivot: None,
            written: None,
        }
    }

    /// Create a controller seeded with the [`BindingPreset::Viewer`] scheme.
    ///
    /// The scheme has changed: left drag still orbits, but right drag no longer pans.
    /// See [`new`](Self::new) for why a session ignores this.
    pub fn viewer() -> Self {
        Self::new(BindingPreset::Viewer)
    }

    /// Former name for [`viewer`](Self::viewer).
    #[deprecated(
        note = "renamed: use OrbitCameraController::viewer, or new_stateless if the host owns the resolver"
    )]
    pub fn viewport_primitives() -> Self {
        Self::new(BindingPreset::Viewer)
    }

    /// Former name for `new(BindingPreset::Default)`, from when that preset was
    /// called `ViewportAll`.
    #[deprecated(
        note = "renamed: use OrbitCameraController::new_stateless if a session or host owns the resolver, or new(BindingPreset::Default) for the standalone path"
    )]
    pub fn viewport_all() -> Self {
        Self::new(BindingPreset::Default)
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

    /// Record the viewport size used for pan when applying a frame through
    /// [`apply`](Self::apply) without going through
    /// [`begin_frame`](Self::begin_frame).
    pub fn set_viewport_size(&mut self, viewport_size: [f32; 2]) {
        self.viewport_size = viewport_size;
    }

    /// Push a single viewport-scoped event into the controller's own accumulator.
    ///
    /// Superseded: own a [`ViewportInput`] (or a `ViewportInstance`), push events
    /// into that, and drive the camera with [`apply`](Self::apply). Bindings belong
    /// to the resolver, and a controller holding a second one is how a session and a
    /// controller end up disagreeing about the control scheme.
    #[deprecated(
        note = "own a ViewportInput and use OrbitCameraController::apply(&mut camera, &frame, dt)"
    )]
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
    /// - [`NavigationMode::Turntable`]: yaw around world Z, pitch clamped to +/-89 deg.
    /// - [`NavigationMode::Planar`]: pan only, orbit input is ignored.
    /// - [`NavigationMode::Fly`]: mouselook + WASD translation. Requires
    ///   the `Default` binding preset so that movement keys are resolved.
    ///
    /// Takes no frame time, so fly movement assumes 60 frames per second.
    #[deprecated(
        note = "own a ViewportInput and use OrbitCameraController::apply(&mut camera, &frame, dt)"
    )]
    pub fn apply_to_camera(&mut self, camera: &mut Camera) -> ActionFrame {
        let frame = self.input.resolve();
        self.apply(camera, &frame, 1.0 / 60.0);
        frame
    }

    /// Orbit `camera` by a pointer delta in pixels, as a drag in the viewport
    /// would: scaled by [`orbit_sensitivity`](Self::orbit_sensitivity) and
    /// applied in [`navigation_mode`](Self::navigation_mode). Planar mode
    /// ignores it.
    ///
    /// Use this for a drag the viewport did not see itself, such as a drag on
    /// an orientation widget, so it moves the camera the same way a drag in the
    /// scene does.
    pub fn orbit_by(&mut self, camera: &mut Camera, delta: glam::Vec2) {
        if delta == glam::Vec2::ZERO {
            return;
        }
        let mut target = self.pivot(camera);
        self.orbit_target_by(&mut target, delta);
        self.write(camera, target);
    }

    /// The orbit centre, distance and orientation the controller last wrote,
    /// or `None` before it has moved a camera.
    pub fn target(&self) -> Option<CameraTarget> {
        self.pivot
    }

    /// Read the pivot from `camera`'s focus point now rather than on the next
    /// call that sees the camera has moved.
    pub fn sync_from_camera(&mut self, camera: &Camera) {
        self.pivot = Some(camera.orbit_target());
        self.written = Some((camera.pose, camera.focus_distance()));
    }

    /// The pivot to work from: the controller's own, unless the camera has
    /// moved since the controller last wrote it.
    fn pivot(&mut self, camera: &Camera) -> CameraTarget {
        let current = (camera.pose, camera.focus_distance());
        match self.pivot {
            Some(pivot) if self.written == Some(current) => pivot,
            _ => {
                self.sync_from_camera(camera);
                camera.orbit_target()
            }
        }
    }

    fn write(&mut self, camera: &mut Camera, target: CameraTarget) {
        camera.set_orbit_target(target);
        self.pivot = Some(target);
        self.written = Some((camera.pose, camera.focus_distance()));
    }

    fn orbit_target_by(&self, target: &mut CameraTarget, delta: glam::Vec2) {
        if delta == glam::Vec2::ZERO {
            return;
        }
        let yaw = delta.x * self.orbit_sensitivity;
        let pitch = delta.y * self.orbit_sensitivity;
        orbit_in_mode(target, self.navigation_mode, yaw, pitch);
    }

    /// Apply an already-resolved [`ActionFrame`] to the camera without draining
    /// the event queue.
    ///
    /// Use this when the host owns a single [`ViewportInput`] resolver shared
    /// across several controllers (orbit, first-person, third-person) and feeds
    /// the same frame to whichever one is active.
    /// [`apply_to_camera`](Self::apply_to_camera) is the convenience wrapper
    /// that resolves and applies in one call.
    ///
    /// Pan needs the viewport height, read from the size recorded by the last
    /// [`begin_frame`](Self::begin_frame) or
    /// [`set_viewport_size`](Self::set_viewport_size).
    ///
    /// `dt` is the frame time in seconds. Only held input reads it: fly movement
    /// in [`NavigationMode::Fly`]. Drags, the wheel and touch arrive as deltas
    /// and move the camera the same at any frame rate.
    pub fn apply(&mut self, camera: &mut Camera, frame: &ActionFrame, dt: f32) {
        let nav = &frame.navigation;
        let h = self.viewport_size[1];
        let start = self.pivot(camera);
        let mut t = start;
        let gesture = nav.twist * self.gesture_sensitivity;
        let pan = |t: &mut CameraTarget| {
            if nav.pan != glam::Vec2::ZERO {
                let scale = camera.view_height_at(t.distance) / h.max(1.0);
                t.pan_world(nav.pan.x * scale, nav.pan.y * scale);
            }
        };
        let zoom = |t: &mut CameraTarget| {
            if nav.zoom != 0.0 {
                t.zoom_by_factor(1.0 - nav.zoom * self.zoom_sensitivity);
            }
        };

        match self.navigation_mode {
            NavigationMode::Arcball => {
                self.orbit_target_by(&mut t, nav.orbit);
                if gesture != 0.0 {
                    t.orbit(gesture, 0.0);
                }
                pan(&mut t);
                zoom(&mut t);
            }

            NavigationMode::Turntable => {
                self.orbit_target_by(&mut t, nav.orbit);
                if gesture != 0.0 {
                    // Gesture twist is yaw-only in turntable mode.
                    apply_turntable(&mut t, gesture, 0.0);
                }
                pan(&mut t);
                zoom(&mut t);
            }

            NavigationMode::Planar => {
                // Orbit input is silently ignored; pan and zoom still work.
                pan(&mut t);
                zoom(&mut t);
            }

            NavigationMode::Fly => {
                // Mouselook: drag rotates the view while the eye stays fixed.
                self.orbit_target_by(&mut t, nav.orbit);
                if gesture != 0.0 {
                    apply_fly_look(&mut t, gesture, 0.0);
                }

                // WASD / QE translation in the view's own basis.
                let q = t.orientation;
                let forward = -(q * glam::Vec3::Z);
                let wish = wish_from_actions(frame, forward, q * glam::Vec3::X, q * glam::Vec3::Y);
                let speed = if frame.is_active(Action::FlySpeedBoost) {
                    self.fly_speed * self.fly_boost
                } else {
                    self.fly_speed
                };
                // Translate the centre (and so the eye) without turning or zooming.
                t.centre += (wish * speed * dt).as_dvec3();
            }

            // NavigationMode is non_exhaustive; no-op for modes not handled here.
            _ => {}
        }

        if t != start {
            self.write(camera, t);
        }
    }
}

// ---------------------------------------------------------------------------
// Navigation mode helpers
// ---------------------------------------------------------------------------

/// Highest eye elevation, above or below the horizon, that turntable and fly
/// look reach: just short of the poles, where yaw stops meaning anything.
const POLE_LIMIT: f32 = 89.0 * std::f32::consts::PI / 180.0;

/// Rotate an orbit view by `yaw` and `pitch` radians the way `mode` interprets
/// an orbit drag. Planar ignores it.
///
/// The one rotation path for the orbit controller and for a
/// [`CameraAnimator`](crate::controllers::CameraAnimator) coasting in a mode, so
/// a coast after a drag turns the camera the way the drag did.
pub(crate) fn orbit_in_mode(target: &mut CameraTarget, mode: NavigationMode, yaw: f32, pitch: f32) {
    match mode {
        NavigationMode::Arcball => target.orbit(yaw, pitch),
        NavigationMode::Turntable => apply_turntable(target, yaw, pitch),
        NavigationMode::Fly => apply_fly_look(target, yaw, pitch),
        // Planar ignores orbit; NavigationMode is non_exhaustive.
        _ => {}
    }
}

/// Apply a turntable yaw + clamped pitch to an orbit view.
///
/// Yaw rotates around world Z. Pitch changes elevation in camera-local space,
/// stopping at [`POLE_LIMIT`] above or below the XY plane. A step that would
/// cross the limit takes the part of it that reaches the limit, so a fast drag
/// ends at the limit rather than short of it, and a step that starts past the
/// limit can still come back toward the horizon.
fn apply_turntable(view: &mut CameraTarget, yaw: f32, pitch: f32) {
    // Yaw: pre-multiply by a world-Z rotation (same as the yaw component of Camera::orbit).
    if yaw != 0.0 {
        view.orientation = (glam::Quat::from_rotation_z(-yaw) * view.orientation).normalize();
    }

    if pitch != 0.0 {
        // Elevation of the eye above the XY plane: the eye direction is
        // `orientation * Z`, and its z-component is the sine of the elevation.
        let elevation = |q: glam::Quat| (q * glam::Vec3::Z).z.clamp(-1.0, 1.0).asin();
        let pitched =
            |amount: f32| (view.orientation * glam::Quat::from_rotation_x(-amount)).normalize();

        // How fast elevation moves per radian of pitch, from a small probe step:
        // +/-1 when the camera carries no roll, less when it does. Working from the
        // rate rather than the end orientation keeps a step large enough to pass
        // over the pole from looking like a small one.
        const PROBE: f32 = 1e-3;
        let current = elevation(view.orientation);
        let rate = (elevation(pitched(PROBE)) - current) / PROBE;
        if rate.abs() < 1e-3 {
            // Rolled on its side: pitch does not move the elevation.
            view.orientation = pitched(pitch);
            return;
        }

        let target = current + rate * pitch;
        if target.abs() <= POLE_LIMIT {
            view.orientation = pitched(pitch);
        } else if target.abs() > current.abs() && current.abs() >= POLE_LIMIT {
            // Already at or past the limit and heading further out.
        } else {
            view.orientation = pitched((POLE_LIMIT.copysign(target) - current) / rate);
        }
    }
}

/// Apply mouselook (yaw + pitch) while keeping the camera eye position fixed.
///
/// Yaw turns around world Z and pitch tilts the view, held to [`POLE_LIMIT`]
/// so the view cannot go over the top and roll. Roll the camera carried in from
/// another mode is levelled out. The orbit centre moves so the eye stays put.
fn apply_fly_look(view: &mut CameraTarget, yaw: f32, pitch: f32) {
    let eye = view.eye();
    // A positive pitch delta is a downward drag, which looks down.
    let look = LookAngles::from_orientation(view.orientation)
        .turned(yaw, -pitch)
        .with_pitch_clamped(-POLE_LIMIT, POLE_LIMIT);
    view.orientation = look.orientation();
    view.centre = eye + (look.direction() * view.distance).as_dvec3();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::binding::KeyCode;
    use glam::DVec3;

    fn camera_facing(orientation: glam::Quat) -> Camera {
        Camera::orbit(DVec3::ZERO, 5.0, orientation)
    }

    fn default_target() -> CameraTarget {
        Camera::default().orbit_target()
    }
    use crate::input::event::{ButtonState, ScrollUnits, ViewportEvent};

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
        let mut ctrl = OrbitCameraController::new(BindingPreset::Default);
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
        let d0 = cam.focus_distance();
        ctrl.push_event(ViewportEvent::Wheel {
            delta: glam::Vec2::new(0.0, 100.0),
            units: ScrollUnits::Pixels,
        });
        ctrl.apply_to_camera(&mut cam);
        assert!(
            (cam.focus_distance() - d0).abs() > 1e-4,
            "zoom should change camera distance"
        );
    }

    #[test]
    fn orbit_by_matches_a_drag_in_every_mode() {
        let delta = glam::Vec2::new(37.0, -21.0);
        let mut frame = ActionFrame::default();
        frame.navigation.orbit = delta;
        let start =
            || camera_facing(glam::Quat::from_rotation_z(0.7) * glam::Quat::from_rotation_x(0.9));
        for mode in [
            NavigationMode::Arcball,
            NavigationMode::Turntable,
            NavigationMode::Fly,
            NavigationMode::Planar,
        ] {
            let mut ctrl = OrbitCameraController::viewer();
            ctrl.navigation_mode = mode;
            let mut by_drag = start();
            ctrl.apply(&mut by_drag, &frame, 1.0 / 60.0);
            let mut by_call = start();
            ctrl.orbit_by(&mut by_call, delta);
            assert!(
                by_drag.orientation().angle_between(by_call.orientation()) < 1e-5,
                "{mode:?}: orientation differs"
            );
            assert!(
                (by_drag.target() - by_call.target()).length() < 1e-4,
                "{mode:?}: centre differs"
            );
            match mode {
                NavigationMode::Planar => assert_eq!(by_call.orientation(), start().orientation()),
                // Mouselook keeps the eye still, so the centre moves: proof the
                // mode's own rotation ran rather than the arcball one.
                NavigationMode::Fly => {
                    assert!((by_call.target() - start().target()).length() > 1e-3)
                }
                _ => assert!(by_call.orientation().angle_between(start().orientation()) > 1e-3),
            }
        }
    }

    #[test]
    fn planar_mode_ignores_orbit() {
        let mut ctrl = OrbitCameraController::viewport_primitives();
        ctrl.navigation_mode = NavigationMode::Planar;
        ctrl.begin_frame(make_ctx());
        let mut cam = Camera::default();
        let orient_before = cam.orientation();
        // Simulate a left-drag (orbit gesture in primitives preset)
        ctrl.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(100.0, 100.0),
        });
        ctrl.push_event(ViewportEvent::MouseButton {
            button: crate::input::binding::MouseButton::Left,
            state: ButtonState::Pressed,
        });
        ctrl.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(200.0, 200.0),
        });
        ctrl.apply_to_camera(&mut cam);
        let after = cam.orientation();
        assert!(
            (after.x - orient_before.x).abs() < 1e-6
                && (after.y - orient_before.y).abs() < 1e-6
                && (after.z - orient_before.z).abs() < 1e-6
                && (after.w - orient_before.w).abs() < 1e-6,
            "planar mode should not change orientation"
        );
    }

    #[test]
    fn turntable_pitch_clamped() {
        let mut cam = default_target();
        // Try to apply extreme pitch
        for _ in 0..1000 {
            apply_turntable(&mut cam, 0.0, 0.1);
        }
        // Check eye direction Z component stays within sin(89 deg)
        let eye_z = (cam.orientation * glam::Vec3::Z).z;
        let max_sin = 89.0_f32.to_radians().sin();
        assert!(
            eye_z.abs() <= max_sin + 1e-4,
            "turntable pitch should be clamped: eye_z={eye_z}"
        );
    }

    #[test]
    fn fly_look_preserves_eye() {
        let mut cam = default_target();
        let eye_before = cam.eye();
        apply_fly_look(&mut cam, 0.3, 0.2);
        let eye_after = cam.eye();
        let diff = (eye_after - eye_before).length();
        assert!(
            diff < 1e-3,
            "firstperson look should preserve eye position, diff={diff}"
        );
    }

    #[test]
    fn fly_moves_camera() {
        let mut ctrl = OrbitCameraController::new(BindingPreset::Default);
        ctrl.navigation_mode = NavigationMode::Fly;
        // The deprecated path assumes 60 Hz: 60 units per second is 1 per frame.
        ctrl.fly_speed = 60.0;
        ctrl.begin_frame(make_ctx());
        let mut cam = camera_facing(glam::Quat::IDENTITY);
        let centre_before = cam.target();
        ctrl.push_event(ViewportEvent::Key {
            key: KeyCode::W,
            state: ButtonState::Pressed,
            repeat: false,
            logical: None,
        });
        ctrl.apply_to_camera(&mut cam);
        assert!(
            (cam.target() - centre_before).length() > 0.5,
            "FlyForward should move camera centre"
        );
    }

    #[test]
    fn apply_consumes_external_frame() {
        // The frame-consumer path applies an already-resolved ActionFrame
        // without draining events, the basis for sharing one resolver across
        // the controller family.
        let mut ctrl = OrbitCameraController::viewport_primitives();
        let mut cam = Camera::default();
        let before = cam.orientation();
        let mut frame = ActionFrame::default();
        frame.navigation.orbit = glam::Vec2::new(20.0, 0.0);
        ctrl.apply(&mut cam, &frame, 1.0 / 60.0);
        assert_ne!(
            cam.orientation(),
            before,
            "apply(&frame) should orbit from the frame's nav delta"
        );
    }

    #[test]
    fn pivot_follows_a_camera_moved_elsewhere() {
        let mut ctrl = OrbitCameraController::new_stateless();
        let mut cam = Camera::default();
        let mut frame = ActionFrame::default();
        frame.navigation.orbit = glam::Vec2::new(10.0, 0.0);
        ctrl.apply(&mut cam, &frame, 1.0 / 60.0);
        assert_eq!(ctrl.target().unwrap().centre, DVec3::ZERO);

        // A preset or fly-to moves the camera; the next drag orbits about the
        // new focus point at the new distance.
        let centre = DVec3::new(10.0, -3.0, 2.0);
        cam = Camera::orbit(centre, 4.0, cam.orientation());
        ctrl.apply(&mut cam, &frame, 1.0 / 60.0);
        assert!((cam.target() - centre).length() < 1e-4);
        assert_eq!(cam.focus_distance(), 4.0);
        assert_eq!(ctrl.target().unwrap().centre, centre);
    }

    #[test]
    fn no_input_leaves_the_camera_alone() {
        let mut ctrl = OrbitCameraController::new_stateless();
        let mut cam = Camera::default();
        cam.set_orthographic(true);
        let before = cam.clone();
        ctrl.apply(&mut cam, &ActionFrame::default(), 1.0 / 60.0);
        assert_eq!(cam, before);
    }

    #[test]
    fn zoom_scales_an_orthographic_view() {
        let mut ctrl = OrbitCameraController::new_stateless();
        let mut cam = Camera::default();
        cam.set_orthographic(true);
        let height = cam.view_height_at(1.0);
        let mut frame = ActionFrame::default();
        frame.navigation.zoom = 500.0;
        ctrl.apply(&mut cam, &frame, 1.0 / 60.0);
        let ratio = cam.focus_distance() / 5.0;
        assert!(ratio < 1.0);
        assert!((cam.view_height_at(1.0) - height * ratio).abs() < 1e-4);
    }

    fn fly_frame(actions: &[Action]) -> ActionFrame {
        let mut frame = ActionFrame::default();
        for &a in actions {
            frame
                .actions
                .insert(a, crate::input::action_frame::ResolvedActionState::Held);
        }
        frame
    }

    fn fly_controller() -> OrbitCameraController {
        let mut ctrl = OrbitCameraController::new_stateless();
        ctrl.navigation_mode = NavigationMode::Fly;
        ctrl.fly_speed = 1.0;
        ctrl
    }

    #[test]
    fn fly_diagonal_moves_as_far_as_straight() {
        let mut ctrl = fly_controller();
        let level = LookAngles::FORWARD.orientation();
        let mut moved = |actions: &[Action]| {
            let mut cam = camera_facing(level);
            let before = cam.target();
            ctrl.apply(&mut cam, &fly_frame(actions), 1.0);
            (cam.target() - before).length()
        };
        let straight = moved(&[Action::FlyForward]);
        let diagonal = moved(&[Action::FlyForward, Action::FlyRight]);
        let three = moved(&[Action::FlyForward, Action::FlyRight, Action::FlyUp]);
        assert!((straight - 1.0).abs() < 1e-5);
        assert!((diagonal - straight).abs() < 1e-5, "diagonal={diagonal}");
        assert!((three - straight).abs() < 1e-5, "three axes={three}");
    }

    #[test]
    fn fly_look_stops_short_of_vertical() {
        let mut cam = default_target();
        cam.orientation = LookAngles::FORWARD.orientation();
        // Drag up a long way: negative pitch delta looks up.
        for _ in 0..200 {
            apply_fly_look(&mut cam, 0.0, -0.1);
        }
        let look = LookAngles::from_orientation(cam.orientation);
        assert!(
            (look.pitch - POLE_LIMIT).abs() < 1e-3,
            "pitch={}",
            look.pitch
        );
        // No roll: the camera's right vector stays level.
        assert!((cam.orientation * glam::Vec3::X).z.abs() < 1e-4);
    }

    #[test]
    fn fly_look_matches_the_old_turn_directions() {
        let mut cam = default_target();
        cam.orientation = LookAngles::FORWARD.orientation();
        apply_fly_look(&mut cam, 0.2, 0.1);
        let look = LookAngles::from_orientation(cam.orientation);
        // Positive yaw turns right, a positive (downward) pitch delta looks down.
        assert!(look.yaw > 0.0 && look.pitch < 0.0, "{look:?}");
    }

    #[test]
    fn turntable_fast_step_reaches_the_limit() {
        let mut cam = default_target();
        cam.orientation = LookAngles::FORWARD.orientation();
        // One step far past the limit lands on it instead of being dropped.
        apply_turntable(&mut cam, 0.0, 2.0);
        let eye_elevation = (cam.orientation * glam::Vec3::Z).z.asin();
        assert!(
            (eye_elevation.abs() - POLE_LIMIT).abs() < 1e-3,
            "elevation={eye_elevation}"
        );
    }

    #[test]
    fn turntable_can_back_away_from_past_the_limit() {
        let mut cam = default_target();
        // Arcball left the eye past the limit, looking almost straight down.
        cam.orientation = LookAngles::new(0.0, -1.565).orientation();
        let before = (cam.orientation * glam::Vec3::Z).z;
        // A drag further over the top does nothing.
        apply_turntable(&mut cam, 0.0, 0.01);
        assert!(((cam.orientation * glam::Vec3::Z).z - before).abs() < 1e-6);
        // A drag back toward the horizon applies.
        apply_turntable(&mut cam, 0.0, -0.01);
        let after = (cam.orientation * glam::Vec3::Z).z;
        assert!(after < before, "before={before} after={after}");
    }

    #[test]
    fn fly_speed_is_per_second() {
        let mut ctrl = fly_controller();
        let mut moved = |dt: f32, frames: usize| {
            let mut cam = camera_facing(LookAngles::FORWARD.orientation());
            let before = cam.target();
            for _ in 0..frames {
                ctrl.apply(&mut cam, &fly_frame(&[Action::FlyForward]), dt);
            }
            (cam.target() - before).length()
        };
        // One second at 30, 60 and 144 Hz covers the same ground.
        for (dt, frames) in [(1.0 / 30.0, 30), (1.0 / 60.0, 60), (1.0 / 144.0, 144)] {
            assert!((moved(dt, frames) - 1.0).abs() < 1e-3, "dt={dt}");
        }
    }

    #[test]
    fn fly_boost_multiplies_speed() {
        let mut ctrl = fly_controller();
        let mut cam = camera_facing(LookAngles::FORWARD.orientation());
        let before = cam.target();
        ctrl.apply(
            &mut cam,
            &fly_frame(&[Action::FlyForward, Action::FlySpeedBoost]),
            1.0,
        );
        let moved = (cam.target() - before).length();
        assert!(
            (moved - f64::from(ctrl.fly_boost)).abs() < 1e-5,
            "moved={moved}"
        );
    }

    #[test]
    fn shift_resolves_to_fly_speed_boost() {
        use crate::input::binding::Modifiers;
        let mut input = ViewportInput::new(crate::input::viewport_default_bindings());
        input.begin_frame(make_ctx());
        input.push_event(ViewportEvent::ModifiersChanged(Modifiers::SHIFT));
        assert!(input.resolve().is_active(Action::FlySpeedBoost));
        input.push_event(ViewportEvent::ModifiersChanged(Modifiers::NONE));
        assert!(!input.resolve().is_active(Action::FlySpeedBoost));
    }
}
