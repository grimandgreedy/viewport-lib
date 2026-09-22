//! Stateful viewport input accumulator and resolver.
//!
//! [`ViewportInput`] is the lower-level input resolver. Most consumers should
//! use [`crate::controllers::orbit::OrbitCameraController`] which wraps it.

use std::collections::HashSet;

use super::action::Action;
use super::action_frame::{ActionFrame, NavigationActions, PointerFrame, ResolvedActionState};
use super::binding::{KeyCode, Modifiers, MouseButton};
use super::context::ViewportContext;
use super::event::{ButtonState, ScrollUnits, ViewportEvent};
use super::preset::{BindingPreset, viewer_bindings, viewport_default_bindings};
use super::viewport_binding::{ViewportBinding, ViewportGesture};

/// Pixels-per-line conversion for scroll delta normalisation.
const PIXELS_PER_LINE: f32 = 28.0;

/// Maximum pointer displacement (in viewport pixels) between primary press and
/// release for the gesture to count as a click rather than a drag.
const CLICK_THRESHOLD_PX: f32 = 5.0;

/// Pinch-to-zoom conversion: wheel-equivalent pixels per unit of pinch delta.
///
/// The counterpart of [`PIXELS_PER_LINE`] for the trackpad. A pinch delta is a
/// log-scale magnification (see [`ViewportEvent::TrackpadPinch`]), so a full
/// spread of the fingers accumulates to roughly `1.0`. Feeding that through the
/// default zoom sensitivity gives about a halving of the camera distance, which
/// puts one full pinch at roughly one doubling of apparent size: the same ballpark
/// as the platform's own pinch-to-zoom, and close to what egui's `exp()` mapping
/// produces for its own widgets.
const PINCH_PIXELS_PER_UNIT: f32 = 500.0;

/// Stateful viewport input accumulator.
///
/// Maintains pointer and button state across frames and resolves raw
/// [`ViewportEvent`]s into semantic [`ActionFrame`] output.
///
/// # Frame lifecycle
///
/// ```text
/// // --- AppState construction ---
/// input.begin_frame(ctx);          // prime the accumulator
///
/// // --- Per winit window_event ---
/// input.push_event(translated_event);
///
/// // --- RedrawRequested ---
/// let actions = input.resolve();   // apply to camera / interactions
/// input.begin_frame(ctx);          // reset for next frame's events
/// ```
pub struct ViewportInput {
    bindings: Vec<ViewportBinding>,

    // Per-frame accumulated deltas
    drag_delta: glam::Vec2,
    wheel_delta: glam::Vec2, // always in pixels
    rotate_gesture: f32,     // accumulated two-finger rotation this frame, radians
    pinch_gesture: f32,      // accumulated two-finger pinch this frame, log-scale magnification
    pan_gesture: glam::Vec2, // accumulated trackpad pan this frame, logical pixels

    // Per-frame pointer/click state (reset by begin_frame). Unlike `drag_delta`,
    // `pointer_delta` accumulates every pointer move, not only while a button is held.
    pointer_delta: glam::Vec2,
    /// True on the frame the primary (left) button was pressed.
    left_drag_started: bool,
    /// True on the frame the primary button was released within the click threshold.
    left_clicked: bool,

    // Per-frame key accumulators (reset by begin_frame)
    keys_pressed: HashSet<KeyCode>,
    /// Characters typed this frame (reset by begin_frame, drained into ActionFrame).
    typed_chars: Vec<char>,

    // Persistent state
    pointer_pos: Option<glam::Vec2>,
    /// Which buttons are currently held. Tracks three buttons.
    button_held: [bool; 3], // [Left, Right, Middle]
    /// Position at which each button was first pressed (to detect in-viewport press).
    button_press_pos: [Option<glam::Vec2>; 3],
    modifiers: Modifiers,
    /// Keys currently held down (persistent across frames).
    keys_held: HashSet<KeyCode>,

    ctx: ViewportContext,
}

/// Index into the tracked-button arrays, or `None` for buttons the resolver does
/// not track for drag/hold (Back, Forward, Other). Those still reach the consumer as
/// raw events; they just do not drive orbit/pan/click gestures.
fn button_index(b: MouseButton) -> Option<usize> {
    match b {
        MouseButton::Left => Some(0),
        MouseButton::Right => Some(1),
        MouseButton::Middle => Some(2),
        _ => None,
    }
}

impl ViewportInput {
    /// Create a new resolver with the given binding list.
    pub fn new(bindings: Vec<ViewportBinding>) -> Self {
        Self {
            bindings,
            drag_delta: glam::Vec2::ZERO,
            wheel_delta: glam::Vec2::ZERO,
            rotate_gesture: 0.0,
            pinch_gesture: 0.0,
            pan_gesture: glam::Vec2::ZERO,
            pointer_delta: glam::Vec2::ZERO,
            left_drag_started: false,
            left_clicked: false,
            keys_pressed: HashSet::new(),
            typed_chars: Vec::new(),
            pointer_pos: None,
            button_held: [false; 3],
            button_press_pos: [None, None, None],
            modifiers: Modifiers::NONE,
            keys_held: HashSet::new(),
            ctx: ViewportContext::default(),
        }
    }

    /// Create a resolver for a named [`BindingPreset`].
    pub fn from_preset(preset: BindingPreset) -> Self {
        let bindings = match preset {
            BindingPreset::Default => viewport_default_bindings(),
            BindingPreset::Viewer => viewer_bindings(),
        };
        Self::new(bindings)
    }

    /// Whether a pointer gesture of ours is in flight: a button is held and its
    /// press landed while we were being fed events.
    ///
    /// A host uses this before taking the pointer for itself, so its own UI does not
    /// steal a drag (or, once touch lands, a pinch) half-way through. The reverse
    /// direction, telling the viewport an input is not its own, is simply not
    /// forwarding it: see
    /// [`forward_to_viewport`](crate::input::forward_to_viewport).
    pub fn is_gesture_active(&self) -> bool {
        self.button_held
            .iter()
            .zip(self.button_press_pos.iter())
            .any(|(held, press)| *held && press.is_some())
    }

    /// End any gesture in flight without completing it.
    ///
    /// Releases every held button and forgets where each was pressed, so a drag
    /// stops contributing and does not resume when the pointer next moves. Per-frame
    /// deltas already accumulated are cleared too, so the frame this is called on
    /// does not apply a partial drag.
    ///
    /// Call this when something outside the viewport invalidates the gesture and no
    /// release will arrive: the host taking the pointer for its own UI, a mode change,
    /// a tool switch, or a cancelled touch. Focus loss and the pointer leaving the
    /// window do this already, through
    /// [`ViewportEvent::FocusLost`](crate::input::ViewportEvent::FocusLost) and
    /// [`PointerLeft`](crate::input::ViewportEvent::PointerLeft).
    ///
    /// Keyboard state is left alone: a cancelled pointer gesture says nothing about
    /// which keys are held.
    pub fn cancel_gesture(&mut self) {
        for held in &mut self.button_held {
            *held = false;
        }
        for pos in &mut self.button_press_pos {
            *pos = None;
        }
        self.drag_delta = glam::Vec2::ZERO;
        self.pointer_delta = glam::Vec2::ZERO;
        self.left_drag_started = false;
        self.left_clicked = false;
    }

    /// Begin a new frame.
    ///
    /// Resets per-frame accumulators and records the current viewport context.
    /// Call this at the END of each render so it's ready to accumulate the next
    /// batch of events. Also call once during initialisation.
    pub fn begin_frame(&mut self, ctx: ViewportContext) {
        self.ctx = ctx;
        self.drag_delta = glam::Vec2::ZERO;
        self.wheel_delta = glam::Vec2::ZERO;
        self.rotate_gesture = 0.0;
        self.pinch_gesture = 0.0;
        self.pan_gesture = glam::Vec2::ZERO;
        self.pointer_delta = glam::Vec2::ZERO;
        self.left_drag_started = false;
        self.left_clicked = false;
        self.keys_pressed.clear();
        self.typed_chars.clear();
        // Note: persistent state (button_held, pointer_pos, modifiers, keys_held) is NOT reset.
    }

    /// Push a single viewport-scoped event into the accumulator.
    pub fn push_event(&mut self, event: ViewportEvent) {
        match event {
            ViewportEvent::PointerMoved { position } => {
                if let Some(prev) = self.pointer_pos {
                    // Only accumulate drag delta when at least one button is held
                    if self.button_held.iter().any(|&h| h) {
                        self.drag_delta += position - prev;
                    }
                    // Pointer delta tracks all movement this frame, button or not.
                    self.pointer_delta += position - prev;
                }
                self.pointer_pos = Some(position);
            }
            ViewportEvent::MouseButton { button, state } => {
                // Untracked buttons (Back/Forward/Other) have no drag/hold slot; they
                // pass through as raw events without driving gestures.
                if let Some(idx) = button_index(button) {
                    match state {
                        ButtonState::Pressed => {
                            self.button_held[idx] = true;
                            self.button_press_pos[idx] = self.pointer_pos;
                            if button == MouseButton::Left {
                                self.left_drag_started = true;
                            }
                        }
                        ButtonState::Released => {
                            if button == MouseButton::Left {
                                // Click if the pointer barely moved since the press.
                                let is_click = self.button_press_pos[idx]
                                    .zip(self.pointer_pos)
                                    .map(|(origin, cur)| {
                                        (cur - origin).length() < CLICK_THRESHOLD_PX
                                    })
                                    .unwrap_or(false);
                                if is_click {
                                    self.left_clicked = true;
                                }
                            }
                            self.button_held[idx] = false;
                            self.button_press_pos[idx] = None;
                        }
                    }
                }
            }
            ViewportEvent::Wheel { delta, units } => {
                let scale = match units {
                    ScrollUnits::Lines => PIXELS_PER_LINE,
                    ScrollUnits::Pixels => 1.0,
                    ScrollUnits::Pages => self.ctx.viewport_size[1].max(1.0),
                };
                // Only accumulate if viewport is hovered
                if self.ctx.hovered {
                    self.wheel_delta += delta * scale;
                }
            }
            ViewportEvent::ModifiersChanged(mods) => {
                self.modifiers = mods;
            }
            ViewportEvent::Key { key, state, repeat } => {
                // Only process key events when the viewport is focused
                if !self.ctx.focused {
                    return;
                }
                match state {
                    ButtonState::Pressed => {
                        if !repeat {
                            self.keys_pressed.insert(key);
                        }
                        self.keys_held.insert(key);
                    }
                    ButtonState::Released => {
                        self.keys_held.remove(&key);
                    }
                }
            }
            ViewportEvent::Character(c) => {
                // Only accept characters that are valid in a numeric expression.
                // The app is responsible for only pushing this event while a
                // manipulation session is active (see ViewportEvent::Character docs).
                if c.is_ascii_digit() || c == '.' || c == '-' {
                    self.typed_chars.push(c);
                }
            }
            ViewportEvent::PointerLeft => {
                self.pointer_pos = None;
                // Release all buttons on pointer leave to avoid stuck state
                for held in &mut self.button_held {
                    *held = false;
                }
                for pos in &mut self.button_press_pos {
                    *pos = None;
                }
            }
            ViewportEvent::FocusLost => {
                // Release all buttons and keys on focus loss
                for held in &mut self.button_held {
                    *held = false;
                }
                for pos in &mut self.button_press_pos {
                    *pos = None;
                }
                self.keys_held.clear();
                self.keys_pressed.clear();
            }
            ViewportEvent::TrackpadRotate(angle) => {
                if self.ctx.hovered {
                    self.rotate_gesture += angle;
                }
            }
            ViewportEvent::TrackpadPinch(delta) => {
                if self.ctx.hovered {
                    self.pinch_gesture += delta;
                }
            }
            ViewportEvent::TrackpadPan(delta) => {
                if self.ctx.hovered {
                    self.pan_gesture += delta;
                }
            }
            // ViewportEvent is non_exhaustive; ignore events this pipeline does not handle.
            _ => {}
        }
    }

    /// Resolve accumulated events into an [`ActionFrame`].
    ///
    /// This does NOT reset state : call [`begin_frame`](Self::begin_frame) for that.
    pub fn resolve(&self) -> ActionFrame {
        let mut orbit = glam::Vec2::ZERO;
        let mut pan = glam::Vec2::ZERO;
        let mut zoom = 0.0f32;
        let mut actions = std::collections::HashMap::new();

        // Skip pointer/wheel gesture evaluation if viewport is not hovered
        // (and no button is actively held from a press that started inside).
        let any_held_with_press = self
            .button_held
            .iter()
            .enumerate()
            .any(|(i, &held)| held && self.button_press_pos[i].is_some());
        let pointer_active = self.ctx.hovered || any_held_with_press;

        for binding in &self.bindings {
            match &binding.gesture {
                ViewportGesture::Drag { button, modifiers } => {
                    if !pointer_active {
                        continue;
                    }
                    let Some(idx) = button_index(*button) else {
                        continue;
                    };
                    let held = self.button_held[idx];
                    let press_started = self.button_press_pos[idx].is_some();
                    if held && press_started && modifiers.matches(self.modifiers) {
                        let delta = self.drag_delta;
                        match binding.action {
                            Action::Orbit => {
                                if orbit == glam::Vec2::ZERO {
                                    orbit += delta;
                                    actions
                                        .entry(binding.action)
                                        .or_insert(ResolvedActionState::Delta(delta));
                                }
                            }
                            Action::Pan => {
                                if pan == glam::Vec2::ZERO {
                                    pan += delta;
                                    actions
                                        .entry(binding.action)
                                        .or_insert(ResolvedActionState::Delta(delta));
                                }
                            }
                            Action::Zoom => {
                                if zoom == 0.0 {
                                    zoom += delta.y;
                                    actions
                                        .entry(binding.action)
                                        .or_insert(ResolvedActionState::Delta(delta));
                                }
                            }
                            _ => {
                                actions
                                    .entry(binding.action)
                                    .or_insert(ResolvedActionState::Delta(delta));
                            }
                        }
                    }
                }
                ViewportGesture::WheelY { modifiers } => {
                    if !pointer_active {
                        continue;
                    }
                    if modifiers.matches(self.modifiers) && self.wheel_delta.y != 0.0 {
                        let y = self.wheel_delta.y;
                        match binding.action {
                            Action::Zoom => zoom += y,
                            Action::Orbit => orbit.y += y,
                            Action::Pan => pan.y += y,
                            _ => {}
                        }
                        actions
                            .entry(binding.action)
                            .or_insert(ResolvedActionState::Delta(glam::Vec2::new(0.0, y)));
                    }
                }
                ViewportGesture::WheelXY { modifiers } => {
                    if !pointer_active {
                        continue;
                    }
                    if modifiers.matches(self.modifiers) && self.wheel_delta != glam::Vec2::ZERO {
                        let delta = self.wheel_delta;
                        match binding.action {
                            Action::Orbit => orbit += delta,
                            Action::Pan => pan += delta,
                            Action::Zoom => zoom += delta.y,
                            _ => {}
                        }
                        actions
                            .entry(binding.action)
                            .or_insert(ResolvedActionState::Delta(delta));
                    }
                }
                ViewportGesture::KeyPress { key, modifiers } => {
                    if self.keys_pressed.contains(key) && modifiers.matches(self.modifiers) {
                        actions
                            .entry(binding.action)
                            .or_insert(ResolvedActionState::Pressed);
                    }
                }
                ViewportGesture::KeyHold { key, modifiers } => {
                    if self.keys_held.contains(key) && modifiers.matches(self.modifiers) {
                        actions
                            .entry(binding.action)
                            .or_insert(ResolvedActionState::Held);
                    }
                }
            }
        }

        // ActionFrame and PointerFrame are non_exhaustive (built here, read by
        // consumers), so populate them through Default rather than a literal.
        let mut pointer = PointerFrame::default();
        pointer.cursor = self.pointer_pos;
        pointer.delta = self.pointer_delta;
        pointer.clicked = self.left_clicked;
        pointer.drag_started = self.left_drag_started;
        pointer.dragging =
            button_index(MouseButton::Left).is_some_and(|i| self.button_held[i]);

        // The trackpad gestures resolve straight into navigation rather than through the
        // binding table: they are already a named camera intent, and there is no button or
        // modifier to key a binding on. Same treatment as the rotation gesture.
        let mut frame = ActionFrame::default();
        frame.navigation = NavigationActions {
            orbit,
            pan: pan + self.pan_gesture,
            zoom: zoom + self.pinch_gesture * PINCH_PIXELS_PER_UNIT,
            twist: self.rotate_gesture,
        };
        frame.actions = actions;
        frame.typed_chars = self.typed_chars.clone();
        frame.pointer = pointer;
        frame
    }

    /// Current modifier state.
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::event::ButtonState;
    use crate::input::preset::viewport_default_bindings;

    fn focused_ctx() -> ViewportContext {
        ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [800.0, 600.0],
        }
    }

    /// Drag a held left button from `from` to `to`, returning the resolved frame.
    fn drag(input: &mut ViewportInput, from: glam::Vec2, to: glam::Vec2) -> ActionFrame {
        input.push_event(ViewportEvent::PointerMoved { position: from });
        input.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
        });
        input.push_event(ViewportEvent::PointerMoved { position: to });
        input.resolve()
    }

    #[test]
    fn a_gesture_is_active_only_between_press_and_release() {
        let mut input = ViewportInput::new(viewer_bindings());
        input.begin_frame(focused_ctx());
        assert!(!input.is_gesture_active(), "nothing held to begin with");

        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(10.0, 10.0),
        });
        assert!(!input.is_gesture_active(), "a bare move is not a gesture");

        input.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
        });
        assert!(input.is_gesture_active(), "the press starts one");

        input.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Released,
        });
        assert!(!input.is_gesture_active(), "the release ends it");
    }

    #[test]
    fn cancelling_stops_the_drag_on_the_frame_it_happens() {
        let mut input = ViewportInput::new(viewer_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(10.0, 10.0),
        });
        input.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
        });
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(40.0, 10.0),
        });

        input.cancel_gesture();
        assert!(!input.is_gesture_active());
        let frame = input.resolve();
        assert_eq!(
            frame.navigation.orbit,
            glam::Vec2::ZERO,
            "the partial drag must not be applied on the frame it was cancelled",
        );
    }

    #[test]
    fn a_cancelled_drag_does_not_resume_when_the_pointer_moves_again() {
        // The failure this guards: clearing the held flag but keeping the press position
        // (or vice versa) lets the next move re-enter the drag as though nothing happened.
        let mut input = ViewportInput::new(viewer_bindings());
        input.begin_frame(focused_ctx());
        drag(
            &mut input,
            glam::Vec2::new(10.0, 10.0),
            glam::Vec2::new(40.0, 10.0),
        );
        input.cancel_gesture();

        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(90.0, 10.0),
        });
        let frame = input.resolve();
        assert_eq!(
            frame.navigation.orbit,
            glam::Vec2::ZERO,
            "the button is no longer held, so moving must not orbit",
        );
        assert!(!input.is_gesture_active());
    }

    #[test]
    fn cancelling_leaves_keyboard_state_alone() {
        // A cancelled pointer gesture says nothing about which keys are held.
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Key {
            key: KeyCode::W,
            state: ButtonState::Pressed,
            repeat: false,
        });
        input.cancel_gesture();
        let frame = input.resolve();
        assert!(
            frame.is_active(Action::FlyForward),
            "W is still held after cancelling a pointer gesture",
        );
    }

    #[test]
    fn key_press_fires_once_then_clears() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Key {
            key: KeyCode::F,
            state: ButtonState::Pressed,
            repeat: false,
        });
        let frame = input.resolve();
        assert!(
            frame.is_active(Action::FocusObject),
            "FocusObject should be active on first frame"
        );

        // Second frame without a new press should not fire
        input.begin_frame(focused_ctx());
        let frame2 = input.resolve();
        assert!(
            !frame2.is_active(Action::FocusObject),
            "FocusObject should not be active on second frame"
        );
    }

    #[test]
    fn key_ignored_when_not_focused() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(ViewportContext {
            hovered: true,
            focused: false,
            viewport_size: [800.0, 600.0],
        });
        input.push_event(ViewportEvent::Key {
            key: KeyCode::F,
            state: ButtonState::Pressed,
            repeat: false,
        });
        let frame = input.resolve();
        assert!(
            !frame.is_active(Action::FocusObject),
            "key should be ignored without focus"
        );
    }

    #[test]
    fn resolve_no_events_is_zero() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        let frame = input.resolve();
        assert_eq!(frame.navigation.orbit, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.pan, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.zoom, 0.0);
        assert_eq!(frame.navigation.twist, 0.0);
        assert!(frame.actions.is_empty());
    }

    #[test]
    fn scroll_produces_zoom() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Wheel {
            delta: glam::Vec2::new(0.0, 3.0),
            units: ScrollUnits::Lines,
        });
        let frame = input.resolve();
        // Lines are scaled by PIXELS_PER_LINE (28.0), so zoom = 3 * 28 = 84
        assert!((frame.navigation.zoom - 84.0).abs() < 1e-3);
    }

    #[test]
    fn scroll_pixel_units_no_scaling() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Wheel {
            delta: glam::Vec2::new(0.0, 10.0),
            units: ScrollUnits::Pixels,
        });
        let frame = input.resolve();
        assert!((frame.navigation.zoom - 10.0).abs() < 1e-3);
    }

    #[test]
    fn scroll_ignored_when_not_hovered() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(ViewportContext {
            hovered: false,
            focused: true,
            viewport_size: [800.0, 600.0],
        });
        input.push_event(ViewportEvent::Wheel {
            delta: glam::Vec2::new(0.0, 5.0),
            units: ScrollUnits::Lines,
        });
        let frame = input.resolve();
        assert_eq!(frame.navigation.zoom, 0.0);
    }

    #[test]
    fn neither_left_nor_right_drag_touches_the_camera() {
        // Both belong to the application: selection, gizmo dragging, tools, context
        // menus. A camera binding on either gives that gesture two claimants.
        for button in [MouseButton::Left, MouseButton::Right] {
            let mut input = ViewportInput::new(viewport_default_bindings());
            input.begin_frame(focused_ctx());
            input.push_event(ViewportEvent::PointerMoved {
                position: glam::Vec2::new(100.0, 100.0),
            });
            input.push_event(ViewportEvent::MouseButton {
                button,
                state: ButtonState::Pressed,
            });
            input.push_event(ViewportEvent::PointerMoved {
                position: glam::Vec2::new(110.0, 105.0),
            });
            let frame = input.resolve();
            assert_eq!(frame.navigation.pan, glam::Vec2::ZERO, "{button:?} panned");
            assert_eq!(frame.navigation.orbit, glam::Vec2::ZERO, "{button:?} orbited");
        }
    }

    #[test]
    fn middle_drag_orbits_and_shift_middle_pans() {
        // The drag half of the default scheme, and the rule that shift means pan.
        let cases = [
            (Modifiers::NONE, true, false),
            (Modifiers::SHIFT, false, true),
        ];
        for (modifiers, want_orbit, want_pan) in cases {
            let mut input = ViewportInput::new(viewport_default_bindings());
            input.begin_frame(focused_ctx());
            input.push_event(ViewportEvent::ModifiersChanged(modifiers));
            input.push_event(ViewportEvent::PointerMoved {
                position: glam::Vec2::new(100.0, 100.0),
            });
            input.push_event(ViewportEvent::MouseButton {
                button: MouseButton::Middle,
                state: ButtonState::Pressed,
            });
            input.push_event(ViewportEvent::PointerMoved {
                position: glam::Vec2::new(110.0, 105.0),
            });
            let frame = input.resolve();
            assert_eq!(
                frame.navigation.orbit != glam::Vec2::ZERO,
                want_orbit,
                "orbit with {modifiers:?}"
            );
            assert_eq!(
                frame.navigation.pan != glam::Vec2::ZERO,
                want_pan,
                "pan with {modifiers:?}"
            );
        }
    }

    #[test]
    fn the_viewer_preset_adds_left_drag_orbit() {
        let mut input = ViewportInput::new(viewer_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(100.0, 100.0),
        });
        input.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
        });
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(110.0, 105.0),
        });
        assert_ne!(input.resolve().navigation.orbit, glam::Vec2::ZERO);
    }

    #[test]
    fn pointer_move_without_button_no_drag() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(100.0, 100.0),
        });
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(200.0, 200.0),
        });
        let frame = input.resolve();
        assert_eq!(frame.navigation.orbit, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.pan, glam::Vec2::ZERO);
    }

    #[test]
    fn begin_frame_resets_accumulators() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Wheel {
            delta: glam::Vec2::new(0.0, 5.0),
            units: ScrollUnits::Pixels,
        });
        // First resolve should have zoom
        let frame1 = input.resolve();
        assert!(frame1.navigation.zoom != 0.0);
        // begin_frame resets accumulators
        input.begin_frame(focused_ctx());
        let frame2 = input.resolve();
        assert_eq!(frame2.navigation.zoom, 0.0);
    }

    #[test]
    fn pointer_left_releases_buttons() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(100.0, 100.0),
        });
        input.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Right,
            state: ButtonState::Pressed,
        });
        input.push_event(ViewportEvent::PointerLeft);
        // Now move again and check no drag delta accumulates
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(200.0, 200.0),
        });
        let frame = input.resolve();
        assert_eq!(frame.navigation.pan, glam::Vec2::ZERO);
    }

    #[test]
    fn focus_lost_clears_keys() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Key {
            key: KeyCode::W,
            state: ButtonState::Pressed,
            repeat: false,
        });
        input.push_event(ViewportEvent::FocusLost);
        let frame = input.resolve();
        // FlyForward is bound to W hold; after FocusLost, keys_held is cleared
        assert!(
            !frame.is_active(Action::FlyForward),
            "FlyForward should not be active after focus lost"
        );
    }

    #[test]
    fn character_event_populates_typed_chars() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Character('3'));
        input.push_event(ViewportEvent::Character('.'));
        input.push_event(ViewportEvent::Character('5'));
        input.push_event(ViewportEvent::Character('a')); // filtered out
        let frame = input.resolve();
        assert_eq!(frame.typed_chars, vec!['3', '.', '5']);
    }

    #[test]
    fn trackpad_rotate_accumulates_twist() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::TrackpadRotate(0.1));
        input.push_event(ViewportEvent::TrackpadRotate(0.2));
        let frame = input.resolve();
        assert!((frame.navigation.twist - 0.3).abs() < 1e-5);
    }

    // Events the resolver does not model (drag-drop, raw motion, theme, occlusion,
    // extra mouse buttons) must pass through without perturbing the resolved navigation
    // or pointer state. They reach a consumer via the raw event stream instead.
    #[test]
    fn unmodelled_events_do_not_perturb_navigation() {
        use crate::input::event::Theme;
        use std::path::PathBuf;

        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::FileDropped(PathBuf::from("/x")));
        input.push_event(ViewportEvent::FileHovered(PathBuf::from("/x")));
        input.push_event(ViewportEvent::FileHoverCancelled);
        input.push_event(ViewportEvent::RawMotion {
            delta: glam::Vec2::new(9.0, 9.0),
        });
        input.push_event(ViewportEvent::ThemeChanged(Theme::Dark));
        input.push_event(ViewportEvent::Occluded(true));
        input.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Back,
            state: ButtonState::Pressed,
        });
        let frame = input.resolve();
        assert_eq!(frame.navigation.orbit, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.pan, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.zoom, 0.0);
        assert!(frame.navigation.twist.abs() < 1e-6);
        assert!(!frame.pointer.clicked);
        assert!(!frame.pointer.dragging);
    }

    #[test]
    fn a_trackpad_pinch_zooms_and_a_pan_pans() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::TrackpadPinch(0.1));
        input.push_event(ViewportEvent::TrackpadPan(glam::Vec2::new(5.0, -3.0)));
        let frame = input.resolve();
        assert!(
            frame.navigation.zoom > 0.0,
            "spreading the fingers zooms in, got {}",
            frame.navigation.zoom
        );
        assert_eq!(frame.navigation.pan, glam::Vec2::new(5.0, -3.0));
    }

    #[test]
    fn pinching_together_zooms_the_other_way() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::TrackpadPinch(-0.1));
        assert!(input.resolve().navigation.zoom < 0.0);
    }

    #[test]
    fn gestures_accumulate_within_a_frame_and_reset_between_them() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::TrackpadPinch(0.05));
        input.push_event(ViewportEvent::TrackpadPinch(0.05));
        let both = input.resolve().navigation.zoom;

        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::TrackpadPinch(0.1));
        let one = input.resolve().navigation.zoom;
        assert!(
            (both - one).abs() < 1e-3,
            "two half-pinches must equal one whole: {both} vs {one}"
        );

        input.begin_frame(focused_ctx());
        assert_eq!(
            input.resolve().navigation.zoom,
            0.0,
            "a new frame starts from nothing"
        );
    }

    #[test]
    fn trackpad_gestures_are_dropped_when_not_hovered() {
        // Same gate as the wheel: a gesture over someone else's pane is not ours.
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(ViewportContext {
            hovered: false,
            focused: false,
            viewport_size: [800.0, 600.0],
        });
        input.push_event(ViewportEvent::TrackpadPinch(0.5));
        input.push_event(ViewportEvent::TrackpadPan(glam::Vec2::new(5.0, 5.0)));
        input.push_event(ViewportEvent::TrackpadRotate(0.5));
        let frame = input.resolve();
        assert_eq!(frame.navigation.zoom, 0.0);
        assert_eq!(frame.navigation.pan, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.twist, 0.0);
    }

    #[test]
    fn key_hold_active_every_frame() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Key {
            key: KeyCode::W,
            state: ButtonState::Pressed,
            repeat: false,
        });
        let frame1 = input.resolve();
        assert!(frame1.is_active(Action::FlyForward));
        // Next frame: key is still held (no release event), so KeyHold should still fire
        input.begin_frame(focused_ctx());
        let frame2 = input.resolve();
        assert!(
            frame2.is_active(Action::FlyForward),
            "FlyForward should persist while key is held"
        );
    }

    #[test]
    fn key_release_stops_hold() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Key {
            key: KeyCode::W,
            state: ButtonState::Pressed,
            repeat: false,
        });
        let frame1 = input.resolve();
        assert!(frame1.is_active(Action::FlyForward));
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Key {
            key: KeyCode::W,
            state: ButtonState::Released,
            repeat: false,
        });
        let frame2 = input.resolve();
        assert!(
            !frame2.is_active(Action::FlyForward),
            "FlyForward should stop after key release"
        );
    }

    #[test]
    fn modifiers_changed_affects_bindings() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        // Press Shift modifier, then press X -> should fire ExcludeX (Shift+X)
        input.push_event(ViewportEvent::ModifiersChanged(Modifiers::SHIFT));
        input.push_event(ViewportEvent::Key {
            key: KeyCode::X,
            state: ButtonState::Pressed,
            repeat: false,
        });
        let frame = input.resolve();
        assert!(
            frame.is_active(Action::ExcludeX),
            "Shift+X should fire ExcludeX"
        );
    }

    #[test]
    fn repeat_key_does_not_fire_press() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Key {
            key: KeyCode::G,
            state: ButtonState::Pressed,
            repeat: true,
        });
        let frame = input.resolve();
        // BeginMove is a KeyPress binding; repeat should not trigger it
        assert!(
            !frame.is_active(Action::BeginMove),
            "repeat should not fire KeyPress bindings"
        );
    }

    #[test]
    fn middle_drag_shift_produces_pan() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(50.0, 50.0),
        });
        input.push_event(ViewportEvent::ModifiersChanged(Modifiers::SHIFT));
        input.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Middle,
            state: ButtonState::Pressed,
        });
        input.push_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(60.0, 55.0),
        });
        let frame = input.resolve();
        assert!((frame.navigation.pan.x - 10.0).abs() < 1e-3);
        assert!((frame.navigation.pan.y - 5.0).abs() < 1e-3);
    }
}
