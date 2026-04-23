//! Stateful viewport input accumulator and resolver.
//!
//! [`ViewportInput`] is the lower-level input resolver. Most consumers should
//! use [`super::controller::OrbitCameraController`] which wraps it.

use std::collections::HashSet;

use super::action::Action;
use super::action_frame::{ActionFrame, NavigationActions, ResolvedActionState};
use super::binding::{KeyCode, Modifiers, MouseButton};
use super::context::ViewportContext;
use super::event::{ButtonState, ScrollUnits, ViewportEvent};
use super::preset::{BindingPreset, viewport_all_bindings, viewport_primitives_bindings};
use super::viewport_binding::{ViewportBinding, ViewportGesture};

/// Pixels-per-line conversion for scroll delta normalisation.
const PIXELS_PER_LINE: f32 = 28.0;

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

fn button_index(b: MouseButton) -> usize {
    match b {
        MouseButton::Left => 0,
        MouseButton::Right => 1,
        MouseButton::Middle => 2,
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
            BindingPreset::ViewportPrimitives => viewport_primitives_bindings(),
            BindingPreset::ViewportAll => viewport_all_bindings(),
        };
        Self::new(bindings)
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
                }
                self.pointer_pos = Some(position);
            }
            ViewportEvent::MouseButton { button, state } => {
                let idx = button_index(button);
                match state {
                    ButtonState::Pressed => {
                        self.button_held[idx] = true;
                        self.button_press_pos[idx] = self.pointer_pos;
                    }
                    ButtonState::Released => {
                        self.button_held[idx] = false;
                        self.button_press_pos[idx] = None;
                    }
                }
            }
            ViewportEvent::Wheel { delta, units } => {
                let scale = match units {
                    ScrollUnits::Lines => PIXELS_PER_LINE,
                    ScrollUnits::Pixels => 1.0,
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
        }
    }

    /// Resolve accumulated events into an [`ActionFrame`].
    ///
    /// This does NOT reset state — call [`begin_frame`](Self::begin_frame) for that.
    pub fn resolve(&self) -> ActionFrame {
        let mut orbit = glam::Vec2::ZERO;
        let mut pan = glam::Vec2::ZERO;
        let mut zoom = 0.0f32;
        let mut actions = std::collections::HashMap::new();

        // Skip pointer/wheel gesture evaluation if viewport is not hovered
        // (and no button is actively held from a press that started inside).
        let any_held_with_press = self.button_held.iter().enumerate().any(|(i, &held)| {
            held && self.button_press_pos[i].is_some()
        });
        let pointer_active = self.ctx.hovered || any_held_with_press;

        for binding in &self.bindings {
            match &binding.gesture {
                ViewportGesture::Drag { button, modifiers } => {
                    if !pointer_active {
                        continue;
                    }
                    let idx = button_index(*button);
                    let held = self.button_held[idx];
                    let press_started = self.button_press_pos[idx].is_some();
                    if held && press_started && modifiers.matches(self.modifiers) {
                        let delta = self.drag_delta;
                        match binding.action {
                            Action::Orbit => {
                                if orbit == glam::Vec2::ZERO {
                                    orbit += delta;
                                    actions.entry(binding.action).or_insert(
                                        ResolvedActionState::Delta(delta),
                                    );
                                }
                            }
                            Action::Pan => {
                                if pan == glam::Vec2::ZERO {
                                    pan += delta;
                                    actions.entry(binding.action).or_insert(
                                        ResolvedActionState::Delta(delta),
                                    );
                                }
                            }
                            Action::Zoom => {
                                if zoom == 0.0 {
                                    zoom += delta.y;
                                    actions.entry(binding.action).or_insert(
                                        ResolvedActionState::Delta(delta),
                                    );
                                }
                            }
                            _ => {
                                actions.entry(binding.action).or_insert(
                                    ResolvedActionState::Delta(delta),
                                );
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
                        actions.entry(binding.action).or_insert(
                            ResolvedActionState::Delta(glam::Vec2::new(0.0, y)),
                        );
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
                        actions.entry(binding.action).or_insert(
                            ResolvedActionState::Delta(delta),
                        );
                    }
                }
                ViewportGesture::KeyPress { key, modifiers } => {
                    if self.keys_pressed.contains(key) && modifiers.matches(self.modifiers) {
                        actions.entry(binding.action).or_insert(ResolvedActionState::Pressed);
                    }
                }
                ViewportGesture::KeyHold { key, modifiers } => {
                    if self.keys_held.contains(key) && modifiers.matches(self.modifiers) {
                        actions.entry(binding.action).or_insert(ResolvedActionState::Held);
                    }
                }
            }
        }

        ActionFrame {
            navigation: NavigationActions { orbit, pan, zoom, twist: self.rotate_gesture },
            actions,
            typed_chars: self.typed_chars.clone(),
        }
    }

    /// Current modifier state.
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::input::preset::viewport_all_bindings;
    use crate::interaction::input::event::ButtonState;

    fn focused_ctx() -> ViewportContext {
        ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [800.0, 600.0],
        }
    }

    #[test]
    fn key_press_fires_once_then_clears() {
        let mut input = ViewportInput::new(viewport_all_bindings());
        input.begin_frame(focused_ctx());
        input.push_event(ViewportEvent::Key {
            key: KeyCode::F,
            state: ButtonState::Pressed,
            repeat: false,
        });
        let frame = input.resolve();
        assert!(frame.is_active(Action::FocusObject), "FocusObject should be active on first frame");

        // Second frame without a new press should not fire
        input.begin_frame(focused_ctx());
        let frame2 = input.resolve();
        assert!(!frame2.is_active(Action::FocusObject), "FocusObject should not be active on second frame");
    }

    #[test]
    fn key_ignored_when_not_focused() {
        let mut input = ViewportInput::new(viewport_all_bindings());
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
        assert!(!frame.is_active(Action::FocusObject), "key should be ignored without focus");
    }
}
