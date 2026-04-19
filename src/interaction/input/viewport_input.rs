//! Stateful viewport input accumulator and resolver.
//!
//! [`ViewportInput`] is the lower-level input resolver. Most consumers should
//! use [`super::controller::OrbitCameraController`] which wraps it.

use super::action_frame::{ActionFrame, NavigationActions};
use super::binding::{Modifiers, MouseButton};
use super::context::ViewportContext;
use super::event::{ButtonState, ScrollUnits, ViewportEvent};
use super::preset::{BindingPreset, viewport_primitives_bindings};
use super::viewport_binding::{ViewportAction, ViewportBinding, ViewportGesture};

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

    // Persistent state
    pointer_pos: Option<glam::Vec2>,
    /// Which buttons are currently held. Tracks three buttons.
    button_held: [bool; 3], // [Left, Right, Middle]
    /// Position at which each button was first pressed (to detect in-viewport press).
    button_press_pos: [Option<glam::Vec2>; 3],
    modifiers: Modifiers,

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
            pointer_pos: None,
            button_held: [false; 3],
            button_press_pos: [None, None, None],
            modifiers: Modifiers::NONE,
            ctx: ViewportContext::default(),
        }
    }

    /// Create a resolver for a named [`BindingPreset`].
    pub fn from_preset(preset: BindingPreset) -> Self {
        let bindings = match preset {
            BindingPreset::ViewportPrimitives => viewport_primitives_bindings(),
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
        // Note: persistent state (button_held, pointer_pos, modifiers) is NOT reset.
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
                // Release all buttons on focus loss
                for held in &mut self.button_held {
                    *held = false;
                }
                for pos in &mut self.button_press_pos {
                    *pos = None;
                }
            }
            ViewportEvent::Key { .. } => {
                // Key events not used for camera navigation in this resolver
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

        // Skip all gesture evaluation if viewport is not hovered (and no button is actively held)
        // Buttons that started in the viewport are still allowed to complete their drag.
        let any_held_with_press = self.button_held.iter().enumerate().any(|(i, &held)| {
            held && self.button_press_pos[i].is_some()
        });
        let active = self.ctx.hovered || any_held_with_press;
        if !active {
            return ActionFrame::default();
        }

        for binding in &self.bindings {
            match &binding.gesture {
                ViewportGesture::Drag { button, modifiers } => {
                    let idx = button_index(*button);
                    let held = self.button_held[idx];
                    let press_started = self.button_press_pos[idx].is_some();
                    if held && press_started && modifiers.matches(self.modifiers) {
                        // First matching drag binding wins for the accumulated delta
                        match binding.action {
                            ViewportAction::Orbit => {
                                if orbit == glam::Vec2::ZERO {
                                    orbit += self.drag_delta;
                                }
                            }
                            ViewportAction::Pan => {
                                if pan == glam::Vec2::ZERO {
                                    pan += self.drag_delta;
                                }
                            }
                            ViewportAction::Zoom => {
                                if zoom == 0.0 {
                                    zoom += self.drag_delta.y;
                                }
                            }
                        }
                        break; // first matching drag wins
                    }
                }
                ViewportGesture::WheelY { modifiers } => {
                    if modifiers.matches(self.modifiers) && self.wheel_delta.y != 0.0 {
                        match binding.action {
                            ViewportAction::Zoom => {
                                zoom += self.wheel_delta.y;
                            }
                            ViewportAction::Orbit => {
                                orbit.y += self.wheel_delta.y;
                            }
                            ViewportAction::Pan => {
                                pan.y += self.wheel_delta.y;
                            }
                        }
                    }
                }
                ViewportGesture::WheelXY { modifiers } => {
                    if modifiers.matches(self.modifiers)
                        && self.wheel_delta != glam::Vec2::ZERO
                    {
                        match binding.action {
                            ViewportAction::Orbit => {
                                orbit += self.wheel_delta;
                            }
                            ViewportAction::Pan => {
                                pan += self.wheel_delta;
                            }
                            ViewportAction::Zoom => {
                                zoom += self.wheel_delta.y;
                            }
                        }
                    }
                }
            }
        }

        ActionFrame {
            navigation: NavigationActions { orbit, pan, zoom },
        }
    }

    /// Current modifier state.
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }
}
