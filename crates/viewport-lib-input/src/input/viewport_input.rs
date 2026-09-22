//! Stateful viewport input accumulator and resolver.
//!
//! [`ViewportInput`] is the lower-level input resolver. Most consumers should
//! use [`crate::controllers::orbit::OrbitCameraController`] which wraps it.

use std::collections::HashSet;

use super::action::Action;
use super::action_frame::{ActionFrame, NavigationActions, PointerFrame, ResolvedActionState};
use super::binding::{KeyCode, Modifiers, MouseButton};
use super::context::ViewportContext;
use super::event::{ButtonState, ScrollUnits, TouchId, TouchPhase, ViewportEvent};
use super::preset::{BindingPreset, viewer_bindings, viewport_default_bindings};
use super::touch::TouchSettings;
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

/// One touch contact the resolver is tracking.
#[derive(Debug, Clone, Copy)]
struct Contact {
    id: TouchId,
    /// Where the contact went down.
    origin: glam::Vec2,
    /// Where it is now.
    position: glam::Vec2,
    /// The clock reading when it went down. Zero unless the host sets a clock.
    start_time: f32,
    /// The furthest it has been from its origin, which is what disqualifies a tap or
    /// a long press. Distance from the origin alone would let a finger wander out and
    /// come back.
    travel: f32,
    /// A long press already fired for this contact, so lifting it is not a tap.
    long_pressed: bool,
}

/// What the contact set measures to: centroid, the distance between the two
/// contacts, and the angle of the line between them.
///
/// Gestures are the change in this between one touch event and the next, which is
/// why the count has to be unchanged for a delta to mean anything.
#[derive(Debug, Clone, Copy)]
struct TouchMeasure {
    centroid: glam::Vec2,
    spread: f32,
    angle: f32,
}

/// Wrap an angle difference into `-PI..PI`, so a twist across the atan2 branch cut
/// does not read as a full turn the other way.
fn wrap_angle(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (a + PI).rem_euclid(TAU) - PI
}

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

    // Per-frame touch accumulators (reset by begin_frame)
    /// Motion of the contact centroid, logical pixels.
    touch_drag: glam::Vec2,
    /// Change in contact spread, log scale, matching the trackpad pinch convention.
    touch_pinch: f32,
    /// Change in the angle between two contacts, radians.
    touch_twist: f32,
    tapped: bool,
    double_tapped: bool,
    long_pressed: bool,

    // Persistent state
    pointer_pos: Option<glam::Vec2>,
    /// Contacts currently down, in the order they landed.
    contacts: Vec<Contact>,
    /// What the contact set measured to when we last looked. `None` whenever a delta
    /// would be meaningless: no contacts, or more than the two a gesture reads.
    touch_ref: Option<TouchMeasure>,
    /// Where the last touch event landed. Outlives the contact so a tap can be
    /// hit-tested on the frame it is reported.
    touch_position: Option<glam::Vec2>,
    /// Origin and time of the last tap, for double-tap recognition.
    last_tap: Option<(glam::Vec2, f32)>,
    touch_settings: TouchSettings,
    /// The host's clock, in seconds, as of the last `begin_frame_at`.
    time: f32,
    /// Whether the host has ever given us a clock. The timed recognisers stay quiet
    /// until it has, rather than firing off a frozen zero.
    has_clock: bool,
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
            touch_drag: glam::Vec2::ZERO,
            touch_pinch: 0.0,
            touch_twist: 0.0,
            tapped: false,
            double_tapped: false,
            long_pressed: false,
            pointer_pos: None,
            contacts: Vec::new(),
            touch_ref: None,
            touch_position: None,
            last_tap: None,
            touch_settings: TouchSettings::default(),
            time: 0.0,
            has_clock: false,
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
            // BindingPreset is non_exhaustive; an unknown preset gets the default set.
            _ => viewport_default_bindings(),
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
        !self.contacts.is_empty()
            || self
                .button_held
                .iter()
                .zip(self.button_press_pos.iter())
                .any(|(held, press)| *held && press.is_some())
    }

    /// The numbers touch recognition is calibrated on.
    pub fn touch_settings(&self) -> TouchSettings {
        self.touch_settings
    }

    /// Replace the touch calibration. Takes effect on the next event; a gesture in
    /// flight keeps accumulating, it is only scaled differently from here.
    pub fn set_touch_settings(&mut self, settings: TouchSettings) {
        self.touch_settings = settings;
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
        // Contacts go with the buttons: a cancelled gesture is cancelled whichever
        // pointer was making it. The last tap is kept, so cancelling one gesture does
        // not silently break a double tap that had already landed its first half.
        self.contacts.clear();
        self.touch_ref = None;
        self.touch_drag = glam::Vec2::ZERO;
        self.touch_pinch = 0.0;
        self.touch_twist = 0.0;
        self.tapped = false;
        self.double_tapped = false;
        self.long_pressed = false;
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
        self.touch_drag = glam::Vec2::ZERO;
        self.touch_pinch = 0.0;
        self.touch_twist = 0.0;
        self.tapped = false;
        self.double_tapped = false;
        self.long_pressed = false;
        // Note: persistent state (button_held, pointer_pos, contacts, modifiers,
        // keys_held) is NOT reset.
    }

    /// Begin a new frame, and tell the resolver what time it is.
    ///
    /// `time_seconds` is your own elapsed-seconds clock, the same one you drive
    /// animation from; only the difference between readings matters, so any origin
    /// will do. The viewport never reads a clock of its own.
    ///
    /// Prefer this to [`begin_frame`](Self::begin_frame) on anything with a
    /// touchscreen: double tap and long press are the two recognisers that cannot be
    /// decided from positions alone, and they stay quiet until a clock arrives.
    pub fn begin_frame_at(&mut self, ctx: ViewportContext, time_seconds: f32) {
        self.has_clock = true;
        self.time = time_seconds;
        self.begin_frame(ctx);
        // After the per-frame clear, or it would be cleared away again. A long press
        // is the one gesture recognised by time passing rather than by an event, so
        // this is the only place it can be noticed.
        if self.contacts.len() == 1 {
            let dwell = self.touch_settings.long_press_dwell;
            let tolerance = self.touch_settings.long_press_tolerance;
            let contact = &mut self.contacts[0];
            if !contact.long_pressed
                && contact.travel <= tolerance
                && time_seconds - contact.start_time >= dwell
            {
                contact.long_pressed = true;
                self.long_pressed = true;
            }
        }
    }

    /// Track one touch contact.
    fn push_touch(&mut self, id: TouchId, phase: TouchPhase, position: glam::Vec2) {
        self.touch_position = Some(position);
        match phase {
            TouchPhase::Started => {
                // Gate the landing on hover, as the wheel is gated: a contact that
                // lands outside is not ours. Once it is tracked the gesture runs to
                // its end wherever the finger goes, like a drag whose press landed
                // inside.
                if !self.ctx.hovered || self.contacts.iter().any(|c| c.id == id) {
                    return;
                }
                self.contacts.push(Contact {
                    id,
                    origin: position,
                    position,
                    start_time: self.time,
                    travel: 0.0,
                    long_pressed: false,
                });
                self.rebaseline_touch();
            }
            TouchPhase::Moved => {
                let Some(contact) = self.contacts.iter_mut().find(|c| c.id == id) else {
                    return;
                };
                contact.position = position;
                contact.travel = contact.travel.max((position - contact.origin).length());
                self.accumulate_touch();
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                let Some(index) = self.contacts.iter().position(|c| c.id == id) else {
                    return;
                };
                let contact = self.contacts.remove(index);
                if matches!(phase, TouchPhase::Ended) {
                    self.finish_tap(&contact);
                }
                self.rebaseline_touch();
            }
        }
    }

    /// What the contact set measures to, or `None` when a delta from it would be
    /// meaningless: nothing down, or a third finger the gesture vocabulary has no
    /// meaning for. A third finger parks the gesture rather than distorting it.
    fn measure_touch(&self) -> Option<TouchMeasure> {
        match self.contacts.len() {
            1 => Some(TouchMeasure {
                centroid: self.contacts[0].position,
                spread: 0.0,
                angle: 0.0,
            }),
            2 => {
                let a = self.contacts[0].position;
                let b = self.contacts[1].position;
                let d = b - a;
                Some(TouchMeasure {
                    centroid: (a + b) * 0.5,
                    spread: d.length(),
                    angle: d.y.atan2(d.x),
                })
            }
            _ => None,
        }
    }

    /// Re-measure without accumulating, which is what a contact landing or lifting
    /// has to do: the centroid and the spread both jump when the set changes, and
    /// that jump is not a gesture. This is what makes the two-to-one-finger handoff
    /// continue smoothly instead of throwing the camera.
    fn rebaseline_touch(&mut self) {
        self.touch_ref = self.measure_touch();
    }

    /// Accumulate the change since the last measurement into this frame's gestures.
    fn accumulate_touch(&mut self) {
        let Some(now) = self.measure_touch() else {
            self.touch_ref = None;
            return;
        };
        if let Some(prev) = self.touch_ref {
            self.touch_drag += now.centroid - prev.centroid;
            // Two fingers almost on top of each other make the spread ratio and the
            // angle both wildly unstable, so leave them alone until there is a gesture
            // to measure.
            if prev.spread > 1.0 && now.spread > 1.0 {
                self.touch_pinch += (now.spread / prev.spread).ln();
                self.touch_twist += wrap_angle(now.angle - prev.angle);
            }
        }
        self.touch_ref = Some(now);
    }

    /// Decide whether a lifted contact was a tap, and whether it was the second half
    /// of a double one.
    fn finish_tap(&mut self, contact: &Contact) {
        if contact.long_pressed || contact.travel > self.touch_settings.tap_tolerance {
            return;
        }
        self.tapped = true;
        if self.has_clock
            && let Some((origin, at)) = self.last_tap
            && self.time - at <= self.touch_settings.double_tap_interval
            && (contact.origin - origin).length() <= self.touch_settings.double_tap_distance
        {
            self.double_tapped = true;
        }
        self.last_tap = Some((contact.origin, self.time));
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
                // Release all buttons, contacts and keys on focus loss
                for held in &mut self.button_held {
                    *held = false;
                }
                for pos in &mut self.button_press_pos {
                    *pos = None;
                }
                self.contacts.clear();
                self.touch_ref = None;
                self.keys_held.clear();
                self.keys_pressed.clear();
            }
            ViewportEvent::Touch {
                id,
                phase,
                position,
            } => self.push_touch(id, phase, position),
            // One source at a time. A host that sends contacts and also has the OS
            // recognise gestures from the same fingers would otherwise apply both,
            // which reads as a viewport twice as sensitive on that device alone.
            // Contacts win, because they are the ones the recogniser above is
            // calibrated for.
            ViewportEvent::TrackpadRotate(angle) => {
                if self.ctx.hovered && self.contacts.is_empty() {
                    self.rotate_gesture += angle;
                }
            }
            ViewportEvent::TrackpadPinch(delta) => {
                if self.ctx.hovered && self.contacts.is_empty() {
                    self.pinch_gesture += delta;
                }
            }
            ViewportEvent::TrackpadPan(delta) => {
                if self.ctx.hovered && self.contacts.is_empty() {
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
                ViewportGesture::TouchDrag { contacts } => {
                    if usize::from(*contacts) != self.contacts.len()
                        || self.touch_drag == glam::Vec2::ZERO
                    {
                        continue;
                    }
                    // The sensitivity that applies is the one named for what the
                    // gesture resolved to, not for how many fingers made it.
                    let delta = match binding.action {
                        Action::Orbit => self.touch_drag * self.touch_settings.orbit_sensitivity,
                        Action::Pan => self.touch_drag * self.touch_settings.pan_sensitivity,
                        _ => self.touch_drag,
                    };
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
                ViewportGesture::TouchPinch => {
                    if self.contacts.len() != 2 || self.touch_pinch == 0.0 {
                        continue;
                    }
                    // Converted through the same constant as the trackpad pinch, so
                    // the two feel alike and there is one number to calibrate.
                    let delta = self.touch_pinch
                        * self.touch_settings.pinch_sensitivity
                        * PINCH_PIXELS_PER_UNIT;
                    match binding.action {
                        Action::Zoom => zoom += delta,
                        Action::Orbit => orbit.y += delta,
                        Action::Pan => pan.y += delta,
                        _ => {}
                    }
                    actions
                        .entry(binding.action)
                        .or_insert(ResolvedActionState::Delta(glam::Vec2::new(0.0, delta)));
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
                // ViewportGesture is non_exhaustive; a gesture this resolver does not
                // model contributes nothing rather than failing to build.
                _ => {}
            }
        }

        // ActionFrame and PointerFrame are non_exhaustive (built here, read by
        // consumers), so populate them through Default rather than a literal.
        let mut pointer = PointerFrame::default();
        pointer.cursor = self.pointer_pos;
        pointer.delta = self.pointer_delta;
        pointer.clicked = self.left_clicked;
        pointer.drag_started = self.left_drag_started;
        pointer.dragging = button_index(MouseButton::Left).is_some_and(|i| self.button_held[i]);
        pointer.tapped = self.tapped;
        pointer.double_tapped = self.double_tapped;
        pointer.long_pressed = self.long_pressed;
        pointer.touch_position = self.touch_position;
        pointer.contacts = self.contacts.len() as u8;

        // The trackpad gestures resolve straight into navigation rather than through the
        // binding table: they are already a named camera intent, and there is no button or
        // modifier to key a binding on. Same treatment as the rotation gesture.
        // NavigationActions is non_exhaustive as well, so it is built through Default.
        let mut navigation = NavigationActions::default();
        navigation.orbit = orbit;
        navigation.pan = pan + self.pan_gesture;
        navigation.zoom = zoom + self.pinch_gesture * PINCH_PIXELS_PER_UNIT;
        // Twist has no Action to bind to, so both sources land here directly.
        navigation.twist =
            self.rotate_gesture + self.touch_twist * self.touch_settings.twist_sensitivity;

        let mut frame = ActionFrame::default();
        frame.navigation = navigation;
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
            assert_eq!(
                frame.navigation.orbit,
                glam::Vec2::ZERO,
                "{button:?} orbited"
            );
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
    /// Helpers for driving synthetic contact sequences.
    fn touch(input: &mut ViewportInput, id: u64, phase: TouchPhase, x: f32, y: f32) {
        input.push_event(ViewportEvent::Touch {
            id: TouchId(id),
            phase,
            position: glam::Vec2::new(x, y),
        });
    }

    #[test]
    fn one_finger_drags_the_camera_and_two_pan_it() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 1, TouchPhase::Moved, 130.0, 100.0);
        let frame = input.resolve();
        assert_eq!(frame.navigation.orbit, glam::Vec2::new(30.0, 0.0));
        assert_eq!(
            frame.navigation.pan,
            glam::Vec2::ZERO,
            "one finger does not pan"
        );
        assert_eq!(frame.pointer.contacts, 1);

        // A second finger changes what the same motion means.
        input.begin_frame(focused_ctx());
        touch(&mut input, 2, TouchPhase::Started, 200.0, 100.0);
        touch(&mut input, 1, TouchPhase::Moved, 150.0, 100.0);
        touch(&mut input, 2, TouchPhase::Moved, 220.0, 100.0);
        let frame = input.resolve();
        assert_eq!(frame.navigation.pan, glam::Vec2::new(20.0, 0.0));
        assert_eq!(
            frame.navigation.orbit,
            glam::Vec2::ZERO,
            "two fingers do not orbit"
        );
    }

    #[test]
    fn spreading_two_fingers_zooms_in_and_turning_them_twists() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 2, TouchPhase::Started, 200.0, 100.0);
        // Spread about the centroid: no pan, pure pinch.
        touch(&mut input, 1, TouchPhase::Moved, 50.0, 100.0);
        touch(&mut input, 2, TouchPhase::Moved, 250.0, 100.0);
        let frame = input.resolve();
        assert!(frame.navigation.zoom > 0.0, "got {}", frame.navigation.zoom);
        assert!(
            frame.navigation.pan.length() < 1e-4,
            "a symmetric spread does not pan"
        );

        // Turn the pair about the centroid: pure twist.
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Moved, 150.0, 0.0);
        touch(&mut input, 2, TouchPhase::Moved, 150.0, 200.0);
        let frame = input.resolve();
        assert!(
            frame.navigation.twist.abs() > 1.0,
            "got {}",
            frame.navigation.twist
        );
    }

    #[test]
    fn lifting_one_of_two_fingers_does_not_throw_the_camera() {
        // The handoff: the centroid jumps to the remaining finger when its partner
        // lifts, and that jump is not a gesture.
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 2, TouchPhase::Started, 300.0, 100.0);

        input.begin_frame(focused_ctx());
        touch(&mut input, 2, TouchPhase::Ended, 300.0, 100.0);
        let frame = input.resolve();
        assert_eq!(
            frame.navigation.orbit,
            glam::Vec2::ZERO,
            "the centroid moving 100px because a finger lifted is not an orbit",
        );

        // And the survivor still drives the camera from where it actually is.
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Moved, 110.0, 100.0);
        let frame = input.resolve();
        assert_eq!(frame.navigation.orbit, glam::Vec2::new(10.0, 0.0));
    }

    #[test]
    fn a_third_finger_parks_the_gesture() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 2, TouchPhase::Started, 200.0, 100.0);
        touch(&mut input, 3, TouchPhase::Started, 300.0, 100.0);

        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Moved, 150.0, 150.0);
        let frame = input.resolve();
        assert_eq!(frame.pointer.contacts, 3);
        assert_eq!(frame.navigation.orbit, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.pan, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.zoom, 0.0);
    }

    #[test]
    fn a_cancelled_contact_leaves_nothing_behind() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 1, TouchPhase::Moved, 105.0, 100.0);
        touch(&mut input, 1, TouchPhase::Cancelled, 105.0, 100.0);
        let frame = input.resolve();
        assert_eq!(frame.pointer.contacts, 0);
        assert!(!frame.pointer.tapped, "a cancelled contact is not a tap");
        assert!(!input.is_gesture_active());

        // A move for a contact we are no longer tracking is inert, not a resurrection.
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Moved, 400.0, 100.0);
        assert_eq!(input.resolve().navigation.orbit, glam::Vec2::ZERO);
    }

    #[test]
    fn a_short_contact_taps_and_a_long_travel_does_not() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 1, TouchPhase::Ended, 102.0, 100.0);
        let frame = input.resolve();
        assert!(frame.pointer.tapped);
        assert_eq!(
            frame.pointer.touch_position,
            Some(glam::Vec2::new(102.0, 100.0))
        );

        input.begin_frame(focused_ctx());
        touch(&mut input, 2, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 2, TouchPhase::Moved, 300.0, 100.0);
        touch(&mut input, 2, TouchPhase::Ended, 300.0, 100.0);
        assert!(!input.resolve().pointer.tapped, "a drag is not a tap");
    }

    #[test]
    fn a_second_tap_doubles_only_when_it_is_soon_and_close() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        let tap = |input: &mut ViewportInput, id, x: f32| {
            touch(input, id, TouchPhase::Started, x, 100.0);
            touch(input, id, TouchPhase::Ended, x, 100.0);
        };

        input.begin_frame_at(focused_ctx(), 0.0);
        tap(&mut input, 1, 100.0);
        assert!(
            !input.resolve().pointer.double_tapped,
            "the first tap is not a double"
        );

        input.begin_frame_at(focused_ctx(), 0.1);
        tap(&mut input, 2, 105.0);
        assert!(input.resolve().pointer.double_tapped);
        assert!(input.resolve().pointer.tapped, "a double tap is also a tap");

        // Too late.
        input.begin_frame_at(focused_ctx(), 5.0);
        tap(&mut input, 3, 105.0);
        assert!(!input.resolve().pointer.double_tapped);

        // Soon enough, but across the screen.
        input.begin_frame_at(focused_ctx(), 5.1);
        tap(&mut input, 4, 600.0);
        assert!(!input.resolve().pointer.double_tapped);
    }

    #[test]
    fn a_held_finger_long_presses_once_and_then_is_not_a_tap() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame_at(focused_ctx(), 0.0);
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        assert!(!input.resolve().pointer.long_pressed, "not yet");

        input.begin_frame_at(focused_ctx(), 1.0);
        assert!(input.resolve().pointer.long_pressed, "the dwell has passed");

        input.begin_frame_at(focused_ctx(), 1.1);
        assert!(!input.resolve().pointer.long_pressed, "it fires once");

        input.begin_frame_at(focused_ctx(), 1.2);
        touch(&mut input, 1, TouchPhase::Ended, 100.0, 100.0);
        assert!(
            !input.resolve().pointer.tapped,
            "lifting after a long press is not also a tap",
        );
    }

    #[test]
    fn a_long_press_needs_a_clock_and_a_still_finger() {
        // Without begin_frame_at there is no clock, so the dwell can never elapse.
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        for _ in 0..10 {
            input.begin_frame(focused_ctx());
            assert!(!input.resolve().pointer.long_pressed);
        }

        // With one, a finger that wandered is disqualified.
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame_at(focused_ctx(), 0.0);
        touch(&mut input, 2, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 2, TouchPhase::Moved, 200.0, 100.0);
        input.begin_frame_at(focused_ctx(), 2.0);
        assert!(!input.resolve().pointer.long_pressed);
    }

    #[test]
    fn contacts_win_over_the_hosts_own_gesture_events() {
        // A device that reports both would otherwise apply both and feel twice as
        // sensitive as the same gesture on any other device.
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 2, TouchPhase::Started, 200.0, 100.0);
        input.push_event(ViewportEvent::TrackpadPinch(0.5));
        input.push_event(ViewportEvent::TrackpadPan(glam::Vec2::new(30.0, 0.0)));
        input.push_event(ViewportEvent::TrackpadRotate(0.5));
        let frame = input.resolve();
        assert_eq!(frame.navigation.zoom, 0.0);
        assert_eq!(frame.navigation.pan, glam::Vec2::ZERO);
        assert_eq!(frame.navigation.twist, 0.0);

        // Once the fingers are up the trackpad is heard again.
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Ended, 100.0, 100.0);
        touch(&mut input, 2, TouchPhase::Ended, 200.0, 100.0);
        input.push_event(ViewportEvent::TrackpadPinch(0.5));
        assert!(input.resolve().navigation.zoom > 0.0);
    }

    #[test]
    fn a_contact_landing_outside_the_viewport_is_not_ours() {
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.begin_frame(ViewportContext {
            hovered: false,
            focused: false,
            viewport_size: [800.0, 600.0],
        });
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 1, TouchPhase::Moved, 200.0, 100.0);
        let frame = input.resolve();
        assert_eq!(frame.pointer.contacts, 0);
        assert_eq!(frame.navigation.orbit, glam::Vec2::ZERO);
    }

    #[test]
    fn touch_sensitivities_scale_and_invert() {
        let mut settings = TouchSettings::default();
        settings.orbit_sensitivity = glam::Vec2::new(2.0, -1.0);
        let mut input = ViewportInput::new(viewport_default_bindings());
        input.set_touch_settings(settings);
        input.begin_frame(focused_ctx());
        touch(&mut input, 1, TouchPhase::Started, 100.0, 100.0);
        touch(&mut input, 1, TouchPhase::Moved, 110.0, 110.0);
        let frame = input.resolve();
        assert_eq!(frame.navigation.orbit, glam::Vec2::new(20.0, -10.0));
    }

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
