//! Framework-agnostic viewport events for the new input pipeline.

use super::binding::{KeyCode, Modifiers, MouseButton};

/// Button press or release state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ButtonState {
    /// Button was pressed.
    Pressed,
    /// Button was released.
    Released,
}

/// Scroll delta units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScrollUnits {
    /// Delta in logical line units (one notch ≈ 1.0).
    Lines,
    /// Delta in physical pixels.
    Pixels,
}

/// A framework-agnostic event delivered to the viewport input pipeline.
///
/// Host applications translate their native windowing events into
/// `ViewportEvent` values and push them to [`super::controller::OrbitCameraController`]
/// (or [`super::viewport_input::ViewportInput`] for the lower-level path).
#[derive(Debug, Clone)]
pub enum ViewportEvent {
    /// The pointer moved to the given viewport-local position.
    PointerMoved {
        /// Viewport-local position in logical pixels, origin at top-left.
        position: glam::Vec2,
    },
    /// A mouse button was pressed or released.
    MouseButton {
        /// Which button changed state.
        button: MouseButton,
        /// New button state.
        state: ButtonState,
    },
    /// The scroll wheel moved.
    Wheel {
        /// Scroll delta. Positive Y = scroll up / zoom in (conventional).
        delta: glam::Vec2,
        /// Whether the delta is in lines or pixels.
        units: ScrollUnits,
    },
    /// A keyboard key changed state.
    Key {
        /// Which key changed state.
        key: KeyCode,
        /// New key state.
        state: ButtonState,
        /// True if the event is a key-repeat (key held down).
        repeat: bool,
    },
    /// Modifier key state changed.
    ModifiersChanged(Modifiers),
    /// The pointer left the viewport area.
    PointerLeft,
    /// The viewport lost keyboard focus.
    FocusLost,
    /// A character was typed (Unicode).
    ///
    /// Only push this event when the manipulation controller is active
    /// (`ManipulationController::is_active()`) to avoid swallowing other keypresses.
    /// The library filters the character stream to digits, `.`, and `-` before
    /// passing it to the numeric input buffer.
    Character(char),
}
