//! Viewport events for the new input pipeline.

use std::path::PathBuf;

use super::binding::{KeyCode, Modifiers, MouseButton};

/// Button press or release state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ButtonState {
    /// Button was pressed.
    Pressed,
    /// Button was released.
    Released,
}

/// OS colour theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Theme {
    /// Light theme.
    Light,
    /// Dark theme.
    Dark,
}

/// Scroll delta units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScrollUnits {
    /// Delta in logical line units (one notch ~ 1.0).
    Lines,
    /// Delta in physical pixels.
    Pixels,
    /// Delta in viewport pages (one unit = viewport height).
    Pages,
}

/// An event delivered to the viewport input pipeline.
///
/// Host applications translate their native windowing events into
/// `ViewportEvent` values and push them to the `OrbitCameraController`
/// (or `ViewportInput` for direct input handling).
#[non_exhaustive]
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

    /// Two-finger trackpad rotation gesture.
    ///
    /// `delta` is the change in angle this event, in radians.
    /// Positive = counter-clockwise (matches winit's `RotationGesture` convention,
    /// converted from degrees to radians by the host).
    ///
    /// ## Platform-specific
    /// Only emitted on macOS (and iOS). Silently unused on Windows and Linux.
    TrackpadRotate(f32),

    /// Two-finger trackpad pinch (magnify) gesture.
    ///
    /// `delta` is the change in scale this event (winit's `PinchGesture` delta):
    /// positive = pinch out / zoom in. Pass-through by default; a consumer maps it to
    /// camera zoom if desired.
    ///
    /// ## Platform-specific
    /// Only emitted on macOS (and iOS). Silently unused on Windows and Linux.
    TrackpadPinch(f32),

    /// Two-finger trackpad pan gesture.
    ///
    /// `delta` is the pan this event in logical points. Pass-through by default.
    ///
    /// ## Platform-specific
    /// Only emitted on macOS (and iOS). Silently unused on Windows and Linux.
    TrackpadPan(glam::Vec2),

    /// Raw, unaccelerated relative pointer motion from the input device, not tied to
    /// the window or surface. `delta` is in raw device units. Use this for
    /// first-person / mouselook navigation while the cursor is grabbed; the ordinary
    /// cursor position comes from [`PointerMoved`](ViewportEvent::PointerMoved).
    RawMotion {
        /// Relative motion since the last event, in raw device units.
        delta: glam::Vec2,
    },

    /// The OS colour theme changed. A consumer can follow the system light/dark
    /// preference (for overlay UI colours, for example).
    ThemeChanged(Theme),

    /// The window's occlusion state changed. `true` when the window became fully
    /// hidden (behind others, or minimised); `false` when it is visible again. A
    /// consumer can pause rendering while occluded to save power.
    Occluded(bool),

    /// A file was dropped onto the window, at the OS level (not viewport-local).
    FileDropped(PathBuf),

    /// A file is being dragged over the window but not yet dropped. May arrive more
    /// than once as the drag moves; a consumer uses it to show a drop target.
    FileHovered(PathBuf),

    /// A file drag left the window without dropping, cancelling a prior
    /// [`FileHovered`](ViewportEvent::FileHovered).
    FileHoverCancelled,
}

impl ViewportEvent {
    /// Whether this event would be claimed by both the viewport and anything drawn
    /// over it, so at most one of them may act on it.
    ///
    /// A left press, its release, and the pointer motion between them are contested:
    /// a press-and-move is exactly what a camera orbit, a gizmo drag and a UI drag
    /// would all claim, so it has to belong to one of them. Everything else is
    /// **ambient**: the wheel, the other buttons and the modifiers can be acted on by
    /// more than one claimant in the same frame.
    ///
    /// That asymmetry is deliberate. Withholding the whole event stream whenever
    /// something over the viewport holds the pointer is the obvious implementation
    /// and it fails in the most visible place: the moment the cursor crosses a gizmo
    /// handle, scrolling stops zooming, which is exactly where someone is most likely
    /// to scroll.
    ///
    /// The set is fixed and does not consult the active bindings. Right and middle
    /// drag stay ambient even though the default scheme binds them to camera
    /// navigation, because chrome overwhelmingly uses the primary button, and
    /// contesting the others would break the commoner case of a camera drag
    /// interrupted by the cursor passing over a readout. An application whose own
    /// chrome binds right or middle drag withholds those itself before forwarding.
    ///
    /// Used by [`forward_to_viewport`](super::context::forward_to_viewport), which is
    /// what most callers want.
    pub fn is_contested(&self) -> bool {
        matches!(
            self,
            ViewportEvent::MouseButton {
                button: MouseButton::Left,
                ..
            } | ViewportEvent::PointerMoved { .. }
        )
    }
}

#[cfg(test)]
mod contested_tests {
    use super::*;

    #[test]
    fn only_the_left_button_and_pointer_motion_are_contested() {
        assert!(
            ViewportEvent::PointerMoved {
                position: glam::Vec2::ZERO
            }
            .is_contested()
        );
        for state in [ButtonState::Pressed, ButtonState::Released] {
            assert!(
                ViewportEvent::MouseButton {
                    button: MouseButton::Left,
                    state
                }
                .is_contested()
            );
        }

        // Ambient: acted on by the camera even while something over it holds the pointer.
        for button in [MouseButton::Right, MouseButton::Middle] {
            assert!(
                !ViewportEvent::MouseButton {
                    button,
                    state: ButtonState::Pressed
                }
                .is_contested(),
                "{button:?} must stay ambient"
            );
        }
        assert!(
            !ViewportEvent::Wheel {
                delta: glam::Vec2::Y,
                units: ScrollUnits::Lines
            }
            .is_contested()
        );
        assert!(!ViewportEvent::ModifiersChanged(Modifiers::NONE).is_contested());
        assert!(!ViewportEvent::PointerLeft.is_contested());
    }
}
