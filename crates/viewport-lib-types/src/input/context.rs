//! Per-frame viewport context for the new input pipeline.

use super::event::ViewportEvent;

/// Who owns the pointer for this frame, as the application's own routing decided.
///
/// The application hit-tests its widgets, resolves its z-order, and honours any
/// gesture already in flight; the result is one of these. The viewport never makes
/// this decision (it cannot see the application's widgets), it only acts on it.
///
/// Two questions hide behind the one word "hovered", and this separates them:
/// whether the pointer is *geometrically* inside the viewport, and whether the
/// viewport *won* the input. Chrome drawn over a viewport makes the second false
/// while the first stays true, which is the [`Inside`](Self::Inside) case and the
/// one most easily missed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PointerOwnership {
    /// The pointer is outside the viewport and no gesture of ours is in flight.
    #[default]
    Elsewhere,
    /// The pointer is inside the viewport's rect, but something drawn over it owns
    /// the input this frame: a tool button, a gizmo handle, an open menu.
    ///
    /// Ambient events still reach the viewport, so the wheel keeps zooming while the
    /// cursor is over a tool handle. Contested ones do not, so a press on that handle
    /// is not also a press on the scene behind it.
    Inside,
    /// The viewport owns the input this frame; everything reaches it.
    Owned,
    /// The pointer is grabbed by the viewport, as for mouselook. Everything reaches
    /// it, hover has no meaning, and [`ViewportEvent::PointerLeft`] is suppressed:
    /// a grabbed pointer has no rect to leave, and the event releases every held
    /// button, which would end the session.
    Grabbed,
}

impl PointerOwnership {
    /// Whether the pointer is geometrically inside the viewport. True for every
    /// state except [`Elsewhere`](Self::Elsewhere). Gates the wheel and the
    /// trackpad gestures.
    pub fn pointer_inside(self) -> bool {
        !matches!(self, PointerOwnership::Elsewhere)
    }

    /// Whether the viewport won the input this frame. True for
    /// [`Owned`](Self::Owned) and [`Grabbed`](Self::Grabbed). Gates contested
    /// events (see [`ViewportEvent::is_contested`]).
    pub fn owns_input(self) -> bool {
        matches!(self, PointerOwnership::Owned | PointerOwnership::Grabbed)
    }
}

/// Whether to forward `event` to the viewport, given who owns the pointer this frame.
///
/// The pointer half of the routing rule. Contested events (see
/// [`ViewportEvent::is_contested`]) reach the viewport only when it owns the input;
/// everything else is ambient and reaches it regardless, so a camera keeps zooming
/// while a tool over the viewport holds the pointer.
///
/// `ViewportInput::push_event` applies this rule itself, with the ownership in
/// the [`ViewportContext`] it was given, so a host with one viewport pushes every
/// event and does not call this. It is for a host that routes events between
/// several resolvers itself.
///
/// Teardown always lands. A left release, and a touch contact ending or being
/// cancelled, are forwarded whoever owns the input, so a press or contact the
/// viewport was already tracking cannot be stranded down by ownership changing
/// mid-gesture. A release or end for something the viewport never saw start is a
/// no-op, so this cannot make it act on a gesture that was never its own.
/// [`ViewportEvent::PointerLeft`] is dropped under [`PointerOwnership::Grabbed`]
/// (see that variant).
///
/// ```
/// # use viewport_lib_types::input::{forward_to_viewport, PointerOwnership, ViewportEvent, ScrollUnits};
/// let wheel = ViewportEvent::Wheel { delta: glam::Vec2::Y, units: ScrollUnits::Lines };
/// // A tool drawn over the viewport holds the pointer, but the wheel is still the camera's.
/// assert!(forward_to_viewport(&wheel, PointerOwnership::Inside));
/// ```
pub fn forward_to_viewport(event: &ViewportEvent, ownership: PointerOwnership) -> bool {
    match event {
        ViewportEvent::PointerLeft => !matches!(ownership, PointerOwnership::Grabbed),
        _ if event.is_contested() => ownership.owns_input(),
        _ => true,
    }
}

/// Whether the viewport owns the keyboard this frame.
///
/// Separate from [`PointerOwnership`] because the two answers differ: a text
/// field in a side panel can hold the keyboard while the pointer is over the
/// viewport. Keys, typed text and held-modifier bindings reach the viewport only
/// under [`Owned`](Self::Owned). Modifier state itself is always tracked, so a
/// shift-drag still pans while a text field has the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KeyboardOwnership {
    /// Something else has the keyboard: a text field, another viewport, another
    /// window.
    #[default]
    Elsewhere,
    /// The viewport has the keyboard.
    Owned,
}

/// Who owns the pointer and who owns the keyboard, as the application's routing
/// decided this frame.
///
/// The per-framework helpers (`adapters::egui::ownership`,
/// `adapters::winit::ownership` in `viewport-lib-input`) compute it from
/// framework state; a host that hit-tests its own widgets builds it directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct InputOwnership {
    /// Who owns the pointer.
    pub pointer: PointerOwnership,
    /// Who owns the keyboard.
    pub keyboard: KeyboardOwnership,
}

impl InputOwnership {
    /// Neither the pointer nor the keyboard is the viewport's.
    pub const NONE: Self = Self {
        pointer: PointerOwnership::Elsewhere,
        keyboard: KeyboardOwnership::Elsewhere,
    };

    /// A viewport that fills its window with nothing drawn over it: it owns both.
    pub const fn full_window() -> Self {
        Self {
            pointer: PointerOwnership::Owned,
            keyboard: KeyboardOwnership::Owned,
        }
    }

    /// Pointer ownership `pointer`, with the keyboard following it: owned when
    /// the pointer is [`Owned`](PointerOwnership::Owned) or
    /// [`Grabbed`](PointerOwnership::Grabbed). For a host with no text fields of
    /// its own; one that has them sets the keyboard axis from their focus.
    pub const fn pointer_only(pointer: PointerOwnership) -> Self {
        let keyboard = match pointer {
            PointerOwnership::Owned | PointerOwnership::Grabbed => KeyboardOwnership::Owned,
            _ => KeyboardOwnership::Elsewhere,
        };
        Self { pointer, keyboard }
    }
}

/// Per-frame viewport context passed to `OrbitCameraController::begin_frame`
/// (and `ViewportInput::begin_frame`): who owns the input, and the viewport size.
#[derive(Debug, Clone, Copy)]
pub struct ViewportContext {
    /// Who owns the pointer and the keyboard this frame.
    pub ownership: InputOwnership,
    /// Viewport size in logical pixels `[width, height]`.
    pub viewport_size: [f32; 2],
}

impl ViewportContext {
    /// A context from the application's routing decision and the viewport size.
    pub const fn new(ownership: InputOwnership, viewport_size: [f32; 2]) -> Self {
        Self {
            ownership,
            viewport_size,
        }
    }

    /// A context from pointer ownership alone, the keyboard following the pointer.
    #[deprecated(
        note = "use ViewportContext::new(InputOwnership::pointer_only(ownership), size), or set the keyboard axis from your text focus"
    )]
    pub const fn with_ownership(ownership: PointerOwnership, viewport_size: [f32; 2]) -> Self {
        Self::new(InputOwnership::pointer_only(ownership), viewport_size)
    }
}

impl Default for ViewportContext {
    fn default() -> Self {
        Self::new(InputOwnership::NONE, [1.0, 1.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{ButtonState, MouseButton, ScrollUnits, TouchId, TouchPhase};

    fn left(state: ButtonState) -> ViewportEvent {
        ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state,
        }
    }

    fn wheel() -> ViewportEvent {
        ViewportEvent::Wheel {
            delta: glam::Vec2::new(0.0, 1.0),
            units: ScrollUnits::Lines,
        }
    }

    #[test]
    fn pointer_only_keeps_the_keyboard_with_the_pointer() {
        let cases = [
            (PointerOwnership::Elsewhere, KeyboardOwnership::Elsewhere),
            (PointerOwnership::Inside, KeyboardOwnership::Elsewhere),
            (PointerOwnership::Owned, KeyboardOwnership::Owned),
            (PointerOwnership::Grabbed, KeyboardOwnership::Owned),
        ];
        for (pointer, keyboard) in cases {
            let o = InputOwnership::pointer_only(pointer);
            assert_eq!(o.pointer, pointer);
            assert_eq!(o.keyboard, keyboard, "keyboard for {pointer:?}");
            #[allow(deprecated)]
            let ctx = ViewportContext::with_ownership(pointer, [800.0, 600.0]);
            assert_eq!(ctx.ownership, o);
            assert_eq!(ctx.viewport_size, [800.0, 600.0]);
        }
    }

    #[test]
    fn the_named_ownerships() {
        assert_eq!(InputOwnership::NONE, InputOwnership::default());
        assert_eq!(InputOwnership::NONE.pointer, PointerOwnership::Elsewhere);
        assert_eq!(InputOwnership::NONE.keyboard, KeyboardOwnership::Elsewhere);
        let full = InputOwnership::full_window();
        assert_eq!(full.pointer, PointerOwnership::Owned);
        assert_eq!(full.keyboard, KeyboardOwnership::Owned);
        assert_eq!(ViewportContext::default().ownership, InputOwnership::NONE);
    }

    #[test]
    fn a_tool_over_the_viewport_keeps_the_wheel_with_the_camera() {
        // The case this whole distinction exists for: chrome holds the pointer, so the
        // press is not the scene's, but scrolling to zoom must keep working.
        let o = PointerOwnership::Inside;
        assert!(!forward_to_viewport(&left(ButtonState::Pressed), o));
        assert!(!forward_to_viewport(
            &ViewportEvent::PointerMoved {
                position: glam::Vec2::ZERO
            },
            o
        ));
        assert!(forward_to_viewport(&wheel(), o));
        assert!(forward_to_viewport(
            &ViewportEvent::MouseButton {
                button: MouseButton::Right,
                state: ButtonState::Pressed
            },
            o
        ));
    }

    #[test]
    fn owning_the_input_forwards_everything() {
        for o in [PointerOwnership::Owned, PointerOwnership::Grabbed] {
            assert!(forward_to_viewport(&left(ButtonState::Pressed), o));
            assert!(forward_to_viewport(&left(ButtonState::Released), o));
            assert!(forward_to_viewport(&wheel(), o));
        }
    }

    #[test]
    fn a_left_release_always_lands() {
        // Forwarding the press but not the release would leave the button stuck down.
        assert!(left(ButtonState::Pressed).is_contested());
        assert!(!left(ButtonState::Released).is_contested());
        for o in [
            PointerOwnership::Elsewhere,
            PointerOwnership::Inside,
            PointerOwnership::Owned,
            PointerOwnership::Grabbed,
        ] {
            assert!(
                forward_to_viewport(&left(ButtonState::Released), o),
                "{o:?}"
            );
        }
    }

    #[test]
    fn a_contact_is_withheld_but_its_end_is_not() {
        let touch = |phase| ViewportEvent::Touch {
            id: TouchId(1),
            phase,
            position: glam::Vec2::ZERO,
        };
        // A finger that lands on a control over the viewport is not also the camera's.
        assert!(!forward_to_viewport(
            &touch(TouchPhase::Started),
            PointerOwnership::Inside
        ));
        // But teardown always lands, or ownership changing mid-gesture strands the contact.
        for phase in [TouchPhase::Ended, TouchPhase::Cancelled] {
            for o in [
                PointerOwnership::Elsewhere,
                PointerOwnership::Inside,
                PointerOwnership::Owned,
            ] {
                assert!(
                    forward_to_viewport(&touch(phase), o),
                    "{phase:?} under {o:?}"
                );
            }
        }
    }

    #[test]
    fn a_grabbed_pointer_never_hears_pointer_left() {
        // PointerLeft releases every held button, which would end a look session; and a
        // grabbed pointer has no rect to leave in the first place.
        assert!(!forward_to_viewport(
            &ViewportEvent::PointerLeft,
            PointerOwnership::Grabbed
        ));
        for o in [
            PointerOwnership::Elsewhere,
            PointerOwnership::Inside,
            PointerOwnership::Owned,
        ] {
            assert!(forward_to_viewport(&ViewportEvent::PointerLeft, o));
        }
    }
}
