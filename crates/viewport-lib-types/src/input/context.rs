//! Per-frame viewport context for the new input pipeline.

use super::event::{TouchPhase, ViewportEvent};

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
    /// state except [`Elsewhere`](Self::Elsewhere); this is what
    /// [`ViewportContext::hovered`] wants.
    pub fn pointer_inside(self) -> bool {
        !matches!(self, PointerOwnership::Elsewhere)
    }

    /// Whether the viewport won the input this frame. True for
    /// [`Owned`](Self::Owned) and [`Grabbed`](Self::Grabbed); this is what
    /// [`ViewportContext::focused`] wants.
    pub fn owns_input(self) -> bool {
        matches!(self, PointerOwnership::Owned | PointerOwnership::Grabbed)
    }
}

/// Whether to forward `event` to the viewport, given who owns the pointer this frame.
///
/// The routing rule in one call. Contested events (see
/// [`ViewportEvent::is_contested`]) reach the viewport only when it owns the input;
/// everything else is ambient and reaches it regardless, so a camera keeps zooming
/// while a tool over the viewport holds the pointer. Ambient events are still gated
/// inside the resolver by the [`ViewportContext`] you set: the wheel accrues only
/// while hovered, keys only while focused.
///
/// Two special cases, both about teardown. [`ViewportEvent::PointerLeft`] is dropped
/// under [`PointerOwnership::Grabbed`] (see that variant), and a touch contact ending
/// or being cancelled is forwarded whoever owns the input, so a contact the viewport
/// was already tracking cannot be stranded down by ownership changing mid-gesture. An
/// end for a contact the viewport never saw start is a no-op, so this cannot make it
/// act on a gesture that was never its own.
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
        ViewportEvent::Touch {
            phase: TouchPhase::Ended | TouchPhase::Cancelled,
            ..
        } => true,
        _ if event.is_contested() => ownership.owns_input(),
        _ => true,
    }
}

/// Per-frame viewport context passed to `OrbitCameraController::begin_frame`
/// (and `ViewportInput::begin_frame`).
///
/// Carries the hover, focus, and size state needed to interpret viewport input.
///
/// The two flags answer different questions and are easy to conflate. Prefer
/// [`with_ownership`](Self::with_ownership), which sets both from one
/// [`PointerOwnership`] and cannot get them out of step.
#[derive(Debug, Clone, Copy)]
pub struct ViewportContext {
    /// Whether the pointer is geometrically inside the viewport, whatever owns it.
    ///
    /// Chrome drawn over the viewport does **not** make this false: the pointer is
    /// still in the viewport, and the camera should still answer the wheel there.
    /// Gates the wheel and pointer gestures.
    pub hovered: bool,
    /// Whether the viewport won the input this frame.
    ///
    /// Chrome drawn over the viewport **does** make this false. Gates keys, so
    /// setting it from `hovered` makes a hovered viewport swallow keystrokes a
    /// focused text field elsewhere should receive.
    pub focused: bool,
    /// Viewport size in logical pixels `[width, height]`.
    pub viewport_size: [f32; 2],
}

impl ViewportContext {
    /// Build a context from the application's routing decision and the viewport size.
    ///
    /// Sets `hovered` and `focused` from the one ownership state, so they cannot
    /// disagree:
    ///
    /// | Ownership | `hovered` | `focused` |
    /// |---|---|---|
    /// | `Elsewhere` | false | false |
    /// | `Inside` | true | false |
    /// | `Owned` | true | true |
    /// | `Grabbed` | true | true |
    pub fn with_ownership(ownership: PointerOwnership, viewport_size: [f32; 2]) -> Self {
        Self {
            hovered: ownership.pointer_inside(),
            focused: ownership.owns_input(),
            viewport_size,
        }
    }
}

impl Default for ViewportContext {
    fn default() -> Self {
        Self {
            hovered: false,
            focused: false,
            viewport_size: [1.0, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{ButtonState, MouseButton, ScrollUnits, TouchId};

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
    fn ownership_maps_onto_the_two_flags() {
        let cases = [
            (PointerOwnership::Elsewhere, false, false),
            (PointerOwnership::Inside, true, false),
            (PointerOwnership::Owned, true, true),
            (PointerOwnership::Grabbed, true, true),
        ];
        for (ownership, hovered, focused) in cases {
            let ctx = ViewportContext::with_ownership(ownership, [800.0, 600.0]);
            assert_eq!(ctx.hovered, hovered, "hovered for {ownership:?}");
            assert_eq!(ctx.focused, focused, "focused for {ownership:?}");
            assert_eq!(ctx.viewport_size, [800.0, 600.0]);
        }
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
    fn a_left_release_is_contested_too() {
        // Forwarding the press but not the release would leave the button stuck down.
        assert!(left(ButtonState::Pressed).is_contested());
        assert!(left(ButtonState::Released).is_contested());
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
        assert!(!forward_to_viewport(
            &touch(TouchPhase::Moved),
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
