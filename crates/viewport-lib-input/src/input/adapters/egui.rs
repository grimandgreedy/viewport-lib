//! egui -> [`ViewportEvent`] translation.

use crate::input::{ButtonState, KeyCode, MouseButton, ScrollUnits, ViewportEvent};

/// Translate an [`egui::Event`](::egui::Event) into a [`ViewportEvent`], or
/// `None` for events the viewport does not consume.
///
/// `viewport_origin` is the top-left of the viewport rectangle in egui points
/// (logical coordinates); it is subtracted from pointer positions to make them
/// viewport-local. Coordinates stay in logical points, which is the space the
/// viewport's screen-space math and `viewport_size` use; HiDPI sharpness comes
/// from the render-target size and `pixels_per_point`, not from the pointer
/// coordinates. Modifier state is not carried on `ViewportEvent`; push a
/// [`ViewportEvent::ModifiersChanged`] from the frame's `InputState` separately.
pub fn from_egui(event: &::egui::Event, viewport_origin: glam::Vec2) -> Option<ViewportEvent> {
    use ::egui::Event;

    match event {
        Event::PointerMoved(pos) => Some(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(pos.x, pos.y) - viewport_origin,
        }),
        Event::PointerButton {
            button, pressed, ..
        } => Some(ViewportEvent::MouseButton {
            button: map_button(*button)?,
            state: if *pressed {
                ButtonState::Pressed
            } else {
                ButtonState::Released
            },
        }),
        Event::PointerGone => Some(ViewportEvent::PointerLeft),
        Event::MouseWheel { unit, delta, .. } => {
            let units = match unit {
                ::egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                ::egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                ::egui::MouseWheelUnit::Page => ScrollUnits::Pages,
            };
            Some(ViewportEvent::Wheel {
                delta: glam::Vec2::new(delta.x, delta.y),
                units,
            })
        }
        Event::Key {
            key,
            pressed,
            repeat,
            ..
        } => Some(ViewportEvent::Key {
            key: map_key(*key)?,
            state: if *pressed {
                ButtonState::Pressed
            } else {
                ButtonState::Released
            },
            repeat: *repeat,
        }),
        _ => None,
    }
}

fn map_button(button: ::egui::PointerButton) -> Option<MouseButton> {
    match button {
        ::egui::PointerButton::Primary => Some(MouseButton::Left),
        ::egui::PointerButton::Secondary => Some(MouseButton::Right),
        ::egui::PointerButton::Middle => Some(MouseButton::Middle),
        _ => None,
    }
}

fn map_key(key: ::egui::Key) -> Option<KeyCode> {
    use ::egui::Key as E;
    let k = match key {
        E::A => KeyCode::A,
        E::B => KeyCode::B,
        E::C => KeyCode::C,
        E::D => KeyCode::D,
        E::E => KeyCode::E,
        E::F => KeyCode::F,
        E::G => KeyCode::G,
        E::H => KeyCode::H,
        E::I => KeyCode::I,
        E::J => KeyCode::J,
        E::K => KeyCode::K,
        E::L => KeyCode::L,
        E::M => KeyCode::M,
        E::N => KeyCode::N,
        E::O => KeyCode::O,
        E::P => KeyCode::P,
        E::Q => KeyCode::Q,
        E::R => KeyCode::R,
        E::S => KeyCode::S,
        E::T => KeyCode::T,
        E::U => KeyCode::U,
        E::V => KeyCode::V,
        E::W => KeyCode::W,
        E::X => KeyCode::X,
        E::Y => KeyCode::Y,
        E::Z => KeyCode::Z,
        E::Num0 => KeyCode::Num0,
        E::Num1 => KeyCode::Num1,
        E::Num2 => KeyCode::Num2,
        E::Num3 => KeyCode::Num3,
        E::Num4 => KeyCode::Num4,
        E::Num5 => KeyCode::Num5,
        E::Num6 => KeyCode::Num6,
        E::Num7 => KeyCode::Num7,
        E::Num8 => KeyCode::Num8,
        E::Num9 => KeyCode::Num9,
        E::Escape => KeyCode::Escape,
        E::Enter => KeyCode::Enter,
        E::Tab => KeyCode::Tab,
        E::Space => KeyCode::Space,
        E::Backspace => KeyCode::Backspace,
        E::ArrowUp => KeyCode::Up,
        E::ArrowDown => KeyCode::Down,
        E::ArrowLeft => KeyCode::Left,
        E::ArrowRight => KeyCode::Right,
        E::Minus => KeyCode::Minus,
        E::Comma => KeyCode::Comma,
        E::Period => KeyCode::Period,
        E::Slash => KeyCode::Slash,
        E::Backtick => KeyCode::Backtick,
        E::Backslash => KeyCode::Backslash,
        E::OpenBracket => KeyCode::LeftBracket,
        E::CloseBracket => KeyCode::RightBracket,
        E::Semicolon => KeyCode::Semicolon,
        E::Quote => KeyCode::Apostrophe,
        E::Equals => KeyCode::Equals,
        E::Delete => KeyCode::Delete,
        E::Insert => KeyCode::Insert,
        E::Home => KeyCode::Home,
        E::End => KeyCode::End,
        E::PageUp => KeyCode::PageUp,
        E::PageDown => KeyCode::PageDown,
        E::F1 => KeyCode::F1,
        E::F2 => KeyCode::F2,
        E::F3 => KeyCode::F3,
        E::F4 => KeyCode::F4,
        E::F5 => KeyCode::F5,
        E::F6 => KeyCode::F6,
        E::F7 => KeyCode::F7,
        E::F8 => KeyCode::F8,
        E::F9 => KeyCode::F9,
        E::F10 => KeyCode::F10,
        E::F11 => KeyCode::F11,
        E::F12 => KeyCode::F12,
        E::F13 => KeyCode::F13,
        E::F14 => KeyCode::F14,
        E::F15 => KeyCode::F15,
        E::F16 => KeyCode::F16,
        E::F17 => KeyCode::F17,
        E::F18 => KeyCode::F18,
        E::F19 => KeyCode::F19,
        E::F20 => KeyCode::F20,
        E::F21 => KeyCode::F21,
        E::F22 => KeyCode::F22,
        E::F23 => KeyCode::F23,
        E::F24 => KeyCode::F24,
        _ => return None,
    };
    Some(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::egui::{Event, Key as E, Modifiers, MouseWheelUnit, PointerButton, Pos2, Vec2 as EVec2};

    /// egui keys the adapter maps, and the library key each must produce. egui's
    /// key set is a superset of what the viewport models (it also carries Copy /
    /// Paste / punctuation the viewport has no binding for), so this is not
    /// exhaustive over `KeyCode`; it locks the mappings that exist, including the
    /// ones easy to forget: F13-F24, `=`, and `'`.
    const EGUI_KEYS: &[(E, KeyCode)] = &[
        (E::A, KeyCode::A),
        (E::Z, KeyCode::Z),
        (E::Num0, KeyCode::Num0),
        (E::Num9, KeyCode::Num9),
        (E::Escape, KeyCode::Escape),
        (E::Enter, KeyCode::Enter),
        (E::Space, KeyCode::Space),
        (E::ArrowUp, KeyCode::Up),
        (E::ArrowLeft, KeyCode::Left),
        (E::OpenBracket, KeyCode::LeftBracket),
        (E::CloseBracket, KeyCode::RightBracket),
        (E::Backtick, KeyCode::Backtick),
        (E::Semicolon, KeyCode::Semicolon),
        (E::Quote, KeyCode::Apostrophe),
        (E::Equals, KeyCode::Equals),
        (E::Minus, KeyCode::Minus),
        (E::PageDown, KeyCode::PageDown),
        (E::F1, KeyCode::F1),
        (E::F12, KeyCode::F12),
        (E::F13, KeyCode::F13),
        (E::F24, KeyCode::F24),
    ];

    #[test]
    fn egui_keys_map_to_the_right_code() {
        for &(key, code) in EGUI_KEYS {
            assert_eq!(map_key(key), Some(code), "egui {key:?}");
        }
    }

    /// egui keys the viewport has no binding for translate to `None`, not to a
    /// wrong code. `F25`+ is past the library's `F24`; `Copy`/`Plus`/`BrowserBack`
    /// have no `KeyCode`.
    #[test]
    fn egui_keys_without_a_code_are_none() {
        for key in [
            E::F25,
            E::F35,
            E::Copy,
            E::Cut,
            E::Paste,
            E::Plus,
            E::Pipe,
            E::BrowserBack,
        ] {
            assert_eq!(map_key(key), None, "egui {key:?} should not map");
        }
    }

    #[test]
    fn pointer_moved_is_made_viewport_local() {
        let ev = from_egui(
            &Event::PointerMoved(Pos2::new(100.0, 80.0)),
            glam::Vec2::new(10.0, 20.0),
        );
        match ev {
            Some(ViewportEvent::PointerMoved { position }) => {
                assert_eq!(position, glam::Vec2::new(90.0, 60.0));
            }
            other => panic!("expected PointerMoved, got {other:?}"),
        }
    }

    #[test]
    fn pointer_buttons_map_with_state() {
        let press = from_egui(
            &Event::PointerButton {
                pos: Pos2::ZERO,
                button: PointerButton::Secondary,
                pressed: true,
                modifiers: Modifiers::default(),
            },
            glam::Vec2::ZERO,
        );
        assert!(matches!(
            press,
            Some(ViewportEvent::MouseButton {
                button: MouseButton::Right,
                state: ButtonState::Pressed
            })
        ));

        let release = from_egui(
            &Event::PointerButton {
                pos: Pos2::ZERO,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Modifiers::default(),
            },
            glam::Vec2::ZERO,
        );
        assert!(matches!(
            release,
            Some(ViewportEvent::MouseButton {
                button: MouseButton::Left,
                state: ButtonState::Released
            })
        ));
    }

    #[test]
    fn extra_pointer_buttons_drop() {
        let ev = from_egui(
            &Event::PointerButton {
                pos: Pos2::ZERO,
                button: PointerButton::Extra1,
                pressed: true,
                modifiers: Modifiers::default(),
            },
            glam::Vec2::ZERO,
        );
        assert!(ev.is_none());
    }

    #[test]
    fn wheel_units_are_carried_through() {
        let cases = [
            (MouseWheelUnit::Line, ScrollUnits::Lines),
            (MouseWheelUnit::Point, ScrollUnits::Pixels),
            (MouseWheelUnit::Page, ScrollUnits::Pages),
        ];
        for (unit, expected) in cases {
            let ev = from_egui(
                &Event::MouseWheel {
                    unit,
                    delta: EVec2::new(1.0, -2.0),
                    modifiers: Modifiers::default(),
                },
                glam::Vec2::ZERO,
            );
            match ev {
                Some(ViewportEvent::Wheel { delta, units }) => {
                    assert_eq!(units, expected, "unit {unit:?}");
                    assert_eq!(delta, glam::Vec2::new(1.0, -2.0));
                }
                other => panic!("expected Wheel, got {other:?}"),
            }
        }
    }

    #[test]
    fn pointer_gone_becomes_pointer_left() {
        assert!(matches!(
            from_egui(&Event::PointerGone, glam::Vec2::ZERO),
            Some(ViewportEvent::PointerLeft)
        ));
    }

    #[test]
    fn key_event_carries_state_and_repeat() {
        let ev = from_egui(
            &Event::Key {
                key: E::W,
                physical_key: None,
                pressed: true,
                repeat: true,
                modifiers: Modifiers::default(),
            },
            glam::Vec2::ZERO,
        );
        match ev {
            Some(ViewportEvent::Key { key, state, repeat }) => {
                assert_eq!(key, KeyCode::W);
                assert_eq!(state, ButtonState::Pressed);
                assert!(repeat);
            }
            other => panic!("expected Key, got {other:?}"),
        }
    }
}
