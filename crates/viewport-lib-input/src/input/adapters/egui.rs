//! egui -> [`ViewportEvent`] translation.
//!
//! egui's event types are versioned, and a host embedding a viewport runs
//! whichever egui its framework pins, so the mapping below is written once and
//! generated per supported egui version. Enable the adapter feature matching
//! your egui (`egui-adapter` for 0.33, `egui-adapter-035`, `egui-adapter-036`)
//! and call [`from_egui`] as normal: the enabled version's copy is what gets
//! re-exported.

#[cfg(any(
    all(feature = "egui-adapter", feature = "egui-adapter-035"),
    all(feature = "egui-adapter", feature = "egui-adapter-036"),
    all(feature = "egui-adapter-035", feature = "egui-adapter-036"),
))]
compile_error!(
    "enable exactly one egui adapter feature: egui-adapter (0.33), egui-adapter-035, or egui-adapter-036"
);

/// The mapping, instantiated against one aliased egui crate.
///
/// Everything egui-shaped is named through `$egui`, so the only thing that
/// differs between the generated copies is which crate they read the event enum
/// from. A version whose enum diverges would show up here as a compile error in
/// that copy alone.
macro_rules! egui_adapter {
    ($egui:ident) => {
        use $egui as egui;

        use crate::input::{
            ButtonState, KeyCode, MouseButton, ScrollUnits, TouchId, TouchPhase, ViewportEvent,
        };

        /// Translate an `egui::Event` into a [`ViewportEvent`], or `None` for
        /// events the viewport does not consume.
        ///
        /// `viewport_origin` is the top-left of the viewport rectangle in egui
        /// points (logical coordinates); it is subtracted from pointer positions
        /// to make them viewport-local. Coordinates stay in logical points, which
        /// is the space the viewport's screen-space math and `viewport_size` use;
        /// HiDPI sharpness comes from the render-target size and
        /// `pixels_per_point`, not from the pointer coordinates. Modifier state is
        /// not carried on `ViewportEvent`; push a
        /// [`ViewportEvent::ModifiersChanged`] from the frame's `InputState`
        /// separately.
        pub fn from_egui(
            event: &egui::Event,
            viewport_origin: glam::Vec2,
        ) -> Option<ViewportEvent> {
            use egui::Event;

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
                        egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                        egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                        egui::MouseWheelUnit::Page => ScrollUnits::Pages,
                    };
                    Some(ViewportEvent::Wheel {
                        delta: glam::Vec2::new(delta.x, delta.y),
                        units,
                    })
                }
                // egui reports a pinch as a multiplicative zoom factor (1.0 = no
                // change); the viewport convention is winit's additive, log-scale
                // delta. `ln` is the exact inverse of the `exp` egui-winit applies
                // on the way in, so a pinch that arrives through egui and the same
                // pinch that arrives through winit resolve to the same zoom.
                Event::Zoom(factor) => {
                    (*factor > 0.0).then(|| ViewportEvent::TrackpadPinch(factor.ln()))
                }
                // egui's angle convention is the opposite of winit's, which is why
                // egui-winit negates on the way in. Negate back, so twist turns the
                // same way whichever shell hosts the viewport.
                Event::Rotate(radians) => Some(ViewportEvent::TrackpadRotate(-*radians)),
                // Raw contacts, not egui's derived multi-touch state: the viewport
                // recognises gestures itself so the feel is the same through every
                // host. egui also warns its own translation delta may not be in
                // screen points, which the recogniser would have to undo.
                Event::Touch { id, phase, pos, .. } => Some(ViewportEvent::Touch {
                    id: TouchId(id.0),
                    phase: map_touch_phase(*phase),
                    position: glam::Vec2::new(pos.x, pos.y) - viewport_origin,
                }),
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

        fn map_touch_phase(phase: egui::TouchPhase) -> TouchPhase {
            match phase {
                egui::TouchPhase::Start => TouchPhase::Started,
                egui::TouchPhase::Move => TouchPhase::Moved,
                egui::TouchPhase::End => TouchPhase::Ended,
                egui::TouchPhase::Cancel => TouchPhase::Cancelled,
            }
        }

        fn map_button(button: egui::PointerButton) -> Option<MouseButton> {
            match button {
                egui::PointerButton::Primary => Some(MouseButton::Left),
                egui::PointerButton::Secondary => Some(MouseButton::Right),
                egui::PointerButton::Middle => Some(MouseButton::Middle),
                _ => None,
            }
        }

        fn map_key(key: egui::Key) -> Option<KeyCode> {
            use egui::Key as E;
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
            use egui::{Event, Key as E, Modifiers, MouseWheelUnit, PointerButton, Pos2};

            /// egui keys the adapter maps, and the library key each must produce.
            /// egui's key set is a superset of what the viewport models (it also
            /// carries Copy / Paste / punctuation the viewport has no binding
            /// for), so this is not exhaustive over `KeyCode`; it locks the
            /// mappings that exist, including the ones easy to forget: F13-F24,
            /// `=`, and `'`.
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

            /// egui keys the viewport has no binding for translate to `None`, not
            /// to a wrong code. `F25`+ is past the library's `F24`;
            /// `Copy`/`Plus`/`BrowserBack` have no `KeyCode`.
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
                            delta: egui::Vec2::new(1.0, -2.0),
                            // egui 0.35 added a scroll phase to this event. The
                            // adapter ignores it (it matches with `..`), but the
                            // literal here has to carry it.
                            #[cfg(any(
                                feature = "egui-adapter-035",
                                feature = "egui-adapter-036"
                            ))]
                            phase: egui::TouchPhase::Move,
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
            fn a_pinch_becomes_the_viewport_log_scale_delta() {
                // egui hands us a factor; the viewport speaks winit's additive delta.
                // A factor of 1.0 is no change, so it must resolve to zero.
                match from_egui(&Event::Zoom(1.0), glam::Vec2::ZERO) {
                    Some(ViewportEvent::TrackpadPinch(d)) => assert!(d.abs() < 1e-6),
                    other => panic!("expected TrackpadPinch, got {other:?}"),
                }
                // Spreading (factor > 1) zooms in, which is a positive delta.
                match from_egui(&Event::Zoom(std::f32::consts::E), glam::Vec2::ZERO) {
                    Some(ViewportEvent::TrackpadPinch(d)) => assert!((d - 1.0).abs() < 1e-5),
                    other => panic!("expected TrackpadPinch, got {other:?}"),
                }
                // Pinching together is negative.
                match from_egui(&Event::Zoom(0.5), glam::Vec2::ZERO) {
                    Some(ViewportEvent::TrackpadPinch(d)) => assert!(d < 0.0),
                    other => panic!("expected TrackpadPinch, got {other:?}"),
                }
                // A zero or negative factor has no logarithm; drop it rather than
                // feed the resolver an infinity.
                assert!(from_egui(&Event::Zoom(0.0), glam::Vec2::ZERO).is_none());
            }

            #[test]
            fn rotation_is_negated_onto_the_winit_convention() {
                // egui's positive angle is winit's negative one. Getting this wrong
                // makes twist spin opposite ways depending on the host, which is the
                // whole reason the adapter converts rather than passing through.
                match from_egui(&Event::Rotate(0.25), glam::Vec2::ZERO) {
                    Some(ViewportEvent::TrackpadRotate(r)) => assert!((r + 0.25).abs() < 1e-6),
                    other => panic!("expected TrackpadRotate, got {other:?}"),
                }
            }

            #[test]
            fn a_touch_keeps_its_id_and_is_made_viewport_local() {
                let ev = from_egui(
                    &Event::Touch {
                        device_id: egui::TouchDeviceId(0),
                        id: egui::TouchId(7),
                        phase: egui::TouchPhase::Move,
                        pos: Pos2::new(60.0, 50.0),
                        force: None,
                    },
                    glam::Vec2::new(10.0, 20.0),
                );
                match ev {
                    Some(ViewportEvent::Touch {
                        id,
                        phase,
                        position,
                    }) => {
                        assert_eq!(id, TouchId(7));
                        assert_eq!(phase, TouchPhase::Moved);
                        assert_eq!(position, glam::Vec2::new(50.0, 30.0));
                    }
                    other => panic!("expected Touch, got {other:?}"),
                }
            }

            #[test]
            fn egui_touch_phases_map() {
                let cases = [
                    (egui::TouchPhase::Start, TouchPhase::Started),
                    (egui::TouchPhase::Move, TouchPhase::Moved),
                    (egui::TouchPhase::End, TouchPhase::Ended),
                    (egui::TouchPhase::Cancel, TouchPhase::Cancelled),
                ];
                for (egui_phase, expected) in cases {
                    assert_eq!(map_touch_phase(egui_phase), expected);
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
    };
}

#[cfg(feature = "egui-adapter")]
mod v033 {
    egui_adapter!(egui033);
}
#[cfg(feature = "egui-adapter")]
pub use v033::from_egui;

#[cfg(feature = "egui-adapter-035")]
mod v035 {
    egui_adapter!(egui035);
}
#[cfg(feature = "egui-adapter-035")]
pub use v035::from_egui;

#[cfg(feature = "egui-adapter-036")]
mod v036 {
    egui_adapter!(egui036);
}
#[cfg(feature = "egui-adapter-036")]
pub use v036::from_egui;
