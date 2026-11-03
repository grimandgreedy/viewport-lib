//! egui -> [`ViewportEvent`] translation.
//!
//! egui's event types are versioned, and a host embedding a viewport runs
//! whichever egui its framework pins, so the mapping below is written once and
//! generated per supported egui version. Enable the adapter feature matching
//! your egui (`egui-adapter` for 0.33, `egui-adapter-035`, `egui-adapter-036`)
//! and call [`translate`], [`begin_frame`] and [`ownership`] as normal: the
//! enabled version's copy is what gets re-exported.

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

        use super::super::AdapterCtx;
        use crate::input::{
            ButtonState, ImeEvent, InputOwnership, KeyCode, KeyboardOwnership, Modifiers,
            MouseButton, PointerOwnership, ScrollUnits, TouchId, TouchPhase, ViewportEvent,
        };

        /// Who owns the input for a viewport drawn as an egui widget.
        ///
        /// `response` is the viewport widget's response and `rect` the viewport's
        /// rectangle, usually `response.rect`. `session_active` is whether the
        /// application has a manipulation session in flight (a G/R/S move driven by
        /// the keyboard and the pointer), which keeps the pointer for the session.
        ///
        /// The pointer is [`Grabbed`](PointerOwnership::Grabbed) while the widget
        /// is dragged or a session is active, [`Owned`](PointerOwnership::Owned)
        /// while it is hovered, [`Inside`](PointerOwnership::Inside) when the
        /// pointer is in `rect` but egui gave the hover to something drawn over it,
        /// and [`Elsewhere`](PointerOwnership::Elsewhere) otherwise. The keyboard is
        /// the viewport's while it has the pointer and no egui widget has keyboard
        /// focus, so typing into a text field beside a hovered viewport does not
        /// reach it.
        ///
        /// A camera drag in flight needs no escalation: the resolver keeps its own
        /// held press's events when the pointer crosses egui chrome or leaves the
        /// rect. [`Grabbed`](PointerOwnership::Grabbed) from the host is for
        /// pointer capture. With a session active, a click on an egui button both
        /// presses the button and confirms the session.
        ///
        /// A tool drawn inside the viewport (a gizmo handle) is not something egui
        /// can see, so this reports the pointer owned over it. Give the camera's
        /// resolver [`Inside`](PointerOwnership::Inside) while the tool wants the
        /// pointer, and call its `cancel_gesture()` when the tool takes a press.
        pub fn ownership(
            response: &egui::Response,
            ctx: &egui::Context,
            rect: egui::Rect,
            session_active: bool,
        ) -> InputOwnership {
            let pointer = if response.dragged() || session_active {
                PointerOwnership::Grabbed
            } else if response.hovered() {
                PointerOwnership::Owned
            } else if ctx
                .input(|i| i.pointer.latest_pos())
                .is_some_and(|p| rect.contains(p))
            {
                PointerOwnership::Inside
            } else {
                PointerOwnership::Elsewhere
            };
            // Any focused egui widget is listening for keys (what egui's
            // `wants_keyboard_input` reports, under a name that changed in 0.35).
            let text_focus = ctx.memory(|m| m.focused().is_some());
            let keyboard = if !text_focus && pointer.owns_input() {
                KeyboardOwnership::Owned
            } else {
                KeyboardOwnership::Elsewhere
            };
            InputOwnership { pointer, keyboard }
        }

        /// Translate egui's modifier state into [`Modifiers`], for a
        /// [`ViewportEvent::ModifiersChanged`]. Ctrl and Cmd (`mac_cmd`) stay
        /// apart; bindings still treat them as one key.
        pub fn modifiers_from_egui(m: &egui::Modifiers) -> Modifiers {
            Modifiers::NONE
                .with_alt(m.alt)
                .with_shift(m.shift)
                .with_ctrl(m.ctrl)
                .with_super(m.mac_cmd)
        }

        /// The character egui's layout-following key stands for, lower case, or
        /// `None` for keys with no printable character.
        fn logical_char(key: egui::Key) -> Option<char> {
            let mut chars = key.symbol_or_name().chars();
            let c = chars.next()?;
            (chars.next().is_none() && c.is_ascii_graphic()).then(|| c.to_ascii_lowercase())
        }

        /// Translate an `egui::Event`, appending what it means for the viewport
        /// to `out`.
        ///
        /// Call [`begin_frame`] first each frame, then this for every event in
        /// the frame's `InputState::events`. Typed text (`Event::Text`) becomes
        /// one [`ViewportEvent::Character`] per character, control characters
        /// left out. Window focus is left to `begin_frame`, which turns its loss
        /// into [`ViewportEvent::FocusLost`]. Events the viewport has no use for
        /// append nothing.
        ///
        /// Positions are divided by `ctx.scale_factor` (`1.0` for egui, whose
        /// points are already logical) and offset by `ctx.viewport_origin`, the
        /// viewport rectangle's top-left in egui points, so they come out
        /// viewport-local.
        ///
        /// `ViewportEvent::Key` carries the key's position (its US-layout name),
        /// taken from the event's `physical_key`, and the layout character from
        /// egui's layout-following `key`. A host that leaves `physical_key`
        /// empty gets the layout key for both.
        pub fn translate(event: &egui::Event, ctx: &AdapterCtx, out: &mut Vec<ViewportEvent>) {
            if let egui::Event::Text(text) = event {
                out.extend(
                    text.chars()
                        .filter(|c| !c.is_control())
                        .map(ViewportEvent::Character),
                );
            } else {
                out.extend(translate_one(event, ctx));
            }
        }

        /// The events egui reports as frame state rather than as events: the
        /// modifier state as a [`ViewportEvent::ModifiersChanged`], a
        /// [`ViewportEvent::FocusLost`] when the window lost focus this frame
        /// (which releases held keys and buttons), and a
        /// [`ViewportEvent::FileDropped`] per file dropped this frame.
        ///
        /// Call it once per frame, before [`translate`]ing the frame's events.
        /// Nothing it emits carries a position, so `ctx` is not read.
        pub fn begin_frame(
            input: &egui::InputState,
            ctx: &AdapterCtx,
            out: &mut Vec<ViewportEvent>,
        ) {
            let _ = ctx;
            out.push(ViewportEvent::ModifiersChanged(modifiers_from_egui(
                &input.modifiers,
            )));
            if input
                .events
                .iter()
                .any(|e| matches!(e, egui::Event::WindowFocused(false)))
            {
                out.push(ViewportEvent::FocusLost);
            }
            #[cfg(not(feature = "egui-adapter-036"))]
            let dropped = input.raw.dropped_files.iter().filter_map(|f| f.path.clone());
            #[cfg(feature = "egui-adapter-036")]
            let dropped = input
                .raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf());
            out.extend(dropped.map(ViewportEvent::FileDropped));
        }

        /// Translate an `egui::Event` into one [`ViewportEvent`], or `None` for
        /// events the viewport does not consume. Typed text is not included.
        #[deprecated(note = "use translate, which also emits typed text, and begin_frame")]
        pub fn from_egui(
            event: &egui::Event,
            viewport_origin: glam::Vec2,
        ) -> Option<ViewportEvent> {
            translate_one(event, &AdapterCtx::new(1.0, viewport_origin))
        }

        /// The one event an egui event maps to, without typed text.
        #[allow(deprecated)] // egui 0.35 deprecates the IME enable and disable events.
        fn translate_one(event: &egui::Event, ctx: &AdapterCtx) -> Option<ViewportEvent> {
            use egui::Event;

            match event {
                Event::PointerMoved(pos) => Some(ViewportEvent::PointerMoved {
                    position: ctx.local(pos.x, pos.y),
                }),
                // Unaccelerated motion, the counterpart of winit's device motion.
                Event::MouseMoved(delta) => Some(ViewportEvent::RawMotion {
                    delta: glam::Vec2::new(delta.x, delta.y),
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
                    position: ctx.local(pos.x, pos.y),
                }),
                // The key position, as the winit adapter reports it, so a binding
                // means the same key through either adapter. egui's `key` follows
                // the layout; it is the fallback when the host gives no physical
                // key.
                Event::Key {
                    key,
                    physical_key,
                    pressed,
                    repeat,
                    ..
                } => Some(ViewportEvent::Key {
                    key: map_key(physical_key.unwrap_or(*key))?,
                    state: if *pressed {
                        ButtonState::Pressed
                    } else {
                        ButtonState::Released
                    },
                    repeat: *repeat,
                    logical: logical_char(*key),
                }),
                #[cfg(feature = "egui-adapter-036")]
                Event::ModifiersChanged(m) => {
                    Some(ViewportEvent::ModifiersChanged(modifiers_from_egui(m)))
                }
                Event::Ime(ime) => Some(ViewportEvent::Ime(match ime {
                    egui::ImeEvent::Enabled => ImeEvent::Enabled,
                    #[cfg(feature = "egui-adapter")]
                    egui::ImeEvent::Preedit(text) => preedit(text),
                    #[cfg(not(feature = "egui-adapter"))]
                    egui::ImeEvent::Preedit { text, .. } => preedit(text),
                    egui::ImeEvent::Commit(text) => ImeEvent::Commit(text.clone()),
                    egui::ImeEvent::Disabled => ImeEvent::Disabled,
                    #[allow(unreachable_patterns)]
                    _ => return None,
                })),
                _ => None,
            }
        }

        /// A preedit with the caret at its end: egui does not report one.
        fn preedit(text: &str) -> ImeEvent {
            ImeEvent::Preedit {
                text: text.to_owned(),
                cursor: Some((text.len(), text.len())),
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

            /// A file dropped this frame, in the shape this egui version uses.
            #[cfg(not(feature = "egui-adapter-036"))]
            fn dropped(path: &str) -> egui::DroppedFile {
                egui::DroppedFile {
                    path: Some(path.into()),
                    ..Default::default()
                }
            }

            #[cfg(feature = "egui-adapter-036")]
            fn dropped(path: &str) -> egui::DroppedFileHandle {
                #[derive(Debug)]
                struct File(std::path::PathBuf);
                impl egui::DroppedFile for File {
                    fn path(&self) -> &std::path::Path {
                        &self.0
                    }
                    fn bytes(&self) -> Result<Vec<u8>, String> {
                        Ok(Vec::new())
                    }
                }
                std::sync::Arc::new(File(path.into()))
            }

            /// The single event `event` translates to with the viewport at
            /// `origin`, if any.
            fn one(event: &Event, origin: glam::Vec2) -> Option<ViewportEvent> {
                let mut out = Vec::new();
                translate(event, &AdapterCtx::new(1.0, origin), &mut out);
                assert!(out.len() <= 1, "{out:?}");
                out.pop()
            }

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
                let ev = one(
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
                let press = one(
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

                let release = one(
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
                let ev = one(
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
                    let ev = one(
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
                match one(&Event::Zoom(1.0), glam::Vec2::ZERO) {
                    Some(ViewportEvent::TrackpadPinch(d)) => assert!(d.abs() < 1e-6),
                    other => panic!("expected TrackpadPinch, got {other:?}"),
                }
                // Spreading (factor > 1) zooms in, which is a positive delta.
                match one(&Event::Zoom(std::f32::consts::E), glam::Vec2::ZERO) {
                    Some(ViewportEvent::TrackpadPinch(d)) => assert!((d - 1.0).abs() < 1e-5),
                    other => panic!("expected TrackpadPinch, got {other:?}"),
                }
                // Pinching together is negative.
                match one(&Event::Zoom(0.5), glam::Vec2::ZERO) {
                    Some(ViewportEvent::TrackpadPinch(d)) => assert!(d < 0.0),
                    other => panic!("expected TrackpadPinch, got {other:?}"),
                }
                // A zero or negative factor has no logarithm; drop it rather than
                // feed the resolver an infinity.
                assert!(one(&Event::Zoom(0.0), glam::Vec2::ZERO).is_none());
            }

            #[test]
            fn rotation_is_negated_onto_the_winit_convention() {
                // egui's positive angle is winit's negative one. Getting this wrong
                // makes twist spin opposite ways depending on the host, which is the
                // whole reason the adapter converts rather than passing through.
                match one(&Event::Rotate(0.25), glam::Vec2::ZERO) {
                    Some(ViewportEvent::TrackpadRotate(r)) => assert!((r + 0.25).abs() < 1e-6),
                    other => panic!("expected TrackpadRotate, got {other:?}"),
                }
            }

            #[test]
            fn a_touch_keeps_its_id_and_is_made_viewport_local() {
                let ev = one(
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
                    one(&Event::PointerGone, glam::Vec2::ZERO),
                    Some(ViewportEvent::PointerLeft)
                ));
            }

            // AZERTY: the key in the US W position types Z. The binding follows the
            // position, as through winit.
            #[test]
            fn key_event_uses_the_physical_key() {
                let ev = one(
                    &Event::Key {
                        key: E::Z,
                        physical_key: Some(E::W),
                        pressed: true,
                        repeat: false,
                        modifiers: Modifiers::default(),
                    },
                    glam::Vec2::ZERO,
                );
                assert!(matches!(
                    ev,
                    Some(ViewportEvent::Key {
                        key: KeyCode::W,
                        logical: Some('z'),
                        ..
                    })
                ));
            }

            #[test]
            fn modifiers_keep_cmd_apart_from_ctrl() {
                let cmd = modifiers_from_egui(&Modifiers {
                    mac_cmd: true,
                    command: true,
                    ..Default::default()
                });
                assert!(cmd.super_key && !cmd.ctrl);
                assert_eq!(logical_char(E::ArrowLeft), None);
                assert_eq!(logical_char(E::Num1), Some('1'));
            }

            #[test]
            fn key_event_carries_state_and_repeat() {
                let ev = one(
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
                    Some(ViewportEvent::Key {
                        key, state, repeat, ..
                    }) => {
                        assert_eq!(key, KeyCode::W);
                        assert_eq!(state, ButtonState::Pressed);
                        assert!(repeat);
                    }
                    other => panic!("expected Key, got {other:?}"),
                }
            }

            /// Run one egui frame with `events`, laying out the viewport widget at
            /// `VIEWPORT` and a second widget beside it, which takes keyboard focus
            /// when `focus_field` is set. Returns what `ownership` reports.
            fn frame(
                ctx: &egui::Context,
                events: Vec<Event>,
                session_active: bool,
                focus_field: bool,
            ) -> InputOwnership {
                let raw = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        Pos2::ZERO,
                        egui::vec2(400.0, 300.0),
                    )),
                    events,
                    ..Default::default()
                };
                let mut out = InputOwnership::NONE;
                let mut body = |ui: &mut egui::Ui| {
                    let field = ui.interact(FIELD, egui::Id::new("field"), egui::Sense::click());
                    if focus_field {
                        field.request_focus();
                    }
                    let response = ui.interact(
                        VIEWPORT,
                        egui::Id::new("viewport"),
                        egui::Sense::click_and_drag(),
                    );
                    out = ownership(&response, ui.ctx(), BIG, session_active);
                };
                #[cfg(feature = "egui-adapter")]
                let _ = ctx.run(raw, |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| body(ui));
                });
                #[cfg(not(feature = "egui-adapter"))]
                let _ = ctx.run_ui(raw, |ui| body(ui));
                out
            }

            const VIEWPORT: egui::Rect = egui::Rect {
                min: Pos2::new(100.0, 100.0),
                max: Pos2::new(200.0, 200.0),
            };
            /// A larger rect around the widget, so a pointer in it but off the
            /// widget reads as inside the viewport with the hover elsewhere.
            const BIG: egui::Rect = egui::Rect {
                min: Pos2::new(50.0, 50.0),
                max: Pos2::new(250.0, 250.0),
            };
            const FIELD: egui::Rect = egui::Rect {
                min: Pos2::new(300.0, 0.0),
                max: Pos2::new(380.0, 20.0),
            };

            fn moved(x: f32, y: f32) -> Event {
                Event::PointerMoved(Pos2::new(x, y))
            }

            fn primary(x: f32, y: f32, pressed: bool) -> Event {
                Event::PointerButton {
                    pos: Pos2::new(x, y),
                    button: PointerButton::Primary,
                    pressed,
                    modifiers: Modifiers::default(),
                }
            }

            #[test]
            fn ownership_follows_hover() {
                let ctx = egui::Context::default();
                frame(&ctx, vec![moved(150.0, 150.0)], false, false);
                let o = frame(&ctx, vec![], false, false);
                assert_eq!(o, InputOwnership::full_window(), "hovered");

                frame(&ctx, vec![moved(70.0, 70.0)], false, false);
                let o = frame(&ctx, vec![], false, false);
                assert_eq!(
                    o.pointer,
                    PointerOwnership::Inside,
                    "in the rect, off the widget"
                );
                assert_eq!(o.keyboard, KeyboardOwnership::Elsewhere);

                frame(&ctx, vec![moved(390.0, 290.0)], false, false);
                let o = frame(&ctx, vec![], false, false);
                assert_eq!(o, InputOwnership::NONE, "outside");
            }

            #[test]
            fn a_session_or_a_drag_grabs_the_pointer() {
                let ctx = egui::Context::default();
                frame(&ctx, vec![moved(390.0, 290.0)], false, false);
                let o = frame(&ctx, vec![], true, false);
                assert_eq!(o.pointer, PointerOwnership::Grabbed, "session");
                assert_eq!(o.keyboard, KeyboardOwnership::Owned);

                frame(&ctx, vec![moved(150.0, 150.0)], false, false);
                frame(&ctx, vec![primary(150.0, 150.0, true)], false, false);
                frame(&ctx, vec![moved(170.0, 150.0)], false, false);
                // Dragged past the widget's edge and out of the rect: still ours.
                let o = frame(&ctx, vec![moved(390.0, 290.0)], false, false);
                assert_eq!(o.pointer, PointerOwnership::Grabbed, "drag");
            }

            #[test]
            fn a_focused_text_field_keeps_the_keyboard() {
                let ctx = egui::Context::default();
                frame(&ctx, vec![moved(150.0, 150.0)], false, true);
                let o = frame(&ctx, vec![], false, false);
                assert_eq!(o.pointer, PointerOwnership::Owned);
                assert_eq!(o.keyboard, KeyboardOwnership::Elsewhere);
            }

            #[test]
            fn begin_frame_reports_modifiers_focus_loss_and_drops() {
                let mut input = egui::InputState::default();
                input.modifiers = Modifiers {
                    shift: true,
                    ..Default::default()
                };
                let mut out = Vec::new();
                begin_frame(&input, &AdapterCtx::default(), &mut out);
                assert!(matches!(
                    out.as_slice(),
                    [ViewportEvent::ModifiersChanged(m)] if m.shift && !m.ctrl
                ));

                input.events.push(Event::WindowFocused(false));
                input.raw.dropped_files.push(dropped("model.obj"));
                out.clear();
                begin_frame(&input, &AdapterCtx::default(), &mut out);
                assert!(matches!(
                    out.as_slice(),
                    [
                        ViewportEvent::ModifiersChanged(_),
                        ViewportEvent::FocusLost,
                        ViewportEvent::FileDropped(p),
                    ] if p.as_os_str() == "model.obj"
                ));

                // Focus loss is begin_frame's to report, so it is not reported twice.
                out.clear();
                translate(
                    &Event::WindowFocused(false),
                    &AdapterCtx::default(),
                    &mut out,
                );
                assert!(out.is_empty());
            }

            #[test]
            fn positions_are_scaled_then_made_viewport_local() {
                let mut out = Vec::new();
                translate(
                    &Event::PointerMoved(Pos2::new(100.0, 80.0)),
                    &AdapterCtx::new(2.0, glam::Vec2::new(10.0, 20.0)),
                    &mut out,
                );
                assert!(matches!(
                    out.as_slice(),
                    [ViewportEvent::PointerMoved { position }]
                        if *position == glam::Vec2::new(40.0, 20.0)
                ));
            }

            /// Every `ViewportEvent` variant is either produced from egui here
            /// (by `translate` or `begin_frame`) or listed as one egui cannot
            /// produce. A new variant fails this until it is one or the other.
            #[test]
            fn every_viewport_event_is_carried_or_listed() {
                /// Variants egui has no event for, with the reason.
                const UNREPRESENTABLE: &[(&str, &str)] = &[
                    ("TrackpadPan", "egui folds a trackpad pan into MouseWheel"),
                    ("ThemeChanged", "egui reports the theme as state, not a change"),
                    ("Occluded", "egui reports occlusion as viewport state"),
                    ("FileHovered", "egui reports hovered files as state, with no cancel"),
                    ("FileHoverCancelled", "egui reports hovered files as state"),
                ];

                let mut input = egui::InputState::default();
                input.events.push(Event::WindowFocused(false));
                input.raw.dropped_files.push(dropped("a"));
                let ctx = AdapterCtx::default();
                let mut out = Vec::new();
                begin_frame(&input, &ctx, &mut out);
                let events = [
                    Event::PointerMoved(Pos2::new(1.0, 2.0)),
                    Event::MouseMoved(egui::vec2(1.0, 1.0)),
                    Event::PointerButton {
                        pos: Pos2::ZERO,
                        button: PointerButton::Primary,
                        pressed: true,
                        modifiers: Modifiers::default(),
                    },
                    Event::PointerGone,
                    Event::MouseWheel {
                        unit: MouseWheelUnit::Line,
                        delta: egui::vec2(0.0, 1.0),
                        #[cfg(any(feature = "egui-adapter-035", feature = "egui-adapter-036"))]
                        phase: egui::TouchPhase::Move,
                        modifiers: Modifiers::default(),
                    },
                    Event::Zoom(1.5),
                    Event::Rotate(0.5),
                    Event::Touch {
                        device_id: egui::TouchDeviceId(0),
                        id: egui::TouchId(1),
                        phase: egui::TouchPhase::Start,
                        pos: Pos2::ZERO,
                        force: None,
                    },
                    // QWERTZ: the key in the US Y position is labelled Z.
                    Event::Key {
                        key: E::Z,
                        physical_key: Some(E::Y),
                        pressed: true,
                        repeat: false,
                        modifiers: Modifiers::default(),
                    },
                    Event::Text("z".into()),
                    Event::Ime(egui::ImeEvent::Commit("x".into())),
                ];
                for ev in &events {
                    translate(ev, &ctx, &mut out);
                }

                let produced: std::collections::HashSet<&str> =
                    out.iter().map(ViewportEvent::variant_name).collect();
                for name in ViewportEvent::VARIANTS {
                    let listed = UNREPRESENTABLE.iter().any(|(n, _)| n == name);
                    assert!(
                        produced.contains(name) != listed,
                        "{name}: produced {}, listed {listed}",
                        produced.contains(name)
                    );
                }
                let key = out
                    .iter()
                    .find(|e| matches!(e, ViewportEvent::Key { .. }))
                    .unwrap();
                assert!(matches!(
                    key,
                    ViewportEvent::Key {
                        key: KeyCode::Y,
                        logical: Some('z'),
                        ..
                    }
                ));
            }

            #[test]
            fn text_becomes_one_character_per_char() {
                let mut out = Vec::new();
                translate(&Event::Text("2.5\t".into()), &AdapterCtx::default(), &mut out);
                let chars: Vec<char> = out
                    .iter()
                    .map(|ev| match ev {
                        ViewportEvent::Character(c) => *c,
                        other => panic!("expected Character, got {other:?}"),
                    })
                    .collect();
                assert_eq!(chars, vec!['2', '.', '5']);
            }

            #[test]
            fn into_appends_the_translated_event() {
                let mut out = vec![ViewportEvent::PointerLeft];
                translate(&Event::PointerGone, &AdapterCtx::default(), &mut out);
                translate(&Event::Copy, &AdapterCtx::default(), &mut out);
                assert_eq!(out.len(), 2);
                assert!(matches!(out[1], ViewportEvent::PointerLeft));
            }
        }
    };
}

#[cfg(feature = "egui-adapter")]
mod v033 {
    egui_adapter!(egui033);
}
#[cfg(feature = "egui-adapter")]
#[allow(deprecated)]
pub use v033::{begin_frame, from_egui, modifiers_from_egui, ownership, translate};

#[cfg(feature = "egui-adapter-035")]
mod v035 {
    egui_adapter!(egui035);
}
#[cfg(feature = "egui-adapter-035")]
#[allow(deprecated)]
pub use v035::{begin_frame, from_egui, modifiers_from_egui, ownership, translate};

#[cfg(feature = "egui-adapter-036")]
mod v036 {
    egui_adapter!(egui036);
}
#[cfg(feature = "egui-adapter-036")]
#[allow(deprecated)]
pub use v036::{begin_frame, from_egui, modifiers_from_egui, ownership, translate};
