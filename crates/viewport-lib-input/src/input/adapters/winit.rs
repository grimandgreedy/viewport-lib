//! winit -> [`ViewportEvent`] translation.

use super::AdapterCtx;
use crate::input::{
    ButtonState, ImeEvent, InputOwnership, KeyCode, KeyboardOwnership, Modifiers, MouseButton,
    PointerOwnership, ScrollUnits, Theme, TouchId, TouchPhase, ViewportEvent,
};

/// Who owns the input for a viewport in a winit window, from what a winit host
/// knows.
///
/// - `window_focused`: the window has keyboard focus (`WindowEvent::Focused`).
/// - `cursor_in_rect`: the cursor is inside the viewport's rect.
/// - `captured`: the host has grabbed the pointer, as for mouselook.
/// - `ui_claims_pointer`: the host's own UI drawn over the viewport has the
///   pointer this frame (a button, a panel, a tool handle).
/// - `ui_claims_keyboard`: the host's own UI has the keyboard (a text field).
///
/// The pointer is [`Grabbed`](PointerOwnership::Grabbed) while captured,
/// [`Owned`](PointerOwnership::Owned) when the cursor is in the rect of a
/// focused window and the UI does not claim it,
/// [`Inside`](PointerOwnership::Inside) when the cursor is in the rect
/// otherwise, and [`Elsewhere`](PointerOwnership::Elsewhere) outside it. The
/// keyboard is the viewport's while the window is focused and the UI does not
/// claim it.
///
/// A camera drag in flight needs no argument: the resolver keeps its own held
/// press's events when the cursor crosses the UI or leaves the rect. `captured`
/// is for pointer capture only, where there is no rect to leave and
/// [`ViewportEvent::PointerLeft`] must not end the session.
pub fn ownership(
    window_focused: bool,
    cursor_in_rect: bool,
    captured: bool,
    ui_claims_pointer: bool,
    ui_claims_keyboard: bool,
) -> InputOwnership {
    let pointer = if captured {
        PointerOwnership::Grabbed
    } else if window_focused && cursor_in_rect && !ui_claims_pointer {
        PointerOwnership::Owned
    } else if cursor_in_rect {
        PointerOwnership::Inside
    } else {
        PointerOwnership::Elsewhere
    };
    let keyboard = if window_focused && !ui_claims_keyboard {
        KeyboardOwnership::Owned
    } else {
        KeyboardOwnership::Elsewhere
    };
    InputOwnership { pointer, keyboard }
}

/// Translate a winit [`WindowEvent`](::winit::event::WindowEvent), appending what
/// it means for the viewport to `out`.
///
/// Covers window events: pointer, buttons, wheel, keys and the text they type,
/// modifiers, focus loss, trackpad gestures, touch, theme, occlusion, file
/// drag-and-drop and input method composition. A key press is followed by one
/// [`ViewportEvent::Character`] per character it typed, control characters left
/// out. Events the viewport has no use for append nothing. Raw relative motion
/// for mouselook arrives as a device event: see [`translate_device`].
///
/// winit reports positions in physical pixels; they are divided by
/// `ctx.scale_factor` (`window.scale_factor()`) and offset by
/// `ctx.viewport_origin`, so they come out in viewport-local logical points.
pub fn translate(
    event: &::winit::event::WindowEvent,
    ctx: &AdapterCtx,
    out: &mut Vec<ViewportEvent>,
) {
    out.extend(translate_one(event, ctx));
    if let ::winit::event::WindowEvent::KeyboardInput { event: key, .. } = event {
        typed_text(key.state, key.text.as_deref(), out);
    }
}

/// Translate a winit [`DeviceEvent`](::winit::event::DeviceEvent), appending what
/// it means for the viewport to `out`.
///
/// Raw relative pointer motion arrives as a device event, not a window event, so
/// a runner routes its `device_event` callback through here (typically to the
/// focused window) to get [`ViewportEvent::RawMotion`] for mouselook.
pub fn translate_device(event: &::winit::event::DeviceEvent, out: &mut Vec<ViewportEvent>) {
    use ::winit::event::DeviceEvent;
    if let DeviceEvent::MouseMotion { delta } = event {
        out.push(ViewportEvent::RawMotion {
            delta: glam::Vec2::new(delta.0 as f32, delta.1 as f32),
        });
    }
}

/// Translate a winit [`WindowEvent`](::winit::event::WindowEvent) into one
/// [`ViewportEvent`], or `None` for events the viewport does not consume. A key
/// press's typed text is not included.
#[deprecated(note = "use translate, which also emits the text a key press types")]
pub fn from_winit(event: &::winit::event::WindowEvent, scale_factor: f32) -> Option<ViewportEvent> {
    translate_one(event, &AdapterCtx::new(scale_factor, glam::Vec2::ZERO))
}

/// The one event a window event maps to, without the text a key types.
fn translate_one(event: &::winit::event::WindowEvent, ctx: &AdapterCtx) -> Option<ViewportEvent> {
    use ::winit::event::{MouseScrollDelta, WindowEvent};

    let inv_scale = 1.0 / ctx.scale_factor.max(0.001);
    match event {
        WindowEvent::CursorMoved { position, .. } => Some(ViewportEvent::PointerMoved {
            position: ctx.local(position.x as f32, position.y as f32),
        }),
        WindowEvent::CursorLeft { .. } => Some(ViewportEvent::PointerLeft),
        WindowEvent::MouseInput { state, button, .. } => Some(ViewportEvent::MouseButton {
            button: map_button(*button)?,
            state: map_state(*state),
        }),
        WindowEvent::MouseWheel { delta, .. } => {
            let (delta, units) = match delta {
                MouseScrollDelta::LineDelta(x, y) => (glam::Vec2::new(*x, *y), ScrollUnits::Lines),
                MouseScrollDelta::PixelDelta(p) => {
                    (glam::Vec2::new(p.x as f32, p.y as f32), ScrollUnits::Pixels)
                }
            };
            Some(ViewportEvent::Wheel { delta, units })
        }
        WindowEvent::ModifiersChanged(mods) => Some(ViewportEvent::ModifiersChanged(
            modifiers_from_winit(mods.state()),
        )),
        WindowEvent::Focused(false) => Some(ViewportEvent::FocusLost),
        WindowEvent::KeyboardInput { event, .. } => key_event(
            &event.physical_key,
            logical_char(event),
            event.state,
            event.repeat,
        ),
        WindowEvent::ThemeChanged(theme) => Some(ViewportEvent::ThemeChanged(match theme {
            ::winit::window::Theme::Light => Theme::Light,
            ::winit::window::Theme::Dark => Theme::Dark,
        })),
        WindowEvent::Occluded(occluded) => Some(ViewportEvent::Occluded(*occluded)),
        // winit reports the rotation gesture in degrees; the viewport convention is
        // radians (see ViewportEvent::TrackpadRotate).
        WindowEvent::RotationGesture { delta, .. } => {
            Some(ViewportEvent::TrackpadRotate(delta.to_radians()))
        }
        WindowEvent::PinchGesture { delta, .. } => {
            Some(ViewportEvent::TrackpadPinch(*delta as f32))
        }
        WindowEvent::PanGesture { delta, .. } => Some(ViewportEvent::TrackpadPan(
            glam::Vec2::new(delta.x, delta.y) * inv_scale,
        )),
        // Touch locations are physical, like the cursor's, so they scale the same way.
        WindowEvent::Touch(touch) => Some(ViewportEvent::Touch {
            id: TouchId(touch.id),
            phase: map_touch_phase(touch.phase),
            position: ctx.local(touch.location.x as f32, touch.location.y as f32),
        }),
        WindowEvent::DroppedFile(path) => Some(ViewportEvent::FileDropped(path.clone())),
        WindowEvent::HoveredFile(path) => Some(ViewportEvent::FileHovered(path.clone())),
        WindowEvent::HoveredFileCancelled => Some(ViewportEvent::FileHoverCancelled),
        WindowEvent::Ime(ime) => Some(ViewportEvent::Ime(match ime {
            ::winit::event::Ime::Enabled => ImeEvent::Enabled,
            ::winit::event::Ime::Preedit(text, cursor) => ImeEvent::Preedit {
                text: text.clone(),
                cursor: *cursor,
            },
            ::winit::event::Ime::Commit(text) => ImeEvent::Commit(text.clone()),
            ::winit::event::Ime::Disabled => ImeEvent::Disabled,
        })),
        _ => None,
    }
}

/// A key event from its parts, or `None` for a physical key with no `KeyCode`.
/// Split out of `translate_one` because winit's `KeyEvent` cannot be built
/// outside winit, so this is what the tests reach.
fn key_event(
    physical: &::winit::keyboard::PhysicalKey,
    logical: Option<char>,
    state: ::winit::event::ElementState,
    repeat: bool,
) -> Option<ViewportEvent> {
    Some(ViewportEvent::Key {
        key: map_key(physical)?,
        state: map_state(state),
        repeat,
        logical,
    })
}

/// The characters a key press typed, control characters left out.
fn typed_text(
    state: ::winit::event::ElementState,
    text: Option<&str>,
    out: &mut Vec<ViewportEvent>,
) {
    if state != ::winit::event::ElementState::Pressed {
        return;
    }
    if let Some(text) = text {
        out.extend(
            text.chars()
                .filter(|c| !c.is_control())
                .map(ViewportEvent::Character),
        );
    }
}

/// Translate winit's modifier state into [`Modifiers`], with Ctrl and Super
/// (Cmd, the Windows key) kept apart.
pub fn modifiers_from_winit(state: ::winit::keyboard::ModifiersState) -> Modifiers {
    Modifiers::NONE
        .with_alt(state.alt_key())
        .with_shift(state.shift_key())
        .with_ctrl(state.control_key())
        .with_super(state.super_key())
}

/// The character the layout gives a key, unshifted and lower case. Where winit
/// offers the key without modifiers (desktop platforms) it is used; elsewhere the
/// logical key is lowercased, which is right for letters but leaves a shifted
/// symbol (`!`) as it is.
fn logical_char(event: &::winit::event::KeyEvent) -> Option<char> {
    #[cfg(any(
        target_os = "windows",
        target_os = "macos",
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
    ))]
    let key = {
        use ::winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
        event.key_without_modifiers()
    };
    #[cfg(not(any(
        target_os = "windows",
        target_os = "macos",
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
    )))]
    let key = event.logical_key.clone();
    single_lower_char(&key)
}

fn single_lower_char(key: &::winit::keyboard::Key) -> Option<char> {
    let ::winit::keyboard::Key::Character(text) = key else {
        return None;
    };
    let mut chars = text.chars();
    let c = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    let mut lower = c.to_lowercase();
    let l = lower.next()?;
    lower.next().is_none().then_some(l)
}

fn map_state(state: ::winit::event::ElementState) -> ButtonState {
    match state {
        ::winit::event::ElementState::Pressed => ButtonState::Pressed,
        ::winit::event::ElementState::Released => ButtonState::Released,
    }
}

fn map_touch_phase(phase: ::winit::event::TouchPhase) -> TouchPhase {
    use ::winit::event::TouchPhase as W;
    match phase {
        W::Started => TouchPhase::Started,
        W::Moved => TouchPhase::Moved,
        W::Ended => TouchPhase::Ended,
        W::Cancelled => TouchPhase::Cancelled,
    }
}

fn map_button(button: ::winit::event::MouseButton) -> Option<MouseButton> {
    match button {
        ::winit::event::MouseButton::Left => Some(MouseButton::Left),
        ::winit::event::MouseButton::Right => Some(MouseButton::Right),
        ::winit::event::MouseButton::Middle => Some(MouseButton::Middle),
        ::winit::event::MouseButton::Back => Some(MouseButton::Back),
        ::winit::event::MouseButton::Forward => Some(MouseButton::Forward),
        ::winit::event::MouseButton::Other(n) => Some(MouseButton::Other(n)),
    }
}

fn map_key(key: &::winit::keyboard::PhysicalKey) -> Option<KeyCode> {
    use ::winit::keyboard::{KeyCode as W, PhysicalKey};
    let PhysicalKey::Code(code) = key else {
        return None;
    };
    let k = match code {
        W::KeyA => KeyCode::A,
        W::KeyB => KeyCode::B,
        W::KeyC => KeyCode::C,
        W::KeyD => KeyCode::D,
        W::KeyE => KeyCode::E,
        W::KeyF => KeyCode::F,
        W::KeyG => KeyCode::G,
        W::KeyH => KeyCode::H,
        W::KeyI => KeyCode::I,
        W::KeyJ => KeyCode::J,
        W::KeyK => KeyCode::K,
        W::KeyL => KeyCode::L,
        W::KeyM => KeyCode::M,
        W::KeyN => KeyCode::N,
        W::KeyO => KeyCode::O,
        W::KeyP => KeyCode::P,
        W::KeyQ => KeyCode::Q,
        W::KeyR => KeyCode::R,
        W::KeyS => KeyCode::S,
        W::KeyT => KeyCode::T,
        W::KeyU => KeyCode::U,
        W::KeyV => KeyCode::V,
        W::KeyW => KeyCode::W,
        W::KeyX => KeyCode::X,
        W::KeyY => KeyCode::Y,
        W::KeyZ => KeyCode::Z,
        W::Digit0 => KeyCode::Num0,
        W::Digit1 => KeyCode::Num1,
        W::Digit2 => KeyCode::Num2,
        W::Digit3 => KeyCode::Num3,
        W::Digit4 => KeyCode::Num4,
        W::Digit5 => KeyCode::Num5,
        W::Digit6 => KeyCode::Num6,
        W::Digit7 => KeyCode::Num7,
        W::Digit8 => KeyCode::Num8,
        W::Digit9 => KeyCode::Num9,
        W::Escape => KeyCode::Escape,
        W::Enter => KeyCode::Enter,
        W::Tab => KeyCode::Tab,
        W::Space => KeyCode::Space,
        W::Backspace => KeyCode::Backspace,
        W::ArrowUp => KeyCode::Up,
        W::ArrowDown => KeyCode::Down,
        W::ArrowLeft => KeyCode::Left,
        W::ArrowRight => KeyCode::Right,
        W::ShiftLeft => KeyCode::LeftShift,
        W::ShiftRight => KeyCode::RightShift,
        W::Minus => KeyCode::Minus,
        W::Comma => KeyCode::Comma,
        W::Period => KeyCode::Period,
        W::Slash => KeyCode::Slash,
        W::Backquote => KeyCode::Backtick,
        W::Backslash => KeyCode::Backslash,
        W::BracketLeft => KeyCode::LeftBracket,
        W::BracketRight => KeyCode::RightBracket,
        W::Semicolon => KeyCode::Semicolon,
        W::Quote => KeyCode::Apostrophe,
        W::Equal => KeyCode::Equals,
        W::Delete => KeyCode::Delete,
        W::Insert => KeyCode::Insert,
        W::Home => KeyCode::Home,
        W::End => KeyCode::End,
        W::PageUp => KeyCode::PageUp,
        W::PageDown => KeyCode::PageDown,
        W::CapsLock => KeyCode::CapsLock,
        W::ControlLeft => KeyCode::LeftCtrl,
        W::ControlRight => KeyCode::RightCtrl,
        W::AltLeft => KeyCode::LeftAlt,
        W::AltRight => KeyCode::RightAlt,
        W::SuperLeft => KeyCode::LeftSuper,
        W::SuperRight => KeyCode::RightSuper,
        W::NumLock => KeyCode::NumLock,
        W::PrintScreen => KeyCode::PrintScreen,
        W::Pause => KeyCode::Pause,
        W::ScrollLock => KeyCode::ScrollLock,
        W::F1 => KeyCode::F1,
        W::F2 => KeyCode::F2,
        W::F3 => KeyCode::F3,
        W::F4 => KeyCode::F4,
        W::F5 => KeyCode::F5,
        W::F6 => KeyCode::F6,
        W::F7 => KeyCode::F7,
        W::F8 => KeyCode::F8,
        W::F9 => KeyCode::F9,
        W::F10 => KeyCode::F10,
        W::F11 => KeyCode::F11,
        W::F12 => KeyCode::F12,
        W::F13 => KeyCode::F13,
        W::F14 => KeyCode::F14,
        W::F15 => KeyCode::F15,
        W::F16 => KeyCode::F16,
        W::F17 => KeyCode::F17,
        W::F18 => KeyCode::F18,
        W::F19 => KeyCode::F19,
        W::F20 => KeyCode::F20,
        W::F21 => KeyCode::F21,
        W::F22 => KeyCode::F22,
        W::F23 => KeyCode::F23,
        W::F24 => KeyCode::F24,
        W::Numpad0 => KeyCode::Numpad0,
        W::Numpad1 => KeyCode::Numpad1,
        W::Numpad2 => KeyCode::Numpad2,
        W::Numpad3 => KeyCode::Numpad3,
        W::Numpad4 => KeyCode::Numpad4,
        W::Numpad5 => KeyCode::Numpad5,
        W::Numpad6 => KeyCode::Numpad6,
        W::Numpad7 => KeyCode::Numpad7,
        W::Numpad8 => KeyCode::Numpad8,
        W::Numpad9 => KeyCode::Numpad9,
        W::NumpadAdd => KeyCode::NumpadAdd,
        W::NumpadSubtract => KeyCode::NumpadSubtract,
        W::NumpadMultiply => KeyCode::NumpadMultiply,
        W::NumpadDivide => KeyCode::NumpadDivide,
        W::NumpadDecimal => KeyCode::NumpadDecimal,
        W::NumpadEnter => KeyCode::NumpadEnter,
        _ => return None,
    };
    Some(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::winit::event::{ElementState, MouseButton as WinitMouseButton};

    /// The single event `event` translates to, if any.
    fn one(event: &::winit::event::WindowEvent) -> Option<ViewportEvent> {
        let mut out = Vec::new();
        translate(event, &AdapterCtx::default(), &mut out);
        assert!(out.len() <= 1, "{out:?}");
        out.pop()
    }
    use ::winit::keyboard::{KeyCode as Wk, PhysicalKey};

    // One pairs list, expanded into both the full list of library keys and the
    // exhaustive library-key -> winit-key mapping used to test coverage. Because
    // the generated `match` has no wildcard, adding a `KeyCode` variant fails to
    // compile here until its winit key is listed: that is the tripwire that keeps
    // the adapter from silently dropping a newly defined key.
    macro_rules! key_pairs {
        ($($lib:ident => $winit:ident),* $(,)?) => {
            const ALL_KEYCODES: &[KeyCode] = &[$(KeyCode::$lib),*];
            /// The winit key that must translate to each library key.
            fn winit_key_for(k: KeyCode) -> Wk {
                match k { $(KeyCode::$lib => Wk::$winit),* }
            }
        };
    }

    key_pairs! {
        A => KeyA, B => KeyB, C => KeyC, D => KeyD, E => KeyE, F => KeyF, G => KeyG,
        H => KeyH, I => KeyI, J => KeyJ, K => KeyK, L => KeyL, M => KeyM, N => KeyN,
        O => KeyO, P => KeyP, Q => KeyQ, R => KeyR, S => KeyS, T => KeyT, U => KeyU,
        V => KeyV, W => KeyW, X => KeyX, Y => KeyY, Z => KeyZ,
        Num0 => Digit0, Num1 => Digit1, Num2 => Digit2, Num3 => Digit3, Num4 => Digit4,
        Num5 => Digit5, Num6 => Digit6, Num7 => Digit7, Num8 => Digit8, Num9 => Digit9,
        Tab => Tab, Enter => Enter, Escape => Escape, Backtick => Backquote,
        Backspace => Backspace, Comma => Comma, Period => Period,
        LeftBracket => BracketLeft, RightBracket => BracketRight, Slash => Slash,
        LeftShift => ShiftLeft, RightShift => ShiftRight, Space => Space,
        Up => ArrowUp, Down => ArrowDown, Left => ArrowLeft, Right => ArrowRight,
        F1 => F1, F2 => F2, F3 => F3, F4 => F4, F5 => F5, F6 => F6, F7 => F7,
        F8 => F8, F9 => F9, F10 => F10, F11 => F11, F12 => F12, F13 => F13,
        F14 => F14, F15 => F15, F16 => F16, F17 => F17, F18 => F18, F19 => F19,
        F20 => F20, F21 => F21, F22 => F22, F23 => F23, F24 => F24,
        Semicolon => Semicolon, Apostrophe => Quote, Backslash => Backslash,
        Minus => Minus, Equals => Equal, LeftCtrl => ControlLeft,
        RightCtrl => ControlRight, LeftAlt => AltLeft, RightAlt => AltRight,
        LeftSuper => SuperLeft, RightSuper => SuperRight, CapsLock => CapsLock,
        Delete => Delete, Insert => Insert, Home => Home, End => End,
        PageUp => PageUp, PageDown => PageDown,
        Numpad0 => Numpad0, Numpad1 => Numpad1, Numpad2 => Numpad2, Numpad3 => Numpad3,
        Numpad4 => Numpad4, Numpad5 => Numpad5, Numpad6 => Numpad6, Numpad7 => Numpad7,
        Numpad8 => Numpad8, Numpad9 => Numpad9, NumpadAdd => NumpadAdd,
        NumpadSubtract => NumpadSubtract, NumpadMultiply => NumpadMultiply,
        NumpadDivide => NumpadDivide, NumpadDecimal => NumpadDecimal,
        NumpadEnter => NumpadEnter, NumLock => NumLock, PrintScreen => PrintScreen,
        Pause => Pause, ScrollLock => ScrollLock,
    }

    /// Every key the library defines must translate from some winit key. This is
    /// the guard against the "key not in the defined set" class: a defined key
    /// the adapter forgets to map (F13-F24 were exactly this) fails here because
    /// `map_key` returns `None` for it.
    #[test]
    fn ownership_follows_focus_cursor_and_claims() {
        use KeyboardOwnership as K;
        use PointerOwnership as P;
        // (focused, in rect, captured, ui pointer, ui keyboard) -> (pointer, keyboard)
        let cases = [
            ((true, true, false, false, false), (P::Owned, K::Owned)),
            ((true, false, false, false, false), (P::Elsewhere, K::Owned)),
            ((true, true, false, true, false), (P::Inside, K::Owned)),
            (
                (false, true, false, false, false),
                (P::Inside, K::Elsewhere),
            ),
            (
                (false, false, false, false, false),
                (P::Elsewhere, K::Elsewhere),
            ),
            ((true, true, false, false, true), (P::Owned, K::Elsewhere)),
            ((true, false, true, false, false), (P::Grabbed, K::Owned)),
            ((true, true, true, true, true), (P::Grabbed, K::Elsewhere)),
            (
                (false, false, true, false, false),
                (P::Grabbed, K::Elsewhere),
            ),
        ];
        for ((focused, inside, captured, ui_p, ui_k), (pointer, keyboard)) in cases {
            let o = ownership(focused, inside, captured, ui_p, ui_k);
            assert_eq!(
                (o.pointer, o.keyboard),
                (pointer, keyboard),
                "focused {focused}, in rect {inside}, captured {captured}, ui {ui_p}/{ui_k}"
            );
        }
    }

    #[test]
    fn every_defined_key_maps_from_winit() {
        for &code in ALL_KEYCODES {
            let winit = winit_key_for(code);
            assert_eq!(
                map_key(&PhysicalKey::Code(winit)),
                Some(code),
                "winit {winit:?} should map to KeyCode::{code:?}"
            );
        }
    }

    /// An unmapped physical key is dropped (returns `None`), not mis-mapped. The
    /// `Fn` lock keys are examples winit exposes that the viewport does not use.
    #[test]
    fn unmapped_physical_key_is_none() {
        assert_eq!(map_key(&PhysicalKey::Code(Wk::Fn)), None);
        assert_eq!(map_key(&PhysicalKey::Code(Wk::ContextMenu)), None);
        assert_eq!(
            map_key(&PhysicalKey::Unidentified(
                ::winit::keyboard::NativeKeyCode::Unidentified
            )),
            None
        );
    }

    #[test]
    fn mouse_buttons_map_including_extras() {
        assert_eq!(map_button(WinitMouseButton::Left), Some(MouseButton::Left));
        assert_eq!(
            map_button(WinitMouseButton::Right),
            Some(MouseButton::Right)
        );
        assert_eq!(
            map_button(WinitMouseButton::Middle),
            Some(MouseButton::Middle)
        );
        // Back/forward and hardware extras now map through; the resolver ignores
        // them for gestures, but a consumer can act on them via the raw event stream.
        assert_eq!(map_button(WinitMouseButton::Back), Some(MouseButton::Back));
        assert_eq!(
            map_button(WinitMouseButton::Forward),
            Some(MouseButton::Forward)
        );
        assert_eq!(
            map_button(WinitMouseButton::Other(9)),
            Some(MouseButton::Other(9))
        );
    }

    #[test]
    fn element_state_maps() {
        assert_eq!(map_state(ElementState::Pressed), ButtonState::Pressed);
        assert_eq!(map_state(ElementState::Released), ButtonState::Released);
    }

    #[test]
    fn theme_and_occlusion_translate() {
        use ::winit::event::WindowEvent;
        use ::winit::window::Theme as WinitTheme;

        assert!(matches!(
            one(&WindowEvent::ThemeChanged(WinitTheme::Dark)),
            Some(ViewportEvent::ThemeChanged(Theme::Dark))
        ));
        assert!(matches!(
            one(&WindowEvent::ThemeChanged(WinitTheme::Light)),
            Some(ViewportEvent::ThemeChanged(Theme::Light))
        ));
        assert!(matches!(
            one(&WindowEvent::Occluded(true)),
            Some(ViewportEvent::Occluded(true))
        ));
        assert!(matches!(
            one(&WindowEvent::Occluded(false)),
            Some(ViewportEvent::Occluded(false))
        ));
    }

    #[test]
    fn modifiers_keep_super_apart_from_ctrl() {
        use ::winit::keyboard::ModifiersState as M;
        let cmd = modifiers_from_winit(M::SUPER);
        assert!(cmd.super_key && !cmd.ctrl);
        let ctrl = modifiers_from_winit(M::CONTROL | M::SHIFT);
        assert!(ctrl.ctrl && ctrl.shift && !ctrl.super_key);
    }

    #[test]
    fn logical_char_is_one_lower_case_character() {
        use ::winit::keyboard::{Key, NamedKey};
        assert_eq!(single_lower_char(&Key::Character("Z".into())), Some('z'));
        assert_eq!(single_lower_char(&Key::Character("1".into())), Some('1'));
        assert_eq!(single_lower_char(&Key::Character("ab".into())), None);
        assert_eq!(single_lower_char(&Key::Named(NamedKey::ArrowLeft)), None);
    }

    #[test]
    fn ime_events_translate() {
        use ::winit::event::{Ime, WindowEvent};

        assert!(matches!(
            one(&WindowEvent::Ime(Ime::Enabled)),
            Some(ViewportEvent::Ime(ImeEvent::Enabled))
        ));
        assert!(matches!(
            one(&WindowEvent::Ime(Ime::Preedit("ni".into(), Some((2, 2))))),
            Some(ViewportEvent::Ime(ImeEvent::Preedit { text, cursor: Some((2, 2)) })) if text == "ni"
        ));
        assert!(matches!(
            one(&WindowEvent::Ime(Ime::Commit("\u{65e5}".into()))),
            Some(ViewportEvent::Ime(ImeEvent::Commit(text))) if text == "\u{65e5}"
        ));
        assert!(matches!(
            one(&WindowEvent::Ime(Ime::Disabled)),
            Some(ViewportEvent::Ime(ImeEvent::Disabled))
        ));
    }

    #[test]
    fn file_drag_and_drop_translates() {
        use ::winit::event::WindowEvent;
        use std::path::PathBuf;

        let path = PathBuf::from("/tmp/model.obj");
        assert!(matches!(
            one(&WindowEvent::DroppedFile(path.clone())),
            Some(ViewportEvent::FileDropped(p)) if p == path
        ));
        assert!(matches!(
            one(&WindowEvent::HoveredFile(path.clone())),
            Some(ViewportEvent::FileHovered(p)) if p == path
        ));
        assert!(matches!(
            one(&WindowEvent::HoveredFileCancelled),
            Some(ViewportEvent::FileHoverCancelled)
        ));
    }

    #[test]
    fn touch_phases_map() {
        use ::winit::event::TouchPhase as W;
        let cases = [
            (W::Started, TouchPhase::Started),
            (W::Moved, TouchPhase::Moved),
            (W::Ended, TouchPhase::Ended),
            (W::Cancelled, TouchPhase::Cancelled),
        ];
        for (winit, expected) in cases {
            assert_eq!(map_touch_phase(winit), expected);
        }
    }

    #[test]
    fn raw_device_motion_translates() {
        use ::winit::event::DeviceEvent;
        let mut out = Vec::new();
        translate_device(&DeviceEvent::MouseMotion { delta: (3.0, -4.0) }, &mut out);
        assert!(matches!(
            out.as_slice(),
            [ViewportEvent::RawMotion { delta }] if *delta == glam::Vec2::new(3.0, -4.0)
        ));
        // A device event the viewport does not consume appends nothing.
        translate_device(
            &DeviceEvent::MouseWheel {
                delta: ::winit::event::MouseScrollDelta::LineDelta(0.0, 1.0),
            },
            &mut out,
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn positions_are_scaled_then_made_viewport_local() {
        use ::winit::dpi::PhysicalPosition;
        use ::winit::event::{DeviceId, WindowEvent};
        let ctx = AdapterCtx::new(2.0, glam::Vec2::new(10.0, 20.0));
        let mut out = Vec::new();
        translate(
            &WindowEvent::CursorMoved {
                device_id: DeviceId::dummy(),
                position: PhysicalPosition::new(100.0, 80.0),
            },
            &ctx,
            &mut out,
        );
        assert!(matches!(
            out.as_slice(),
            [ViewportEvent::PointerMoved { position }] if *position == glam::Vec2::new(40.0, 20.0)
        ));
    }

    #[test]
    fn a_key_carries_its_layout_character_and_then_its_text() {
        use ::winit::keyboard::Key;
        // AZERTY: the key in the US Q position types 'a'.
        let mut out = Vec::new();
        out.extend(key_event(
            &PhysicalKey::Code(Wk::KeyQ),
            single_lower_char(&Key::Character("A".into())),
            ElementState::Pressed,
            false,
        ));
        typed_text(ElementState::Pressed, Some("A\r"), &mut out);
        assert!(matches!(
            out.as_slice(),
            [
                ViewportEvent::Key {
                    key: KeyCode::Q,
                    logical: Some('a'),
                    state: ButtonState::Pressed,
                    repeat: false,
                },
                ViewportEvent::Character('A'),
            ]
        ));
        // A release types nothing.
        typed_text(ElementState::Released, Some("a"), &mut out);
        assert_eq!(out.len(), 2);
    }

    /// Every `ViewportEvent` variant is either produced from a winit event here or
    /// listed as one winit cannot produce. A new variant fails this until it is
    /// one or the other.
    #[test]
    fn every_viewport_event_is_carried_or_listed() {
        use ::winit::dpi::PhysicalPosition;
        use ::winit::event::{
            DeviceEvent, DeviceId, Ime, MouseScrollDelta, Touch, TouchPhase as W, WindowEvent,
        };
        use ::winit::keyboard::{Key, ModifiersState};
        use std::path::PathBuf;

        /// Variants no winit event maps to.
        const UNREPRESENTABLE: &[&str] = &[];

        let d = DeviceId::dummy();
        let windows = [
            WindowEvent::CursorMoved {
                device_id: d,
                position: PhysicalPosition::new(1.0, 2.0),
            },
            WindowEvent::MouseInput {
                device_id: d,
                state: ElementState::Pressed,
                button: WinitMouseButton::Left,
            },
            WindowEvent::MouseWheel {
                device_id: d,
                delta: MouseScrollDelta::LineDelta(0.0, 1.0),
                phase: W::Moved,
            },
            WindowEvent::ModifiersChanged(ModifiersState::SHIFT.into()),
            WindowEvent::CursorLeft { device_id: d },
            WindowEvent::Focused(false),
            WindowEvent::RotationGesture {
                device_id: d,
                delta: 1.0,
                phase: W::Moved,
            },
            WindowEvent::PinchGesture {
                device_id: d,
                delta: 0.1,
                phase: W::Moved,
            },
            WindowEvent::PanGesture {
                device_id: d,
                delta: PhysicalPosition::new(1.0, 1.0),
                phase: W::Moved,
            },
            WindowEvent::Touch(Touch {
                device_id: d,
                phase: W::Started,
                location: PhysicalPosition::new(1.0, 1.0),
                force: None,
                id: 1,
            }),
            WindowEvent::ThemeChanged(::winit::window::Theme::Dark),
            WindowEvent::Occluded(true),
            WindowEvent::DroppedFile(PathBuf::from("a")),
            WindowEvent::HoveredFile(PathBuf::from("a")),
            WindowEvent::HoveredFileCancelled,
            WindowEvent::Ime(Ime::Commit("x".into())),
        ];
        let ctx = AdapterCtx::default();
        let mut out = Vec::new();
        for ev in &windows {
            translate(ev, &ctx, &mut out);
        }
        translate_device(&DeviceEvent::MouseMotion { delta: (1.0, 1.0) }, &mut out);
        // winit's KeyEvent cannot be built outside winit; these are the two
        // functions its KeyboardInput arm calls.
        out.extend(key_event(
            &PhysicalKey::Code(Wk::KeyZ),
            single_lower_char(&Key::Character("y".into())),
            ElementState::Pressed,
            false,
        ));
        typed_text(ElementState::Pressed, Some("y"), &mut out);

        let produced: std::collections::HashSet<&str> =
            out.iter().map(ViewportEvent::variant_name).collect();
        for name in ViewportEvent::VARIANTS {
            let listed = UNREPRESENTABLE.contains(name);
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
                logical: Some('y'),
                ..
            }
        ));
    }
}
