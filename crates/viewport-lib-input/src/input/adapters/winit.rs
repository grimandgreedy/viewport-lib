//! winit -> [`ViewportEvent`] translation.

use crate::input::{ButtonState, KeyCode, Modifiers, MouseButton, ScrollUnits, ViewportEvent};

/// Translate a winit [`WindowEvent`](::winit::event::WindowEvent) into a
/// [`ViewportEvent`], or `None` for events the viewport does not consume.
///
/// winit reports cursor positions in physical pixels; `scale_factor`
/// (`window.scale_factor()`) divides them down to logical points, the space the
/// viewport's screen-space math and `viewport_size` use. Pass `1.0` to keep
/// positions in physical pixels. For a window hosting a single full-window
/// viewport the result is already viewport-local; an embedded winit viewport
/// should offset by the viewport rect origin as well.
pub fn from_winit(event: &::winit::event::WindowEvent, scale_factor: f32) -> Option<ViewportEvent> {
    use ::winit::event::{MouseScrollDelta, WindowEvent};

    let inv_scale = 1.0 / scale_factor.max(0.001);
    match event {
        WindowEvent::CursorMoved { position, .. } => Some(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(position.x as f32, position.y as f32) * inv_scale,
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
        WindowEvent::ModifiersChanged(mods) => {
            let s = mods.state();
            Some(ViewportEvent::ModifiersChanged(Modifiers {
                alt: s.alt_key(),
                shift: s.shift_key(),
                ctrl: s.control_key() || s.super_key(),
            }))
        }
        WindowEvent::Focused(false) => Some(ViewportEvent::FocusLost),
        WindowEvent::KeyboardInput { event, .. } => {
            let key = map_key(&event.physical_key)?;
            Some(ViewportEvent::Key {
                key,
                state: map_state(event.state),
                repeat: event.repeat,
            })
        }
        _ => None,
    }
}

fn map_state(state: ::winit::event::ElementState) -> ButtonState {
    match state {
        ::winit::event::ElementState::Pressed => ButtonState::Pressed,
        ::winit::event::ElementState::Released => ButtonState::Released,
    }
}

fn map_button(button: ::winit::event::MouseButton) -> Option<MouseButton> {
    match button {
        ::winit::event::MouseButton::Left => Some(MouseButton::Left),
        ::winit::event::MouseButton::Right => Some(MouseButton::Right),
        ::winit::event::MouseButton::Middle => Some(MouseButton::Middle),
        _ => None,
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
    fn mouse_buttons_map_and_extras_drop() {
        assert_eq!(map_button(WinitMouseButton::Left), Some(MouseButton::Left));
        assert_eq!(
            map_button(WinitMouseButton::Right),
            Some(MouseButton::Right)
        );
        assert_eq!(
            map_button(WinitMouseButton::Middle),
            Some(MouseButton::Middle)
        );
        // Back/forward and hardware extras are not viewport buttons.
        assert_eq!(map_button(WinitMouseButton::Back), None);
        assert_eq!(map_button(WinitMouseButton::Forward), None);
        assert_eq!(map_button(WinitMouseButton::Other(9)), None);
    }

    #[test]
    fn element_state_maps() {
        assert_eq!(map_state(ElementState::Pressed), ButtonState::Pressed);
        assert_eq!(map_state(ElementState::Released), ButtonState::Released);
    }
}
