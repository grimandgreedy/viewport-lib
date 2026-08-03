//! winit -> [`ViewportEvent`] translation.

use crate::interaction::input::{
    ButtonState, KeyCode, Modifiers, MouseButton, ScrollUnits, ViewportEvent,
};

/// Translate a winit [`WindowEvent`](::winit::event::WindowEvent) into a
/// [`ViewportEvent`], or `None` for events the viewport does not consume.
///
/// Cursor positions are passed through in physical pixels; for a window that
/// hosts a single full-window viewport that is the viewport-local coordinate.
/// An embedded winit viewport should offset by the viewport rect origin instead
/// of using this directly.
pub fn from_winit(event: &::winit::event::WindowEvent) -> Option<ViewportEvent> {
    use ::winit::event::{MouseScrollDelta, WindowEvent};

    match event {
        WindowEvent::CursorMoved { position, .. } => Some(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(position.x as f32, position.y as f32),
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
        _ => return None,
    };
    Some(k)
}
