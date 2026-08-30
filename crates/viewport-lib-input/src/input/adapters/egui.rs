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
        _ => return None,
    };
    Some(k)
}
