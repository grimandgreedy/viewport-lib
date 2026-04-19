//! Named control presets for the viewport input pipeline.

use super::binding::{Modifiers, MouseButton};
use super::viewport_binding::{ModifiersMatch, ViewportAction, ViewportBinding, ViewportGesture};

/// Named viewport control presets.
///
/// A preset packages a complete set of [`ViewportBinding`]s that define
/// the camera navigation behavior for a given interaction style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingPreset {
    /// The canonical control scheme matching `examples/winit_primitives`.
    ///
    /// - Left drag → Orbit
    /// - Right drag → Pan
    /// - Middle drag → Orbit
    /// - Middle + Shift drag → Pan
    /// - Scroll → Zoom
    /// - Ctrl + Scroll → Orbit (two-axis)
    /// - Shift + Scroll → Pan (two-axis)
    ViewportPrimitives,
}

/// Returns the viewport bindings for the [`BindingPreset::ViewportPrimitives`] preset.
///
/// This is the canonical reference control scheme, matching `examples/winit_primitives`.
pub fn viewport_primitives_bindings() -> Vec<ViewportBinding> {
    vec![
        // Left drag → Orbit (no modifiers)
        ViewportBinding::new(
            ViewportAction::Orbit,
            ViewportGesture::Drag {
                button: MouseButton::Left,
                modifiers: ModifiersMatch::Exact(Modifiers::NONE),
            },
        ),
        // Right drag → Pan (any modifiers, pan takes priority over orbit for right)
        ViewportBinding::new(
            ViewportAction::Pan,
            ViewportGesture::Drag {
                button: MouseButton::Right,
                modifiers: ModifiersMatch::Any,
            },
        ),
        // Middle drag + Shift → Pan
        ViewportBinding::new(
            ViewportAction::Pan,
            ViewportGesture::Drag {
                button: MouseButton::Middle,
                modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
            },
        ),
        // Middle drag (no shift) → Orbit
        ViewportBinding::new(
            ViewportAction::Orbit,
            ViewportGesture::Drag {
                button: MouseButton::Middle,
                modifiers: ModifiersMatch::Exact(Modifiers::NONE),
            },
        ),
        // Ctrl + Scroll → Orbit (two-axis)
        ViewportBinding::new(
            ViewportAction::Orbit,
            ViewportGesture::WheelXY {
                modifiers: ModifiersMatch::Contains(Modifiers::CTRL),
            },
        ),
        // Shift + Scroll → Pan (two-axis)
        ViewportBinding::new(
            ViewportAction::Pan,
            ViewportGesture::WheelXY {
                modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
            },
        ),
        // Plain Scroll → Zoom
        ViewportBinding::new(
            ViewportAction::Zoom,
            ViewportGesture::WheelY {
                modifiers: ModifiersMatch::Exact(Modifiers::NONE),
            },
        ),
    ]
}
