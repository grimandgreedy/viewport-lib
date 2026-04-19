//! Named control presets for the viewport input pipeline.

use super::action::Action;
use super::binding::{KeyCode, Modifiers, MouseButton};
use super::viewport_binding::{ModifiersMatch, ViewportBinding, ViewportGesture};

/// Named viewport control presets.
///
/// A preset packages a complete set of [`ViewportBinding`]s that define
/// the viewport interaction behavior for a given interaction style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingPreset {
    /// The canonical camera-navigation control scheme matching `examples/winit_primitives`.
    ///
    /// - Left drag → Orbit
    /// - Right drag → Pan
    /// - Middle drag → Orbit
    /// - Middle + Shift drag → Pan
    /// - Scroll → Zoom
    /// - Ctrl + Scroll → Orbit (two-axis)
    /// - Shift + Scroll → Pan (two-axis)
    ViewportPrimitives,

    /// Full binding set: camera navigation from [`ViewportPrimitives`] plus all
    /// keyboard shortcuts (normal mode, fly mode, manipulation mode, global).
    ///
    /// Use this preset to replace [`crate::InputSystem`] entirely.
    ViewportAll,
}

/// Returns the viewport bindings for the [`BindingPreset::ViewportPrimitives`] preset.
///
/// This is the canonical reference control scheme, matching `examples/winit_primitives`.
pub fn viewport_primitives_bindings() -> Vec<ViewportBinding> {
    vec![
        // Left drag → Orbit (no modifiers)
        ViewportBinding::new(
            Action::Orbit,
            ViewportGesture::Drag {
                button: MouseButton::Left,
                modifiers: ModifiersMatch::Exact(Modifiers::NONE),
            },
        ),
        // Right drag → Pan (any modifiers, pan takes priority over orbit for right)
        ViewportBinding::new(
            Action::Pan,
            ViewportGesture::Drag {
                button: MouseButton::Right,
                modifiers: ModifiersMatch::Any,
            },
        ),
        // Middle drag + Shift → Pan
        ViewportBinding::new(
            Action::Pan,
            ViewportGesture::Drag {
                button: MouseButton::Middle,
                modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
            },
        ),
        // Middle drag (no shift) → Orbit
        ViewportBinding::new(
            Action::Orbit,
            ViewportGesture::Drag {
                button: MouseButton::Middle,
                modifiers: ModifiersMatch::Exact(Modifiers::NONE),
            },
        ),
        // Ctrl + Scroll → Orbit (two-axis)
        ViewportBinding::new(
            Action::Orbit,
            ViewportGesture::WheelXY {
                modifiers: ModifiersMatch::Contains(Modifiers::CTRL),
            },
        ),
        // Shift + Scroll → Pan (two-axis)
        ViewportBinding::new(
            Action::Pan,
            ViewportGesture::WheelXY {
                modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
            },
        ),
        // Plain Scroll → Zoom
        ViewportBinding::new(
            Action::Zoom,
            ViewportGesture::WheelY {
                modifiers: ModifiersMatch::Exact(Modifiers::NONE),
            },
        ),
    ]
}

/// Returns the full binding set for [`BindingPreset::ViewportAll`].
///
/// Extends [`viewport_primitives_bindings`] with all keyboard shortcuts:
/// normal-mode actions, fly-mode movement, manipulation constraints, and
/// global actions (undo/redo).
///
/// Consumers are responsible for applying mode awareness — key bindings for
/// fly mode and manipulation mode are always present in the resolved
/// [`crate::ActionFrame`], so callers should gate on the current [`crate::InputMode`].
pub fn viewport_all_bindings() -> Vec<ViewportBinding> {
    let none = ModifiersMatch::Exact(Modifiers::NONE);
    let any = ModifiersMatch::Any;

    let mut bindings = viewport_primitives_bindings();

    // -- Normal mode: object manipulation shortcuts --
    bindings.push(ViewportBinding::new(
        Action::BeginMove,
        ViewportGesture::KeyPress { key: KeyCode::G, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::BeginRotate,
        ViewportGesture::KeyPress { key: KeyCode::R, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::BeginScale,
        ViewportGesture::KeyPress { key: KeyCode::S, modifiers: none },
    ));

    // -- Normal mode: object shortcuts --
    bindings.push(ViewportBinding::new(
        Action::OpenAddMenu,
        ViewportGesture::KeyPress {
            key: KeyCode::A,
            modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::DeleteSelected,
        ViewportGesture::KeyPress { key: KeyCode::X, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::FocusObject,
        ViewportGesture::KeyPress { key: KeyCode::F, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::ResetView,
        ViewportGesture::KeyPress { key: KeyCode::R, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::ToggleWireframe,
        ViewportGesture::KeyPress { key: KeyCode::W, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::CycleGizmoMode,
        ViewportGesture::KeyPress { key: KeyCode::Tab, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::ToggleGizmoSpace,
        ViewportGesture::KeyPress { key: KeyCode::Backtick, modifiers: none },
    ));

    // -- Fly mode entry --
    bindings.push(ViewportBinding::new(
        Action::EnterFlyMode,
        ViewportGesture::KeyPress {
            key: KeyCode::Backtick,
            modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
        },
    ));

    // -- Fly mode movement (KeyHold; callers must gate on InputMode::FlyMode) --
    bindings.push(ViewportBinding::new(
        Action::FlyForward,
        ViewportGesture::KeyHold { key: KeyCode::W, modifiers: any },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyBackward,
        ViewportGesture::KeyHold { key: KeyCode::S, modifiers: any },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyLeft,
        ViewportGesture::KeyHold { key: KeyCode::A, modifiers: any },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyRight,
        ViewportGesture::KeyHold { key: KeyCode::D, modifiers: any },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyUp,
        ViewportGesture::KeyHold { key: KeyCode::E, modifiers: any },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyDown,
        ViewportGesture::KeyHold { key: KeyCode::Q, modifiers: any },
    ));

    // -- Manipulation mode: axis constraints --
    bindings.push(ViewportBinding::new(
        Action::ConstrainX,
        ViewportGesture::KeyPress { key: KeyCode::X, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::ConstrainY,
        ViewportGesture::KeyPress { key: KeyCode::Y, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::ConstrainZ,
        ViewportGesture::KeyPress { key: KeyCode::Z, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::ExcludeX,
        ViewportGesture::KeyPress {
            key: KeyCode::X,
            modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::ExcludeY,
        ViewportGesture::KeyPress {
            key: KeyCode::Y,
            modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::ExcludeZ,
        ViewportGesture::KeyPress {
            key: KeyCode::Z,
            modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
        },
    ));

    // -- Confirm / Cancel (fly mode + manipulation mode; callers must gate on mode) --
    bindings.push(ViewportBinding::new(
        Action::Confirm,
        ViewportGesture::KeyPress { key: KeyCode::Enter, modifiers: none },
    ));
    bindings.push(ViewportBinding::new(
        Action::Cancel,
        ViewportGesture::KeyPress { key: KeyCode::Escape, modifiers: none },
    ));

    // -- Global: undo/redo --
    bindings.push(ViewportBinding::new(
        Action::Redo,
        ViewportGesture::KeyPress {
            key: KeyCode::Z,
            modifiers: ModifiersMatch::Contains(Modifiers::CTRL_SHIFT),
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::Undo,
        ViewportGesture::KeyPress {
            key: KeyCode::Z,
            modifiers: ModifiersMatch::Contains(Modifiers::CTRL),
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::Redo,
        ViewportGesture::KeyPress {
            key: KeyCode::Y,
            modifiers: ModifiersMatch::Contains(Modifiers::CTRL),
        },
    ));

    bindings
}
