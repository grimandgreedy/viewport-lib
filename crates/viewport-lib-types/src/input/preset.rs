//! Named control presets for the viewport input pipeline.

use super::action::Action;
use super::binding::{KeyCode, Modifiers, MouseButton};
use super::viewport_binding::{ModifiersMatch, ViewportBinding, ViewportGesture};

/// Named viewport control presets.
///
/// A preset packages a complete set of [`ViewportBinding`]s that define the
/// viewport interaction behaviour for a given interaction style. Bindings belong
/// to the resolver that owns them, not to a camera controller: one
/// `ViewportInput` per viewport holds the set, and controllers consume the
/// `ActionFrame` it produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BindingPreset {
    /// The default scheme: the library takes the middle button and the wheel, and
    /// nothing else.
    ///
    /// - Middle drag -> Orbit
    /// - Shift + Middle drag -> Pan
    /// - Scroll -> Zoom
    /// - Ctrl + Scroll -> Orbit
    /// - Shift + Scroll -> Pan
    ///
    /// Plus the viewport keyboard shortcuts (manipulation, axis constraints, fly
    /// movement, focus, reset view, wireframe, gizmo mode and space).
    ///
    /// **Left drag and right drag are deliberately unbound** and belong to the
    /// application: selection, gizmo dragging, tools, context menus.
    ///
    /// Two complete schemes with one rule connecting them. The middle button covers
    /// orbit and pan; the wheel with modifiers covers all three with no buttons at
    /// all, which is what makes a viewport usable on a trackpad. Across both, shift
    /// means pan and ctrl means orbit. Zoom is scroll only.
    #[default]
    Default,

    /// [`Default`](Self::Default) plus `Left drag -> Orbit`.
    ///
    /// Use this **only if the application never uses left drag**, which in practice
    /// means it has no selection, no gizmo and no tools: a model viewer, a plot, a
    /// thumbnail preview. In anything with a selection, left drag has two claimants
    /// and the camera will fight the tool.
    Viewer,
}

/// Former name for the camera-only scheme. Resolves to [`BindingPreset::Viewer`],
/// which keeps left-drag orbit; note that right-drag pan is no longer bound.
#[deprecated(note = "renamed: use BindingPreset::Viewer (left drag still orbits) or ::Default")]
pub const VIEWPORT_PRIMITIVES: BindingPreset = BindingPreset::Viewer;

/// Former name for the full scheme. Resolves to [`BindingPreset::Default`].
#[deprecated(note = "renamed: use BindingPreset::Default")]
pub const VIEWPORT_ALL: BindingPreset = BindingPreset::Default;

/// The camera-navigation half of [`BindingPreset::Default`]: the middle button and
/// the wheel, and nothing else.
///
/// Left drag and right drag are unbound and belong to the application. Use this when
/// you want the camera scheme without the viewport's keyboard shortcuts.
pub fn viewport_camera_bindings() -> Vec<ViewportBinding> {
    vec![
        // Middle drag (no modifiers) -> Orbit. Exact, so shift+middle can mean pan.
        ViewportBinding::new(
            Action::Orbit,
            ViewportGesture::Drag {
                button: MouseButton::Middle,
                modifiers: ModifiersMatch::Exact(Modifiers::NONE),
            },
        ),
        // Middle + Shift drag -> Pan. Shift means pan here and on the wheel below.
        ViewportBinding::new(
            Action::Pan,
            ViewportGesture::Drag {
                button: MouseButton::Middle,
                modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
            },
        ),
        // Ctrl + Scroll -> Orbit (two-axis). Ctrl means orbit here and nowhere else.
        ViewportBinding::new(
            Action::Orbit,
            ViewportGesture::WheelXY {
                modifiers: ModifiersMatch::Contains(Modifiers::CTRL),
            },
        ),
        // Shift + Scroll -> Pan (two-axis).
        ViewportBinding::new(
            Action::Pan,
            ViewportGesture::WheelXY {
                modifiers: ModifiersMatch::Contains(Modifiers::SHIFT),
            },
        ),
        // Plain Scroll -> Zoom. Zoom is scroll only; there is no drag-to-zoom.
        ViewportBinding::new(
            Action::Zoom,
            ViewportGesture::WheelY {
                modifiers: ModifiersMatch::Exact(Modifiers::NONE),
            },
        ),
    ]
}

/// Left drag bound to orbit, the one addition [`BindingPreset::Viewer`] makes.
fn left_drag_orbit() -> ViewportBinding {
    ViewportBinding::new(
        Action::Orbit,
        ViewportGesture::Drag {
            button: MouseButton::Left,
            modifiers: ModifiersMatch::Exact(Modifiers::NONE),
        },
    )
}

/// Returns the bindings for [`BindingPreset::Viewer`]: the default set plus
/// `Left drag -> Orbit`.
///
/// Only for an application that never uses left drag itself.
pub fn viewer_bindings() -> Vec<ViewportBinding> {
    let mut bindings = viewport_default_bindings();
    bindings.push(left_drag_orbit());
    bindings
}

/// Former name for the camera-only set. The scheme has changed: left drag and right
/// drag are no longer bound to the camera.
#[deprecated(note = "renamed: use viewport_camera_bindings (note the new scheme) or viewer_bindings")]
pub fn viewport_primitives_bindings() -> Vec<ViewportBinding> {
    viewer_bindings()
}

/// Returns the full binding set for [`BindingPreset::Default`]: the camera scheme
/// from [`viewport_camera_bindings`] plus the viewport's keyboard shortcuts.
///
/// Covers normal-mode actions, fly-mode movement, and manipulation constraints.
/// Undo and redo are deliberately **not** bound: they belong to the application's
/// document, not to the viewport, so binding them here would make every consumer
/// notice and unbind them.
///
/// Consumers are responsible for applying mode awareness : key bindings for
/// fly mode and manipulation mode are always present in the resolved
/// [`ActionFrame`](crate::input::action_frame::ActionFrame), so callers should
/// gate on the current [`InputMode`](crate::input::mode::InputMode).
pub fn viewport_default_bindings() -> Vec<ViewportBinding> {
    let none = ModifiersMatch::Exact(Modifiers::NONE);
    let any = ModifiersMatch::Any;

    let mut bindings = viewport_camera_bindings();

    // -- Normal mode: object manipulation shortcuts --
    bindings.push(ViewportBinding::new(
        Action::BeginMove,
        ViewportGesture::KeyPress {
            key: KeyCode::G,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::BeginRotate,
        ViewportGesture::KeyPress {
            key: KeyCode::R,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::BeginScale,
        ViewportGesture::KeyPress {
            key: KeyCode::S,
            modifiers: none,
        },
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
        ViewportGesture::KeyPress {
            key: KeyCode::X,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::FocusObject,
        ViewportGesture::KeyPress {
            key: KeyCode::F,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::ResetView,
        ViewportGesture::KeyPress {
            key: KeyCode::R,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::ToggleWireframe,
        ViewportGesture::KeyPress {
            key: KeyCode::W,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::CycleGizmoMode,
        ViewportGesture::KeyPress {
            key: KeyCode::Tab,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::ToggleGizmoSpace,
        ViewportGesture::KeyPress {
            key: KeyCode::Backtick,
            modifiers: none,
        },
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
        ViewportGesture::KeyHold {
            key: KeyCode::W,
            modifiers: any,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyBackward,
        ViewportGesture::KeyHold {
            key: KeyCode::S,
            modifiers: any,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyLeft,
        ViewportGesture::KeyHold {
            key: KeyCode::A,
            modifiers: any,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyRight,
        ViewportGesture::KeyHold {
            key: KeyCode::D,
            modifiers: any,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyUp,
        ViewportGesture::KeyHold {
            key: KeyCode::E,
            modifiers: any,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::FlyDown,
        ViewportGesture::KeyHold {
            key: KeyCode::Q,
            modifiers: any,
        },
    ));

    // -- Manipulation mode: axis constraints --
    bindings.push(ViewportBinding::new(
        Action::ConstrainX,
        ViewportGesture::KeyPress {
            key: KeyCode::X,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::ConstrainY,
        ViewportGesture::KeyPress {
            key: KeyCode::Y,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::ConstrainZ,
        ViewportGesture::KeyPress {
            key: KeyCode::Z,
            modifiers: none,
        },
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

    // -- Manipulation mode: numeric input --
    bindings.push(ViewportBinding::new(
        Action::NumericBackspace,
        ViewportGesture::KeyPress {
            key: KeyCode::Backspace,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::NumericNextAxis,
        ViewportGesture::KeyPress {
            key: KeyCode::Tab,
            modifiers: none,
        },
    ));

    // -- Confirm / Cancel (fly mode + manipulation mode; callers must gate on mode) --
    bindings.push(ViewportBinding::new(
        Action::Confirm,
        ViewportGesture::KeyPress {
            key: KeyCode::Enter,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::Cancel,
        ViewportGesture::KeyPress {
            key: KeyCode::Escape,
            modifiers: none,
        },
    ));

    // -- Pivot mode cycling (normal + manipulation mode) --
    bindings.push(ViewportBinding::new(
        Action::CyclePivotModeForward,
        ViewportGesture::KeyPress {
            key: KeyCode::LeftBracket,
            modifiers: none,
        },
    ));
    bindings.push(ViewportBinding::new(
        Action::CyclePivotModeBackward,
        ViewportGesture::KeyPress {
            key: KeyCode::RightBracket,
            modifiers: none,
        },
    ));

    bindings
}

/// Former name for the full set. Renamed; note that undo and redo are no longer
/// bound, and the camera scheme has changed.
#[deprecated(note = "renamed: use viewport_default_bindings")]
pub fn viewport_all_bindings() -> Vec<ViewportBinding> {
    viewport_default_bindings()
}
