//! Viewport gesture and binding types for the new input pipeline.

use super::action::Action;
use super::binding::{KeyCode, Modifiers, MouseButton};

/// Modifier matching policy for viewport gestures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModifiersMatch {
    /// All specified modifier bits must match exactly : no extras.
    Exact(Modifiers),
    /// All specified modifier bits must be set; extras are allowed.
    Contains(Modifiers),
    /// Any modifier state is accepted.
    Any,
}

impl ModifiersMatch {
    /// Test whether the given modifier state satisfies this policy.
    pub fn matches(&self, mods: Modifiers) -> bool {
        match self {
            ModifiersMatch::Exact(required) => mods == *required,
            ModifiersMatch::Contains(required) => {
                // every set bit in `required` must also be set in `mods`
                (!required.alt || mods.alt)
                    && (!required.shift || mods.shift)
                    && (!required.ctrl || mods.ctrl)
            }
            ModifiersMatch::Any => true,
        }
    }
}

/// A viewport gesture : the richer gesture vocabulary used by the new pipeline.
#[derive(Debug, Clone)]
pub enum ViewportGesture {
    /// A mouse drag with a specific button and modifier policy.
    Drag {
        /// Which button must be held to drag.
        button: MouseButton,
        /// Required modifier state.
        modifiers: ModifiersMatch,
    },
    /// Vertical wheel delta with a modifier policy.
    ///
    /// Produces a single scalar output (the `y` component of the wheel delta).
    WheelY {
        /// Required modifier state.
        modifiers: ModifiersMatch,
    },
    /// Full two-axis wheel delta with a modifier policy.
    ///
    /// Produces a [`glam::Vec2`] output.
    WheelXY {
        /// Required modifier state.
        modifiers: ModifiersMatch,
    },
    /// A single key press : fires once on the initial press, not on repeat.
    KeyPress {
        /// The key that must be pressed.
        key: KeyCode,
        /// Required modifier state.
        modifiers: ModifiersMatch,
    },
    /// A key held down : fires every frame while the key is held.
    KeyHold {
        /// The key that must be held.
        key: KeyCode,
        /// Required modifier state.
        modifiers: ModifiersMatch,
    },
}

/// Binds an [`Action`] to a [`ViewportGesture`].
#[derive(Debug, Clone)]
pub struct ViewportBinding {
    /// The action this binding fires.
    pub action: Action,
    /// The gesture that activates it.
    pub gesture: ViewportGesture,
}

impl ViewportBinding {
    /// Convenience constructor.
    pub fn new(action: Action, gesture: ViewportGesture) -> Self {
        Self { action, gesture }
    }
}
