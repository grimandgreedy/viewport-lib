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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_none_matches_no_modifiers() {
        assert!(ModifiersMatch::Exact(Modifiers::NONE).matches(Modifiers::NONE));
    }

    #[test]
    fn exact_none_rejects_alt() {
        assert!(!ModifiersMatch::Exact(Modifiers::NONE).matches(Modifiers::ALT));
    }

    #[test]
    fn exact_shift_matches_shift_only() {
        assert!(ModifiersMatch::Exact(Modifiers::SHIFT).matches(Modifiers::SHIFT));
        assert!(!ModifiersMatch::Exact(Modifiers::SHIFT).matches(Modifiers::NONE));
        assert!(!ModifiersMatch::Exact(Modifiers::SHIFT).matches(Modifiers::CTRL_SHIFT));
    }

    #[test]
    fn contains_shift_allows_extras() {
        assert!(ModifiersMatch::Contains(Modifiers::SHIFT).matches(Modifiers::SHIFT));
        assert!(ModifiersMatch::Contains(Modifiers::SHIFT).matches(Modifiers::CTRL_SHIFT));
        assert!(!ModifiersMatch::Contains(Modifiers::SHIFT).matches(Modifiers::NONE));
        assert!(!ModifiersMatch::Contains(Modifiers::SHIFT).matches(Modifiers::CTRL));
    }

    #[test]
    fn contains_ctrl_shift_requires_both() {
        assert!(ModifiersMatch::Contains(Modifiers::CTRL_SHIFT).matches(Modifiers::CTRL_SHIFT));
        assert!(!ModifiersMatch::Contains(Modifiers::CTRL_SHIFT).matches(Modifiers::CTRL));
        assert!(!ModifiersMatch::Contains(Modifiers::CTRL_SHIFT).matches(Modifiers::SHIFT));
    }

    #[test]
    fn any_matches_everything() {
        assert!(ModifiersMatch::Any.matches(Modifiers::NONE));
        assert!(ModifiersMatch::Any.matches(Modifiers::ALT));
        assert!(ModifiersMatch::Any.matches(Modifiers::CTRL_SHIFT));
    }
}
