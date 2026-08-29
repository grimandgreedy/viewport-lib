//! Built-in selection system for ViewportRuntime.

use crate::runtime::context::RuntimeFrameContext;
use crate::runtime::output::RuntimeOutput;
use crate::runtime::output::SelectionOp;

/// Built-in click-to-select system.
///
/// When enabled on [`super::super::ViewportRuntime`], handles primary click and
/// shift-click selection automatically in the `Select` phase. The app supplies
/// `RuntimeFrameContext::pick_hit`, `clicked`, and `shift_held`; this system
/// produces the corresponding [`SelectionOp`] in [`RuntimeOutput::selection_ops`].
pub struct SelectionSystem;

impl Default for SelectionSystem {
    fn default() -> Self {
        Self
    }
}

impl SelectionSystem {
    /// Create a new SelectionSystem.
    pub fn new() -> Self {
        Self
    }

    pub(crate) fn step(&self, frame: &RuntimeFrameContext, output: &mut RuntimeOutput) {
        if !frame.clicked {
            return;
        }
        match &frame.pick_hit {
            Some(hit) => {
                if frame.shift_held {
                    output.selection_ops.push(SelectionOp::Toggle(hit.id));
                } else {
                    output.selection_ops.push(SelectionOp::SelectOne(hit.id));
                }
            }
            None => {
                if !frame.shift_held {
                    output.selection_ops.push(SelectionOp::Clear);
                }
            }
        }
    }
}
