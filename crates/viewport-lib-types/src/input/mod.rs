//! Input vocabulary: events, semantic actions, input modes, and the binding
//! tables that map triggers to actions. Not tied to any particular windowing
//! or GUI framework.
//!
//! This is pure data. The renderer's input pipeline (the stateful accumulator,
//! the query-evaluation engine, and the native-event adapters for winit/egui)
//! consumes this vocabulary but lives in `viewport-lib`.

// Some binding/query vocabulary is deprecated but still defined and re-exported
// here for the compatibility layer; suppress the lint for these re-exports.
#![allow(deprecated)]

/// Semantic action enum.
pub mod action;
/// Per-frame resolved action output.
pub mod action_frame;
/// Binding, trigger, and modifier types.
pub mod binding;
/// Per-frame viewport context.
pub mod context;
/// Default key/mouse bindings for the viewport.
pub mod defaults;
/// Viewport events, translated from native windowing/GUI events.
pub mod event;
/// Input mode enum (Normal, FlyMode, Manipulating).
pub mod mode;
/// Named control presets.
pub mod preset;
/// Viewport gesture and binding types.
pub mod viewport_binding;

pub use action::Action;
pub use action_frame::{ActionFrame, NavigationActions, PointerFrame, ResolvedActionState};
pub use binding::{ActivationMode, Binding, KeyCode, Modifiers, MouseButton, Trigger, TriggerKind};
pub use context::ViewportContext;
pub use defaults::default_bindings;
pub use event::{ButtonState, ScrollUnits, ViewportEvent};
pub use mode::{InputMode, NavigationMode};
pub use preset::{BindingPreset, viewport_all_bindings};
pub use viewport_binding::{ModifiersMatch, ViewportBinding, ViewportGesture};
