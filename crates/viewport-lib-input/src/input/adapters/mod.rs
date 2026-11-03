//! Native-event translation into [`ViewportEvent`](crate::input::ViewportEvent),
//! and the per-framework helpers that work out who owns the input.
//!
//! Each host framework delivers input as its own event type, but the mapping
//! to [`ViewportEvent`](crate::input::ViewportEvent) (buttons, wheel units,
//! modifiers, click-vs-drag) is otherwise identical, so it lives here once
//! per framework instead of in every host application. These adapters are
//! feature-gated per framework and can be used standalone, without any other
//! part of this crate.
//!
//! Every adapter has the same shape: a `translate(event, &AdapterCtx, &mut Vec)`
//! that appends what one native event means, and an `ownership(..)` helper that
//! computes the [`InputOwnership`](crate::input::InputOwnership) for the frame.
//! A host pushes everything `translate` produces; the resolver routes it.
//!
//! # What an adapter carries
//!
//! An adapter emits every native event that a `ViewportEvent` can represent,
//! and drops a native event with no representation silently. A
//! `ViewportEvent` variant the framework has no way to produce is listed as
//! unrepresentable in that adapter's tests, which walk every variant, so a new
//! variant fails them until each adapter carries it or lists it. A key event
//! always carries its layout character in `logical` where the framework gives
//! one.

use super::binding::Modifiers;

/// What an adapter needs besides the native event itself.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdapterCtx {
    /// Physical pixels per logical point. Positions are divided by it. winit
    /// reports physical pixels, so pass `window.scale_factor()`; egui already
    /// reports logical points, so pass `1.0`.
    pub scale_factor: f32,
    /// The viewport's top-left corner in logical points, subtracted from every
    /// position so they come out viewport-local. Zero for a viewport that fills
    /// its window.
    pub viewport_origin: glam::Vec2,
    /// The current modifier state, for a framework that reports modifiers only
    /// as state and never as a change event. winit and egui do not read it: winit
    /// sends a change event, and the egui adapter's `begin_frame` reads the
    /// modifiers from egui's input state.
    pub modifiers: Modifiers,
}

impl AdapterCtx {
    /// A context for a viewport at `viewport_origin` in a window with
    /// `scale_factor`, with no modifiers held.
    pub fn new(scale_factor: f32, viewport_origin: glam::Vec2) -> Self {
        Self {
            scale_factor,
            viewport_origin,
            modifiers: Modifiers::NONE,
        }
    }

    /// The same context with the modifier state set.
    pub fn with_modifiers(mut self, modifiers: Modifiers) -> Self {
        self.modifiers = modifiers;
        self
    }

    /// A native position, in the framework's units, as a viewport-local one.
    #[allow(dead_code)]
    pub(crate) fn local(&self, x: f32, y: f32) -> glam::Vec2 {
        glam::Vec2::new(x, y) / self.scale_factor.max(0.001) - self.viewport_origin
    }
}

impl Default for AdapterCtx {
    /// Scale `1.0`, origin zero, no modifiers: a full-window viewport in logical
    /// points.
    fn default() -> Self {
        Self::new(1.0, glam::Vec2::ZERO)
    }
}

#[cfg(feature = "winit-adapter")]
pub mod winit;
#[cfg(feature = "winit-adapter")]
#[allow(deprecated)]
pub use self::winit::{from_winit, modifiers_from_winit};

// One module for every egui version the adapter supports; the enabled one
// re-exports its functions.
#[cfg(any(
    feature = "egui-adapter",
    feature = "egui-adapter-035",
    feature = "egui-adapter-036"
))]
pub mod egui;
#[cfg(any(
    feature = "egui-adapter",
    feature = "egui-adapter-035",
    feature = "egui-adapter-036"
))]
#[allow(deprecated)]
pub use self::egui::{from_egui, modifiers_from_egui};
