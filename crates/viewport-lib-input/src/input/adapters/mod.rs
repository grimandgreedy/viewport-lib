//! Native-event translation into [`ViewportEvent`](super::ViewportEvent).
//!
//! Each host framework delivers input as its own event type, but the mapping
//! to [`ViewportEvent`](super::ViewportEvent) (buttons, wheel units,
//! modifiers, click-vs-drag) is otherwise identical, so it lives here once
//! per framework instead of in every host application. These adapters are
//! feature-gated per framework and can be used standalone, without any other
//! part of this crate.

#[cfg(feature = "winit-adapter")]
mod winit;
#[cfg(feature = "winit-adapter")]
pub use winit::from_winit;

#[cfg(feature = "egui-adapter")]
mod egui;
#[cfg(feature = "egui-adapter")]
pub use egui::from_egui;
