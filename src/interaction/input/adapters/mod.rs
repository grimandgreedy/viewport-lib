//! Native-event translation into [`ViewportEvent`](super::ViewportEvent).
//!
//! Each host framework delivers input as its own event type; the mapping to
//! [`ViewportEvent`](super::ViewportEvent) (buttons, wheel units, modifiers,
//! click-vs-drag) is otherwise identical and was previously copy-pasted per
//! example. These adapters are feature-gated per framework and are usable on
//! their own, without a [`ViewportSession`](crate::session::ViewportSession).

#[cfg(feature = "winit-adapter")]
mod winit;
#[cfg(feature = "winit-adapter")]
pub use winit::from_winit;

#[cfg(feature = "egui-adapter")]
mod egui;
#[cfg(feature = "egui-adapter")]
pub use egui::from_egui;
