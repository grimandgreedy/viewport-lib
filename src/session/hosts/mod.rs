//! Host integrations that own the outer shell around a [`ViewportSession`].
//!
//! Unlike the event adapters (which are pure translation), these touch the
//! render path and the window/event loop, so they are gated per host.

#[cfg(feature = "app")]
mod winit;
#[cfg(feature = "app")]
pub use winit::{AppConfig, FrameCtx, ViewportApp};
