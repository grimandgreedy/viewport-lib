//! Host crate for the winit-hosted `viewport-lib` examples: the hand-written
//! event loops (`winit-*`) and the built-in runners (`app-*`).
//!
//! Run one with:
//!
//! ```text
//! cargo run --release -p viewport-lib-examples-winit --example winit-minimal
//! ```
//!
//! The `wgpu27` / `wgpu29` / `wgpu30` features pick viewport-lib's wgpu leg;
//! exactly one must be enabled.
//!
//! # Per-example leg status
//!
//! winit does not embed wgpu, so the leg is viewport-lib's alone. What limits an
//! example is whether it drives the wgpu surface itself: wgpu 29 changed
//! `Surface::get_current_texture` from `Result<_, SurfaceError>` to a
//! `CurrentSurfaceTexture` enum, so an example owning its own acquire and
//! present loop is written against one wgpu version. Examples that let the
//! built-in runners own the surface build on every leg.
//!
//! | example | legs |
//! | --- | --- |
//! | `winit-minimal`, `winit-basic-interaction`, `app-multi-window`, `app-in-window-viewports` | 27, 29, 30 |
//! | `winit-viewport`, `winit-multi-viewport`, `winit-hdr`, `winit-web`, `raytrace-interactive`, `raytrace-backends` | 27 only (own the surface loop) |
//!
//! Check the portable set on another leg with:
//!
//! ```text
//! cargo build -p viewport-lib-examples-winit --all-targets --no-default-features --features wgpu29
//! ```

#[cfg(not(any(feature = "wgpu27", feature = "wgpu29", feature = "wgpu30")))]
compile_error!("enable exactly one wgpu leg: wgpu27, wgpu29, or wgpu30");

#[cfg(any(
    all(feature = "wgpu27", feature = "wgpu29"),
    all(feature = "wgpu27", feature = "wgpu30"),
    all(feature = "wgpu29", feature = "wgpu30"),
))]
compile_error!("enable exactly one wgpu leg: wgpu27, wgpu29, or wgpu30");
