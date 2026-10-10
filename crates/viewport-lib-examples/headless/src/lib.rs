//! Host crate for the headless `viewport-lib` examples: offscreen rendering,
//! the job runtime, the path tracer, and the leg-agnostic build check. None of
//! them opens a window.
//!
//! Run one with:
//!
//! ```text
//! cargo run --release -p viewport-lib-examples-headless --example raytrace-reference
//! ```
//!
//! The `wgpu27` / `wgpu29` / `wgpu30` features pick viewport-lib's wgpu leg;
//! exactly one must be enabled. These examples name wgpu only through
//! `viewport_lib::wgpu`, so the same source builds on every leg.

#[cfg(not(any(feature = "wgpu27", feature = "wgpu29", feature = "wgpu30")))]
compile_error!("enable exactly one wgpu leg: wgpu27, wgpu29, or wgpu30");

#[cfg(any(
    all(feature = "wgpu27", feature = "wgpu29"),
    all(feature = "wgpu27", feature = "wgpu30"),
    all(feature = "wgpu29", feature = "wgpu30"),
))]
compile_error!("enable exactly one wgpu leg: wgpu27, wgpu29, or wgpu30");
