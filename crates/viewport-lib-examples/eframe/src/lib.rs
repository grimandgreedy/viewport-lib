//! Host crate for the eframe/egui `viewport-lib` examples, and the eframe
//! version shim they build against.
//!
//! Run one with:
//!
//! ```text
//! cargo run --release -p viewport-lib-examples-eframe --example eframe-showcase
//! ```
//!
//! # The leg shim
//!
//! eframe embeds wgpu, so an eframe version only works against the matching
//! viewport-lib leg: 0.33 with wgpu 27, 0.35 with 29, 0.36 with 30. The three
//! are aliased dependencies selected by this crate's `wgpu27` / `wgpu29` /
//! `wgpu30` features and re-exported here under one name, so an example writes
//!
//! ```ignore
//! use viewport_lib_examples_eframe::eframe;
//! ```
//!
//! and builds on whichever leg is selected, instead of needing one source per
//! version.
//!
//! # Per-example leg status
//!
//! `from_egui`, viewport-lib's egui event translation, takes egui 0.33's
//! `Event` type (the `egui-adapter` feature pins egui 0.33), so an example that
//! uses the adapter only compiles on the 27 leg. Examples that translate events
//! themselves build on every leg.
//!
//! | example | legs |
//! | --- | --- |
//! | `eframe-minimal-callback` | 27, 29, 30 |
//! | every other example here | 27 only (all use `from_egui` or egui widgets against the pinned version) |
//!
//! Check a non-default leg with:
//!
//! ```text
//! cargo run --release -p viewport-lib-examples-eframe --no-default-features \
//!     --features wgpu30 --example eframe-minimal-callback
//! ```

#[cfg(not(any(feature = "wgpu27", feature = "wgpu29", feature = "wgpu30")))]
compile_error!("enable exactly one wgpu leg: wgpu27, wgpu29, or wgpu30");

#[cfg(any(
    all(feature = "wgpu27", feature = "wgpu29"),
    all(feature = "wgpu27", feature = "wgpu30"),
    all(feature = "wgpu29", feature = "wgpu30"),
))]
compile_error!("enable exactly one wgpu leg: wgpu27, wgpu29, or wgpu30");

#[cfg(feature = "wgpu27")]
pub use eframe033 as eframe;
#[cfg(feature = "wgpu29")]
pub use eframe035 as eframe;
#[cfg(feature = "wgpu30")]
pub use eframe036 as eframe;
