//! Host crate for the Slint-hosted `viewport-lib` example.
//!
//! Slint pairs with wgpu 29, so this crate sits outside the parent workspace and
//! bakes the `wgpu29` leg. Build it by manifest path:
//!
//! ```text
//! cargo run --release --manifest-path crates/viewport-lib-examples/slint/Cargo.toml \
//!     --example slint-minimal
//! ```
//!
//! When Slint moves to a newer wgpu, only this manifest changes.
