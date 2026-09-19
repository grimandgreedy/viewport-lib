//! Host crate for the Bevy-hosted `viewport-lib` example.
//!
//! Bevy 0.19 pins wgpu 29, so this crate sits outside the parent workspace and
//! bakes the `wgpu29` leg. Build it by manifest path:
//!
//! ```text
//! cargo run --release --manifest-path crates/viewport-lib-examples/bevy/Cargo.toml \
//!     --example bevy-swarm
//! ```
//!
//! The example shares Bevy's device: viewport-lib renders into a Bevy GPU
//! texture with no CPU copy. When Bevy moves to a newer wgpu, only this manifest
//! changes.
