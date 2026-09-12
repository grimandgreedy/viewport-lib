//! Host crate for the iced-hosted `viewport-lib` example.
//!
//! ```text
//! cargo run --release -p viewport-lib-examples-iced --example iced-viewport
//! ```
//!
//! iced 0.14 embeds wgpu 27, so this crate has no leg features: it pins
//! viewport-lib's `wgpu27` leg directly. It grows the `wgpu27` / `wgpu29` /
//! `wgpu30` triple the other example crates carry once iced ships against a
//! newer wgpu.
