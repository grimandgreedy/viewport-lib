//! Frame-facing render-item descriptors that carry no post-upload store id and
//! no GPU type, so a consumer can build them without the renderer.
//!
//! Most `*Item` descriptors reference uploaded store ids and stay in
//! `viewport-lib`; this module holds the wgpu-free, store-id-free exceptions.

pub mod glyph;
