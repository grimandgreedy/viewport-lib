//! Frame-facing render-item descriptors that carry no post-upload store id and
//! no GPU type, so a consumer can build them without the renderer.
//!
//! Most `*Item` descriptors reference uploaded store ids and stay in
//! `viewport-lib`; this module holds the wgpu-free, store-id-free exceptions.

mod blend;
mod painter;

pub use blend::SpriteBlend;
pub use painter::{MaskTest, PAINTER_MASK_SLOTS, PAINTER_MEMBERS, PainterPlacement, painter_key};
