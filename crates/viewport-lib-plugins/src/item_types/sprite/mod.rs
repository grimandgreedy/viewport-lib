//! The sprite item type as an [`ItemTypePlugin`]: camera-facing textured
//! billboards, optionally lit, with soft-particle fade against scene depth and
//! optional refractive distortion. Consumers submit [`SpriteItem`]s with
//! `frame.scene.submit::<SpriteItem>(..)`, or reference a pre-uploaded batch
//! with [`SpriteSetRefItem`] or [`SpriteInstanceSetRefItem`]. All three forms
//! arrive at this plugin under the one type name.
//!
//! A sprite batch draws in one of four places, chosen by its own fields rather
//! than by a separate setting:
//!
//! - writes depth: the opaque scene pass, through `paint`.
//! - refractive: its own pass over a copy of the scene colour, through
//!   `encode` at [`EncoderScope::AfterOpaque`].
//! - alpha or premultiplied blend in depth order, with no soft fade and no
//!   refraction: the OIT pass, through `paint_transparent`.
//! - anything else transparent, and every batch in painter's order: the
//!   read-only-depth pass, through `paint_depth_read`, which is the only one
//!   of the four that can sample scene depth for the soft fade. Painter
//!   batches draw there last, sorted by layer and then by sort key; on the
//!   LDR path, which has no such pass, they draw in `paint` after the
//!   depth-writing batches.
//!
//! A batch with [`SpriteEffects`] (flash, dissolve, palette swap, outline, or a
//! custom material registered through [`SpriteMaterials`]) draws through effect
//! pipelines composed from the plain shaders, in the same place it would draw
//! otherwise except that it never goes through order-independent transparency.
//! A frame without effects never builds or binds them.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin
//! [`EncoderScope::AfterOpaque`]: viewport_lib::plugin_api::EncoderScope::AfterOpaque
//! [`SpriteMaterials`]: crate::item_types::sprite::SpriteMaterials

mod types;

pub use types::{
    PaletteIndex, SpriteEffects, SpriteFlip, SpriteInstanceSetId, SpriteInstanceSetRefItem,
    SpriteItem, SpriteLitParams, SpriteMaterial, SpriteMaterialId, SpriteNormalMode, SpriteOrder,
    SpriteOrientation, SpriteOutline, SpritePalette, SpriteSetId, SpriteSetRefItem, SpriteSizeMode,
};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.sprite";

/// The structs and helpers every sprite shader shares, spliced in front of
/// each body.
#[cfg(feature = "gpu")]
const SPRITE_COMMON: &str = crate::item_types::shader::wgsl_source!("sprite_common");

#[cfg(feature = "gpu")]
pub mod channels;
#[cfg(feature = "gpu")]
mod effects;
#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;
#[cfg(feature = "gpu")]
mod store;
#[cfg(feature = "gpu")]
mod uploads;

#[cfg(feature = "gpu")]
pub use plugin::SpritePlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub(crate) use store::{encode_sprites, painter_batch_order};
#[cfg(feature = "gpu")]
pub use uploads::{SpriteInstanceUploads, SpriteMaterials};
