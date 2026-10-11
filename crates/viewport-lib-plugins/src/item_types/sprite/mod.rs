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
//! - blended with no soft fade and no refraction: the OIT pass, through
//!   `paint_transparent`.
//! - anything else transparent: the read-only-depth pass, through
//!   `paint_depth_read`, which is the only one of the four that can sample
//!   scene depth for the soft fade.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::{
    SpriteInstanceSetId, SpriteInstanceSetRefItem, SpriteItem, SpriteLitParams, SpriteNormalMode,
    SpriteOrientation, SpriteSetId, SpriteSetRefItem, SpriteSizeMode,
};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.sprite";

#[cfg(feature = "gpu")]
pub mod channels;
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
pub(crate) use store::encode_sprites;
#[cfg(feature = "gpu")]
pub use uploads::SpriteInstanceUploads;
