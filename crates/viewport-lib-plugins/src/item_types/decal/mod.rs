//! The screen-space decal item type as an [`ItemTypePlugin`]: textures
//! projected onto opaque surfaces along a box. A decal reads the surface mask
//! to stay off surfaces which opted out. Consumers submit [`DecalItem`]s with
//! `SceneFrame::items_mut`.
//!
//! All three passes are encoded from [`ItemTypePlugin::encode`] at
//! [`EncoderScope::OnOpaqueSurfaces`], in the order they have to run: the
//! colour projection, then the outline mask and edge trace when a decal is
//! selected.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin
//! [`EncoderScope::OnOpaqueSurfaces`]: viewport_lib::plugin_api::EncoderScope::OnOpaqueSurfaces
//! [`ItemTypePlugin::encode`]: viewport_lib::plugin_api::ItemTypePlugin::encode

mod live;
mod types;

pub use live::{DecalHandle, LiveDecal, LiveDecals};
pub use types::{CylindricalFacing, DecalAnimation, DecalBlendMode, DecalItem, DecalProjection};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.decal";

#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub use plugin::DecalPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
