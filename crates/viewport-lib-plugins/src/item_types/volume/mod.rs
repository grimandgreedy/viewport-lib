//! The volume item type as an [`ItemTypePlugin`]: an uploaded 3D scalar field
//! ray-marched through its bounding box with colour and opacity transfer
//! functions. Consumers submit [`VolumeItem`]s with
//! `frame.scene.submit::<VolumeItem>(..)`.
//!
//! A wireframe volume draws no ray-march at all: the core line substrate
//! renders an oriented bounding box polyline for it instead, so this plugin
//! only skips the item.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::VolumeItem;

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.volume";

#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub use plugin::VolumePlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
