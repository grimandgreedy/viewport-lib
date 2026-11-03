//! The image slice item type as an [`ItemTypePlugin`]: an axis-aligned,
//! colourmapped slice of an uploaded 3D volume, drawn as a shader-generated
//! quad. Consumers submit [`ImageSliceItem`]s with
//! `frame.scene.items_mut::<ImageSliceItem>()`;
//! the renderer routes that field to this plugin.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::{ImageSliceItem, SliceAxis};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.image_slice";

#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub use plugin::ImageSlicePlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
