//! The volume surface slice item type as an [`ItemTypePlugin`]: an arbitrary
//! uploaded mesh coloured by sampling a 3D scalar field at each fragment's
//! world position. Consumers submit [`VolumeSurfaceSliceItem`]s on
//! `frame.scene.items_mut::<VolumeSurfaceSliceItem>()`; the renderer routes them to this
//! plugin.
//!
//! The geometry belongs to the consumer's mesh store, so the draw hooks bind it
//! through [`MeshDraw`](viewport_lib::resources::MeshDraw) and the pick hooks read its
//! CPU arrays through [`MeshGeometry`](viewport_lib::resources::MeshGeometry) rather
//! than keeping copies.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::VolumeSurfaceSliceItem;

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.volume_surface_slice";

#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub use plugin::VolumeSurfaceSlicePlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
