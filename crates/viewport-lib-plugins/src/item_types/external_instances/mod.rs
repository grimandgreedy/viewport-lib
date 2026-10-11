//! The external instance set item type as an [`ItemTypePlugin`]: one uploaded
//! mesh drawn once per element of a consumer-owned GPU buffer.
//!
//! The buffer holds tightly packed `[x, y, z]` `f32` triples, 12 bytes per
//! instance, and is bound directly as a read-only storage buffer: no CPU copy,
//! no per-frame upload. Whatever the consumer's compute passes last wrote is
//! what renders, and synchronisation is queue-submission order. Register the
//! buffer once with
//! [`ViewportRenderer::create_external_instance_set`](viewport_lib::renderer::ViewportRenderer::create_external_instance_set),
//! then submit an [`ExternalInstancesItem`] with
//! `frame.scene.items_mut::<ExternalInstancesItem>()`
//! per frame. `first_instance` / `instance_count` on the item select a window
//! of the buffer, so several items can render disjoint regions of one pool.
//!
//! The instances are opaque and depth-written, so they draw in `paint`, inside
//! the scene pass, and occlude like ordinary opaque geometry. HDR path only;
//! no shadows, no picking, no culling.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::{ExternalInstanceSetId, ExternalInstancesItem};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.external_instances";

#[cfg(feature = "gpu")]
mod config;
#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;
#[cfg(feature = "gpu")]
pub(crate) mod store;
#[cfg(feature = "gpu")]
mod uploads;

#[cfg(feature = "gpu")]
pub use config::ExternalInstanceSetConfig;
#[cfg(feature = "gpu")]
pub use plugin::ExternalInstancesPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub use uploads::ExternalInstanceUploads;
