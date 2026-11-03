//! The point cloud item type: a set of world-space points drawn as
//! screen-space discs, coloured by a scalar through a colourmap or by
//! per-point colours.
//!
//! Submit [`PointCloudItem`]s with `frame.scene.submit::<PointCloudItem>(..)`,
//! or [`PointCloudRefItem`]s to draw a cloud uploaded once through
//! [`Uploads`](viewport_lib::plugin_api::Uploads).
//! Both forms arrive at this plugin under the one type name.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.

mod types;

pub use types::{PointCloudId, PointCloudItem, PointCloudRefItem, PointRenderMode};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.point_cloud";

#[cfg(feature = "gpu")]
pub mod channels;
#[cfg(feature = "gpu")]
mod debug_draw;
#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;
#[cfg(feature = "gpu")]
mod store;
#[cfg(feature = "gpu")]
mod uploads;

#[cfg(feature = "gpu")]
pub use plugin::PointCloudPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
/// Which channel a write names, as the store sees it. The public face is the
/// marker types in [`channels`].
#[cfg(feature = "gpu")]
pub(crate) use store::PointChannel;
