//! The GPU marching cubes item type as an [`ItemTypePlugin`]: an isosurface
//! extracted from an uploaded scalar field by three compute passes each frame
//! and drawn with indirect draws. Consumers submit [`GpuMarchingCubesItem`]s
//! with `frame.scene.submit::<GpuMarchingCubesItem>(..)`.
//!
//! The plugin owns the pipelines, the per-frame extraction, and the uploaded
//! volumes themselves: [`McVolumes::upload_volume_for_mc`]
//! hands back a [`McVolumeId`] that names a volume held here.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::{GpuMarchingCubesItem, McVolumeId};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.gpu_marching_cubes";

#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;
#[cfg(feature = "gpu")]
mod store;
#[cfg(feature = "gpu")]
mod uploads;

#[cfg(feature = "gpu")]
pub use plugin::GpuMarchingCubesPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub use uploads::McVolumes;
