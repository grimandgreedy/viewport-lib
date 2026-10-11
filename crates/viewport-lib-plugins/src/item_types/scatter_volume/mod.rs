//! The scatter-volume item type as an [`ItemTypePlugin`]: participating media
//! (fog, smoke, fire) ray-marched through a box or sphere and composited over
//! the lit scene. Consumers submit [`ScatterVolumeItem`]s with
//! `frame.scene.submit::<ScatterVolumeItem>(..)`.
//!
//! Every pass is encoded from [`ItemTypePlugin::encode`] at
//! [`EncoderScope::AfterTransparent`], in the order they have to run: the
//! optional refraction distortion, the per-volume ray-march into an
//! accumulation target, the optional temporal blend against last frame's
//! history, and the composite onto the scene colour.
//!
//! Two things a scatter volume does are deliberately not here, because they
//! are not rendering. An emissive volume contributes a virtual point light
//! through [`ItemTypePlugin::contribute_lights`], and a selected volume gets a
//! bounds wireframe through the renderer's line substrate.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;
pub(crate) mod volume;

pub use types::ScatterVolumeItem;
pub use volume::{
    ColourSource, DensityRemap, Emission, EmissionCurve, MAX_SCATTER_VOLUMES, NoiseDriver,
    RefractionParams, ScatterShape, ScatterVolume,
};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.scatter_volume";

#[cfg(feature = "gpu")]
mod packing;
#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub use plugin::ScatterVolumePlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
