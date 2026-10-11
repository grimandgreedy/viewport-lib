//! The Gaussian splat item type: alpha-blended, back-to-front sorted billboard
//! splats over an uploaded set.
//!
//! The data a consumer builds ([`GaussianSplatData`], [`GaussianSplatItem`],
//! [`ShDegree`] and the [`GaussianSplatId`] handle) compiles without the `gpu`
//! feature, so a loader or a scene file can name it without linking the
//! renderer. The plugin, its store and its upload calls need `gpu`.

mod types;

pub use types::{GaussianSplatData, GaussianSplatId, GaussianSplatItem, ShDegree};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.gaussian_splat";

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
pub use plugin::GaussianSplatPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub(crate) use store::GaussianSplatGpuSet;
/// Which channel a write names, as the store sees it. The public face is the
/// marker types in [`channels`].
#[cfg(feature = "gpu")]
pub(crate) use store::SplatChannel;
