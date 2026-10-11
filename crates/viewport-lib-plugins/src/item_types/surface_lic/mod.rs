//! Surface line integral convolution as an [`ItemTypePlugin`]: streaks drawn
//! along a vector field on a mesh. Consumers submit [`SurfaceLicItem`]s with
//! `SceneFrame::items_mut`, beside the surface item that draws the mesh.
//!
//! Both passes are encoded from [`ItemTypePlugin::encode`] at
//! [`EncoderScope::OnOpaqueSurfaces`]: each flow surface into a vector target
//! the plugin owns, depth-tested against the scene, then one fullscreen draw
//! that advects noise along those vectors and multiplies the scene colour by
//! the result.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin
//! [`ItemTypePlugin::encode`]: viewport_lib::plugin_api::ItemTypePlugin::encode
//! [`EncoderScope::OnOpaqueSurfaces`]: viewport_lib::plugin_api::EncoderScope::OnOpaqueSurfaces

mod types;

pub use types::{SurfaceLicConfig, SurfaceLicItem};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.surface_lic";

#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub use plugin::SurfaceLicPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
