//! Contour lines on a mesh as an [`ItemTypePlugin`]: lines where a per-vertex
//! scalar attribute crosses a set of levels. Consumers submit
//! [`SurfaceContourItem`]s with `SceneFrame::items_mut`, beside the surface
//! item that draws the mesh.
//!
//! The lines are found per pixel in a fragment shader, so a static field costs
//! one extra draw of the mesh and a changing one costs only the attribute
//! upload. The mesh is drawn again from [`ItemTypePlugin::paint`], inside the
//! scene pass after the opaque geometry, depth-tested against what the
//! surface wrote.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin
//! [`ItemTypePlugin::paint`]: viewport_lib::plugin_api::ItemTypePlugin::paint

mod types;

pub use types::{ContourLevels, MAX_CONTOUR_LEVELS, SurfaceContourItem};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.surface_contour";

#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub use plugin::SurfaceContourPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
