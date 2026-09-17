//! Item structs: the per-frame submission types a consumer fills in.
//!
//! Each item type implemented as an [`ItemTypePlugin`](crate::plugin_api::ItemTypePlugin)
//! keeps its own struct in its plugin directory, beside the code that draws it,
//! and this module re-exports them so the crate-root paths and every internal
//! `items::` path resolve unchanged. What is declared here rather than
//! re-exported belongs to the geometry substrate (surface mesh, mesh-instance
//! batch, volume mesh, external instances), which is not an item type, plus the
//! bits the item structs share.

pub(crate) mod common;
mod compute_filter;
mod external_instances;
mod mesh;
mod mesh_instance;
mod volume_mesh;

pub use self::common::*;
pub use self::compute_filter::*;
pub use self::external_instances::*;
pub use crate::renderer::item_plugins::decal::types::*;
pub use crate::renderer::item_plugins::gaussian_splat::types::*;
pub use self::mesh::*;
pub use self::mesh_instance::*;
pub use self::volume_mesh::*;
pub use crate::renderer::item_plugins::curves::types::*;
pub use crate::renderer::item_plugins::glyph::types::*;
pub use crate::renderer::item_plugins::gpu_implicit::types::*;
pub use crate::renderer::item_plugins::gpu_marching_cubes::types::*;
pub use crate::renderer::item_plugins::image_slice::types::*;
pub use crate::renderer::item_plugins::point_cloud::types::*;
pub use crate::renderer::item_plugins::polyline::types::*;
pub use crate::renderer::item_plugins::scatter_volume::types::*;
pub use crate::renderer::item_plugins::sprite::types::*;
pub use crate::renderer::item_plugins::tensor_glyph::types::*;
pub use crate::renderer::item_plugins::volume::types::*;
pub use crate::renderer::item_plugins::volume_surface_slice::types::*;
// The glyph and tensor-glyph structs are wgpu-free and carry no store id, so
// they live in `viewport-lib-types` rather than in their plugin directories.
// Their reference forms do sit with their plugins, above.
pub use viewport_lib_types::render_item::glyph::{GlyphItem, GlyphType, TensorGlyphItem};
