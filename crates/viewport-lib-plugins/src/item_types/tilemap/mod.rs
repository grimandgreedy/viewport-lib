//! The tilemap item type as an [`ItemTypePlugin`]: grids of tiles from one
//! atlas, drawn on a world plane for 2D scenes under `Camera2d`.
//!
//! A [`Tilemap`] is uploaded once and returns a [`TilemapId`]; each frame a
//! [`TilemapItem`] places it on a plane with an origin, an elevation, a
//! parallax factor and an animation clock. The map is held in square chunks:
//! each chunk is one instance buffer (one 16-byte instance per occupied cell)
//! and one instanced draw, a chunk outside the view is not drawn, and
//! `set_tile` rewrites only the chunk holding the cell.
//!
//! Animated tiles are resolved on the CPU once per animation per frame into a
//! small table in the layer's uniform, which the vertex shader looks up, so an
//! animation never touches the chunk buffers.
//!
//! A cutout layer draws in the opaque scene pass. A blended layer draws after
//! the opaque scene in painter's order, in the read-only-depth pass on the HDR
//! path and after the cutout layers on the LDR path, like painter sprites.
//!
//! The draws use instanced vertex streams and uniforms only, so the type runs
//! on the compatibility tier.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::{
    MAX_TILE_ANIMATIONS, MAX_TILEMAP_SIDE, Tile, TileAnimation, TileChunk, TileFlip, TileFrame,
    Tilemap, TilemapDraw, TilemapId, TilemapItem, TilemapPlane, Tileset,
};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.tilemap";

/// The bindings and vertex stage every tilemap shader shares, spliced in front
/// of each body.
#[cfg(feature = "gpu")]
const TILEMAP_COMMON: &str = crate::item_types::shader::wgsl_source!("tilemap_common");

#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;
#[cfg(feature = "gpu")]
mod store;
#[cfg(feature = "gpu")]
mod uploads;

#[cfg(feature = "gpu")]
pub use plugin::TilemapPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub use uploads::TilemapEdits;
