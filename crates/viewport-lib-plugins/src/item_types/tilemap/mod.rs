//! The tilemap item type as an [`ItemTypePlugin`]: grids of tiles from one
//! atlas, drawn on a world plane for 2D scenes under `Camera2d`.
//!
//! A [`Tileset`] lays its cells out square-on, isometric, staggered or
//! hexagonal ([`TileOrientation`]), with Tiled's conventions, so a map read
//! from Tiled draws as Tiled draws it.
//!
//! A [`Tilemap`] is uploaded once and returns a [`TilemapId`]; each frame a
//! [`TilemapItem`] places it on a plane with an origin, an elevation, a
//! parallax factor and an animation clock. The map is held in square chunks
//! keyed by chunk coordinate, created as cells are set and removed whole:
//! each chunk is one instance buffer (one 16-byte instance per occupied cell)
//! and one instanced draw, a chunk outside the view is not drawn, and
//! `set_tile` rewrites only the chunk holding the cell.
//!
//! Animated tiles are resolved on the CPU once per animation per frame into a
//! small table in the layer's uniform, which the vertex shader looks up, so an
//! animation never touches the chunk buffers.
//!
//! A cutout layer draws in the opaque scene pass. A blended layer draws after
//! the opaque scene in painter's order through `paint_painter`, in the
//! painter pass after the resolved transparency on the HDR path and after
//! every item type's opaque draws on the LDR path, like painter sprites,
//! interleaved by layer with world overlay layers. A blended layer is one
//! painter batch, or one per drawn row of each chunk so walls and tall props
//! sort against sprites by where they stand.
//!
//! The draws use instanced vertex streams and uniforms only, so the type runs
//! on the compatibility tier.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod grid;
mod types;

pub(crate) use grid::GridLayout;
pub use grid::{StaggerAxis, StaggerIndex, TileOrientation};
pub use types::{
    MAX_TILE_ANIMATIONS, MAX_TILE_CELL, MAX_TILE_SIZES, MIN_TILE_CELL, Tile, TileAnimation,
    TileChunk, TileFlip, TileFrame, TileMaterial, TileMaterialId, TileSize, Tilemap, TilemapDraw,
    TilemapId, TilemapItem, TilemapPainter, TilemapPlane, Tileset,
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
pub(crate) use pipeline::tile_sources;
#[cfg(feature = "gpu")]
pub use plugin::TilemapPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub use uploads::{TileMaterials, TilemapEdits};
