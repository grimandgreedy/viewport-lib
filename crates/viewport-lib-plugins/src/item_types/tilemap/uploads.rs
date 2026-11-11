//! How a renderer uploads, edits and releases tilemaps.

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

use super::*;
use crate::item_types::{host, plugin_mut};

impl viewport_lib::plugin_api::Uploads<Tilemap> for ViewportRenderer {
    type Id = TilemapId;

    fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &Tilemap,
    ) -> viewport_lib::error::ViewportResult<TilemapId> {
        let host = host::<TilemapPlugin>(self, TYPE_NAME)?;
        Ok(host.plugin.upload(device, queue, host.resources, item))
    }

    fn begin_upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: Tilemap,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<TilemapPlugin>(self, TYPE_NAME)?;
        Ok(host
            .plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item))
    }

    fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        item: &Tilemap,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<TilemapPlugin>(self, TYPE_NAME)?;
        host.plugin.replace(device, queue, host.resources, id, item)
    }
}

impl viewport_lib::plugin_api::Handles<TilemapId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<TilemapId> {
        let host = host::<TilemapPlugin>(self, TYPE_NAME)?;
        host.plugin.take_upload_result(&host.jobs, job)
    }

    fn release(&mut self, id: TilemapId) -> bool {
        plugin_mut::<TilemapPlugin>(self, TYPE_NAME).is_ok_and(|p| p.drop_stored(id))
    }
}

/// Edits to an uploaded tilemap. Each call rewrites only the chunks holding
/// the cells it changes.
pub trait TilemapEdits {
    /// Put `tile` at (`col`, `row`) and re-upload its chunk, creating the
    /// chunk when the map has none there. `Ok(false)`, changing nothing, when
    /// the cell is outside the cell range or the map was uploaded without a
    /// CPU copy ([`Tilemap::keep_cpu_copy`]), whose edits go through
    /// [`set_chunk`](Self::set_chunk).
    fn set_tile(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        col: i32,
        row: i32,
        tile: Tile,
    ) -> viewport_lib::error::ViewportResult<bool>;

    /// Put each `(col, row, tile)` and re-upload every chunk that changed,
    /// once. `Ok(false)` when any cell was outside the cell range, which is
    /// skipped while the rest are written, or the map keeps no CPU copy,
    /// which writes nothing.
    fn set_tiles(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        tiles: &[(i32, i32, Tile)],
    ) -> viewport_lib::error::ViewportResult<bool>;

    /// Replace the whole chunk at `coord` (`[cx, cy]`) and upload it,
    /// creating it when the map has none there; its buffer holds one
    /// instance per occupied cell. Works whether or not the map keeps a CPU
    /// copy. `Ok(false)` when the chunk lies outside the cell range.
    fn set_chunk(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        coord: [i32; 2],
        chunk: TileChunk,
    ) -> viewport_lib::error::ViewportResult<bool>;

    /// Drop the chunk at `coord` and free its buffer. `Ok(false)` when the
    /// map had no chunk there.
    fn remove_chunk(
        &mut self,
        id: TilemapId,
        coord: [i32; 2],
    ) -> viewport_lib::error::ViewportResult<bool>;

    /// The renderer's copy of an uploaded map, as edits have left it. A map
    /// uploaded without a CPU copy comes back with its tileset and no
    /// chunks.
    fn tilemap(&self, id: TilemapId) -> Option<&Tilemap>;
}

impl TilemapEdits for ViewportRenderer {
    fn set_tile(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        col: i32,
        row: i32,
        tile: Tile,
    ) -> viewport_lib::error::ViewportResult<bool> {
        plugin_mut::<TilemapPlugin>(self, TYPE_NAME)?.set_tiles(
            device,
            queue,
            id,
            [(col, row, tile)],
        )
    }

    fn set_tiles(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        tiles: &[(i32, i32, Tile)],
    ) -> viewport_lib::error::ViewportResult<bool> {
        plugin_mut::<TilemapPlugin>(self, TYPE_NAME)?.set_tiles(
            device,
            queue,
            id,
            tiles.iter().copied(),
        )
    }

    fn set_chunk(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        coord: [i32; 2],
        chunk: TileChunk,
    ) -> viewport_lib::error::ViewportResult<bool> {
        plugin_mut::<TilemapPlugin>(self, TYPE_NAME)?.set_chunk(device, queue, id, coord, chunk)
    }

    fn remove_chunk(
        &mut self,
        id: TilemapId,
        coord: [i32; 2],
    ) -> viewport_lib::error::ViewportResult<bool> {
        plugin_mut::<TilemapPlugin>(self, TYPE_NAME)?.remove_chunk(id, coord)
    }

    fn tilemap(&self, id: TilemapId) -> Option<&Tilemap> {
        self.item_type_plugin::<TilemapPlugin>(TYPE_NAME)?
            .tilemap(id)
    }
}

/// Registering tile materials with the tilemap type.
pub trait TileMaterials {
    /// Register `plugin` as a tile material: a body defining
    /// `fn displace_sprite(v: SpriteVertex) -> vec3<f32>`, the sprite vertex
    /// hook, which then moves each corner of every tile of a layer whose
    /// [`TilemapItem::material`] names it. `v.local_uv` runs across the
    /// tile's quad (one cell, or a [`TileSize`]'s quad) and `v.anchor` is the
    /// bottom-left corner of the cell's bounding box, where the grid's
    /// orientation puts it. The body reads its
    /// parameters through `params` and `hook_params.lanes`; tile materials
    /// take no textures.
    ///
    /// # Errors
    ///
    /// [`ViewportError::ShadeNameTaken`](viewport_lib::error::ViewportError::ShadeNameTaken)
    /// when a tile material of that name exists;
    /// [`ViewportError::ShadeShaderInvalid`](viewport_lib::error::ViewportError::ShadeShaderInvalid)
    /// when the name is not a WGSL identifier, the body defines no
    /// `displace_sprite`, it declares textures or too many parameters, or a
    /// composed module fails validation.
    fn register_tile_material(
        &mut self,
        device: &gpu::Device,
        plugin: &dyn viewport_lib::MaterialPlugin,
    ) -> viewport_lib::error::ViewportResult<TileMaterialId>;

    /// The tile material registered as `name`.
    fn tile_material_id(&self, name: &str) -> Option<TileMaterialId>;
}

impl TileMaterials for ViewportRenderer {
    fn register_tile_material(
        &mut self,
        device: &gpu::Device,
        plugin: &dyn viewport_lib::MaterialPlugin,
    ) -> viewport_lib::error::ViewportResult<TileMaterialId> {
        plugin_mut::<TilemapPlugin>(self, TYPE_NAME)?.register_material(device, plugin)
    }

    fn tile_material_id(&self, name: &str) -> Option<TileMaterialId> {
        self.item_type_plugin::<TilemapPlugin>(TYPE_NAME)?
            .material_id(name)
    }
}
