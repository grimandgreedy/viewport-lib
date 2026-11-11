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
    /// Put `tile` at (`col`, `row`) and re-upload its chunk. `Ok(false)` when
    /// the cell is off the map, which changes nothing.
    fn set_tile(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        col: u32,
        row: u32,
        tile: Tile,
    ) -> viewport_lib::error::ViewportResult<bool>;

    /// Put each `(col, row, tile)` and re-upload every chunk that changed,
    /// once. `Ok(false)` when any cell was off the map; the rest are written.
    fn set_tiles(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        tiles: &[(u32, u32, Tile)],
    ) -> viewport_lib::error::ViewportResult<bool>;

    /// The renderer's copy of an uploaded map, as edits have left it.
    fn tilemap(&self, id: TilemapId) -> Option<&Tilemap>;
}

impl TilemapEdits for ViewportRenderer {
    fn set_tile(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: TilemapId,
        col: u32,
        row: u32,
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
        tiles: &[(u32, u32, Tile)],
    ) -> viewport_lib::error::ViewportResult<bool> {
        plugin_mut::<TilemapPlugin>(self, TYPE_NAME)?.set_tiles(
            device,
            queue,
            id,
            tiles.iter().copied(),
        )
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
    /// [`TilemapItem::material`] names it. `v.local_uv` runs across the cell
    /// and `v.anchor` is the cell's bottom-left corner. The body reads its
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
