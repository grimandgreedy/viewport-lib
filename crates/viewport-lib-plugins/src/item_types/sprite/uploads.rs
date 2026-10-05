//! How a renderer uploads, writes and releases sprites.

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

use super::*;
use crate::item_types::{cast_bytes, host, plugin_mut};

/// Ranged writes into a stored sprite batch, one implementation per channel.
///
/// The positions are their own vertex stream and the rest of a sprite is one
/// interleaved record, so a particle feed that only moves its sprites writes a
/// quarter of the bytes a full update would.
macro_rules! sprite_writes {
    ($marker:ty, positions = $positions:expr, encode = $encode:expr) => {
        impl viewport_lib::plugin_api::Writes<$marker> for ViewportRenderer {
            fn write_range(
                &mut self,
                _channel: $marker,
                queue: &gpu::Queue,
                id: SpriteSetId,
                first_element: u32,
                data: &[<$marker as viewport_lib::plugin_api::Channel>::Input],
            ) -> viewport_lib::error::ViewportResult<()> {
                let bytes = $encode(data);
                plugin_mut::<SpritePlugin>(self, TYPE_NAME).write_set_channel(
                    queue,
                    id,
                    $positions,
                    first_element,
                    &bytes,
                )
            }

            fn reserve(
                &mut self,
                _channel: $marker,
                device: &gpu::Device,
                queue: &gpu::Queue,
                id: SpriteSetId,
                capacity: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<SpritePlugin>(self, TYPE_NAME).reserve_set(device, queue, id, capacity)
            }

            fn set_len(
                &mut self,
                _channel: $marker,
                id: SpriteSetId,
                len: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<SpritePlugin>(self, TYPE_NAME).set_set_len(id, len)
            }

            fn extent(
                &self,
                _channel: $marker,
                id: SpriteSetId,
            ) -> Option<viewport_lib::plugin_api::Extent> {
                self.item_type_plugin::<SpritePlugin>(TYPE_NAME)?
                    .set_extent(id, $positions)
            }
        }
    };
}

sprite_writes!(channels::Positions, positions = true, encode = cast_bytes);
sprite_writes!(
    channels::Sprites,
    positions = false,
    encode = encode_sprites
);

/// Sprites keep two stores behind one item struct: a batch drawn as one set,
/// and an instance set drawn per transform. `Uploads` carries one `Id` per
/// implementation, so it covers the plain set and the instance set keeps calls
/// of its own on [`SpriteInstanceUploads`].
impl viewport_lib::plugin_api::Uploads<SpriteItem> for ViewportRenderer {
    type Id = SpriteSetId;

    fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<SpriteSetId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME);
        Ok(host.plugin.upload_set(device, queue, host.resources, item))
    }

    fn begin_upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME);
        Ok(host
            .plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item))
    }

    fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<SpritePlugin>(self, TYPE_NAME);
        host.plugin
            .replace_set(device, queue, host.resources, id, item)
    }
}

impl viewport_lib::plugin_api::Handles<SpriteSetId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<SpriteSetId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME);
        host.plugin.take_set_result(&host.jobs, job)
    }

    fn release(&mut self, id: SpriteSetId) -> bool {
        plugin_mut::<SpritePlugin>(self, TYPE_NAME).drop_set(id)
    }
}

/// The sprite instance-set store, which shares `SpriteItem` with the plain set
/// store and so cannot share its `Uploads` implementation. Its handle type is
/// its own, so taking a finished job and releasing a set go through
/// [`Handles`](viewport_lib::plugin_api::Handles) like every other store.
pub trait SpriteInstanceUploads {
    /// Upload an instance set, returning a handle valid until
    /// [`release`](viewport_lib::plugin_api::Handles::release).
    fn upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> SpriteInstanceSetId;

    /// Start an off-thread upload of an instance set.
    fn begin_upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::resources::JobId;

    /// Replace the sprites behind an instance-set handle, keeping the handle.
    fn replace_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteInstanceSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()>;
}

impl SpriteInstanceUploads for ViewportRenderer {
    fn upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> SpriteInstanceSetId {
        let host = host::<SpritePlugin>(self, TYPE_NAME);
        host.plugin
            .upload_instance_set(device, queue, host.resources, item)
    }

    fn begin_upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::resources::JobId {
        let host = host::<SpritePlugin>(self, TYPE_NAME);
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    fn replace_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteInstanceSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<SpritePlugin>(self, TYPE_NAME);
        host.plugin
            .replace_instance_set(device, queue, host.resources, id, item)
    }
}

impl viewport_lib::plugin_api::Handles<SpriteInstanceSetId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<SpriteInstanceSetId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME);
        host.plugin.take_instance_set_result(&host.jobs, job)
    }

    fn release(&mut self, id: SpriteInstanceSetId) -> bool {
        plugin_mut::<SpritePlugin>(self, TYPE_NAME).drop_instance_set(id)
    }
}
