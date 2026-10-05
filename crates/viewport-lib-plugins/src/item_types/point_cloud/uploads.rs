//! How a renderer uploads, writes and releases point clouds.

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

use super::*;
use crate::item_types::{cast_bytes, host, plugin_mut};

impl viewport_lib::plugin_api::Uploads<PointCloudItem> for ViewportRenderer {
    type Id = PointCloudId;

    fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &PointCloudItem,
    ) -> viewport_lib::error::ViewportResult<PointCloudId> {
        let host = host::<PointCloudPlugin>(self, TYPE_NAME);
        Ok(host.plugin.upload(device, queue, host.resources, item))
    }

    fn begin_upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: PointCloudItem,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<PointCloudPlugin>(self, TYPE_NAME);
        Ok(host
            .plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item))
    }

    fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: PointCloudId,
        item: &PointCloudItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<PointCloudPlugin>(self, TYPE_NAME);
        host.plugin.replace(device, queue, host.resources, id, item)
    }
}

impl viewport_lib::plugin_api::Handles<PointCloudId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<PointCloudId> {
        let host = host::<PointCloudPlugin>(self, TYPE_NAME);
        host.plugin.take_upload_result(&host.jobs, job)
    }

    fn release(&mut self, id: PointCloudId) -> bool {
        plugin_mut::<PointCloudPlugin>(self, TYPE_NAME).drop_stored(id)
    }
}

/// Ranged writes into a stored point cloud, one implementation per channel.
///
/// The bodies differ only in which channel they name and how the caller's
/// elements become bytes, which is the encoding step the channel exists to keep
/// on this side of the API: `Colour` is a caller-facing type and the buffer holds
/// linear RGBA.
macro_rules! point_cloud_writes {
    ($marker:ty, $variant:ident, encode = $encode:expr) => {
        impl viewport_lib::plugin_api::Writes<$marker> for ViewportRenderer {
            fn write_range(
                &mut self,
                _channel: $marker,
                queue: &gpu::Queue,
                id: PointCloudId,
                first_element: u32,
                data: &[<$marker as viewport_lib::plugin_api::Channel>::Input],
            ) -> viewport_lib::error::ViewportResult<()> {
                let bytes: std::borrow::Cow<'_, [u8]> = $encode(data);
                plugin_mut::<PointCloudPlugin>(self, TYPE_NAME).write_channel(
                    queue,
                    id,
                    PointChannel::$variant,
                    <$marker as viewport_lib::plugin_api::Channel>::NAME,
                    first_element,
                    &bytes,
                )
            }

            fn reserve(
                &mut self,
                _channel: $marker,
                device: &gpu::Device,
                queue: &gpu::Queue,
                id: PointCloudId,
                capacity: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<PointCloudPlugin>(self, TYPE_NAME)
                    .reserve_stored(device, queue, id, capacity)
            }

            fn set_len(
                &mut self,
                _channel: $marker,
                id: PointCloudId,
                len: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<PointCloudPlugin>(self, TYPE_NAME).set_stored_len(id, len)
            }

            fn extent(
                &self,
                _channel: $marker,
                id: PointCloudId,
            ) -> Option<viewport_lib::plugin_api::Extent> {
                self.item_type_plugin::<PointCloudPlugin>(TYPE_NAME)?
                    .stored_extent(id, PointChannel::$variant)
            }
        }
    };
}

/// Colours are supplied as [`Colour`](viewport_lib::Colour) and stored as linear
/// RGBA, so this channel is the one that has to build a buffer.
fn encode_colours(data: &[viewport_lib::Colour]) -> std::borrow::Cow<'_, [u8]> {
    let linear: Vec<[f32; 4]> = data.iter().map(|c| c.to_linear_rgba()).collect();
    std::borrow::Cow::Owned(bytemuck::cast_slice(&linear).to_vec())
}

point_cloud_writes!(channels::Positions, Positions, encode = cast_bytes);
point_cloud_writes!(channels::Scalars, Scalars, encode = cast_bytes);
point_cloud_writes!(channels::Colours, Colours, encode = encode_colours);
point_cloud_writes!(channels::Sizes, Sizes, encode = cast_bytes);
point_cloud_writes!(
    channels::Transparencies,
    Transparencies,
    encode = cast_bytes
);

/// Caller-owned buffers for a stored point cloud's channels.
///
/// The counterpart to the writes above: a producer whose points are already on the
/// device points the cloud at its buffer and no bytes move at all.
macro_rules! point_cloud_sources {
    ($marker:ty, $variant:ident) => {
        impl viewport_lib::plugin_api::Sourced<$marker> for ViewportRenderer {
            fn set_source(
                &mut self,
                _channel: $marker,
                device: &gpu::Device,
                id: PointCloudId,
                source: Option<gpu::Buffer>,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<PointCloudPlugin>(self, TYPE_NAME).set_channel_source(
                    device,
                    id,
                    PointChannel::$variant,
                    <$marker as viewport_lib::plugin_api::Channel>::NAME,
                    source,
                )
            }

            fn has_source(&self, _channel: $marker, id: PointCloudId) -> Option<bool> {
                self.item_type_plugin::<PointCloudPlugin>(TYPE_NAME)?
                    .channel_has_source(id, PointChannel::$variant)
            }
        }
    };
}

point_cloud_sources!(channels::Positions, Positions);
point_cloud_sources!(channels::Scalars, Scalars);
point_cloud_sources!(channels::Colours, Colours);
point_cloud_sources!(channels::Sizes, Sizes);
point_cloud_sources!(channels::Transparencies, Transparencies);
