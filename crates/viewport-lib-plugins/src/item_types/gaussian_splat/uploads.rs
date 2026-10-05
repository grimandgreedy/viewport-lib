//! How a renderer uploads, writes and releases Gaussian splat sets.

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

use super::*;
use crate::item_types::{cast_bytes, host, plugin_mut};

impl viewport_lib::plugin_api::Uploads<GaussianSplatData> for ViewportRenderer {
    type Id = GaussianSplatId;

    fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        data: &GaussianSplatData,
    ) -> viewport_lib::error::ViewportResult<GaussianSplatId> {
        plugin_mut::<GaussianSplatPlugin>(self, TYPE_NAME).upload(device, queue, data)
    }

    fn begin_upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        data: GaussianSplatData,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<GaussianSplatPlugin>(self, TYPE_NAME);
        host.plugin.begin_upload(&host.jobs, device, queue, data)
    }

    fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: GaussianSplatId,
        data: &GaussianSplatData,
    ) -> viewport_lib::error::ViewportResult<()> {
        plugin_mut::<GaussianSplatPlugin>(self, TYPE_NAME).replace(device, queue, id, data)
    }
}

impl viewport_lib::plugin_api::Handles<GaussianSplatId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<GaussianSplatId> {
        let host = host::<GaussianSplatPlugin>(self, TYPE_NAME);
        host.plugin.take_upload_result(&host.jobs, job)
    }

    fn release(&mut self, id: GaussianSplatId) -> bool {
        plugin_mut::<GaussianSplatPlugin>(self, TYPE_NAME).free(id)
    }
}

/// Ranged writes into a stored splat set, one implementation per channel.
///
/// Three of the five pad the caller's element on the way in, and two of those
/// also carry the caller's own values through to the CPU mirror the pick paths
/// read.
macro_rules! splat_writes {
    ($marker:ty, $variant:ident, encode = $encode:expr, mirror = $mirror:expr) => {
        impl viewport_lib::plugin_api::Writes<$marker> for ViewportRenderer {
            fn write_range(
                &mut self,
                _channel: $marker,
                queue: &gpu::Queue,
                id: GaussianSplatId,
                first_element: u32,
                data: &[<$marker as viewport_lib::plugin_api::Channel>::Input],
            ) -> viewport_lib::error::ViewportResult<()> {
                let bytes: std::borrow::Cow<'_, [u8]> = $encode(data);
                let mirror: Option<&[[f32; 3]]> = $mirror(data);
                plugin_mut::<GaussianSplatPlugin>(self, TYPE_NAME).write_channel(
                    queue,
                    id,
                    SplatChannel::$variant,
                    <$marker as viewport_lib::plugin_api::Channel>::NAME,
                    first_element,
                    &bytes,
                    mirror,
                )
            }

            fn reserve(
                &mut self,
                _channel: $marker,
                device: &gpu::Device,
                queue: &gpu::Queue,
                id: GaussianSplatId,
                capacity: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<GaussianSplatPlugin>(self, TYPE_NAME)
                    .reserve_stored(device, queue, id, capacity)
            }

            fn set_len(
                &mut self,
                _channel: $marker,
                id: GaussianSplatId,
                len: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<GaussianSplatPlugin>(self, TYPE_NAME).set_stored_len(id, len)
            }

            fn extent(
                &self,
                _channel: $marker,
                id: GaussianSplatId,
            ) -> Option<viewport_lib::plugin_api::Extent> {
                self.item_type_plugin::<GaussianSplatPlugin>(TYPE_NAME)?
                    .stored_extent(id, SplatChannel::$variant)
            }
        }
    };
}

/// Pad a `[f32; 3]` channel to the `vec4` the GPU holds, with a fixed `w`.
fn pad_vec3(data: &[[f32; 3]], w: f32) -> std::borrow::Cow<'static, [u8]> {
    let padded: Vec<[f32; 4]> = data.iter().map(|v| [v[0], v[1], v[2], w]).collect();
    std::borrow::Cow::Owned(bytemuck::cast_slice(&padded).to_vec())
}

fn pad_positions(data: &[[f32; 3]]) -> std::borrow::Cow<'static, [u8]> {
    pad_vec3(data, 1.0)
}

fn pad_scales(data: &[[f32; 3]]) -> std::borrow::Cow<'static, [u8]> {
    pad_vec3(data, 0.0)
}

/// Channels that keep a CPU copy hand the caller's own elements through to it.
fn mirrored(data: &[[f32; 3]]) -> Option<&[[f32; 3]]> {
    Some(data)
}

fn unmirrored<T>(_data: &[T]) -> Option<&'static [[f32; 3]]> {
    None
}

splat_writes!(
    channels::Positions,
    Positions,
    encode = pad_positions,
    mirror = mirrored
);
splat_writes!(
    channels::Scales,
    Scales,
    encode = pad_scales,
    mirror = mirrored
);
splat_writes!(
    channels::Rotations,
    Rotations,
    encode = cast_bytes,
    mirror = unmirrored
);
splat_writes!(
    channels::Opacities,
    Opacities,
    encode = cast_bytes,
    mirror = unmirrored
);
splat_writes!(
    channels::ShCoefficients,
    ShCoefficients,
    encode = cast_bytes,
    mirror = unmirrored
);
