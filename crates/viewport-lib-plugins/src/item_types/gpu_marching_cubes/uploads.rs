//! How a renderer uploads, writes and releases marching-cubes volumes.

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

use super::*;
use crate::item_types::{host, plugin_mut};

/// The marching-cubes volume surface, on the renderer.
///
/// `VolumeData` belongs to `viewport-lib-geometry` and this crate owns neither
/// it nor `ViewportRenderer`, so `Uploads<VolumeData>` cannot be written here:
/// no type in the implementation would be local to this crate. Two of these
/// calls are not uploads anyway, so the content-keyed half of the surface gets
/// verbs of its own. The handle is this crate's, so taking a finished job and
/// releasing a volume go through
/// [`Handles`](viewport_lib::plugin_api::Handles) as usual.
pub trait McVolumes {
    /// Upload a scalar field, pre-allocating every slab's buffers, and return
    /// its handle.
    fn upload_volume_for_mc(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        vol: &viewport_lib_geometry::volume::grid::VolumeData,
    ) -> viewport_lib::error::ViewportResult<McVolumeId>;

    /// Start an off-thread upload of a scalar field. Ownership of `vol`
    /// transfers into the worker, and the handle is minted by
    /// [`upload_result`](viewport_lib::plugin_api::Handles::upload_result).
    fn begin_upload_volume_for_mc(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        vol: viewport_lib_geometry::volume::grid::VolumeData,
    ) -> viewport_lib::resources::JobId;

    /// Feed a volume from a caller-supplied buffer, refreshed before every
    /// dispatch so the isosurface tracks it with no CPU upload.
    ///
    /// The buffer holds one `f32` per volume node in x-fastest order
    /// (`index = x + y * nx + z * nx * ny`), matching `VolumeData::data`,
    /// starting at `offset_bytes`. It needs `COPY_SRC` usage and
    /// `offset_bytes` must be a multiple of 4. The renderer keeps a clone of
    /// the buffer handle; if the consumer reallocates it, call this again with
    /// the new buffer.
    fn set_mc_scalar_source_buffer(
        &mut self,
        id: McVolumeId,
        buffer: gpu::Buffer,
        offset_bytes: u64,
    ) -> viewport_lib::error::ViewportResult<()>;

    /// Detach the external scalar source, freezing the isosurface at the last
    /// field copied in.
    fn clear_mc_scalar_source(&mut self, id: McVolumeId)
    -> viewport_lib::error::ViewportResult<()>;
}

impl McVolumes for ViewportRenderer {
    fn upload_volume_for_mc(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        vol: &viewport_lib_geometry::volume::grid::VolumeData,
    ) -> viewport_lib::error::ViewportResult<McVolumeId> {
        plugin_mut::<GpuMarchingCubesPlugin>(self, TYPE_NAME).upload(device, queue, vol)
    }

    fn begin_upload_volume_for_mc(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        vol: viewport_lib_geometry::volume::grid::VolumeData,
    ) -> viewport_lib::resources::JobId {
        let host = host::<GpuMarchingCubesPlugin>(self, TYPE_NAME);
        host.plugin.begin_upload(&host.jobs, device, queue, vol)
    }

    fn set_mc_scalar_source_buffer(
        &mut self,
        id: McVolumeId,
        buffer: gpu::Buffer,
        offset_bytes: u64,
    ) -> viewport_lib::error::ViewportResult<()> {
        plugin_mut::<GpuMarchingCubesPlugin>(self, TYPE_NAME).set_scalar_source(
            id,
            buffer,
            offset_bytes,
        )
    }

    fn clear_mc_scalar_source(
        &mut self,
        id: McVolumeId,
    ) -> viewport_lib::error::ViewportResult<()> {
        plugin_mut::<GpuMarchingCubesPlugin>(self, TYPE_NAME).clear_scalar_source(id)
    }
}

impl viewport_lib::plugin_api::Handles<McVolumeId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<McVolumeId> {
        let host = host::<GpuMarchingCubesPlugin>(self, TYPE_NAME);
        host.plugin.take_upload_result(&host.jobs, job)
    }

    fn release(&mut self, id: McVolumeId) -> bool {
        plugin_mut::<GpuMarchingCubesPlugin>(self, TYPE_NAME).free(id)
    }
}
