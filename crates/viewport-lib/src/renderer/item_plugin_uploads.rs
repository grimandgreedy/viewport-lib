//! The per-type upload calls for the item types that hold their own content.
//!
//! Each of these forwards to the item type that owns the content, so they live
//! on [`ViewportRenderer`] rather than on `DeviceResources`: the renderer owns
//! the registered item types, and an upload that has to reach a type's own
//! storage can only be reached from this level.
//!
//! The private `*_host` / `*_plugin_mut` accessors interleaved with them turn a
//! built-in type name into that type's concrete plugin, which is the lookup
//! plus downcast every call here needs. One per type, rather than one per
//! method.
//!
//! Those lookups cannot fail. A type the renderer installs registers at
//! construction, there is no call that unregisters one, and
//! [`with_item_type_plugin`](crate::renderer::ViewportRenderer::with_item_type_plugin)
//! refuses a plugin claiming one of their names, so the name resolves and the
//! downcast is to the type that put itself there.

use super::*;

impl ViewportRenderer {
    /// The registered polyline item type, which holds the uploaded curves.
    fn polyline_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::polyline::PolylinePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::polyline::TYPE_NAME)
            .expect(
                "the built-in polyline item type registers at construction, under a name nothing else can take",
            )
    }

    /// The registered glyph item type, which holds the uploaded sets.
    fn glyph_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::glyph::GlyphPlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::glyph::TYPE_NAME)
            .expect(
                "the built-in glyph item type registers at construction, under a name nothing else can take",
            )
    }

    /// Upload a scalar volume for GPU marching cubes, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_volume_for_mc(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vol: &crate::geometry::marching_cubes::VolumeData,
    ) -> crate::ViewportResult<crate::resources::McVolumeId> {
        self.gpu_marching_cubes_plugin_mut()?
            .upload(device, queue, vol)
    }

    /// Release a marching-cubes volume and its slab buffers.
    ///
    /// Dropping the buffers takes the volume out of
    /// [`resident_bytes`](Self::resident_bytes) immediately; wgpu defers the
    /// real GPU free until in-flight commands referencing them complete. The
    /// emptied slot is reused by a later upload, at a new generation, so the
    /// freed handle cannot alias its successor.
    pub fn free_mc_volume(&mut self, id: crate::resources::McVolumeId) {
        if let Ok(plugin) = self.gpu_marching_cubes_plugin_mut() {
            plugin.free(id);
        }
    }

    /// Feed a marching-cubes volume from a caller-supplied buffer, refreshed
    /// before every dispatch so the isosurface tracks it with no CPU upload.
    ///
    /// The buffer holds one `f32` per volume node in x-fastest order
    /// (`index = x + y * nx + z * nx * ny`), matching `VolumeData::data`,
    /// starting at `offset_bytes`. It needs `COPY_SRC` usage and
    /// `offset_bytes` must be a multiple of 4. The renderer keeps a clone of
    /// the buffer handle; if the consumer reallocates it, call this again with
    /// the new buffer.
    ///
    /// # Errors
    ///
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` does
    /// not resolve to a live volume,
    /// [`ExternalBufferUsageMissing`](crate::error::ViewportError::ExternalBufferUsageMissing)
    /// if the buffer lacks `COPY_SRC`, or
    /// [`McScalarSourceMismatch`](crate::error::ViewportError::McScalarSourceMismatch)
    /// if the offset is misaligned or the volume's scalars do not fit in the
    /// buffer past `offset_bytes`.
    pub fn set_mc_scalar_source_buffer(
        &mut self,
        id: crate::resources::McVolumeId,
        buffer: crate::gpu::Buffer,
        offset_bytes: u64,
    ) -> crate::ViewportResult<()> {
        self.gpu_marching_cubes_plugin_mut()?
            .set_scalar_source(id, buffer, offset_bytes)
    }

    /// Detach the external scalar source, freezing the isosurface at the last
    /// field copied in.
    ///
    /// # Errors
    ///
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` does
    /// not resolve to a live volume.
    pub fn clear_mc_scalar_source(
        &mut self,
        id: crate::resources::McVolumeId,
    ) -> crate::ViewportResult<()> {
        self.gpu_marching_cubes_plugin_mut()?
            .clear_scalar_source(id)
    }

    /// The registered GPU marching cubes item type, which holds the uploaded
    /// volumes.
    fn gpu_marching_cubes_plugin_mut(
        &mut self,
    ) -> crate::error::ViewportResult<
        &mut crate::renderer::item_plugins::gpu_marching_cubes::GpuMarchingCubesPlugin,
    > {
        let name = crate::renderer::item_plugins::gpu_marching_cubes::TYPE_NAME;
        self.item_type_plugin_mut(name)
            .ok_or(crate::error::ViewportError::ItemTypePluginMissing { type_name: name })
    }
}

// ---------------------------------------------------------------------------
// The standard upload surface, one implementation per item type with a store
// ---------------------------------------------------------------------------

/// Implement [`Uploads`](crate::plugin_api::Uploads) for one item type whose
/// plugin exposes the five standard store calls under their usual names.
macro_rules! standard_uploads {
    ($item:ty, $id:ty, $host:ident) => {
        impl crate::plugin_api::Uploads<$item> for ViewportRenderer {
            type Id = $id;

            fn upload(
                &mut self,
                device: &crate::gpu::Device,
                queue: &crate::gpu::Queue,
                item: &$item,
            ) -> crate::error::ViewportResult<$id> {
                let host = self.$host();
                Ok(host.plugin.upload(device, queue, host.resources, item))
            }

            fn begin_upload(
                &mut self,
                device: &crate::gpu::Device,
                queue: &crate::gpu::Queue,
                item: $item,
            ) -> crate::error::ViewportResult<crate::resources::JobId> {
                let host = self.$host();
                Ok(host
                    .plugin
                    .begin_upload(&host.jobs, device, queue, host.resources, item))
            }

            fn replace(
                &mut self,
                device: &crate::gpu::Device,
                queue: &crate::gpu::Queue,
                id: $id,
                item: &$item,
            ) -> crate::error::ViewportResult<()> {
                let host = self.$host();
                host.plugin.replace(device, queue, host.resources, id, item)
            }
        }

        impl crate::plugin_api::Handles<$id> for ViewportRenderer {
            fn upload_result(
                &mut self,
                job: crate::resources::JobId,
            ) -> crate::error::ViewportResult<$id> {
                let host = self.$host();
                host.plugin.take_upload_result(&host.jobs, job)
            }

            fn release(&mut self, id: $id) -> bool {
                self.$host().plugin.drop_stored(id)
            }
        }
    };
}

standard_uploads!(
    crate::renderer::PolylineItem,
    crate::resources::PolylineId,
    polyline_host
);
standard_uploads!(
    crate::renderer::GlyphItem,
    crate::resources::GlyphSetId,
    glyph_host
);
