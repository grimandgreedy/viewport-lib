//! GPU marching cubes.
//!
//! Three-pass GPU compute pipeline for isosurface extraction:
//!   1. Classify       computes case index and triangle count per cell.
//!   2. Prefix sum     hierarchical exclusive scan to build triangle offsets.
//!   3. Generate       interpolates vertex positions and normals into a vertex buffer.
//!
//! The compute pipelines and the draws that consume their output belong to the
//! marching-cubes item type; what lives here is the store of uploaded volumes
//! and their per-slab buffers, which is consumer API (`upload_volume_for_mc`
//! and [`McVolumeId`]).

use crate::gpu::util::DeviceExt as _;

use crate::{geometry::marching_cubes::VolumeData, resources::DeviceResources};

/// The store of uploaded marching-cubes volumes, each split into Z-axis slabs.
///
/// The extraction pipelines and the draws that consume their output belong to
/// the item type; this is only the resident volume data, which consumers
/// address by [`McVolumeId`].
#[derive(Default)]
pub(crate) struct McResources {
    pub(crate) volumes: crate::resources::handle::SlotStore<McVolumeGpuData, McVolumeId>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

crate::resources::handle::slot_handle! {
    /// Handle to a volume scalar field uploaded for GPU marching cubes.
    ///
    /// Returned by [`DeviceResources::upload_volume_for_mc`]. Pass to
    /// [`GpuMarchingCubesItem`] to select which volume to triangulate each frame.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. A handle whose volume was removed (its slot freed and reused
    /// by a later upload) resolves to nothing on lookup, so it cannot alias the
    /// volume now in its slot.
    pub struct McVolumeId;
}

// ---------------------------------------------------------------------------
// GPU-internal types
// ---------------------------------------------------------------------------

/// GPU buffers for one Z-axis slab of an uploaded volume.
///
/// A slab covers `dims[2]` scalar Z-layers (`dims[2] - 1` cell layers).
/// Adjacent slabs share exactly one scalar Z-layer at their boundary so MC
/// edge interpolation produces no seams.
pub(crate) struct McSlabGpuData {
    pub scalar_buf: crate::gpu::Buffer, // f32 per slab node; STORAGE | COPY_DST
    /// Byte offset of this slab's first scalar in the full linear volume
    /// (x-fastest node order). Used to source the slab's range out of an
    /// external scalar buffer with one `copy_buffer_to_buffer` per slab.
    pub scalar_byte_offset: u64,
    pub counts_buf: crate::gpu::Buffer, // u32 per slab cell; STORAGE
    pub case_idx_buf: crate::gpu::Buffer, // u32 per slab cell; STORAGE
    pub offsets_buf: crate::gpu::Buffer, // u32 per slab cell; STORAGE
    pub block_sums_buf: crate::gpu::Buffer, // u32 per slab block; STORAGE
    pub vertex_buf: crate::gpu::Buffer, // f32 * 6 per vertex; STORAGE | VERTEX
    pub indirect_buf: crate::gpu::Buffer, // 4 u32; STORAGE | INDIRECT (surface draw)
    pub wire_indirect_buf: crate::gpu::Buffer, // 4 u32; STORAGE | INDIRECT (wireframe draw)
    pub dims: [u32; 3],                 // [nx, ny, slab_nz] (scalar layers)
    pub origin: [f32; 3],               // world origin; z is offset per slab
    pub spacing: [f32; 3],
    pub cell_count: u32,
    pub block_count: u32,
}

/// Persistent GPU resources for one uploaded volume, split into Z-axis slabs.
///
/// Z-axis chunking keeps every allocation within `device.limits().max_buffer_size`
/// regardless of volume size. The single-slab path is equivalent to the old layout.
pub(crate) struct McVolumeGpuData {
    pub slabs: Vec<McSlabGpuData>,
    /// Full-volume scalar dims `[nx, ny, nz]`, kept for validating an
    /// external scalar source against the volume's node count.
    pub dims: [u32; 3],
    /// When `Some`, the slab scalar buffers are refreshed from this
    /// caller-supplied buffer before every MC dispatch, so the isosurface
    /// tracks the buffer's contents with no CPU upload.
    pub external_scalar: Option<McExternalScalarSource>,
}

/// A caller-supplied buffer feeding a volume's scalar field.
pub(crate) struct McExternalScalarSource {
    pub buffer: crate::gpu::Buffer,
    /// Byte offset of the volume's first scalar inside `buffer`.
    pub offset_bytes: u64,
}

impl crate::resources::handle::GpuByteSize for McVolumeGpuData {
    /// Resident GPU bytes across every slab buffer of this volume.
    fn gpu_bytes(&self) -> u64 {
        self.slabs
            .iter()
            .map(|s| {
                s.scalar_buf.size()
                    + s.counts_buf.size()
                    + s.case_idx_buf.size()
                    + s.offsets_buf.size()
                    + s.block_sums_buf.size()
                    + s.vertex_buf.size()
                    + s.indirect_buf.size()
                    + s.wire_indirect_buf.size()
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Volume upload (impl DeviceResources)
// ---------------------------------------------------------------------------

impl DeviceResources {
    /// Upload a [`VolumeData`] to GPU, pre-allocating all intermediate and output
    /// buffers for GPU marching cubes.
    ///
    /// The returned [`McVolumeId`] is stable until [`free_mc_volume`] is called.
    ///
    /// Returns `Err(ViewportError::McBufferTooLarge)` if any required buffer exceeds
    /// the device's `max_buffer_size`; the caller should fall back to CPU isosurface
    /// extraction.
    ///
    /// Prefer [`ViewportRenderer::upload_volume_for_mc`](crate::renderer::ViewportRenderer::upload_volume_for_mc),
    /// which stays reachable when an item type holds its own storage.
    pub fn upload_volume_for_mc(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vol: &VolumeData,
    ) -> crate::ViewportResult<McVolumeId> {
        let gpu_data = build_mc_volume_gpu_data(device, queue, vol)?;
        Ok(self.insert_mc_volume_gpu_data(gpu_data))
    }

    /// Main-thread half of an async marching-cubes volume upload: insert
    /// pre-built GPU data into the store and return its handle. Reuses a freed
    /// slot when one is available, carrying that slot's current generation so a
    /// stale handle to the previous occupant no longer resolves.
    pub(crate) fn insert_mc_volume_gpu_data(&mut self, gpu_data: McVolumeGpuData) -> McVolumeId {
        self.mc.volumes.insert_sized(gpu_data)
    }

    /// Look up a live volume by handle, validating the generation. Returns
    /// `None` for a stale handle, a freed slot, or an out-of-range index.
    pub(crate) fn mc_volume(&self, id: McVolumeId) -> Option<&McVolumeGpuData> {
        self.mc.volumes.get(id)
    }

    /// Feed the volume's scalar field from a caller-supplied same-device buffer.
    ///
    /// The buffer holds one `f32` per volume node in x-fastest order
    /// (`index = x + y * nx + z * nx * ny`), matching `VolumeData::data`,
    /// starting at `offset_bytes`. While the source is set, the renderer
    /// copies the field into its internal slab buffers (GPU to GPU, one copy
    /// per slab) before every marching-cubes dispatch, so the isosurface
    /// tracks whatever the consumer's compute passes last wrote with no CPU
    /// upload. This is also the path for animating a density field.
    ///
    /// The buffer needs `COPY_SRC` usage. `offset_bytes` must be a multiple
    /// of 4. The renderer keeps a clone of the buffer handle; if the
    /// consumer reallocates it, call this again with the new buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `id` does not resolve to a live volume,
    /// [`ViewportError::ExternalBufferUsageMissing`](crate::error::ViewportError::ExternalBufferUsageMissing)
    /// if the buffer lacks `COPY_SRC`, or
    /// [`ViewportError::McScalarSourceMismatch`](crate::error::ViewportError::McScalarSourceMismatch)
    /// if the offset is misaligned or the volume's scalars do not fit in the
    /// buffer past `offset_bytes`.
    ///
    /// Prefer [`ViewportRenderer::set_mc_scalar_source_buffer`](crate::renderer::ViewportRenderer::set_mc_scalar_source_buffer),
    /// which stays reachable when an item type holds its own storage.
    pub fn set_mc_scalar_source_buffer(
        &mut self,
        id: McVolumeId,
        buffer: crate::gpu::Buffer,
        offset_bytes: u64,
    ) -> crate::ViewportResult<()> {
        if !buffer.usage().contains(crate::gpu::BufferUsages::COPY_SRC) {
            return Err(crate::ViewportError::ExternalBufferUsageMissing {
                missing: "COPY_SRC",
            });
        }
        let store_len = self.mc.volumes.slot_count();
        let vol = self
            .mc
            .volumes
            .get_mut(id)
            .ok_or(crate::ViewportError::StaleHandle {
                index: id.index(),
                count: store_len,
            })?;
        let [nx, ny, nz] = vol.dims;
        let needed_bytes = nx as u64 * ny as u64 * nz as u64 * 4;
        let available_bytes = buffer.size().saturating_sub(offset_bytes);
        if offset_bytes % 4 != 0 || needed_bytes > available_bytes {
            return Err(crate::ViewportError::McScalarSourceMismatch {
                needed_bytes,
                available_bytes,
                offset_bytes,
            });
        }
        vol.external_scalar = Some(McExternalScalarSource {
            buffer,
            offset_bytes,
        });
        Ok(())
    }

    /// Detach the external scalar source. The slab buffers keep whatever was
    /// last copied in, so the isosurface freezes at the final field.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `id` does not resolve to a live volume.
    ///
    /// Prefer [`ViewportRenderer::clear_mc_scalar_source`](crate::renderer::ViewportRenderer::clear_mc_scalar_source),
    /// which stays reachable when an item type holds its own storage.
    pub fn clear_mc_scalar_source(&mut self, id: McVolumeId) -> crate::ViewportResult<()> {
        let store_len = self.mc.volumes.slot_count();
        let vol = self
            .mc
            .volumes
            .get_mut(id)
            .ok_or(crate::ViewportError::StaleHandle {
                index: id.index(),
                count: store_len,
            })?;
        vol.external_scalar = None;
        Ok(())
    }
}

/// CPU + GPU-buffer work for an MC volume upload, factored out so the same
/// code can run on a worker thread for the async path.
pub(crate) fn build_mc_volume_gpu_data(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    vol: &VolumeData,
) -> crate::ViewportResult<McVolumeGpuData> {
    {
        let [nx, ny, nz] = vol.dims;
        // The vertex buffer is bound as both STORAGE (compute) and VERTEX (render).
        // The binding limit for compute shaders is max_storage_buffer_binding_size, which
        // is often half of max_buffer_size (e.g. 128 MiB vs 256 MiB). Use the smaller of
        // the two so slab sizing respects both constraints.
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        let max_buf = device.limits().max_buffer_size;
        let max_limit = max_binding.min(max_buf);

        // Worst-case vertex buffer bytes per Z-cell-layer:
        // (nx-1)*(ny-1) cells x 5 triangles x 3 vertices x 24 bytes = cells_xy x 360.
        // Compute how many Z-cell layers fit within the effective limit.
        let cells_xy = (nx - 1) as u64 * (ny - 1) as u64;
        let max_cells_per_slab = max_limit / (15 * 24);
        let z_cells_per_slab = if cells_xy > 0 {
            (max_cells_per_slab / cells_xy).min((nz - 1) as u64) as u32
        } else {
            nz - 1
        };
        if z_cells_per_slab == 0 {
            // Even a single Z-layer of cells exceeds the effective binding limit.
            return Err(crate::ViewportError::McBufferTooLarge {
                buffer: "vertex_buf",
                needed: cells_xy * 15 * 24,
                limit: max_limit,
            });
        }

        let nz_cells_total = nz - 1;
        let slab_count = nz_cells_total.div_ceil(z_cells_per_slab);
        let nodes_per_z = (nx * ny) as usize;

        let mut slabs = Vec::with_capacity(slab_count as usize);

        for s in 0..slab_count {
            let z_cell_start = s * z_cells_per_slab;
            let z_cell_end = (z_cell_start + z_cells_per_slab).min(nz_cells_total);
            let slab_z_cells = z_cell_end - z_cell_start; // cell layers in this slab
            let slab_nz = slab_z_cells + 1; // scalar layers in this slab

            // slab_cell_count is bounded by max_cells_per_slab, which fits in u32
            // at any realistic max_buffer_size value.
            let slab_cell_count = (cells_xy * slab_z_cells as u64) as u32;
            let slab_block_count = slab_cell_count.div_ceil(256);
            let slab_cell_bytes = (slab_cell_count as u64) * 4;
            let slab_block_bytes = (slab_block_count as u64) * 4;
            // At most 15 vertices per cell (5 triangles x 3 vertices) x 24 bytes each.
            let slab_vertex_bytes = (slab_cell_count as u64) * 15 * 24;

            // Scalar data is x-fastest: index = x + y*nx + z*nx*ny.
            // A Z-slab covering scalar layers z_cell_start..z_cell_start+slab_nz is
            // a contiguous slice, no copying required.
            let scalar_start = z_cell_start as usize * nodes_per_z;
            let scalar_end = (z_cell_start + slab_nz) as usize * nodes_per_z;
            let slab_origin_z = vol.origin[2] + z_cell_start as f32 * vol.spacing[2];

            let scalar_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("mc_scalar_buf"),
                contents: bytemuck::cast_slice(&vol.data[scalar_start..scalar_end]),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            });
            let counts_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_counts_buf"),
                size: slab_cell_bytes,
                usage: crate::gpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let case_idx_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_case_idx_buf"),
                size: slab_cell_bytes,
                usage: crate::gpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let offsets_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_offsets_buf"),
                size: slab_cell_bytes,
                usage: crate::gpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let block_sums_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_block_sums_buf"),
                size: slab_block_bytes,
                usage: crate::gpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let vertex_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_vertex_buf"),
                size: slab_vertex_bytes,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
            let initial_indirect = bytemuck::cast_slice(&[0u32, 1u32, 0u32, 0u32]);
            let indirect_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("mc_indirect_buf"),
                // Initial: 0 vertices, 1 instance, 0 first_vertex, 0 first_instance.
                contents: initial_indirect,
                usage: crate::gpu::BufferUsages::STORAGE
                    | crate::gpu::BufferUsages::INDIRECT
                    | crate::gpu::BufferUsages::COPY_DST,
            });
            let wire_indirect_buf =
                device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                    label: Some("mc_wire_indirect_buf"),
                    contents: initial_indirect,
                    usage: crate::gpu::BufferUsages::STORAGE
                        | crate::gpu::BufferUsages::INDIRECT
                        | crate::gpu::BufferUsages::COPY_DST,
                });

            slabs.push(McSlabGpuData {
                scalar_buf,
                scalar_byte_offset: scalar_start as u64 * 4,
                counts_buf,
                case_idx_buf,
                offsets_buf,
                block_sums_buf,
                vertex_buf,
                indirect_buf,
                wire_indirect_buf,
                dims: [nx, ny, slab_nz],
                origin: [vol.origin[0], vol.origin[1], slab_origin_z],
                spacing: vol.spacing,
                cell_count: slab_cell_count,
                block_count: slab_block_count,
            });
        }

        let _ = queue;

        Ok(McVolumeGpuData {
            slabs,
            dims: vol.dims,
            external_scalar: None,
        })
    }
}

impl DeviceResources {
    /// Start an asynchronous marching-cubes-ready volume upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Slab
    /// sizing, scalar buffer allocation, and intermediate / output buffer
    /// allocation run on a worker thread on cloned `Device` and `Queue`
    /// handles. The apply step inserts the resulting GPU buffers into the
    /// MC volume store; once `UploadStatus::Ready`, call
    /// [`upload_result_volume_mc`](Self::upload_result_volume_mc) to take
    /// the [`McVolumeId`].
    ///
    /// Ownership of `vol` transfers into the worker.
    ///
    /// # Errors
    ///
    /// The worker surfaces
    /// [`ViewportError::McBufferTooLarge`](crate::error::ViewportError::McBufferTooLarge)
    /// through [`UploadStatus::Failed`] when the device's
    /// `max_storage_buffer_binding_size` cannot fit a single Z-cell layer.
    pub fn begin_upload_volume_for_mc(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vol: VolumeData,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<McVolumeId>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let queue_for_worker = queue.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let gpu_data =
                    build_mc_volume_gpu_data(&device_for_worker, &queue_for_worker, &vol)?;
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let id = resources.insert_mc_volume_gpu_data(gpu_data);
                        slot_for_apply.set(id);
                    }),
                ))
            })
        };

        self.job_results
            .volume_mc
            .lock()
            .expect("volume mc result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`McVolumeId`] produced by a completed
    /// [`begin_upload_volume_for_mc`](Self::begin_upload_volume_for_mc) job.
    pub fn upload_result_volume_mc(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<McVolumeId> {
        let mut map = self
            .job_results
            .volume_mc
            .lock()
            .expect("volume mc result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(vid) => {
                map.remove(&id);
                Ok(vid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Free a MC volume: drop its slab GPU buffers to reclaim the memory now,
    /// mark the slot free, and bump its generation so a stale handle no longer
    /// resolves. The emptied slot is reused by a later upload. Dropping the
    /// buffers drops this volume out of [`resident_bytes`](Self::resident_bytes)
    /// immediately (wgpu defers the real GPU free until in-flight commands that
    /// reference the buffers complete).
    ///
    /// Prefer [`ViewportRenderer::free_mc_volume`](crate::renderer::ViewportRenderer::free_mc_volume),
    /// which stays reachable when an item type holds its own storage.
    pub fn free_mc_volume(&mut self, id: McVolumeId) {
        self.mc.volumes.remove(id);
    }

    /// Total resident GPU bytes across every live MC volume.
    pub(crate) fn mc_volume_resident_bytes(&self) -> u64 {
        self.mc.volumes.allocated_bytes()
    }
}

#[cfg(test)]
mod residency_tests {
    use crate::DeviceResources;
    use crate::geometry::marching_cubes::VolumeData;

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn sample_volume() -> VolumeData {
        let dims = [4u32, 4, 4];
        let data = (0..(dims[0] * dims[1] * dims[2]))
            .map(|i| (i % 2) as f32)
            .collect();
        VolumeData {
            data,
            dims,
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn stale_mc_volume_handle_does_not_alias_after_slot_reuse() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let id1 = resources
            .upload_volume_for_mc(&device, &queue, &sample_volume())
            .unwrap();
        assert!(resources.mc_volume(id1).is_some());
        resources.free_mc_volume(id1);
        assert!(
            resources.mc_volume(id1).is_none(),
            "a removed handle must not resolve"
        );

        // The next upload reuses the freed slot at a new generation.
        let id2 = resources
            .upload_volume_for_mc(&device, &queue, &sample_volume())
            .unwrap();
        assert_eq!(id1.index(), id2.index(), "the freed slot should be reused");
        assert_ne!(id1, id2, "the reused slot must carry a new generation");
        assert!(resources.mc_volume(id2).is_some());
        assert!(
            resources.mc_volume(id1).is_none(),
            "the stale handle must not alias the volume now occupying its slot"
        );
    }

    fn scalar_buffer(
        device: &crate::gpu::Device,
        bytes: u64,
        usage: crate::gpu::BufferUsages,
    ) -> crate::gpu::Buffer {
        device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("test_scalar_src"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn mc_scalar_source_set_clear_roundtrip() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources
            .upload_volume_for_mc(&device, &queue, &sample_volume())
            .unwrap();

        // 4x4x4 volume = 64 nodes = 256 bytes; source sits at offset 64.
        let buf = scalar_buffer(
            &device,
            256 + 64,
            crate::gpu::BufferUsages::COPY_SRC | crate::gpu::BufferUsages::COPY_DST,
        );
        resources.set_mc_scalar_source_buffer(id, buf, 64).unwrap();
        {
            let vol = resources.mc_volume(id).unwrap();
            let src = vol.external_scalar.as_ref().unwrap();
            assert_eq!(src.offset_bytes, 64);
        }

        resources.clear_mc_scalar_source(id).unwrap();
        assert!(resources.mc_volume(id).unwrap().external_scalar.is_none());
    }

    #[test]
    fn mc_scalar_source_rejects_bad_inputs() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources
            .upload_volume_for_mc(&device, &queue, &sample_volume())
            .unwrap();

        // Too small: 64 nodes need 256 bytes.
        let small = scalar_buffer(&device, 128, crate::gpu::BufferUsages::COPY_SRC);
        assert!(matches!(
            resources.set_mc_scalar_source_buffer(id, small, 0),
            Err(crate::ViewportError::McScalarSourceMismatch {
                needed_bytes: 256,
                available_bytes: 128,
                ..
            })
        ));

        // Misaligned offset.
        let buf = scalar_buffer(&device, 512, crate::gpu::BufferUsages::COPY_SRC);
        assert!(matches!(
            resources.set_mc_scalar_source_buffer(id, buf, 2),
            Err(crate::ViewportError::McScalarSourceMismatch { .. })
        ));

        // Missing COPY_SRC usage.
        let storage_only = scalar_buffer(&device, 256, crate::gpu::BufferUsages::STORAGE);
        assert!(matches!(
            resources.set_mc_scalar_source_buffer(id, storage_only, 0),
            Err(crate::ViewportError::ExternalBufferUsageMissing {
                missing: "COPY_SRC"
            })
        ));

        // Stale handle after free.
        resources.free_mc_volume(id);
        let buf = scalar_buffer(&device, 256, crate::gpu::BufferUsages::COPY_SRC);
        assert!(matches!(
            resources.set_mc_scalar_source_buffer(id, buf, 0),
            Err(crate::ViewportError::StaleHandle { .. })
        ));
        assert!(matches!(
            resources.clear_mc_scalar_source(id),
            Err(crate::ViewportError::StaleHandle { .. })
        ));
    }

    #[test]
    fn free_mc_volume_reclaims_resident_bytes() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let start = resources.resident_bytes().mc_volume_bytes;
        let id = resources
            .upload_volume_for_mc(&device, &queue, &sample_volume())
            .unwrap();
        let after_upload = resources.resident_bytes().mc_volume_bytes;
        assert!(
            after_upload > start,
            "uploading a volume must increase resident volume bytes"
        );

        resources.free_mc_volume(id);
        assert_eq!(
            resources.resident_bytes().mc_volume_bytes,
            start,
            "freeing a volume must drop its slab buffers out of the resident total"
        );
    }
}
