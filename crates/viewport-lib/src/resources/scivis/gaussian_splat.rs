use super::*;

pub use viewport_lib_types::data::point::{GaussianSplatData, ShDegree};
pub use viewport_lib_types::ids::GaussianSplatId;

/// Check that a splat set is non-empty and its per-attribute vectors agree in
/// length. Shared by the sync, async, and replace upload paths.
fn validate_gaussian_splat_data(data: &GaussianSplatData) -> crate::error::ViewportResult<()> {
    if data.positions.is_empty() {
        return Err(crate::error::ViewportError::InvalidGaussianSplatData {
            reason: "empty splat list",
        });
    }
    let n = data.positions.len();
    if data.scales.len() != n || data.rotations.len() != n || data.opacities.len() != n {
        return Err(crate::error::ViewportError::InvalidGaussianSplatData {
            reason: "mismatched buffer lengths",
        });
    }
    Ok(())
}

/// Build the persistent GPU buffers for a splat set and assemble the
/// `GaussianSplatGpuSet`. Assumes `data` already passed
/// [`validate_gaussian_splat_data`]. Shared by the sync `upload_gaussian_splat`,
/// the async worker, and `replace_gaussian_splat` so all three produce identical
/// resources.
fn build_gaussian_splat_set(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    data: &GaussianSplatData,
) -> GaussianSplatGpuSet {
    let count = data.positions.len() as u32;

    // Pad positions/scales/rotations to vec4 (w=1 / w=0 / raw).
    let pos_data: Vec<[f32; 4]> = data
        .positions
        .iter()
        .map(|p| [p[0], p[1], p[2], 1.0])
        .collect();
    let scale_data: Vec<[f32; 4]> = data
        .scales
        .iter()
        .map(|s| [s[0], s[1], s[2], 0.0])
        .collect();
    let rotation_data: Vec<[f32; 4]> = data
        .rotations
        .iter()
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect();

    let buf_size_pos = (pos_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
    let buf_size_scale = (scale_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
    let buf_size_rot = (rotation_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
    let buf_size_opa = (data.opacities.len() * 4).max(4) as u64;
    let buf_size_sh = (data.sh_coefficients.len() * 4).max(4) as u64;

    let position_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_position_buf"),
        size: buf_size_pos,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&position_buf, 0, bytemuck::cast_slice(&pos_data));

    let scale_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_scale_buf"),
        size: buf_size_scale,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&scale_buf, 0, bytemuck::cast_slice(&scale_data));

    let rotation_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_rotation_buf"),
        size: buf_size_rot,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&rotation_buf, 0, bytemuck::cast_slice(&rotation_data));

    let opacity_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_opacity_buf"),
        size: buf_size_opa,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&opacity_buf, 0, bytemuck::cast_slice(&data.opacities));

    let sh_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_sh_buf"),
        size: buf_size_sh,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.sh_coefficients.is_empty() {
        queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&data.sh_coefficients));
    }

    GaussianSplatGpuSet {
        position_buf,
        scale_buf,
        rotation_buf,
        opacity_buf,
        sh_buf,
        sh_degree: data.sh_degree,
        count,
        cpu_positions: std::sync::Arc::new(data.positions.clone()),
        cpu_scales: std::sync::Arc::new(data.scales.clone()),
    }
}

impl DeviceResources {
    /// Upload one Gaussian splat set to the GPU and return its handle.
    ///
    /// Call once per splat set at startup (or when the set changes). The returned
    /// [`GaussianSplatId`] is stable until [`free_gaussian_splat`] is called.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// if `data.positions` is empty or if the lengths of `positions`, `scales`,
    /// `rotations`, and `opacities` do not all match.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use viewport_lib::error::ViewportError;
    /// # use viewport_lib::renderer::{GaussianSplatData, ViewportRenderer};
    /// # fn demo(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device, queue: &viewport_lib::wgpu::Queue) {
    /// let result = renderer.upload_gaussian_splat(device, queue, &GaussianSplatData::default());
    /// assert!(matches!(result, Err(ViewportError::InvalidGaussianSplatData { .. })));
    /// # }
    /// ```
    pub fn upload_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &GaussianSplatData,
    ) -> crate::error::ViewportResult<GaussianSplatId> {
        validate_gaussian_splat_data(data)?;
        let gpu_set = build_gaussian_splat_set(device, queue, data);
        Ok(self.content.gaussian_splat_store.insert_sized(gpu_set))
    }

    /// Replace the contents of an uploaded Gaussian splat set in place, keeping
    /// the same [`GaussianSplatId`](GaussianSplatId).
    ///
    /// Items holding the handle pick up the new splats on the next frame with no
    /// reassignment. The generation check is the in-flight guard: a stale handle
    /// (its slot freed and reused) returns
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) instead of
    /// overwriting whatever now occupies the slot. Use this for content that
    /// changes over time (a re-trained or streamed splat set).
    ///
    /// # Errors
    ///
    /// [`InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// when `data` is empty or its per-attribute vectors disagree in length, or
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` does not
    /// resolve to a live set.
    pub fn replace_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: GaussianSplatId,
        data: &GaussianSplatData,
    ) -> crate::error::ViewportResult<()> {
        validate_gaussian_splat_data(data)?;
        let gpu_set = build_gaussian_splat_set(device, queue, data);
        if self
            .content
            .gaussian_splat_store
            .replace_sized(id, gpu_set)
            .is_some()
        {
            Ok(())
        } else {
            Err(crate::error::ViewportError::StaleHandle {
                index: id.index() as usize,
                count: self.content.gaussian_splat_store.slot_count(),
            })
        }
    }

    /// Remove an uploaded Gaussian splat set by handle.
    pub fn free_gaussian_splat(&mut self, id: GaussianSplatId) {
        self.content.gaussian_splat_store.remove(id);
    }

    /// Start an asynchronous Gaussian splat upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Vec4
    /// padding for positions / scales / rotations and storage buffer
    /// creation + writes all run on a worker thread on a cloned `Device`
    /// and `Queue`. The apply step inserts the prepared `GaussianSplatGpuSet`
    /// into the store and surfaces the resulting [`GaussianSplatId`].
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidGaussianSplatData`] before any job
    /// is submitted when `data.positions` is empty or the per-attribute
    /// vectors disagree in length.
    pub fn begin_upload_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: GaussianSplatData,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        validate_gaussian_splat_data(&data)?;

        let slot = crate::resources::ResultSlot::<GaussianSplatId>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let queue_for_worker = queue.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let gpu_set =
                    build_gaussian_splat_set(&device_for_worker, &queue_for_worker, &data);
                progress.set(0.95);

                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let id = resources.content.gaussian_splat_store.insert_sized(gpu_set);
                        slot_for_apply.set(id);
                    }),
                ))
            })
        };

        self.job_results
            .gaussian_splat
            .lock()
            .expect("gaussian splat result map poisoned")
            .insert(id, slot);
        Ok(id)
    }

    /// Take the [`GaussianSplatId`](GaussianSplatId) produced by a
    /// completed [`begin_upload_gaussian_splat`](Self::begin_upload_gaussian_splat) job.
    pub fn upload_result_gaussian_splat(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<GaussianSplatId> {
        let mut map = self
            .job_results
            .gaussian_splat
            .lock()
            .expect("gaussian splat result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(splat_id) => {
                map.remove(&id);
                Ok(splat_id)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }
}

#[cfg(test)]
mod async_tests {
    use super::GaussianSplatData;
    use crate::DeviceResources;
    use crate::resources::UploadStatus;

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

    fn sample_splats(n: usize) -> GaussianSplatData {
        let mut data = GaussianSplatData::default();
        data.positions = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
        data.scales = vec![[0.1, 0.1, 0.1]; n];
        data.rotations = vec![[0.0, 0.0, 0.0, 1.0]; n];
        data.opacities = vec![0.5; n];
        data
    }

    #[test]
    fn stale_splat_handle_does_not_alias_after_slot_reuse() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // Upload a splat set, then remove it. The handle is now stale.
        let id1 = resources
            .upload_gaussian_splat(&device, &queue, &sample_splats(8))
            .unwrap();
        assert!(resources.content.gaussian_splat_store.get(id1).is_some());
        resources.free_gaussian_splat(id1);
        assert!(
            resources.content.gaussian_splat_store.get(id1).is_none(),
            "a removed handle must not resolve"
        );

        // The next upload reuses the freed slot at a new generation.
        let id2 = resources
            .upload_gaussian_splat(&device, &queue, &sample_splats(4))
            .unwrap();
        assert_eq!(id1.index(), id2.index(), "the freed slot should be reused");
        assert_ne!(id1, id2, "the reused slot must carry a new generation");
        assert!(resources.content.gaussian_splat_store.get(id2).is_some());
        assert!(
            resources.content.gaussian_splat_store.get(id1).is_none(),
            "the stale handle must not alias the set now occupying its slot"
        );
    }

    #[test]
    fn begin_upload_gaussian_splat_validates() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let err = resources
            .begin_upload_gaussian_splat(&device, &queue, GaussianSplatData::default())
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::InvalidGaussianSplatData { .. }
        ));
    }

    #[test]
    fn begin_upload_gaussian_splat_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources
            .begin_upload_gaussian_splat(&device, &queue, sample_splats(8))
            .expect("job submitted");
        for _ in 0..200 {
            resources.process_uploads(&device, &queue);
            match resources.upload_status(job) {
                UploadStatus::Ready => break,
                UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                UploadStatus::Unknown => panic!("job id disappeared"),
            }
        }
        let _id = resources.upload_result_gaussian_splat(job).expect("ready");
    }

    #[test]
    fn replace_gaussian_splat_keeps_handle_and_updates_bytes() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let id = resources
            .upload_gaussian_splat(&device, &queue, &sample_splats(8))
            .unwrap();
        let bytes_before = resources.resident_bytes().gaussian_splat_bytes;
        assert!(
            bytes_before > 0,
            "an uploaded set must count resident bytes"
        );

        // Replacing with a smaller set keeps the handle valid and shrinks bytes.
        resources
            .replace_gaussian_splat(&device, &queue, id, &sample_splats(2))
            .expect("replace on a live handle succeeds");
        assert!(resources.content.gaussian_splat_store.get(id).is_some());
        let bytes_after = resources.resident_bytes().gaussian_splat_bytes;
        assert!(
            bytes_after < bytes_before,
            "replacing with fewer splats must reduce resident bytes"
        );

        // A stale handle is rejected, not silently applied.
        resources.free_gaussian_splat(id);
        assert_eq!(resources.resident_bytes().gaussian_splat_bytes, 0);
        let err = resources
            .replace_gaussian_splat(&device, &queue, id, &sample_splats(2))
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::StaleHandle { .. }
        ));
    }
}

/// Persistent GPU state for one uploaded Gaussian splat set.
pub(crate) struct GaussianSplatGpuSet {
    /// Positions as vec4<f32> (w=1), one per splat.
    pub position_buf: crate::gpu::Buffer,
    /// Scales as vec4<f32> (w=0), one per splat.
    pub scale_buf: crate::gpu::Buffer,
    /// Rotations as vec4<f32> [x,y,z,w], one per splat.
    pub rotation_buf: crate::gpu::Buffer,
    /// Opacities as f32, one per splat.
    pub opacity_buf: crate::gpu::Buffer,
    /// SH coefficients as f32, count = splat_count * sh_degree.coeff_count().
    pub sh_buf: crate::gpu::Buffer,
    /// SH degree for this set.
    pub sh_degree: ShDegree,
    /// Number of splats.
    pub count: u32,
    /// CPU positions kept for picking and the wireframe overlay
    /// (object-space). Shared so per-frame consumers snapshot without
    /// copying the set.
    pub cpu_positions: std::sync::Arc<Vec<[f32; 3]>>,
    /// CPU scales kept for picking and the wireframe overlay.
    pub cpu_scales: std::sync::Arc<Vec<[f32; 3]>>,
}

impl crate::resources::handle::GpuByteSize for GaussianSplatGpuSet {
    /// Resident GPU bytes for the persistent source buffers (position, scale,
    /// rotation, opacity, SH). Per-viewport sort scratch is derived and grows
    /// lazily, so it is not counted here.
    fn gpu_bytes(&self) -> u64 {
        self.position_buf.size()
            + self.scale_buf.size()
            + self.rotation_buf.size()
            + self.opacity_buf.size()
            + self.sh_buf.size()
    }
}

/// Slotted store for Gaussian splat sets.
///
/// A removed set leaves an empty slot that a later insert reuses. Each slot
/// carries a generation bumped on removal, and a [`GaussianSplatId`] captures
/// the generation it was issued against, so a stale handle resolves to `None`
/// rather than aliasing the set now in its slot. An entry's byte charge is its
/// [`GpuByteSize::gpu_bytes`](crate::resources::handle::GpuByteSize::gpu_bytes),
/// and its revision is what the item type keys its per-viewport sort scratch on.
pub(crate) type GaussianSplatStore =
    crate::resources::handle::SlotStore<GaussianSplatGpuSet, GaussianSplatId>;
