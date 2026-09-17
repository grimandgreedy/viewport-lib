use crate::resources::*;

/// Choose the direct-volume 3D texture format from the device's capabilities.
///
/// The scalar field must be linearly filterable so the ray-march reconstructs it
/// with trilinear interpolation rather than blocky nearest-neighbor. `R32Float`
/// is only filterable when the device enabled `FLOAT32_FILTERABLE`, so:
///
/// - feature enabled  -> `R32Float`: full precision, native trilinear.
/// - feature absent   -> `R16Float`: filterable on baseline WebGPU (no feature),
///   so trilinear still works, at half the texture bandwidth and reduced (f16)
///   precision.
///
/// The renderer adds `FLOAT32_FILTERABLE` to
/// [`recommended_device_features`](crate::ViewportRenderer::recommended_device_features),
/// so a consumer that requests those features gets the full-precision path
/// automatically wherever the adapter supports it (all common discrete GPUs);
/// everyone else gets the graceful f16 fallback. The scatter density path binds
/// this same texture non-filtered, so either format works there unchanged.
pub(crate) fn volume_texture_format(device: &crate::gpu::Device) -> crate::gpu::TextureFormat {
    if device
        .features()
        .contains(crate::gpu::Features::FLOAT32_FILTERABLE)
    {
        crate::gpu::TextureFormat::R32Float
    } else {
        crate::gpu::TextureFormat::R16Float
    }
}

/// Bytes per texel for the two direct-volume formats.
fn volume_bytes_per_texel(format: crate::gpu::TextureFormat) -> u32 {
    match format {
        crate::gpu::TextureFormat::R32Float => 4,
        crate::gpu::TextureFormat::R16Float => 2,
        other => panic!("unexpected direct-volume texture format {other:?}"),
    }
}

/// Encode a scalar field into the texel bytes for `format`: raw `f32` for
/// `R32Float`, converted to `f16` for `R16Float`.
fn encode_volume_texels(format: crate::gpu::TextureFormat, data: &[f32]) -> Vec<u8> {
    match format {
        crate::gpu::TextureFormat::R32Float => bytemuck::cast_slice(data).to_vec(),
        crate::gpu::TextureFormat::R16Float => {
            let halves: Vec<u16> = data
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            bytemuck::cast_slice(&halves).to_vec()
        }
        other => panic!("unexpected direct-volume texture format {other:?}"),
    }
}

impl DeviceResources {
    /// Upload a 3D scalar field to the GPU as a filterable 3D texture
    /// ([`volume_texture_format`]: `R32Float` at full precision, or the
    /// `R16Float` fallback), sampled with trilinear interpolation at draw time.
    ///
    /// `data` must be a flat array of `dims[0] * dims[1] * dims[2]` scalars in
    /// x-fastest order (index = x + y*nx + z*nx*ny).
    ///
    /// Returns a [`VolumeId`](crate::resources::VolumeId) that can be stored in [`VolumeItem::volume_id`](crate::renderer::VolumeItem::volume_id).
    pub fn upload_volume(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &[f32],
        dims: [u32; 3],
    ) -> VolumeId {
        let (texture, view, volume_bytes) = Self::build_volume_texture(device, queue, data, dims);
        self.content
            .volume_textures
            .insert((texture, view), volume_bytes)
    }

    /// Overwrite the 3D texture behind `id` in place, keeping the same slot and
    /// handle.
    ///
    /// For a time-series where a field is played back over a fixed grid, calling
    /// this each timestep reuses one slot instead of leaking a fresh 3D texture
    /// per step through [`upload_volume`](Self::upload_volume): resident volume
    /// memory stays flat at one field's size rather than growing without bound.
    /// The old texture is dropped and the store's byte charge is updated to the
    /// new field's size. `dims` may differ from the original upload.
    ///
    /// Returns `false` (and uploads nothing) if `id` is stale or was freed.
    pub fn replace_volume(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: VolumeId,
        data: &[f32],
        dims: [u32; 3],
    ) -> bool {
        if !self.content.volume_textures.contains(id) {
            return false;
        }
        let (texture, view, volume_bytes) = Self::build_volume_texture(device, queue, data, dims);
        let replaced = self
            .content
            .volume_textures
            .replace(id, (texture, view), volume_bytes)
            .is_some();
        if replaced {
            // Drop any scatter bind group built against this slot's old texture
            // so the previous field's GPU memory is actually released.
        }
        replaced
    }

    /// Free the 3D texture behind `id`, reclaiming its slot and byte charge.
    ///
    /// A later [`upload_volume`](Self::upload_volume) reuses the freed slot. Any
    /// handle still holding `id` resolves to nothing afterwards rather than
    /// aliasing whatever next occupies the slot. Returns `false` if `id` was
    /// already freed or is stale.
    pub fn free_volume(&mut self, id: VolumeId) -> bool {
        let freed = self.content.volume_textures.remove(id).is_some();
        if freed {}
        freed
    }

    /// Create a filterable 3D texture from `data`, upload it, and return the
    /// texture, its default view, and the GPU bytes it occupies.
    ///
    /// The format is [`volume_texture_format`] (`R32Float` or the `R16Float`
    /// fallback), so `data` is written raw or converted to `f16` accordingly.
    /// Shared by [`upload_volume`](Self::upload_volume) and
    /// [`replace_volume`](Self::replace_volume).
    fn build_volume_texture(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &[f32],
        dims: [u32; 3],
    ) -> (crate::gpu::Texture, crate::gpu::TextureView, u64) {
        let expected = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        assert_eq!(
            data.len(),
            expected,
            "volume data length {} does not match dims {:?} (expected {})",
            data.len(),
            dims,
            expected
        );

        let format = volume_texture_format(device);
        let bpt = volume_bytes_per_texel(format);

        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("volume_3d_texture"),
            size: crate::gpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D3,
            format,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texels = encode_volume_texels(format, data);
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            &texels,
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims[0] * bpt),
                rows_per_image: Some(dims[1]),
            },
            crate::gpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
        );

        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        (texture, view, (expected as u64) * (bpt as u64))
    }

    /// Start an asynchronous volume upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. The 3D
    /// texture creation and `queue.write_texture` run on a worker thread on
    /// cloned `Device` and `Queue` handles; once the job reports
    /// `UploadStatus::Ready`, call
    /// [`upload_result_volume`](Self::upload_result_volume) to take the
    /// resulting [`VolumeId`](crate::resources::VolumeId).
    ///
    /// Ownership of `data` transfers into the worker.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::VolumeDataLengthMismatch`](crate::error::ViewportError::VolumeDataLengthMismatch)
    /// if `data.len() != dims[0] * dims[1] * dims[2]` before any job is
    /// submitted.
    pub fn begin_upload_volume(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: Vec<f32>,
        dims: [u32; 3],
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        let expected = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        if data.len() != expected {
            return Err(crate::error::ViewportError::VolumeDataLengthMismatch {
                actual: data.len(),
                expected,
                dims,
            });
        }

        let slot = crate::resources::ResultSlot::<VolumeId>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let queue_for_worker = queue.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let format = volume_texture_format(&device_for_worker);
                let bpt = volume_bytes_per_texel(format);
                let texture = device_for_worker.create_texture(&crate::gpu::TextureDescriptor {
                    label: Some("volume_3d_texture"),
                    size: crate::gpu::Extent3d {
                        width: dims[0],
                        height: dims[1],
                        depth_or_array_layers: dims[2],
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: crate::gpu::TextureDimension::D3,
                    format,
                    usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                        | crate::gpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let texels = encode_volume_texels(format, &data);
                queue_for_worker.write_texture(
                    crate::gpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: crate::gpu::Origin3d::ZERO,
                        aspect: crate::gpu::TextureAspect::All,
                    },
                    &texels,
                    crate::gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(dims[0] * bpt),
                        rows_per_image: Some(dims[1]),
                    },
                    crate::gpu::Extent3d {
                        width: dims[0],
                        height: dims[1],
                        depth_or_array_layers: dims[2],
                    },
                );
                let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let volume_bytes = (expected as u64) * (bpt as u64);
                        let id = resources
                            .content
                            .volume_textures
                            .insert((texture, view), volume_bytes);
                        slot_for_apply.set(id);
                    }),
                ))
            })
        };

        self.job_results
            .volume
            .lock()
            .expect("volume result map poisoned")
            .insert(id, slot);
        Ok(id)
    }

    /// Take the [`VolumeId`](crate::resources::VolumeId) produced by a
    /// completed [`begin_upload_volume`](Self::begin_upload_volume) job.
    ///
    /// Returns `JobNotReady` while the upload is still in flight, and
    /// `JobResultMissing` for ids that have already been taken, were
    /// issued by a different upload type, or never existed.
    pub fn upload_result_volume(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<VolumeId> {
        let mut map = self
            .job_results
            .volume
            .lock()
            .expect("volume result map poisoned");
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
}

#[cfg(test)]
mod tests {
    use crate::DeviceResources;
    use crate::geometry::marching_cubes::VolumeData;
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

    fn sample_volume_data() -> Vec<f32> {
        let n: usize = 8;
        let mut data = Vec::with_capacity(n * n * n);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let v = (x + y + z) as f32 / (3.0 * n as f32);
                    data.push(v);
                }
            }
        }
        data
    }

    fn sample_volume_struct() -> VolumeData {
        VolumeData {
            data: sample_volume_data(),
            dims: [8, 8, 8],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        }
    }

    fn drive_until_ready(
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::JobId,
        label: &str,
    ) {
        for _ in 0..200 {
            resources.process_uploads(device, queue);
            match resources.upload_status(id) {
                UploadStatus::Ready => return,
                UploadStatus::Failed(e) => panic!("{label} upload failed: {e:?}"),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                UploadStatus::Unknown => panic!("{label} job id disappeared"),
            }
        }
        panic!("{label} upload did not complete in time");
    }

    #[test]
    fn sync_upload_volume_still_works() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let data = sample_volume_data();
        let _id = resources.upload_volume(&device, &queue, &data, [8, 8, 8]);
    }

    #[test]
    fn upload_volume_charges_resident_bytes() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // Bytes per texel depend on the chosen format: 4 for R32Float (with
        // FLOAT32_FILTERABLE), 2 for the R16Float fallback (the default test
        // device requests no features, so this is usually the 2-byte path).
        let bpt = super::volume_bytes_per_texel(super::volume_texture_format(&device)) as u64;
        assert_eq!(resources.resident_bytes().volume_bytes, 0);
        let _id = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        // 8*8*8 = 512 texels * bytes-per-texel.
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * bpt);
        // A second distinct upload takes a second slot, so the charge adds. This
        // is the per-timestep growth a time-series avoids by calling
        // replace_volume on one handle instead (see replace_volume_reuses_slot).
        let _id2 = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(resources.resident_bytes().volume_bytes, 2 * 8 * 8 * 8 * bpt);
    }

    #[test]
    fn replace_volume_reuses_slot() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let bpt = super::volume_bytes_per_texel(super::volume_texture_format(&device)) as u64;
        let id = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * bpt);

        // Replacing the field in place keeps the same handle and one slot, so the
        // charge stays flat rather than doubling (the whole point of S1).
        for _ in 0..10 {
            assert!(resources.replace_volume(
                &device,
                &queue,
                id,
                &sample_volume_data(),
                [8, 8, 8]
            ));
        }
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * bpt);

        // A larger field updates the charge to the new size, still one slot.
        let big = vec![0.5_f32; 16 * 16 * 16];
        assert!(resources.replace_volume(&device, &queue, id, &big, [16, 16, 16]));
        assert_eq!(resources.resident_bytes().volume_bytes, 16 * 16 * 16 * bpt);
    }

    #[test]
    fn free_volume_reclaims_bytes_and_slot() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let bpt = super::volume_bytes_per_texel(super::volume_texture_format(&device)) as u64;
        let id = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * bpt);

        assert!(resources.free_volume(id));
        assert_eq!(resources.resident_bytes().volume_bytes, 0);
        // Second free of the same handle is a no-op.
        assert!(!resources.free_volume(id));

        // The freed slot is reused by the next upload, so the charge is one
        // field's worth, not two.
        let _id2 = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * bpt);
    }

    #[test]
    fn stale_volume_handle_does_not_alias_reused_slot() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let old = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert!(resources.free_volume(old));
        // Reusing the freed slot mints a handle with a bumped generation.
        let new = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(old.index(), new.index(), "same slot reused");
        assert_ne!(old, new, "generation bumped so the stale handle differs");
        // The stale handle no longer resolves and cannot be replaced or refreed.
        assert!(!resources.replace_volume(&device, &queue, old, &sample_volume_data(), [8, 8, 8]));
        assert!(!resources.free_volume(old));
        // The live handle still works.
        assert!(resources.replace_volume(&device, &queue, new, &sample_volume_data(), [8, 8, 8]));
    }

    #[test]
    fn begin_upload_volume_validates_dims() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let err = resources
            .begin_upload_volume(&device, &queue, vec![0.0_f32; 7], [8, 8, 8])
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::VolumeDataLengthMismatch { .. }
        ));
    }

    #[test]
    fn begin_upload_volume_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources
            .begin_upload_volume(&device, &queue, sample_volume_data(), [8, 8, 8])
            .expect("job submitted");
        drive_until_ready(&mut resources, &device, &queue, job, "volume");
        let _id = resources.upload_result_volume(job).expect("ready");
        let err = resources.upload_result_volume(job).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::JobResultMissing { .. }
        ));
    }

    #[test]
    fn begin_upload_volume_for_mc_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_volume_for_mc(&device, &queue, sample_volume_struct());
        drive_until_ready(&mut resources, &device, &queue, job, "volume_mc");
        let _id = resources.upload_result_volume_mc(job).expect("ready");
    }

    #[test]
    fn sync_upload_volume_for_mc_still_works() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let vol = sample_volume_struct();
        let _id = resources
            .upload_volume_for_mc(&device, &queue, &vol)
            .expect("upload ok");
    }
}
