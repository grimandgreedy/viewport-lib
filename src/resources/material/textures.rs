use crate::resources::*;

impl DeviceResources {
    /// Upload an RGBA texture to the GPU and return its texture ID.
    ///
    /// The ID can be stored in `Material::texture_id` to apply the texture to objects.
    /// `rgba_data` must be exactly `width * height * 4` bytes in RGBA8 format.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData) if the data length is incorrect.
    pub fn upload_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> crate::error::ViewportResult<crate::resources::TextureId> {
        // Sync wrapper around the async path: build the job, drive it to
        // completion on the calling thread, and take the typed result. The
        // worker performs the same texture creation, sampler setup, and
        // bind-group build that the apply closure runs from
        // `process_uploads`. The data is copied into an owned `Vec` for the
        // worker thread; small textures absorb this trivially, large
        // textures pay one extra memcpy.
        let id = self.begin_upload_texture(device, queue, width, height, rgba_data.to_vec())?;
        self.drain_until(device, queue, id)?;
        self.upload_result_texture(id)
    }

    /// Upload an RGBA texture as a normal map and return its texture ID.
    ///
    /// Uses Rgba8Unorm format (not sRGB) so values are linear : required for correct
    /// normal map decoding. `rgba_data` must be `width * height * 4` bytes.
    pub fn upload_normal_map(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> crate::error::ViewportResult<crate::resources::TextureId> {
        // Sync wrapper; see `upload_texture` for the worker / apply layout.
        let id = self.begin_upload_normal_map(device, queue, width, height, rgba_data.to_vec())?;
        self.drain_until(device, queue, id)?;
        self.upload_result_texture(id)
    }

    // -----------------------------------------------------------------------
    // Async texture upload (routed through the upload-job runner)
    // -----------------------------------------------------------------------

    /// Start an asynchronous albedo texture upload.
    ///
    /// Returns a `JobId` immediately. The mip chain is built on a worker
    /// thread; the texture creation and pixel copy then run on the device
    /// thread during a `process_uploads` call, under the frame budget when
    /// one is set, and the runner gates the job on a submission that
    /// flushes those writes. Once the status is `Ready`, take the resulting
    /// texture id with `upload_result_texture` and store it in
    /// `Material::texture_id`.
    ///
    /// `rgba` transfers into the worker; clone at the call site to retain
    /// it. Format and binding match the synchronous `upload_texture`.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// when `rgba.len() != width * height * 4`, reported before any job is
    /// submitted.
    pub fn begin_upload_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        let expected = (width * height * 4) as usize;
        if rgba.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba.len(),
            });
        }
        Ok(self.spawn_texture_upload(
            device,
            queue,
            TextureUploadSpec {
                width,
                height,
                format: crate::gpu::TextureFormat::Rgba8UnormSrgb,
                is_normal_map: false,
                mip_levels: vec![rgba],
            },
        ))
    }

    /// Start an asynchronous normal-map upload.
    ///
    /// Same shape as `begin_upload_texture`, but the texture is created
    /// with the linear `Rgba8Unorm` format and bound into the normal-map
    /// slot. Take the result with `upload_result_texture` once `Ready`.
    ///
    /// # Errors
    ///
    /// Same as `begin_upload_texture`.
    pub fn begin_upload_normal_map(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        let expected = (width * height * 4) as usize;
        if rgba.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba.len(),
            });
        }
        Ok(self.spawn_texture_upload(
            device,
            queue,
            TextureUploadSpec {
                width,
                height,
                format: crate::gpu::TextureFormat::Rgba8Unorm,
                is_normal_map: true,
                mip_levels: vec![rgba],
            },
        ))
    }

    /// Upload a linear HDR RGBA texture (`Rgba16Float`) and return its texture ID.
    ///
    /// For baked lightmaps and other data whose values exceed 1.0: the 8-bit
    /// [`upload_texture`](Self::upload_texture) path clamps at upload, so bright
    /// baked radiance is lost before it reaches the HDR render path. This keeps
    /// the full range. `rgba` is `width * height * 4` linear `f32` values (RGBA,
    /// row-major); they are converted to half floats. No mip chain is built
    /// (lightmaps sample the base level), so the texture is single-mip.
    pub fn upload_texture_hdr(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba: &[f32],
    ) -> crate::error::ViewportResult<crate::resources::TextureId> {
        let id = self.begin_upload_texture_hdr(device, queue, width, height, rgba.to_vec())?;
        self.drain_until(device, queue, id)?;
        self.upload_result_texture(id)
    }

    /// Start an asynchronous linear-HDR (`Rgba16Float`) texture upload.
    ///
    /// Same shape as [`begin_upload_texture`](Self::begin_upload_texture) but the
    /// texture is created as `Rgba16Float` (single mip) so values above 1.0
    /// survive. `rgba` is `width * height * 4` linear `f32` values, converted to
    /// half floats before upload. Take the result with
    /// [`upload_result_texture`](Self::upload_result_texture) once `Ready`.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// when `rgba.len() != width * height * 4`.
    pub fn begin_upload_texture_hdr(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<f32>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        let expected = (width * height * 4) as usize;
        if rgba.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba.len(),
            });
        }
        // Pack to half-float bytes (2 bytes per channel, little-endian).
        let mut bytes = Vec::with_capacity(rgba.len() * 2);
        for &v in &rgba {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        Ok(self.spawn_texture_upload(
            device,
            queue,
            TextureUploadSpec {
                width,
                height,
                format: crate::gpu::TextureFormat::Rgba16Float,
                is_normal_map: false,
                mip_levels: vec![bytes],
            },
        ))
    }

    /// Take the texture id produced by a completed `begin_upload_texture`
    /// or `begin_upload_normal_map` job.
    ///
    /// Returns `JobNotReady` while the upload is still in flight, and
    /// `JobResultMissing` for ids that have already been taken, were
    /// issued by a different upload type, or never existed.
    pub fn upload_result_texture(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TextureId> {
        let mut map = self
            .job_results
            .texture
            .lock()
            .expect("texture result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(tex_id) => {
                map.remove(&id);
                Ok(tex_id)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Upload a pre-compressed, pre-mipped texture and return its texture ID.
    ///
    /// Sync wrapper around [`begin_upload_compressed_texture`](Self::begin_upload_compressed_texture):
    /// see that method for the format and layout requirements. Blocks the
    /// calling thread until the upload completes.
    ///
    /// # Errors
    ///
    /// See [`begin_upload_compressed_texture`](Self::begin_upload_compressed_texture).
    pub fn upload_compressed_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        desc: CompressedTextureDesc<'_>,
    ) -> crate::error::ViewportResult<crate::resources::TextureId> {
        let id = self.begin_upload_compressed_texture(device, queue, desc)?;
        self.drain_until(device, queue, id)?;
        self.upload_result_texture(id)
    }

    /// Start an asynchronous upload of pre-compressed, pre-mipped texture data.
    ///
    /// Returns a `JobId` immediately; take the resulting texture id with
    /// [`upload_result_texture`](Self::upload_result_texture) once the status is
    /// `Ready`, then store it in a `Material` slot. The data is validated and
    /// copied into the worker before the job is submitted.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`ViewportError::UnsupportedTextureFormat`](crate::error::ViewportError::UnsupportedTextureFormat)
    /// when `desc.format` is not block-compressed or the device lacks its
    /// required feature (check first with [`supports_texture_format`]), and
    /// [`ViewportError::InvalidCompressedTextureData`](crate::error::ViewportError::InvalidCompressedTextureData)
    /// when `mip_levels` is empty or a level's byte length does not match its
    /// block-packed size.
    pub fn begin_upload_compressed_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        desc: CompressedTextureDesc<'_>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        if !desc.format.is_compressed() {
            return Err(crate::error::ViewportError::UnsupportedTextureFormat {
                format: desc.format,
            });
        }
        // wgpu requires block-compressed textures to be created with
        // block-aligned base dimensions. Reject early with a clear error rather
        // than letting `create_texture` fail on the worker (which would leave an
        // invalid texture bound into a bind group). Checked before device
        // support so the caller gets the specific reason either way.
        let (block_w, block_h) = desc.format.block_dimensions();
        if desc.width % block_w != 0 || desc.height % block_h != 0 {
            return Err(
                crate::error::ViewportError::CompressedTextureNotBlockAligned {
                    width: desc.width,
                    height: desc.height,
                    block_width: block_w,
                    block_height: block_h,
                },
            );
        }
        if !supports_texture_format(device, desc.format) {
            return Err(crate::error::ViewportError::UnsupportedTextureFormat {
                format: desc.format,
            });
        }
        if desc.mip_levels.is_empty() {
            let (_, _, expected) = mip_block_layout(desc.format, desc.width, desc.height);
            return Err(crate::error::ViewportError::InvalidCompressedTextureData {
                level: 0,
                expected,
                actual: 0,
            });
        }
        // Validate each level against its block-packed size and copy into owned
        // buffers for the worker thread.
        let mut mip_levels = Vec::with_capacity(desc.mip_levels.len());
        for (level, data) in desc.mip_levels.iter().enumerate() {
            let lw = (desc.width >> level).max(1);
            let lh = (desc.height >> level).max(1);
            let (_, _, expected) = mip_block_layout(desc.format, lw, lh);
            if data.len() != expected {
                return Err(crate::error::ViewportError::InvalidCompressedTextureData {
                    level: level as u32,
                    expected,
                    actual: data.len(),
                });
            }
            mip_levels.push(data.to_vec());
        }
        Ok(self.spawn_texture_upload(
            device,
            queue,
            TextureUploadSpec {
                width: desc.width,
                height: desc.height,
                format: desc.format,
                is_normal_map: desc.is_normal_map,
                mip_levels,
            },
        ))
    }

    /// Shared spawn path for the RGBA8 and compressed upload entry points.
    /// The spec carries the texture format, dimensions, mip chain, and the
    /// slot (albedo vs normal map) the texture occupies.
    fn spawn_texture_upload(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        spec: TextureUploadSpec,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::TextureId>::new();
        let slot_for_apply = slot.clone();
        // The GPU stage receives device and queue from `process` on the
        // device thread; the handles passed in here are unused, kept so the
        // public entry points keep their signatures.
        let _ = (device, queue);

        // Clone the fallback views and the bind-group layout into the job
        // so its stages can build the GpuTexture and bind group without
        // touching `self`.
        let bgl = self.texture_bind_group_layout.clone();
        let fallback_albedo_view = self.fallback_texture.view.clone();
        let fallback_normal_view = self.fallback_normal_map_view.clone();
        let fallback_ao_view = self.fallback_ao_map_view.clone();
        let TextureUploadSpec {
            width,
            height,
            format,
            is_normal_map,
            mip_levels,
        } = spec;
        // Resident-byte accounting. When the worker will generate a mip chain
        // (single uncompressed RGBA level), charge the full chain size.
        let data_bytes: u64 = if mip_levels.len() == 1
            && matches!(
                format,
                crate::gpu::TextureFormat::Rgba8UnormSrgb | crate::gpu::TextureFormat::Rgba8Unorm
            ) {
            let mut total = 0u64;
            let (mut w, mut h) = (width, height);
            loop {
                total += w as u64 * h as u64 * 4;
                if w == 1 && h == 1 {
                    break;
                }
                w = (w / 2).max(1);
                h = (h / 2).max(1);
            }
            total
        } else {
            mip_levels.iter().map(|l| l.len() as u64).sum()
        };

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu_then_gpu_chunked(move |progress| {
                // A single supplied level means an uncompressed RGBA base
                // image: build its mip chain here on the worker so minified
                // sampling is trilinear instead of full-resolution fetches.
                // Compressed uploads pass their own chains and skip this.
                // The chain build is pure CPU and by far the expensive part
                // of the job, so it must not run on the render thread; only
                // the texture creation and copies below need the device
                // thread.
                let mip_levels = if mip_levels.len() == 1
                    && matches!(
                        format,
                        crate::gpu::TextureFormat::Rgba8UnormSrgb
                            | crate::gpu::TextureFormat::Rgba8Unorm
                    ) {
                    let srgb = format == crate::gpu::TextureFormat::Rgba8UnormSrgb;
                    let mut levels = mip_levels;
                    let base = levels.pop().expect("one base level");
                    build_rgba_mip_chain(base, width, height, srgb)
                } else {
                    mip_levels
                };
                progress.set(0.2);

                // GPU stage: create the texture once, then write the chain
                // in row bands of at most TEXTURE_CHUNK_BYTES, as many per
                // turn as the frame budget allows (at least one, so an
                // already-elapsed budget still makes progress). The typed
                // result is published only after the final band and flush,
                // so no partially written texture is ever observable.
                let total_bytes: u64 = mip_levels.iter().map(|l| l.len() as u64).sum();
                let mut texture: Option<crate::gpu::Texture> = None;
                let mut level: usize = 0;
                let mut block_row: u32 = 0;
                let mut written: u64 = 0;
                let mut slot_for_apply = Some(slot_for_apply);
                Ok(Box::new(
                    move |dev: &crate::gpu::Device,
                          q: &crate::gpu::Queue,
                          progress: &crate::resources::ProgressHandle,
                          budget: &crate::resources::upload_jobs::FrameBudget| {
                        let tex_label = if is_normal_map {
                            "user_normal_map_texture"
                        } else {
                            "user_texture"
                        };
                        let tex = texture.get_or_insert_with(|| {
                            dev.create_texture(&crate::gpu::TextureDescriptor {
                                label: Some(tex_label),
                                size: crate::gpu::Extent3d {
                                    width,
                                    height,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: mip_levels.len() as u32,
                                sample_count: 1,
                                dimension: crate::gpu::TextureDimension::D2,
                                format,
                                usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                                    | crate::gpu::TextureUsages::COPY_DST,
                                view_formats: &[],
                            })
                        });
                        while level < mip_levels.len() {
                            let lw = (width >> level).max(1);
                            let lh = (height >> level).max(1);
                            let (bytes_per_row, blocks_high, _) = mip_block_layout(format, lw, lh);
                            let band_rows = ((TEXTURE_CHUNK_BYTES / bytes_per_row as u64).max(1)
                                as u32)
                                .min(blocks_high - block_row);
                            let (_, block_h) = format.block_dimensions();
                            let origin_y = block_row * block_h;
                            let end_row = block_row + band_rows;
                            // The last band's texel height absorbs any
                            // non-multiple-of-block edge.
                            let band_height = if end_row == blocks_high {
                                lh - origin_y
                            } else {
                                band_rows * block_h
                            };
                            let data = &mip_levels[level][(block_row as usize
                                * bytes_per_row as usize)
                                ..(end_row as usize * bytes_per_row as usize)];
                            q.write_texture(
                                crate::gpu::TexelCopyTextureInfo {
                                    texture: tex,
                                    mip_level: level as u32,
                                    origin: crate::gpu::Origin3d {
                                        x: 0,
                                        y: origin_y,
                                        z: 0,
                                    },
                                    aspect: crate::gpu::TextureAspect::All,
                                },
                                data,
                                crate::gpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(bytes_per_row),
                                    rows_per_image: Some(band_rows),
                                },
                                crate::gpu::Extent3d {
                                    width: lw,
                                    height: band_height,
                                    depth_or_array_layers: 1,
                                },
                            );
                            written += data.len() as u64;
                            block_row = end_row;
                            if block_row == blocks_high {
                                level += 1;
                                block_row = 0;
                            }
                            progress.set(0.2 + 0.7 * (written as f32 / total_bytes.max(1) as f32));
                            if budget.exhausted() {
                                break;
                            }
                        }
                        if level < mip_levels.len() {
                            return Ok(crate::resources::upload_jobs::GpuStep::Continue);
                        }

                        let gpu_texture = finish_gpu_texture(
                            dev,
                            texture.take().expect("texture created on first turn"),
                            mip_levels.len() as u32,
                            is_normal_map,
                            &bgl,
                            &fallback_albedo_view,
                            &fallback_normal_view,
                            &fallback_ao_view,
                        );
                        // Flush so the runner has a submission to gate on.
                        // Writes queued above are folded into this submit.
                        let encoder =
                            dev.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                                label: Some("async_texture_flush"),
                            });
                        let submission = q.submit(std::iter::once(encoder.finish()));
                        progress.set(1.0);

                        let slot = slot_for_apply.take().expect("done reached once");
                        Ok(crate::resources::upload_jobs::GpuStep::Done(
                            crate::resources::upload_jobs::JobProduct::with_gpu_and_apply(
                                submission,
                                Box::new(move |resources: &mut DeviceResources| {
                                    let tex_id =
                                        resources.content.textures.insert(gpu_texture, data_bytes);
                                    slot.set(tex_id);
                                }),
                            ),
                        ))
                    },
                )
                    as crate::resources::upload_jobs::ChunkedGpuWorkFn)
            })
        };

        self.job_results
            .texture
            .lock()
            .expect("texture result map poisoned")
            .insert(id, slot);
        id
    }

    // -----------------------------------------------------------------------
    // VRAM budget query
    // -----------------------------------------------------------------------

    /// Current GPU memory usage for user-uploaded textures.
    ///
    /// Counts bytes from `upload_texture`, `upload_normal_map`, and the
    /// async upload entries. Internal resources (shadow maps, colourmaps,
    /// IBL, post-processing targets) are not included.
    pub fn texture_memory_stats(&self) -> TextureMemoryStats {
        TextureMemoryStats {
            used_bytes: self.content.textures.allocated_bytes(),
            texture_count: self.content.textures.len() as u32,
        }
    }

    /// Release a user-uploaded texture, reclaiming its slot and GPU memory.
    ///
    /// Drops the `GpuTexture` (wgpu defers the real free until in-flight
    /// commands that reference it complete), bumps the slot generation so `id`
    /// no longer resolves, and evicts the cached bind groups that named the
    /// texture: the shared `material_bind_groups` / `instance_bind_groups`
    /// entries whose key contains `id`, and any per-mesh object bind group built
    /// against it (invalidated so the next `prepare` rebinds the fallback).
    ///
    /// Returns `true` if a texture was released, `false` if `id` did not resolve
    /// to a live texture (already freed, never uploaded, or a stale handle).
    /// Materials still holding `id` are not rewritten; they fall back to the
    /// fallback texture until reassigned.
    pub fn free_texture(&mut self, id: crate::resources::TextureId) -> bool {
        if !self.content.textures.remove(id) {
            return false;
        }
        self.evict_texture_bind_group_caches(id.raw());
        self.resource_free_epoch += 1;
        true
    }

    /// Replace the pixels of an already-uploaded texture in place, keeping the
    /// same `TextureId`.
    ///
    /// The handle stays valid: materials and items holding `id` pick up the new
    /// pixels on the next frame with no reassignment. The generation check is the
    /// in-flight guard, so a stale handle (its slot freed and reused) returns
    /// `StaleHandle` instead of overwriting whatever now occupies the slot. Use
    /// this for content that changes over time (a streamed or animated texture)
    /// where re-uploading and reassigning a fresh id would be wasteful.
    ///
    /// The texture is recreated as an `Rgba8UnormSrgb` albedo texture, matching
    /// [`upload_texture`](Self::upload_texture); `rgba_data` must be exactly
    /// `width * height * 4` bytes. Dimensions and format need not match the
    /// original upload.
    ///
    /// # Errors
    ///
    /// [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// if the data length is wrong, or
    /// [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `id` does not resolve to a live texture.
    /// Note: replaced textures are single-mip. This path is synchronous and
    /// sized for per-frame dynamic content (video frames, procedural
    /// updates), where building a mip chain on the caller thread every frame
    /// would cost more than the minified sampling it saves. Static textures
    /// uploaded through `upload_texture`/`begin_upload_texture` get a full
    /// mip chain built on the worker.
    pub fn replace_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::TextureId,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> crate::error::ViewportResult<()> {
        let expected = (width * height * 4) as usize;
        if rgba_data.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba_data.len(),
            });
        }
        let gpu_texture = build_gpu_texture(
            device,
            queue,
            width,
            height,
            crate::gpu::TextureFormat::Rgba8UnormSrgb,
            false,
            std::slice::from_ref(&rgba_data.to_vec()),
            &self.texture_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
        );
        let bytes = rgba_data.len() as u64;
        if self
            .content
            .textures
            .replace(id, gpu_texture, bytes)
            .is_none()
        {
            let (index, count) = (id.index(), self.content.textures.len());
            return Err(crate::error::ViewportError::StaleHandle { index, count });
        }
        // The slot's view changed, so bind groups cached against this id now
        // sample the old texture. Evict them exactly as `free_texture` does; they
        // rebuild against the new view on the next `prepare`.
        self.evict_texture_bind_group_caches(id.raw());
        Ok(())
    }

    /// Drop cached bind groups that named user texture slot `raw` so they rebuild
    /// against the current occupant (or the fallback). Shared by `free_texture`
    /// (slot now empty) and `replace_texture` (slot holds a new view).
    fn evict_texture_bind_group_caches(&mut self, raw: u64) {
        self.content
            .material_bind_groups
            .retain(|&(a, n, ao), _| a != raw && n != raw && ao != raw);
        self.instancing
            .bind_groups
            .retain(|&(a, n, ao), _| a != raw && n != raw && ao != raw);

        // Invalidate per-mesh object bind groups that sampled the texture so
        // `update_mesh_texture_bind_group` rebuilds them. `last_tex_key`
        // positions holding user-texture ids are albedo, normal, ao,
        // metallic-roughness, and emissive.
        const INVALID_KEY: (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) = (
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
        );
        for (_, mesh) in self.mesh_store.iter_mut() {
            let k = mesh.last_tex_key;
            if k.0 == raw || k.1 == raw || k.2 == raw || k.7 == raw || k.8 == raw {
                mesh.last_tex_key = INVALID_KEY;
            }
        }
    }
}

/// Everything `spawn_texture_upload` needs to create and fill a texture.
/// The RGBA8 entry points build a single-level spec; the compressed entry
/// point builds one with the caller's format and full mip chain.
struct TextureUploadSpec {
    width: u32,
    height: u32,
    format: crate::gpu::TextureFormat,
    /// Bind into the normal-map slot and label as such.
    is_normal_map: bool,
    /// One entry per mip level, level 0 first.
    mip_levels: Vec<Vec<u8>>,
}

/// Build a full RGBA8 mip chain from a single base level by 2x2 box
/// filtering, down to 1x1. For sRGB textures the RGB channels are averaged
/// in linear space (decode, average, re-encode); alpha and linear-format
/// channels average directly. Odd dimensions clamp the second sample row or
/// column to the edge.
fn build_rgba_mip_chain(base: Vec<u8>, width: u32, height: u32, srgb: bool) -> Vec<Vec<u8>> {
    // 256-entry sRGB decode table; encode goes through powf per output
    // texel, which the halving series keeps cheap (the whole chain is one
    // third of the base size).
    fn srgb_to_linear(v: u8) -> f32 {
        let f = v as f32 / 255.0;
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    }
    fn linear_to_srgb(f: f32) -> u8 {
        let f = f.clamp(0.0, 1.0);
        let s = if f <= 0.0031308 {
            f * 12.92
        } else {
            1.055 * f.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    }
    let decode: Vec<f32> = (0..=255u32).map(|v| srgb_to_linear(v as u8)).collect();

    let mut levels = vec![base];
    let (mut w, mut h) = (width as usize, height as usize);
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let src = levels.last().expect("previous level");
        let mut dst = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            let (sy0, sy1) = (2 * y, (2 * y + 1).min(h - 1));
            for x in 0..nw {
                let (sx0, sx1) = (2 * x, (2 * x + 1).min(w - 1));
                let corners = [
                    (sy0 * w + sx0) * 4,
                    (sy0 * w + sx1) * 4,
                    (sy1 * w + sx0) * 4,
                    (sy1 * w + sx1) * 4,
                ];
                let o = (y * nw + x) * 4;
                for c in 0..3 {
                    if srgb {
                        let sum: f32 = corners.iter().map(|&s| decode[src[s + c] as usize]).sum();
                        dst[o + c] = linear_to_srgb(sum * 0.25);
                    } else {
                        let sum: u32 = corners.iter().map(|&s| src[s + c] as u32).sum();
                        dst[o + c] = ((sum + 2) / 4) as u8;
                    }
                }
                let sum_a: u32 = corners.iter().map(|&s| src[s + 3] as u32).sum();
                dst[o + 3] = ((sum_a + 2) / 4) as u8;
            }
        }
        levels.push(dst);
        w = nw;
        h = nh;
    }
    levels
}

/// Slice size for the chunked async texture writes, in bytes. Small enough
/// that one band fits a sub-millisecond frame budget at staging-copy
/// bandwidth (a few GB/s), large enough to amortise per-write overhead.
/// Bands are whole block rows, so a mip smaller than this uploads in one
/// write.
const TEXTURE_CHUNK_BYTES: u64 = 4 << 20;

/// Block-packed layout for a single mip level: `(bytes_per_row, rows_of_blocks,
/// total_bytes)`. Uses the format's block dimensions and size, so it is correct
/// for uncompressed formats (1x1 blocks) and block-compressed formats alike.
fn mip_block_layout(
    format: crate::gpu::TextureFormat,
    level_width: u32,
    level_height: u32,
) -> (u32, u32, usize) {
    let (block_w, block_h) = format.block_dimensions();
    let block_bytes = format
        .block_copy_size(None)
        .expect("texture format has no single block-copy size");
    let blocks_x = level_width.div_ceil(block_w);
    let blocks_y = level_height.div_ceil(block_h);
    let bytes_per_row = blocks_x * block_bytes;
    let total = (blocks_x * blocks_y * block_bytes) as usize;
    (bytes_per_row, blocks_y, total)
}

/// Build a `GpuTexture` (texture, view, sampler, bind group) from one or more
/// mip levels. Shared by the async upload worker and the synchronous
/// `replace_texture` path so both create identical resources.
///
/// `mip_levels` holds level 0 first; a single level keeps nearest-mip sampling,
/// more than one enables trilinear. `is_normal_map` selects which bind-group
/// slot the new view occupies (albedo slot 0 or normal slot 2), with the other
/// colour slots filled by the fallback views.
#[allow(clippy::too_many_arguments)]
fn build_gpu_texture(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    width: u32,
    height: u32,
    format: crate::gpu::TextureFormat,
    is_normal_map: bool,
    mip_levels: &[Vec<u8>],
    bgl: &crate::gpu::BindGroupLayout,
    fallback_albedo_view: &crate::gpu::TextureView,
    fallback_normal_view: &crate::gpu::TextureView,
    fallback_ao_view: &crate::gpu::TextureView,
) -> GpuTexture {
    let tex_label = if is_normal_map {
        "user_normal_map_texture"
    } else {
        "user_texture"
    };
    let mip_level_count = mip_levels.len() as u32;
    let texture = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some(tex_label),
        size: crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format,
        usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Upload each mip level. Row/size math is block-based so it is correct for
    // both uncompressed (1x1 blocks) and block-compressed formats.
    for (level, data) in mip_levels.iter().enumerate() {
        let lw = (width >> level).max(1);
        let lh = (height >> level).max(1);
        let (bytes_per_row, blocks_high, _) = mip_block_layout(format, lw, lh);
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            data,
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(blocks_high),
            },
            crate::gpu::Extent3d {
                width: lw,
                height: lh,
                depth_or_array_layers: 1,
            },
        );
    }

    finish_gpu_texture(
        device,
        texture,
        mip_level_count,
        is_normal_map,
        bgl,
        fallback_albedo_view,
        fallback_normal_view,
        fallback_ao_view,
    )
}

/// Build the view, sampler, and bind group around an already-written
/// texture. Tail of [`build_gpu_texture`], shared with the chunked async
/// upload path, which creates and fills the texture across several frames
/// before finishing it here.
#[allow(clippy::too_many_arguments)]
fn finish_gpu_texture(
    device: &crate::gpu::Device,
    texture: crate::gpu::Texture,
    mip_level_count: u32,
    is_normal_map: bool,
    bgl: &crate::gpu::BindGroupLayout,
    fallback_albedo_view: &crate::gpu::TextureView,
    fallback_normal_view: &crate::gpu::TextureView,
    fallback_ao_view: &crate::gpu::TextureView,
) -> GpuTexture {
    let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
    let mipmap_filter = if mip_level_count > 1 {
        crate::gpu::FilterMode::Linear
    } else {
        crate::gpu::FilterMode::Nearest
    };
    let sampler_label = if is_normal_map {
        "user_normal_map_sampler"
    } else {
        "user_texture_sampler"
    };
    let sampler =
        crate::resources::builders::repeat_linear_sampler(device, sampler_label, mipmap_filter);
    let (slot0_view, slot2_view) = if is_normal_map {
        (fallback_albedo_view, &view)
    } else {
        (&view, fallback_normal_view)
    };
    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some(if is_normal_map {
            "user_normal_map_bg"
        } else {
            "user_texture_bg"
        }),
        layout: bgl,
        entries: &[
            crate::gpu::BindGroupEntry {
                binding: 0,
                resource: crate::gpu::BindingResource::TextureView(slot0_view),
            },
            crate::gpu::BindGroupEntry {
                binding: 1,
                resource: crate::gpu::BindingResource::Sampler(&sampler),
            },
            crate::gpu::BindGroupEntry {
                binding: 2,
                resource: crate::gpu::BindingResource::TextureView(slot2_view),
            },
            crate::gpu::BindGroupEntry {
                binding: 3,
                resource: crate::gpu::BindingResource::TextureView(fallback_ao_view),
            },
        ],
    });
    GpuTexture {
        texture,
        view,
        sampler,
        bind_group,
    }
}

/// True when `device` can sample `format`.
///
/// Checks the format's required wgpu feature (for example
/// `TEXTURE_COMPRESSION_BC`, `_ASTC`, or `_ETC2` for block-compressed
/// formats). Call this before `upload_compressed_texture` to decide whether to
/// hand the renderer compressed data or fall back to an uncompressed upload.
pub fn supports_texture_format(
    device: &crate::gpu::Device,
    format: crate::gpu::TextureFormat,
) -> bool {
    device.features().contains(format.required_features())
}

/// Pre-compressed, pre-mipped texture data, uploaded to the GPU as-is.
///
/// The library does no encoding or decoding: compress and build the mip chain
/// offline in your asset pipeline, then hand the block bytes here. `format`
/// must be a block-compressed format (BC, ASTC, or ETC2) that the device
/// supports (check with [`supports_texture_format`]). `mip_levels` holds one
/// tightly block-packed byte slice per level, level 0 (full size) first;
/// `mip_levels.len()` becomes the texture's mip count.
///
/// Color space is carried by `format` (for example `Bc7RgbaUnormSrgb` for
/// albedo versus `Bc5RgUnorm` for normals); `is_normal_map` only selects which
/// internal bind-group slot the texture occupies.
pub struct CompressedTextureDesc<'a> {
    /// Width of mip level 0, in texels.
    pub width: u32,
    /// Height of mip level 0, in texels.
    pub height: u32,
    /// Block-compressed texture format.
    pub format: crate::gpu::TextureFormat,
    /// Bind into the normal-map slot rather than the albedo slot.
    pub is_normal_map: bool,
    /// Block bytes per mip level, level 0 first.
    pub mip_levels: &'a [&'a [u8]],
}

impl DeviceResources {
    /// Get or create a cached material bind group for (albedo, normal_map, ao_map) texture combo.
    ///
    /// `u64::MAX` sentinel means "use fallback texture for that slot".
    /// The bind group is cached in `material_bind_groups` keyed by the 3-tuple.
    #[allow(dead_code)]
    pub(crate) fn get_material_bind_group(
        &mut self,
        device: &crate::gpu::Device,
        albedo_id: Option<crate::resources::TextureId>,
        normal_map_id: Option<crate::resources::TextureId>,
        ao_map_id: Option<crate::resources::TextureId>,
    ) -> &crate::gpu::BindGroup {
        let key = (
            albedo_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
        );

        if !self.content.material_bind_groups.contains_key(&key) {
            let albedo_view = match albedo_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("material_bg"),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(albedo_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::TextureView(normal_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: crate::gpu::BindingResource::TextureView(ao_view),
                    },
                ],
            });
            self.content.material_bind_groups.insert(key, bg);
        }

        self.content.material_bind_groups.get(&key).unwrap()
    }

    /// Rebuild `mesh.object_bind_group` so it includes the texture views, LUT, and scalar
    /// buffer for the given material + attribute key. Called from `prepare()` when
    /// `mesh.last_tex_key` differs from the current frame's material/attribute state.
    ///
    /// The bind group layout is `object_bgl`:
    ///   binding 0 -> object uniform buffer
    ///   binding 1 -> albedo texture view
    ///   binding 2 -> material sampler (also used for LUT sampling)
    ///   binding 3 -> normal map view
    ///   binding 4 -> AO map view
    ///   binding 5 -> LUT (colourmap) texture view
    ///   binding 6 -> scalar attribute storage buffer
    pub(crate) fn update_mesh_texture_bind_group(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        albedo_id: Option<crate::resources::TextureId>,
        normal_map_id: Option<crate::resources::TextureId>,
        ao_map_id: Option<crate::resources::TextureId>,
        lut_id: Option<ColourmapId>,
        active_attr: Option<&str>,
        matcap_id: Option<crate::resources::MatcapId>,
        warp_attr: Option<&str>,
        metallic_roughness_id: Option<crate::resources::TextureId>,
        emissive_texture_id: Option<crate::resources::TextureId>,
    ) {
        let hash_str = |name: &str| -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            name.hash(&mut h);
            h.finish()
        };
        let attr_hash = active_attr.map(|n| hash_str(n)).unwrap_or(u64::MAX);
        let warp_hash = warp_attr.map(|n| hash_str(n)).unwrap_or(u64::MAX);

        // The last two slots track GPU position/normal override (re)bind events.
        // Bumped by `set_*_override_buffer` / `clear_*_override`, so a fresh
        // override forces a bind-group rebuild here.
        let (pos_override_gen, nrm_override_gen, has_extension_attr, lightmap_gen, lightmap_tex_id) = {
            let Some(mesh) = self.mesh_store.get(mesh_id) else {
                return;
            };
            (
                mesh.position_override_gen,
                mesh.normal_override_gen,
                mesh.extension_attr_buffer.is_some() as u64,
                mesh.lightmap_gen,
                mesh.lightmap.as_ref().map(|lm| lm.texture_id),
            )
        };

        let key = (
            albedo_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            lut_id.map(|id| id.0 as u64).unwrap_or(u64::MAX),
            attr_hash,
            matcap_id.map(|id| id.index as u64).unwrap_or(u64::MAX),
            warp_hash,
            metallic_roughness_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            emissive_texture_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            pos_override_gen,
            nrm_override_gen,
            // Pack the extension-attr presence bit with the lightmap gen so a
            // lightmap set/clear invalidates the cached bind group. The tuple
            // stays at 12 elements (its PartialEq arity limit).
            has_extension_attr | (lightmap_gen << 1),
        );

        {
            let Some(mesh) = self.mesh_store.get(mesh_id) else {
                return;
            };
            if mesh.last_tex_key == key {
                return;
            }
        }

        let albedo_view = match albedo_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_texture.view,
        };
        let normal_view = match normal_map_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_normal_map_view,
        };
        let ao_view = match ao_map_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_ao_map_view,
        };
        let lut_view = match lut_id {
            Some(id) if id.0 < self.content.colourmap_views.len() => {
                &self.content.colourmap_views[id.0]
            }
            _ => &self.content.fallback_lut_view,
        };
        // The lightmap texture is resolved from the mesh's own registration, not
        // a caller-supplied id, so it is looked up here before the mutable mesh
        // borrow below.
        let lightmap_view = match lightmap_tex_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_texture.view,
        };

        let Some(mesh) = self.mesh_store.get_mut(mesh_id) else {
            return;
        };

        let scalar_buf: &crate::gpu::Buffer = match active_attr {
            Some(name) => {
                let found_vertex = mesh.attribute_buffers.get(name);
                let found_face = mesh.face_attribute_buffers.get(name);
                found_vertex
                    .or(found_face)
                    .unwrap_or(&self.content.fallback_scalar_buf)
            }
            None => &self.content.fallback_scalar_buf,
        };

        let face_colour_buf: &crate::gpu::Buffer = match active_attr {
            Some(name) => mesh
                .face_colour_buffers
                .get(name)
                .unwrap_or(&self.content.fallback_face_colour_buf),
            None => &self.content.fallback_face_colour_buf,
        };

        // Resolve matcap texture view : fallback to 1x1 white when no matcap active.
        let matcap_view: &crate::gpu::TextureView = match matcap_id {
            Some(id) if id.index < self.content.matcap_views.len() => {
                &self.content.matcap_views[id.index]
            }
            _ => self
                .content
                .fallback_matcap_view
                .as_ref()
                .unwrap_or(&self.fallback_texture.view),
        };

        let warp_buf: &crate::gpu::Buffer = match warp_attr {
            Some(name) => mesh
                .vector_attribute_buffers
                .get(name)
                .unwrap_or(&self.content.fallback_warp_buf),
            None => &self.content.fallback_warp_buf,
        };

        let position_override_buf: &crate::gpu::Buffer = mesh
            .position_override_buffer
            .as_ref()
            .unwrap_or(&self.content.fallback_position_override_buf);
        let normal_override_buf: &crate::gpu::Buffer = mesh
            .normal_override_buffer
            .as_ref()
            .unwrap_or(&self.content.fallback_normal_override_buf);
        // The binding-15 vec4 sidecar carries the plugin vertex attribute or a
        // baked lightmap's UV1; the lightmap wins the slot when both are set.
        let sidecar_buf: &crate::gpu::Buffer = mesh
            .lightmap
            .as_ref()
            .map(|lm| &lm.uv1_buffer)
            .or(mesh.extension_attr_buffer.as_ref())
            .unwrap_or(&self.content.fallback_extension_attr_buf);

        let metallic_roughness_view: &crate::gpu::TextureView = match metallic_roughness_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_metallic_roughness_texture_view,
        };
        let emissive_view: &crate::gpu::TextureView = match emissive_texture_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_emissive_texture_view,
        };

        mesh.object_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("object_bind_group"),
            layout: &self.object_bind_group_layout,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh.object_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(albedo_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&self.material_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(normal_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(ao_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: scalar_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: crate::gpu::BindingResource::TextureView(matcap_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: face_colour_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 9,
                    resource: warp_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 10,
                    resource: crate::gpu::BindingResource::Sampler(&self.lut_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 11,
                    resource: crate::gpu::BindingResource::TextureView(metallic_roughness_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 12,
                    resource: crate::gpu::BindingResource::TextureView(emissive_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 13,
                    resource: position_override_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 14,
                    resource: normal_override_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 15,
                    resource: sidecar_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 17,
                    resource: crate::gpu::BindingResource::TextureView(lightmap_view),
                },
            ],
        });
        mesh.last_tex_key = key;
    }

    /// Build an object bind group that pairs an external per-item uniform buffer with
    /// the mesh's textures/LUT/matcap/scalar/override resources. Returns the bind group
    /// and the cache key that was used to construct it.
    ///
    /// This mirrors the resource-resolution in `update_mesh_texture_bind_group`, but
    /// reads from a caller-supplied uniform buffer instead of the mesh's shared
    /// `object_uniform_buf`. The per-object draw path uses one of these per scene item
    /// so that items sharing a `MeshId` each get their own transform.
    pub(crate) fn build_per_item_object_bind_group(
        &self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        item_uniform_buf: &crate::gpu::Buffer,
        albedo_id: Option<crate::resources::TextureId>,
        normal_map_id: Option<crate::resources::TextureId>,
        ao_map_id: Option<crate::resources::TextureId>,
        lut_id: Option<ColourmapId>,
        active_attr: Option<&str>,
        matcap_id: Option<crate::resources::MatcapId>,
        warp_attr: Option<&str>,
        metallic_roughness_id: Option<crate::resources::TextureId>,
        emissive_texture_id: Option<crate::resources::TextureId>,
        prev_key: Option<u64>,
    ) -> Option<(crate::gpu::BindGroup, u64)> {
        let hash_str = |name: &str| -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            name.hash(&mut h);
            h.finish()
        };
        let attr_hash = active_attr.map(|n| hash_str(n)).unwrap_or(u64::MAX);
        let warp_hash = warp_attr.map(|n| hash_str(n)).unwrap_or(u64::MAX);

        let mesh = self.mesh_store.get(mesh_id)?;
        let pos_override_gen = mesh.position_override_gen;
        let nrm_override_gen = mesh.normal_override_gen;

        let cache_key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            // Index and generation both: cached per-object entries can now
            // outlive a mesh slot's occupant, so a freed-and-reused slot must
            // not alias the old occupant's bind group.
            mesh_id.index().hash(&mut h);
            mesh_id.generation.hash(&mut h);
            albedo_id.map(|t| t.raw()).unwrap_or(u64::MAX).hash(&mut h);
            normal_map_id
                .map(|t| t.raw())
                .unwrap_or(u64::MAX)
                .hash(&mut h);
            ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX).hash(&mut h);
            lut_id
                .map(|id| id.0 as u64)
                .unwrap_or(u64::MAX)
                .hash(&mut h);
            attr_hash.hash(&mut h);
            matcap_id
                .map(|id| id.index as u64)
                .unwrap_or(u64::MAX)
                .hash(&mut h);
            warp_hash.hash(&mut h);
            metallic_roughness_id
                .map(|t| t.raw())
                .unwrap_or(u64::MAX)
                .hash(&mut h);
            emissive_texture_id
                .map(|t| t.raw())
                .unwrap_or(u64::MAX)
                .hash(&mut h);
            pos_override_gen.hash(&mut h);
            nrm_override_gen.hash(&mut h);
            mesh.extension_attr_buffer.is_some().hash(&mut h);
            mesh.lightmap_gen.hash(&mut h);
            h.finish()
        };

        // Cache hit: the previously built bind group is still valid, so skip the
        // create_bind_group below. The caller keeps its existing bind group. The
        // per-item uniform write happens at the call site regardless, since the
        // transform changes each frame.
        if prev_key == Some(cache_key) {
            return None;
        }

        let albedo_view = match albedo_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_texture.view,
        };
        let normal_view = match normal_map_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_normal_map_view,
        };
        let ao_view = match ao_map_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_ao_map_view,
        };
        let lut_view = match lut_id {
            Some(id) if id.0 < self.content.colourmap_views.len() => {
                &self.content.colourmap_views[id.0]
            }
            _ => &self.content.fallback_lut_view,
        };

        let scalar_buf: &crate::gpu::Buffer = match active_attr {
            Some(name) => {
                let found_vertex = mesh.attribute_buffers.get(name);
                let found_face = mesh.face_attribute_buffers.get(name);
                found_vertex
                    .or(found_face)
                    .unwrap_or(&self.content.fallback_scalar_buf)
            }
            None => &self.content.fallback_scalar_buf,
        };

        let face_colour_buf: &crate::gpu::Buffer = match active_attr {
            Some(name) => mesh
                .face_colour_buffers
                .get(name)
                .unwrap_or(&self.content.fallback_face_colour_buf),
            None => &self.content.fallback_face_colour_buf,
        };

        let matcap_view: &crate::gpu::TextureView = match matcap_id {
            Some(id) if id.index < self.content.matcap_views.len() => {
                &self.content.matcap_views[id.index]
            }
            _ => self
                .content
                .fallback_matcap_view
                .as_ref()
                .unwrap_or(&self.fallback_texture.view),
        };

        let warp_buf: &crate::gpu::Buffer = match warp_attr {
            Some(name) => mesh
                .vector_attribute_buffers
                .get(name)
                .unwrap_or(&self.content.fallback_warp_buf),
            None => &self.content.fallback_warp_buf,
        };

        let position_override_buf: &crate::gpu::Buffer = mesh
            .position_override_buffer
            .as_ref()
            .unwrap_or(&self.content.fallback_position_override_buf);
        let normal_override_buf: &crate::gpu::Buffer = mesh
            .normal_override_buffer
            .as_ref()
            .unwrap_or(&self.content.fallback_normal_override_buf);
        // Binding-15 vec4 sidecar: lightmap UV1 wins the slot, else the plugin
        // vertex attribute, else the shared zero fallback.
        let sidecar_buf: &crate::gpu::Buffer = mesh
            .lightmap
            .as_ref()
            .map(|lm| &lm.uv1_buffer)
            .or(mesh.extension_attr_buffer.as_ref())
            .unwrap_or(&self.content.fallback_extension_attr_buf);
        let lightmap_view: &crate::gpu::TextureView =
            match mesh.lightmap.as_ref().map(|lm| lm.texture_id) {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_texture.view,
            };

        let metallic_roughness_view: &crate::gpu::TextureView = match metallic_roughness_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_metallic_roughness_texture_view,
        };
        let emissive_view: &crate::gpu::TextureView = match emissive_texture_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_emissive_texture_view,
        };

        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("per_item_object_bind_group"),
            layout: &self.object_bind_group_layout,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: item_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(albedo_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&self.material_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(normal_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(ao_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: scalar_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: crate::gpu::BindingResource::TextureView(matcap_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: face_colour_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 9,
                    resource: warp_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 10,
                    resource: crate::gpu::BindingResource::Sampler(&self.lut_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 11,
                    resource: crate::gpu::BindingResource::TextureView(metallic_roughness_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 12,
                    resource: crate::gpu::BindingResource::TextureView(emissive_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 13,
                    resource: position_override_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 14,
                    resource: normal_override_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 15,
                    resource: sidecar_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 17,
                    resource: crate::gpu::BindingResource::TextureView(lightmap_view),
                },
            ],
        });
        Some((bg, cache_key))
    }

    /// Upload a 256-sample RGBA colourmap to the GPU and return its `ColourmapId`.
    ///
    /// The returned ID can be stored in `SceneRenderItem::colourmap_id`.
    /// Use `BuiltinColourmap` variants + [`Self::builtin_colourmap_id`] for the built-in presets.
    pub fn upload_colourmap(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        rgba_data: &[[u8; 4]; 256],
    ) -> ColourmapId {
        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("lut_texture"),
            size: crate::gpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let flat: Vec<u8> = rgba_data.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            &flat,
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(1),
            },
            crate::gpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        let id = ColourmapId(self.content.colourmap_textures.len());
        self.content.colourmap_textures.push(texture);
        self.content.colourmap_views.push(view);
        self.content.colourmaps_cpu.push(*rgba_data);
        id
    }

    /// Return the CPU-side colourmap LUT for `id` as 256 RGBA8 entries, or `None` if the id is invalid.
    ///
    /// Useful for any non-GPU colourmap output: PDF export, table cell colouring, custom legend
    /// widgets, or sampling a colour at a specific scalar value. The data is always in memory
    /// (kept for GPU upload) so this accessor is free.
    pub fn get_colourmap_rgba(&self, id: ColourmapId) -> Option<&[[u8; 4]; 256]> {
        self.content.colourmaps_cpu.get(id.0)
    }

    /// Return the `ColourmapId` for a built-in preset.
    ///
    /// Call [`Self::ensure_colourmaps_initialized`] first (done automatically by
    /// `ViewportRenderer::prepare`).  Panics if colourmaps have not been initialized yet.
    pub fn builtin_colourmap_id(&self, preset: BuiltinColourmap) -> ColourmapId {
        self.content
            .builtin_colourmap_ids
            .expect("call ensure_colourmaps_initialized before using built-in colourmaps")
            [preset as usize]
    }

    /// Ensure built-in colourmaps are uploaded to the GPU.
    ///
    /// Called automatically by `ViewportRenderer::prepare()` on the first frame.
    /// Safe to call multiple times : no-op after first invocation.
    pub fn ensure_colourmaps_initialized(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        if self.content.colourmaps_initialized {
            return;
        }
        let viridis = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::viridis_rgba(),
        );
        let plasma = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::plasma_rgba(),
        );
        let greyscale = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::greyscale_rgba(),
        );
        let coolwarm = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::coolwarm_rgba(),
        );
        let rainbow = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::rainbow_rgba(),
        );
        let magma = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::magma_rgba(),
        );
        let inferno = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::inferno_rgba(),
        );
        let turbo = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::turbo_rgba(),
        );
        let jet = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::jet_rgba(),
        );
        let rdbu = self.upload_colourmap(
            device,
            queue,
            &crate::resources::material::colourmap_data::rdbu_r_rgba(),
        );
        self.content.builtin_colourmap_ids = Some([
            viridis, plasma, greyscale, coolwarm, rainbow, magma, inferno, turbo, jet, rdbu,
        ]);
        self.content.colourmaps_initialized = true;
    }

    // -----------------------------------------------------------------------
    // Matcap texture API
    // -----------------------------------------------------------------------

    /// Upload a 256x256 RGBA matcap texture and return its `MatcapId`.
    ///
    /// `rgba_data` must be exactly `256 * 256 * 4 = 262_144` bytes.
    /// Set `blendable = true` for matcaps whose alpha channel tints the base
    /// geometry colour; `false` for static matcaps that fully replace the colour.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// if `rgba_data` has the wrong length.
    pub fn upload_matcap(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        rgba_data: &[u8],
        blendable: bool,
    ) -> crate::error::ViewportResult<crate::resources::MatcapId> {
        let (width, height) = (256u32, 256u32);
        let expected = (width * height * 4) as usize;
        if rgba_data.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba_data.len(),
            });
        }

        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("matcap_texture"),
            size: crate::gpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            rgba_data,
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            crate::gpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Ensure the shared clamp sampler is created.
        if self.content.matcap_sampler.is_none() {
            self.content.matcap_sampler = Some(crate::resources::builders::clamp_linear_sampler(
                device,
                "matcap_sampler",
            ));
        }

        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        let index = self.content.matcap_textures.len();
        self.content.matcap_textures.push(texture);
        self.content.matcap_views.push(view);

        // Lazily initialise the fallback matcap view to binding 7 of the
        // first uploaded texture (a plain white 1x1 is fine as fallback).
        if self.content.fallback_matcap_view.is_none() {
            self.content.fallback_matcap_view = Some(
                self.fallback_texture
                    .texture
                    .create_view(&crate::gpu::TextureViewDescriptor::default()),
            );
        }

        tracing::debug!(matcap_index = index, blendable, "matcap uploaded");
        Ok(crate::resources::MatcapId { index, blendable })
    }

    /// Return the `MatcapId` for a built-in preset.
    ///
    /// Panics if called before the renderer has run at least one prepare pass
    /// (which calls [`Self::ensure_matcaps_initialized`] automatically).
    pub fn builtin_matcap_id(
        &self,
        preset: crate::resources::BuiltinMatcap,
    ) -> crate::resources::MatcapId {
        self.content.builtin_matcap_ids
            .expect("call ensure_matcaps_initialized (or run one prepare frame) before using built-in matcaps")
            [preset as usize]
    }

    /// Upload the eight built-in matcaps to the GPU if not already done.
    ///
    /// Called automatically by `ViewportRenderer::prepare()`. Safe to call
    /// multiple times : no-op after first invocation.
    pub fn ensure_matcaps_initialized(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        if self.content.matcaps_initialized {
            return;
        }
        use crate::resources::material::matcap_data;
        let clay = self
            .upload_matcap(device, queue, &matcap_data::clay(), true)
            .unwrap();
        let wax = self
            .upload_matcap(device, queue, &matcap_data::wax(), true)
            .unwrap();
        let candy = self
            .upload_matcap(device, queue, &matcap_data::candy(), true)
            .unwrap();
        let flat = self
            .upload_matcap(device, queue, &matcap_data::flat(), true)
            .unwrap();
        let ceramic = self
            .upload_matcap(device, queue, &matcap_data::ceramic(), false)
            .unwrap();
        let jade = self
            .upload_matcap(device, queue, &matcap_data::jade(), false)
            .unwrap();
        let mud = self
            .upload_matcap(device, queue, &matcap_data::mud(), false)
            .unwrap();
        let normal = self
            .upload_matcap(device, queue, &matcap_data::normal(), false)
            .unwrap();
        self.content.builtin_matcap_ids =
            Some([clay, wax, candy, flat, ceramic, jade, mud, normal]);
        self.content.matcaps_initialized = true;
    }
}

#[cfg(test)]
mod async_texture_tests {
    use crate::DeviceResources;
    use crate::resources::UploadStatus;

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    /// Device with `TEXTURE_COMPRESSION_BC` enabled, or `None` when the adapter
    /// does not support BC (e.g. a software / mobile adapter) so the caller can
    /// skip the test.
    fn try_make_bc_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        if !adapter
            .features()
            .contains(crate::gpu::Features::TEXTURE_COMPRESSION_BC)
        {
            return None;
        }
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
            required_features: crate::gpu::Features::TEXTURE_COMPRESSION_BC,
            ..Default::default()
        }))
        .ok()
    }

    fn drive_until_ready(
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::JobId,
    ) {
        for _ in 0..200 {
            resources.process_uploads(device, queue);
            match resources.upload_status(id) {
                UploadStatus::Ready => return,
                UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                UploadStatus::Unknown => panic!("job id disappeared"),
            }
        }
        panic!("texture upload did not complete in time");
    }

    #[test]
    fn invalid_size_errors_synchronously() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // 2x2 image requires 16 bytes. Pass 12 and confirm the error fires
        // before any job is submitted.
        let rgba = vec![0u8; 12];
        let err = resources
            .begin_upload_texture(&device, &queue, 2, 2, rgba)
            .expect_err("invalid size should error");
        assert!(matches!(
            err,
            crate::error::ViewportError::InvalidTextureData {
                expected: 16,
                actual: 12
            }
        ));
        assert_eq!(resources.uploads_pending(), 0);
    }

    #[test]
    fn begin_upload_texture_completes_and_yields_id() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let rgba = vec![128u8; 4 * 4 * 4];
        let id = resources
            .begin_upload_texture(&device, &queue, 4, 4, rgba)
            .unwrap();
        assert_eq!(resources.uploads_pending(), 1);

        // Result is not available until the worker finishes.
        let err = resources.upload_result_texture(id).unwrap_err();
        assert!(matches!(err, crate::error::ViewportError::JobNotReady));

        drive_until_ready(&mut resources, &device, &queue, id);

        let tex_id = resources.upload_result_texture(id).expect("ready result");
        // The first uploaded texture lands at index 0.
        assert_eq!(tex_id, crate::resources::TextureId(0));

        // Taking the result again reports missing.
        let err = resources.upload_result_texture(id).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::JobResultMissing { .. }
        ));
    }

    #[test]
    fn begin_upload_normal_map_routes_to_same_result_accessor() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let rgba = vec![64u8; 8 * 8 * 4];
        let id = resources
            .begin_upload_normal_map(&device, &queue, 8, 8, rgba)
            .unwrap();
        drive_until_ready(&mut resources, &device, &queue, id);
        let tex_id = resources.upload_result_texture(id).expect("ready result");
        assert_eq!(tex_id, crate::resources::TextureId(0));
    }

    #[test]
    fn sync_upload_still_works() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let rgba = vec![200u8; 4 * 4 * 4];
        let tex_id = resources
            .upload_texture(&device, &queue, 4, 4, &rgba)
            .unwrap();
        assert_eq!(tex_id, crate::resources::TextureId(0));
    }

    #[test]
    fn replace_texture_keeps_handle_and_updates_bytes() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let id = resources
            .upload_texture(&device, &queue, 4, 4, &vec![200u8; 4 * 4 * 4])
            .unwrap();
        let bytes_4x4 = resources.resident_bytes().texture_bytes;

        // Replace in place with a larger image: same handle, larger byte total.
        resources
            .replace_texture(&device, &queue, id, 8, 8, &vec![10u8; 8 * 8 * 4])
            .expect("replace on a live handle succeeds");
        assert!(resources.texture_view(id).is_some(), "handle stays valid");
        let bytes_8x8 = resources.resident_bytes().texture_bytes;
        assert!(
            bytes_8x8 > bytes_4x4,
            "replacing with a larger image must grow resident bytes"
        );

        // Wrong data length is rejected before touching the slot.
        let err = resources
            .replace_texture(&device, &queue, id, 8, 8, &[0u8; 3])
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::InvalidTextureData { .. }
        ));

        // A stale handle is rejected.
        assert!(resources.free_texture(id));
        let err = resources
            .replace_texture(&device, &queue, id, 4, 4, &vec![0u8; 4 * 4 * 4])
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::StaleHandle { .. }
        ));
    }

    #[test]
    fn block_layout_matches_format() {
        use super::mip_block_layout;
        use crate::gpu::TextureFormat as F;

        // Uncompressed RGBA8: 1x1 blocks, 4 bytes/texel.
        assert_eq!(mip_block_layout(F::Rgba8Unorm, 4, 4), (16, 4, 64));

        // BC7: 4x4 blocks, 16 bytes/block.
        assert_eq!(mip_block_layout(F::Bc7RgbaUnormSrgb, 4, 4), (16, 1, 16));
        assert_eq!(mip_block_layout(F::Bc7RgbaUnormSrgb, 8, 8), (32, 2, 64));
        // Non-4-aligned dimensions round up to whole blocks.
        assert_eq!(mip_block_layout(F::Bc7RgbaUnormSrgb, 6, 6), (32, 2, 64));

        // BC4: 4x4 blocks, 8 bytes/block.
        assert_eq!(mip_block_layout(F::Bc4RUnorm, 8, 8), (16, 2, 32));

        // ASTC 8x8: 8x8 blocks, 16 bytes/block.
        let astc = F::Astc {
            block: crate::gpu::AstcBlock::B8x8,
            channel: crate::gpu::AstcChannel::Unorm,
        };
        assert_eq!(mip_block_layout(astc, 16, 16), (32, 2, 64));
    }

    #[test]
    fn block_layout_full_mip_pyramid_sum() {
        use super::mip_block_layout;
        // 64x64 BC7 pyramid down to 1x1: 4096+1024+256+64+16+16+16.
        let mut total = 0usize;
        let (w, h) = (64u32, 64u32);
        for level in 0..=6 {
            let lw = (w >> level).max(1);
            let lh = (h >> level).max(1);
            let (_, _, bytes) =
                mip_block_layout(crate::gpu::TextureFormat::Bc7RgbaUnormSrgb, lw, lh);
            total += bytes;
        }
        assert_eq!(total, 5488);
    }

    #[test]
    fn supports_texture_format_false_without_feature() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        // The default device requests no features, so BC is never enabled even
        // when the adapter could support it.
        assert!(!crate::resources::supports_texture_format(
            &device,
            crate::gpu::TextureFormat::Bc7RgbaUnormSrgb
        ));
        // An uncompressed format needs no feature and is always supported.
        assert!(crate::resources::supports_texture_format(
            &device,
            crate::gpu::TextureFormat::Rgba8Unorm
        ));
    }

    #[test]
    fn compressed_upload_rejects_unsupported_and_non_block_formats() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // BC7 without the feature: rejected up front, no job submitted.
        let block = vec![0u8; 16];
        let err = resources
            .begin_upload_compressed_texture(
                &device,
                &queue,
                crate::resources::CompressedTextureDesc {
                    width: 4,
                    height: 4,
                    format: crate::gpu::TextureFormat::Bc7RgbaUnormSrgb,
                    is_normal_map: false,
                    mip_levels: &[&block],
                },
            )
            .expect_err("BC7 upload without the feature should error");
        assert!(matches!(
            err,
            crate::error::ViewportError::UnsupportedTextureFormat { .. }
        ));

        // A non-compressed format is also rejected by this path.
        let rgba = vec![0u8; 4 * 4 * 4];
        let err = resources
            .begin_upload_compressed_texture(
                &device,
                &queue,
                crate::resources::CompressedTextureDesc {
                    width: 4,
                    height: 4,
                    format: crate::gpu::TextureFormat::Rgba8Unorm,
                    is_normal_map: false,
                    mip_levels: &[&rgba],
                },
            )
            .expect_err("non-compressed format should error");
        assert!(matches!(
            err,
            crate::error::ViewportError::UnsupportedTextureFormat { .. }
        ));
        assert_eq!(resources.uploads_pending(), 0);
    }

    #[test]
    fn compressed_upload_validates_level_lengths() {
        let Some((device, queue)) = try_make_bc_device() else {
            eprintln!("skipping: no adapter with TEXTURE_COMPRESSION_BC");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // 4x4 BC7 needs one 16-byte block; pass 15 and confirm the level check
        // fires before any job is submitted.
        let bad = vec![0u8; 15];
        let err = resources
            .begin_upload_compressed_texture(
                &device,
                &queue,
                crate::resources::CompressedTextureDesc {
                    width: 4,
                    height: 4,
                    format: crate::gpu::TextureFormat::Bc7RgbaUnormSrgb,
                    is_normal_map: false,
                    mip_levels: &[&bad],
                },
            )
            .expect_err("wrong block length should error");
        assert!(matches!(
            err,
            crate::error::ViewportError::InvalidCompressedTextureData {
                level: 0,
                expected: 16,
                actual: 15,
            }
        ));

        // Empty mip chain is rejected too.
        let err = resources
            .begin_upload_compressed_texture(
                &device,
                &queue,
                crate::resources::CompressedTextureDesc {
                    width: 4,
                    height: 4,
                    format: crate::gpu::TextureFormat::Bc7RgbaUnormSrgb,
                    is_normal_map: false,
                    mip_levels: &[],
                },
            )
            .expect_err("empty mip chain should error");
        assert!(matches!(
            err,
            crate::error::ViewportError::InvalidCompressedTextureData { actual: 0, .. }
        ));
        assert_eq!(resources.uploads_pending(), 0);
    }

    #[test]
    fn compressed_upload_completes_and_counts_bytes() {
        let Some((device, queue)) = try_make_bc_device() else {
            eprintln!("skipping: no adapter with TEXTURE_COMPRESSION_BC");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let before = resources.texture_memory_stats().used_bytes;
        // 4x4 BC7 = one 16-byte block. Contents need not decode to anything in
        // particular; we only exercise the upload path and byte accounting.
        let block = vec![0u8; 16];
        let id = resources
            .begin_upload_compressed_texture(
                &device,
                &queue,
                crate::resources::CompressedTextureDesc {
                    width: 4,
                    height: 4,
                    format: crate::gpu::TextureFormat::Bc7RgbaUnormSrgb,
                    is_normal_map: false,
                    mip_levels: &[&block],
                },
            )
            .unwrap();
        drive_until_ready(&mut resources, &device, &queue, id);
        let tex_id = resources.upload_result_texture(id).expect("ready result");
        assert_eq!(tex_id, crate::resources::TextureId(0));

        let stats = resources.texture_memory_stats();
        assert_eq!(stats.used_bytes - before, 16);
        assert_eq!(stats.texture_count, 1);
    }

    #[test]
    fn compressed_upload_rejects_non_block_aligned_dimensions() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // wgpu cannot create a BC texture whose dimensions are not multiples of
        // the 4x4 block. The upload must reject this up front (before any GPU
        // work) so a consumer can fall back to an uncompressed upload rather
        // than binding an invalid texture. 1419 mirrors a real asset dimension.
        for (w, h) in [(6u32, 6u32), (1419, 1024), (1024, 1419)] {
            let blocks_x = w.div_ceil(4);
            let blocks_y = h.div_ceil(4);
            let block = vec![0u8; (blocks_x * blocks_y * 16) as usize];
            let err = resources
                .begin_upload_compressed_texture(
                    &device,
                    &queue,
                    crate::resources::CompressedTextureDesc {
                        width: w,
                        height: h,
                        format: crate::gpu::TextureFormat::Bc7RgbaUnormSrgb,
                        is_normal_map: false,
                        mip_levels: &[&block],
                    },
                )
                .expect_err("non-block-aligned dimensions must be rejected");
            assert!(matches!(
                err,
                crate::error::ViewportError::CompressedTextureNotBlockAligned { .. }
            ));
        }
        assert_eq!(resources.uploads_pending(), 0);
    }
}

// ---------------------------------------------------------------------------
// GpuTexture: GPU texture with sampler and bind group
// ---------------------------------------------------------------------------

/// A GPU texture with its view, sampler, and bind group for shader binding.
pub struct GpuTexture {
    /// Underlying wgpu texture object.
    pub texture: crate::gpu::Texture,
    /// Full-texture view used for sampling.
    pub view: crate::gpu::TextureView,
    /// Sampler bound alongside the view.
    pub sampler: crate::gpu::Sampler,
    /// Bind group that binds `view` and `sampler` for use in shaders. Built at
    /// upload time and kept with the texture; draw paths currently build their
    /// own material bind groups, so this one is not read.
    #[allow(dead_code)]
    pub bind_group: crate::gpu::BindGroup,
}
