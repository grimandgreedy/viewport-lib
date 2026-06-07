use super::*;

impl ViewportGpuResources {
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> crate::error::ViewportResult<u64> {
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> crate::error::ViewportResult<u64> {
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
    /// Returns a `JobId` immediately. The texture and bind group are built
    /// on a worker thread; `queue.write_texture` queues the pixel copy and
    /// the runner gates the job on a fresh submission that flushes those
    /// writes. Once the status is `Ready`, take the resulting texture id
    /// with `upload_result_texture` and store it in `Material::texture_id`.
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        let label = TextureAsyncLabel::Albedo;
        Ok(self.spawn_texture_upload(device, queue, width, height, rgba, label))
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        let label = TextureAsyncLabel::NormalMap;
        Ok(self.spawn_texture_upload(device, queue, width, height, rgba, label))
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
    ) -> crate::error::ViewportResult<u64> {
        let mut map = self
            .job_texture_results
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

    /// Shared spawn path for `begin_upload_texture` and
    /// `begin_upload_normal_map`. The label decides texture format and
    /// which bind-group slot the texture occupies.
    fn spawn_texture_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        label: TextureAsyncLabel,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<u64>::new();
        let slot_for_apply = slot.clone();

        // Clone the fallback views and the bind-group layout into the
        // worker so it can build the GpuTexture and bind group without
        // touching `self` from the worker thread.
        let bgl = self.texture_bind_group_layout.clone();
        let fallback_albedo_view = self.fallback_texture.view.clone();
        let fallback_normal_view = self.fallback_normal_map_view.clone();
        let fallback_ao_view = self.fallback_ao_map_view.clone();
        let data_bytes = rgba.len() as u64;

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_with_gpu(device, queue, move |dev, q, progress| {
                progress.set(0.2);
                let format = match label {
                    TextureAsyncLabel::Albedo => wgpu::TextureFormat::Rgba8UnormSrgb,
                    TextureAsyncLabel::NormalMap => wgpu::TextureFormat::Rgba8Unorm,
                };
                let tex_label = match label {
                    TextureAsyncLabel::Albedo => "async_user_texture",
                    TextureAsyncLabel::NormalMap => "async_normal_map_texture",
                };
                let texture = dev.create_texture(&wgpu::TextureDescriptor {
                    label: Some(tex_label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                q.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                progress.set(0.7);

                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler_label = match label {
                    TextureAsyncLabel::Albedo => "async_user_texture_sampler",
                    TextureAsyncLabel::NormalMap => "async_normal_map_sampler",
                };
                let sampler = dev.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some(sampler_label),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                });
                // Bind-group entries differ between albedo and normal map:
                // albedo binds the new view to slot 0 and the fallback
                // normal to slot 2; normal map binds the fallback albedo
                // to slot 0 and the new view to slot 2.
                let bg_label = match label {
                    TextureAsyncLabel::Albedo => "async_user_texture_bg",
                    TextureAsyncLabel::NormalMap => "async_normal_map_bg",
                };
                let (slot0_view, slot2_view) = match label {
                    TextureAsyncLabel::Albedo => (&view, &fallback_normal_view),
                    TextureAsyncLabel::NormalMap => (&fallback_albedo_view, &view),
                };
                let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(bg_label),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(slot0_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(slot2_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&fallback_ao_view),
                        },
                    ],
                });
                progress.set(0.9);

                // Flush so the runner has a submission to gate on. Implicit
                // writes queued above are folded into this submit by wgpu.
                let encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("async_texture_flush"),
                });
                let submission = q.submit(std::iter::once(encoder.finish()));
                progress.set(1.0);

                let gpu_texture = GpuTexture {
                    texture,
                    view,
                    sampler,
                    bind_group,
                };
                Ok(crate::resources::upload_jobs::JobProduct::with_gpu_and_apply(
                    submission,
                    Box::new(move |resources: &mut ViewportGpuResources| {
                        let tex_id = resources.textures.len() as u64;
                        resources.textures.push(gpu_texture);
                        resources.texture_allocated_bytes += data_bytes;
                        slot_for_apply.set(tex_id);
                    }),
                ))
            })
        };

        self.job_texture_results
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
            used_bytes: self.texture_allocated_bytes,
            texture_count: self.textures.len() as u32,
        }
    }
}

/// Discriminator used by `spawn_texture_upload` to switch between the
/// albedo and normal-map paths without duplicating the worker body.
#[derive(Clone, Copy)]
enum TextureAsyncLabel {
    Albedo,
    NormalMap,
}

impl ViewportGpuResources {
    /// Get or create a cached material bind group for (albedo, normal_map, ao_map) texture combo.
    ///
    /// `u64::MAX` sentinel means "use fallback texture for that slot".
    /// The bind group is cached in `material_bind_groups` keyed by the 3-tuple.
    #[allow(dead_code)]
    pub(crate) fn get_material_bind_group(
        &mut self,
        device: &wgpu::Device,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
    ) -> &wgpu::BindGroup {
        let key = (
            albedo_id.unwrap_or(u64::MAX),
            normal_map_id.unwrap_or(u64::MAX),
            ao_map_id.unwrap_or(u64::MAX),
        );

        if !self.material_bind_groups.contains_key(&key) {
            let albedo_view = match albedo_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_bg"),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                ],
            });
            self.material_bind_groups.insert(key, bg);
        }

        self.material_bind_groups.get(&key).unwrap()
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
        device: &wgpu::Device,
        mesh_id: crate::resources::mesh_store::MeshId,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
        lut_id: Option<ColourmapId>,
        active_attr: Option<&str>,
        matcap_id: Option<crate::resources::MatcapId>,
        warp_attr: Option<&str>,
        metallic_roughness_id: Option<u64>,
        emissive_texture_id: Option<u64>,
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
        let (pos_override_gen, nrm_override_gen) = {
            let Some(mesh) = self.mesh_store.get(mesh_id) else {
                return;
            };
            (mesh.position_override_gen, mesh.normal_override_gen)
        };

        let key = (
            albedo_id.unwrap_or(u64::MAX),
            normal_map_id.unwrap_or(u64::MAX),
            ao_map_id.unwrap_or(u64::MAX),
            lut_id.map(|id| id.0 as u64).unwrap_or(u64::MAX),
            attr_hash,
            matcap_id.map(|id| id.index as u64).unwrap_or(u64::MAX),
            warp_hash,
            metallic_roughness_id.unwrap_or(u64::MAX),
            emissive_texture_id.unwrap_or(u64::MAX),
            pos_override_gen,
            nrm_override_gen,
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
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_texture.view,
        };
        let normal_view = match normal_map_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_normal_map_view,
        };
        let ao_view = match ao_map_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_ao_map_view,
        };
        let lut_view = match lut_id {
            Some(id) if id.0 < self.colourmap_views.len() => &self.colourmap_views[id.0],
            _ => &self.fallback_lut_view,
        };

        let Some(mesh) = self.mesh_store.get_mut(mesh_id) else {
            return;
        };

        let scalar_buf: &wgpu::Buffer = match active_attr {
            Some(name) => {
                let found_vertex = mesh.attribute_buffers.get(name);
                let found_face = mesh.face_attribute_buffers.get(name);
                found_vertex
                    .or(found_face)
                    .unwrap_or(&self.fallback_scalar_buf)
            }
            None => &self.fallback_scalar_buf,
        };

        let face_colour_buf: &wgpu::Buffer = match active_attr {
            Some(name) => mesh
                .face_colour_buffers
                .get(name)
                .unwrap_or(&self.fallback_face_colour_buf),
            None => &self.fallback_face_colour_buf,
        };

        // Resolve matcap texture view : fallback to 1x1 white when no matcap active.
        let matcap_view: &wgpu::TextureView = match matcap_id {
            Some(id) if id.index < self.matcap_views.len() => &self.matcap_views[id.index],
            _ => self
                .fallback_matcap_view
                .as_ref()
                .unwrap_or(&self.fallback_texture.view),
        };

        let warp_buf: &wgpu::Buffer = match warp_attr {
            Some(name) => mesh
                .vector_attribute_buffers
                .get(name)
                .unwrap_or(&self.fallback_warp_buf),
            None => &self.fallback_warp_buf,
        };

        let position_override_buf: &wgpu::Buffer = mesh
            .position_override_buffer
            .as_ref()
            .unwrap_or(&self.fallback_position_override_buf);
        let normal_override_buf: &wgpu::Buffer = mesh
            .normal_override_buffer
            .as_ref()
            .unwrap_or(&self.fallback_normal_override_buf);

        let metallic_roughness_view: &wgpu::TextureView = match metallic_roughness_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_metallic_roughness_texture_view,
        };
        let emissive_view: &wgpu::TextureView = match emissive_texture_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_emissive_texture_view,
        };

        mesh.object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object_bind_group"),
            layout: &self.object_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh.object_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scalar_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(matcap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: face_colour_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: warp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(metallic_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(emissive_view),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: position_override_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: normal_override_buf.as_entire_binding(),
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
        device: &wgpu::Device,
        mesh_id: crate::resources::mesh_store::MeshId,
        item_uniform_buf: &wgpu::Buffer,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
        lut_id: Option<ColourmapId>,
        active_attr: Option<&str>,
        matcap_id: Option<crate::resources::MatcapId>,
        warp_attr: Option<&str>,
        metallic_roughness_id: Option<u64>,
        emissive_texture_id: Option<u64>,
    ) -> Option<(wgpu::BindGroup, u64)> {
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
            mesh_id.index().hash(&mut h);
            albedo_id.unwrap_or(u64::MAX).hash(&mut h);
            normal_map_id.unwrap_or(u64::MAX).hash(&mut h);
            ao_map_id.unwrap_or(u64::MAX).hash(&mut h);
            lut_id.map(|id| id.0 as u64).unwrap_or(u64::MAX).hash(&mut h);
            attr_hash.hash(&mut h);
            matcap_id
                .map(|id| id.index as u64)
                .unwrap_or(u64::MAX)
                .hash(&mut h);
            warp_hash.hash(&mut h);
            metallic_roughness_id.unwrap_or(u64::MAX).hash(&mut h);
            emissive_texture_id.unwrap_or(u64::MAX).hash(&mut h);
            pos_override_gen.hash(&mut h);
            nrm_override_gen.hash(&mut h);
            h.finish()
        };

        let albedo_view = match albedo_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_texture.view,
        };
        let normal_view = match normal_map_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_normal_map_view,
        };
        let ao_view = match ao_map_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_ao_map_view,
        };
        let lut_view = match lut_id {
            Some(id) if id.0 < self.colourmap_views.len() => &self.colourmap_views[id.0],
            _ => &self.fallback_lut_view,
        };

        let scalar_buf: &wgpu::Buffer = match active_attr {
            Some(name) => {
                let found_vertex = mesh.attribute_buffers.get(name);
                let found_face = mesh.face_attribute_buffers.get(name);
                found_vertex
                    .or(found_face)
                    .unwrap_or(&self.fallback_scalar_buf)
            }
            None => &self.fallback_scalar_buf,
        };

        let face_colour_buf: &wgpu::Buffer = match active_attr {
            Some(name) => mesh
                .face_colour_buffers
                .get(name)
                .unwrap_or(&self.fallback_face_colour_buf),
            None => &self.fallback_face_colour_buf,
        };

        let matcap_view: &wgpu::TextureView = match matcap_id {
            Some(id) if id.index < self.matcap_views.len() => &self.matcap_views[id.index],
            _ => self
                .fallback_matcap_view
                .as_ref()
                .unwrap_or(&self.fallback_texture.view),
        };

        let warp_buf: &wgpu::Buffer = match warp_attr {
            Some(name) => mesh
                .vector_attribute_buffers
                .get(name)
                .unwrap_or(&self.fallback_warp_buf),
            None => &self.fallback_warp_buf,
        };

        let position_override_buf: &wgpu::Buffer = mesh
            .position_override_buffer
            .as_ref()
            .unwrap_or(&self.fallback_position_override_buf);
        let normal_override_buf: &wgpu::Buffer = mesh
            .normal_override_buffer
            .as_ref()
            .unwrap_or(&self.fallback_normal_override_buf);

        let metallic_roughness_view: &wgpu::TextureView = match metallic_roughness_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_metallic_roughness_texture_view,
        };
        let emissive_view: &wgpu::TextureView = match emissive_texture_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_emissive_texture_view,
        };

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("per_item_object_bind_group"),
            layout: &self.object_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: item_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scalar_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(matcap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: face_colour_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: warp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(metallic_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(emissive_view),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: position_override_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: normal_override_buf.as_entire_binding(),
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba_data: &[[u8; 4]; 256],
    ) -> ColourmapId {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lut_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let flat: Vec<u8> = rgba_data.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &flat,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = ColourmapId(self.colourmap_textures.len());
        self.colourmap_textures.push(texture);
        self.colourmap_views.push(view);
        self.colourmaps_cpu.push(*rgba_data);
        id
    }

    /// Return the CPU-side colourmap LUT for `id` as 256 RGBA8 entries, or `None` if the id is invalid.
    ///
    /// Useful for any non-GPU colourmap output: PDF export, table cell colouring, custom legend
    /// widgets, or sampling a colour at a specific scalar value. The data is always in memory
    /// (kept for GPU upload) so this accessor is free.
    pub fn get_colourmap_rgba(&self, id: ColourmapId) -> Option<&[[u8; 4]; 256]> {
        self.colourmaps_cpu.get(id.0)
    }

    /// Return the `ColourmapId` for a built-in preset.
    ///
    /// Call [`Self::ensure_colourmaps_initialized`] first (done automatically by
    /// `ViewportRenderer::prepare`).  Panics if colourmaps have not been initialized yet.
    pub fn builtin_colourmap_id(&self, preset: BuiltinColourmap) -> ColourmapId {
        self.builtin_colourmap_ids
            .expect("call ensure_colourmaps_initialized before using built-in colourmaps")
            [preset as usize]
    }

    /// Ensure built-in colourmaps are uploaded to the GPU.
    ///
    /// Called automatically by `ViewportRenderer::prepare()` on the first frame.
    /// Safe to call multiple times : no-op after first invocation.
    pub fn ensure_colourmaps_initialized(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.colourmaps_initialized {
            return;
        }
        let viridis = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::viridis_rgba(),
        );
        let plasma = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::plasma_rgba(),
        );
        let greyscale = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::greyscale_rgba(),
        );
        let coolwarm = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::coolwarm_rgba(),
        );
        let rainbow = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::rainbow_rgba(),
        );
        let magma = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::magma_rgba(),
        );
        let inferno = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::inferno_rgba(),
        );
        let turbo = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::turbo_rgba(),
        );
        let jet =
            self.upload_colourmap(device, queue, &crate::resources::colourmap_data::jet_rgba());
        let rdbu = self.upload_colourmap(
            device,
            queue,
            &crate::resources::colourmap_data::rdbu_r_rgba(),
        );
        self.builtin_colourmap_ids = Some([
            viridis, plasma, greyscale, coolwarm, rainbow, magma, inferno, turbo, jet, rdbu,
        ]);
        self.colourmaps_initialized = true;
    }

    // -----------------------------------------------------------------------
    // Matcap texture API
    // -----------------------------------------------------------------------

    /// Upload a 256×256 RGBA matcap texture and return its `MatcapId`.
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("matcap_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Ensure the shared clamp sampler is created.
        if self.matcap_sampler.is_none() {
            self.matcap_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("matcap_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let index = self.matcap_textures.len();
        self.matcap_textures.push(texture);
        self.matcap_views.push(view);

        // Lazily initialise the fallback matcap view to binding 7 of the
        // first uploaded texture (a plain white 1×1 is fine as fallback).
        if self.fallback_matcap_view.is_none() {
            self.fallback_matcap_view = Some(
                self.fallback_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
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
        self.builtin_matcap_ids
            .expect("call ensure_matcaps_initialized (or run one prepare frame) before using built-in matcaps")
            [preset as usize]
    }

    /// Upload the eight built-in matcaps to the GPU if not already done.
    ///
    /// Called automatically by `ViewportRenderer::prepare()`. Safe to call
    /// multiple times : no-op after first invocation.
    pub fn ensure_matcaps_initialized(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.matcaps_initialized {
            return;
        }
        use crate::resources::matcap_data;
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
        self.builtin_matcap_ids = Some([clay, wax, candy, flat, ceramic, jade, mud, normal]);
        self.matcaps_initialized = true;
    }
}

#[cfg(test)]
mod async_texture_tests {
    use crate::ViewportGpuResources;
    use crate::resources::UploadStatus;

    fn try_make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn drive_until_ready(
        resources: &mut ViewportGpuResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

        // 2x2 image requires 16 bytes. Pass 12 and confirm the error fires
        // before any job is submitted.
        let rgba = vec![0u8; 12];
        let err = resources
            .begin_upload_texture(&device, &queue, 2, 2, rgba)
            .expect_err("invalid size should error");
        assert!(matches!(
            err,
            crate::error::ViewportError::InvalidTextureData { expected: 16, actual: 12 }
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
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

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
        assert_eq!(tex_id, 0);

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
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

        let rgba = vec![64u8; 8 * 8 * 4];
        let id = resources
            .begin_upload_normal_map(&device, &queue, 8, 8, rgba)
            .unwrap();
        drive_until_ready(&mut resources, &device, &queue, id);
        let tex_id = resources.upload_result_texture(id).expect("ready result");
        assert_eq!(tex_id, 0);
    }

    #[test]
    fn sync_upload_still_works() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

        let rgba = vec![200u8; 4 * 4 * 4];
        let tex_id = resources
            .upload_texture(&device, &queue, 4, 4, &rgba)
            .unwrap();
        assert_eq!(tex_id, 0);
    }
}
