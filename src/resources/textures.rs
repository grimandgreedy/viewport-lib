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
        let expected = (width * height * 4) as usize;
        if rgba_data.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba_data.len(),
            });
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("user_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
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

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("user_texture_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_texture_bg"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_normal_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_ao_map_view),
                },
            ],
        });

        let id = self.textures.len() as u64;
        self.textures.push(GpuTexture {
            texture,
            view,
            sampler,
            bind_group,
        });
        tracing::debug!(texture_id = id, width, height, "texture uploaded");
        Ok(id)
    }

    /// Upload an RGBA texture as a normal map and return its texture ID.
    ///
    /// Uses Rgba8Unorm format (not sRGB) so values are linear — required for correct
    /// normal map decoding. `rgba_data` must be `width * height * 4` bytes.
    pub fn upload_normal_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> crate::error::ViewportResult<u64> {
        let expected = (width * height * 4) as usize;
        if rgba_data.len() != expected {
            return Err(crate::error::ViewportError::InvalidTextureData {
                expected,
                actual: rgba_data.len(),
            });
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normal_map_texture"),
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

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("normal_map_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal_map_bg"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_ao_map_view),
                },
            ],
        });

        let id = self.textures.len() as u64;
        self.textures.push(GpuTexture {
            texture,
            view,
            sampler,
            bind_group,
        });
        tracing::debug!(texture_id = id, width, height, "normal map uploaded");
        Ok(id)
    }

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
    ///   binding 0 → object uniform buffer
    ///   binding 1 → albedo texture view
    ///   binding 2 → material sampler (also used for LUT sampling)
    ///   binding 3 → normal map view
    ///   binding 4 → AO map view
    ///   binding 5 → LUT (colormap) texture view
    ///   binding 6 → scalar attribute storage buffer
    pub(crate) fn update_mesh_texture_bind_group(
        &mut self,
        device: &wgpu::Device,
        mesh_index: usize,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
        lut_id: Option<ColormapId>,
        active_attr: Option<&str>,
    ) {
        let attr_hash = active_attr
            .map(|name| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                name.hash(&mut h);
                h.finish()
            })
            .unwrap_or(u64::MAX);

        let key = (
            albedo_id.unwrap_or(u64::MAX),
            normal_map_id.unwrap_or(u64::MAX),
            ao_map_id.unwrap_or(u64::MAX),
            lut_id.map(|id| id.0 as u64).unwrap_or(u64::MAX),
            attr_hash,
        );

        {
            let Some(mesh) = self
                .mesh_store
                .get(crate::resources::mesh_store::MeshId(mesh_index))
            else {
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
            Some(id) if id.0 < self.colormap_views.len() => &self.colormap_views[id.0],
            _ => &self.fallback_lut_view,
        };

        let Some(mesh) = self
            .mesh_store
            .get_mut(crate::resources::mesh_store::MeshId(mesh_index))
        else {
            return;
        };

        let scalar_buf: &wgpu::Buffer = match active_attr {
            Some(name) => mesh
                .attribute_buffers
                .get(name)
                .unwrap_or(&self.fallback_scalar_buf),
            None => &self.fallback_scalar_buf,
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
            ],
        });
        mesh.last_tex_key = key;
    }

    /// Upload a 256-sample RGBA colormap to the GPU and return its `ColormapId`.
    ///
    /// The returned ID can be stored in `SceneRenderItem::colormap_id`.
    /// Use `BuiltinColormap` variants + [`Self::builtin_colormap_id`] for the built-in presets.
    pub fn upload_colormap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba_data: &[[u8; 4]; 256],
    ) -> ColormapId {
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
        let id = ColormapId(self.colormap_textures.len());
        self.colormap_textures.push(texture);
        self.colormap_views.push(view);
        self.colormaps_cpu.push(*rgba_data);
        id
    }

    /// Return the CPU-side colormap data for `id`, or `None` if the id is invalid.
    ///
    /// Use this to draw an egui scalar bar gradient strip via `egui::Painter::image`.
    pub fn get_colormap_rgba(&self, id: ColormapId) -> Option<&[[u8; 4]; 256]> {
        self.colormaps_cpu.get(id.0)
    }

    /// Return the `ColormapId` for a built-in preset.
    ///
    /// Call [`Self::ensure_colormaps_initialized`] first (done automatically by
    /// `ViewportRenderer::prepare`).  Panics if colormaps have not been initialized yet.
    pub fn builtin_colormap_id(&self, preset: BuiltinColormap) -> ColormapId {
        self.builtin_colormap_ids
            .expect("call ensure_colormaps_initialized before using built-in colormaps")
            [preset as usize]
    }

    /// Ensure built-in colormaps are uploaded to the GPU.
    ///
    /// Called automatically by `ViewportRenderer::prepare()` on the first frame.
    /// Safe to call multiple times — no-op after first invocation.
    pub fn ensure_colormaps_initialized(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.colormaps_initialized {
            return;
        }
        let viridis = self.upload_colormap(
            device,
            queue,
            &crate::resources::colormap_data::viridis_rgba(),
        );
        let plasma = self.upload_colormap(
            device,
            queue,
            &crate::resources::colormap_data::plasma_rgba(),
        );
        let greyscale = self.upload_colormap(
            device,
            queue,
            &crate::resources::colormap_data::greyscale_rgba(),
        );
        let coolwarm = self.upload_colormap(
            device,
            queue,
            &crate::resources::colormap_data::coolwarm_rgba(),
        );
        let rainbow = self.upload_colormap(
            device,
            queue,
            &crate::resources::colormap_data::rainbow_rgba(),
        );
        self.builtin_colormap_ids = Some([viridis, plasma, greyscale, coolwarm, rainbow]);
        self.colormaps_initialized = true;
    }
}
