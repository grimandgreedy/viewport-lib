use crate::resources::*;

/// Direct volume rendering pipelines, layouts, the cached unit cube geometry,
/// and the default opacity LUT. All lazily built; the uploaded 3D volume
/// textures live in a separate flat store.
#[derive(Default)]
pub(crate) struct VolumeResources {
    /// Volume render pipeline. None until first volume is submitted.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Bind group layout for volume uniforms (group 1).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Cached unit cube vertex buffer for bounding-box rasterization.
    pub(crate) cube_vb: Option<wgpu::Buffer>,
    /// Cached unit cube index buffer.
    pub(crate) cube_ib: Option<wgpu::Buffer>,
    /// Default linear ramp opacity LUT texture (256x1, R8Unorm).
    pub(crate) default_opacity_lut: Option<wgpu::Texture>,
    pub(crate) default_opacity_lut_view: Option<wgpu::TextureView>,
    /// Volume surface slice render pipeline. None until first slice item.
    pub(crate) surface_slice_pipeline: Option<DualPipeline>,
    /// Bind group layout for volume surface slice uniforms (group 1).
    pub(crate) surface_slice_bgl: Option<wgpu::BindGroupLayout>,
    /// Mask-write pipeline for volume AABB cubes. None until first selected volume.
    pub(crate) outline_mask_pipeline: Option<wgpu::RenderPipeline>,
}

impl DeviceResources {
    /// Upload a 3D scalar field to the GPU as an `R32Float` 3D texture.
    ///
    /// `data` must be a flat array of `dims[0] * dims[1] * dims[2]` scalars in
    /// x-fastest order (index = x + y*nx + z*nx*ny).
    ///
    /// Returns a [`VolumeId`](crate::resources::VolumeId) that can be stored in [`VolumeItem::volume_id`](crate::renderer::VolumeItem::volume_id).
    pub fn upload_volume(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[f32],
        dims: [u32; 3],
    ) -> VolumeId {
        let expected = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        assert_eq!(
            data.len(),
            expected,
            "volume data length {} does not match dims {:?} (expected {})",
            data.len(),
            dims,
            expected
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume_3d_texture"),
            size: wgpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let bytes: &[u8] = bytemuck::cast_slice(data);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims[0] * 4),
                rows_per_image: Some(dims[1]),
            },
            wgpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        VolumeId(self.content.volume_textures.push((texture, view)))
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
                let texture = device_for_worker.create_texture(&wgpu::TextureDescriptor {
                    label: Some("volume_3d_texture"),
                    size: wgpu::Extent3d {
                        width: dims[0],
                        height: dims[1],
                        depth_or_array_layers: dims[2],
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let bytes: &[u8] = bytemuck::cast_slice(&data);
                queue_for_worker.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(dims[0] * 4),
                        rows_per_image: Some(dims[1]),
                    },
                    wgpu::Extent3d {
                        width: dims[0],
                        height: dims[1],
                        depth_or_array_layers: dims[2],
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let id = VolumeId(resources.content.volume_textures.push((texture, view)));
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

    /// Create the volume render pipeline and bind group layout (lazy init).
    pub(crate) fn ensure_volume_pipeline(&mut self, device: &wgpu::Device) {
        if self.volume.pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader_src = include_str!(concat!(env!("OUT_DIR"), "/volume.wgsl"));
        let shader = crate::resources::builders::wgsl_module(device, "volume_shader", shader_src);

        let pipeline_layout = crate::resources::builders::standard_scene_layout(
            device,
            "volume_pipeline_layout",
            &self.camera_bind_group_layout,
            &bgl,
        );

        let vol_vert_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        };
        self.volume.pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "volume_pipeline",
                layout: &pipeline_layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[vol_vert_layout.clone()],
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: wgpu::CompareFunction::Less,
                sample_count: self.sample_count,
                ldr_format: self.target_format,
            },
        ));
        self.volume.bgl = Some(bgl);
    }

    /// Ensure the volume outline mask pipeline exists. This pipeline ray-marches the
    /// volume in the R8 mask texture so the outline hugs the actual volume silhouette
    /// rather than the AABB. Requires `ensure_volume_pipeline` to have been called
    /// first (needs `volume_bgl`).
    pub(crate) fn ensure_volume_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.volume.outline_mask_pipeline.is_some() {
            return;
        }
        let bgl = self.volume.bgl.as_ref().expect(
            "ensure_volume_pipeline must be called before ensure_volume_outline_mask_pipeline",
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "volume_outline_mask_shader",
            crate::resources::builders::wgsl_source!("volume_outline_mask"),
        );

        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "volume_outline_mask_pipeline_layout",
            &self.camera_bind_group_layout,
            bgl,
        );

        let vert_attrs = [wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        }];
        let vert_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vert_attrs,
        };

        self.volume.outline_mask_pipeline =
            Some(crate::resources::builders::build_outline_mask_pipeline(
                device,
                "volume_outline_mask_pipeline",
                &layout,
                &shader,
                wgpu::TextureFormat::R8Unorm,
                &[vert_layout],
                None,
                true,
                wgpu::CompareFunction::Less,
            ));
    }

    /// Ensure the unit cube vertex + index buffers for volume bounding box proxy exist.
    pub(crate) fn ensure_volume_cube(&mut self, device: &wgpu::Device) {
        if self.volume.cube_vb.is_some() {
            return;
        }

        #[rustfmt::skip]
        let vertices: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];

        #[rustfmt::skip]
        let indices: [u32; 36] = [
            0, 2, 1,  0, 3, 2,
            4, 5, 6,  4, 6, 7,
            0, 4, 7,  0, 7, 3,
            1, 2, 6,  1, 6, 5,
            0, 1, 5,  0, 5, 4,
            3, 7, 6,  3, 6, 2,
        ];

        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_cube_vb"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = vbuf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&vertices));
        }
        vbuf.unmap();

        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_cube_ib"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = ibuf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&indices));
        }
        ibuf.unmap();

        self.volume.cube_vb = Some(vbuf);
        self.volume.cube_ib = Some(ibuf);
    }

    /// Ensure the default linear ramp opacity LUT exists.
    fn ensure_default_opacity_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.volume.default_opacity_lut.is_some() {
            return;
        }

        let mut data = [0u8; 256];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as u8;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume_default_opacity_lut"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
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
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.volume.default_opacity_lut = Some(texture);
        self.volume.default_opacity_lut_view = Some(view);
    }

    /// Prepare per-frame GPU data for a single volume item.
    pub(crate) fn upload_volume_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::VolumeItem,
        clip_objects: &[crate::renderer::ClipObject],
        // Multiplier applied to the computed step size (1.0 = normal, >1.0 = coarser/faster).
        step_scale_multiplier: f32,
    ) -> VolumeGpuData {
        self.ensure_volume_cube(device);
        self.ensure_default_opacity_lut(device, queue);

        let vol_id = item.volume_id.0;
        let dims = {
            let uploaded = self.content.volume_textures.len();
            let (tex, _) = self.content.volume_textures.get(vol_id).unwrap_or_else(|| {
                panic!("invalid VolumeId: {vol_id} (only {uploaded} volumes uploaded)")
            });
            let size = tex.size();
            [size.width, size.height, size.depth_or_array_layers]
        };

        let item_model = glam::Mat4::from_cols_array_2d(&item.model);
        let bbox_min = glam::Vec3::from(item.bbox_min);
        let bbox_max = glam::Vec3::from(item.bbox_max);
        let extent = bbox_max - bbox_min;
        let bbox_model = glam::Mat4::from_translation(bbox_min) * glam::Mat4::from_scale(extent);
        let model = item_model * bbox_model;
        let inv_model = model.inverse();

        let max_dim = dims[0].max(dims[1]).max(dims[2]) as f32;
        let step_size = (item.step_scale * step_scale_multiplier) / max_dim.max(1.0);

        let mut clip_plane_data = [[0.0f32; 4]; 6];
        let mut num_clip = 0u32;
        for obj in clip_objects.iter().filter(|o| o.enabled) {
            if num_clip >= 6 {
                break;
            }
            if let crate::renderer::ClipShape::Plane {
                normal, distance, ..
            } = obj.shape
            {
                clip_plane_data[num_clip as usize] = [normal[0], normal[1], normal[2], distance];
                num_clip += 1;
            }
        }

        let mut uniform_data = [0u8; 304];
        {
            let mut offset = 0usize;
            let model_arr = model.to_cols_array();
            uniform_data[offset..offset + 64].copy_from_slice(bytemuck::bytes_of(&model_arr));
            offset += 64;
            let inv_model_arr = inv_model.to_cols_array();
            uniform_data[offset..offset + 64].copy_from_slice(bytemuck::bytes_of(&inv_model_arr));
            offset += 64;
            uniform_data[offset..offset + 12].copy_from_slice(bytemuck::bytes_of(&item.bbox_min));
            offset += 12;
            uniform_data[offset..offset + 4].copy_from_slice(bytemuck::bytes_of(&step_size));
            offset += 4;
            uniform_data[offset..offset + 12].copy_from_slice(bytemuck::bytes_of(&item.bbox_max));
            offset += 12;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.opacity_scale));
            offset += 4;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.scalar_range.0));
            offset += 4;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.scalar_range.1));
            offset += 4;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.threshold_min));
            offset += 4;
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&item.threshold_max));
            offset += 4;
            // ItemSettings.unlit forces gradient shading off regardless of the
            // per-item enable_shading toggle. The volume ray-marcher's only
            // lighting path is gradient Phong, gated by VolumeUniform.enable_shading;
            // ORing unlit here gives consumers a uniform way to disable lighting
            // across all item types.
            let shading_u32: u32 = if item.enable_shading && !item.settings.unlit {
                1
            } else {
                0
            };
            uniform_data[offset..offset + 4].copy_from_slice(bytemuck::bytes_of(&shading_u32));
            offset += 4;
            uniform_data[offset..offset + 4].copy_from_slice(bytemuck::bytes_of(&num_clip));
            offset += 4;
            let use_nan_colour_u32: u32 = if item.nan_colour.is_some() { 1 } else { 0 };
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&use_nan_colour_u32));
            offset += 4;
            offset += 4;
            let nan_colour = item.nan_colour.unwrap_or([0.0f32; 4]);
            uniform_data[offset..offset + 16].copy_from_slice(bytemuck::bytes_of(&nan_colour));
            offset += 16;
            for cp in &clip_plane_data {
                uniform_data[offset..offset + 16].copy_from_slice(bytemuck::bytes_of(cp));
                offset += 16;
            }
            debug_assert_eq!(offset, 304);
        }

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_uniform_buf"),
            size: 304,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = uniform_buf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(&uniform_data);
        }
        uniform_buf.unmap();

        let volume_view = &self
            .content
            .volume_textures
            .get(vol_id)
            .expect("VolumeId validated above")
            .1;

        let colour_lut_view = if let Some(cmap_id) = item.colour_lut {
            self.content
                .colourmap_views
                .get(cmap_id.0)
                .unwrap_or(&self.content.fallback_lut_view)
        } else if let Some(ids) = &self.content.builtin_colourmap_ids {
            self.content
                .colourmap_views
                .get(ids[0].0)
                .unwrap_or(&self.content.fallback_lut_view)
        } else {
            &self.content.fallback_lut_view
        };

        let opacity_lut_view = if let Some(cmap_id) = item.opacity_lut {
            self.content
                .colourmap_views
                .get(cmap_id.0)
                .unwrap_or(self.volume.default_opacity_lut_view.as_ref().unwrap())
        } else {
            self.volume.default_opacity_lut_view.as_ref().unwrap()
        };

        let nearest_sampler =
            crate::resources::builders::clamp_nearest_sampler(device, "volume_nearest_sampler");

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume_lut_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bgl = self
            .volume
            .bgl
            .as_ref()
            .expect("ensure_volume_pipeline not called");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(colour_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(opacity_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });

        #[rustfmt::skip]
        let vertices: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        #[rustfmt::skip]
        let indices: [u32; 36] = [
            0,2,1, 0,3,2, 4,5,6, 4,6,7,
            0,4,7, 0,7,3, 1,2,6, 1,6,5,
            0,1,5, 0,5,4, 3,7,6, 3,6,2,
        ];

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_cube_vb_frame"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = vertex_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&vertices));
        }
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume_cube_ib_frame"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = index_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&indices));
        }
        index_buffer.unmap();

        VolumeGpuData {
            bind_group,
            vertex_buffer,
            index_buffer,
            _dims: dims,
            _uniform_buf: uniform_buf,
            wireframe: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::DeviceResources;
    use crate::geometry::marching_cubes::VolumeData;
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let data = sample_volume_data();
        let _id = resources.upload_volume(&device, &queue, &data, [8, 8, 8]);
    }

    #[test]
    fn begin_upload_volume_validates_dims() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
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
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
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
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
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
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let vol = sample_volume_struct();
        let _id = resources
            .upload_volume_for_mc(&device, &queue, &vol)
            .expect("upload ok");
    }
}
