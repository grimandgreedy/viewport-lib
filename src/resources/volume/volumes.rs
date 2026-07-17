use crate::resources::*;

/// Direct volume rendering pipelines, layouts, the cached unit cube geometry,
/// and the default opacity LUT. All lazily built; the uploaded 3D volume
/// textures live in a separate flat store.
#[derive(Default)]
pub(crate) struct VolumeResources {
    /// Volume render pipeline. None until first volume is submitted.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Bind group layout for volume uniforms (group 1).
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    /// Cached unit cube vertex buffer for bounding-box rasterization.
    pub(crate) cube_vb: Option<crate::gpu::Buffer>,
    /// Cached unit cube index buffer.
    pub(crate) cube_ib: Option<crate::gpu::Buffer>,
    /// Default linear ramp opacity LUT texture (256x1, R8Unorm).
    pub(crate) default_opacity_lut: Option<crate::gpu::Texture>,
    pub(crate) default_opacity_lut_view: Option<crate::gpu::TextureView>,
    /// Volume surface slice render pipeline. None until first slice item.
    pub(crate) surface_slice_pipeline: Option<DualPipeline>,
    /// Bind group layout for volume surface slice uniforms (group 1).
    pub(crate) surface_slice_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Mask-write pipeline for volume AABB cubes. None until first selected volume.
    pub(crate) outline_mask_pipeline: Option<crate::gpu::RenderPipeline>,
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
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &[f32],
        dims: [u32; 3],
    ) -> VolumeId {
        // Build the ray-march pipeline now so a load-time upload also pays the
        // pipeline compile, not the first frame that draws the volume.
        self.ensure_volume_pipeline(device);
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
        self.ensure_volume_pipeline(device);
        let (texture, view, volume_bytes) = Self::build_volume_texture(device, queue, data, dims);
        let replaced = self
            .content
            .volume_textures
            .replace(id, (texture, view), volume_bytes)
            .is_some();
        if replaced {
            // Drop any scatter bind group built against this slot's old texture
            // so the previous field's GPU memory is actually released.
            self.invalidate_scatter_density(id.index() as u32);
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
        if freed {
            self.invalidate_scatter_density(id.index() as u32);
        }
        freed
    }

    /// Create an `R32Float` 3D texture from `data`, upload it, and return the
    /// texture, its default view, and the GPU bytes it occupies (4 per texel).
    ///
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
            format: crate::gpu::TextureFormat::R32Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let bytes: &[u8] = bytemuck::cast_slice(data);
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            bytes,
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dims[0] * 4),
                rows_per_image: Some(dims[1]),
            },
            crate::gpu::Extent3d {
                width: dims[0],
                height: dims[1],
                depth_or_array_layers: dims[2],
            },
        );

        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        (texture, view, (expected as u64) * 4)
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
                    format: crate::gpu::TextureFormat::R32Float,
                    usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                        | crate::gpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let bytes: &[u8] = bytemuck::cast_slice(&data);
                queue_for_worker.write_texture(
                    crate::gpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: crate::gpu::Origin3d::ZERO,
                        aspect: crate::gpu::TextureAspect::All,
                    },
                    bytes,
                    crate::gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(dims[0] * 4),
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
                        let volume_bytes = (expected as u64) * 4;
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

    /// Create the volume render pipeline and bind group layout (lazy init).
    pub(crate) fn ensure_volume_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.volume.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
                        view_dimension: crate::gpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(
                        crate::gpu::SamplerBindingType::NonFiltering,
                    ),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
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

        let vol_vert_layout = crate::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &[crate::gpu::VertexAttribute {
                format: crate::gpu::VertexFormat::Float32x3,
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
                blend: Some(crate::gpu::BlendState {
                    color: crate::gpu::BlendComponent {
                        src_factor: crate::gpu::BlendFactor::SrcAlpha,
                        dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                        operation: crate::gpu::BlendOperation::Add,
                    },
                    alpha: crate::gpu::BlendComponent {
                        src_factor: crate::gpu::BlendFactor::One,
                        dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                        operation: crate::gpu::BlendOperation::Add,
                    },
                }),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: crate::gpu::CompareFunction::Less,
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
    pub(crate) fn ensure_volume_outline_mask_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.volume.outline_mask_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
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

        let vert_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }];
        let vert_layout = crate::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &vert_attrs,
        };

        self.volume.outline_mask_pipeline =
            Some(crate::resources::builders::build_outline_mask_pipeline(
                device,
                "volume_outline_mask_pipeline",
                &layout,
                &shader,
                crate::gpu::TextureFormat::R8Unorm,
                &[vert_layout],
                None,
                true,
                crate::gpu::CompareFunction::Less,
            ));
    }

    /// Ensure the unit cube vertex + index buffers for volume bounding box proxy exist.
    pub(crate) fn ensure_volume_cube(&mut self, device: &crate::gpu::Device) {
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

        let vbuf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_cube_vb"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(vbuf.slice(..), bytemuck::cast_slice(&vertices));
        vbuf.unmap();

        let ibuf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_cube_ib"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(ibuf.slice(..), bytemuck::cast_slice(&indices));
        ibuf.unmap();

        self.volume.cube_vb = Some(vbuf);
        self.volume.cube_ib = Some(ibuf);
    }

    /// Ensure the default linear ramp opacity LUT exists.
    fn ensure_default_opacity_lut(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        if self.volume.default_opacity_lut.is_some() {
            return;
        }

        let mut data = [0u8; 256];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as u8;
        }

        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("volume_default_opacity_lut"),
            size: crate::gpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R8Unorm,
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
            &data,
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
            crate::gpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        self.volume.default_opacity_lut = Some(texture);
        self.volume.default_opacity_lut_view = Some(view);
    }

    /// Prepare per-frame GPU data for a single volume item.
    pub(crate) fn upload_volume_frame(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::VolumeItem,
        clip_objects: &[crate::renderer::ClipObject],
        // Multiplier applied to the computed step size (1.0 = normal, >1.0 = coarser/faster).
        step_scale_multiplier: f32,
    ) -> VolumeGpuData {
        self.ensure_volume_cube(device);
        self.ensure_default_opacity_lut(device, queue);

        let vol_id = item.volume_id;
        let dims = {
            let uploaded = self.content.volume_textures.len();
            let (tex, _) = self.content.volume_textures.get(vol_id).unwrap_or_else(|| {
                panic!("invalid VolumeId: {vol_id:?} (only {uploaded} volumes live)")
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

        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_uniform_buf"),
            size: 304,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(uniform_buf.slice(..), &uniform_data);
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

        let linear_sampler =
            crate::resources::builders::clamp_linear_mip_sampler(device, "volume_lut_sampler");

        let bgl = self
            .volume
            .bgl
            .as_ref()
            .expect("ensure_volume_pipeline not called");

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("volume_bind_group"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(volume_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&nearest_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(colour_lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(opacity_lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::Sampler(&linear_sampler),
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

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_cube_vb_frame"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            vertex_buffer.slice(..),
            bytemuck::cast_slice(&vertices),
        );
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_cube_ib_frame"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            index_buffer.slice(..),
            bytemuck::cast_slice(&indices),
        );
        index_buffer.unmap();

        VolumeGpuData {
            bind_group,
            vertex_buffer,
            index_buffer,
            _dims: dims,
            _uniform_buf: uniform_buf,
            wireframe: false,
            pick_id: item.settings.pick_id,
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

        assert_eq!(resources.resident_bytes().volume_bytes, 0);
        let _id = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        // 8*8*8 R32Float = 512 texels * 4 bytes.
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * 4);
        // A second distinct upload takes a second slot, so the charge adds. This
        // is the per-timestep growth a time-series avoids by calling
        // replace_volume on one handle instead (see replace_volume_reuses_slot).
        let _id2 = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(resources.resident_bytes().volume_bytes, 2 * 8 * 8 * 8 * 4);
    }

    #[test]
    fn replace_volume_reuses_slot() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let id = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * 4);

        // Replacing the field in place keeps the same handle and one slot, so the
        // charge stays flat rather than doubling (the whole point of S1).
        for _ in 0..10 {
            assert!(resources.replace_volume(&device, &queue, id, &sample_volume_data(), [8, 8, 8]));
        }
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * 4);

        // A larger field updates the charge to the new size, still one slot.
        let big = vec![0.5_f32; 16 * 16 * 16];
        assert!(resources.replace_volume(&device, &queue, id, &big, [16, 16, 16]));
        assert_eq!(resources.resident_bytes().volume_bytes, 16 * 16 * 16 * 4);
    }

    #[test]
    fn free_volume_reclaims_bytes_and_slot() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let id = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * 4);

        assert!(resources.free_volume(id));
        assert_eq!(resources.resident_bytes().volume_bytes, 0);
        // Second free of the same handle is a no-op.
        assert!(!resources.free_volume(id));

        // The freed slot is reused by the next upload, so the charge is one
        // field's worth, not two.
        let _id2 = resources.upload_volume(&device, &queue, &sample_volume_data(), [8, 8, 8]);
        assert_eq!(resources.resident_bytes().volume_bytes, 8 * 8 * 8 * 4);
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

/// Per-frame GPU data for one volume item, created in `prepare()`.
pub struct VolumeGpuData {
    /// Bind group (group 1): volume uniform + 3D texture + sampler + colour LUT + opacity LUT.
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Vertex buffer for the unit cube bounding box proxy.
    pub(crate) vertex_buffer: crate::gpu::Buffer,
    /// Index buffer for the unit cube (36 indices).
    pub(crate) index_buffer: crate::gpu::Buffer,
    /// Grid dimensions (stored for reference).
    pub(crate) _dims: [u32; 3],
    // Keep the uniform buffer alive.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    /// When true, skip the volume ray-march draw; an OBB wireframe polyline is rendered instead.
    pub(crate) wireframe: bool,
    /// Item pick id, used by the GPU pick pass to raymarch this volume's cube and
    /// tag the first in-threshold voxel. `NONE` when the item is not pickable.
    pub(crate) pick_id: crate::renderer::PickId,
}

/// Per-frame GPU data for one image slice item, created in `prepare()`.
pub(crate) struct ImageSliceGpuData {
    /// Bind group (group 1): uniform + 3D texture + sampler + LUT + LUT sampler.
    pub(crate) bind_group: crate::gpu::BindGroup,
    // Keep buffers/samplers alive.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    /// The item's pick id (from `settings.pick_id`); `PickId::NONE` when not pickable.
    pub(crate) pick_id: crate::renderer::PickId,
}

/// Per-frame GPU data for one volume surface slice item, created in `prepare()`.
pub(crate) struct VolumeSurfaceSliceGpuData {
    /// Bind group (group 1): uniform + 3D texture + sampler + LUT + LUT sampler.
    pub(crate) bind_group: crate::gpu::BindGroup,
    // Keep uniform buffer alive.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    /// Mesh to draw (vertex + index buffers looked up from mesh_store at render time).
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// The item's pick id (from `settings.pick_id`); `PickId::NONE` when not pickable.
    pub(crate) pick_id: crate::renderer::PickId,
}
