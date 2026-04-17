use super::*;

impl ViewportGpuResources {
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
        let id = VolumeId(self.volume_textures.len());
        self.volume_textures.push((texture, view));
        id
    }

    /// Create the volume render pipeline and bind group layout (lazy init).
    pub(crate) fn ensure_volume_pipeline(&mut self, device: &wgpu::Device) {
        if self.volume_pipeline.is_some() {
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

        let shader_src = include_str!("../shaders/volume.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volume_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volume_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
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
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: self.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        self.volume_pipeline = Some(pipeline);
        self.volume_bgl = Some(bgl);
    }

    /// Ensure the unit cube vertex + index buffers for volume bounding box proxy exist.
    fn ensure_volume_cube(&mut self, device: &wgpu::Device) {
        if self.volume_cube_vb.is_some() {
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

        self.volume_cube_vb = Some(vbuf);
        self.volume_cube_ib = Some(ibuf);
    }

    /// Ensure the default linear ramp opacity LUT exists.
    fn ensure_default_opacity_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.volume_default_opacity_lut.is_some() {
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
        self.volume_default_opacity_lut = Some(texture);
        self.volume_default_opacity_lut_view = Some(view);
    }

    /// Prepare per-frame GPU data for a single volume item.
    pub(crate) fn upload_volume_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::VolumeItem,
        clip_planes: &[crate::renderer::ClipPlane],
    ) -> VolumeGpuData {
        self.ensure_volume_cube(device);
        self.ensure_default_opacity_lut(device, queue);

        let vol_id = item.volume_id.0;
        assert!(
            vol_id < self.volume_textures.len(),
            "invalid VolumeId: {} (only {} volumes uploaded)",
            vol_id,
            self.volume_textures.len()
        );

        let dims = {
            let tex = &self.volume_textures[vol_id].0;
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
        let step_size = item.step_scale / max_dim.max(1.0);

        let mut clip_plane_data = [[0.0f32; 4]; 6];
        let mut num_clip = 0u32;
        for cp in clip_planes.iter().filter(|c| c.enabled).take(6) {
            clip_plane_data[num_clip as usize] =
                [cp.normal[0], cp.normal[1], cp.normal[2], cp.distance];
            num_clip += 1;
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
            let shading_u32: u32 = if item.enable_shading { 1 } else { 0 };
            uniform_data[offset..offset + 4].copy_from_slice(bytemuck::bytes_of(&shading_u32));
            offset += 4;
            uniform_data[offset..offset + 4].copy_from_slice(bytemuck::bytes_of(&num_clip));
            offset += 4;
            let use_nan_color_u32: u32 = if item.nan_color.is_some() { 1 } else { 0 };
            uniform_data[offset..offset + 4]
                .copy_from_slice(bytemuck::bytes_of(&use_nan_color_u32));
            offset += 4;
            offset += 4;
            let nan_color = item.nan_color.unwrap_or([0.0f32; 4]);
            uniform_data[offset..offset + 16].copy_from_slice(bytemuck::bytes_of(&nan_color));
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

        let volume_view = &self.volume_textures[vol_id].1;

        let color_lut_view = if let Some(cmap_id) = item.color_lut {
            self.colormap_views
                .get(cmap_id.0)
                .unwrap_or(&self.fallback_lut_view)
        } else if let Some(ids) = &self.builtin_colormap_ids {
            self.colormap_views
                .get(ids[0].0)
                .unwrap_or(&self.fallback_lut_view)
        } else {
            &self.fallback_lut_view
        };

        let opacity_lut_view = if let Some(cmap_id) = item.opacity_lut {
            self.colormap_views
                .get(cmap_id.0)
                .unwrap_or(self.volume_default_opacity_lut_view.as_ref().unwrap())
        } else {
            self.volume_default_opacity_lut_view.as_ref().unwrap()
        };

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

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
            .volume_bgl
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
                    resource: wgpu::BindingResource::TextureView(color_lut_view),
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
        }
    }
}
