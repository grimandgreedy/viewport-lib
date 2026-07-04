use super::*;

/// Arrow/sphere/cube glyph pipelines, layouts, and cached base meshes.
/// All lazily built; the uploaded glyph sets live in a separate flat store.
#[derive(Default)]
pub(crate) struct GlyphResources {
    /// Glyph render pipeline. None until first glyph set is submitted.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Glyph wireframe pipeline (LineList, same bind groups as `pipeline`).
    pub(crate) wireframe_pipeline: Option<DualPipeline>,
    /// Bind group layout for glyph uniforms (group 1).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for glyph instance storage (group 2).
    pub(crate) instance_bgl: Option<wgpu::BindGroupLayout>,
    /// Cached glyph base mesh for the Arrow shape.
    pub(crate) arrow_mesh: Option<GlyphBaseMesh>,
    /// Cached glyph base mesh for the Sphere shape.
    pub(crate) sphere_mesh: Option<GlyphBaseMesh>,
    /// Cached glyph base mesh for the Cube shape.
    pub(crate) cube_mesh: Option<GlyphBaseMesh>,
    /// Instanced mask pipeline for arrow/sphere glyph outlines.
    pub(crate) outline_mask_pipeline: Option<wgpu::RenderPipeline>,
}

/// Tensor glyph (ellipsoid / superquadric) pipelines and layouts.
#[derive(Default)]
pub(crate) struct TensorGlyphResources {
    /// Tensor glyph render pipeline. None until first tensor glyph set is submitted.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Tensor glyph wireframe pipeline (LineList, same bind groups as `pipeline`).
    pub(crate) wireframe_pipeline: Option<DualPipeline>,
    /// Bind group layout for tensor glyph uniforms (group 1).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for tensor glyph instance storage (group 2).
    pub(crate) instance_bgl: Option<wgpu::BindGroupLayout>,
    /// Instanced mask pipeline for tensor glyph outlines.
    pub(crate) outline_mask_pipeline: Option<wgpu::RenderPipeline>,
}

impl DeviceResources {
    /// Lazily create the glyph render pipeline (instanced TriangleList).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.glyphs` is non-empty.
    pub(crate) fn ensure_glyph_pipeline(&mut self, device: &wgpu::Device) {
        if self.glyph.pipeline.is_some() {
            return;
        }

        let glyph_bgl = crate::resources::builders::uniform_texture_sampler_bgl(
            device,
            "glyph_bgl",
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            wgpu::ShaderStages::VERTEX,
        );

        let glyph_instance_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("glyph_instance_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glyph_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/glyph.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glyph_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &glyph_bgl,
                &glyph_instance_bgl,
            ],
            push_constant_ranges: &[],
        });

        self.glyph.bgl = Some(glyph_bgl);
        self.glyph.instance_bgl = Some(glyph_instance_bgl);
        self.glyph.pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "glyph_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                depth_write: true,
                depth_compare: wgpu::CompareFunction::Less,
                sample_count: self.sample_count,
                ldr_format: self.target_format,
            },
        ));

        // Wireframe variant: same bind groups, LineList topology, no culling.
        self.glyph.wireframe_pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "glyph_wireframe_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
                blend: None,
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                depth_write: true,
                depth_compare: wgpu::CompareFunction::Less,
                sample_count: self.sample_count,
                ldr_format: self.target_format,
            },
        ));
    }

    /// Upload one [`GlyphItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.scene.glyphs`.
    /// The glyph base mesh is cached in `glyph_arrow_mesh` / `glyph_sphere_mesh` / `glyph_cube_mesh`.
    pub(crate) fn upload_glyph_set_per_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::GlyphItem,
        wireframe: bool,
    ) -> GlyphGpuData {
        let instance_count = item.positions.len() as u32;

        self.ensure_glyph_mesh(device, item.glyph_type);

        let (mesh_vbuf, mesh_ibuf, mesh_idx_count, mesh_edge_ibuf, mesh_edge_count) = {
            let mesh = match item.glyph_type {
                crate::renderer::GlyphType::Arrow => self.glyph.arrow_mesh.as_ref(),
                crate::renderer::GlyphType::Sphere => self.glyph.sphere_mesh.as_ref(),
                crate::renderer::GlyphType::Cube => self.glyph.cube_mesh.as_ref(),
            }
            .expect("glyph mesh should have been created by ensure_glyph_mesh");

            let vbuf: &'static wgpu::Buffer = unsafe { &*(&mesh.vertex_buffer as *const _) };
            let ibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.index_buffer as *const _) };
            let eibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.edge_index_buffer as *const _) };
            (vbuf, ibuf, mesh.index_count, eibuf, mesh.edge_index_count)
        };

        let mags: Vec<f32> = item
            .vectors
            .iter()
            .map(|v| glam::Vec3::from(*v).length())
            .collect();

        let (scalar_min, scalar_max) = if !item.scalars.is_empty() {
            item.scalar_range.unwrap_or_else(|| {
                let min = item.scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = item
                    .scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
        } else {
            item.scalar_range.unwrap_or_else(|| {
                let min = mags.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = mags.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
        };

        let (mag_clamp_min, mag_clamp_max, has_mag_clamp) = item
            .magnitude_clamp
            .map(|(mn, mx)| (mn, mx, 1u32))
            .unwrap_or((0.0, 1.0, 0u32));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GlyphInstance {
            position: [f32; 3],
            _pad0: f32,
            direction: [f32; 3],
            scalar: f32,
        }

        let instances: Vec<GlyphInstance> = (0..item.positions.len())
            .map(|i| GlyphInstance {
                position: item.positions[i],
                _pad0: 0.0,
                direction: item.vectors.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
                scalar: item
                    .scalars
                    .get(i)
                    .copied()
                    .unwrap_or(mags.get(i).copied().unwrap_or(0.0)),
            })
            .collect();

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_instance_buf"),
            size: (std::mem::size_of::<GlyphInstance>() * instances.len()).max(32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GlyphUniform {
            model: [[f32; 4]; 4],
            global_scale: f32,
            scale_by_magnitude: u32,
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            mag_clamp_min: f32,
            mag_clamp_max: f32,
            has_mag_clamp: u32,
            default_colour: [f32; 4],
            use_default_colour: u32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
        }
        let uniform_data = GlyphUniform {
            model: item.model,
            global_scale: item.scale,
            scale_by_magnitude: if item.scale_by_magnitude { 1 } else { 0 },
            has_scalars: if !item.scalars.is_empty() { 1 } else { 0 },
            scalar_min,
            scalar_max,
            mag_clamp_min,
            mag_clamp_max,
            has_mag_clamp,
            default_colour: item.default_colour,
            use_default_colour: if item.default_colour[3] > 0.0 && item.use_default_colour {
                1
            } else {
                0
            },
            unlit: if item.settings.unlit { 1 } else { 0 },
            opacity: item.settings.opacity,
            wireframe: if wireframe { 1 } else { 0 },
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_uniform_buf"),
            size: std::mem::size_of::<GlyphUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = self
            .content
            .builtin_colourmap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colourmap_id
                    .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                self.content.colourmap_views.get(preset_id.0)
            })
            .unwrap_or(&self.content.fallback_lut_view);

        let lut_sampler = &self.material_sampler;

        let bgl1 = self
            .glyph
            .bgl
            .as_ref()
            .expect("ensure_glyph_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glyph_uniform_bg"),
            layout: bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        let bgl2 = self
            .glyph
            .instance_bgl
            .as_ref()
            .expect("ensure_glyph_pipeline not called");
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glyph_instance_bg"),
            layout: bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        GlyphGpuData {
            mesh_vertex_buffer: mesh_vbuf,
            mesh_index_buffer: mesh_ibuf,
            mesh_index_count: mesh_idx_count,
            mesh_edge_index_buffer: mesh_edge_ibuf,
            mesh_edge_index_count: mesh_edge_count,
            instance_count,
            wireframe,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Ensure a glyph base mesh is cached for the given [`GlyphType`].
    /// Creates and uploads the mesh on first call for that type.
    fn ensure_glyph_mesh(&mut self, device: &wgpu::Device, glyph_type: crate::renderer::GlyphType) {
        use crate::renderer::GlyphType;

        let already_cached = match glyph_type {
            GlyphType::Arrow => self.glyph.arrow_mesh.is_some(),
            GlyphType::Sphere => self.glyph.sphere_mesh.is_some(),
            GlyphType::Cube => self.glyph.cube_mesh.is_some(),
        };
        if already_cached {
            return;
        }

        let (verts, indices) = match glyph_type {
            GlyphType::Arrow => build_glyph_arrow(),
            GlyphType::Sphere => build_glyph_sphere(),
            GlyphType::Cube => build_unit_cube(),
        };

        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_mesh_vbuf"),
            size: (std::mem::size_of::<Vertex>() * verts.len()).max(64) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vbuf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&verts));
        vbuf.unmap();

        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_mesh_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()).max(12) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        ibuf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&indices));
        ibuf.unmap();

        let edge_indices = crate::resources::extra_impls::generate_edge_indices(&indices);
        let edge_buf_size = (std::mem::size_of::<u32>() * edge_indices.len().max(2)) as u64;
        let edge_ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_mesh_edge_ibuf"),
            size: edge_buf_size,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut mapped = edge_ibuf.slice(..).get_mapped_range_mut();
            let bytes = bytemuck::cast_slice::<u32, u8>(&edge_indices);
            mapped[..bytes.len()].copy_from_slice(bytes);
        }
        edge_ibuf.unmap();

        let mesh = GlyphBaseMesh {
            vertex_buffer: vbuf,
            index_buffer: ibuf,
            index_count: indices.len() as u32,
            edge_index_buffer: edge_ibuf,
            edge_index_count: edge_indices.len() as u32,
        };

        match glyph_type {
            GlyphType::Arrow => self.glyph.arrow_mesh = Some(mesh),
            GlyphType::Sphere => self.glyph.sphere_mesh = Some(mesh),
            GlyphType::Cube => self.glyph.cube_mesh = Some(mesh),
        }
    }

    /// Lazily create the tensor glyph render pipeline (instanced ellipsoids).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.tensor_glyphs`
    /// is non-empty.
    pub(crate) fn ensure_tensor_glyph_pipeline(&mut self, device: &wgpu::Device) {
        if self.tensor_glyph.pipeline.is_some() {
            return;
        }

        let tg_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor_glyph_bgl"),
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
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let tg_instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor_glyph_instance_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor_glyph_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/tensor_glyph.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tensor_glyph_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &tg_bgl, &tg_instance_bgl],
            push_constant_ranges: &[],
        });

        self.tensor_glyph.bgl = Some(tg_bgl);
        self.tensor_glyph.instance_bgl = Some(tg_instance_bgl);
        self.tensor_glyph.pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "tensor_glyph_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                depth_write: true,
                depth_compare: wgpu::CompareFunction::Less,
                sample_count: self.sample_count,
                ldr_format: self.target_format,
            },
        ));

        // Wireframe variant: same bind groups, LineList topology, no culling.
        self.tensor_glyph.wireframe_pipeline =
            Some(crate::resources::builders::build_dual_pipeline(
                device,
                &crate::resources::builders::DualPipelineDesc {
                    label: "tensor_glyph_wireframe_pipeline",
                    layout: &layout,
                    shader: &shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &[Vertex::buffer_layout()],
                    blend: None,
                    topology: wgpu::PrimitiveTopology::LineList,
                    cull_mode: None,
                    depth_write: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    sample_count: self.sample_count,
                    ldr_format: self.target_format,
                },
            ));
    }

    /// Upload one [`TensorGlyphItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.scene.tensor_glyphs`.
    /// Reuses the sphere base mesh cached by the glyph pipeline.
    pub(crate) fn upload_tensor_glyph_set_per_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::TensorGlyphItem,
        wireframe: bool,
    ) -> TensorGlyphGpuData {
        use crate::renderer::GlyphType;

        let instance_count = item.positions.len() as u32;

        // Reuse the cached sphere mesh from the glyph pipeline.
        self.ensure_glyph_mesh(device, GlyphType::Sphere);
        let (mesh_vbuf, mesh_ibuf, mesh_idx_count, mesh_edge_ibuf, mesh_edge_count) = {
            let mesh = self
                .glyph
                .sphere_mesh
                .as_ref()
                .expect("sphere mesh should be present after ensure_glyph_mesh");
            let vbuf: &'static wgpu::Buffer = unsafe { &*(&mesh.vertex_buffer as *const _) };
            let ibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.index_buffer as *const _) };
            let eibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.edge_index_buffer as *const _) };
            (vbuf, ibuf, mesh.index_count, eibuf, mesh.edge_index_count)
        };

        // Pre-compute per-instance model and normal matrices on the CPU.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TensorInstance {
            model_col0: [f32; 4],
            model_col1: [f32; 4],
            model_col2: [f32; 4],
            model_col3: [f32; 4],
            normal_col0: [f32; 4],
            normal_col1: [f32; 4],
            normal_col2: [f32; 4],
            scalar: f32,
            _pad: [f32; 3],
        }

        // `item.model` is uploaded into `TensorGlyphUniform.model`; the
        // shader composes it on top of the per-instance ellipsoid model so
        // pre-uploaded sets can be moved per frame without rebuilding the
        // instance buffer.

        // Determine scalars for LUT lookup.
        let has_scalars = item.colour_attribute.is_some();
        let (scalar_min, scalar_max) = if let Some(ref scalars) = item.colour_attribute {
            item.scalar_range.unwrap_or_else(|| {
                let mn = scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (mn, mx)
            })
        } else {
            // Sign colouring: map [-1, 1] so LUT midpoint = neutral.
            item.scalar_range.unwrap_or((-1.0, 1.0))
        };

        let instances: Vec<TensorInstance> = (0..item.positions.len())
            .map(|i| {
                let pos = glam::Vec3::from(item.positions[i]);
                let ev = if i < item.eigenvalues.len() {
                    item.eigenvalues[i]
                } else {
                    [1.0, 1.0, 1.0]
                };
                let vecs = if i < item.eigenvectors.len() {
                    item.eigenvectors[i]
                } else {
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                };

                // Scale by |eigenvalue| * global_scale, minimum 1e-6 to avoid degenerate.
                let s0 = (ev[0].abs() * item.scale).max(1e-6_f32);
                let s1 = (ev[1].abs() * item.scale).max(1e-6_f32);
                let s2 = (ev[2].abs() * item.scale).max(1e-6_f32);

                // Rotation matrix: columns are the eigenvectors.
                let col0 = glam::Vec3::from(vecs[0]);
                let col1 = glam::Vec3::from(vecs[1]);
                let col2 = glam::Vec3::from(vecs[2]);

                // Rotation-scale block: RS = R * diag(s0, s1, s2).
                let rs = glam::Mat3::from_cols(col0 * s0, col1 * s1, col2 * s2);

                // 4x4 model matrix.
                let local_model = glam::Mat4::from_mat3(rs) * glam::Mat4::IDENTITY;
                let mut world_model = local_model;
                world_model.w_axis = glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);

                // Normal matrix: R * diag(1/s0, 1/s1, 1/s2).
                let nm = glam::Mat3::from_cols(col0 / s0, col1 / s1, col2 / s2);

                // Scalar for LUT.
                let scalar = if has_scalars {
                    item.colour_attribute
                        .as_ref()
                        .and_then(|sc| sc.get(i))
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    // Sign of dominant eigenvalue.
                    if i < item.eigenvalues.len() {
                        item.eigenvalues[i][0]
                    } else {
                        0.0
                    }
                };

                let mc = world_model.to_cols_array_2d();
                TensorInstance {
                    model_col0: mc[0],
                    model_col1: mc[1],
                    model_col2: mc[2],
                    model_col3: mc[3],
                    normal_col0: [nm.x_axis.x, nm.x_axis.y, nm.x_axis.z, 0.0],
                    normal_col1: [nm.y_axis.x, nm.y_axis.y, nm.y_axis.z, 0.0],
                    normal_col2: [nm.z_axis.x, nm.z_axis.y, nm.z_axis.z, 0.0],
                    scalar,
                    _pad: [0.0; 3],
                }
            })
            .collect();

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor_glyph_instance_buf"),
            size: (std::mem::size_of::<TensorInstance>() * instances.len()).max(128) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TensorGlyphUniform {
            model: [[f32; 4]; 4],
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
            _pad1b: f32,
            _pad1c: f32,
            _pad2: [[f32; 4]; 2],
        }
        let uniform_data = TensorGlyphUniform {
            model: item.model,
            has_scalars: if has_scalars { 1 } else { 0 },
            scalar_min,
            scalar_max,
            unlit: item.settings.unlit as u32,
            opacity: item.settings.opacity,
            wireframe: if wireframe { 1 } else { 0 },
            _pad1b: 0.0,
            _pad1c: 0.0,
            _pad2: [[0.0; 4]; 2],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor_glyph_uniform_buf"),
            size: std::mem::size_of::<TensorGlyphUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = self
            .content
            .builtin_colourmap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colourmap_id
                    .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                self.content.colourmap_views.get(preset_id.0)
            })
            .unwrap_or(&self.content.fallback_lut_view);

        let lut_sampler = &self.material_sampler;

        let bgl1 = self
            .tensor_glyph
            .bgl
            .as_ref()
            .expect("ensure_tensor_glyph_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tensor_glyph_uniform_bg"),
            layout: bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        let bgl2 = self
            .tensor_glyph
            .instance_bgl
            .as_ref()
            .expect("ensure_tensor_glyph_pipeline not called");
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tensor_glyph_instance_bg"),
            layout: bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        TensorGlyphGpuData {
            mesh_vertex_buffer: mesh_vbuf,
            mesh_index_buffer: mesh_ibuf,
            mesh_index_count: mesh_idx_count,
            mesh_edge_index_buffer: mesh_edge_ibuf,
            mesh_edge_index_count: mesh_edge_count,
            instance_count,
            wireframe,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Lazily create the glyph outline mask pipeline.
    ///
    /// Renders the instanced glyph mesh into the R8 outline mask texture so
    /// outlines follow the actual arrow/sphere shape.  Reuses the bind group
    /// layouts from the main glyph pipeline (must be called after
    /// `ensure_glyph_pipeline`).
    pub(crate) fn ensure_glyph_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.glyph.outline_mask_pipeline.is_some() {
            return;
        }
        let glyph_bgl = self
            .glyph
            .bgl
            .as_ref()
            .expect("ensure_glyph_pipeline must be called first");
        let glyph_instance_bgl = self
            .glyph
            .instance_bgl
            .as_ref()
            .expect("ensure_glyph_pipeline must be called first");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glyph_outline_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/glyph_outline_mask.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glyph_outline_mask_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                glyph_bgl,
                glyph_instance_bgl,
            ],
            push_constant_ranges: &[],
        });

        self.glyph.outline_mask_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("glyph_outline_mask_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            },
        ));
    }

    /// Lazily create the tensor glyph outline mask pipeline.
    ///
    /// Same idea as `ensure_glyph_outline_mask_pipeline` but for tensor
    /// glyph ellipsoids.  Must be called after `ensure_tensor_glyph_pipeline`.
    pub(crate) fn ensure_tensor_glyph_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.tensor_glyph.outline_mask_pipeline.is_some() {
            return;
        }
        let tg_bgl = self
            .tensor_glyph
            .bgl
            .as_ref()
            .expect("ensure_tensor_glyph_pipeline must be called first");
        let tg_instance_bgl = self
            .tensor_glyph
            .instance_bgl
            .as_ref()
            .expect("ensure_tensor_glyph_pipeline must be called first");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor_glyph_outline_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/tensor_glyph_outline_mask.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tensor_glyph_outline_mask_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, tg_bgl, tg_instance_bgl],
            push_constant_ranges: &[],
        });

        self.tensor_glyph.outline_mask_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("tensor_glyph_outline_mask_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            },
        ));
    }

    /// Pre-upload a glyph set and return a typed handle.
    pub fn upload_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::GlyphItem,
    ) -> crate::resources::GlyphSetId {
        self.ensure_glyph_pipeline(device);
        let gpu = self.upload_glyph_set_per_frame(device, queue, item, false);
        self.content.glyph_set_store.insert(gpu)
    }

    /// Remove a pre-uploaded glyph set.
    pub fn drop_glyph_set(&mut self, id: crate::resources::GlyphSetId) -> bool {
        self.content.glyph_set_store.remove(id)
    }

    /// Replace the geometry of a pre-uploaded glyph set, keeping the same id.
    pub fn replace_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: crate::resources::GlyphSetId,
        item: &crate::renderer::GlyphItem,
    ) -> bool {
        if !self.content.glyph_set_store.contains(id) {
            return false;
        }
        self.ensure_glyph_pipeline(device);
        let gpu = self.upload_glyph_set_per_frame(device, queue, item, false);
        self.content.glyph_set_store.replace(id, gpu)
    }

    /// Start an asynchronous glyph set upload.
    pub fn begin_upload_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: crate::renderer::GlyphItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::GlyphSetId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let gid =
                            resources.upload_glyph_set(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(gid);
                    }),
                ))
            })
        };
        self.job_results
            .glyph_set
            .lock()
            .expect("glyph set result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`GlyphSetId`](crate::resources::GlyphSetId) produced by a
    /// completed [`begin_upload_glyph_set`](Self::begin_upload_glyph_set) job.
    pub fn upload_result_glyph_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::GlyphSetId> {
        let mut map = self
            .job_results
            .glyph_set
            .lock()
            .expect("glyph set result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(gid) => {
                map.remove(&id);
                Ok(gid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Pre-upload a tensor glyph set and return a typed handle.
    pub fn upload_tensor_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::TensorGlyphItem,
    ) -> crate::resources::TensorGlyphSetId {
        self.ensure_tensor_glyph_pipeline(device);
        let gpu = self.upload_tensor_glyph_set_per_frame(device, queue, item, false);
        self.content.tensor_glyph_set_store.insert(gpu)
    }

    /// Remove a pre-uploaded tensor glyph set.
    pub fn drop_tensor_glyph_set(&mut self, id: crate::resources::TensorGlyphSetId) -> bool {
        self.content.tensor_glyph_set_store.remove(id)
    }

    /// Replace the geometry of a pre-uploaded tensor glyph set, keeping the same id.
    pub fn replace_tensor_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: crate::resources::TensorGlyphSetId,
        item: &crate::renderer::TensorGlyphItem,
    ) -> bool {
        if !self.content.tensor_glyph_set_store.contains(id) {
            return false;
        }
        self.ensure_tensor_glyph_pipeline(device);
        let gpu = self.upload_tensor_glyph_set_per_frame(device, queue, item, false);
        self.content.tensor_glyph_set_store.replace(id, gpu)
    }

    /// Start an asynchronous tensor glyph set upload.
    pub fn begin_upload_tensor_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: crate::renderer::TensorGlyphItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::TensorGlyphSetId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let tid = resources.upload_tensor_glyph_set(
                            &device_for_apply,
                            &queue_for_apply,
                            &item,
                        );
                        slot_for_apply.set(tid);
                    }),
                ))
            })
        };
        self.job_results
            .tensor_glyph_set
            .lock()
            .expect("tensor glyph set result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`TensorGlyphSetId`](crate::resources::TensorGlyphSetId) produced by a
    /// completed [`begin_upload_tensor_glyph_set`](Self::begin_upload_tensor_glyph_set) job.
    pub fn upload_result_tensor_glyph_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TensorGlyphSetId> {
        let mut map = self
            .job_results
            .tensor_glyph_set
            .lock()
            .expect("tensor glyph set result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(tid) => {
                map.remove(&id);
                Ok(tid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::DeviceResources;
    use crate::renderer::{GlyphItem, TensorGlyphItem};
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

    fn sample_glyph_set() -> GlyphItem {
        let mut item = GlyphItem::default();
        item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        item.vectors = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        item
    }

    fn sample_tensor_glyph_set() -> TensorGlyphItem {
        let mut item = TensorGlyphItem::default();
        item.positions = vec![[0.0, 0.0, 0.0]];
        item.eigenvalues = vec![[1.0, 0.5, 0.25]];
        item.eigenvectors = vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]];
        item
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
    fn upload_glyph_set_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_glyph_set(&device, &queue, &sample_glyph_set());
        assert!(resources.content.glyph_set_store.contains(id));
        assert!(resources.drop_glyph_set(id));
    }

    #[test]
    fn upload_tensor_glyph_set_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_tensor_glyph_set(&device, &queue, &sample_tensor_glyph_set());
        assert!(resources.content.tensor_glyph_set_store.contains(id));
        assert!(resources.drop_tensor_glyph_set(id));
    }

    #[test]
    fn begin_upload_glyph_set_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_glyph_set(&device, &queue, sample_glyph_set());
        drive_until_ready(&mut resources, &device, &queue, job, "glyph_set");
        let id = resources.upload_result_glyph_set(job).expect("ready");
        assert!(resources.content.glyph_set_store.contains(id));
    }

    #[test]
    fn begin_upload_tensor_glyph_set_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job =
            resources.begin_upload_tensor_glyph_set(&device, &queue, sample_tensor_glyph_set());
        drive_until_ready(&mut resources, &device, &queue, job, "tensor_glyph_set");
        let id = resources
            .upload_result_tensor_glyph_set(job)
            .expect("ready");
        assert!(resources.content.tensor_glyph_set_store.contains(id));
    }
}
