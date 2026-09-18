use super::*;

/// The glyph bind group layouts and the cached arrow / sphere / cube base
/// meshes. Uploads build their bind groups against the layouts, so they live
/// here with the store rather than with the item type's pipelines, and are
/// created up front: they are layouts, not compiled pipelines. The meshes are
/// still built on first use, but through a shared reference, so an item-type
/// plugin can reach them from `prepare`, which holds `&DeviceResources`.
pub(crate) struct GlyphResources {
    /// Bind group layout for glyph uniforms (group 1).
    pub(crate) bgl: crate::gpu::BindGroupLayout,
    /// Bind group layout for glyph instance storage (group 2).
    pub(crate) instance_bgl: crate::gpu::BindGroupLayout,
    /// Cached glyph base mesh for the Arrow shape.
    pub(crate) arrow_mesh: std::sync::OnceLock<GlyphBaseMesh>,
    /// Cached glyph base mesh for the Sphere shape.
    pub(crate) sphere_mesh: std::sync::OnceLock<GlyphBaseMesh>,
    /// Cached glyph base mesh for the Cube shape.
    pub(crate) cube_mesh: std::sync::OnceLock<GlyphBaseMesh>,
}

/// The tensor glyph bind group layouts. Uploads build their bind groups against
/// them, so they live here with the store rather than with the item type's
/// pipelines, and are created up front: they are layouts, not compiled
/// pipelines.
pub(crate) struct TensorGlyphResources {
    /// Bind group layout for tensor glyph uniforms (group 1).
    pub(crate) bgl: crate::gpu::BindGroupLayout,
    /// Bind group layout for tensor glyph instance storage (group 2).
    pub(crate) instance_bgl: crate::gpu::BindGroupLayout,
}

impl TensorGlyphResources {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = crate::resources::builders::uniform_texture_sampler_bgl(
            device,
            "tensor_glyph_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            crate::gpu::ShaderStages::VERTEX,
        );
        let instance_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("tensor_glyph_instance_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        Self { bgl, instance_bgl }
    }
}

impl GlyphResources {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = crate::resources::builders::uniform_texture_sampler_bgl(
            device,
            "glyph_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            crate::gpu::ShaderStages::VERTEX,
        );
        let instance_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("glyph_instance_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        Self {
            bgl,
            instance_bgl,
            arrow_mesh: std::sync::OnceLock::new(),
            sphere_mesh: std::sync::OnceLock::new(),
            cube_mesh: std::sync::OnceLock::new(),
        }
    }
}

impl DeviceResources {
    /// Upload one [`GlyphItem`] to the GPU and return draw data.
    ///
    /// Shared by the per-frame item upload and the pre-upload store. The base
    /// mesh comes from the shared glyph mesh cache.
    pub(crate) fn upload_glyph_set_per_frame(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::GlyphItem,
        wireframe: bool,
    ) -> GlyphGpuData {
        let instance_count = item.positions.len() as u32;

        let (mesh_vbuf, mesh_ibuf, mesh_idx_count, mesh_edge_ibuf, mesh_edge_count) = {
            let mesh = self.ensure_glyph_mesh(device, item.glyph_type);
            (
                mesh.vertex_buffer.clone(),
                mesh.index_buffer.clone(),
                mesh.index_count,
                mesh.edge_index_buffer.clone(),
                mesh.edge_index_count,
            )
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

        let instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("glyph_instance_buf"),
            size: (std::mem::size_of::<GlyphInstance>() * instances.len()).max(32) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
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
            default_colour: item.default_colour.to_linear_rgba(),
            use_default_colour: if item.default_colour.alpha() > 0.0 && item.use_default_colour {
                1
            } else {
                0
            },
            unlit: if item.settings.unlit { 1 } else { 0 },
            opacity: item.settings.opacity,
            wireframe: if wireframe { 1 } else { 0 },
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("glyph_uniform_buf"),
            size: std::mem::size_of::<GlyphUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
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

        let lut_sampler = &self.material.sampler;

        let uniform_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("glyph_uniform_bg"),
            layout: &self.glyph.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        let instance_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("glyph_instance_bg"),
            layout: &self.glyph.instance_bgl,
            entries: &[crate::gpu::BindGroupEntry {
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
            pick_id: item.settings.pick_id,
            wireframe,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Ensure a glyph base mesh is cached for the given [`GlyphType`].
    /// Creates and uploads the mesh on first call for that type.
    pub(crate) fn ensure_glyph_mesh(
        &self,
        device: &crate::gpu::Device,
        glyph_type: crate::renderer::GlyphType,
    ) -> &GlyphBaseMesh {
        use crate::renderer::GlyphType;

        let slot = match glyph_type {
            GlyphType::Arrow => &self.glyph.arrow_mesh,
            GlyphType::Sphere => &self.glyph.sphere_mesh,
            GlyphType::Cube => &self.glyph.cube_mesh,
        };
        slot.get_or_init(|| build_glyph_base_mesh(device, glyph_type))
    }
}

/// Build one glyph base mesh: the shape's vertex and index buffers plus the
/// edge index buffer the wireframe variant draws.
fn build_glyph_base_mesh(
    device: &crate::gpu::Device,
    glyph_type: crate::renderer::GlyphType,
) -> GlyphBaseMesh {
    use crate::renderer::GlyphType;
    {
        let (verts, indices) = match glyph_type {
            GlyphType::Arrow => build_glyph_arrow(),
            GlyphType::Sphere => build_glyph_sphere(),
            GlyphType::Cube => build_unit_cube(),
        };

        let vbuf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("glyph_mesh_vbuf"),
            size: (std::mem::size_of::<Vertex>() * verts.len()).max(64) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(vbuf.slice(..), bytemuck::cast_slice(&verts));
        vbuf.unmap();

        let ibuf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("glyph_mesh_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()).max(12) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(ibuf.slice(..), bytemuck::cast_slice(&indices));
        ibuf.unmap();

        let edge_indices = crate::resources::mesh::geometry::generate_edge_indices(&indices);
        let edge_buf_size = (std::mem::size_of::<u32>() * edge_indices.len().max(2)) as u64;
        let edge_ibuf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("glyph_mesh_edge_ibuf"),
            size: edge_buf_size,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            edge_ibuf.slice(..),
            bytemuck::cast_slice::<u32, u8>(&edge_indices),
        );
        edge_ibuf.unmap();

        GlyphBaseMesh {
            vertex_buffer: vbuf,
            index_buffer: ibuf,
            index_count: indices.len() as u32,
            edge_index_buffer: edge_ibuf,
            edge_index_count: edge_indices.len() as u32,
        }
    }
}

impl DeviceResources {
    /// Upload one [`TensorGlyphItem`] to the GPU and return draw data.
    ///
    /// Shared by the per-frame item upload and the pre-upload store. Reuses the
    /// shared sphere base mesh.
    pub(crate) fn upload_tensor_glyph_set_per_frame(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::TensorGlyphItem,
        wireframe: bool,
    ) -> TensorGlyphGpuData {
        use crate::renderer::GlyphType;

        let instance_count = item.positions.len() as u32;

        // Reuse the shared sphere base mesh.
        let (mesh_vbuf, mesh_ibuf, mesh_idx_count, mesh_edge_ibuf, mesh_edge_count) = {
            let mesh = self.ensure_glyph_mesh(device, GlyphType::Sphere);
            (
                mesh.vertex_buffer.clone(),
                mesh.index_buffer.clone(),
                mesh.index_count,
                mesh.edge_index_buffer.clone(),
                mesh.edge_index_count,
            )
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

        let instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tensor_glyph_instance_buf"),
            size: (std::mem::size_of::<TensorInstance>() * instances.len()).max(128) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
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
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tensor_glyph_uniform_buf"),
            size: std::mem::size_of::<TensorGlyphUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
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

        let lut_sampler = &self.material.sampler;

        let uniform_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tensor_glyph_uniform_bg"),
            layout: &self.tensor_glyph.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        let instance_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tensor_glyph_instance_bg"),
            layout: &self.tensor_glyph.instance_bgl,
            entries: &[crate::gpu::BindGroupEntry {
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
            pick_id: item.settings.pick_id,
            wireframe,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Pre-upload a glyph set and return a typed handle.
    ///
    /// Prefer [`ViewportRenderer::upload_glyph_set`](crate::renderer::ViewportRenderer::upload_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::GlyphItem,
    ) -> crate::resources::GlyphSetId {
        let gpu = self.upload_glyph_set_per_frame(device, queue, item, false);
        self.content.glyph_set_store.insert_sized(gpu)
    }

    /// Remove a pre-uploaded glyph set.
    ///
    /// Prefer [`ViewportRenderer::drop_glyph_set`](crate::renderer::ViewportRenderer::drop_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::drop_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn drop_glyph_set(&mut self, id: crate::resources::GlyphSetId) -> bool {
        self.content.glyph_set_store.remove(id).is_some()
    }

    /// Replace the geometry of a pre-uploaded glyph set, keeping the same id.
    ///
    /// Prefer [`ViewportRenderer::replace_glyph_set`](crate::renderer::ViewportRenderer::replace_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::replace_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn replace_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::GlyphSetId,
        item: &crate::renderer::GlyphItem,
    ) -> bool {
        if !self.content.glyph_set_store.contains(id) {
            return false;
        }
        let gpu = self.upload_glyph_set_per_frame(device, queue, item, false);
        self.content
            .glyph_set_store
            .replace_sized(id, gpu)
            .is_some()
    }

    /// Start an asynchronous glyph set upload.
    ///
    /// Prefer [`ViewportRenderer::begin_upload_glyph_set`](crate::renderer::ViewportRenderer::begin_upload_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::begin_upload_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    // The apply step inserts through the synchronous upload, which goes with it.
    #[allow(deprecated)]
    pub fn begin_upload_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
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
    ///
    /// Prefer [`ViewportRenderer::upload_result_glyph_set`](crate::renderer::ViewportRenderer::upload_result_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_result_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
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
    ///
    /// Prefer [`ViewportRenderer::upload_tensor_glyph_set`](crate::renderer::ViewportRenderer::upload_tensor_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_tensor_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::TensorGlyphItem,
    ) -> crate::resources::TensorGlyphSetId {
        let gpu = self.upload_tensor_glyph_set_per_frame(device, queue, item, false);
        self.content.tensor_glyph_set_store.insert_sized(gpu)
    }

    /// Remove a pre-uploaded tensor glyph set.
    ///
    /// Prefer [`ViewportRenderer::drop_tensor_glyph_set`](crate::renderer::ViewportRenderer::drop_tensor_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::drop_tensor_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn drop_tensor_glyph_set(&mut self, id: crate::resources::TensorGlyphSetId) -> bool {
        self.content.tensor_glyph_set_store.remove(id).is_some()
    }

    /// Replace the geometry of a pre-uploaded tensor glyph set, keeping the same id.
    ///
    /// Prefer [`ViewportRenderer::replace_tensor_glyph_set`](crate::renderer::ViewportRenderer::replace_tensor_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::replace_tensor_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn replace_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::TensorGlyphSetId,
        item: &crate::renderer::TensorGlyphItem,
    ) -> bool {
        if !self.content.tensor_glyph_set_store.contains(id) {
            return false;
        }
        let gpu = self.upload_tensor_glyph_set_per_frame(device, queue, item, false);
        self.content
            .tensor_glyph_set_store
            .replace_sized(id, gpu)
            .is_some()
    }

    /// Start an asynchronous tensor glyph set upload.
    ///
    /// Prefer [`ViewportRenderer::begin_upload_tensor_glyph_set`](crate::renderer::ViewportRenderer::begin_upload_tensor_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::begin_upload_tensor_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    // The apply step inserts through the synchronous upload, which goes with it.
    #[allow(deprecated)]
    pub fn begin_upload_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
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
    ///
    /// Prefer [`ViewportRenderer::upload_result_tensor_glyph_set`](crate::renderer::ViewportRenderer::upload_result_tensor_glyph_set),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_result_tensor_glyph_set instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
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
    // These drive the DeviceResources upload calls directly, which is the
    // point: they test the methods the renderer-level ones forward to.
    #![allow(deprecated)]
    use crate::DeviceResources;
    use crate::renderer::{GlyphItem, TensorGlyphItem};
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
    fn upload_glyph_set_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
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
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
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
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
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
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job =
            resources.begin_upload_tensor_glyph_set(&device, &queue, sample_tensor_glyph_set());
        drive_until_ready(&mut resources, &device, &queue, job, "tensor_glyph_set");
        let id = resources
            .upload_result_tensor_glyph_set(job)
            .expect("ready");
        assert!(resources.content.tensor_glyph_set_store.contains(id));
    }
}

/// Cached GPU vertex + index buffers for a glyph base mesh (arrow, sphere, cube).
pub(crate) struct GlyphBaseMesh {
    /// Vertex buffer using the full `Vertex` layout (64 bytes stride).
    pub vertex_buffer: crate::gpu::Buffer,
    /// Triangle index buffer.
    pub index_buffer: crate::gpu::Buffer,
    /// Number of indices.
    pub index_count: u32,
    /// Edge index buffer (deduplicated pairs) for wireframe LineList rendering.
    pub edge_index_buffer: crate::gpu::Buffer,
    /// Number of indices in the edge buffer.
    pub edge_index_count: u32,
}
/// Per-frame GPU data for one glyph item, created in `prepare()`.
#[derive(Clone)]
pub struct GlyphGpuData {
    /// Vertex buffer for the glyph base mesh. A cloned handle on the shared
    /// mesh `DeviceResources` caches, so the draw data outlives the borrow the
    /// upload took.
    pub(crate) mesh_vertex_buffer: crate::gpu::Buffer,
    /// Triangle index buffer for the glyph base mesh.
    pub(crate) mesh_index_buffer: crate::gpu::Buffer,
    /// Number of triangle mesh indices.
    pub(crate) mesh_index_count: u32,
    /// Edge index buffer for wireframe LineList rendering.
    pub(crate) mesh_edge_index_buffer: crate::gpu::Buffer,
    /// Number of edge indices.
    pub(crate) mesh_edge_index_count: u32,
    /// Number of glyph instances.
    pub(crate) instance_count: u32,
    /// Object-level pick id shared by every instance in the set (from the item's
    /// `settings.pick_id`); `PickId::NONE` when the set is not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Whether this batch should be drawn with the wireframe pipeline.
    pub(crate) wireframe: bool,
    /// Bind group (group 1): glyph uniform + LUT texture + sampler.
    pub(crate) uniform_bind_group: crate::gpu::BindGroup,
    /// Bind group (group 2): instance storage buffer.
    pub(crate) instance_bind_group: crate::gpu::BindGroup,
    // Keep the buffers alive.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) _instance_buf: crate::gpu::Buffer,
}

/// Per-frame GPU data for one tensor glyph item, created in `prepare()`.
///
/// The sphere base mesh comes from the shared glyph mesh cache; the buffer
/// handles here are clones of it.
#[derive(Clone)]
pub struct TensorGlyphGpuData {
    /// Vertex buffer for the sphere base mesh.
    pub(crate) mesh_vertex_buffer: crate::gpu::Buffer,
    /// Triangle index buffer for the sphere base mesh.
    pub(crate) mesh_index_buffer: crate::gpu::Buffer,
    /// Number of triangle mesh indices.
    pub(crate) mesh_index_count: u32,
    /// Edge index buffer for wireframe LineList rendering.
    pub(crate) mesh_edge_index_buffer: crate::gpu::Buffer,
    /// Number of edge indices.
    pub(crate) mesh_edge_index_count: u32,
    /// Number of tensor glyph instances.
    pub(crate) instance_count: u32,
    /// Object-level pick id shared by every instance in the set (from the item's
    /// `settings.pick_id`); `PickId::NONE` when the set is not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Whether this batch should be drawn with the wireframe pipeline.
    pub(crate) wireframe: bool,
    /// Bind group (group 1): uniform + LUT texture + sampler.
    pub(crate) uniform_bind_group: crate::gpu::BindGroup,
    /// Bind group (group 2): per-instance storage buffer.
    pub(crate) instance_bind_group: crate::gpu::BindGroup,
    // Keep buffers alive.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) _instance_buf: crate::gpu::Buffer,
}
