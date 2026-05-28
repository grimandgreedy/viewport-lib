use super::*;
use rayon::prelude::*;

impl ViewportGpuResources {
    /// Create a GpuMesh from vertex/index slices and register it into the resource list.
    ///
    /// Returns the `MeshId` of the new mesh.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::EmptyMesh`](crate::error::ViewportError::EmptyMesh) if
    /// `vertices` or `indices` is empty.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use viewport_lib::error::ViewportError;
    /// # fn demo(resources: &mut viewport_lib::resources::ViewportGpuResources, device: &wgpu::Device) {
    /// let result = resources.upload_mesh(device, &[], &[]);
    /// assert!(matches!(result, Err(ViewportError::EmptyMesh { .. })));
    /// # }
    /// ```
    pub fn upload_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> crate::error::ViewportResult<crate::resources::mesh_store::MeshId> {
        if vertices.is_empty() || indices.is_empty() {
            return Err(crate::error::ViewportError::EmptyMesh {
                positions: vertices.len(),
                indices: indices.len(),
            });
        }
        self.frame_upload_bytes += (vertices.len() * std::mem::size_of::<Vertex>()
            + indices.len() * std::mem::size_of::<u32>()) as u64;
        let mesh = Self::create_mesh(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.lut_sampler,
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.fallback_face_colour_buf,
            &self.fallback_warp_buf,
            &self.fallback_metallic_roughness_texture_view,
            &self.fallback_emissive_texture_view,
            vertices,
            indices,
        );
        Ok(self.mesh_store.insert(mesh))
    }

    /// Upload a `MeshData` (from the geometry primitives module) directly.
    ///
    /// Converts positions/normals/indices to the GPU `Vertex` layout (white colour)
    /// and creates a normal visualization line buffer (light blue #a0c4ff, length 0.1).
    /// Returns the `MeshId`.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::EmptyMesh`](crate::error::ViewportError::EmptyMesh) if positions or indices are empty,
    /// [`ViewportError::MeshLengthMismatch`](crate::error::ViewportError::MeshLengthMismatch) if positions and normals differ in length,
    /// or [`ViewportError::InvalidVertexIndex`](crate::error::ViewportError::InvalidVertexIndex) if an index references a nonexistent vertex.
    pub fn upload_mesh_data(
        &mut self,
        device: &wgpu::Device,
        data: &MeshData,
    ) -> crate::error::ViewportResult<crate::resources::mesh_store::MeshId> {
        Self::validate_mesh_data(data)?;

        let computed_tangents: Option<Vec<[f32; 4]>> = if data.tangents.is_none() {
            data.uvs.as_ref().map(|uvs| {
                Self::compute_tangents(&data.positions, &data.normals, uvs, &data.indices)
            })
        } else {
            None
        };
        let tangent_slice = data.tangents.as_deref().or(computed_tangents.as_deref());

        let vertices: Vec<Vertex> = data
            .positions
            .iter()
            .zip(data.normals.iter())
            .enumerate()
            .map(|(i, (p, n))| {
                let uv = data
                    .uvs
                    .as_ref()
                    .and_then(|uvs| uvs.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                let tangent = tangent_slice
                    .and_then(|ts| ts.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                Vertex {
                    position: *p,
                    normal: *n,
                    colour: [1.0, 1.0, 1.0, 1.0],
                    uv,
                    tangent,
                }
            })
            .collect();

        let normal_line_verts = Self::build_normal_lines(data);

        let mut mesh = Self::create_mesh_with_normals(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.lut_sampler,
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.fallback_face_colour_buf,
            &self.fallback_warp_buf,
            &self.fallback_metallic_roughness_texture_view,
            &self.fallback_emissive_texture_view,
            &vertices,
            &data.indices,
            Some(&normal_line_verts),
        );
        mesh.cpu_positions = Some(data.positions.clone());
        mesh.cpu_indices = Some(data.indices.clone());
        let (attr_bufs, attr_ranges, face_vbuf, face_attr_bufs, face_colour_bufs, vector_attr_bufs) =
            Self::upload_attributes(
                device,
                &data.attributes,
                &data.positions,
                &data.normals,
                &data.indices,
                data.uvs.as_deref(),
                tangent_slice,
            );
        mesh.attribute_buffers = attr_bufs;
        mesh.attribute_ranges = attr_ranges;
        mesh.face_vertex_buffer = face_vbuf;
        mesh.face_attribute_buffers = face_attr_bufs;
        mesh.face_colour_buffers = face_colour_bufs;
        mesh.vector_attribute_buffers = vector_attr_bufs;
        self.frame_upload_bytes += (vertices.len() * std::mem::size_of::<Vertex>()
            + data.indices.len() * std::mem::size_of::<u32>())
            as u64;
        let id = self.mesh_store.insert(mesh);
        tracing::debug!(
            mesh_index = id.index(),
            vertices = data.positions.len(),
            indices = data.indices.len(),
            "mesh uploaded"
        );
        Ok(id)
    }

    /// Upload a `MeshData` and retain CPU positions and indices for picking.
    ///
    /// Equivalent to [`upload_mesh_data`](Self::upload_mesh_data). The CPU
    /// position and index data is kept so that `renderer.pick()` can test
    /// FACE, EDGE, and VERTEX hits against this mesh. Use this variant to
    /// make the intent explicit at the call site.
    ///
    /// # Errors
    ///
    /// Same as [`upload_mesh_data`](Self::upload_mesh_data).
    pub fn upload_mesh_data_pickable(
        &mut self,
        device: &wgpu::Device,
        data: &MeshData,
    ) -> crate::error::ViewportResult<crate::resources::mesh_store::MeshId> {
        self.upload_mesh_data(device, data)
    }

    /// Free or retain the CPU position and index data for an already-uploaded mesh.
    ///
    /// `set_pickable(id, false)` drops the retained CPU data, freeing memory.
    /// The mesh continues to render normally; it will be silently skipped for
    /// FACE, EDGE, and VERTEX picks after this call.
    ///
    /// `set_pickable(id, true)` is a no-op: CPU data is either already present
    /// (the mesh was uploaded via [`upload_mesh_data`] or
    /// [`upload_mesh_data_pickable`]) or it was freed and cannot be recovered
    /// without re-uploading.
    ///
    /// Has no effect if `mesh_id` is not found.
    pub fn set_pickable(&mut self, mesh_id: crate::resources::mesh_store::MeshId, pickable: bool) {
        if let Some(mesh) = self.mesh_store.get_mut(mesh_id) {
            if !pickable {
                mesh.cpu_positions = None;
                mesh.cpu_indices = None;
            }
        }
    }

    /// Write new positions and normals into an existing mesh without reallocating GPU buffers.
    ///
    /// The vertex count must match the original upload exactly. Use this for deforming meshes
    /// where topology is stable across frames: the index buffer, edge buffer, and bind groups
    /// are all reused. Colour, UVs, and tangents are written as defaults (white, zero, [0,0,0,1]).
    ///
    /// The normal line visualization buffer is also updated in place if it was created at upload time.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::MeshIndexOutOfBounds`](crate::error::ViewportError::MeshIndexOutOfBounds)
    /// if `mesh_id` is out of range, [`ViewportError::MeshLengthMismatch`](crate::error::ViewportError::MeshLengthMismatch)
    /// if `positions` and `normals` differ in length or do not match the existing vertex count.
    pub fn write_mesh_positions_normals(
        &mut self,
        queue: &wgpu::Queue,
        mesh_id: crate::resources::mesh_store::MeshId,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
    ) -> crate::error::ViewportResult<()> {
        use bytemuck::cast_slice;

        if !self.mesh_store.contains(mesh_id) {
            return Err(crate::error::ViewportError::MeshIndexOutOfBounds {
                index: mesh_id.index(),
                count: self.mesh_store.len(),
            });
        }
        if positions.len() != normals.len() {
            return Err(crate::error::ViewportError::MeshLengthMismatch {
                positions: positions.len(),
                normals: normals.len(),
            });
        }

        let existing_vertex_count = {
            let mesh = self.mesh_store.get(mesh_id).unwrap();
            (mesh.vertex_buffer.size() / std::mem::size_of::<Vertex>() as u64) as usize
        };
        if positions.len() != existing_vertex_count {
            return Err(crate::error::ViewportError::MeshLengthMismatch {
                positions: positions.len(),
                normals: existing_vertex_count,
            });
        }

        let vertices: Vec<Vertex> = positions
            .iter()
            .zip(normals.iter())
            .map(|(p, n)| Vertex {
                position: *p,
                normal: *n,
                colour: [1.0, 1.0, 1.0, 1.0],
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            })
            .collect();

        let has_normal_lines = self
            .mesh_store
            .get(mesh_id)
            .unwrap()
            .normal_line_buffer
            .is_some();
        let normal_line_verts: Option<Vec<Vertex>> = if has_normal_lines {
            let normal_length = 0.1_f32;
            let normal_colour = [0.627_f32, 0.769, 1.0, 1.0];
            let mut verts = Vec::with_capacity(positions.len() * 2);
            for (p, n) in positions.iter().zip(normals.iter()) {
                let tip = [
                    p[0] + n[0] * normal_length,
                    p[1] + n[1] * normal_length,
                    p[2] + n[2] * normal_length,
                ];
                verts.push(Vertex {
                    position: *p,
                    normal: *n,
                    colour: normal_colour,
                    uv: [0.0, 0.0],
                    tangent: [0.0, 0.0, 0.0, 1.0],
                });
                verts.push(Vertex {
                    position: tip,
                    normal: *n,
                    colour: normal_colour,
                    uv: [0.0, 0.0],
                    tangent: [0.0, 0.0, 0.0, 1.0],
                });
            }
            Some(verts)
        } else {
            None
        };

        let aabb = crate::scene::aabb::Aabb::from_positions(positions);
        let mesh = self.mesh_store.get_mut(mesh_id).unwrap();
        queue.write_buffer(&mesh.vertex_buffer, 0, cast_slice(&vertices));
        if let (Some(nl_buf), Some(nl_verts)) = (&mesh.normal_line_buffer, &normal_line_verts) {
            queue.write_buffer(nl_buf, 0, cast_slice(nl_verts.as_slice()));
        }
        mesh.aabb = aabb;
        if let Some(ref mut cp) = mesh.cpu_positions {
            *cp = positions.to_vec();
        }

        self.frame_upload_bytes += (vertices.len() * std::mem::size_of::<Vertex>()) as u64;
        if let Some(ref nl) = normal_line_verts {
            self.frame_upload_bytes += (nl.len() * std::mem::size_of::<Vertex>()) as u64;
        }

        Ok(())
    }

    /// Replace the mesh at `mesh_index` with new geometry data.
    ///
    /// When the new vertex and index counts match the existing mesh and no attributes are
    /// present, the existing GPU buffers are reused and data is written in place, avoiding
    /// GPU memory allocation. When topology changes, new buffers are allocated.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::MeshIndexOutOfBounds`](crate::error::ViewportError::MeshIndexOutOfBounds) if `mesh_index` is out of range,
    /// or any mesh validation error from the new data.
    pub fn replace_mesh_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh_id: crate::resources::mesh_store::MeshId,
        data: &MeshData,
    ) -> crate::error::ViewportResult<()> {
        if !self.mesh_store.contains(mesh_id) {
            return Err(crate::error::ViewportError::MeshIndexOutOfBounds {
                index: mesh_id.index(),
                count: self.mesh_store.len(),
            });
        }
        Self::validate_mesh_data(data)?;

        let computed_tangents: Option<Vec<[f32; 4]>> = if data.tangents.is_none() {
            data.uvs.as_ref().map(|uvs| {
                Self::compute_tangents(&data.positions, &data.normals, uvs, &data.indices)
            })
        } else {
            None
        };
        let tangent_slice = data.tangents.as_deref().or(computed_tangents.as_deref());

        let vertices: Vec<Vertex> = data
            .positions
            .iter()
            .zip(data.normals.iter())
            .enumerate()
            .map(|(i, (p, n))| {
                let uv = data
                    .uvs
                    .as_ref()
                    .and_then(|uvs| uvs.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                let tangent = tangent_slice
                    .and_then(|ts| ts.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                Vertex {
                    position: *p,
                    normal: *n,
                    colour: [1.0, 1.0, 1.0, 1.0],
                    uv,
                    tangent,
                }
            })
            .collect();

        // Fast path: when topology is unchanged and no attributes need updating, write
        // directly to the existing GPU buffers to avoid re-allocation.
        {
            let existing = self.mesh_store.get(mesh_id).unwrap();
            let existing_vc =
                (existing.vertex_buffer.size() / std::mem::size_of::<Vertex>() as u64) as usize;
            let in_place = existing_vc == vertices.len()
                && existing.index_count as usize == data.indices.len()
                && data.attributes.is_empty();

            if in_place {
                use bytemuck::cast_slice;
                let edge_indices =
                    crate::resources::extra_impls::generate_edge_indices(&data.indices);
                let normal_line_verts = Self::build_normal_lines(data);
                let aabb = crate::scene::aabb::Aabb::from_positions(&data.positions);

                let mesh = self.mesh_store.get_mut(mesh_id).unwrap();
                queue.write_buffer(&mesh.vertex_buffer, 0, cast_slice(&vertices));
                queue.write_buffer(&mesh.index_buffer, 0, cast_slice(data.indices.as_slice()));
                let edge_byte_len = (edge_indices.len() * std::mem::size_of::<u32>()) as u64;
                if edge_byte_len <= mesh.edge_index_buffer.size() {
                    queue.write_buffer(&mesh.edge_index_buffer, 0, cast_slice(&edge_indices));
                    mesh.edge_index_count = edge_indices.len() as u32;
                }
                if let Some(ref nl_buf) = mesh.normal_line_buffer {
                    queue.write_buffer(nl_buf, 0, cast_slice(&normal_line_verts));
                }
                mesh.aabb = aabb;
                mesh.cpu_positions = Some(data.positions.clone());
                mesh.cpu_indices = Some(data.indices.clone());

                self.frame_upload_bytes += (vertices.len() * std::mem::size_of::<Vertex>()
                    + data.indices.len() * std::mem::size_of::<u32>())
                    as u64;
                tracing::debug!(
                    mesh_index = mesh_id.index(),
                    vertices = data.positions.len(),
                    "mesh updated in place"
                );
                return Ok(());
            }
        }

        let normal_line_verts = Self::build_normal_lines(data);
        let mut new_mesh = Self::create_mesh_with_normals(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.lut_sampler,
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.fallback_face_colour_buf,
            &self.fallback_warp_buf,
            &self.fallback_metallic_roughness_texture_view,
            &self.fallback_emissive_texture_view,
            &vertices,
            &data.indices,
            Some(&normal_line_verts),
        );
        new_mesh.cpu_positions = Some(data.positions.clone());
        new_mesh.cpu_indices = Some(data.indices.clone());
        let (attr_bufs, attr_ranges, face_vbuf, face_attr_bufs, face_colour_bufs, vector_attr_bufs) =
            Self::upload_attributes(
                device,
                &data.attributes,
                &data.positions,
                &data.normals,
                &data.indices,
                data.uvs.as_deref(),
                tangent_slice,
            );
        new_mesh.attribute_buffers = attr_bufs;
        new_mesh.attribute_ranges = attr_ranges;
        new_mesh.face_vertex_buffer = face_vbuf;
        new_mesh.face_attribute_buffers = face_attr_bufs;
        new_mesh.face_colour_buffers = face_colour_bufs;
        new_mesh.vector_attribute_buffers = vector_attr_bufs;
        self.frame_upload_bytes += (vertices.len() * std::mem::size_of::<Vertex>()
            + data.indices.len() * std::mem::size_of::<u32>())
            as u64;
        let _ = self.mesh_store.replace(mesh_id, new_mesh);
        tracing::debug!(
            mesh_index = mesh_id.index(),
            vertices = data.positions.len(),
            indices = data.indices.len(),
            "mesh replaced"
        );
        Ok(())
    }

    /// Get a reference to the mesh at the given index, or `None` if the slot is empty/invalid.
    pub fn mesh(&self, id: crate::resources::mesh_store::MeshId) -> Option<&GpuMesh> {
        self.mesh_store.get(id)
    }

    /// Total number of mesh slots (including empty/removed slots).
    pub fn mesh_slot_count(&self) -> usize {
        self.mesh_store.slot_count()
    }

    /// Remove a mesh, dropping its GPU buffers and freeing its slot for reuse.
    ///
    /// Returns `true` if a mesh was removed, `false` if the slot was already empty.
    pub fn remove_mesh(&mut self, id: crate::resources::mesh_store::MeshId) -> bool {
        self.mesh_store.remove(id)
    }

    /// Upload an unstructured volume mesh by extracting its boundary surface and uploading
    /// the result via [`upload_mesh_data`](Self::upload_mesh_data).
    ///
    /// Interior faces (shared by two cells) are discarded; only boundary faces (belonging
    /// to exactly one cell) are kept. Per-cell scalar and colour attributes are remapped to
    /// per-face attributes so the face-colouring path handles them automatically.
    ///
    /// Returns the `MeshId`, identical to what [`upload_mesh_data`](Self::upload_mesh_data)
    /// would return. Reference cell attributes via
    /// [`AttributeRef { kind: AttributeKind::Face, .. }`](crate::resources::AttributeRef).
    pub fn upload_volume_mesh_data(
        &mut self,
        device: &wgpu::Device,
        data: &crate::resources::volume_mesh::VolumeMeshData,
    ) -> crate::error::ViewportResult<(crate::resources::mesh_store::MeshId, Vec<u32>)> {
        let (mesh_data, face_to_cell) = crate::resources::volume_mesh::extract_boundary_faces(data);
        let mesh_id = self.upload_mesh_data(device, &mesh_data)?;
        Ok((mesh_id, face_to_cell))
    }

    /// Upload a clipped volume mesh by extracting boundary and section faces for the
    /// given clip planes and uploading the result via [`upload_mesh_data`](Self::upload_mesh_data).
    ///
    /// Each entry in `clip_planes` is `[nx, ny, nz, d]` where a point `p` is kept when
    /// `dot(p, [nx,ny,nz]) + d >= 0`.  An empty slice is equivalent to
    /// [`upload_volume_mesh_data`](Self::upload_volume_mesh_data).
    ///
    /// Returns the `MeshId`.  Reference cell attributes via
    /// [`AttributeRef { kind: AttributeKind::Face, .. }`](crate::resources::AttributeRef).
    pub fn upload_clipped_volume_mesh_data(
        &mut self,
        device: &wgpu::Device,
        data: &crate::resources::volume_mesh::VolumeMeshData,
        clip_planes: &[[f32; 4]],
    ) -> crate::error::ViewportResult<(crate::resources::mesh_store::MeshId, Vec<u32>)> {
        let (mesh_data, face_to_cell) =
            crate::resources::volume_mesh::extract_clipped_volume_faces(data, clip_planes);
        let mesh_id = self.upload_mesh_data(device, &mesh_data)?;
        Ok((mesh_id, face_to_cell))
    }

    /// Replace an existing mesh slot with a freshly-extracted clipped volume mesh.
    ///
    /// Equivalent to calling [`upload_clipped_volume_mesh_data`](Self::upload_clipped_volume_mesh_data)
    /// and then [`replace_mesh_data`](Self::replace_mesh_data), but without allocating a new slot.
    /// Use this for per-frame clip-plane updates to avoid leaking GPU memory.
    pub fn replace_clipped_volume_mesh_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh_id: crate::resources::mesh_store::MeshId,
        data: &crate::resources::volume_mesh::VolumeMeshData,
        clip_planes: &[[f32; 4]],
    ) -> crate::error::ViewportResult<Vec<u32>> {
        let (mesh_data, face_to_cell) =
            crate::resources::volume_mesh::extract_clipped_volume_faces(data, clip_planes);
        self.replace_mesh_data(device, queue, mesh_id, &mesh_data)?;
        Ok(face_to_cell)
    }

    /// Replace a previously uploaded sparse voxel grid in place.
    ///
    /// Equivalent to calling [`upload_sparse_volume_grid_data`](Self::upload_sparse_volume_grid_data)
    /// and then [`replace_mesh_data`](Self::replace_mesh_data), but without allocating a new slot.
    /// Use this for per-frame or per-interaction updates (e.g. voxel paint) to avoid leaking GPU memory.
    pub fn replace_sparse_volume_grid_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh_id: crate::resources::mesh_store::MeshId,
        data: &crate::resources::sparse_volume::SparseVolumeGridData,
    ) -> crate::error::ViewportResult<()> {
        let mesh_data = crate::resources::sparse_volume::extract_sparse_boundary(data);
        self.replace_mesh_data(device, queue, mesh_id, &mesh_data)
    }

    /// Upload a sparse voxel grid by extracting its boundary surface and uploading
    /// the result via [`upload_mesh_data`](Self::upload_mesh_data).
    ///
    /// Only quad faces not shared between two active cells are kept.  Per-cell
    /// scalars and colours are remapped to per-face attributes, and per-node
    /// scalars are averaged over the 4 quad corners to produce per-face scalars.
    ///
    /// Returns the `MeshId`.  Reference cell and node attributes via
    /// [`AttributeRef { kind: AttributeKind::Face, .. }`](crate::resources::AttributeRef).
    pub fn upload_sparse_volume_grid_data(
        &mut self,
        device: &wgpu::Device,
        data: &crate::resources::sparse_volume::SparseVolumeGridData,
    ) -> crate::error::ViewportResult<crate::resources::mesh_store::MeshId> {
        let mesh_data = crate::resources::sparse_volume::extract_sparse_boundary(data);
        self.upload_mesh_data(device, &mesh_data)
    }

    /// Upload per-vertex, per-cell, per-face scalar, and per-face colour attributes to GPU buffers.
    ///
    /// Returns `(attribute_buffers, attribute_ranges, face_vertex_buffer, face_attribute_buffers,
    /// face_colour_buffers)`.
    ///
    /// - `attribute_buffers`: per-vertex storage buffers for `Vertex` and `Cell` kinds.
    /// - `attribute_ranges`: `(min, max)` per attribute name (all scalar kinds).
    /// - `face_vertex_buffer`: non-indexed 3N-vertex buffer (built once if any `Face`/`FaceColour` attr exists).
    /// - `face_attribute_buffers`: per-face scalar storage buffers (3N `f32` entries, replicated).
    /// - `face_colour_buffers`: per-face colour storage buffers (3N `[f32;4]` entries, replicated).
    fn upload_attributes(
        device: &wgpu::Device,
        attributes: &std::collections::HashMap<String, AttributeData>,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        uvs: Option<&[[f32; 2]]>,
        tangents: Option<&[[f32; 4]]>,
    ) -> (
        std::collections::HashMap<String, wgpu::Buffer>,
        std::collections::HashMap<String, (f32, f32)>,
        Option<wgpu::Buffer>,
        std::collections::HashMap<String, wgpu::Buffer>,
        std::collections::HashMap<String, wgpu::Buffer>,
        std::collections::HashMap<String, wgpu::Buffer>,
    ) {
        let mut bufs = std::collections::HashMap::new();
        let mut ranges = std::collections::HashMap::new();
        let mut face_attr_bufs: std::collections::HashMap<String, wgpu::Buffer> =
            std::collections::HashMap::new();
        let mut vector_attr_bufs: std::collections::HashMap<String, wgpu::Buffer> =
            std::collections::HashMap::new();
        let mut face_colour_bufs: std::collections::HashMap<String, wgpu::Buffer> =
            std::collections::HashMap::new();
        let mut face_vbuf: Option<wgpu::Buffer> = None;

        let n_tris = indices.len() / 3;

        for (name, attr_data) in attributes {
            match attr_data {
                AttributeData::Vertex(v) => {
                    let scalars = v.clone();
                    if scalars.is_empty() {
                        continue;
                    }
                    let min = scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let buf =
                        Self::create_storage_buffer_f32(device, &format!("attr_{name}"), &scalars);
                    bufs.insert(name.clone(), buf);
                    ranges.insert(name.clone(), (min, max));
                }
                AttributeData::Cell(c) => {
                    let scalars = Self::expand_cell_to_vertex(c, positions, indices);
                    if scalars.is_empty() {
                        continue;
                    }
                    let min = scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let buf =
                        Self::create_storage_buffer_f32(device, &format!("attr_{name}"), &scalars);
                    bufs.insert(name.clone(), buf);
                    ranges.insert(name.clone(), (min, max));
                }
                AttributeData::Face(f) => {
                    // Build the shared face vertex buffer on first Face/FaceColour attribute.
                    if face_vbuf.is_none() {
                        face_vbuf = Some(Self::build_face_vertex_buffer(
                            device, positions, normals, indices, uvs, tangents,
                        ));
                    }
                    let expanded = Self::expand_face_scalars_to_3n(f, n_tris);
                    if expanded.is_empty() {
                        continue;
                    }
                    let min = expanded.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = expanded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let buf = Self::create_storage_buffer_f32(
                        device,
                        &format!("face_attr_{name}"),
                        &expanded,
                    );
                    face_attr_bufs.insert(name.clone(), buf);
                    ranges.insert(name.clone(), (min, max));
                }
                AttributeData::FaceColour(colours) => {
                    // Build the shared face vertex buffer on first Face/FaceColour attribute.
                    if face_vbuf.is_none() {
                        face_vbuf = Some(Self::build_face_vertex_buffer(
                            device, positions, normals, indices, uvs, tangents,
                        ));
                    }
                    let expanded = Self::expand_face_colours_to_3n(colours, n_tris);
                    if expanded.is_empty() {
                        continue;
                    }
                    let byte_len = std::mem::size_of::<[f32; 4]>() * expanded.len();
                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("face_colour_{name}")),
                        size: byte_len as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    {
                        let mut view = buf.slice(..).get_mapped_range_mut();
                        view.copy_from_slice(bytemuck::cast_slice(&expanded));
                    }
                    buf.unmap();
                    face_colour_bufs.insert(name.clone(), buf);
                }
                AttributeData::Edge(e) => {
                    // Average edge values to vertex values (each edge's scalar is
                    // distributed to its two endpoint vertices).
                    let scalars = Self::expand_edge_to_vertex(e, positions, indices);
                    if scalars.is_empty() {
                        continue;
                    }
                    let min = scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let buf =
                        Self::create_storage_buffer_f32(device, &format!("attr_{name}"), &scalars);
                    bufs.insert(name.clone(), buf);
                    ranges.insert(name.clone(), (min, max));
                }
                AttributeData::Halfedge(h) | AttributeData::Corner(h) => {
                    // Per-corner scalars: already 3*n_tris values (one per corner),
                    // matching the face vertex buffer layout. Store directly.
                    if face_vbuf.is_none() {
                        face_vbuf = Some(Self::build_face_vertex_buffer(
                            device, positions, normals, indices, uvs, tangents,
                        ));
                    }
                    if h.is_empty() {
                        continue;
                    }
                    let expanded = h.as_slice();
                    let min = expanded.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = expanded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let buf = Self::create_storage_buffer_f32(
                        device,
                        &format!("face_attr_{name}"),
                        &expanded,
                    );
                    face_attr_bufs.insert(name.clone(), buf);
                    ranges.insert(name.clone(), (min, max));
                }
                AttributeData::VertexVector(v) => {
                    // Flatten [f32; 3] -> [f32] with 12-byte per-vertex stride.
                    // Bound as vertex buffer 1 in the LIC surface pass (location 1).
                    if v.is_empty() {
                        continue;
                    }
                    let flat: Vec<f32> = v.iter().flat_map(|&[x, y, z]| [x, y, z]).collect();
                    let byte_len = (std::mem::size_of::<f32>() * flat.len()) as u64;
                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("vec_attr_{name}")),
                        size: byte_len,
                        usage: wgpu::BufferUsages::VERTEX
                            | wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    {
                        let mut view = buf.slice(..).get_mapped_range_mut();
                        view.copy_from_slice(bytemuck::cast_slice(&flat));
                    }
                    buf.unmap();
                    vector_attr_bufs.insert(name.clone(), buf);
                }
            }
        }
        (
            bufs,
            ranges,
            face_vbuf,
            face_attr_bufs,
            face_colour_bufs,
            vector_attr_bufs,
        )
    }

    /// Allocate and fill a STORAGE buffer from a slice of `f32` values.
    fn create_storage_buffer_f32(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (std::mem::size_of::<f32>() * data.len()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = buf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(data));
        }
        buf.unmap();
        buf
    }

    /// Build a non-indexed 3N-vertex buffer: one vertex per triangle corner, geometry only.
    fn build_face_vertex_buffer(
        device: &wgpu::Device,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        uvs: Option<&[[f32; 2]]>,
        tangents: Option<&[[f32; 4]]>,
    ) -> wgpu::Buffer {
        let n_tris = indices.len() / 3;
        let mut verts: Vec<Vertex> = Vec::with_capacity(n_tris * 3);
        for tri in indices.chunks(3) {
            for &vi in tri {
                let vi = vi as usize;
                let uv = uvs.and_then(|u| u.get(vi)).copied().unwrap_or([0.0, 0.0]);
                let tangent = tangents
                    .and_then(|t| t.get(vi))
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                verts.push(Vertex {
                    position: positions.get(vi).copied().unwrap_or([0.0, 0.0, 0.0]),
                    normal: normals.get(vi).copied().unwrap_or([0.0, 1.0, 0.0]),
                    colour: [1.0, 1.0, 1.0, 1.0],
                    uv,
                    tangent,
                });
            }
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("face_vertex_buf"),
            size: (std::mem::size_of::<Vertex>() * verts.len().max(1)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = buf.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(&verts));
        }
        buf.unmap();
        buf
    }

    /// Expand N face scalar values to 3N by repeating each value three times.
    fn expand_face_scalars_to_3n(values: &[f32], n_tris: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n_tris * 3);
        for i in 0..n_tris {
            let v = values.get(i).copied().unwrap_or(0.0);
            out.push(v);
            out.push(v);
            out.push(v);
        }
        out
    }

    /// Expand N face RGBA colours to 3N by repeating each colour three times.
    fn expand_face_colours_to_3n(colours: &[[f32; 4]], n_tris: usize) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity(n_tris * 3);
        for i in 0..n_tris {
            let c = colours.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]);
            out.push(c);
            out.push(c);
            out.push(c);
        }
        out
    }

    /// Expand per-directed-edge scalars to per-vertex by averaging over incident edges.
    ///
    /// Edge ordering: `edge_values[3*t + k]` is the k-th edge of triangle `t`,
    /// running from vertex `k` to vertex `(k+1)%3` of that triangle.
    /// Each edge's value is added to both endpoint vertices; the final per-vertex
    /// value is the average over all incident edge contributions.
    fn expand_edge_to_vertex(
        edge_values: &[f32],
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Vec<f32> {
        let n = positions.len();
        let mut sum = vec![0.0f32; n];
        let mut count = vec![0u32; n];
        for (tri_idx, chunk) in indices.chunks(3).enumerate() {
            for k in 0..3 {
                let v = edge_values.get(3 * tri_idx + k).copied().unwrap_or(0.0);
                let vi0 = chunk[k] as usize;
                let vi1 = chunk[(k + 1) % 3] as usize;
                if vi0 < n {
                    sum[vi0] += v;
                    count[vi0] += 1;
                }
                if vi1 < n {
                    sum[vi1] += v;
                    count[vi1] += 1;
                }
            }
        }
        (0..n)
            .map(|i| {
                if count[i] > 0 {
                    sum[i] / count[i] as f32
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Expand per-cell (per-triangle) scalar values to per-vertex by averaging contributions.
    fn expand_cell_to_vertex(
        cell_values: &[f32],
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Vec<f32> {
        let n = positions.len();
        let mut sum = vec![0.0f32; n];
        let mut count = vec![0u32; n];
        for (tri_idx, chunk) in indices.chunks(3).enumerate() {
            let v = cell_values.get(tri_idx).copied().unwrap_or(0.0);
            for &vi in chunk {
                let vi = vi as usize;
                if vi < n {
                    sum[vi] += v;
                    count[vi] += 1;
                }
            }
        }
        (0..n)
            .map(|i| {
                if count[i] > 0 {
                    sum[i] / count[i] as f32
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compute per-vertex tangents using Gram-Schmidt orthogonalization with handedness.
    ///
    /// Returns a `Vec<[f32; 4]>` of length `positions.len()` where each element is
    /// `[tx, ty, tz, w]` with `w = ±1.0` encoding bitangent handedness.
    ///
    /// Requires triangulated indices (every 3 indices = one triangle).
    /// If any triangle is degenerate (zero-area or zero UV area), its contribution is skipped.
    fn compute_tangents(
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        uvs: &[[f32; 2]],
        indices: &[u32],
    ) -> Vec<[f32; 4]> {
        let n = positions.len();
        let tri_count = indices.len() / 3;

        // Accumulate sdir/tdir contributions per vertex.
        // Above 1024 triangles: parallel fold/reduce with thread-local arrays.
        // Below: sequential loop to avoid per-thread allocation overhead.
        let (tan1, tan2) = if tri_count >= 1024 {
            indices
                .par_chunks(3)
                .fold(
                    || (vec![[0.0f32; 3]; n], vec![[0.0f32; 3]; n]),
                    |(mut t1, mut t2), chunk| {
                        if chunk.len() < 3 {
                            return (t1, t2);
                        }
                        let i0 = chunk[0] as usize;
                        let i1 = chunk[1] as usize;
                        let i2 = chunk[2] as usize;

                        let p0 = positions[i0];
                        let p1 = positions[i1];
                        let p2 = positions[i2];
                        let uv0 = uvs[i0];
                        let uv1 = uvs[i1];
                        let uv2 = uvs[i2];

                        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                        let du1 = uv1[0] - uv0[0];
                        let dv1 = uv1[1] - uv0[1];
                        let du2 = uv2[0] - uv0[0];
                        let dv2 = uv2[1] - uv0[1];

                        let det = du1 * dv2 - du2 * dv1;
                        if det.abs() < 1e-10 {
                            return (t1, t2);
                        }
                        let r = 1.0 / det;

                        let sdir = [
                            (dv2 * e1[0] - dv1 * e2[0]) * r,
                            (dv2 * e1[1] - dv1 * e2[1]) * r,
                            (dv2 * e1[2] - dv1 * e2[2]) * r,
                        ];
                        let tdir = [
                            (du1 * e2[0] - du2 * e1[0]) * r,
                            (du1 * e2[1] - du2 * e1[1]) * r,
                            (du1 * e2[2] - du2 * e1[2]) * r,
                        ];

                        for &vi in &[i0, i1, i2] {
                            for k in 0..3 {
                                t1[vi][k] += sdir[k];
                                t2[vi][k] += tdir[k];
                            }
                        }
                        (t1, t2)
                    },
                )
                .reduce(
                    || (vec![[0.0f32; 3]; n], vec![[0.0f32; 3]; n]),
                    |(mut a1, mut a2), (b1, b2)| {
                        for i in 0..n {
                            for k in 0..3 {
                                a1[i][k] += b1[i][k];
                                a2[i][k] += b2[i][k];
                            }
                        }
                        (a1, a2)
                    },
                )
        } else {
            let mut tan1 = vec![[0.0f32; 3]; n];
            let mut tan2 = vec![[0.0f32; 3]; n];
            for t in 0..tri_count {
                let i0 = indices[t * 3] as usize;
                let i1 = indices[t * 3 + 1] as usize;
                let i2 = indices[t * 3 + 2] as usize;

                let p0 = positions[i0];
                let p1 = positions[i1];
                let p2 = positions[i2];
                let uv0 = uvs[i0];
                let uv1 = uvs[i1];
                let uv2 = uvs[i2];

                let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                let du1 = uv1[0] - uv0[0];
                let dv1 = uv1[1] - uv0[1];
                let du2 = uv2[0] - uv0[0];
                let dv2 = uv2[1] - uv0[1];

                let det = du1 * dv2 - du2 * dv1;
                if det.abs() < 1e-10 {
                    continue;
                }
                let r = 1.0 / det;

                let sdir = [
                    (dv2 * e1[0] - dv1 * e2[0]) * r,
                    (dv2 * e1[1] - dv1 * e2[1]) * r,
                    (dv2 * e1[2] - dv1 * e2[2]) * r,
                ];
                let tdir = [
                    (du1 * e2[0] - du2 * e1[0]) * r,
                    (du1 * e2[1] - du2 * e1[1]) * r,
                    (du1 * e2[2] - du2 * e1[2]) * r,
                ];

                for &vi in &[i0, i1, i2] {
                    for k in 0..3 {
                        tan1[vi][k] += sdir[k];
                        tan2[vi][k] += tdir[k];
                    }
                }
            }
            (tan1, tan2)
        };

        // Gram-Schmidt orthogonalization per vertex -- trivially parallel.
        (0..n)
            .into_par_iter()
            .map(|i| {
                let n_v = normals[i];
                let t = tan1[i];
                let dot = n_v[0] * t[0] + n_v[1] * t[1] + n_v[2] * t[2];
                let tx = t[0] - n_v[0] * dot;
                let ty = t[1] - n_v[1] * dot;
                let tz = t[2] - n_v[2] * dot;
                let len = (tx * tx + ty * ty + tz * tz).sqrt();
                let (tx, ty, tz) = if len > 1e-7 {
                    (tx / len, ty / len, tz / len)
                } else {
                    (1.0, 0.0, 0.0)
                };
                let cx = n_v[1] * tz - n_v[2] * ty;
                let cy = n_v[2] * tx - n_v[0] * tz;
                let cz = n_v[0] * ty - n_v[1] * tx;
                let w = if cx * tan2[i][0] + cy * tan2[i][1] + cz * tan2[i][2] < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                [tx, ty, tz, w]
            })
            .collect()
    }

    /// Validate mesh data before upload.
    fn validate_mesh_data(data: &MeshData) -> crate::error::ViewportResult<()> {
        if data.positions.is_empty() || data.indices.is_empty() {
            return Err(crate::error::ViewportError::EmptyMesh {
                positions: data.positions.len(),
                indices: data.indices.len(),
            });
        }
        if data.positions.len() != data.normals.len() {
            return Err(crate::error::ViewportError::MeshLengthMismatch {
                positions: data.positions.len(),
                normals: data.normals.len(),
            });
        }
        let vertex_count = data.positions.len();
        for &idx in &data.indices {
            if (idx as usize) >= vertex_count {
                return Err(crate::error::ViewportError::InvalidVertexIndex {
                    vertex_index: idx,
                    vertex_count,
                });
            }
        }
        Ok(())
    }

    /// Build per-vertex normal visualization lines from mesh data.
    fn build_normal_lines(data: &MeshData) -> Vec<Vertex> {
        let normal_colour = [0.627_f32, 0.769, 1.0, 1.0];
        let normal_length = 0.1_f32;
        let mut normal_line_verts: Vec<Vertex> = Vec::with_capacity(data.positions.len() * 2);
        for (p, n) in data.positions.iter().zip(data.normals.iter()) {
            let tip = [
                p[0] + n[0] * normal_length,
                p[1] + n[1] * normal_length,
                p[2] + n[2] * normal_length,
            ];
            normal_line_verts.push(Vertex {
                position: *p,
                normal: *n,
                colour: normal_colour,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
            normal_line_verts.push(Vertex {
                position: tip,
                normal: *n,
                colour: normal_colour,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
        normal_line_verts
    }

    pub(crate) fn create_mesh(
        device: &wgpu::Device,
        object_bgl: &wgpu::BindGroupLayout,
        fallback_albedo_view: &wgpu::TextureView,
        fallback_normal_view: &wgpu::TextureView,
        fallback_ao_view: &wgpu::TextureView,
        fallback_sampler: &wgpu::Sampler,
        lut_sampler: &wgpu::Sampler,
        fallback_lut_view: &wgpu::TextureView,
        fallback_scalar_buf: &wgpu::Buffer,
        fallback_matcap_view: &wgpu::TextureView,
        fallback_face_colour_buf: &wgpu::Buffer,
        fallback_warp_buf: &wgpu::Buffer,
        fallback_metallic_roughness_view: &wgpu::TextureView,
        fallback_emissive_view: &wgpu::TextureView,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> GpuMesh {
        Self::create_mesh_with_normals(
            device,
            object_bgl,
            fallback_albedo_view,
            fallback_normal_view,
            fallback_ao_view,
            fallback_sampler,
            lut_sampler,
            fallback_lut_view,
            fallback_scalar_buf,
            fallback_matcap_view,
            fallback_face_colour_buf,
            fallback_warp_buf,
            fallback_metallic_roughness_view,
            fallback_emissive_view,
            vertices,
            indices,
            None,
        )
    }

    pub(crate) fn create_mesh_with_normals(
        device: &wgpu::Device,
        object_bgl: &wgpu::BindGroupLayout,
        fallback_albedo_view: &wgpu::TextureView,
        fallback_normal_view: &wgpu::TextureView,
        fallback_ao_view: &wgpu::TextureView,
        fallback_sampler: &wgpu::Sampler,
        lut_sampler: &wgpu::Sampler,
        fallback_lut_view: &wgpu::TextureView,
        fallback_scalar_buf: &wgpu::Buffer,
        fallback_matcap_view: &wgpu::TextureView,
        fallback_face_colour_buf: &wgpu::Buffer,
        fallback_warp_buf: &wgpu::Buffer,
        fallback_metallic_roughness_view: &wgpu::TextureView,
        fallback_emissive_view: &wgpu::TextureView,
        vertices: &[Vertex],
        indices: &[u32],
        normal_line_verts: Option<&[Vertex]>,
    ) -> GpuMesh {
        use bytemuck::cast_slice;
        use wgpu;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex_buf"),
            size: (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index_buf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(indices));
        index_buffer.unmap();

        let edge_indices = generate_edge_indices(indices);
        let edge_buf_size = (std::mem::size_of::<u32>() * edge_indices.len().max(2)) as u64;
        let edge_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_index_buf"),
            size: edge_buf_size,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut mapped = edge_index_buffer.slice(..).get_mapped_range_mut();
            let edge_bytes = cast_slice::<u32, u8>(&edge_indices);
            mapped[..edge_bytes.len()].copy_from_slice(edge_bytes);
        }
        edge_index_buffer.unmap();

        let identity = glam::Mat4::IDENTITY.to_cols_array_2d();
        let object_uniform = ObjectUniform {
            model: identity,
            colour: [1.0, 1.0, 1.0, 1.0],
            selected: 0,
            wireframe: 0,
            ambient: 0.15,
            diffuse: 0.75,
            specular: 0.4,
            shininess: 32.0,
            has_texture: 0,
            use_pbr: 0,
            metallic: 0.0,
            roughness: 0.5,
            has_normal_map: 0,
            has_ao_map: 0,
            has_attribute: 0,
            scalar_min: 0.0,
            scalar_max: 1.0,
            _pad_scalar: 0,
            nan_colour: [0.0, 0.0, 0.0, 0.0],
            use_nan_colour: 0,
            use_matcap: 0,
            matcap_blendable: 0,
            unlit: 0,
            use_face_colour: 0,
            uv_vis_mode: 0,
            uv_vis_scale: 8.0,
            backface_policy: 0,
            backface_colour: [0.0; 4],
            has_warp: 0,
            warp_scale: 1.0,
            _pad_warp: [0; 2],
            emissive: [0.0; 3],
            _pad_emissive: 0,
            alpha_mode: 0,
            alpha_cutoff: 0.5,
            has_metallic_roughness_tex: 0,
            has_emissive_tex: 0,
        };
        let object_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_uniform_buf"),
            size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        object_uniform_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[object_uniform]));
        object_uniform_buf.unmap();

        let object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object_bind_group"),
            layout: object_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: object_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(fallback_albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(fallback_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(fallback_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(fallback_ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(fallback_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: fallback_scalar_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(fallback_matcap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: fallback_face_colour_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: fallback_warp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(fallback_metallic_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(fallback_emissive_view),
                },
            ],
        });

        let normal_override_uniform = ObjectUniform {
            model: identity,
            colour: [1.0, 1.0, 1.0, 1.0],
            selected: 0,
            wireframe: 0,
            ambient: 0.15,
            diffuse: 0.75,
            specular: 0.4,
            shininess: 32.0,
            has_texture: 0,
            use_pbr: 0,
            metallic: 0.0,
            roughness: 0.5,
            has_normal_map: 0,
            has_ao_map: 0,
            has_attribute: 0,
            scalar_min: 0.0,
            scalar_max: 1.0,
            _pad_scalar: 0,
            nan_colour: [0.0, 0.0, 0.0, 0.0],
            use_nan_colour: 0,
            use_matcap: 0,
            matcap_blendable: 0,
            unlit: 0,
            use_face_colour: 0,
            uv_vis_mode: 0,
            uv_vis_scale: 8.0,
            backface_policy: 0,
            backface_colour: [0.0; 4],
            has_warp: 0,
            warp_scale: 1.0,
            _pad_warp: [0; 2],
            emissive: [0.0; 3],
            _pad_emissive: 0,
            alpha_mode: 0,
            alpha_cutoff: 0.5,
            has_metallic_roughness_tex: 0,
            has_emissive_tex: 0,
        };
        let normal_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("normal_uniform_buf"),
            size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        normal_uniform_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[normal_override_uniform]));
        normal_uniform_buf.unmap();

        let normal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal_bind_group"),
            layout: object_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normal_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(fallback_albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(fallback_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(fallback_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(fallback_ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(fallback_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: fallback_scalar_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(fallback_matcap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: fallback_face_colour_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: fallback_warp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(fallback_metallic_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(fallback_emissive_view),
                },
            ],
        });

        let (normal_line_buffer, normal_line_count) = if let Some(nl_verts) = normal_line_verts {
            if nl_verts.is_empty() {
                (None, 0)
            } else {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("normal_line_buf"),
                    size: (std::mem::size_of::<Vertex>() * nl_verts.len()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buf.slice(..)
                    .get_mapped_range_mut()
                    .copy_from_slice(cast_slice(nl_verts));
                buf.unmap();
                let count = nl_verts.len() as u32;
                (Some(buf), count)
            }
        } else {
            (None, 0)
        };

        let aabb = crate::scene::aabb::Aabb::from_positions(
            &vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
        );

        GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            edge_index_buffer,
            edge_index_count: edge_indices.len() as u32,
            normal_line_buffer,
            normal_line_count,
            object_uniform_buf,
            object_bind_group,
            last_tex_key: (
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
            ),
            normal_uniform_buf,
            normal_bind_group,
            aabb,
            cpu_positions: None,
            cpu_indices: None,
            attribute_buffers: std::collections::HashMap::new(),
            attribute_ranges: std::collections::HashMap::new(),
            face_vertex_buffer: None,
            face_attribute_buffers: std::collections::HashMap::new(),
            face_colour_buffers: std::collections::HashMap::new(),
            vector_attribute_buffers: std::collections::HashMap::new(),
        }
    }

    // ---------------------------------------------------------------------------
    // Projected tetrahedra upload
    // ---------------------------------------------------------------------------

    /// Ensure the projected-tetrahedra bind group layout exists.
    ///
    /// No-op after the first call. Called internally by
    /// [`upload_projected_tet_mesh`](Self::upload_projected_tet_mesh) and
    /// [`ensure_pt_pipeline`](Self::ensure_pt_pipeline).
    pub(crate) fn ensure_pt_bind_group_layout(&mut self, device: &wgpu::Device) {
        if self.pt_bind_group_layout.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt_bgl"),
            entries: &[
                // binding 0: PT uniforms (density, scalar_min, scalar_max, _pad)
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
                // binding 1: tet storage buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: colourmap texture (256x1 D2, same format as all other LUT textures)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: colourmap sampler (linear clamp)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        self.pt_bind_group_layout = Some(bgl);
    }

    /// Decompose all cells in `data` into tetrahedra and upload to the GPU.
    ///
    /// `scalar_attribute` names a key in `data.cell_scalars`; cells without the attribute
    /// get scalar 0.0.  The scalar range is auto-detected from the data.
    ///
    /// Returns a [`ProjectedTetId`] that can be placed in a
    /// [`TransparentVolumeMeshItem`](crate::renderer::types::TransparentVolumeMeshItem)
    /// each frame.
    /// Upload a projected-tet mesh and return both the GPU handle and the actual scalar
    /// range stored in the GPU buffer. Callers should use the returned scalar range for
    /// threshold computations so that brimcast and the GPU always agree on the data range
    /// (including the constant-data `scalar_min + 1.0` adjustment in `decompose_into_chunks`).
    pub fn upload_projected_tet_mesh(
        &mut self,
        device: &wgpu::Device,
        data: &crate::resources::volume_mesh::VolumeMeshData,
        scalar_attribute: &str,
        colourmap_id: ColourmapId,
    ) -> crate::error::ViewportResult<(ProjectedTetId, f32, f32)> {
        self.ensure_pt_bind_group_layout(device);

        let (pending, scalar_range, uniform_buffer) =
            Self::decompose_into_chunks(device, data, scalar_attribute);

        // Build bind groups: one per chunk, all sharing the same uniform buffer + colourmap.
        let chunks = {
            let bgl = self
                .pt_bind_group_layout
                .as_ref()
                .expect("pt_bind_group_layout must exist after ensure_pt_bind_group_layout");
            let lut_view = self
                .colourmap_views
                .get(colourmap_id.0)
                .unwrap_or(&self.fallback_lut_view);
            let lut_sampler = &self.material_sampler;
            pending
                .into_iter()
                .map(|(tet_buffer, tet_count)| {
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pt_bind_group"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: tet_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(lut_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(lut_sampler),
                            },
                        ],
                    });
                    crate::resources::types::ProjectedTetChunk {
                        tet_buffer,
                        tet_count,
                        bind_group,
                    }
                })
                .collect::<Vec<_>>()
        };

        let id = ProjectedTetId(self.projected_tet_store.len());
        self.projected_tet_store.push(GpuProjectedTetMesh {
            chunks,
            uniform_buffer,
            scalar_range,
        });
        Ok((id, scalar_range.0, scalar_range.1))
    }

    /// Replace the tet buffer and colourmap for an existing projected-tet mesh in-place.
    ///
    /// Rebuilds the tet buffer with the new scalar attribute and recreates the bind
    /// group with the new colourmap LUT. The uniform buffer (density, scalar range) is
    /// updated to reflect the new scalar range; the existing GPU buffer is reused.
    pub fn replace_projected_tet_mesh(
        &mut self,
        device: &wgpu::Device,
        id: ProjectedTetId,
        data: &crate::resources::volume_mesh::VolumeMeshData,
        scalar_attribute: &str,
        colourmap_id: ColourmapId,
    ) -> crate::error::ViewportResult<()> {
        self.ensure_pt_bind_group_layout(device);

        let (pending, scalar_range, _new_uniform) =
            Self::decompose_into_chunks(device, data, scalar_attribute);

        // Build bind groups referencing the existing uniform buffer (reuse the GPU allocation).
        let chunks = {
            let bgl = self
                .pt_bind_group_layout
                .as_ref()
                .expect("pt_bind_group_layout must exist after ensure_pt_bind_group_layout");
            let lut_view = self
                .colourmap_views
                .get(colourmap_id.0)
                .unwrap_or(&self.fallback_lut_view);
            let lut_sampler = &self.material_sampler;
            let uniform_buf = &self.projected_tet_store[id.0].uniform_buffer;
            pending
                .into_iter()
                .map(|(tet_buffer, tet_count)| {
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pt_bind_group"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: tet_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(lut_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(lut_sampler),
                            },
                        ],
                    });
                    crate::resources::types::ProjectedTetChunk {
                        tet_buffer,
                        tet_count,
                        bind_group,
                    }
                })
                .collect::<Vec<_>>()
        };

        let slot = &mut self.projected_tet_store[id.0];
        slot.chunks = chunks;
        slot.scalar_range = scalar_range;

        Ok(())
    }

    /// Decompose `data` into device-limit-bounded tet buffers and a shared uniform buffer.
    ///
    /// Returns `(pending_chunks, scalar_range, uniform_buffer)` where each element of
    /// `pending_chunks` is a `(wgpu::Buffer, tet_count)` pair ready for bind group creation.
    /// Bind groups are created separately so callers can supply the correct uniform buffer
    /// reference (new for upload, existing for replace).
    fn decompose_into_chunks(
        device: &wgpu::Device,
        data: &crate::resources::volume_mesh::VolumeMeshData,
        scalar_attribute: &str,
    ) -> (Vec<(wgpu::Buffer, u32)>, (f32, f32), wgpu::Buffer) {
        // Determine the maximum tets per chunk from device limits.
        // Each tet is 64 bytes (4 x vec4<f32>).
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        let max_buf = device.limits().max_buffer_size;
        let chunk_size_tets = ((max_binding.min(max_buf)) / 64).max(1) as usize;

        let mut pending: Vec<(wgpu::Buffer, u32)> = Vec::new();
        let mut current_raw: Vec<f32> = Vec::with_capacity(chunk_size_tets * 16);
        let mut scalar_min = f32::INFINITY;
        let mut scalar_max = f32::NEG_INFINITY;

        let flush = |raw: &mut Vec<f32>, pending: &mut Vec<(wgpu::Buffer, u32)>| {
            let tet_count = (raw.len() / 16) as u32;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_tet_buffer"),
                size: (raw.len() * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            buf.slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(raw));
            buf.unmap();
            pending.push((buf, tet_count));
            raw.clear();
        };

        crate::resources::volume_mesh::for_each_tet(data, scalar_attribute, |verts, scalar| {
            scalar_min = scalar_min.min(scalar);
            scalar_max = scalar_max.max(scalar);
            current_raw.extend_from_slice(&[verts[0][0], verts[0][1], verts[0][2], scalar]);
            current_raw.extend_from_slice(&[verts[1][0], verts[1][1], verts[1][2], 0.0]);
            current_raw.extend_from_slice(&[verts[2][0], verts[2][1], verts[2][2], 0.0]);
            current_raw.extend_from_slice(&[verts[3][0], verts[3][1], verts[3][2], 0.0]);
            if current_raw.len() == chunk_size_tets * 16 {
                flush(&mut current_raw, &mut pending);
            }
        });

        if !current_raw.is_empty() {
            flush(&mut current_raw, &mut pending);
        }

        let scalar_range = if scalar_min.is_infinite() {
            (0.0f32, 1.0f32)
        } else {
            let max_s = if (scalar_max - scalar_min).abs() < 1e-12 {
                scalar_min + 1.0
            } else {
                scalar_max
            };
            (scalar_min, max_s)
        };

        let initial_uniform = crate::resources::types::ProjectedTetUniform {
            density: 1.0,
            scalar_min: scalar_range.0,
            scalar_max: scalar_range.1,
            threshold_min: f32::NEG_INFINITY,
            threshold_max: f32::INFINITY,
            _pad: 0.0,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt_uniform_buf"),
            size: std::mem::size_of::<crate::resources::types::ProjectedTetUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::bytes_of(&initial_uniform));
        uniform_buffer.unmap();

        (pending, scalar_range, uniform_buffer)
    }
}
