use super::*;

impl ViewportGpuResources {
    /// Create a GpuMesh from vertex/index slices and register it into the resource list.
    ///
    /// Returns the index of the new mesh in `self.meshes`.
    pub fn upload_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> usize {
        let mesh = Self::create_mesh(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.fallback_face_color_buf,
            vertices,
            indices,
        );
        self.mesh_store.insert(mesh).index()
    }

    /// Upload a `MeshData` (from the geometry primitives module) directly.
    ///
    /// Converts positions/normals/indices to the GPU `Vertex` layout (white color)
    /// and creates a normal visualization line buffer (light blue #a0c4ff, length 0.1).
    /// Returns the mesh index.
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
    ) -> crate::error::ViewportResult<usize> {
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
                    color: [1.0, 1.0, 1.0, 1.0],
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
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.fallback_face_color_buf,
            &vertices,
            &data.indices,
            Some(&normal_line_verts),
        );
        mesh.cpu_positions = Some(data.positions.clone());
        mesh.cpu_indices = Some(data.indices.clone());
        let (attr_bufs, attr_ranges, face_vbuf, face_attr_bufs, face_color_bufs) =
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
        mesh.face_color_buffers = face_color_bufs;
        let id = self.mesh_store.insert(mesh);
        tracing::debug!(
            mesh_index = id.index(),
            vertices = data.positions.len(),
            indices = data.indices.len(),
            "mesh uploaded"
        );
        Ok(id.index())
    }

    /// Replace the mesh at `mesh_index` with new geometry data.
    ///
    /// Used when primitive parameters change (re-tessellation).
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::MeshIndexOutOfBounds`](crate::error::ViewportError::MeshIndexOutOfBounds) if `mesh_index` is out of range,
    /// or any mesh validation error from the new data.
    pub fn replace_mesh_data(
        &mut self,
        device: &wgpu::Device,
        mesh_index: usize,
        data: &MeshData,
    ) -> crate::error::ViewportResult<()> {
        let mesh_id = crate::resources::mesh_store::MeshId(mesh_index);
        if !self.mesh_store.contains(mesh_id) {
            return Err(crate::error::ViewportError::MeshIndexOutOfBounds {
                index: mesh_index,
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
                    color: [1.0, 1.0, 1.0, 1.0],
                    uv,
                    tangent,
                }
            })
            .collect();
        let normal_line_verts = Self::build_normal_lines(data);
        let mut new_mesh = Self::create_mesh_with_normals(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.fallback_lut_view,
            &self.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.fallback_face_color_buf,
            &vertices,
            &data.indices,
            Some(&normal_line_verts),
        );
        new_mesh.cpu_positions = Some(data.positions.clone());
        new_mesh.cpu_indices = Some(data.indices.clone());
        let (attr_bufs, attr_ranges, face_vbuf, face_attr_bufs, face_color_bufs) =
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
        new_mesh.face_color_buffers = face_color_bufs;
        let _ = self.mesh_store.replace(mesh_id, new_mesh);
        tracing::debug!(
            mesh_index,
            vertices = data.positions.len(),
            indices = data.indices.len(),
            "mesh replaced"
        );
        Ok(())
    }

    /// Get a reference to the mesh at the given index, or `None` if the slot is empty/invalid.
    pub fn mesh(&self, index: usize) -> Option<&GpuMesh> {
        self.mesh_store
            .get(crate::resources::mesh_store::MeshId(index))
    }

    /// Total number of mesh slots (including empty/removed slots).
    pub fn mesh_slot_count(&self) -> usize {
        self.mesh_store.slot_count()
    }

    /// Remove a mesh, dropping its GPU buffers and freeing its slot for reuse.
    ///
    /// Returns `true` if a mesh was removed, `false` if the slot was already empty.
    pub fn remove_mesh(&mut self, index: usize) -> bool {
        self.mesh_store
            .remove(crate::resources::mesh_store::MeshId(index))
    }

    /// Upload an unstructured volume mesh by extracting its boundary surface and uploading
    /// the result via [`upload_mesh_data`](Self::upload_mesh_data).
    ///
    /// Interior faces (shared by two cells) are discarded; only boundary faces (belonging
    /// to exactly one cell) are kept. Per-cell scalar and color attributes are remapped to
    /// per-face attributes so the Phase 2 face-coloring path handles them automatically.
    ///
    /// Returns the mesh index, identical to what [`upload_mesh_data`](Self::upload_mesh_data)
    /// would return. Reference cell attributes via
    /// [`AttributeRef { kind: AttributeKind::Face, .. }`](crate::resources::AttributeRef).
    pub fn upload_volume_mesh_data(
        &mut self,
        device: &wgpu::Device,
        data: &crate::resources::volume_mesh::VolumeMeshData,
    ) -> crate::error::ViewportResult<usize> {
        let mesh_data = crate::resources::volume_mesh::extract_boundary_faces(data);
        self.upload_mesh_data(device, &mesh_data)
    }

    /// Upload per-vertex, per-cell, per-face scalar, and per-face color attributes to GPU buffers.
    ///
    /// Returns `(attribute_buffers, attribute_ranges, face_vertex_buffer, face_attribute_buffers,
    /// face_color_buffers)`.
    ///
    /// - `attribute_buffers`: per-vertex storage buffers for `Vertex` and `Cell` kinds.
    /// - `attribute_ranges`: `(min, max)` per attribute name (all scalar kinds).
    /// - `face_vertex_buffer`: non-indexed 3N-vertex buffer (built once if any `Face`/`FaceColor` attr exists).
    /// - `face_attribute_buffers`: per-face scalar storage buffers (3N `f32` entries, replicated).
    /// - `face_color_buffers`: per-face color storage buffers (3N `[f32;4]` entries, replicated).
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
    ) {
        let mut bufs = std::collections::HashMap::new();
        let mut ranges = std::collections::HashMap::new();
        let mut face_attr_bufs: std::collections::HashMap<String, wgpu::Buffer> =
            std::collections::HashMap::new();
        let mut face_color_bufs: std::collections::HashMap<String, wgpu::Buffer> =
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
                    // Build the shared face vertex buffer on first Face/FaceColor attribute.
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
                AttributeData::FaceColor(colors) => {
                    // Build the shared face vertex buffer on first Face/FaceColor attribute.
                    if face_vbuf.is_none() {
                        face_vbuf = Some(Self::build_face_vertex_buffer(
                            device, positions, normals, indices, uvs, tangents,
                        ));
                    }
                    let expanded = Self::expand_face_colors_to_3n(colors, n_tris);
                    if expanded.is_empty() {
                        continue;
                    }
                    let byte_len = std::mem::size_of::<[f32; 4]>() * expanded.len();
                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("face_color_{name}")),
                        size: byte_len as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    {
                        let mut view = buf.slice(..).get_mapped_range_mut();
                        view.copy_from_slice(bytemuck::cast_slice(&expanded));
                    }
                    buf.unmap();
                    face_color_bufs.insert(name.clone(), buf);
                }
            }
        }
        (bufs, ranges, face_vbuf, face_attr_bufs, face_color_bufs)
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
                    color: [1.0, 1.0, 1.0, 1.0],
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

    /// Expand N face RGBA colors to 3N by repeating each color three times.
    fn expand_face_colors_to_3n(colors: &[[f32; 4]], n_tris: usize) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity(n_tris * 3);
        for i in 0..n_tris {
            let c = colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]);
            out.push(c);
            out.push(c);
            out.push(c);
        }
        out
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
        let mut tan1 = vec![[0.0f32; 3]; n];
        let mut tan2 = vec![[0.0f32; 3]; n];

        let tri_count = indices.len() / 3;
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
                }
                for k in 0..3 {
                    tan2[vi][k] += tdir[k];
                }
            }
        }

        (0..n)
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
        let normal_color = [0.627_f32, 0.769, 1.0, 1.0];
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
                color: normal_color,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
            normal_line_verts.push(Vertex {
                position: tip,
                normal: *n,
                color: normal_color,
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
        fallback_lut_view: &wgpu::TextureView,
        fallback_scalar_buf: &wgpu::Buffer,
        fallback_matcap_view: &wgpu::TextureView,
        fallback_face_color_buf: &wgpu::Buffer,
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
            fallback_lut_view,
            fallback_scalar_buf,
            fallback_matcap_view,
            fallback_face_color_buf,
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
        fallback_lut_view: &wgpu::TextureView,
        fallback_scalar_buf: &wgpu::Buffer,
        fallback_matcap_view: &wgpu::TextureView,
        fallback_face_color_buf: &wgpu::Buffer,
        vertices: &[Vertex],
        indices: &[u32],
        normal_line_verts: Option<&[Vertex]>,
    ) -> GpuMesh {
        use bytemuck::cast_slice;
        use wgpu;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex_buf"),
            size: (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
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
            color: [1.0, 1.0, 1.0, 1.0],
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
            nan_color: [0.0, 0.0, 0.0, 0.0],
            use_nan_color: 0,
            use_matcap: 0,
            matcap_blendable: 0,
            _pad2: 0,
            use_face_color: 0,
            uv_vis_mode: 0,
            uv_vis_scale: 8.0,
            backface_policy: 0,
            backface_color: [0.0; 4],
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
                    resource: fallback_face_color_buf.as_entire_binding(),
                },
            ],
        });

        let normal_override_uniform = ObjectUniform {
            model: identity,
            color: [1.0, 1.0, 1.0, 1.0],
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
            nan_color: [0.0, 0.0, 0.0, 0.0],
            use_nan_color: 0,
            use_matcap: 0,
            matcap_blendable: 0,
            _pad2: 0,
            use_face_color: 0,
            uv_vis_mode: 0,
            uv_vis_scale: 8.0,
            backface_policy: 0,
            backface_color: [0.0; 4],
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
                    resource: fallback_face_color_buf.as_entire_binding(),
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
            last_tex_key: (u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            normal_uniform_buf,
            normal_bind_group,
            aabb,
            cpu_positions: None,
            cpu_indices: None,
            attribute_buffers: std::collections::HashMap::new(),
            attribute_ranges: std::collections::HashMap::new(),
            face_vertex_buffer: None,
            face_attribute_buffers: std::collections::HashMap::new(),
            face_color_buffers: std::collections::HashMap::new(),
        }
    }
}
