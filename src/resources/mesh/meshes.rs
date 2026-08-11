use crate::resources::*;

/// Slice size for the chunked async mesh buffer fills, in bytes. The copies
/// are plain memcpys into mapped buffers, so the size only has to fit a
/// sub-millisecond frame budget at memcpy bandwidth while amortising
/// per-slice overhead.
const MESH_CHUNK_BYTES: usize = 4 << 20;

/// Bytes per override-buffer element: one tightly packed `vec3<f32>`.
const OVERRIDE_ELEMENT_BYTES: u64 = 12;

/// Element window into a position/normal override buffer, for meshes that
/// read out of a pooled buffer shared with other meshes.
///
/// Units are vec3 elements (12 bytes each), not bytes: `base_element` is the
/// first element this mesh reads and `element_count` how many are readable
/// from there. Because the window is applied by the shader rather than a
/// buffer binding offset, `base_element` needs no alignment.
///
/// Used with `set_position_override_buffer_sliced` and
/// `set_normal_override_buffer_sliced`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct OverrideBufferSlice {
    /// Index of the first vec3 element this mesh reads.
    pub base_element: u32,
    /// Number of vec3 elements readable starting at `base_element`.
    pub element_count: u32,
}

impl OverrideBufferSlice {
    /// Window of `element_count` vec3 elements starting at `base_element`.
    pub fn new(base_element: u32, element_count: u32) -> Self {
        Self {
            base_element,
            element_count,
        }
    }
}

/// CPU-prepared vertex stream and ancillary buffers needed to finish a mesh
/// upload on the main thread.
///
/// Produced by `DeviceResources::prep_mesh_data` and consumed by
/// `DeviceResources::assemble_mesh_data`. Sits between the worker
/// thread and the apply step of `begin_upload_mesh_data`.
pub(crate) struct MeshPrep {
    /// Interleaved GPU vertex stream (position + normal + uv + tangent +
    /// colour) ready for upload as `Vec<Vertex>`.
    pub vertices: Vec<Vertex>,
    /// Tangents computed from positions, normals, UVs, and indices when the
    /// source `MeshData` did not carry its own tangents. `None` means the
    /// source tangents (if any) should be used directly.
    pub computed_tangents: Option<Vec<[f32; 4]>>,
    /// CPU copies retained on the mesh for picking. Cloned here rather
    /// than in `assemble_mesh_data` so the async path pays the memcpy on
    /// the worker thread instead of inside the apply step.
    pub cpu_positions: Vec<[f32; 3]>,
    pub cpu_normals: Vec<[f32; 3]>,
    pub cpu_indices: Vec<u32>,
}

impl DeviceResources {
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
    /// # fn demo(resources: &mut viewport_lib::resources::DeviceResources, device: &viewport_lib::wgpu::Device) {
    /// let result = resources.upload_mesh(device, &[], &[]);
    /// assert!(matches!(result, Err(ViewportError::EmptyMesh { .. })));
    /// # }
    /// ```
    pub fn upload_mesh(
        &mut self,
        device: &crate::gpu::Device,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> crate::error::ViewportResult<crate::resources::mesh::mesh_store::MeshId> {
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
            &self.fallback_texture_array_view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.lut_sampler,
            &self.content.fallback_lut_view,
            &self.content.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.content.fallback_face_colour_buf,
            &self.content.fallback_warp_buf,
            &self.content.fallback_position_override_buf,
            &self.content.fallback_normal_override_buf,
            &self.content.fallback_extension_attr_buf,
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
        device: &crate::gpu::Device,
        data: &MeshData,
    ) -> crate::error::ViewportResult<crate::resources::mesh::mesh_store::MeshId> {
        Self::validate_mesh_data(data)?;
        Self::validate_mesh_size(device, data)?;
        let prep = Self::prep_mesh_data(data);
        Ok(self.assemble_mesh_data(device, data, prep))
    }

    /// Refuse meshes whose vertex or index buffer would exceed the device's
    /// `max_buffer_size`: creating such a buffer raises a validation error
    /// that takes the whole device down.
    fn validate_mesh_size(
        device: &crate::gpu::Device,
        data: &MeshData,
    ) -> crate::error::ViewportResult<()> {
        let max = device.limits().max_buffer_size;
        let vertex_bytes = (data.positions.len() * std::mem::size_of::<Vertex>()) as u64;
        let index_bytes = (data.indices.len() * std::mem::size_of::<u32>()) as u64;
        let bytes = vertex_bytes.max(index_bytes);
        if bytes > max {
            return Err(crate::error::ViewportError::MeshTooLarge { bytes, max });
        }
        Ok(())
    }

    /// CPU-side preparation that converts a `MeshData` into the vertex
    /// stream, normal-line visualization vertices, and any tangents the
    /// shader needs.
    ///
    /// Split out so it can run on a worker thread for
    /// `begin_upload_mesh_data`. Returns owned buffers; the caller hands
    /// them to `assemble_mesh_data` on the main thread to finish the
    /// upload.
    pub(crate) fn prep_mesh_data(data: &MeshData) -> MeshPrep {
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
                let colour = data
                    .vertex_colours
                    .as_ref()
                    .and_then(|c| c.get(i))
                    .copied()
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                Vertex {
                    position: *p,
                    normal: *n,
                    colour,
                    uv,
                    tangent,
                }
            })
            .collect();

        MeshPrep {
            vertices,
            computed_tangents,
            cpu_positions: data.positions.clone(),
            cpu_normals: data.normals.clone(),
            cpu_indices: data.indices.clone(),
        }
    }

    /// Main-thread half of `upload_mesh_data`: takes the prep buffers,
    /// creates GPU buffers and bind groups, inserts the mesh into the
    /// store, and returns the new id.
    pub(crate) fn assemble_mesh_data(
        &mut self,
        device: &crate::gpu::Device,
        data: &MeshData,
        prep: MeshPrep,
    ) -> crate::resources::mesh::mesh_store::MeshId {
        let MeshPrep {
            vertices,
            computed_tangents,
            cpu_positions,
            cpu_normals,
            cpu_indices,
        } = prep;
        let tangent_slice = data.tangents.as_deref().or(computed_tangents.as_deref());

        let mut mesh = Self::create_mesh_with_normals(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_texture_array_view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.lut_sampler,
            &self.content.fallback_lut_view,
            &self.content.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.content.fallback_face_colour_buf,
            &self.content.fallback_warp_buf,
            &self.content.fallback_position_override_buf,
            &self.content.fallback_normal_override_buf,
            &self.content.fallback_extension_attr_buf,
            &self.fallback_metallic_roughness_texture_view,
            &self.fallback_emissive_texture_view,
            &vertices,
            &data.indices,
            None,
        );
        mesh.cpu_positions = Some(cpu_positions);
        mesh.cpu_normals = Some(cpu_normals);
        mesh.cpu_indices = Some(cpu_indices);
        mesh.submeshes = data.submeshes.clone();
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
        mesh.extension_attr_buffer =
            Self::upload_extension_attributes(device, data, vertices.len());
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
        id
    }

    /// Start an asynchronous mesh upload.
    ///
    /// Returns immediately with a `JobId`. The CPU prep (tangent
    /// computation, vertex repack, pick-data copies) runs on a worker
    /// thread; the GPU buffers are then filled on the main thread during
    /// `process_uploads` calls, sliced across frames under the upload
    /// budget when one is set, and the mesh record is assembled once the
    /// last slice lands. Once the status is `Ready`, call
    /// `upload_result_mesh` to take the resulting `MeshId`.
    ///
    /// Ownership of `data` transfers into the worker. To upload a mesh
    /// without giving up ownership, clone the `MeshData` at the call site.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as `upload_mesh_data` (empty
    /// mesh, length mismatch, invalid vertex index) before any job is
    /// submitted.
    pub fn begin_upload_mesh_data(
        &mut self,
        device: &crate::gpu::Device,
        data: MeshData,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        Self::validate_mesh_data(&data)?;
        Self::validate_mesh_size(device, &data)?;

        let slot =
            crate::resources::ResultSlot::<crate::resources::mesh::mesh_store::MeshId>::new();
        let slot_for_apply = slot.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu_then_gpu_chunked(move |progress| {
                progress.set(0.1);
                let prep = DeviceResources::prep_mesh_data(&data);
                let aabb = crate::scene::aabb::Aabb::from_positions(&data.positions);
                progress.set(0.5);

                // GPU stage: create the two buffers mapped, memcpy the
                // vertex and index bytes in MESH_CHUNK_BYTES slices as the
                // frame budget allows (at least one slice per turn), unmap,
                // and hand the filled buffers to the apply step. The typed
                // result is published only after the apply, so no partially
                // filled mesh is ever observable.
                let MeshPrep {
                    vertices,
                    computed_tangents,
                    cpu_positions,
                    cpu_normals,
                    cpu_indices,
                } = prep;
                let mut bufs: Option<(crate::gpu::Buffer, crate::gpu::Buffer)> = None;
                let mut voff: usize = 0;
                let mut ioff: usize = 0;
                // One-shot payload for the apply closure; an FnMut cannot
                // move out of its captures, so the final turn takes it.
                let mut payload = Some((
                    data,
                    computed_tangents,
                    cpu_positions,
                    cpu_normals,
                    cpu_indices,
                    aabb,
                    slot_for_apply,
                ));
                Ok(Box::new(
                    move |dev: &crate::gpu::Device,
                          _q: &crate::gpu::Queue,
                          progress: &crate::resources::ProgressHandle,
                          budget: &crate::resources::upload_jobs::FrameBudget| {
                        use bytemuck::cast_slice;
                        let vsrc: &[u8] = cast_slice(&vertices);
                        let indices_bytes = payload
                            .as_ref()
                            .expect("payload present until done")
                            .0
                            .indices
                            .len()
                            * std::mem::size_of::<u32>();
                        let (vbuf, ibuf) = bufs.get_or_insert_with(|| {
                            let vbuf = dev.create_buffer(&crate::gpu::BufferDescriptor {
                                label: Some("vertex_buf"),
                                size: vsrc.len() as u64,
                                usage: crate::gpu::BufferUsages::VERTEX
                                    | crate::gpu::BufferUsages::COPY_DST
                                    | crate::gpu::BufferUsages::STORAGE,
                                mapped_at_creation: true,
                            });
                            let ibuf = dev.create_buffer(&crate::gpu::BufferDescriptor {
                                label: Some("index_buf"),
                                size: indices_bytes as u64,
                                usage: crate::gpu::BufferUsages::INDEX
                                    | crate::gpu::BufferUsages::COPY_DST
                                    | crate::gpu::BufferUsages::STORAGE,
                                mapped_at_creation: true,
                            });
                            (vbuf, ibuf)
                        });
                        let total = vsrc.len() + indices_bytes;
                        loop {
                            if voff < vsrc.len() {
                                let end = (voff + MESH_CHUNK_BYTES).min(vsrc.len());
                                crate::resources::builders::write_mapped(
                                    vbuf.slice(voff as u64..end as u64),
                                    &vsrc[voff..end],
                                );
                                voff = end;
                            } else if ioff < indices_bytes {
                                let isrc: &[u8] = cast_slice(
                                    &payload.as_ref().expect("payload present").0.indices,
                                );
                                let end = (ioff + MESH_CHUNK_BYTES).min(isrc.len());
                                crate::resources::builders::write_mapped(
                                    ibuf.slice(ioff as u64..end as u64),
                                    &isrc[ioff..end],
                                );
                                ioff = end;
                            } else {
                                break;
                            }
                            progress.set(0.5 + 0.45 * ((voff + ioff) as f32 / total as f32));
                            if budget.exhausted() {
                                break;
                            }
                        }
                        if voff < vsrc.len() || ioff < indices_bytes {
                            return Ok(crate::resources::upload_jobs::GpuStep::Continue);
                        }

                        let (vbuf, ibuf) = bufs.take().expect("buffers created on first turn");
                        vbuf.unmap();
                        ibuf.unmap();
                        let (
                            data,
                            computed_tangents,
                            cpu_positions,
                            cpu_normals,
                            cpu_indices,
                            aabb,
                            slot,
                        ) = payload.take().expect("done reached once");
                        let device = dev.clone();
                        let upload_bytes = total as u64;
                        let apply = Box::new(move |resources: &mut DeviceResources| {
                            let tangent_slice =
                                data.tangents.as_deref().or(computed_tangents.as_deref());
                            resources.frame_upload_bytes += upload_bytes;
                            let mut mesh = DeviceResources::create_mesh_from_buffers(
                                &device,
                                &resources.object_bind_group_layout,
                                &resources.fallback_texture.view,
                                &resources.fallback_texture_array_view,
                                &resources.fallback_normal_map_view,
                                &resources.fallback_ao_map_view,
                                &resources.material_sampler,
                                &resources.lut_sampler,
                                &resources.content.fallback_lut_view,
                                &resources.content.fallback_scalar_buf,
                                &resources.fallback_texture.view,
                                &resources.content.fallback_face_colour_buf,
                                &resources.content.fallback_warp_buf,
                                &resources.content.fallback_position_override_buf,
                                &resources.content.fallback_normal_override_buf,
                                &resources.content.fallback_extension_attr_buf,
                                &resources.fallback_metallic_roughness_texture_view,
                                &resources.fallback_emissive_texture_view,
                                vbuf,
                                ibuf,
                                data.indices.len() as u32,
                                aabb,
                            );
                            mesh.cpu_positions = Some(cpu_positions);
                            mesh.cpu_normals = Some(cpu_normals);
                            mesh.cpu_indices = Some(cpu_indices);
                            mesh.submeshes = data.submeshes.clone();
                            let (
                                attr_bufs,
                                attr_ranges,
                                face_vbuf,
                                face_attr_bufs,
                                face_colour_bufs,
                                vector_attr_bufs,
                            ) = DeviceResources::upload_attributes(
                                &device,
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
                            mesh.extension_attr_buffer =
                                DeviceResources::upload_extension_attributes(
                                    &device,
                                    &data,
                                    data.positions.len(),
                                );
                            let mesh_id = resources.mesh_store.insert(mesh);
                            tracing::debug!(
                                mesh_index = mesh_id.index(),
                                vertices = data.positions.len(),
                                indices = data.indices.len(),
                                "mesh uploaded"
                            );
                            slot.set(mesh_id);
                        })
                            as crate::resources::upload_jobs::ApplyFn;
                        progress.set(0.95);
                        Ok(crate::resources::upload_jobs::GpuStep::Done(
                            crate::resources::upload_jobs::JobProduct::with_apply(apply),
                        ))
                    },
                )
                    as crate::resources::upload_jobs::ChunkedGpuWorkFn)
            })
        };

        self.job_results
            .mesh
            .lock()
            .expect("mesh result map poisoned")
            .insert(id, slot);
        Ok(id)
    }

    /// Take the `MeshId` produced by a completed `begin_upload_mesh_data`
    /// job.
    ///
    /// Returns `JobNotReady` while the upload is still in flight, and
    /// `JobResultMissing` for ids that have already been taken, were
    /// issued by a different upload type, or never existed.
    pub fn upload_result_mesh(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::mesh::mesh_store::MeshId> {
        let mut map = self
            .job_results
            .mesh
            .lock()
            .expect("mesh result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(mesh_id) => {
                map.remove(&id);
                Ok(mesh_id)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
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
        device: &crate::gpu::Device,
        data: &MeshData,
    ) -> crate::error::ViewportResult<crate::resources::mesh::mesh_store::MeshId> {
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
    pub fn set_pickable(
        &mut self,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        pickable: bool,
    ) {
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
    /// Mutually exclusive per frame with
    /// [`set_position_override_buffer`](Self::set_position_override_buffer) and
    /// [`set_normal_override_buffer`](Self::set_normal_override_buffer) for the
    /// same mesh: the two write paths race. A debug assertion fires if both
    /// are active. To switch from GPU-compute deformation back to CPU writes,
    /// call [`clear_position_override`](Self::clear_position_override) /
    /// [`clear_normal_override`](Self::clear_normal_override) first.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is out of range, [`ViewportError::MeshLengthMismatch`](crate::error::ViewportError::MeshLengthMismatch)
    /// if `positions` and `normals` differ in length or do not match the existing vertex count.
    pub fn write_mesh_positions_normals(
        &mut self,
        queue: &crate::gpu::Queue,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
    ) -> crate::error::ViewportResult<()> {
        use bytemuck::cast_slice;

        if !self.mesh_store.contains(mesh_id) {
            return Err(crate::error::ViewportError::StaleHandle {
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
            debug_assert!(
                mesh.position_override_buffer.is_none() && mesh.normal_override_buffer.is_none(),
                "write_mesh_positions_normals called on mesh {} that has a GPU position/normal override bound. The CPU write and the GPU override race; call clear_position_override / clear_normal_override first.",
                mesh_id.index(),
            );
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
        // The normal-line sidecar is built lazily; refresh it only when a
        // normals view has already materialised it.
        let normal_line_verts: Option<Vec<Vertex>> =
            has_normal_lines.then(|| Self::build_normal_lines(positions, normals));

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
        if let Some(ref mut cn) = mesh.cpu_normals {
            *cn = normals.to_vec();
        }
        mesh.content_rev += 1;

        self.frame_upload_bytes += (vertices.len() * std::mem::size_of::<Vertex>()) as u64;
        if let Some(ref nl) = normal_line_verts {
            self.frame_upload_bytes += (nl.len() * std::mem::size_of::<Vertex>()) as u64;
        }

        Ok(())
    }

    /// Bind a GPU storage buffer of per-vertex positions to `mesh_id`. The
    /// standard mesh and skinned-mesh pipelines read positions from this
    /// buffer instead of the vertex buffer's position attribute on every frame
    /// the binding is present.
    ///
    /// Intended consumer: a [`GpuPlugin`](crate::runtime::GpuPlugin) that
    /// computes deformed positions on the GPU in `pre_prepare` (cloth, hair,
    /// GPU particles, audio-reactive displacement). The override path
    /// sidesteps the CPU round-trip that
    /// [`write_mesh_positions_normals`](Self::write_mesh_positions_normals)
    /// requires.
    ///
    /// The buffer must:
    ///   - have [`wgpu::BufferUsages::STORAGE`],
    ///   - hold at least 3 `f32` per vertex (12 bytes each), in flat
    ///     `[x, y, z, x, y, z, ...]` order. This matches the warp-attribute
    ///     buffer layout and avoids WGSL's 16-byte vec3 stride padding, so a
    ///     consumer compute shader can write tight `vec3` data directly.
    ///
    /// The shader bounds-checks `arrayLength` before reading, so a smaller
    /// buffer falls back to `in.position` for out-of-range vertex indices.
    ///
    /// Override and skinning compose: when both are active, the override
    /// replaces the bind-pose input and skinning is then applied on top.
    ///
    /// Do not call this in the same frame as `write_mesh_positions_normals`
    /// for the same mesh; the two write paths race and the result is
    /// undefined. Pick one source for positions per frame.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered.
    pub fn set_position_override_buffer(
        &mut self,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        buffer: crate::gpu::Buffer,
    ) -> crate::error::ViewportResult<()> {
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        mesh.position_override_buffer = Some(buffer);
        mesh.position_override_slice = None;
        // Bump only the gen counter; don't touch `last_tex_key.9` here. The
        // bind-group rebuild path reads `position_override_gen` into the new
        // key and compares against `last_tex_key`; the mismatch is what
        // triggers the rebuild that actually swaps the fallback binding for
        // this buffer.
        mesh.position_override_gen = mesh.position_override_gen.wrapping_add(1);
        Ok(())
    }

    /// Like [`set_position_override_buffer`](Self::set_position_override_buffer)
    /// but reads a window of `buffer` instead of the whole thing, so several
    /// meshes can share one pooled buffer (for example a physics solver that
    /// keeps every body's particles in a single allocation).
    ///
    /// `slice` is in vec3 elements, 12 bytes each: the mesh's vertex `i` reads
    /// elements `slice.base_element + i`. There is no alignment requirement on
    /// `base_element`; the window is applied by the shader, not by a buffer
    /// binding offset. Vertices at or past `slice.element_count` fall back to
    /// the vertex buffer's position attribute.
    ///
    /// If the pool grows and the consumer reallocates it (a new
    /// `wgpu::Buffer` identity), call this again with the new buffer; the
    /// renderer rebinds on the next frame. Holding the old handle just renders
    /// the last data written to the old allocation.
    ///
    /// All items drawing this `mesh_id` read the same window: the slice is
    /// per-mesh state, not per-item.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered, or
    /// [`ViewportError::OverrideSliceOutOfRange`](crate::error::ViewportError::OverrideSliceOutOfRange)
    /// if the slice does not fit in `buffer`.
    pub fn set_position_override_buffer_sliced(
        &mut self,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        buffer: crate::gpu::Buffer,
        slice: OverrideBufferSlice,
    ) -> crate::error::ViewportResult<()> {
        Self::validate_override_slice(&buffer, slice)?;
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        mesh.position_override_buffer = Some(buffer);
        mesh.position_override_slice = Some(slice);
        // See `set_position_override_buffer` for why only the gen moves.
        mesh.position_override_gen = mesh.position_override_gen.wrapping_add(1);
        Ok(())
    }

    /// Same idea as [`set_position_override_buffer`](Self::set_position_override_buffer)
    /// but for per-vertex normals (bound at group 1 binding 14).
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered.
    pub fn set_normal_override_buffer(
        &mut self,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        buffer: crate::gpu::Buffer,
    ) -> crate::error::ViewportResult<()> {
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        mesh.normal_override_buffer = Some(buffer);
        mesh.normal_override_slice = None;
        // See `set_position_override_buffer` for why this only bumps the gen
        // counter and not `last_tex_key.10`.
        mesh.normal_override_gen = mesh.normal_override_gen.wrapping_add(1);
        Ok(())
    }

    /// Sliced variant of [`set_normal_override_buffer`](Self::set_normal_override_buffer);
    /// see [`set_position_override_buffer_sliced`](Self::set_position_override_buffer_sliced)
    /// for the windowing contract.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered, or
    /// [`ViewportError::OverrideSliceOutOfRange`](crate::error::ViewportError::OverrideSliceOutOfRange)
    /// if the slice does not fit in `buffer`.
    pub fn set_normal_override_buffer_sliced(
        &mut self,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        buffer: crate::gpu::Buffer,
        slice: OverrideBufferSlice,
    ) -> crate::error::ViewportResult<()> {
        Self::validate_override_slice(&buffer, slice)?;
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        mesh.normal_override_buffer = Some(buffer);
        mesh.normal_override_slice = Some(slice);
        mesh.normal_override_gen = mesh.normal_override_gen.wrapping_add(1);
        Ok(())
    }

    /// Shared slice-fits-in-buffer check for the sliced override setters.
    fn validate_override_slice(
        buffer: &crate::gpu::Buffer,
        slice: OverrideBufferSlice,
    ) -> crate::error::ViewportResult<()> {
        let needed =
            (slice.base_element as u64 + slice.element_count as u64) * OVERRIDE_ELEMENT_BYTES;
        if needed > buffer.size() {
            return Err(crate::error::ViewportError::OverrideSliceOutOfRange {
                base_element: slice.base_element,
                element_count: slice.element_count,
                buffer_elements: buffer.size() / OVERRIDE_ELEMENT_BYTES,
            });
        }
        Ok(())
    }

    /// Revert the position source to the mesh's vertex buffer attribute.
    /// Drops the override buffer handle; if no other owner holds it, wgpu
    /// frees it after the in-flight frames complete.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered.
    pub fn clear_position_override(
        &mut self,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
    ) -> crate::error::ViewportResult<()> {
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        mesh.position_override_buffer = None;
        mesh.position_override_slice = None;
        mesh.position_override_gen = mesh.position_override_gen.wrapping_add(1);
        Ok(())
    }

    /// Revert the normal source to the mesh's vertex buffer attribute.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered.
    pub fn clear_normal_override(
        &mut self,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
    ) -> crate::error::ViewportResult<()> {
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        mesh.normal_override_buffer = None;
        mesh.normal_override_slice = None;
        mesh.normal_override_gen = mesh.normal_override_gen.wrapping_add(1);
        Ok(())
    }

    /// Attach a baked lightmap to a mesh.
    ///
    /// `uv1` is the second UV set, one `Vec2` per vertex (zero-padded or
    /// truncated to the mesh's vertex count). Unlike UV0, it must be a unique,
    /// non-overlapping unwrap in `[0, 1]` so each surface point maps to its own
    /// lightmap texel: overlapping UV0 tiling would make two surfaces share a
    /// baked colour. Producing that unwrap and the texture is a baking-pipeline
    /// concern; this call only consumes the result.
    ///
    /// `data` names the baked texture and `mode` how it combines with the
    /// shader's ambient term (see [`LightmapMode`](crate::resources::LightmapMode)).
    /// The texture is sampled in the lit fragment shader on the non-instanced
    /// mesh paths (`mesh`, `mesh_oit`). Call again to change the lightmap;
    /// [`clear_lightmap`](Self::clear_lightmap) removes it.
    ///
    /// The `TextureId` in `data` must stay uploaded for as long as the lightmap
    /// is set. Removing the mesh clears the registration.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered.
    pub fn set_lightmap(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        uv1: &[glam::Vec2],
        data: crate::resources::lightmap::LightmapData,
        mode: crate::resources::lightmap::LightmapMode,
    ) -> crate::error::ViewportResult<()> {
        self.set_lightmap_impl(device, mesh_id, uv1, None, data, mode)
    }

    /// Attach a multi-page baked lightmap to a mesh.
    ///
    /// Like [`set_lightmap`](Self::set_lightmap), but for an unwrap whose charts
    /// spilled across several atlas pages. `pages` is one atlas-page (array-layer)
    /// index per vertex, parallel to `uv1`; every vertex of a chart shares one
    /// page. `data` must name a texture-array lightmap uploaded with
    /// [`upload_texture_hdr_layers`](crate::resources::ViewportGpuResources::upload_texture_hdr_layers)
    /// whose layer count covers the largest page index. A single-page lightmap
    /// (all pages `0`) is identical to `set_lightmap`.
    ///
    /// `pages` is zero-padded / truncated to the vertex count like `uv1`.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered.
    pub fn set_lightmap_paged(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        uv1: &[glam::Vec2],
        pages: &[u32],
        data: crate::resources::lightmap::LightmapData,
        mode: crate::resources::lightmap::LightmapMode,
    ) -> crate::error::ViewportResult<()> {
        self.set_lightmap_impl(device, mesh_id, uv1, Some(pages), data, mode)
    }

    fn set_lightmap_impl(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        uv1: &[glam::Vec2],
        pages: Option<&[u32]>,
        data: crate::resources::lightmap::LightmapData,
        mode: crate::resources::lightmap::LightmapMode,
    ) -> crate::error::ViewportResult<()> {
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        // Zero-pad / truncate UV1 to the vertex count. It rides the same vec4
        // sidecar as the plugin vertex attribute (binding 15), so pack UV into
        // .xy and the atlas page into .z (0 for single-page lightmaps); the shader
        // reads .xy and selects the array layer from .z.
        let vertex_count = mesh.vertex_count().max(1);
        let mut values = vec![[0.0f32; 4]; vertex_count];
        for (i, (dst, src)) in values.iter_mut().zip(uv1.iter()).enumerate() {
            let page = pages.and_then(|p| p.get(i)).copied().unwrap_or(0) as f32;
            *dst = [src.x, src.y, page, 0.0];
        }
        let uv1_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("lightmap_uv1_buf"),
            size: (values.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            uv1_buffer.slice(..),
            bytemuck::cast_slice(&values),
        );
        uv1_buffer.unmap();
        mesh.lightmap = Some(crate::resources::lightmap::MeshLightmap {
            uv1_buffer,
            texture_id: data.texture_id(),
            direction_texture_id: data.direction_texture_id(),
            is_shadowmask: data.is_shadowmask(),
            mode: mode.to_shader(),
            // A per-mesh lightmap owns its whole atlas: identity sub-rect, layer 0.
            // The array layer for a multi-page lightmap comes per vertex (UV1.z),
            // not from here. Scene lightmaps override both via `set_scene_lightmap`.
            scale_bias: [1.0, 1.0, 0.0, 0.0],
            layer: 0,
        });
        // The gen bump forces the object bind-group rebuild that swaps the
        // fallbacks at bindings 16/17 for this UV1 buffer and texture.
        mesh.lightmap_gen = mesh.lightmap_gen.wrapping_add(1);
        Ok(())
    }

    /// Attach a mesh to a shared scene lightmap atlas.
    ///
    /// Scene-level atlasing packs many objects into a handful of shared atlas
    /// pages instead of giving each mesh its own atlas. Every object points at the
    /// same atlas array (one `TextureId`, uploaded with
    /// [`upload_texture_hdr_layers`](crate::resources::ViewportGpuResources::upload_texture_hdr_layers))
    /// and carries its own placement: `layer` selects the atlas page and
    /// `scale_bias` maps its `[0, 1]` unwrap onto its sub-rect of that page
    /// (`lm_uv = uv1 * scale_bias.xy + scale_bias.zw`). This is the Unity/Unreal
    /// per-instance lightmap-index + `LightmapScaleBias` model.
    ///
    /// `uv1` is the object's own unique unwrap in `[0, 1]`, exactly as for
    /// [`set_lightmap`](Self::set_lightmap); the sub-rect transform is applied in
    /// the shader, so the same unwrap works whether the object owns an atlas or
    /// shares one. Radiance only (`NonDirectional`); pass the shared atlas array as
    /// `radiance`.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered.
    pub fn set_scene_lightmap(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        uv1: &[glam::Vec2],
        radiance: crate::resources::TextureId,
        layer: u32,
        scale_bias: [f32; 4],
        mode: crate::resources::lightmap::LightmapMode,
    ) -> crate::error::ViewportResult<()> {
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        // The object's own unwrap, page 0: the shared page comes from `layer` on
        // the uniform, so UV1.z stays 0 (the shader adds them).
        let vertex_count = mesh.vertex_count().max(1);
        let mut values = vec![[0.0f32; 4]; vertex_count];
        for (dst, src) in values.iter_mut().zip(uv1.iter()) {
            *dst = [src.x, src.y, 0.0, 0.0];
        }
        let uv1_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("scene_lightmap_uv1_buf"),
            size: (values.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            uv1_buffer.slice(..),
            bytemuck::cast_slice(&values),
        );
        uv1_buffer.unmap();
        mesh.lightmap = Some(crate::resources::lightmap::MeshLightmap {
            uv1_buffer,
            texture_id: radiance,
            direction_texture_id: None,
            is_shadowmask: false,
            mode: mode.to_shader(),
            scale_bias,
            layer,
        });
        mesh.lightmap_gen = mesh.lightmap_gen.wrapping_add(1);
        Ok(())
    }

    /// Remove a mesh's lightmap, reverting it to the shared fallbacks. No-op if
    /// none was set.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered.
    pub fn clear_lightmap(
        &mut self,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
    ) -> crate::error::ViewportResult<()> {
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        mesh.lightmap = None;
        mesh.lightmap_gen = mesh.lightmap_gen.wrapping_add(1);
        Ok(())
    }

    /// Write per-vertex colours into an uploaded mesh in place, starting at
    /// `start_vertex`, without re-uploading the whole mesh.
    ///
    /// This is the fast path for vertex painting. The per-vertex colour term is
    /// already part of every mesh's interleaved vertex layout and is multiplied
    /// into the base colour before lighting (see [`MeshData::vertex_colours`]),
    /// so painted colours show up with no pipeline or shader change. Only the
    /// colour bytes of the touched vertices are written, so a brush stroke over
    /// `colours.len()` vertices costs O(touched vertices), not O(mesh) the way
    /// [`replace_mesh_data`](Self::replace_mesh_data) (which rewrites the whole
    /// interleaved vertex buffer) would.
    ///
    /// `colours` is one RGBA (linear 0..1) per vertex, covering the contiguous
    /// range `[start_vertex, start_vertex + colours.len())`. Colour is
    /// interleaved (each vertex's colour sits one vertex stride apart), so the
    /// write is issued per vertex; a stroke touching a few hundred vertices is
    /// a few hundred small writes, still bounded by what was painted.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `mesh_id` is not registered, or
    /// [`ViewportError::MeshLengthMismatch`](crate::error::ViewportError::MeshLengthMismatch)
    /// if the run would extend past the mesh's vertex count.
    pub fn update_vertex_colours(
        &self,
        queue: &crate::gpu::Queue,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        start_vertex: usize,
        colours: &[[f32; 4]],
    ) -> crate::error::ViewportResult<()> {
        let store_len = self.mesh_store.len();
        let mesh =
            self.mesh_store
                .get(mesh_id)
                .ok_or(crate::error::ViewportError::StaleHandle {
                    index: mesh_id.index(),
                    count: store_len,
                })?;
        if colours.is_empty() {
            return Ok(());
        }
        let stride = std::mem::size_of::<Vertex>() as u64;
        let vertex_count = (mesh.vertex_buffer.size() / stride) as usize;
        if start_vertex + colours.len() > vertex_count {
            return Err(crate::error::ViewportError::MeshLengthMismatch {
                positions: start_vertex + colours.len(),
                normals: vertex_count,
            });
        }
        // Colour is the `[f32; 4]` at offset 24 in the interleaved `Vertex`
        // (see `Vertex::buffer_layout`). Interleaving puts each vertex's colour
        // one full stride apart, so a contiguous vertex run is written per
        // vertex rather than in a single strided copy.
        const COLOUR_OFFSET: u64 = 24;
        for (i, colour) in colours.iter().enumerate() {
            let byte_offset = (start_vertex as u64 + i as u64) * stride + COLOUR_OFFSET;
            queue.write_buffer(
                &mesh.vertex_buffer,
                byte_offset,
                bytemuck::cast_slice(std::slice::from_ref(colour)),
            );
        }
        Ok(())
    }

    /// Replace the mesh at `mesh_index` with new geometry data.
    ///
    /// When the new vertex and index counts match the existing mesh and no attributes are
    /// present, the existing GPU buffers are reused and data is written in place, avoiding
    /// GPU memory allocation. When topology changes, new buffers are allocated.
    ///
    /// A bound position/normal override buffer (and its slice window) survives
    /// either path. On a topology change the caller is responsible for the
    /// override still covering the new vertex count; vertices past the
    /// buffer's or window's end fall back to the vertex-buffer attribute.
    ///
    /// This is the only slot-targeting mesh operation, so it doubles as the
    /// guard against a free racing a queued replace: the `mesh_id` generation is
    /// checked here (and again in `MeshStore::replace`), so a handle whose mesh
    /// was freed, or freed and its slot reused by a later upload, is rejected
    /// rather than overwriting the mesh now in that slot. The async upload paths
    /// need no such guard because they allocate a fresh slot at apply time.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle) if `mesh_index` is out of range
    /// or the handle is stale, or any mesh validation error from the new data.
    pub fn replace_mesh_data(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        data: &MeshData,
    ) -> crate::error::ViewportResult<()> {
        if !self.mesh_store.contains(mesh_id) {
            return Err(crate::error::ViewportError::StaleHandle {
                index: mesh_id.index(),
                count: self.mesh_store.len(),
            });
        }
        Self::validate_mesh_data(data)?;
        Self::validate_mesh_size(device, data)?;

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
                let colour = data
                    .vertex_colours
                    .as_ref()
                    .and_then(|c| c.get(i))
                    .copied()
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                Vertex {
                    position: *p,
                    normal: *n,
                    colour,
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
                && data.attributes.is_empty()
                && data.extension_attributes.is_none()
                && existing.extension_attr_buffer.is_none();

            if in_place {
                use bytemuck::cast_slice;
                let aabb = crate::scene::aabb::Aabb::from_positions(&data.positions);

                let mesh = self.mesh_store.get_mut(mesh_id).unwrap();
                queue.write_buffer(&mesh.vertex_buffer, 0, cast_slice(&vertices));
                queue.write_buffer(&mesh.index_buffer, 0, cast_slice(data.indices.as_slice()));
                // Sidecars are built lazily; refresh only the ones a view has
                // already materialised.
                if let Some(ref edge_buf) = mesh.edge_index_buffer {
                    let edge_indices =
                        crate::resources::mesh::geometry::generate_edge_indices(&data.indices);
                    let edge_byte_len = (edge_indices.len() * std::mem::size_of::<u32>()) as u64;
                    if edge_byte_len <= edge_buf.size() {
                        queue.write_buffer(edge_buf, 0, cast_slice(&edge_indices));
                        mesh.edge_index_count = edge_indices.len() as u32;
                    }
                }
                if let Some(ref nl_buf) = mesh.normal_line_buffer {
                    let normal_line_verts =
                        Self::build_normal_lines(&data.positions, &data.normals);
                    queue.write_buffer(nl_buf, 0, cast_slice(&normal_line_verts));
                }
                mesh.aabb = aabb;
                mesh.cpu_positions = Some(data.positions.clone());
                mesh.cpu_normals = Some(data.normals.clone());
                mesh.cpu_indices = Some(data.indices.clone());
                mesh.submeshes = data.submeshes.clone();
                mesh.content_rev += 1;

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

        let mut new_mesh = Self::create_mesh_with_normals(
            device,
            &self.object_bind_group_layout,
            &self.fallback_texture.view,
            &self.fallback_texture_array_view,
            &self.fallback_normal_map_view,
            &self.fallback_ao_map_view,
            &self.material_sampler,
            &self.lut_sampler,
            &self.content.fallback_lut_view,
            &self.content.fallback_scalar_buf,
            &self.fallback_texture.view,
            &self.content.fallback_face_colour_buf,
            &self.content.fallback_warp_buf,
            &self.content.fallback_position_override_buf,
            &self.content.fallback_normal_override_buf,
            &self.content.fallback_extension_attr_buf,
            &self.fallback_metallic_roughness_texture_view,
            &self.fallback_emissive_texture_view,
            &vertices,
            &data.indices,
            None,
        );
        new_mesh.cpu_positions = Some(data.positions.clone());
        new_mesh.cpu_normals = Some(data.normals.clone());
        new_mesh.cpu_indices = Some(data.indices.clone());
        new_mesh.submeshes = data.submeshes.clone();
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
        new_mesh.extension_attr_buffer =
            Self::upload_extension_attributes(device, data, vertices.len());
        // Content changed under a stable id: carry the old revision forward
        // plus one so content-keyed caches invalidate.
        new_mesh.content_rev = self
            .mesh_store
            .get(mesh_id)
            .map_or(1, |old| old.content_rev + 1);
        // Carry any bound position/normal override across the rebuild.
        // `create_mesh_with_normals` bakes the fallback override buffers into
        // the fresh bind group, so advance the gens past the old mesh's: the
        // tex-key mismatch on the next prepare rebuilds the bind group with
        // the carried buffer. Without this the consumer's binding silently
        // reverts to the fallback (bind pose) on a topology change.
        if let Some(old) = self.mesh_store.get(mesh_id) {
            new_mesh.position_override_buffer = old.position_override_buffer.clone();
            new_mesh.position_override_slice = old.position_override_slice;
            new_mesh.normal_override_buffer = old.normal_override_buffer.clone();
            new_mesh.normal_override_slice = old.normal_override_slice;
            new_mesh.position_override_gen = old.position_override_gen.wrapping_add(1);
            new_mesh.normal_override_gen = old.normal_override_gen.wrapping_add(1);
        }
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
    pub fn mesh(&self, id: crate::resources::mesh::mesh_store::MeshId) -> Option<&GpuMesh> {
        self.mesh_store.get(id)
    }

    /// Total number of mesh slots (including empty/removed slots).
    pub fn mesh_slot_count(&self) -> usize {
        self.mesh_store.slot_count()
    }

    /// Deprecated alias for [`free_mesh`](Self::free_mesh).
    #[deprecated(note = "renamed to free_mesh")]
    pub fn remove_mesh(&mut self, id: crate::resources::mesh::mesh_store::MeshId) -> bool {
        self.free_mesh(id)
    }

    /// Free a mesh, reclaiming its GPU buffers and slot.
    ///
    /// Drops the `GpuMesh` (vertex, index, attribute buffers and its object bind
    /// group; wgpu defers the real free until in-flight commands complete),
    /// bumps the slot generation so `id` no longer resolves, and frees the slot
    /// for a later upload to reuse. A stale `id` held elsewhere degrades to
    /// fallback rendering rather than aliasing the reused slot.
    ///
    /// Returns `true` if a mesh was freed, `false` if `id` did not resolve to a
    /// live mesh. This is the residency-facing name for [`remove_mesh`]; the two
    /// are equivalent. To free a mesh that is a member of a LOD group, free the
    /// group with [`free_lod_group`](Self::free_lod_group) instead so shared
    /// members are handled.
    ///
    /// [`remove_mesh`]: Self::remove_mesh
    pub fn free_mesh(&mut self, id: crate::resources::mesh::mesh_store::MeshId) -> bool {
        let removed = self.mesh_store.remove(id);
        if removed {
            self.resource_free_epoch += 1;
        }
        removed
    }

    /// Upload an unstructured volume mesh and return a ready-to-submit
    /// [`VolumeMeshItem`](crate::VolumeMeshItem).
    ///
    /// Extracts the boundary surface and uploads it through the standard mesh
    /// pipeline. Interior faces (shared by two cells) are discarded; only
    /// boundary faces (belonging to exactly one cell) are kept. Per-cell scalar
    /// and colour attributes are remapped to per-face attributes so the
    /// face-colouring path handles them automatically.
    ///
    /// The returned item has `transparency: None` and `projected_tet_id: None`;
    /// it renders as an opaque surface mesh. Use
    /// [`upload_volume_mesh_with_transparency`](Self::upload_volume_mesh_with_transparency)
    /// instead if you need to toggle volumetric rendering at runtime.
    pub fn upload_volume_mesh(
        &mut self,
        device: &crate::gpu::Device,
        data: &crate::resources::volume::volume_mesh::VolumeMeshData,
    ) -> crate::error::ViewportResult<crate::VolumeMeshItem> {
        let (mesh_data, face_to_cell) =
            crate::resources::volume::volume_mesh::extract_boundary_faces(data);
        let mesh_id = self.upload_mesh_data(device, &mesh_data)?;
        Ok(crate::VolumeMeshItem::new(mesh_id, face_to_cell))
    }

    /// Upload an unstructured volume mesh with both the boundary surface and
    /// the projected-tet decomposition needed for volumetric rendering.
    ///
    /// The returned item carries:
    ///   - `boundary_mesh_id` + `face_to_cell` for the opaque surface draw and
    ///     boundary-level cell picking,
    ///   - `projected_tet_id` for the volumetric draw, and
    ///   - `volume_mesh_data` (an `Arc` over the input) for interior-inclusive
    ///     cell picking when transparency is on.
    ///
    /// Default `transparency: None`: the item renders as a boundary surface
    /// until the host sets [`VolumeMeshItem::transparency`](crate::VolumeMeshItem::transparency)
    /// to `Some(VolumeTransparency { .. })`. Switching modes at runtime is free
    /// because both GPU artifacts are already resident.
    ///
    /// `scalar_attribute` names a key in `data.cell_scalars`; cells without the
    /// attribute receive scalar 0.0. The scalar range is auto-detected from the
    /// data and stored in the per-volume uniform.
    pub fn upload_volume_mesh_with_transparency(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::volume::volume_mesh::VolumeMeshData,
        scalar_attribute: &str,
    ) -> crate::error::ViewportResult<crate::VolumeMeshItem> {
        let (mesh_data, face_to_cell) =
            crate::resources::volume::volume_mesh::extract_boundary_faces(&data);
        let mesh_id = self.upload_mesh_data(device, &mesh_data)?;
        let (pt_id, _, _) = self.upload_projected_tet(device, &data, scalar_attribute)?;
        let mut item = crate::VolumeMeshItem::new(mesh_id, face_to_cell);
        item.projected_tet_id = Some(pt_id);
        item.volume_mesh_data = Some(std::sync::Arc::new(data));
        Ok(item)
    }

    /// Upload a clipped volume mesh, returning a ready-to-submit
    /// [`VolumeMeshItem`](crate::VolumeMeshItem).
    ///
    /// Each entry in `clip_planes` is `[nx, ny, nz, d]` where a point `p` is
    /// kept when `dot(p, [nx, ny, nz]) + d >= 0`. An empty slice is equivalent
    /// to [`upload_volume_mesh`](Self::upload_volume_mesh).
    ///
    /// The returned item has `transparency: None` and `projected_tet_id: None`.
    pub fn upload_clipped_volume_mesh(
        &mut self,
        device: &crate::gpu::Device,
        data: &crate::resources::volume::volume_mesh::VolumeMeshData,
        clip_planes: &[[f32; 4]],
    ) -> crate::error::ViewportResult<crate::VolumeMeshItem> {
        let (mesh_data, face_to_cell) =
            crate::resources::volume::volume_mesh::extract_clipped_volume_faces(data, clip_planes);
        let mesh_id = self.upload_mesh_data(device, &mesh_data)?;
        Ok(crate::VolumeMeshItem::new(mesh_id, face_to_cell))
    }

    /// Replace an existing boundary-mesh slot with a freshly-extracted clipped
    /// volume mesh, returning the new `face_to_cell` map.
    ///
    /// Equivalent to calling [`upload_clipped_volume_mesh`](Self::upload_clipped_volume_mesh)
    /// and then [`replace_mesh_data`](Self::replace_mesh_data), but without
    /// allocating a new mesh slot. Use this for per-frame clip-plane updates to
    /// avoid leaking GPU memory.
    pub fn replace_clipped_volume_mesh(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        data: &crate::resources::volume::volume_mesh::VolumeMeshData,
        clip_planes: &[[f32; 4]],
    ) -> crate::error::ViewportResult<Vec<u32>> {
        let (mesh_data, face_to_cell) =
            crate::resources::volume::volume_mesh::extract_clipped_volume_faces(data, clip_planes);
        self.replace_mesh_data(device, queue, mesh_id, &mesh_data)?;
        Ok(face_to_cell)
    }

    /// Update a volume-mesh boundary's per-cell scalar field in place, without
    /// re-extracting the boundary or re-uploading any geometry.
    ///
    /// For a time-series where a scalar field animates on a **fixed** volume
    /// mesh (a CFD/FE field played back over a static unstructured grid), the
    /// boundary topology does not change frame to frame -- only the values do.
    /// Rather than rebuilding the mesh with
    /// [`upload_volume_mesh`](Self::upload_volume_mesh) /
    /// [`extract_clipped_volume_faces`](crate::resources::volume::volume_mesh::extract_clipped_volume_faces)
    /// every frame (which re-runs the O(cells) boundary extraction and
    /// re-uploads the whole surface), upload the mesh once and call this each
    /// frame with the new per-cell values.
    ///
    /// The new per-cell `cell_scalars` are remapped to the boundary faces
    /// through `face_to_cell` (the map [`VolumeMeshItem::face_to_cell`](crate::VolumeMeshItem::face_to_cell)
    /// retains from upload) and written to the named per-face scalar buffer
    /// only: no boundary re-extraction, no vertex/index touch, no new mesh slot.
    /// Cost is O(boundary faces) instead of O(total cells).
    ///
    /// `face_to_cell` and the boundary must be the ones the mesh was uploaded
    /// with (unchanged since); only the scalar values may differ. The scalar
    /// range is recomputed so a colourmap on the owning item stays accurate;
    /// items that pin an explicit [`scalar_range`](crate::VolumeMeshItem::scalar_range)
    /// keep using theirs.
    ///
    /// # Errors
    ///
    /// - [`SlotEmpty`](crate::error::ViewportError::SlotEmpty) : `mesh_id` not in the store.
    /// - [`AttributeNotFound`](crate::error::ViewportError::AttributeNotFound) : `name` is not a
    ///   per-face scalar attribute on the mesh (upload the mesh with a
    ///   `cell_scalars` entry of this name first).
    /// - [`AttributeLengthMismatch`](crate::error::ViewportError::AttributeLengthMismatch) :
    ///   `face_to_cell.len()` differs from the mesh's boundary-face count, or a
    ///   `face_to_cell` entry indexes past `cell_scalars`.
    pub fn update_volume_mesh_scalar(
        &mut self,
        queue: &crate::gpu::Queue,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        name: &str,
        face_to_cell: &[u32],
        cell_scalars: &[f32],
    ) -> crate::error::ViewportResult<()> {
        let gpu_mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::SlotEmpty {
                    index: mesh_id.index(),
                })?;

        let buffer = gpu_mesh.face_attribute_buffers.get(name).ok_or_else(|| {
            crate::error::ViewportError::AttributeNotFound {
                mesh_id: mesh_id.index(),
                name: name.to_string(),
            }
        })?;

        // The per-face scalar buffer holds `3 * n_tris` f32 (each face scalar
        // replicated to the three vertices of its triangle; see
        // `expand_face_scalars_to_3n`).
        let n_tris = (buffer.size() / 4) as usize / 3;
        if face_to_cell.len() != n_tris {
            return Err(crate::error::ViewportError::AttributeLengthMismatch {
                expected: n_tris,
                got: face_to_cell.len(),
            });
        }

        // Remap cell -> face and expand to 3N in one pass, tracking the range.
        let mut expanded = Vec::with_capacity(n_tris * 3);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &ci in face_to_cell {
            let v = cell_scalars.get(ci as usize).copied().ok_or(
                crate::error::ViewportError::AttributeLengthMismatch {
                    expected: ci as usize + 1,
                    got: cell_scalars.len(),
                },
            )?;
            min = min.min(v);
            max = max.max(v);
            expanded.push(v);
            expanded.push(v);
            expanded.push(v);
        }

        // Zero-copy in-place write via the wgpu staging belt.
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&expanded));

        // Recompute the scalar range so colourmap LUT mapping stays accurate.
        let range = if min > max { (0.0, 1.0) } else { (min, max) };
        gpu_mesh.attribute_ranges.insert(name.to_string(), range);

        // Force the attribute bind group to rebuild on next prepare() (the range
        // may have moved), matching `replace_attribute`.
        gpu_mesh.last_tex_key = (
            gpu_mesh.last_tex_key.0,
            gpu_mesh.last_tex_key.1,
            gpu_mesh.last_tex_key.2,
            gpu_mesh.last_tex_key.3,
            u64::MAX,
            gpu_mesh.last_tex_key.5,
            gpu_mesh.last_tex_key.6,
            gpu_mesh.last_tex_key.7,
            gpu_mesh.last_tex_key.8,
            gpu_mesh.last_tex_key.9,
            gpu_mesh.last_tex_key.10,
            gpu_mesh.last_tex_key.11,
        );

        Ok(())
    }

    /// Replace a previously uploaded sparse voxel grid in place.
    ///
    /// Equivalent to calling [`upload_sparse_volume_grid_data`](Self::upload_sparse_volume_grid_data)
    /// and then [`replace_mesh_data`](Self::replace_mesh_data), but without allocating a new slot.
    /// Use this for per-frame or per-interaction updates (e.g. voxel paint) to avoid leaking GPU memory.
    pub fn replace_sparse_volume_grid_data(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        data: &crate::resources::volume::sparse_volume::SparseVolumeGridData,
    ) -> crate::error::ViewportResult<()> {
        let mesh_data = crate::resources::volume::sparse_volume::extract_sparse_boundary(data);
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
        device: &crate::gpu::Device,
        data: &crate::resources::volume::sparse_volume::SparseVolumeGridData,
    ) -> crate::error::ViewportResult<crate::resources::mesh::mesh_store::MeshId> {
        let mesh_data = crate::resources::volume::sparse_volume::extract_sparse_boundary(data);
        self.upload_mesh_data(device, &mesh_data)
    }

    /// Start an asynchronous boundary-only volume mesh upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Boundary
    /// extraction (`extract_boundary_faces`) and vertex prep
    /// (`prep_mesh_data`) run on a worker thread; the apply step creates
    /// GPU buffers and inserts the mesh. Take the resulting
    /// [`VolumeMeshItem`](crate::VolumeMeshItem) via
    /// [`upload_result_volume_mesh`](Self::upload_result_volume_mesh).
    ///
    /// Ownership of `data` transfers into the worker. The returned item has
    /// `transparency: None`; this async path does not produce a projected-tet
    /// decomposition. For volumetric rendering use the synchronous
    /// [`upload_volume_mesh_with_transparency`](Self::upload_volume_mesh_with_transparency).
    pub fn begin_upload_volume_mesh(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::volume::volume_mesh::VolumeMeshData,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<(
            crate::resources::mesh::mesh_store::MeshId,
            Vec<u32>,
        )>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let (mesh_data, face_to_cell) =
                    crate::resources::volume::volume_mesh::extract_boundary_faces(&data);
                progress.set(0.5);
                DeviceResources::validate_mesh_data(&mesh_data)?;
                let prep = DeviceResources::prep_mesh_data(&mesh_data);
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let mesh_id =
                            resources.assemble_mesh_data(&device_for_apply, &mesh_data, prep);
                        slot_for_apply.set((mesh_id, face_to_cell));
                    }),
                ))
            })
        };

        self.job_results
            .volume_mesh
            .lock()
            .expect("volume mesh result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`VolumeMeshItem`](crate::VolumeMeshItem) produced
    /// by a completed
    /// [`begin_upload_volume_mesh`](Self::begin_upload_volume_mesh) job.
    pub fn upload_result_volume_mesh(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::VolumeMeshItem> {
        let mut map = self
            .job_results
            .volume_mesh
            .lock()
            .expect("volume mesh result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some((mesh_id, face_to_cell)) => {
                map.remove(&id);
                Ok(crate::VolumeMeshItem::new(mesh_id, face_to_cell))
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Start an asynchronous clipped volume mesh upload. See
    /// [`upload_clipped_volume_mesh`](Self::upload_clipped_volume_mesh) for the
    /// sync analog and the semantics of `clip_planes`.
    pub fn begin_upload_clipped_volume_mesh(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::volume::volume_mesh::VolumeMeshData,
        clip_planes: Vec<[f32; 4]>,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<(
            crate::resources::mesh::mesh_store::MeshId,
            Vec<u32>,
        )>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let (mesh_data, face_to_cell) =
                    crate::resources::volume::volume_mesh::extract_clipped_volume_faces(
                        &data,
                        &clip_planes,
                    );
                progress.set(0.5);
                DeviceResources::validate_mesh_data(&mesh_data)?;
                let prep = DeviceResources::prep_mesh_data(&mesh_data);
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let mesh_id =
                            resources.assemble_mesh_data(&device_for_apply, &mesh_data, prep);
                        slot_for_apply.set((mesh_id, face_to_cell));
                    }),
                ))
            })
        };

        self.job_results
            .clipped_volume_mesh
            .lock()
            .expect("clipped volume mesh result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`VolumeMeshItem`](crate::VolumeMeshItem) produced
    /// by a completed
    /// [`begin_upload_clipped_volume_mesh`](Self::begin_upload_clipped_volume_mesh) job.
    pub fn upload_result_clipped_volume_mesh(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::VolumeMeshItem> {
        let mut map = self
            .job_results
            .clipped_volume_mesh
            .lock()
            .expect("clipped volume mesh result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some((mesh_id, face_to_cell)) => {
                map.remove(&id);
                Ok(crate::VolumeMeshItem::new(mesh_id, face_to_cell))
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Start an asynchronous sparse voxel grid upload.
    pub fn begin_upload_sparse_volume_grid_data(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::volume::sparse_volume::SparseVolumeGridData,
    ) -> crate::resources::JobId {
        let slot =
            crate::resources::ResultSlot::<crate::resources::mesh::mesh_store::MeshId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let mesh_data =
                    crate::resources::volume::sparse_volume::extract_sparse_boundary(&data);
                progress.set(0.5);
                DeviceResources::validate_mesh_data(&mesh_data)?;
                let prep = DeviceResources::prep_mesh_data(&mesh_data);
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let mesh_id =
                            resources.assemble_mesh_data(&device_for_apply, &mesh_data, prep);
                        slot_for_apply.set(mesh_id);
                    }),
                ))
            })
        };

        self.job_results
            .sparse_volume_grid
            .lock()
            .expect("sparse volume grid result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`MeshId`](crate::resources::mesh::mesh_store::MeshId) produced by a completed
    /// [`begin_upload_sparse_volume_grid_data`](Self::begin_upload_sparse_volume_grid_data)
    /// job.
    pub fn upload_result_sparse_volume_grid(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::mesh::mesh_store::MeshId> {
        let mut map = self
            .job_results
            .sparse_volume_grid
            .lock()
            .expect("sparse volume grid result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(mesh_id) => {
                map.remove(&id);
                Ok(mesh_id)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
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
    /// Upload `MeshData::extension_attributes` to a per-mesh storage buffer
    /// (one `vec4<f32>` per vertex, zero-padded / truncated to the vertex
    /// count). `None` when the mesh declares no extension attributes.
    pub(crate) fn upload_extension_attributes(
        device: &crate::gpu::Device,
        data: &MeshData,
        vertex_count: usize,
    ) -> Option<crate::gpu::Buffer> {
        let attrs = data.extension_attributes.as_ref()?;
        let mut values = vec![[0.0f32; 4]; vertex_count.max(1)];
        for (dst, src) in values.iter_mut().zip(attrs.iter()) {
            *dst = *src;
        }
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("extension_attr_buf"),
            size: (values.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(buf.slice(..), bytemuck::cast_slice(&values));
        buf.unmap();
        Some(buf)
    }

    fn upload_attributes(
        device: &crate::gpu::Device,
        attributes: &std::collections::HashMap<String, AttributeData>,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        uvs: Option<&[[f32; 2]]>,
        tangents: Option<&[[f32; 4]]>,
    ) -> (
        std::collections::HashMap<String, crate::gpu::Buffer>,
        std::collections::HashMap<String, (f32, f32)>,
        Option<crate::gpu::Buffer>,
        std::collections::HashMap<String, crate::gpu::Buffer>,
        std::collections::HashMap<String, crate::gpu::Buffer>,
        std::collections::HashMap<String, crate::gpu::Buffer>,
    ) {
        let mut bufs = std::collections::HashMap::new();
        let mut ranges = std::collections::HashMap::new();
        let mut face_attr_bufs: std::collections::HashMap<String, crate::gpu::Buffer> =
            std::collections::HashMap::new();
        let mut vector_attr_bufs: std::collections::HashMap<String, crate::gpu::Buffer> =
            std::collections::HashMap::new();
        let mut face_colour_bufs: std::collections::HashMap<String, crate::gpu::Buffer> =
            std::collections::HashMap::new();
        let mut face_vbuf: Option<crate::gpu::Buffer> = None;

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
                    let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some(&format!("face_colour_{name}")),
                        size: byte_len as u64,
                        usage: crate::gpu::BufferUsages::STORAGE
                            | crate::gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    crate::resources::builders::write_mapped(
                        buf.slice(..),
                        bytemuck::cast_slice(&expanded),
                    );
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
                    let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some(&format!("vec_attr_{name}")),
                        size: byte_len,
                        usage: crate::gpu::BufferUsages::VERTEX
                            | crate::gpu::BufferUsages::STORAGE
                            | crate::gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    crate::resources::builders::write_mapped(
                        buf.slice(..),
                        bytemuck::cast_slice(&flat),
                    );
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
    fn create_storage_buffer_f32(
        device: &crate::gpu::Device,
        label: &str,
        data: &[f32],
    ) -> crate::gpu::Buffer {
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some(label),
            size: (std::mem::size_of::<f32>() * data.len()) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(buf.slice(..), bytemuck::cast_slice(data));
        buf.unmap();
        buf
    }

    /// Build a non-indexed 3N-vertex buffer: one vertex per triangle corner, geometry only.
    fn build_face_vertex_buffer(
        device: &crate::gpu::Device,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        uvs: Option<&[[f32; 2]]>,
        tangents: Option<&[[f32; 4]]>,
    ) -> crate::gpu::Buffer {
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
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("face_vertex_buf"),
            size: (std::mem::size_of::<Vertex>() * verts.len().max(1)) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(buf.slice(..), bytemuck::cast_slice(&verts));
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
    /// `[tx, ty, tz, w]` with `w = +/-1.0` encoding bitangent handedness.
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

        // Accumulate sdir/tdir contributions per vertex. Sequential.
        //
        // **Do not** use rayon parallel iterators in this function. This
        // routine is already invoked from a rayon worker : every mesh
        // upload runs `prep_mesh_data -> compute_tangents` inside a
        // `submit_cpu` job (see `upload_jobs::Runner::submit_cpu`).
        // Adding intra-mesh parallelism causes nested rayon work:
        // a worker enters `par_chunks(3).fold(...)`, parks at a join,
        // steals another mesh's upload (which itself enters compute_tangents
        // and parks again), and so on. Each suspension keeps frames on
        // the worker's 2 MB stack; with the upload queue draining
        // concurrent tangent tasks, stack depth grows unboundedly and
        // overflows.
        //
        // The function is cache-friendly and runs at ~30 ns / triangle
        // sequentially (~ 15 ms for a 500 k-tri mesh). Per-mesh
        // parallelism comes from the upload job pool, not from inside
        // this function.
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

        // Gram-Schmidt orthogonalization per vertex. Sequential, for the
        // same nested-rayon reason as above.
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
        let index_count = data.indices.len() as u64;
        for range in &data.submeshes {
            let end = range.first_index as u64 + range.index_count as u64;
            if end > index_count {
                return Err(crate::error::ViewportError::SubmeshRangeOutOfBounds {
                    first_index: range.first_index,
                    range_count: range.index_count,
                    index_count: data.indices.len(),
                });
            }
        }
        Ok(())
    }

    /// Build per-vertex normal visualization lines from mesh data.
    /// Build the wireframe edge-index buffer for a mesh on first use.
    ///
    /// Edge extraction sorts and dedups three candidate edges per triangle,
    /// which is far too expensive to run eagerly on every upload for a view
    /// most meshes never show. `prepare()` calls this for wireframe items
    /// (and the volume-mesh boundary overlay); it is a no-op when the
    /// buffer already exists.
    pub(crate) fn ensure_edge_indices(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
    ) {
        let Some(mesh) = self.mesh_store.get(mesh_id) else {
            return;
        };
        if mesh.edge_index_buffer.is_some() {
            return;
        }
        let Some(indices) = &mesh.cpu_indices else {
            return;
        };
        let edge_indices = crate::resources::mesh::geometry::generate_edge_indices(indices);
        let bytes = (std::mem::size_of::<u32>() * edge_indices.len().max(2)) as u64;
        if bytes > device.limits().max_buffer_size {
            tracing::warn!(
                mesh_index = mesh_id.index(),
                bytes,
                "edge index buffer would exceed max_buffer_size; wireframe skipped"
            );
            return;
        }
        let buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("edge_index_buf"),
            size: bytes,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            buffer.slice(..),
            bytemuck::cast_slice::<u32, u8>(&edge_indices),
        );
        buffer.unmap();
        let mesh = self.mesh_store.get_mut(mesh_id).unwrap();
        mesh.edge_index_buffer = Some(buffer);
        mesh.edge_index_count = edge_indices.len() as u32;
    }

    /// Build the normal-line visualisation buffer for a mesh on first use.
    ///
    /// The sidecar is two 64-byte vertices per mesh vertex, so building and
    /// uploading it eagerly on every upload costs twice the mesh's own
    /// vertex data for a debug view most meshes never show. `prepare()`
    /// calls this for items with `show_normals` set; it is a no-op when the
    /// buffer already exists or when the mesh was created from raw vertex
    /// slices (which never carried normal lines).
    pub(crate) fn ensure_normal_lines(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
    ) {
        let Some(mesh) = self.mesh_store.get(mesh_id) else {
            return;
        };
        if mesh.normal_line_buffer.is_some() {
            return;
        }
        let (Some(positions), Some(normals)) = (&mesh.cpu_positions, &mesh.cpu_normals) else {
            return;
        };
        let bytes = (positions.len() * 2 * std::mem::size_of::<Vertex>()) as u64;
        if bytes > device.limits().max_buffer_size {
            tracing::warn!(
                mesh_index = mesh_id.index(),
                bytes,
                "normal-line buffer would exceed max_buffer_size; normals view skipped"
            );
            return;
        }
        let verts = Self::build_normal_lines(positions, normals);
        let buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("normal_line_buf"),
            size: bytes,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(buffer.slice(..), bytemuck::cast_slice(&verts));
        buffer.unmap();
        let count = verts.len() as u32;
        let mesh = self.mesh_store.get_mut(mesh_id).unwrap();
        mesh.normal_line_buffer = Some(buffer);
        mesh.normal_line_count = count;
    }

    /// Pre-build the wireframe edge-index and normal-line debug buffers for a
    /// mesh.
    ///
    /// Both buffers normally build on the first frame that draws the mesh in
    /// wireframe or with `show_normals`, which can stall that frame on large
    /// meshes (tens of milliseconds at a few hundred thousand triangles).
    /// Call this after upload for meshes the user is likely to toggle into
    /// those views, so the cost lands at load time instead.
    ///
    /// No-op for buffers that already exist. Meshes uploaded without CPU-side
    /// index or normal data keep the corresponding view unavailable, exactly
    /// as if this had not been called.
    pub fn prebuild_mesh_debug_sidecars(
        &mut self,
        device: &crate::gpu::Device,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
    ) {
        self.ensure_edge_indices(device, mesh_id);
        self.ensure_normal_lines(device, mesh_id);
    }

    fn build_normal_lines(positions: &[[f32; 3]], normals: &[[f32; 3]]) -> Vec<Vertex> {
        let normal_colour = [0.627_f32, 0.769, 1.0, 1.0];
        let normal_length = 0.1_f32;
        let mut normal_line_verts: Vec<Vertex> = Vec::with_capacity(positions.len() * 2);
        for (p, n) in positions.iter().zip(normals.iter()) {
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
        device: &crate::gpu::Device,
        object_bgl: &crate::gpu::BindGroupLayout,
        fallback_albedo_view: &crate::gpu::TextureView,
        fallback_lightmap_array_view: &crate::gpu::TextureView,
        fallback_normal_view: &crate::gpu::TextureView,
        fallback_ao_view: &crate::gpu::TextureView,
        fallback_sampler: &crate::gpu::Sampler,
        lut_sampler: &crate::gpu::Sampler,
        fallback_lut_view: &crate::gpu::TextureView,
        fallback_scalar_buf: &crate::gpu::Buffer,
        fallback_matcap_view: &crate::gpu::TextureView,
        fallback_face_colour_buf: &crate::gpu::Buffer,
        fallback_warp_buf: &crate::gpu::Buffer,
        fallback_position_override_buf: &crate::gpu::Buffer,
        fallback_normal_override_buf: &crate::gpu::Buffer,
        fallback_extension_attr_buf: &crate::gpu::Buffer,
        fallback_metallic_roughness_view: &crate::gpu::TextureView,
        fallback_emissive_view: &crate::gpu::TextureView,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> GpuMesh {
        Self::create_mesh_with_normals(
            device,
            object_bgl,
            fallback_albedo_view,
            fallback_lightmap_array_view,
            fallback_normal_view,
            fallback_ao_view,
            fallback_sampler,
            lut_sampler,
            fallback_lut_view,
            fallback_scalar_buf,
            fallback_matcap_view,
            fallback_face_colour_buf,
            fallback_warp_buf,
            fallback_position_override_buf,
            fallback_normal_override_buf,
            fallback_extension_attr_buf,
            fallback_metallic_roughness_view,
            fallback_emissive_view,
            vertices,
            indices,
            None,
        )
    }

    pub(crate) fn create_mesh_with_normals(
        device: &crate::gpu::Device,
        object_bgl: &crate::gpu::BindGroupLayout,
        fallback_albedo_view: &crate::gpu::TextureView,
        fallback_lightmap_array_view: &crate::gpu::TextureView,
        fallback_normal_view: &crate::gpu::TextureView,
        fallback_ao_view: &crate::gpu::TextureView,
        fallback_sampler: &crate::gpu::Sampler,
        lut_sampler: &crate::gpu::Sampler,
        fallback_lut_view: &crate::gpu::TextureView,
        fallback_scalar_buf: &crate::gpu::Buffer,
        fallback_matcap_view: &crate::gpu::TextureView,
        fallback_face_colour_buf: &crate::gpu::Buffer,
        fallback_warp_buf: &crate::gpu::Buffer,
        fallback_position_override_buf: &crate::gpu::Buffer,
        fallback_normal_override_buf: &crate::gpu::Buffer,
        fallback_extension_attr_buf: &crate::gpu::Buffer,
        fallback_metallic_roughness_view: &crate::gpu::TextureView,
        fallback_emissive_view: &crate::gpu::TextureView,
        vertices: &[Vertex],
        indices: &[u32],
        normal_line_verts: Option<&[Vertex]>,
    ) -> GpuMesh {
        use bytemuck::cast_slice;

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("vertex_buf"),
            size: (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
            usage: crate::gpu::BufferUsages::VERTEX
                | crate::gpu::BufferUsages::COPY_DST
                | crate::gpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(vertex_buffer.slice(..), cast_slice(vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("index_buf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: crate::gpu::BufferUsages::INDEX
                | crate::gpu::BufferUsages::COPY_DST
                | crate::gpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(index_buffer.slice(..), cast_slice(indices));
        index_buffer.unmap();

        let aabb = crate::scene::aabb::Aabb::from_positions(
            &vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
        );
        let mut mesh = Self::create_mesh_from_buffers(
            device,
            object_bgl,
            fallback_albedo_view,
            fallback_lightmap_array_view,
            fallback_normal_view,
            fallback_ao_view,
            fallback_sampler,
            lut_sampler,
            fallback_lut_view,
            fallback_scalar_buf,
            fallback_matcap_view,
            fallback_face_colour_buf,
            fallback_warp_buf,
            fallback_position_override_buf,
            fallback_normal_override_buf,
            fallback_extension_attr_buf,
            fallback_metallic_roughness_view,
            fallback_emissive_view,
            vertex_buffer,
            index_buffer,
            indices.len() as u32,
            aabb,
        );
        if let Some(nl_verts) = normal_line_verts
            && !nl_verts.is_empty()
        {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("normal_line_buf"),
                size: (std::mem::size_of::<Vertex>() * nl_verts.len()) as u64,
                usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            crate::resources::builders::write_mapped(buf.slice(..), cast_slice(nl_verts));
            buf.unmap();
            mesh.normal_line_count = nl_verts.len() as u32;
            mesh.normal_line_buffer = Some(buf);
        }
        mesh.cpu_indices = Some(indices.to_vec());
        mesh
    }

    /// Build a `GpuMesh` around already-created and already-filled vertex
    /// and index buffers: uniforms, bind groups, and the mesh record.
    /// Tail of `create_mesh_with_normals`, shared with the chunked async
    /// upload path, which fills the buffers across several frames before
    /// assembling the mesh here. The normal-line buffer and CPU pick
    /// copies start empty; callers set them as needed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_mesh_from_buffers(
        device: &crate::gpu::Device,
        object_bgl: &crate::gpu::BindGroupLayout,
        fallback_albedo_view: &crate::gpu::TextureView,
        fallback_lightmap_array_view: &crate::gpu::TextureView,
        fallback_normal_view: &crate::gpu::TextureView,
        fallback_ao_view: &crate::gpu::TextureView,
        fallback_sampler: &crate::gpu::Sampler,
        lut_sampler: &crate::gpu::Sampler,
        fallback_lut_view: &crate::gpu::TextureView,
        fallback_scalar_buf: &crate::gpu::Buffer,
        fallback_matcap_view: &crate::gpu::TextureView,
        fallback_face_colour_buf: &crate::gpu::Buffer,
        fallback_warp_buf: &crate::gpu::Buffer,
        fallback_position_override_buf: &crate::gpu::Buffer,
        fallback_normal_override_buf: &crate::gpu::Buffer,
        fallback_extension_attr_buf: &crate::gpu::Buffer,
        fallback_metallic_roughness_view: &crate::gpu::TextureView,
        fallback_emissive_view: &crate::gpu::TextureView,
        vertex_buffer: crate::gpu::Buffer,
        index_buffer: crate::gpu::Buffer,
        index_count: u32,
        aabb: crate::scene::aabb::Aabb,
    ) -> GpuMesh {
        use bytemuck::cast_slice;

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
            receive_shadows: 1,
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
            has_position_override: 0,
            has_normal_override: 0,
            emissive: [0.0; 3],
            use_flat: 0,
            alpha_mode: 0,
            alpha_cutoff: 0.5,
            has_metallic_roughness_tex: 0,
            has_emissive_tex: 0,
            uv_transform: [0.0, 0.0, 1.0, 1.0],
            deform_flags: 0,
            normal_strength: 1.0,
            ao_range: [0.0, 1.0],
            metallic_range: [0.0, 1.0],
            roughness_range: [0.0, 1.0],
            position_override_base: 0,
            position_override_len: u32::MAX,
            normal_override_base: 0,
            normal_override_len: u32::MAX,
            has_light_probe: 0,
            light_probe_index: 0,
            lightmap_mode: 0,
            lightmap_directional: 0,
            lightmap_scale_bias: [1.0, 1.0, 0.0, 0.0],
            lightmap_index: 0,
            has_shadowmask: 0,
            _pad_ls: [0; 2],
        };
        let object_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("object_uniform_buf"),
            size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            object_uniform_buf.slice(..),
            cast_slice(&[object_uniform]),
        );
        object_uniform_buf.unmap();

        let object_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("object_bind_group"),
            layout: object_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: object_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(fallback_albedo_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(fallback_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(fallback_normal_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(fallback_ao_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::TextureView(fallback_lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: fallback_scalar_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: crate::gpu::BindingResource::TextureView(fallback_matcap_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: fallback_face_colour_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 9,
                    resource: fallback_warp_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 10,
                    resource: crate::gpu::BindingResource::Sampler(lut_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 11,
                    resource: crate::gpu::BindingResource::TextureView(
                        fallback_metallic_roughness_view,
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 12,
                    resource: crate::gpu::BindingResource::TextureView(fallback_emissive_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 13,
                    resource: fallback_position_override_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 14,
                    resource: fallback_normal_override_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 15,
                    resource: fallback_extension_attr_buf.as_entire_binding(),
                },
                // 17: lightmap texture (fallback 1x1). Bound for every mesh; the
                // real texture is swapped in by the material bind-group rebuild
                // once a lightmap is set. UV1 rides binding 15 above.
                crate::gpu::BindGroupEntry {
                    binding: 17,
                    resource: crate::gpu::BindingResource::TextureView(
                        fallback_lightmap_array_view,
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 18,
                    resource: crate::gpu::BindingResource::TextureView(
                        fallback_lightmap_array_view,
                    ),
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
            receive_shadows: 1,
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
            has_position_override: 0,
            has_normal_override: 0,
            emissive: [0.0; 3],
            use_flat: 0,
            alpha_mode: 0,
            alpha_cutoff: 0.5,
            has_metallic_roughness_tex: 0,
            has_emissive_tex: 0,
            uv_transform: [0.0, 0.0, 1.0, 1.0],
            deform_flags: 0,
            normal_strength: 1.0,
            ao_range: [0.0, 1.0],
            metallic_range: [0.0, 1.0],
            roughness_range: [0.0, 1.0],
            position_override_base: 0,
            position_override_len: u32::MAX,
            normal_override_base: 0,
            normal_override_len: u32::MAX,
            has_light_probe: 0,
            light_probe_index: 0,
            lightmap_mode: 0,
            lightmap_directional: 0,
            lightmap_scale_bias: [1.0, 1.0, 0.0, 0.0],
            lightmap_index: 0,
            has_shadowmask: 0,
            _pad_ls: [0; 2],
        };
        let normal_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("normal_uniform_buf"),
            size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            normal_uniform_buf.slice(..),
            cast_slice(&[normal_override_uniform]),
        );
        normal_uniform_buf.unmap();

        let normal_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("normal_bind_group"),
            layout: object_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: normal_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(fallback_albedo_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(fallback_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(fallback_normal_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(fallback_ao_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::TextureView(fallback_lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: fallback_scalar_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 7,
                    resource: crate::gpu::BindingResource::TextureView(fallback_matcap_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: fallback_face_colour_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 9,
                    resource: fallback_warp_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 10,
                    resource: crate::gpu::BindingResource::Sampler(lut_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 11,
                    resource: crate::gpu::BindingResource::TextureView(
                        fallback_metallic_roughness_view,
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 12,
                    resource: crate::gpu::BindingResource::TextureView(fallback_emissive_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 13,
                    resource: fallback_position_override_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 14,
                    resource: fallback_normal_override_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 15,
                    resource: fallback_extension_attr_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 17,
                    resource: crate::gpu::BindingResource::TextureView(
                        fallback_lightmap_array_view,
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 18,
                    resource: crate::gpu::BindingResource::TextureView(
                        fallback_lightmap_array_view,
                    ),
                },
            ],
        });

        GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count,
            submeshes: Vec::new(),
            // Wireframe edges are built lazily on first use; the indices are
            // retained so any mesh can materialise them.
            edge_index_buffer: None,
            edge_index_count: 0,
            normal_line_buffer: None,
            normal_line_count: 0,
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
                0,
                0,
                0,
            ),
            normal_uniform_buf,
            normal_bind_group,
            aabb,
            cpu_positions: None,
            cpu_normals: None,
            cpu_indices: None,
            attribute_buffers: std::collections::HashMap::new(),
            attribute_ranges: std::collections::HashMap::new(),
            face_vertex_buffer: None,
            face_attribute_buffers: std::collections::HashMap::new(),
            face_colour_buffers: std::collections::HashMap::new(),
            vector_attribute_buffers: std::collections::HashMap::new(),
            position_override_buffer: None,
            position_override_slice: None,
            normal_override_buffer: None,
            normal_override_slice: None,
            extension_attr_buffer: None,
            lightmap: None,
            lightmap_gen: 0,
            position_override_gen: 0,
            normal_override_gen: 0,
            content_rev: 0,
            pick_trimesh_cache: std::sync::Mutex::new(None),
        }
    }

    // ---------------------------------------------------------------------------
    // Projected tetrahedra upload
    // ---------------------------------------------------------------------------

    /// Ensure the projected-tetrahedra bind group layout exists.
    ///
    /// No-op after the first call. Called internally by
    /// [`upload_volume_mesh_with_transparency`](Self::upload_volume_mesh_with_transparency)
    /// and [`ensure_pt_pipeline`](Self::ensure_pt_pipeline).
    ///
    /// Group 1 carries the per-volume uniform and the tet storage buffer.  The
    /// colourmap LUT lives in group 2 and is bound per-frame from the renderer's
    /// colourmap registry: see [`ensure_pt_lut_bind_group`](Self::ensure_pt_lut_bind_group).
    pub(crate) fn ensure_pt_bind_group_layout(&mut self, device: &crate::gpu::Device) {
        if self.pt.bind_group_layout.is_none() {
            let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("pt_bgl"),
                entries: &[
                    // binding 0: per-volume uniform (density, scalar_min, scalar_max, thresholds, flags)
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
                    // binding 1: tet geometry storage buffer (read-only)
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::gpu::ShaderStages::VERTEX
                            | crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: per-tet scalar storage buffer (read-only)
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: crate::gpu::ShaderStages::VERTEX
                            | crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            self.pt.bind_group_layout = Some(bgl);
        }
        if self.pt.lut_bind_group_layout.is_none() {
            // binding 0: colourmap texture (256x1 D2), binding 1: linear-clamp sampler.
            let bgl = crate::resources::builders::texture_sampler_bgl(
                device,
                "pt_lut_bgl",
                crate::gpu::ShaderStages::FRAGMENT,
            );
            self.pt.lut_bind_group_layout = Some(bgl);
        }
    }

    /// Build (and cache) the projected-tet LUT bind group for `colourmap_id`.
    ///
    /// Returns the bind group keyed by `colourmap_id.0`. Falls back to the
    /// fallback LUT (and its dedicated cached bind group) when the slot is empty.
    /// Callers must ensure the bind group layouts exist
    /// ([`ensure_pt_bind_group_layout`](Self::ensure_pt_bind_group_layout)).
    pub(crate) fn ensure_pt_lut_bind_group(
        &mut self,
        device: &crate::gpu::Device,
        colourmap_id: Option<crate::resources::ColourmapId>,
    ) -> &crate::gpu::BindGroup {
        let bgl = self
            .pt
            .lut_bind_group_layout
            .as_ref()
            .expect("pt_lut_bind_group_layout must exist");
        let sampler = &self.material_sampler;

        match colourmap_id.and_then(|id| self.content.colourmap_views.get(id.0).map(|_| id.0)) {
            Some(slot) => {
                if !self.pt.lut_bind_groups.contains_key(&slot) {
                    let lut_view = &self.content.colourmap_views[slot];
                    let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("pt_lut_bind_group"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::gpu::BindingResource::TextureView(lut_view),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: crate::gpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.pt.lut_bind_groups.insert(slot, bg);
                }
                self.pt.lut_bind_groups.get(&slot).unwrap()
            }
            None => {
                if self.pt.fallback_lut_bind_group.is_none() {
                    let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("pt_fallback_lut_bind_group"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::gpu::BindingResource::TextureView(
                                    &self.content.fallback_lut_view,
                                ),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: crate::gpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.pt.fallback_lut_bind_group = Some(bg);
                }
                self.pt.fallback_lut_bind_group.as_ref().unwrap()
            }
        }
    }

    /// Decompose all cells in `data` into tetrahedra and upload to the GPU.
    ///
    /// `scalar_attribute` names a key in `data.cell_scalars`; cells without the attribute
    /// get scalar 0.0.  The scalar range is auto-detected from the data.
    ///
    /// Returns a [`ProjectedTetId`] that can be placed in a
    /// [`VolumeMeshItem::projected_tet_id`](crate::renderer::types::VolumeMeshItem::projected_tet_id)
    /// each frame to enable the volumetric transparent render mode.
    /// Upload a projected-tet mesh and return both the GPU handle and the actual scalar
    /// range stored in the GPU buffer. Callers should use the returned scalar range for
    /// threshold computations so that brimcast and the GPU always agree on the data range
    /// (including the constant-data `scalar_min + 1.0` adjustment in `decompose_into_chunks`).
    pub(crate) fn upload_projected_tet(
        &mut self,
        device: &crate::gpu::Device,
        data: &crate::resources::volume::volume_mesh::VolumeMeshData,
        scalar_attribute: &str,
    ) -> crate::error::ViewportResult<(ProjectedTetId, f32, f32)> {
        self.ensure_pt_bind_group_layout(device);
        // Build the projected-tet pipeline now so a load-time upload also pays
        // the pipeline compile, not the first frame drawn in transparent mode.
        self.ensure_pt_pipeline(device);

        let (pending, scalar_range, uniform_buffer) =
            Self::decompose_into_chunks(device, data, scalar_attribute);

        // Build bind groups: one per chunk, all sharing the same uniform buffer.
        let chunks = {
            let bgl = self
                .pt
                .bind_group_layout
                .as_ref()
                .expect("pt_bind_group_layout must exist after ensure_pt_bind_group_layout");
            pending
                .into_iter()
                .map(|(tet_buffer, scalar_buffer, tet_count)| {
                    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("pt_bind_group"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buffer.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: tet_buffer.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 2,
                                resource: scalar_buffer.as_entire_binding(),
                            },
                        ],
                    });
                    crate::resources::types::ProjectedTetChunk {
                        tet_buffer,
                        scalar_buffer,
                        tet_count,
                        bind_group,
                    }
                })
                .collect::<Vec<_>>()
        };

        let mesh = GpuProjectedTetMesh {
            chunks,
            uniform_buffer,
            scalar_range,
        };
        let bytes = mesh.gpu_bytes();
        let id = self.content.projected_tet_store.insert(mesh, bytes);
        Ok((id, scalar_range.0, scalar_range.1))
    }

    /// Start an asynchronous projected-tet mesh upload.
    ///
    /// Slab decomposition (`decompose_into_chunks`) and the per-chunk tet
    /// storage buffers are built on a worker thread on a cloned `Device`.
    /// The apply step constructs the per-chunk bind groups against the
    /// renderer's existing `pt_bind_group_layout` and colourmap LUT, then
    /// inserts the mesh. Returns the [`JobId`](crate::resources::JobId);
    /// take the `(ProjectedTetId, scalar_min, scalar_max)` triple via
    /// [`upload_result_projected_tet`](Self::upload_result_projected_tet).
    #[allow(dead_code)]
    pub(crate) fn begin_upload_projected_tet(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::volume::volume_mesh::VolumeMeshData,
        scalar_attribute: String,
    ) -> crate::resources::JobId {
        // Pipeline layout must exist when the apply step builds bind groups.
        self.ensure_pt_bind_group_layout(device);

        let slot = crate::resources::ResultSlot::<(ProjectedTetId, f32, f32)>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let device_for_apply = device.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let (pending, scalar_range, uniform_buffer) =
                    DeviceResources::decompose_into_chunks(
                        &device_for_worker,
                        &data,
                        &scalar_attribute,
                    );
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let chunks = {
                            let bgl = resources
                                .pt
                                .bind_group_layout
                                .as_ref()
                                .expect("pt_bind_group_layout must exist");
                            pending
                                .into_iter()
                                .map(|(tet_buffer, scalar_buffer, tet_count)| {
                                    let bind_group = device_for_apply.create_bind_group(
                                        &crate::gpu::BindGroupDescriptor {
                                            label: Some("pt_bind_group"),
                                            layout: bgl,
                                            entries: &[
                                                crate::gpu::BindGroupEntry {
                                                    binding: 0,
                                                    resource: uniform_buffer.as_entire_binding(),
                                                },
                                                crate::gpu::BindGroupEntry {
                                                    binding: 1,
                                                    resource: tet_buffer.as_entire_binding(),
                                                },
                                                crate::gpu::BindGroupEntry {
                                                    binding: 2,
                                                    resource: scalar_buffer.as_entire_binding(),
                                                },
                                            ],
                                        },
                                    );
                                    crate::resources::types::ProjectedTetChunk {
                                        tet_buffer,
                                        scalar_buffer,
                                        tet_count,
                                        bind_group,
                                    }
                                })
                                .collect::<Vec<_>>()
                        };
                        let mesh = GpuProjectedTetMesh {
                            chunks,
                            uniform_buffer,
                            scalar_range,
                        };
                        let bytes = mesh.gpu_bytes();
                        let pid = resources.content.projected_tet_store.insert(mesh, bytes);
                        slot_for_apply.set((pid, scalar_range.0, scalar_range.1));
                    }),
                ))
            })
        };

        self.job_results
            .projected_tet
            .lock()
            .expect("projected tet result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the `(ProjectedTetId, scalar_min, scalar_max)` triple produced by a
    /// completed [`begin_upload_projected_tet`](Self::begin_upload_projected_tet) job.
    #[allow(dead_code)]
    pub(crate) fn upload_result_projected_tet(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<(ProjectedTetId, f32, f32)> {
        let mut map = self
            .job_results
            .projected_tet
            .lock()
            .expect("projected tet result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(triple) => {
                map.remove(&id);
                Ok(triple)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Replace the tet buffer of an existing projected-tet mesh in-place.
    ///
    /// Rebuilds the tet storage buffer (and its bind group) from the new scalar
    /// attribute. The uniform buffer (density, thresholds, opacity) is reused;
    /// only the cached scalar range is refreshed. Changing the colourmap on the
    /// owning item is now free because the LUT is bound per-frame in render.rs,
    /// so this call no longer takes a `colourmap_id`.
    pub fn replace_projected_tet(
        &mut self,
        device: &crate::gpu::Device,
        id: crate::resources::ProjectedTetId,
        data: &crate::resources::volume::volume_mesh::VolumeMeshData,
        scalar_attribute: &str,
    ) -> crate::error::ViewportResult<()> {
        self.ensure_pt_bind_group_layout(device);

        let (pending, scalar_range, _new_uniform) =
            Self::decompose_into_chunks(device, data, scalar_attribute);

        let chunks = {
            let bgl = self
                .pt
                .bind_group_layout
                .as_ref()
                .expect("pt_bind_group_layout must exist after ensure_pt_bind_group_layout");
            let uniform_buf = &self
                .content
                .projected_tet_store
                .get(id)
                .expect("ProjectedTetId must reference an uploaded mesh")
                .uniform_buffer;
            pending
                .into_iter()
                .map(|(tet_buffer, scalar_buffer, tet_count)| {
                    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("pt_bind_group"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: tet_buffer.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 2,
                                resource: scalar_buffer.as_entire_binding(),
                            },
                        ],
                    });
                    crate::resources::types::ProjectedTetChunk {
                        tet_buffer,
                        scalar_buffer,
                        tet_count,
                        bind_group,
                    }
                })
                .collect::<Vec<_>>()
        };

        let slot = self
            .content
            .projected_tet_store
            .get_mut(id)
            .expect("ProjectedTetId must reference an uploaded mesh");
        slot.chunks = chunks;
        slot.scalar_range = scalar_range;
        // The rebuilt chunks may differ in size from the previous ones, so keep
        // the store's byte charge in step with the new geometry.
        let bytes = slot.gpu_bytes();
        self.content.projected_tet_store.set_bytes(id, bytes);

        Ok(())
    }

    /// Refresh a projected-tet mesh's scalar field in place, without
    /// re-decomposing the cells or re-uploading any tet geometry.
    ///
    /// For a time-series where a scalar field animates on a **fixed** tet mesh
    /// (a CFD/FE field played back over a static unstructured grid), the tet
    /// geometry does not change frame to frame -- only the per-tet scalar does.
    /// [`replace_projected_tet`](Self::replace_projected_tet) rebuilds the whole
    /// tet buffer (all four vertex positions per tet) every call; this rewrites
    /// only the parallel per-tet scalar buffer (one `f32` per tet, 16x less data
    /// than the geometry), leaving the geometry buffers, bind groups, and uniform
    /// untouched. That avoids the redundant geometry re-upload -- the dominant
    /// cost on discrete GPUs, where it crosses PCIe every frame.
    ///
    /// `data` must be the same mesh (same cells, same order) the projected-tet
    /// was uploaded from; only the `cell_scalars` values may differ. The per-cell
    /// scalars are expanded to per-tet scalars through the cell -> tet fan-out
    /// (hex -> 6, wedge -> 3, pyramid -> 2, tet -> 1), matching the tet order the
    /// geometry buffer uses.
    ///
    /// # Errors
    ///
    /// - [`SlotEmpty`](crate::error::ViewportError::SlotEmpty) : `id` does not reference an uploaded mesh.
    /// - [`AttributeLengthMismatch`](crate::error::ViewportError::AttributeLengthMismatch) : `data`
    ///   decomposes to a different tet count than the uploaded mesh (a different
    ///   topology; use [`replace_projected_tet`](Self::replace_projected_tet) for that).
    pub fn replace_projected_tet_scalar(
        &mut self,
        queue: &crate::gpu::Queue,
        id: crate::resources::ProjectedTetId,
        data: &crate::resources::volume::volume_mesh::VolumeMeshData,
        scalar_attribute: &str,
    ) -> crate::error::ViewportResult<()> {
        let scalars = crate::resources::volume::volume_mesh::tet_scalars(data, scalar_attribute);

        let gpu = self
            .content
            .projected_tet_store
            .get_mut(id)
            .ok_or(crate::error::ViewportError::SlotEmpty { index: id.index() })?;

        let total: usize = gpu.chunks.iter().map(|c| c.tet_count as usize).sum();
        if scalars.len() != total {
            return Err(crate::error::ViewportError::AttributeLengthMismatch {
                expected: total,
                got: scalars.len(),
            });
        }

        // Write each chunk's scalar slice; geometry and bind groups are untouched.
        let mut off = 0usize;
        for chunk in &gpu.chunks {
            let n = chunk.tet_count as usize;
            queue.write_buffer(
                &chunk.scalar_buffer,
                0,
                bytemuck::cast_slice(&scalars[off..off + n]),
            );
            off += n;
        }

        // Refresh the cached scalar range so the colourmap mapping stays accurate
        // (matches the range `decompose_into_chunks` derives).
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &s in &scalars {
            min = min.min(s);
            max = max.max(s);
        }
        gpu.scalar_range = if min.is_infinite() {
            (0.0, 1.0)
        } else if (max - min).abs() < 1e-12 {
            (min, min + 1.0)
        } else {
            (min, max)
        };

        Ok(())
    }

    /// Free the projected-tet mesh behind `id`, reclaiming its slot, its tet and
    /// scalar buffers, and its byte charge.
    ///
    /// A transparent time-series that uploads a fresh mesh per timestep (rather
    /// than refreshing one in place through
    /// [`replace_projected_tet`](Self::replace_projected_tet) /
    /// [`replace_projected_tet_scalar`](Self::replace_projected_tet_scalar))
    /// should free the previous handle here so the tet buffers do not accumulate.
    /// A later upload reuses the freed slot, and any handle still holding `id`
    /// resolves to nothing afterwards. Returns `false` if `id` was already freed
    /// or is stale.
    pub fn free_projected_tet(&mut self, id: crate::resources::ProjectedTetId) -> bool {
        self.content.projected_tet_store.remove(id).is_some()
    }

    /// Decompose `data` into device-limit-bounded tet buffers and a shared uniform buffer.
    ///
    /// Returns `(pending_chunks, scalar_range, uniform_buffer)` where each element of
    /// `pending_chunks` is a `(geometry_buffer, scalar_buffer, tet_count)` triple ready
    /// for bind group creation. The geometry buffer holds vertex positions only (per-tet
    /// scalars live in the parallel `scalar_buffer`, so a scalar-only refresh via
    /// [`replace_projected_tet_scalar`](Self::replace_projected_tet_scalar) rewrites just
    /// the scalar buffer and leaves geometry untouched). Bind groups are created separately
    /// so callers can supply the correct uniform buffer reference (new for upload, existing
    /// for replace).
    fn decompose_into_chunks(
        device: &crate::gpu::Device,
        data: &crate::resources::volume::volume_mesh::VolumeMeshData,
        scalar_attribute: &str,
    ) -> (
        Vec<(crate::gpu::Buffer, crate::gpu::Buffer, u32)>,
        (f32, f32),
        crate::gpu::Buffer,
    ) {
        // Determine the maximum tets per chunk from device limits.
        // Each tet is 64 bytes (4 x vec4<f32>) of geometry; the scalar buffer is
        // 4 bytes/tet, far under the limit, so the same chunking bounds both.
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        let max_buf = device.limits().max_buffer_size;
        let chunk_size_tets = ((max_binding.min(max_buf)) / 64).max(1) as usize;

        let mut pending: Vec<(crate::gpu::Buffer, crate::gpu::Buffer, u32)> = Vec::new();
        let mut current_raw: Vec<f32> = Vec::with_capacity(chunk_size_tets * 16);
        let mut current_scalars: Vec<f32> = Vec::with_capacity(chunk_size_tets);
        let mut scalar_min = f32::INFINITY;
        let mut scalar_max = f32::NEG_INFINITY;

        let flush =
            |raw: &mut Vec<f32>,
             scal: &mut Vec<f32>,
             pending: &mut Vec<(crate::gpu::Buffer, crate::gpu::Buffer, u32)>| {
                let tet_count = scal.len() as u32;
                let geom = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("pt_tet_buffer"),
                    size: (raw.len() * std::mem::size_of::<f32>()) as u64,
                    usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                crate::resources::builders::write_mapped(geom.slice(..), bytemuck::cast_slice(raw));
                geom.unmap();
                let scalar_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("pt_scalar_buffer"),
                    size: (scal.len() * std::mem::size_of::<f32>()) as u64,
                    usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                crate::resources::builders::write_mapped(
                    scalar_buf.slice(..),
                    bytemuck::cast_slice(scal),
                );
                scalar_buf.unmap();
                pending.push((geom, scalar_buf, tet_count));
                raw.clear();
                scal.clear();
            };

        crate::resources::volume::volume_mesh::for_each_tet(
            data,
            scalar_attribute,
            |verts, scalar| {
                scalar_min = scalar_min.min(scalar);
                scalar_max = scalar_max.max(scalar);
                // Geometry only; scalar goes to the parallel buffer (v0.w now pad).
                current_raw.extend_from_slice(&[verts[0][0], verts[0][1], verts[0][2], 0.0]);
                current_raw.extend_from_slice(&[verts[1][0], verts[1][1], verts[1][2], 0.0]);
                current_raw.extend_from_slice(&[verts[2][0], verts[2][1], verts[2][2], 0.0]);
                current_raw.extend_from_slice(&[verts[3][0], verts[3][1], verts[3][2], 0.0]);
                current_scalars.push(scalar);
                if current_scalars.len() == chunk_size_tets {
                    flush(&mut current_raw, &mut current_scalars, &mut pending);
                }
            },
        );

        if !current_scalars.is_empty() {
            flush(&mut current_raw, &mut current_scalars, &mut pending);
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
            unlit: 0,
            opacity: 1.0,
            _pad: 0.0,
        };
        let uniform_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pt_uniform_buf"),
            size: std::mem::size_of::<crate::resources::types::ProjectedTetUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            uniform_buffer.slice(..),
            bytemuck::bytes_of(&initial_uniform),
        );
        uniform_buffer.unmap();

        (pending, scalar_range, uniform_buffer)
    }
}

#[cfg(test)]
mod override_tests {
    use super::{OverrideBufferSlice, SubmeshRange};
    use crate::DeviceResources;
    use crate::geometry::primitives;

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

    fn dummy_override_buffer(
        device: &crate::gpu::Device,
        vertex_count: usize,
    ) -> crate::gpu::Buffer {
        device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("test_override_buf"),
            size: (vertex_count * 12) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn set_position_override_roundtrip() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let vertex_count = plane.positions.len();

        // Initial state.
        {
            let mesh = resources.mesh_store.get(mesh_id).unwrap();
            assert!(mesh.position_override_buffer.is_none());
            let gen0 = mesh.position_override_gen;
            assert_eq!(gen0, 0);
        }

        let buf = dummy_override_buffer(&device, vertex_count);
        resources
            .set_position_override_buffer(mesh_id, buf)
            .unwrap();

        {
            let mesh = resources.mesh_store.get(mesh_id).unwrap();
            assert!(mesh.position_override_buffer.is_some());
            assert_eq!(mesh.position_override_gen, 1);
        }

        resources.clear_position_override(mesh_id).unwrap();
        {
            let mesh = resources.mesh_store.get(mesh_id).unwrap();
            assert!(mesh.position_override_buffer.is_none());
            assert_eq!(mesh.position_override_gen, 2);
        }
    }

    #[test]
    fn set_normal_override_roundtrip() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let vertex_count = plane.positions.len();

        let buf = dummy_override_buffer(&device, vertex_count);
        resources.set_normal_override_buffer(mesh_id, buf).unwrap();
        {
            let mesh = resources.mesh_store.get(mesh_id).unwrap();
            assert!(mesh.normal_override_buffer.is_some());
            assert_eq!(mesh.normal_override_gen, 1);
        }

        resources.clear_normal_override(mesh_id).unwrap();
        {
            let mesh = resources.mesh_store.get(mesh_id).unwrap();
            assert!(mesh.normal_override_buffer.is_none());
            assert_eq!(mesh.normal_override_gen, 2);
        }
    }

    #[test]
    fn set_position_override_buffer_sliced_roundtrip() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let vertex_count = plane.positions.len();

        // Pool holding this mesh's window plus 8 extra elements in front.
        let pool = dummy_override_buffer(&device, vertex_count + 8);
        let slice = OverrideBufferSlice::new(8, vertex_count as u32);
        resources
            .set_position_override_buffer_sliced(mesh_id, pool.clone(), slice)
            .unwrap();
        {
            let mesh = resources.mesh_store.get(mesh_id).unwrap();
            assert!(mesh.position_override_buffer.is_some());
            assert_eq!(mesh.position_override_slice, Some(slice));
            assert_eq!(mesh.position_override_gen, 1);
        }

        // The unsliced setter resets the window.
        resources
            .set_position_override_buffer(mesh_id, pool.clone())
            .unwrap();
        {
            let mesh = resources.mesh_store.get(mesh_id).unwrap();
            assert_eq!(mesh.position_override_slice, None);
            assert_eq!(mesh.position_override_gen, 2);
        }

        // Clearing resets both buffer and window.
        resources
            .set_normal_override_buffer_sliced(mesh_id, pool, slice)
            .unwrap();
        resources.clear_normal_override(mesh_id).unwrap();
        {
            let mesh = resources.mesh_store.get(mesh_id).unwrap();
            assert!(mesh.normal_override_buffer.is_none());
            assert_eq!(mesh.normal_override_slice, None);
        }
    }

    #[test]
    fn sliced_override_rejects_out_of_range_slice() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();

        // 4-element buffer cannot hold a window ending at element 6.
        let buf = dummy_override_buffer(&device, 4);
        let err = resources.set_position_override_buffer_sliced(
            mesh_id,
            buf,
            OverrideBufferSlice::new(2, 4),
        );
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::OverrideSliceOutOfRange {
                base_element: 2,
                element_count: 4,
                buffer_elements: 4,
            })
        ));
        // The failed call must not have bound anything.
        let mesh = resources.mesh_store.get(mesh_id).unwrap();
        assert!(mesh.position_override_buffer.is_none());
        assert_eq!(mesh.position_override_gen, 0);
    }

    #[test]
    fn replace_mesh_data_preserves_override_on_topology_change() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let vertex_count = plane.positions.len();

        let pool = dummy_override_buffer(&device, vertex_count + 8);
        let slice = OverrideBufferSlice::new(3, vertex_count as u32);
        resources
            .set_position_override_buffer_sliced(mesh_id, pool, slice)
            .unwrap();
        let gen_before = resources
            .mesh_store
            .get(mesh_id)
            .unwrap()
            .position_override_gen;

        // Different vertex count forces the topology-change (full rebuild)
        // path instead of the in-place write.
        let denser = primitives::grid_plane(1.0, 1.0, 3, 3);
        assert_ne!(denser.positions.len(), vertex_count);
        resources
            .replace_mesh_data(&device, &queue, mesh_id, &denser)
            .unwrap();

        let mesh = resources.mesh_store.get(mesh_id).unwrap();
        assert!(
            mesh.position_override_buffer.is_some(),
            "topology-change replace must carry the override binding, not \
             revert to the fallback"
        );
        assert_eq!(mesh.position_override_slice, Some(slice));
        assert!(
            mesh.position_override_gen > gen_before,
            "gen must advance past the old mesh's so the rebuilt bind group \
             picks up the carried buffer"
        );
    }

    #[test]
    fn sort_triangles_into_submeshes_groups_and_permutes() {
        use crate::resources::AttributeData;

        // Four triangles over six vertices, materials interleaved 2,1,2,1.
        let mut data = super::MeshData::default();
        data.positions = vec![[0.0, 0.0, 0.0]; 6];
        data.normals = vec![[0.0, 0.0, 1.0]; 6];
        data.indices = vec![0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5];
        data.attributes.insert(
            "face_val".to_string(),
            AttributeData::Face(vec![10.0, 11.0, 12.0, 13.0]),
        );
        data.attributes.insert(
            "corner_val".to_string(),
            AttributeData::Corner((0..12).map(|i| i as f32).collect()),
        );

        let ids = data.sort_triangles_into_submeshes(&[2, 1, 2, 1]).unwrap();

        // Distinct ids ascending; stable order within each material.
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(
            data.submeshes,
            vec![
                super::SubmeshRange {
                    first_index: 0,
                    index_count: 6
                },
                super::SubmeshRange {
                    first_index: 6,
                    index_count: 6
                },
            ]
        );
        // Material 1 = original triangles 1, 3; material 2 = triangles 0, 2.
        assert_eq!(data.indices, vec![1, 2, 3, 3, 4, 5, 0, 1, 2, 2, 3, 4]);
        let Some(AttributeData::Face(face)) = data.attributes.get("face_val") else {
            panic!("face attribute lost");
        };
        assert_eq!(face, &vec![11.0, 13.0, 10.0, 12.0]);
        let Some(AttributeData::Corner(corner)) = data.attributes.get("corner_val") else {
            panic!("corner attribute lost");
        };
        assert_eq!(
            corner,
            &vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn sort_triangles_into_submeshes_rejects_count_mismatch() {
        let mut data = super::MeshData::default();
        data.positions = vec![[0.0, 0.0, 0.0]; 3];
        data.normals = vec![[0.0, 0.0, 1.0]; 3];
        data.indices = vec![0, 1, 2];
        let err = data.sort_triangles_into_submeshes(&[0, 1]);
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::SubmeshTriangleCountMismatch {
                triangles: 1,
                material_ids: 2,
            })
        ));
    }

    #[test]
    fn submesh_range_past_index_buffer_rejected() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let index_count = plane.indices.len() as u32;
        plane.submeshes = vec![SubmeshRange {
            first_index: 3,
            index_count,
        }];
        let err = resources.upload_mesh_data(&device, &plane);
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::SubmeshRangeOutOfBounds { first_index: 3, .. })
        ));
    }

    #[test]
    fn submesh_ranges_stored_on_upload_and_replace() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let index_count = plane.indices.len() as u32;
        let ranges = vec![
            SubmeshRange {
                first_index: 0,
                index_count: 3,
            },
            SubmeshRange {
                first_index: 3,
                index_count: index_count - 3,
            },
        ];
        plane.submeshes = ranges.clone();
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        assert_eq!(resources.mesh_store.get(mesh_id).unwrap().submeshes, ranges);

        // Same topology takes the in-place path; ranges must still update.
        plane.submeshes = vec![SubmeshRange {
            first_index: 0,
            index_count,
        }];
        resources
            .replace_mesh_data(&device, &queue, mesh_id, &plane)
            .unwrap();
        assert_eq!(
            resources.mesh_store.get(mesh_id).unwrap().submeshes,
            plane.submeshes
        );
    }

    #[test]
    fn lod_group_rejects_submesh_range_count_mismatch() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mut ranged = primitives::grid_plane(1.0, 1.0, 3, 3);
        ranged.submeshes = vec![
            SubmeshRange {
                first_index: 0,
                index_count: 3,
            },
            SubmeshRange {
                first_index: 3,
                index_count: 3,
            },
        ];
        let level0 = resources.upload_mesh_data(&device, &ranged).unwrap();
        let plain = primitives::grid_plane(1.0, 1.0, 2, 2);
        let level1 = resources.upload_mesh_data(&device, &plain).unwrap();

        let err = resources.register_lod_group(&[level0, level1], &[0.5, 0.1]);
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::LodLevelIncompatible { level: 1, .. })
        ));
    }

    #[test]
    fn update_vertex_colours_writes_in_range_and_rejects_overflow() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let plane = primitives::grid_plane(1.0, 1.0, 2, 2);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let vertex_count = plane.positions.len();

        // A run inside the mesh is accepted.
        let colours = vec![[1.0f32, 0.0, 0.0, 1.0]; vertex_count - 1];
        resources
            .update_vertex_colours(&queue, mesh_id, 1, &colours)
            .unwrap();

        // Empty write is a no-op success.
        resources
            .update_vertex_colours(&queue, mesh_id, 0, &[])
            .unwrap();

        // A run that would extend past the last vertex is rejected.
        let err =
            resources.update_vertex_colours(&queue, mesh_id, vertex_count, &[[0.0, 0.0, 1.0, 1.0]]);
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::MeshLengthMismatch { .. })
        ));

        // Unknown mesh id is rejected.
        let bogus = crate::resources::mesh::mesh_store::MeshId::new(9999, 0);
        let err = resources.update_vertex_colours(&queue, bogus, 0, &[[1.0, 1.0, 1.0, 1.0]]);
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::StaleHandle { .. })
        ));
    }

    #[test]
    fn override_on_unknown_mesh_id_errors() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        // Fabricate an id that is beyond the store.
        let bogus = crate::resources::mesh::mesh_store::MeshId::new(9999, 0);
        let buf = dummy_override_buffer(&device, 4);
        let err = resources.set_position_override_buffer(bogus, buf);
        assert!(matches!(
            err,
            Err(crate::error::ViewportError::StaleHandle { .. })
        ));
    }

    #[test]
    fn stale_mesh_handle_does_not_alias_after_slot_reuse() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // Upload a mesh, then remove it. The handle is now stale.
        let id1 = resources
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .unwrap();
        assert!(resources.mesh_store.get(id1).is_some());
        assert!(resources.free_mesh(id1));
        assert!(
            resources.mesh_store.get(id1).is_none(),
            "a removed handle must not resolve"
        );

        // The next upload reuses the freed slot but at a new generation.
        let id2 = resources
            .upload_mesh_data(&device, &primitives::cube(2.0))
            .unwrap();
        assert_eq!(id1.index(), id2.index(), "the freed slot should be reused");
        assert_ne!(
            id1, id2,
            "the reused slot must carry a new generation so the old handle differs"
        );
        assert!(resources.mesh_store.get(id2).is_some());
        assert!(
            resources.mesh_store.get(id1).is_none(),
            "the stale handle must not alias the mesh now occupying its slot"
        );
    }

    #[test]
    fn stale_handle_replace_is_rejected_after_free_and_reuse() {
        // The in-flight guard: an operation carrying a handle whose mesh was
        // freed (and whose slot a later upload reused) must not land on the new
        // mesh. `replace_mesh_data` is the slot-targeting path; its generation
        // check is what makes a free racing a queued replace safe.
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let stale = resources
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .unwrap();
        assert!(resources.free_mesh(stale));

        // A later upload reuses the freed slot at a new generation.
        let live = resources
            .upload_mesh_data(&device, &primitives::cube(2.0))
            .unwrap();
        assert_eq!(stale.index(), live.index());

        // Replaying the stale handle must be rejected, not silently overwrite
        // the mesh now occupying the slot.
        let err = resources.replace_mesh_data(&device, &queue, stale, &primitives::cube(3.0));
        assert!(
            matches!(err, Err(crate::error::ViewportError::StaleHandle { .. })),
            "replace through a stale handle must fail rather than alias the reused slot"
        );
        assert!(
            resources.mesh_store.get(live).is_some(),
            "the live mesh must be untouched by the rejected replace"
        );
    }

    #[test]
    fn resident_bytes_track_mesh_upload_and_free() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let start = resources.resident_bytes().mesh_bytes;
        let id = resources
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .unwrap();
        let after_upload = resources.resident_bytes().mesh_bytes;
        assert!(
            after_upload > start,
            "uploading a mesh must increase resident mesh bytes"
        );

        assert!(resources.free_mesh(id));
        let after_free = resources.resident_bytes().mesh_bytes;
        assert_eq!(
            after_free, start,
            "freeing the mesh must return resident bytes to the starting total"
        );
    }
}

#[cfg(test)]
mod vertex_colour_tests {
    use crate::DeviceResources;
    use crate::resources::MeshData;

    fn tri() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![0, 1, 2],
            ..MeshData::default()
        }
    }

    #[test]
    fn none_leaves_vertices_white() {
        let prep = DeviceResources::prep_mesh_data(&tri());
        assert!(
            prep.vertices
                .iter()
                .all(|v| v.colour == [1.0, 1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn per_vertex_colours_are_threaded_through() {
        let mut data = tri();
        data.vertex_colours = Some(vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.5],
        ]);
        let prep = DeviceResources::prep_mesh_data(&data);
        assert_eq!(prep.vertices[0].colour, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(prep.vertices[1].colour, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(prep.vertices[2].colour, [0.0, 0.0, 1.0, 0.5]);
    }

    #[test]
    fn short_colour_slice_defaults_missing_entries_white() {
        let mut data = tri();
        data.vertex_colours = Some(vec![[0.2, 0.4, 0.6, 1.0]]);
        let prep = DeviceResources::prep_mesh_data(&data);
        assert_eq!(prep.vertices[0].colour, [0.2, 0.4, 0.6, 1.0]);
        assert_eq!(prep.vertices[1].colour, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(prep.vertices[2].colour, [1.0, 1.0, 1.0, 1.0]);
    }
}

#[cfg(test)]
mod async_upload_tests {
    use crate::DeviceResources;
    use crate::geometry::primitives;
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
        panic!("mesh upload did not complete in time");
    }

    #[test]
    fn invalid_mesh_data_errors_synchronously() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let empty = crate::resources::MeshData::default();
        let err = resources
            .begin_upload_mesh_data(&device, empty)
            .expect_err("empty mesh should be rejected");
        assert!(matches!(err, crate::error::ViewportError::EmptyMesh { .. }));
        assert_eq!(resources.uploads_pending(), 0);
    }

    #[test]
    fn begin_upload_completes_and_yields_mesh_id() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let plane = primitives::grid_plane(1.0, 1.0, 8, 8);
        let id = resources
            .begin_upload_mesh_data(&device, plane.clone())
            .unwrap();
        assert_eq!(resources.uploads_pending(), 1);

        // Result should not be available until the worker finishes.
        let err = resources.upload_result_mesh(id).unwrap_err();
        assert!(matches!(err, crate::error::ViewportError::JobNotReady));

        drive_until_ready(&mut resources, &device, &queue, id);

        let mesh_id = resources.upload_result_mesh(id).expect("ready result");
        assert!(resources.mesh_store.get(mesh_id).is_some());

        // Second take of the same id should now report missing.
        let err = resources.upload_result_mesh(id).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::JobResultMissing { .. }
        ));
    }

    #[test]
    fn sync_upload_still_works_alongside_async() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        assert!(resources.mesh_store.get(mesh_id).is_some());
    }

    #[test]
    fn unknown_job_id_returns_missing() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // Submit an env-map job so we have a live JobId of the wrong type.
        let pixels = vec![0.5f32; 8 * 4 * 4];
        let other_id = crate::resources::material::environment::begin_upload_environment_map(
            &mut resources,
            &device,
            &try_make_device().unwrap().1,
            pixels,
            8,
            4,
        )
        .unwrap();

        let err = resources.upload_result_mesh(other_id).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::JobResultMissing { .. }
        ));
    }
}

#[cfg(test)]
mod c4_volume_mesh_tests {
    use crate::DeviceResources;
    use crate::resources::volume::volume_mesh::VolumeMeshData;
    use crate::resources::{CELL_SENTINEL, SparseVolumeGridData, UploadStatus};

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

    fn single_tet_volume() -> VolumeMeshData {
        let mut v = VolumeMeshData::default();
        v.positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        v.cells = vec![[
            0,
            1,
            2,
            3,
            CELL_SENTINEL,
            CELL_SENTINEL,
            CELL_SENTINEL,
            CELL_SENTINEL,
        ]];
        v.cell_scalars.insert("density".into(), vec![0.5]);
        v
    }

    fn single_cell_sparse() -> SparseVolumeGridData {
        let mut g = SparseVolumeGridData::default();
        g.active_cells = vec![[0, 0, 0]];
        g.cell_size = 1.0;
        g.origin = [0.0, 0.0, 0.0];
        g
    }

    #[test]
    fn begin_upload_volume_mesh_drains_to_pair() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_volume_mesh(&device, single_tet_volume());
        drive_until_ready(&mut resources, &device, &queue, job, "volume_mesh");
        let item = resources.upload_result_volume_mesh(job).expect("ready");
        assert!(resources.mesh_store.get(item.boundary_mesh_id).is_some());
        assert!(!item.face_to_cell.is_empty());
        let res = resources.upload_result_volume_mesh(job);
        assert!(matches!(
            res,
            Err(crate::error::ViewportError::JobResultMissing { .. })
        ));
    }

    #[test]
    fn begin_upload_clipped_volume_mesh_drains_to_pair() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        // Empty clip planes: equivalent to plain volume mesh extraction.
        let job =
            resources.begin_upload_clipped_volume_mesh(&device, single_tet_volume(), Vec::new());
        drive_until_ready(&mut resources, &device, &queue, job, "clipped_volume_mesh");
        let item = resources
            .upload_result_clipped_volume_mesh(job)
            .expect("ready");
        assert!(resources.mesh_store.get(item.boundary_mesh_id).is_some());
        assert!(!item.face_to_cell.is_empty());
    }

    #[test]
    fn begin_upload_sparse_volume_grid_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_sparse_volume_grid_data(&device, single_cell_sparse());
        drive_until_ready(&mut resources, &device, &queue, job, "sparse_volume_grid");
        let mesh_id = resources
            .upload_result_sparse_volume_grid(job)
            .expect("ready");
        assert!(resources.mesh_store.get(mesh_id).is_some());
    }

    #[test]
    fn begin_upload_projected_tet_drains_to_triple() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job =
            resources.begin_upload_projected_tet(&device, single_tet_volume(), "density".into());
        drive_until_ready(&mut resources, &device, &queue, job, "projected_tet");
        let (_id, smin, smax) = resources.upload_result_projected_tet(job).expect("ready");
        assert!(smin <= smax);
    }

    #[test]
    fn sync_paths_still_work() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let vol = single_tet_volume();
        let _item = resources
            .upload_volume_mesh(&device, &vol)
            .expect("sync ok");
        let _item2 = resources
            .upload_clipped_volume_mesh(&device, &vol, &[])
            .expect("clipped sync ok");
        let _grid_id = resources
            .upload_sparse_volume_grid_data(&device, &single_cell_sparse())
            .expect("sparse sync ok");
    }

    #[test]
    fn upload_projected_tet_charges_resident_bytes() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        assert_eq!(resources.resident_bytes().projected_tet_bytes, 0);
        let (_id, _, _) = resources
            .upload_projected_tet(&device, &single_tet_volume(), "density")
            .expect("upload ok");
        assert!(resources.resident_bytes().projected_tet_bytes > 0);
    }

    #[test]
    fn free_projected_tet_reclaims_bytes_and_slot() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let (id, _, _) = resources
            .upload_projected_tet(&device, &single_tet_volume(), "density")
            .expect("upload ok");
        let charged = resources.resident_bytes().projected_tet_bytes;
        assert!(charged > 0);

        assert!(resources.free_projected_tet(id));
        assert_eq!(resources.resident_bytes().projected_tet_bytes, 0);
        // Second free of the same handle is a no-op.
        assert!(!resources.free_projected_tet(id));

        // The freed slot is reused by the next upload; the charge matches one
        // mesh, not two.
        let (_id2, _, _) = resources
            .upload_projected_tet(&device, &single_tet_volume(), "density")
            .expect("upload ok");
        assert_eq!(resources.resident_bytes().projected_tet_bytes, charged);
    }

    #[test]
    fn stale_projected_tet_handle_does_not_alias_reused_slot() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let (old, _, _) = resources
            .upload_projected_tet(&device, &single_tet_volume(), "density")
            .expect("upload ok");
        assert!(resources.free_projected_tet(old));
        let (new, _, _) = resources
            .upload_projected_tet(&device, &single_tet_volume(), "density")
            .expect("upload ok");
        assert_eq!(old.index(), new.index(), "same slot reused");
        assert_ne!(old, new, "generation bumped so the stale handle differs");
        // The stale handle no longer resolves for a scalar refresh.
        assert!(matches!(
            resources.replace_projected_tet_scalar(&queue, old, &single_tet_volume(), "density"),
            Err(crate::error::ViewportError::SlotEmpty { .. })
        ));
        // The live handle still works.
        assert!(
            resources
                .replace_projected_tet_scalar(&queue, new, &single_tet_volume(), "density")
                .is_ok()
        );
    }
}

/// A contiguous run of the index buffer drawn with its own material.
///
/// Ranges partition a mesh so different parts of it can bind different
/// materials: `SceneRenderItem::submesh_materials[i]` supplies the material
/// for `submeshes[i]`. The caller must sort triangles by material before
/// building ranges, so each material's indices form one contiguous run.
/// Asset importers normally do this at import time; for interleaved input,
/// [`MeshData::sort_triangles_into_submeshes`] does the sort and builds the
/// ranges.
///
/// Ranges are expected to partition the index buffer. Overlapping ranges
/// are not rejected, but the overlapped triangles draw once per range that
/// covers them; gaps simply never draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubmeshRange {
    /// First index in the mesh's index buffer.
    pub first_index: u32,
    /// Number of indices in the range (a multiple of 3 for triangle lists).
    pub index_count: u32,
}

/// Raw mesh data for upload to the GPU. Framework-agnostic representation.
#[derive(Clone)]
#[non_exhaustive]
pub struct MeshData {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex normals (must be the same length as `positions`).
    pub normals: Vec<[f32; 3]>,
    /// Triangle index list (every 3 indices form one triangle).
    pub indices: Vec<u32>,
    /// Optional per-vertex UV coordinates. `None` means zero-fill [0.0, 0.0].
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Optional per-vertex tangents [tx, ty, tz, w] where w is handedness (+/-1.0).
    ///
    /// `None` = auto-compute from UVs if available, or zero-fill otherwise.
    /// Tangents are required for correct normal map rendering.
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Optional per-vertex RGBA colour in linear 0..1, multiplied into the base
    /// colour before lighting (same convention as glTF `COLOR_0`). `None` leaves
    /// vertices white [1, 1, 1, 1], a neutral multiply. Entries beyond the slice
    /// length also default to white, matching the forgiving `uvs`/`tangents` lookup.
    pub vertex_colours: Option<Vec<[f32; 4]>>,
    /// Named scalar attributes for per-vertex or per-cell scalar field visualisation.
    ///
    /// Keys are user-defined attribute names (e.g. `"pressure"`, `"velocity_mag"`).
    /// Cell attributes are averaged to vertices at upload time.
    pub attributes: std::collections::HashMap<String, AttributeData>,
    /// Optional per-vertex `vec4<f32>` channel for material plugins.
    ///
    /// Uploaded to a per-mesh storage buffer and delivered to shading-hook
    /// bodies as `surf.attr` when the plugin sets `reads_vertex_attribute`
    /// (interpolated across the triangle, like any vertex attribute). The
    /// meaning of the four components is up to the plugin: blend masks,
    /// wind weights, bake data, etc.
    ///
    /// `None`, entries beyond the slice length, and meshes drawn without a
    /// reading plugin all resolve to `vec4(0.0)`.
    pub extension_attributes: Option<Vec<[f32; 4]>>,
    /// Optional material ranges partitioning the index buffer.
    ///
    /// Empty means the whole mesh draws with the item's single material,
    /// which is the behaviour for meshes that do not need ranges. When
    /// non-empty, an item can bind one material per range via
    /// `SceneRenderItem::submesh_materials`. See [`SubmeshRange`] for the
    /// index-sorting contract.
    pub submeshes: Vec<SubmeshRange>,
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            uvs: None,
            tangents: None,
            vertex_colours: None,
            attributes: std::collections::HashMap::new(),
            extension_attributes: None,
            submeshes: Vec::new(),
        }
    }
}

impl MeshData {
    /// Compute the local-space AABB from vertex positions.
    pub fn compute_aabb(&self) -> crate::scene::aabb::Aabb {
        crate::scene::aabb::Aabb::from_positions(&self.positions)
    }

    /// Sort triangles into contiguous per-material runs and fill `submeshes`.
    ///
    /// `triangle_materials[t]` is the material id of triangle `t` (the
    /// triangle at `indices[3t..3t+3]`). Triangles are stable-sorted by
    /// material id, the index buffer is rewritten in that order, and
    /// per-triangle attributes (`Cell`, `Face`, `FaceColour`, `Edge`,
    /// `Halfedge`, `Corner`) are permuted alongside so they keep addressing
    /// the same triangles. Per-vertex data is untouched.
    ///
    /// One range per distinct material id is written to `submeshes`, ordered
    /// by ascending id. Returns the distinct ids in that same order, so a
    /// caller with sparse or unordered ids can line up
    /// `SceneRenderItem::submesh_materials[i]` with the returned `ids[i]`.
    ///
    /// Importers that already emit per-material contiguous indices do not
    /// need this; it exists for meshes whose triangles arrive interleaved.
    ///
    /// # Errors
    ///
    /// [`ViewportError::SubmeshTriangleCountMismatch`](crate::error::ViewportError::SubmeshTriangleCountMismatch)
    /// if `triangle_materials.len() != indices.len() / 3` (or the index
    /// count is not a multiple of 3).
    pub fn sort_triangles_into_submeshes(
        &mut self,
        triangle_materials: &[u32],
    ) -> crate::error::ViewportResult<Vec<u32>> {
        let tri_count = self.indices.len() / 3;
        if self.indices.len() % 3 != 0 || triangle_materials.len() != tri_count {
            return Err(crate::error::ViewportError::SubmeshTriangleCountMismatch {
                triangles: tri_count,
                material_ids: triangle_materials.len(),
            });
        }

        let mut order: Vec<u32> = (0..tri_count as u32).collect();
        order.sort_by_key(|&t| triangle_materials[t as usize]);

        let old_indices = std::mem::take(&mut self.indices);
        self.indices = Vec::with_capacity(old_indices.len());
        for &t in &order {
            let base = t as usize * 3;
            self.indices.extend_from_slice(&old_indices[base..base + 3]);
        }

        // Per-triangle attribute data must follow its triangle to the new
        // position. Only permute channels whose length matches; a wrong
        // length is caught by upload validation, not silently reshuffled.
        fn permute<T: Copy>(values: &mut Vec<T>, order: &[u32], per_tri: usize) {
            if values.len() != order.len() * per_tri {
                return;
            }
            let old = std::mem::take(values);
            values.reserve(old.len());
            for &t in order {
                let base = t as usize * per_tri;
                values.extend_from_slice(&old[base..base + per_tri]);
            }
        }
        for data in self.attributes.values_mut() {
            match data {
                AttributeData::Cell(v) | AttributeData::Face(v) => permute(v, &order, 1),
                AttributeData::FaceColour(v) => permute(v, &order, 1),
                AttributeData::Edge(v) | AttributeData::Halfedge(v) | AttributeData::Corner(v) => {
                    permute(v, &order, 3)
                }
                AttributeData::Vertex(_) | AttributeData::VertexVector(_) => {}
            }
        }

        self.submeshes.clear();
        let mut ids = Vec::new();
        let mut run_start = 0usize;
        for i in 0..order.len() {
            let id = triangle_materials[order[i] as usize];
            let next_differs = order
                .get(i + 1)
                .map_or(true, |&t| triangle_materials[t as usize] != id);
            if next_differs {
                self.submeshes.push(SubmeshRange {
                    first_index: run_start as u32 * 3,
                    index_count: (i + 1 - run_start) as u32 * 3,
                });
                ids.push(id);
                run_start = i + 1;
            }
        }
        Ok(ids)
    }
}

impl crate::resources::DeviceResources {
    /// Write new scalar data into an existing attribute buffer in-place.
    ///
    /// No GPU buffer reallocation, no mesh re-upload, no bind group rebuild is
    /// required. The attribute bind group *will* be rebuilt on the next
    /// `prepare()` call if the scalar range changes (tracked via `last_tex_key`).
    ///
    /// # Errors
    ///
    /// - [`ViewportError::SlotEmpty`](crate::error::ViewportError::SlotEmpty) : `mesh_id` not found in the store.
    /// - [`ViewportError::AttributeNotFound`](crate::error::ViewportError::AttributeNotFound) : `name` not present on the mesh.
    /// - [`ViewportError::AttributeLengthMismatch`](crate::error::ViewportError::AttributeLengthMismatch) : `data.len()` differs from
    ///   the original upload (same-topology requirement).
    pub fn replace_attribute(
        &mut self,
        queue: &crate::gpu::Queue,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        name: &str,
        data: &[f32],
    ) -> crate::error::ViewportResult<()> {
        // Resolve the mesh.
        let gpu_mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::SlotEmpty {
                    index: mesh_id.index(),
                })?;

        // Find the existing attribute buffer.
        let buffer = gpu_mesh.attribute_buffers.get(name).ok_or_else(|| {
            crate::error::ViewportError::AttributeNotFound {
                mesh_id: mesh_id.index(),
                name: name.to_string(),
            }
        })?;

        // Validate same topology (buffer size must match).
        let expected_elems = (buffer.size() / 4) as usize;
        if data.len() != expected_elems {
            return Err(crate::error::ViewportError::AttributeLengthMismatch {
                expected: expected_elems,
                got: data.len(),
            });
        }

        // Zero-copy in-place write via the wgpu staging belt.
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));

        // Recompute scalar range so LUT mapping stays accurate.
        let (min, max) = data
            .iter()
            .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        let range = if min > max { (0.0, 1.0) } else { (min, max) };
        gpu_mesh.attribute_ranges.insert(name.to_string(), range);

        // Force bind group rebuild on next prepare() by invalidating the key.
        gpu_mesh.last_tex_key = (
            gpu_mesh.last_tex_key.0,
            gpu_mesh.last_tex_key.1,
            gpu_mesh.last_tex_key.2,
            gpu_mesh.last_tex_key.3,
            u64::MAX, // attribute hash component
            gpu_mesh.last_tex_key.5,
            gpu_mesh.last_tex_key.6,
            gpu_mesh.last_tex_key.7,
            gpu_mesh.last_tex_key.8,
            gpu_mesh.last_tex_key.9,
            gpu_mesh.last_tex_key.10,
            gpu_mesh.last_tex_key.11,
        );

        Ok(())
    }
}

/// Linearly interpolate between two attribute buffers element-wise.
///
/// Both slices must have the same length. `t` is clamped to `[0.0, 1.0]`.
/// Returns a new `Vec<f32>` with `a[i] * (1 - t) + b[i] * t`.
///
/// Use this to blend per-vertex scalar attributes between two consecutive
/// timesteps when scrubbing the timeline at sub-frame resolution.
pub fn lerp_attributes(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let t = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| av * one_minus_t + bv * t)
        .collect()
}
