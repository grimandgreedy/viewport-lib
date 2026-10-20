//! Per-mesh GPU buffers and bind group (`GpuMesh`).

use crate::resources::mesh::geometry_slab::SlabSpan;
use crate::resources::types::*;

// ---------------------------------------------------------------------------
// GpuMesh: per-object GPU buffers
// ---------------------------------------------------------------------------

/// GPU buffers and bind group for a single mesh.
pub struct GpuMesh {
    /// Window into the shared vertex slab holding this mesh's interleaved
    /// vertices (Vertex layout). Resolve to a bindable slice via
    /// `DeviceResources::geometry`.
    pub(crate) vertex_span: SlabSpan,
    /// Window into the shared index slab holding this mesh's triangle indices.
    pub(crate) index_span: SlabSpan,
    /// Number of indices in the triangle index buffer.
    pub index_count: u32,
    /// Whether the triangle mesh is a closed, consistently wound surface:
    /// every edge is shared by exactly two triangles that traverse it in
    /// opposite directions, with vertices welded by position so a UV or
    /// normal seam does not split an edge. Computed at upload. A closed mesh
    /// casts shadows through the cull-front pipeline whatever its
    /// `BackfacePolicy`, and its back faces (the inside, seen through a clip
    /// plane) shade as inside the solid's own shadow. The cull-none caster
    /// path is for surfaces that are genuinely open. The check runs on a
    /// worker thread, so this reads `false` from upload until the next
    /// `prepare` settles it.
    pub closed: bool,
    /// The closed check still running for this mesh, if any.
    pub(crate) closed_check: Option<crate::resources::mesh::meshes::ClosedCheck>,
    /// Material ranges partitioning the index buffer, from
    /// `MeshData::submeshes`. Empty for single-material meshes, which draw
    /// the full `0..index_count` range.
    pub submeshes: Vec<crate::resources::mesh::meshes::SubmeshRange>,
    /// Edge index buffer (deduplicated pairs, for wireframe LineList
    /// rendering). Built on the first frame something renders this mesh as
    /// wireframe, not at upload time.
    pub edge_index_buffer: Option<crate::gpu::Buffer>,
    /// Number of indices in the edge index buffer.
    pub edge_index_count: u32,
    /// Vertex buffer for per-vertex normal visualization lines (LineList topology).
    /// Each normal contributes two vertices: the vertex position and position + normal * 0.1.
    /// None if no normal data is available.
    pub normal_line_buffer: Option<crate::gpu::Buffer>,
    /// Number of vertices in the normal line buffer (2 per normal line).
    pub normal_line_count: u32,
    /// Single-record instance buffer (model matrix, flags, material id)
    /// backing the mesh's fallback bind group.
    pub object_uniform_buf: crate::gpu::Buffer,
    /// Bind group (group 1) combining `object_uniform_buf` with texture views.
    /// Texture views are the fallback 1x1 textures by default; rebuilt when material
    /// texture assignment changes (tracked via `last_tex_key`).
    pub object_bind_group: crate::gpu::BindGroup,
    /// Last texture key used to build `object_bind_group`. `u64::MAX` = fallback / none.
    /// Fields: `(albedo, normal_map, ao_map, lut, unused, matcap, unused, metallic_roughness, emissive, position_override_gen, normal_override_gen, lightmap_gen)`.
    /// The override-gen slots are bumped by `set_position_override_buffer`,
    /// `set_normal_override_buffer`, and their `clear_*` counterparts, so the
    /// next `update_mesh_texture_bind_group` call rebuilds with the real
    /// override buffer (or the fallback) at bindings 13/14. The last slot is
    /// `lightmap_gen` (bumped by `set_lightmap` / `clear_lightmap`), so a
    /// lightmap change swaps the lightmap textures at bindings 17/18. Kept at
    /// 12 elements because `PartialEq` is only derived for tuples up to that
    /// arity.
    pub(crate) last_tex_key: (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
    /// Scalar range `(min, max)` per attribute, computed at upload time.
    pub attribute_ranges: std::collections::HashMap<String, (f32, f32)>,
    /// Non-indexed vertex buffer containing 3xN expanded vertices for face-attribute rendering.
    /// `None` if no `Face` or `FaceColour` attributes exist for this mesh.
    pub face_vertex_buffer: Option<crate::gpu::Buffer>,
    /// The mesh's sidecar record in the shared pool: its scalar and face
    /// attributes, face colours, vector attributes, extension attribute and
    /// lightmap UV1, plus the per-mesh draw words. `None` for a mesh with none
    /// of these, which binds the pool's empty header.
    pub(crate) sidecar: Option<crate::resources::mesh::sidecar_pool::SidecarRecord>,
    /// Optional per-vertex position override buffer.
    ///
    /// When `Some`, the standard mesh pipeline reads positions from this buffer
    /// (group 1 binding 13, an `array<vec3<f32>>`) instead of the vertex
    /// buffer's position attribute. Set via
    /// `DeviceResources::set_position_override_buffer` and consumed by a
    /// `GpuPlugin`'s compute output. Mutually exclusive per frame with
    /// `write_mesh_positions_normals`; the two paths race if both are used.
    pub position_override_buffer: Option<crate::gpu::Buffer>,
    /// Element window read from `position_override_buffer` when it holds more
    /// than this mesh (a pooled buffer shared by several meshes). `None`
    /// means the buffer is read from element 0, the pre-slicing behaviour.
    /// Set via `DeviceResources::set_position_override_buffer_sliced`.
    pub(crate) position_override_slice: Option<super::OverrideBufferSlice>,
    /// Optional per-vertex normal override buffer. Same contract as
    /// `position_override_buffer` but bound at group 1 binding 14.
    pub normal_override_buffer: Option<crate::gpu::Buffer>,
    /// Same idea as `position_override_slice` for the normal override.
    pub(crate) normal_override_slice: Option<super::OverrideBufferSlice>,
    /// Baked lightmap registration: the lightmap textures (bindings 17 and 18),
    /// the blend mode and the placement. The per-vertex UV1 is a stream of the
    /// sidecar record. Set via `DeviceResources::set_lightmap`; `None` for
    /// meshes without one, which bind the shared fallbacks and skip the shader
    /// branch.
    pub(crate) lightmap: Option<crate::resources::lightmap::MeshLightmap>,
    /// Whether this mesh carries a second UV set (`MeshData::uvs1`). When true its
    /// object bind group binds the parallel uv1 chunk buffer at binding 19 and its
    /// sidecar record carries the mesh `base_vertex`; when false it binds the
    /// zero fallback and any `uv_set = 1` texture sample reads `vec2(0.0)`.
    pub(crate) has_uv1: bool,
    /// Monotonic counter bumped by `set_lightmap` / `clear_lightmap`. Folded
    /// into `last_tex_key` so the object bind group rebuilds and rebinds the
    /// lightmap textures (or the fallbacks) on the next `prepare()`.
    pub(crate) lightmap_gen: u64,
    /// Monotonic counter bumped each time `set_position_override_buffer` or
    /// `clear_position_override` is called. Folded into `last_tex_key` so the
    /// object bind group rebuilds on the next `prepare()` call.
    pub(crate) position_override_gen: u64,
    /// Same idea for normal override (re)bind tracking.
    pub(crate) normal_override_gen: u64,
    /// Uniform buffer for normal-line rendering: always has selected=0, wireframe=0.
    /// Updated each frame in prepare() with the object's model matrix only.
    pub normal_uniform_buf: crate::gpu::Buffer,
    /// Bind group referencing `normal_uniform_buf` : used when drawing normal lines.
    pub normal_bind_group: crate::gpu::BindGroup,
    /// Local-space axis-aligned bounding box computed from vertex positions at upload time.
    pub aabb: crate::scene::aabb::Aabb,
    /// CPU-side positions retained for cap geometry generation (clip plane cross-section fill).
    pub(crate) cpu_positions: Option<Vec<[f32; 3]>>,
    /// CPU-side normals retained so the normal-line visualisation buffer can
    /// be built on first use instead of at upload time.
    pub(crate) cpu_normals: Option<Vec<[f32; 3]>>,
    /// CPU-side triangle indices retained for cap geometry generation.
    pub(crate) cpu_indices: Option<Vec<u32>>,
    /// Monotonic counter bumped whenever the mesh's rendered geometry
    /// changes under the same `MeshId` (`replace_mesh_data`,
    /// `write_mesh_positions_normals`, position-override changes). Caches
    /// keyed on mesh content (the point-shadow cubemap cache) fold this in
    /// so an in-place vertex update invalidates them.
    pub(crate) content_rev: u64,
    /// Cached mesh-local-space `parry3d::TriMesh` (with its QBVH) used by the
    /// CPU picker, plus the `content_rev` it was built from. `Mutex` (not
    /// `RefCell`) because `ViewportRenderer` (and so `GpuMesh`) must stay
    /// `Send + Sync` for the egui callback trait; `ViewportRenderer::pick`
    /// reads through a shared reference. The cache is rebuilt on first use
    /// after upload or whenever `content_rev` moves past the cached value,
    /// and is shared across every instance of this mesh since the geometry is
    /// local-space (see `cached_pick_trimesh`).
    pub(crate) pick_trimesh_cache:
        std::sync::Mutex<Option<(u64, std::sync::Arc<parry3d::shape::TriMesh>)>>,
}

impl GpuMesh {
    /// Number of vertices in the mesh, derived from the vertex buffer size.
    ///
    /// Useful for sizing or validating a position/normal override buffer
    /// against the mesh it will be bound to.
    pub fn vertex_count(&self) -> usize {
        (self.vertex_span.len / std::mem::size_of::<Vertex>() as u64) as usize
    }

    /// Mesh-local-space `parry3d::TriMesh` for CPU picking, built once from
    /// `cpu_positions` / `cpu_indices` and cached until `content_rev` moves.
    ///
    /// Returns `None` when the mesh carries no CPU-side geometry (GPU-only
    /// meshes) or the triangle list is empty. The returned shape carries no
    /// per-instance transform: callers cast the ray in mesh-local space (via
    /// the item's inverse model matrix) so one cached `TriMesh` serves every
    /// instance of this mesh, and an object moving does not invalidate it.
    pub(crate) fn cached_pick_trimesh(&self) -> Option<std::sync::Arc<parry3d::shape::TriMesh>> {
        let mut cache = self.pick_trimesh_cache.lock().unwrap();
        if let Some((rev, trimesh)) = cache.as_ref() {
            if *rev == self.content_rev {
                return Some(trimesh.clone());
            }
        }

        let positions = self.cpu_positions.as_ref()?;
        let indices = self.cpu_indices.as_ref()?;

        let verts: Vec<parry3d::math::Vector> = positions
            .iter()
            .map(|p| parry3d::math::Vector::new(p[0], p[1], p[2]))
            .collect();
        let tri_indices: Vec<[u32; 3]> = indices
            .chunks(3)
            .filter(|c| c.len() == 3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        if tri_indices.is_empty() {
            return None;
        }

        let trimesh = match parry3d::shape::TriMesh::new(verts, tri_indices) {
            Ok(trimesh) => std::sync::Arc::new(trimesh),
            Err(e) => {
                tracing::warn!(error = %e, "TriMesh build failed for CPU picking");
                return None;
            }
        };
        *cache = Some((self.content_rev, trimesh.clone()));
        Some(trimesh)
    }

    /// Total GPU buffer bytes held by this mesh.
    ///
    /// Sums the geometry, sidecar, override, and per-object uniform buffers.
    /// Used for the resident-bytes accounting so freeing the mesh decrements the
    /// running total by the same amount it added at upload.
    pub(crate) fn gpu_byte_size(&self) -> u64 {
        // The geometry lives in the shared slab; charge this mesh's spans, not
        // the whole chunk buffer (the slab reports its chunk bytes separately).
        let mut bytes = self.vertex_span.len
            + self.index_span.len
            + self.object_uniform_buf.size()
            + self.normal_uniform_buf.size();
        for buf in [
            self.edge_index_buffer.as_ref(),
            self.normal_line_buffer.as_ref(),
            self.face_vertex_buffer.as_ref(),
            self.position_override_buffer.as_ref(),
            self.normal_override_buffer.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            bytes += buf.size();
        }
        // The sidecar record lives in a shared pool chunk; charge its span.
        bytes += self.sidecar.as_ref().map_or(0, |r| r.span.len);
        bytes
    }

    /// Host memory bytes held by the retained CPU geometry copies
    /// (`cpu_positions`, `cpu_normals`, `cpu_indices`).
    ///
    /// These are host allocations, not GPU buffers, so they are not part of
    /// [`gpu_byte_size`](Self::gpu_byte_size). On a discrete GPU they draw from
    /// a different pool; on unified memory they draw from the same one, which is
    /// why they are reported separately rather than ignored.
    ///
    /// The `parry3d` triangle mesh cached by
    /// [`cached_pick_trimesh`](Self::cached_pick_trimesh) is built lazily on the
    /// first CPU pick and is not counted here. It is released alongside these
    /// copies by [`release_cpu_geometry`](Self::release_cpu_geometry).
    pub(crate) fn cpu_byte_size(&self) -> u64 {
        let positions = self.cpu_positions.as_ref().map_or(0, |v| v.len() * 12);
        let normals = self.cpu_normals.as_ref().map_or(0, |v| v.len() * 12);
        let indices = self.cpu_indices.as_ref().map_or(0, |v| v.len() * 4);
        (positions + normals + indices) as u64
    }

    /// Drop the retained CPU geometry copies and the cached pick triangle mesh,
    /// returning the host bytes released as counted by
    /// [`cpu_byte_size`](Self::cpu_byte_size).
    ///
    /// The mesh keeps rendering unchanged. What stops working is every feature
    /// that reads back the geometry on the CPU: CPU picking
    /// (`ViewportRenderer::pick`) resolves no hit against this mesh, clip-plane
    /// cap geometry has nothing to build a cross-section fill from, and the
    /// normal-line visualisation cannot build its buffer if it has not already.
    /// GPU picking is unaffected.
    pub(crate) fn release_cpu_geometry(&mut self) -> u64 {
        let released = self.cpu_byte_size();
        self.cpu_positions = None;
        self.cpu_normals = None;
        self.cpu_indices = None;
        if let Ok(mut cache) = self.pick_trimesh_cache.lock() {
            *cache = None;
        }
        released
    }
}
