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
    /// Per-object uniform buffer (model matrix, material, selection state).
    pub object_uniform_buf: crate::gpu::Buffer,
    /// Bind group (group 1) combining `object_uniform_buf` with texture views.
    /// Texture views are the fallback 1x1 textures by default; rebuilt when material
    /// texture assignment changes (tracked via `last_tex_key`).
    pub object_bind_group: crate::gpu::BindGroup,
    /// Last texture/attribute key used to build `object_bind_group`. `u64::MAX` = fallback / none.
    /// Fields: `(albedo, normal_map, ao_map, lut, attr_hash, matcap, warp_hash, metallic_roughness, emissive, position_override_gen, normal_override_gen, extension_attr)`.
    /// The override-gen slots are bumped by `set_position_override_buffer`,
    /// `set_normal_override_buffer`, and their `clear_*` counterparts, so the
    /// next `update_mesh_texture_bind_group` call rebuilds with the real
    /// override buffer (or the fallback) at bindings 13/14. The final slot
    /// packs the extension-attribute presence bit with `lightmap_gen` (bumped by
    /// `set_lightmap` / `clear_lightmap`), so a lightmap change forces the
    /// rebuild that swaps the UV1 (binding 16) and lightmap texture (binding 17)
    /// for the registered ones (or the fallbacks). Kept at 12 elements because
    /// `PartialEq` is only derived for tuples up to that arity.
    pub(crate) last_tex_key: (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
    /// Per-named-attribute GPU storage buffers (f32 per vertex, STORAGE usage).
    pub attribute_buffers: std::collections::HashMap<String, crate::gpu::Buffer>,
    /// Scalar range `(min, max)` per attribute, computed at upload time.
    pub attribute_ranges: std::collections::HashMap<String, (f32, f32)>,
    /// Non-indexed vertex buffer containing 3xN expanded vertices for face-attribute rendering.
    /// `None` if no `Face` or `FaceColour` attributes exist for this mesh.
    pub face_vertex_buffer: Option<crate::gpu::Buffer>,
    /// Named face scalar buffers: 3N `f32` entries (value replicated for all 3 vertices of each tri).
    pub face_attribute_buffers: std::collections::HashMap<String, crate::gpu::Buffer>,
    /// Named face colour buffers: 3N `[f32; 4]` entries (colour replicated for all 3 vertices of each tri).
    pub face_colour_buffers: std::collections::HashMap<String, crate::gpu::Buffer>,
    /// Per-vertex vector attribute buffers: flat `array<f32>` with 3 values per vertex.
    /// Uploaded from `AttributeData::VertexVector`; used by the Surface LIC surface pass.
    pub vector_attribute_buffers: std::collections::HashMap<String, crate::gpu::Buffer>,
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
    /// Optional per-vertex `vec4<f32>` extension-attribute buffer, uploaded
    /// from `MeshData::extension_attributes` and bound at group 1 binding 15
    /// (the 16-byte zero fallback when `None`). Read by material-plugin
    /// modules whose hook sets `reads_vertex_attribute`; base shaders never
    /// touch it.
    pub extension_attr_buffer: Option<crate::gpu::Buffer>,
    /// Baked lightmap registration: the per-vertex UV1 storage buffer (group 1
    /// binding 16), the lightmap texture (binding 17), and the blend mode. Set
    /// via `DeviceResources::set_lightmap`; `None` for meshes without one, which
    /// bind the shared fallbacks and skip the shader branch.
    pub(crate) lightmap: Option<crate::resources::lightmap::MeshLightmap>,
    /// Monotonic counter bumped by `set_lightmap` / `clear_lightmap`. Folded
    /// into `last_tex_key` so the object bind group rebuilds and rebinds the
    /// real UV1 buffer and lightmap texture (or the fallbacks) on the next
    /// `prepare()`.
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
    /// Sums the geometry, attribute, override, and per-object uniform buffers.
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
            self.extension_attr_buffer.as_ref(),
            self.lightmap.as_ref().map(|lm| &lm.uv1_buffer),
        ]
        .into_iter()
        .flatten()
        {
            bytes += buf.size();
        }
        for map in [
            &self.attribute_buffers,
            &self.face_attribute_buffers,
            &self.face_colour_buffers,
            &self.vector_attribute_buffers,
        ] {
            bytes += map.values().map(|b| b.size()).sum::<u64>();
        }
        bytes
    }
}
