//! Per-mesh GPU buffers and bind group (`GpuMesh`).

use crate::resources::types::*;

// ---------------------------------------------------------------------------
// GpuMesh: per-object GPU buffers
// ---------------------------------------------------------------------------

/// GPU buffers and bind group for a single mesh.
pub struct GpuMesh {
    /// Interleaved position + normal vertex buffer (Vertex layout).
    pub vertex_buffer: crate::gpu::Buffer,
    /// Triangle index buffer (for solid rendering).
    pub index_buffer: crate::gpu::Buffer,
    /// Number of indices in the triangle index buffer.
    pub index_count: u32,
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
    /// Fields: `(albedo, normal_map, ao_map, lut, attr_hash, matcap, warp_hash, metallic_roughness, emissive, position_override_gen, normal_override_gen)`.
    /// The trailing two slots are bumped by `set_position_override_buffer`,
    /// `set_normal_override_buffer`, and their `clear_*` counterparts, so the
    /// next `update_mesh_texture_bind_group` call rebuilds with the real
    /// override buffer (or the fallback) at bindings 13/14.
    pub(crate) last_tex_key: (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
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
    /// Optional per-vertex normal override buffer. Same contract as
    /// `position_override_buffer` but bound at group 1 binding 14.
    pub normal_override_buffer: Option<crate::gpu::Buffer>,
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
}

impl GpuMesh {
    /// Number of vertices in the mesh, derived from the vertex buffer size.
    ///
    /// Useful for sizing or validating a position/normal override buffer
    /// against the mesh it will be bound to.
    pub fn vertex_count(&self) -> usize {
        (self.vertex_buffer.size() / std::mem::size_of::<Vertex>() as u64) as usize
    }

    /// Total GPU buffer bytes held by this mesh.
    ///
    /// Sums the geometry, attribute, override, and per-object uniform buffers.
    /// Used for the resident-bytes accounting so freeing the mesh decrements the
    /// running total by the same amount it added at upload.
    pub(crate) fn gpu_byte_size(&self) -> u64 {
        let mut bytes = self.vertex_buffer.size()
            + self.index_buffer.size()
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
