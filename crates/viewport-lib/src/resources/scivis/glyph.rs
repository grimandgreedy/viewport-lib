//! The arrow, sphere and cube base meshes glyph-shaped item types instance.
//!
//! Two item types draw them (glyph and tensor glyph) and a third builds glyph
//! data from them (the polyline vector decoration), so the cache is shared
//! machinery and stays with the renderer. It is published to item types as
//! `glyph_base_mesh` / `ensure_glyph_base_mesh`. Everything else glyph-shaped,
//! the layouts, the upload and the store, belongs to the glyph item type.
//!
//! The meshes are built on first use behind a `OnceLock`, so an item type can
//! reach them from `prepare`, which holds only `&DeviceResources`.

use super::*;

/// The cached glyph base meshes, one per shape.
pub(crate) struct GlyphResources {
    /// Cached glyph base mesh for the Arrow shape.
    pub(crate) arrow_mesh: std::sync::OnceLock<GlyphBaseMesh>,
    /// Cached glyph base mesh for the Sphere shape.
    pub(crate) sphere_mesh: std::sync::OnceLock<GlyphBaseMesh>,
    /// Cached glyph base mesh for the Cube shape.
    pub(crate) cube_mesh: std::sync::OnceLock<GlyphBaseMesh>,
}

impl GlyphResources {
    pub(crate) fn new() -> Self {
        Self {
            arrow_mesh: std::sync::OnceLock::new(),
            sphere_mesh: std::sync::OnceLock::new(),
            cube_mesh: std::sync::OnceLock::new(),
        }
    }
}

impl DeviceResources {
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
