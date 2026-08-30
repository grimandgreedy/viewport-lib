//! On-surface vector quantities: convert intrinsic (tangent-plane) vector fields,
//! edge one-forms, and volume-mesh vectors into `GlyphItem`s ready for submission.
//!
//! Pure CPU conversions over plain buffers and `VolumeMeshData`; the produced
//! `GlyphItem` comes from `viewport-lib-types`, so a consumer can build glyph sets
//! without the renderer.

pub mod intrinsic_vectors;
pub mod one_forms;
pub mod volume_mesh_vectors;

pub use intrinsic_vectors::{face_intrinsic_to_glyphs, vertex_intrinsic_to_glyphs};
pub use one_forms::edge_one_form_to_glyphs;
pub use volume_mesh_vectors::{
    volume_mesh_cell_vectors_to_glyphs, volume_mesh_vertex_vectors_to_glyphs,
};
