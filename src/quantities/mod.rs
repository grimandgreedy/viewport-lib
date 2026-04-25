//! On-surface vector quantity utilities.
//!
//! Converts intrinsic (tangent-plane) vector fields and edge one-forms into
//! world-space [`GlyphItem`](crate::GlyphItem)s ready for submission to
//! [`SceneFrame::glyphs`](crate::renderer::types::SceneFrame::glyphs).
//!
//! # Quick-start
//!
//! ```rust,ignore
//! use viewport_lib::quantities::{vertex_intrinsic_to_glyphs, edge_one_form_to_glyphs};
//!
//! // Vertex intrinsic vectors — e.g. a tangential vortex field on a sphere.
//! let glyph = vertex_intrinsic_to_glyphs(
//!     &mesh.positions,
//!     &mesh.normals,
//!     mesh.tangents.as_deref(),
//!     &intrinsic_vecs,   // &[[f32; 2]] — (u, v) per vertex
//!     0.3,               // arrow scale
//! );
//! frame.scene.glyphs.push(glyph);
//!
//! // Edge one-forms — Whitney reconstruction per triangle.
//! let glyph = edge_one_form_to_glyphs(
//!     &mesh.positions,
//!     &mesh.indices,
//!     &edge_values,      // &[f32] — 3 values per triangle (e01, e12, e20)
//!     0.3,
//! );
//! frame.scene.glyphs.push(glyph);
//! ```
//!
//! # Tangent frames
//!
//! All functions accept an optional explicit tangent buffer (`[tx, ty, tz, w]`
//! format, matching [`MeshData::tangents`](crate::resources::MeshData::tangents)).
//! When omitted, smooth per-vertex frames are derived from the normals via
//! Gram-Schmidt orthogonalisation.

pub mod intrinsic_vectors;
pub mod one_forms;
pub mod tangent_frames;

pub use intrinsic_vectors::{face_intrinsic_to_glyphs, vertex_intrinsic_to_glyphs};
pub use one_forms::edge_one_form_to_glyphs;
pub use tangent_frames::{
    compute_face_tangent_frames, compute_vertex_tangent_frames, tangents_from_explicit,
};
