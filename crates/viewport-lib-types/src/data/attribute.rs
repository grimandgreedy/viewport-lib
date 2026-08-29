//! Scalar attributes carried on a mesh: what an attribute means per element and
//! how its values are laid out.

/// Scalar attribute interpolation domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeKind {
    /// One value per vertex.
    Vertex,
    /// One value per triangle (cell). Averaged to vertices at upload time.
    Cell,
    /// One value per triangle. NOT averaged : rendered flat via vertex duplication.
    /// Colourmapped through the active LUT just like `Vertex`.
    Face,
    /// One RGBA colour per triangle. NOT averaged : rendered flat via vertex duplication.
    /// Bypasses the colourmap; the per-face colour is used directly.
    FaceColour,
    /// One value per directed triangle edge. `values[3*t + k]` is the scalar on the
    /// k-th edge of triangle `t` (edge from vertex `k` to vertex `(k+1)%3`).
    /// Averaged to the two endpoint vertices for rendering.
    Edge,
    /// One value per directed triangle edge (halfedge). `values[3*t + k]` is the
    /// scalar for the k-th halfedge of triangle `t`.
    /// Rendered flat per triangle corner via vertex duplication (like `Face`).
    Halfedge,
    /// One value per triangle corner. `values[3*t + k]` is the scalar at the
    /// k-th corner of triangle `t`.
    /// Rendered flat per triangle corner via vertex duplication (like `Face`).
    Corner,
}

/// Reference to a named scalar attribute on a mesh.
#[derive(Debug, Clone)]
pub struct AttributeRef {
    /// Name of the attribute as stored in `MeshData::attributes`.
    pub name: String,
    /// Whether the attribute is per-vertex, per-cell, or per-face.
    pub kind: AttributeKind,
}

/// Scalar data for a mesh attribute.
#[derive(Debug, Clone)]
pub enum AttributeData {
    /// One `f32` per vertex.
    Vertex(Vec<f32>),
    /// One `f32` per triangle (cell). Averaged to vertices at upload.
    Cell(Vec<f32>),
    /// One `f32` per triangle. Not averaged; stored in a non-indexed expanded buffer.
    Face(Vec<f32>),
    /// One `[r, g, b, a]` per triangle. Not averaged; stored in a non-indexed expanded buffer.
    FaceColour(Vec<[f32; 4]>),
    /// One `f32` per directed triangle edge. `values[3*t + k]` = k-th edge of triangle `t`.
    /// Averaged to the two endpoint vertices for rendering.
    Edge(Vec<f32>),
    /// One `f32` per directed triangle edge (halfedge). `values[3*t + k]` = k-th halfedge of
    /// triangle `t`. Rendered flat per corner via vertex duplication (like `Face`).
    Halfedge(Vec<f32>),
    /// One `f32` per triangle corner. `values[3*t + k]` = k-th corner of triangle `t`.
    /// Rendered flat per corner via vertex duplication (like `Face`).
    Corner(Vec<f32>),
    /// One `[x, y, z]` per vertex. Uploaded as a flat `array<f32>` storage buffer (3 floats
    /// per vertex) for use in per-vertex vector field rendering (e.g. Surface LIC).
    VertexVector(Vec<[f32; 3]>),
}
