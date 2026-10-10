//! A sampled vector field: one world-space vector at each of a set of positions.
//!
//! The helpers here turn on-surface data into samples: intrinsic `(u, v)`
//! vectors in a tangent frame, and Whitney reconstruction of an edge one-form.
//! Hand the result to [`VectorFieldItem::with_samples`](super::VectorFieldItem::with_samples).

pub mod intrinsic;
pub mod one_forms;
pub mod tangent_frames;

pub use intrinsic::{face_intrinsic_vectors, vertex_intrinsic_vectors};
pub use one_forms::edge_one_form_vectors;
pub use tangent_frames::{
    compute_face_tangent_frames, compute_vertex_tangent_frames, tangents_from_explicit,
};

/// Positions paired with the world-space vector at each position.
///
/// This is what the on-surface and volume-mesh quantity helpers return. The two
/// vectors always have the same length, and entry `i` of one goes with entry `i`
/// of the other.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct VectorSamples {
    /// Sample positions, in the same space as the geometry they came from.
    pub positions: Vec<[f32; 3]>,
    /// One world-space vector per position.
    pub vectors: Vec<[f32; 3]>,
}

impl VectorSamples {
    /// Pair up positions and vectors, truncating to the shorter of the two.
    pub fn new(mut positions: Vec<[f32; 3]>, mut vectors: Vec<[f32; 3]>) -> Self {
        let n = positions.len().min(vectors.len());
        positions.truncate(n);
        vectors.truncate(n);
        Self { positions, vectors }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// True when there are no samples.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_truncates_to_the_shorter_input() {
        let s = VectorSamples::new(vec![[0.0; 3]; 5], vec![[1.0, 0.0, 0.0]; 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(s.vectors.len(), 2);
    }

    #[test]
    fn empty_by_default() {
        assert!(VectorSamples::default().is_empty());
    }
}
