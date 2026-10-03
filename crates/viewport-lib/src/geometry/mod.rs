/// BVH-accelerated ray picking.
pub mod bvh;

// The pure CPU geometry below lives in the `viewport-lib-geometry` crate;
// re-exported here so the renderer and widgets keep their `crate::geometry::*`
// paths.
pub use viewport_lib_geometry::{
    cap_geometry, eigen, intersect, intrinsic_vectors, isoline, marching_cubes, one_forms,
    polyline, primitives, tangent_frames, vector_samples,
};
