/// BVH-accelerated ray picking.
pub mod bvh;
/// CPU sphere-marching of implicit surfaces (signed-distance functions).
pub mod implicit;
/// CPU-side edge-walk isoline (contour line) extraction from triangulated surfaces.
pub mod isoline;

// The pure CPU geometry below lives in the `viewport-lib-geometry` crate;
// re-exported here so the renderer and widgets keep their `crate::geometry::*`
// paths.
pub use viewport_lib_geometry::{cap_geometry, intersect, marching_cubes, polyline, primitives};
