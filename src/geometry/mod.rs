/// BVH-accelerated ray picking.
pub mod bvh;
/// CPU-side marching cubes isosurface extraction from volumetric data.
pub mod marching_cubes;
/// CPU-side edge-walk isoline (contour line) extraction from triangulated surfaces.
pub mod isoline;
pub(crate) mod cap_geometry;
