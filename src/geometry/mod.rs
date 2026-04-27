/// BVH-accelerated ray picking.
pub mod bvh;
pub(crate) mod cap_geometry;
/// CPU sphere-marching of implicit surfaces (signed-distance functions).
pub mod implicit;
/// CPU-side edge-walk isoline (contour line) extraction from triangulated surfaces.
pub mod isoline;
/// CPU-side marching cubes isosurface extraction from volumetric data.
pub mod marching_cubes;
/// Geometry primitives: cube, sphere, plane, cylinder, cone, capsule, torus, icosphere, arrow, disk, frustum, hemisphere, ring, ellipsoid, spring, grid_plane.
pub mod primitives;
