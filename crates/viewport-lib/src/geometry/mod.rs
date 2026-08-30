/// BVH-accelerated ray picking.
pub mod bvh;
pub(crate) mod cap_geometry;
/// CPU sphere-marching of implicit surfaces (signed-distance functions).
pub mod implicit;
/// Ray/primitive intersection helpers shared across interaction widgets.
pub mod intersect;
/// CPU-side edge-walk isoline (contour line) extraction from triangulated surfaces.
pub mod isoline;
/// CPU-side marching cubes isosurface extraction from volumetric data.
///
/// Lives in the `viewport-lib-geometry` crate; re-exported here so the renderer
/// keeps its `crate::geometry::marching_cubes` path.
pub use viewport_lib_geometry::marching_cubes;
/// Polyline construction helpers shared by interaction widget wireframes.
pub mod polyline;
/// Geometry primitives: cube, sphere, plane, cylinder, cone, capsule, torus, torus_ellipse, torus_stadium, icosphere, arrow, disk, frustum, hemisphere, ring, ellipsoid, spring, grid_plane.
pub mod primitives;
