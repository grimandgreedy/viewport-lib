# viewport-lib-geometry

CPU geometry algorithms for [`viewport-lib`](https://github.com/grimandgreedy/viewport-lib), built on [`viewport-lib-types`](https://crates.io/crates/viewport-lib-types).

- `primitives`: built-in meshes (cube, sphere, torus, arrow and the rest), plus `primitives::wire` for wireframe point loops.
- `mesh`: operations on triangle meshes: attribute expansion, tangents, validation, clip-plane caps (`mesh::cap`) and isolines (`mesh::isoline`).
- `volume`: a scalar field on a regular grid (`volume::grid::VolumeData`), its marching cubes isosurface (`volume::marching_cubes::extract_isosurface`), and unstructured volume meshes (`volume::mesh::VolumeMeshData`, tet / pyramid / wedge / hex) with boundary and clipped extraction.
- `maths`: small helpers such as `maths::intersect::ray_plane_intersection`.

Everything returns plain data, usually a `MeshData` the renderer can upload, and nothing here depends on `wgpu`, so loaders and mesh tools can use it without the renderer. `viewport-lib` re-exports the crate as `viewport_lib::vplg`, and the common entry points at its root.
