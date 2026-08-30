# viewport-lib-geometry

CPU geometry algorithms for [`viewport-lib`](https://github.com/grimandgreedy/viewport-lib),
built on [`viewport-lib-types`](https://crates.io/crates/viewport-lib-types).

Two things a consumer needs on its own, without the renderer:

- `marching_cubes`: turn a scalar field (`VolumeData`) into a surface
  (`extract_isosurface`).
- `volume_mesh`: turn unstructured cell connectivity (`VolumeMeshData`, tet /
  pyramid / wedge / hex) into a renderable boundary mesh, or into a `TetMesh`.

Both produce a `MeshData` the renderer can upload, but neither carries a GPU
dependency (no `wgpu`), so loaders and mesh tools can run them standalone. The
renderer-coupled geometry (BVH picking, implicit sphere-marching, the primitive
builders) stays in `viewport-lib`. `viewport-lib` re-exports this crate's
modules, so existing `viewport_lib::` paths keep working.
