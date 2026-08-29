# viewport-lib-types

Pure-data types for [`viewport-lib`](https://github.com/grimandgreedy/viewport-lib):
mesh and volume submission payloads, handle ids, scene and camera types, the
frame/effects config tree, overlay item data, and input events.

It carries no GPU dependency (no `wgpu`), so CPU-side tools such as loaders and
mesh processors can produce and consume these types without pulling the renderer.
`viewport-lib` re-exports this crate's surface, so existing `viewport_lib::`
paths keep working.
