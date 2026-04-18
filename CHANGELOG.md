# Changelog


## [1.3.0]

### Changed

- Updated examples and documentation to present the grouped frame API as the canonical integration style.
- Continued the API-refactor documentation sweep, including remaining-work tracking and a dedicated camera API refactor plan.
- Clarified public guidance around `RenderCamera`, `CameraFrame`, `SceneFrame`, and `FrameData` construction.

### Fixed

- Corrected winding inconsistencies in generated primitive meshes so surfaces render correctly with the library's CCW front-face and back-face culling settings.
- Fixed affected primitives including cylinder, torus, cone, capsule, ellipsoid, spring, arrow, plane, disk, ring, and grid plane.
- Added a regression test covering primitive triangle winding against emitted normals to prevent back-face culling regressions.

## [0.3.0]

### Changed

- Moved `wireframe_mode` into `ViewportFrame` so viewport display state lives entirely under the grouped viewport section.

## [0.2.0]

### Changed

- Refactored the frame submission API around grouped frame sections.
- Introduced `RenderCamera` as the canonical renderer-facing camera type.
- Moved scene, viewport, interaction, effects, and cache-hint state into dedicated frame sub-objects.
- Updated examples to use the grouped frame API.

## [0.1.0]

### Added

- Initial crates.io release of `viewport-lib`.
