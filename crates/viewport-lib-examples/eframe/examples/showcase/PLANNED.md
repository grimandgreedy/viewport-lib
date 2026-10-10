# Planned showcases

The core set below is the plan for this example. Showcases 1-4 exist; 5-7 are
still to add. "Folds in" lists the `eframe_showcase` numbers each one is meant to
cover, so a new showcase distils those rather than copying them.

## Core

| # | Showcase | Demonstrates | Folds in (old #s) |
|---|----------|--------------|-------------------|
| 1 | Objects & manipulation (exists) | Scene graph, selection, G/R/S gizmo, orbit/fly camera | 1, 2, 4, 10 |
| 2 | Overlays & annotations (exists) | Labels, scalar bars, rulers, 2D overlay shapes, HUD, depth compositing | 9, 27, 29, 34, 35 |
| 3 | Picking (exists) | Unified object + sub-object picking (face/vert/point/cell) with masks | 33 |
| 4 | Materials & shading (exists) | PBR / matcap / textured / normal+AO / vertex-colour, one stage + picker | 5, 7, 19, 20, 21, 22, 53 |
| 5 | Lighting & shadows | Movable point/spot/directional lights, CSM shadows, hemisphere ambient | 8, 11, 47, 49 |
| 6 | Post-processing | The HDR stack: bloom, SSAO, DoF, tone-map, FXAA as toggles | 6, 55 |
| 7 | Scientific visualization | Point clouds, glyphs, streamlines/tubes, tensor glyphs, gaussian splats + volumes, isosurfaces, clip volumes | 15, 16, 17, 18, 25, 26, 28, 30, 31, 32, 39, 42 |

Note: the built order differs slightly from the numbering above. Files are
`01_objects`, `02_overlays`, `03_materials`, `04_picking`; picking and materials
are swapped relative to this table.

## Optional / advanced

| Showcase | Folds in |
|----------|----------|
| Scalar fields & colourmaps | 12, 14, 38 |
| Performance & scale (instancing, GPU culling, LOD, async streaming) | 23, 40, 50, 51, 52 |
| Particles & decals | 41, 46 |
| Runtime & animation (plugins, physics, skeletal, debug draw) | 36, 43, 44, 45 |
| Custom shading plugins (MaterialPlugin WGSL hooks) | 54 |

## Notes

- Keep each showcase self-contained: it owns its scene, camera tuning, and
  interaction, and touches the shared `ViewportInstance` only through the public
  accessors.
- `eframe_showcase/CATALOGUE.md` is the reference for what each old number
  covered.
