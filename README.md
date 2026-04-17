# viewport-lib

`viewport-lib` is a gpu-accelerated 3D viewport library for rust. The library gives you a renderer, camera, picking-tools, gizmos, overlays, and scene primitives.

<table>
  <tr>
    <td><img src="assets/demo1.png" alt="demo 1" /></td>
    <td><img src="assets/demo2.png" alt="demo 2" /></td>
  </tr>
  <tr>
    <td><img src="assets/demo3.png" alt="demo 3" /></td>
    <td><img src="assets/demo4.png" alt="demo 4" /></td>
  </tr>
</table>

Whichever gui library you choose to use (`winit`, `eframe`, `Iced`, `Slint`, etc.), the integration model stays the same in each case:
- your application owns the window, event loop, and tool state;
- `viewport-lib` owns rendering and viewport-side maths.


**WARNING**: `viewport-lib` has only recently been extracted as a stand-alone library from a separate project and the API is still somewhat unstable.

## Core features
- mesh, point cloud, polyline, and volume rendering
- directional lighting, shadow mapping, and post-processing
- material shading, normal maps, transparency, outlines, and x-ray views
- clip planes, section views, scalar coloring, and colormaps
- arcball camera, view presets, framing, and smooth camera animation
- CPU/GPU picking, rectangle selection, transform gizmos, and snapping
- annotations, axes indicators
