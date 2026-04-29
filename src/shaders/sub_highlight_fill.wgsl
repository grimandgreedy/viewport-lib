// Sub-object highlight : translucent face fill.
//
// Draws selected triangles from world-space positions with a constant fill
// color. Polygon-offset depth bias is configured on the pipeline so the fill
// sits just above the mesh surface without z-fighting.
//
// Group 0: Camera uniform (view_proj + eye_pos, matching camera_bgl binding 0).
// Group 1: SubHighlightUniform (fill_color, edge_color, edge_width, vertex_size,
//          viewport_width, viewport_height).
//
// Vertex input: xyz float (12 bytes, non-indexed triangleList, 3 verts per triangle).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
}

struct SubHighlight {
    fill_color:      vec4<f32>,
    edge_color:      vec4<f32>,
    edge_width:      f32,
    vertex_size:     f32,
    viewport_width:  f32,
    viewport_height: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> hl:     SubHighlight;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {
    return VsOut(camera.view_proj * vec4<f32>(pos, 1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return hl.fill_color;
}
