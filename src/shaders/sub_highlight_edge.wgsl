// Sub-object highlight : depth-nudged billboard edge lines.
//
// Each draw instance covers one line segment (two world-space endpoints).
// Six vertices are drawn per instance to form a screen-aligned quad
// (two triangles), identical to the polyline billboard approach but without
// miter joints or strip tracking.
//
// A clip-space depth nudge (`clip.z -= 0.0001 / clip.w`) pushes edge geometry
// just in front of the mesh surface to prevent z-fighting without requiring
// hardware polygon offset.
//
// Group 0: Camera uniform (matching camera_bgl binding 0).
// Group 1: SubHighlightUniform (contains edge_color, edge_width, viewport dims).
//
// Instance input (24 bytes per segment):
//   location 0 : pos_a   vec3  (world-space segment start)
//   location 1 : pos_b   vec3  (world-space segment end)

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

// Project a world-space point to screen-pixel coordinates.
fn to_screen(p: vec3<f32>) -> vec2<f32> {
    let c = camera.view_proj * vec4<f32>(p, 1.0);
    let w = max(c.w, 0.0001);
    return (c.xy / w) * vec2<f32>(hl.viewport_width * 0.5, hl.viewport_height * 0.5);
}

// Corner layout (6 verts, TriangleList):
//   Triangle 0: v0=A-left, v1=B-left, v2=A-right
//   Triangle 1: v3=B-left, v4=B-right, v5=A-right
//
//   vid     0 1 2 3 4 5
//   use_b   F T F T T F
//   use_right F F T F T T

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @location(0) pos_a: vec3<f32>,
    @location(1) pos_b: vec3<f32>,
) -> VsOut {
    let use_b     = (vid == 1u || vid == 3u || vid == 4u);
    let use_right = (vid == 2u || vid == 4u || vid == 5u);
    let pos       = select(pos_a, pos_b, use_b);
    let side      = select(-1.0, 1.0, use_right);

    // Project both endpoints to screen pixels.
    let sa = to_screen(pos_a);
    let sb = to_screen(pos_b);

    // Screen-space direction and perpendicular.
    let ab  = sb - sa;
    let len = length(ab);
    let dir = select(vec2<f32>(1.0, 0.0), ab / len, len > 0.001);
    let perp = vec2<f32>(-dir.y, dir.x); // left-perpendicular

    // Clip-space position of the endpoint.
    var clip = camera.view_proj * vec4<f32>(pos, 1.0);

    // Expand by half-width in screen space (convert NDC offset, multiply by w).
    let half_w     = hl.edge_width * 0.5;
    let ndc_offset = side * half_w * perp
                     * vec2<f32>(2.0 / hl.viewport_width, 2.0 / hl.viewport_height);
    clip.x += ndc_offset.x * clip.w;
    clip.y += ndc_offset.y * clip.w;

    // Clip-space depth nudge : shifts edge just in front of the surface.
    clip.z -= 0.0001 / clip.w;

    return VsOut(clip);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return hl.edge_color;
}
