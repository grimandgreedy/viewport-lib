// Sub-object highlight : vertex / point sprite billboards.
//
// Each draw instance is one world-space point. Six vertices are drawn per
// instance to expand it to a screen-aligned quad. Uses a circular disc test
// in the fragment shader to produce a round sprite.
//
// Group 0: Camera uniform (matching camera_bgl binding 0).
// Group 1: SubHighlightUniform (edge_color used for sprite color, vertex_size
//          controls the diameter in pixels, viewport_width/height for expansion).
//
// Instance input (12 bytes per point, step_mode = Instance):
//   location 0 : position  vec3  (world-space point position)

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
    @location(0)       uv:   vec2<f32>,
}

// Unit quad corners (6 vertices, 2 CCW triangles).
fn quad_corner(vi: u32) -> vec2<f32> {
    switch vi {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0,  1.0); }
        case 4u: { return vec2<f32>( 1.0, -1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0)           position: vec3<f32>,
) -> VsOut {
    let center     = camera.view_proj * vec4<f32>(position, 1.0);
    let corner     = quad_corner(vi);
    let half_size  = hl.vertex_size * 0.5;
    let ndc_offset = corner * half_size
                     / vec2<f32>(hl.viewport_width, hl.viewport_height);
    let clip = vec4<f32>(
        center.x + ndc_offset.x * center.w,
        center.y + ndc_offset.y * center.w,
        center.z,
        center.w,
    );
    return VsOut(clip, corner);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Circular disc discard.
    if dot(in.uv, in.uv) > 1.0 { discard; }
    return hl.edge_color;
}
