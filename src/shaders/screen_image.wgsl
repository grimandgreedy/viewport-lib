// Screen-space image overlay shader (Phase 10B).
//
// Renders a textured quad anchored to a viewport corner or center.
// No vertex buffer — corners are generated from vertex_index (0-5).
// No depth test; drawn after all 3D content as a HUD overlay.

struct ScreenImageUniform {
    // Bottom-left corner in NDC space.
    ndc_min: vec2<f32>,
    // Top-right corner in NDC space.
    ndc_max: vec2<f32>,
    // Overall opacity multiplier (applied on top of per-pixel alpha).
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: ScreenImageUniform;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var img_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

// Triangle 0: verts 0-2, Triangle 1: verts 3-5
// Quad corners by vertex index:
//   0 = bottom-left, 1 = bottom-right, 2 = top-left
//   3 = top-left,    4 = bottom-right, 5 = top-right
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // NDC position
    var ndc_x: f32;
    var ndc_y: f32;
    // UV (texture origin = top-left, so y is flipped relative to NDC)
    var uv_x: f32;
    var uv_y: f32;

    switch vi {
        case 0u: { ndc_x = u.ndc_min.x; ndc_y = u.ndc_min.y; uv_x = 0.0; uv_y = 1.0; }
        case 1u: { ndc_x = u.ndc_max.x; ndc_y = u.ndc_min.y; uv_x = 1.0; uv_y = 1.0; }
        case 2u: { ndc_x = u.ndc_min.x; ndc_y = u.ndc_max.y; uv_x = 0.0; uv_y = 0.0; }
        case 3u: { ndc_x = u.ndc_min.x; ndc_y = u.ndc_max.y; uv_x = 0.0; uv_y = 0.0; }
        case 4u: { ndc_x = u.ndc_max.x; ndc_y = u.ndc_min.y; uv_x = 1.0; uv_y = 1.0; }
        default: { ndc_x = u.ndc_max.x; ndc_y = u.ndc_max.y; uv_x = 1.0; uv_y = 0.0; }
    }

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv       = vec2<f32>(uv_x, uv_y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(img, img_sampler, in.uv);
    color.a *= u.alpha;
    return color;
}
