// Dynamic resolution upscale pass.
//
// Bilinear-samples the scaled intermediate color texture and writes it to
// the native-resolution surface. The pipeline is compiled for the surface's
// target_format so no manual color-space conversion is required.
//
// Group 0, binding 0 : intermediate color texture (at render_scale resolution)
// Group 0, binding 1 : linear-clamp sampler

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var s_linear: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    // Fullscreen triangle: (-1,-1), (3,-1), (-1,3).
    let x = f32((vi & 1u) * 2u);
    let y = f32((vi >> 1u) * 2u);
    var out: VertexOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv  = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(t_color, s_linear, in.uv);
}
