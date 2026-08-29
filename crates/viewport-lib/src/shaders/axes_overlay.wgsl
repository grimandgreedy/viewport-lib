// Screen-space axes indicator overlay.
//
// No camera transform : vertex positions are already in clip space (NDC).
// Per-vertex colour with alpha blending for circle backgrounds.

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) colour:    vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) colour: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = vec4<f32>(in.position, 0.0, 1.0);
    out.colour = in.colour;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.colour;
}
