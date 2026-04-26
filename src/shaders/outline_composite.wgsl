// outline_composite.wgsl : fullscreen blit that composites the offscreen outline
// texture onto the main render pass using alpha blending.
// The outline texture contains the solid outline ring on a transparent background.

@group(0) @binding(0) var outline_tex: texture_2d<f32>;
@group(0) @binding(1) var outline_samp: sampler;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

// Fullscreen triangle (no vertex buffer).
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    let uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(outline_tex, outline_samp, in.uv);
}
