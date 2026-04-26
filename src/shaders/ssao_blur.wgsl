// ssao_blur.wgsl : simple 4×4 box blur for the SSAO result.
// Smooths the noisy SSAO output without requiring per-sample blurring.

@group(0) @binding(0) var ssao_tex:  texture_2d<f32>;
@group(0) @binding(1) var ssao_samp: sampler;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

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
    let texel = 1.0 / vec2<f32>(textureDimensions(ssao_tex));
    var result: f32 = 0.0;
    // 4×4 box blur centered at [-1.5, +1.5] in each axis.
    for (var x: i32 = -1; x <= 2; x = x + 1) {
        for (var y: i32 = -1; y <= 2; y = y + 1) {
            let off = vec2<f32>(f32(x), f32(y)) * texel;
            result = result + textureSample(ssao_tex, ssao_samp, in.uv + off).r;
        }
    }
    result = result / 16.0;
    return vec4<f32>(result, result, result, 1.0);
}
