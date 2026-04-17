// bloom_blur.wgsl — separable 9-tap Gaussian blur.
// params.horizontal=1 → horizontal pass; params.horizontal=0 → vertical pass.
// The same shader + pipeline is used for both axes by swapping bind groups.

struct BloomUniform {
    threshold:  f32,  // unused in blur pass
    intensity:  f32,  // unused in blur pass
    horizontal: u32,  // 1 = horizontal, 0 = vertical
    _pad:       u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BloomUniform;

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

// 9-tap Gaussian weights (σ ≈ 1.5, normalised to sum 1).
const WEIGHTS: array<f32, 9> = array<f32, 9>(
    0.0162, 0.0541, 0.1216, 0.1945, 0.2272, 0.1945, 0.1216, 0.0541, 0.0162,
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(input_texture));
    let texel = 1.0 / dim;

    var color = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 9; i = i + 1) {
        let offset = f32(i - 4);
        var uv_off: vec2<f32>;
        if params.horizontal != 0u {
            uv_off = vec2<f32>(texel.x * offset, 0.0);
        } else {
            uv_off = vec2<f32>(0.0, texel.y * offset);
        }
        color = color + textureSample(input_texture, input_sampler, in.uv + uv_off).rgb * WEIGHTS[i];
    }
    return vec4<f32>(color, 1.0);
}
