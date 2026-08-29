// backdrop_blur.wgsl : separable Gaussian blur for the frosted glass backdrop.
//
// Uses a 13-tap kernel on a downsampled scene texture to produce a smooth
// blur suitable for glassmorphism UI panels. The same shader handles both
// horizontal and vertical passes via the `horizontal` uniform.

struct BlurParams {
    horizontal: u32,  // 1 = horizontal pass, 0 = vertical pass
    spread:     f32,  // texel multiplier (controls blur radius)
    _pad0:      u32,
    _pad1:      u32,
}

@group(0) @binding(0) var t_source: texture_2d<f32>;
@group(0) @binding(1) var s_source: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle covering clip space.
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    let uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv);
}

// 13-tap Gaussian weights (sigma ~= 3.0, normalised to sum 1).
const W: array<f32, 7> = array<f32, 7>(
    0.1964, 0.1748, 0.1232, 0.0689, 0.0305, 0.0107, 0.0030,
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(t_source));
    let texel = 1.0 / dim;

    let step = select(
        vec2<f32>(0.0, texel.y),
        vec2<f32>(texel.x, 0.0),
        params.horizontal != 0u,
    ) * params.spread;

    var colour = textureSample(t_source, s_source, in.uv) * W[0];
    for (var i: i32 = 1; i < 7; i = i + 1) {
        let off = step * f32(i);
        colour += textureSample(t_source, s_source, in.uv + off) * W[i];
        colour += textureSample(t_source, s_source, in.uv - off) * W[i];
    }
    // Force alpha to 1.0: the HDR scene clears alpha to 0.0 for OIT
    // compositing, but the blur result must be opaque so the overlay
    // shader can detect it as a backdrop blur source.
    return vec4<f32>(colour.rgb, 1.0);
}
