// One direction of the separable blur over a soft 2D light's shadow mask: a
// Gaussian over `radius` device pixels, taken in `taps` samples each side
// through the linear sampler.
//
// Group 0: the blur's block, at a dynamic offset per pass.
// Group 1: the mask being blurred and its sampler.

struct Blur {
    // One texel along the blur's axis, in texture coordinates.
    step:   vec2<f32>,
    // Radius in texels, and the samples taken each side.
    radius: f32,
    taps:   f32,
};

@group(0) @binding(0) var<uniform> blur: Blur;
@group(1) @binding(0) var source: texture_2d<f32>;
@group(1) @binding(1) var source_sampler: sampler;

struct BlurOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
};

// One triangle over the whole target.
@vertex
fn vs_blur(@builtin(vertex_index) vi: u32) -> BlurOut {
    let c = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: BlurOut;
    out.pos = vec4<f32>(c * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(c.x, 1.0 - c.y);
    return out;
}

@fragment
fn fs_blur(in: BlurOut) -> @location(0) vec4<f32> {
    let taps = u32(blur.taps);
    let spacing = blur.radius / max(blur.taps, 1.0);
    let sigma = max(blur.radius * 0.5, 1e-3);
    var sum = textureSampleLevel(source, source_sampler, in.uv, 0.0).r;
    var weights = 1.0;
    for (var i = 1u; i <= taps; i = i + 1u) {
        let x = f32(i) * spacing;
        let w = exp(-0.5 * (x / sigma) * (x / sigma));
        let d = blur.step * x;
        sum = sum + w * (textureSampleLevel(source, source_sampler, in.uv + d, 0.0).r
            + textureSampleLevel(source, source_sampler, in.uv - d, 0.0).r);
        weights = weights + 2.0 * w;
    }
    return vec4<f32>(sum / weights);
}
