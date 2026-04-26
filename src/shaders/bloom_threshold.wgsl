// bloom_threshold.wgsl : extract bright pixels above a luminance threshold.

struct BloomUniform {
    threshold:  f32,
    intensity:  f32,
    horizontal: u32,  // unused in threshold pass; shared struct with blur
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_texture, input_sampler, in.uv).rgb;
    // Relative luminance (Rec.709).
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Soft knee: how much the pixel exceeds the threshold.
    let brightness = max(lum - params.threshold, 0.0);
    // Re-colour by the original hue but scaled to the above-threshold portion.
    let out_color = color * (brightness / max(lum, 0.001)) * params.intensity;
    return vec4<f32>(out_color, 1.0);
}
