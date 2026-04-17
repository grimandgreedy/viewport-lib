// fxaa.wgsl — FXAA (Fast Approximate Anti-Aliasing) fullscreen post-process pass.
//
// Renders a fullscreen triangle, reads from the tone-mapped LDR texture, and
// applies luminance-based edge detection and sub-pixel blending to reduce aliasing.
//
// Based on the NVIDIA FXAA 3.11 algorithm, simplified for clarity.
// Input: LDR Rgba16Float or Rgba8Unorm tone-mapped texture.
// Output: Anti-aliased LDR color.

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

// Generate a fullscreen triangle from three vertices (no vertex buffer needed).
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

// Perceived luminance of an LDR RGB color.
fn luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(input_texture));
    let texel = 1.0 / dims;

    // Sample center and 4 neighbors.
    let c  = textureSample(input_texture, input_sampler, in.uv).rgb;
    let n  = textureSample(input_texture, input_sampler, in.uv + vec2<f32>( 0.0,  texel.y)).rgb;
    let s  = textureSample(input_texture, input_sampler, in.uv + vec2<f32>( 0.0, -texel.y)).rgb;
    let e  = textureSample(input_texture, input_sampler, in.uv + vec2<f32>( texel.x,  0.0)).rgb;
    let w  = textureSample(input_texture, input_sampler, in.uv + vec2<f32>(-texel.x,  0.0)).rgb;

    let lc = luminance(c);
    let ln = luminance(n);
    let ls = luminance(s);
    let le = luminance(e);
    let lw = luminance(w);

    // Local contrast.
    let l_min = min(lc, min(min(ln, ls), min(le, lw)));
    let l_max = max(lc, max(max(ln, ls), max(le, lw)));
    let l_range = l_max - l_min;

    // FXAA threshold: skip low-contrast regions (thin line threshold = 0.0312).
    let FXAA_EDGE_THRESHOLD      = 0.125;
    let FXAA_EDGE_THRESHOLD_MIN  = 0.0312;
    if l_range < max(FXAA_EDGE_THRESHOLD_MIN, l_max * FXAA_EDGE_THRESHOLD) {
        return vec4<f32>(c, 1.0);
    }

    // Diagonal neighbors for 3x3 sub-pixel blend.
    let nw = textureSample(input_texture, input_sampler, in.uv + vec2<f32>(-texel.x,  texel.y)).rgb;
    let ne = textureSample(input_texture, input_sampler, in.uv + vec2<f32>( texel.x,  texel.y)).rgb;
    let sw = textureSample(input_texture, input_sampler, in.uv + vec2<f32>(-texel.x, -texel.y)).rgb;
    let se = textureSample(input_texture, input_sampler, in.uv + vec2<f32>( texel.x, -texel.y)).rgb;

    let l_nw = luminance(nw);
    let l_ne = luminance(ne);
    let l_sw = luminance(sw);
    let l_se = luminance(se);

    // Sub-pixel blend factor: 3x3 box minus center, weighted by distance.
    let l_subpix_aa = (2.0 * (ln + ls + le + lw) + (l_nw + l_ne + l_sw + l_se)) / 12.0;
    let l_subpix_delta = abs(l_subpix_aa - lc) / l_range;
    let blend = smoothstep(0.0, 1.0, l_subpix_delta) * 0.75;

    // Edge direction: horizontal vs. vertical.
    let horiz = abs(ln + ls - 2.0 * lc) * 2.0 + abs(l_ne + l_se - 2.0 * le) + abs(l_nw + l_sw - 2.0 * lw);
    let vert  = abs(le + lw - 2.0 * lc) * 2.0 + abs(l_ne + l_nw - 2.0 * ln) + abs(l_se + l_sw - 2.0 * ls);
    let is_horiz = horiz >= vert;

    // Step one texel in the direction perpendicular to the edge.
    var step_uv: vec2<f32>;
    if is_horiz {
        // Horizontal edge: blend vertically.
        let l_pos = ln;
        let l_neg = ls;
        step_uv = vec2<f32>(0.0, select(-texel.y, texel.y, l_pos >= l_neg));
    } else {
        // Vertical edge: blend horizontally.
        let l_pos = le;
        let l_neg = lw;
        step_uv = vec2<f32>(select(-texel.x, texel.x, l_pos >= l_neg), 0.0);
    }

    let sample_uv = in.uv + step_uv * blend;
    let blended = textureSample(input_texture, input_sampler, sample_uv).rgb;
    return vec4<f32>(mix(c, blended, blend), 1.0);
}
