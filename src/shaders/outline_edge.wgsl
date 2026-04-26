// outline_edge.wgsl : screen-space edge detection on the selection mask.
//
// Samples the mask texture at multiple radii around each fragment to detect
// the boundary of the selection mask. Draws the outline on both sides of
// the boundary to eliminate dark gaps between the outline and the object.
//
// Group 0: mask texture + sampler + OutlineEdgeUniform.

struct OutlineEdgeUniform {
    color: vec4<f32>,
    radius: f32,
    viewport_w: f32,
    viewport_h: f32,
    _pad: f32,
};

@group(0) @binding(0) var mask_tex: texture_2d<f32>;
@group(0) @binding(1) var mask_samp: sampler;
@group(0) @binding(2) var<uniform> params: OutlineEdgeUniform;

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

// 16-tap ring sample at a given radius, returns (max, min) of mask values.
fn sample_ring(uv: vec2<f32>, r: f32, px: vec2<f32>) -> vec2<f32> {
    var mx: f32 = 0.0;
    var mn: f32 = 1.0;
    // 16 evenly spaced points around a circle.
    let s0  = textureSample(mask_tex, mask_samp, uv + vec2( 1.000,  0.000) * r * px).r;
    let s1  = textureSample(mask_tex, mask_samp, uv + vec2( 0.924,  0.383) * r * px).r;
    let s2  = textureSample(mask_tex, mask_samp, uv + vec2( 0.707,  0.707) * r * px).r;
    let s3  = textureSample(mask_tex, mask_samp, uv + vec2( 0.383,  0.924) * r * px).r;
    let s4  = textureSample(mask_tex, mask_samp, uv + vec2( 0.000,  1.000) * r * px).r;
    let s5  = textureSample(mask_tex, mask_samp, uv + vec2(-0.383,  0.924) * r * px).r;
    let s6  = textureSample(mask_tex, mask_samp, uv + vec2(-0.707,  0.707) * r * px).r;
    let s7  = textureSample(mask_tex, mask_samp, uv + vec2(-0.924,  0.383) * r * px).r;
    let s8  = textureSample(mask_tex, mask_samp, uv + vec2(-1.000,  0.000) * r * px).r;
    let s9  = textureSample(mask_tex, mask_samp, uv + vec2(-0.924, -0.383) * r * px).r;
    let s10 = textureSample(mask_tex, mask_samp, uv + vec2(-0.707, -0.707) * r * px).r;
    let s11 = textureSample(mask_tex, mask_samp, uv + vec2(-0.383, -0.924) * r * px).r;
    let s12 = textureSample(mask_tex, mask_samp, uv + vec2( 0.000, -1.000) * r * px).r;
    let s13 = textureSample(mask_tex, mask_samp, uv + vec2( 0.383, -0.924) * r * px).r;
    let s14 = textureSample(mask_tex, mask_samp, uv + vec2( 0.707, -0.707) * r * px).r;
    let s15 = textureSample(mask_tex, mask_samp, uv + vec2( 0.924, -0.383) * r * px).r;
    mx = max(mx, max(max(max(s0, s1), max(s2, s3)), max(max(s4, s5), max(s6, s7))));
    mx = max(mx, max(max(max(s8, s9), max(s10, s11)), max(max(s12, s13), max(s14, s15))));
    mn = min(mn, min(min(min(s0, s1), min(s2, s3)), min(min(s4, s5), min(s6, s7))));
    mn = min(mn, min(min(min(s8, s9), min(s10, s11)), min(min(s12, s13), min(s14, s15))));
    return vec2(mx, mn);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = vec2<f32>(1.0 / params.viewport_w, 1.0 / params.viewport_h);
    let center = textureSample(mask_tex, mask_samp, in.uv).r;
    let r = params.radius;

    // Sample at two radii for smoother coverage.
    let ring_outer = sample_ring(in.uv, r, px);
    let ring_inner = sample_ring(in.uv, r * 0.5, px);

    let max_mask = max(ring_outer.x, ring_inner.x);
    let min_mask = min(ring_outer.y, ring_inner.y);

    // Outside edge: this pixel is outside the mask but a neighbor is inside.
    let outside_edge = max_mask * (1.0 - center);
    // Inside edge: this pixel is inside the mask but a neighbor is outside.
    // This covers the boundary pixels to eliminate the dark gap.
    let inside_edge = center * (1.0 - min_mask);
    // Combined edge strength.
    let edge = max(outside_edge, inside_edge);

    if edge < 0.001 {
        discard;
    }

    return vec4<f32>(params.color.rgb, params.color.a * edge);
}
