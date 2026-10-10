// ssaa_depth_resolve.wgsl
// Downsamples the supersampled depth target into the scene-resolution depth
// buffer, taking the nearest-to-camera sample of each factor x factor block.
// The render pass has no colour attachment; the fragment writes
// @builtin(frag_depth).
//
// Min rather than a single sub-sample, for two reasons. It is the right
// occlusion answer at a silhouette: the block covers two surfaces and the
// nearer one is what occludes. And it is stable across neighbouring
// destination pixels, which matters more than it looks: the decal pass and the
// sub-object highlight reconstruct world position from this buffer and take
// screen-space derivatives of it to estimate the receiver normal. Picking an
// arbitrary sub-sample per destination pixel makes that normal jump between
// neighbours, and the decal facing test (which rejects receivers more than
// ~84 degrees off the projection axis) then flips on and off across a flat
// surface, stamping decal content in stripes where none belongs.

// Layout matches ssaa_resolve.wgsl's uniform: { factor: u32, _pad: [u32; 3] }.
// Three scalar pads, not a vec3, which would push the struct to 32 bytes.
struct SsaaUniform {
    factor: u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
};

@group(0) @binding(0) var         src_depth: texture_depth_2d;
@group(0) @binding(1) var<uniform> ssaa:     SsaaUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle: covers [0,1] UV across the entire render target.
    let x = f32((vi & 1u) * 2u);
    let y = f32((vi >> 1u) * 2u);
    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv       = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @builtin(frag_depth) f32 {
    let dims   = textureDimensions(src_depth);
    let factor = max(ssaa.factor, 1u);
    // Top-left source texel of the block this destination pixel covers.
    let base = clamp(
        vec2<u32>(in.uv * vec2<f32>(dims)) / factor * factor,
        vec2<u32>(0u),
        dims - vec2<u32>(1u),
    );

    var nearest = 1.0;
    for (var dy = 0u; dy < factor; dy = dy + 1u) {
        for (var dx = 0u; dx < factor; dx = dx + 1u) {
            let coord = min(base + vec2<u32>(dx, dy), dims - vec2<u32>(1u));
            nearest = min(nearest, textureLoad(src_depth, coord, 0));
        }
    }
    return nearest;
}
