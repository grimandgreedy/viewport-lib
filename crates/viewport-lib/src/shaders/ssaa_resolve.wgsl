// SSAA resolve pass : box-filter downsample from an Nx supersampled texture to native resolution.
//
// Group 0, binding 0: ssaa_texture (Rgba16Float, w*factor x h*factor)
// Group 0, binding 1: sampler (nearest : we do the averaging manually)
// Group 0, binding 2: uniform { factor: u32, _pad: [u32;3] }
// Group 0, binding 3: ssaa_fraction (Rg8Unorm, the scene pass's light split)
// Group 0, binding 4: ssaa_depth (the supersampled scene depth)
//
// The vertex shader emits a fullscreen triangle. The fragment shader reads
// factor^2 texels per output pixel and averages them (box filter). The
// light split is averaged over the texels geometry covers, since the
// composite applies it to geometry alone: a silhouette pixel takes its
// surface's fraction, not one diluted by the background's.

struct SsaaUniform {
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var ssaa_texture: texture_2d<f32>;
@group(0) @binding(1) var ssaa_sampler: sampler;
@group(0) @binding(2) var<uniform> ssaa_uniform: SsaaUniform;
@group(0) @binding(3) var ssaa_fraction: texture_2d<f32>;
@group(0) @binding(4) var ssaa_depth: texture_2d<f32>;

// #include "helpers/depth_texel.wgsl"
// #include "helpers/depth.wgsl"

struct ResolveOut {
    @location(0) colour: vec4<f32>,
    @location(1) indirect_fraction: vec4<f32>,
};

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    // Fullscreen triangle: vertices at (-1,-1), (3,-1), (-1,3).
    let x = f32((vi & 1u) * 2u);
    let y = f32((vi >> 1u) * 2u);
    var out: VertexOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv  = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> ResolveOut {
    let factor = ssaa_uniform.factor;
    let dims = vec2<u32>(textureDimensions(ssaa_texture));
    // Output pixel in the native-res target.
    let out_px = vec2<u32>(in.pos.xy);
    // Accumulate factor^2 SSAA texels.
    var sum = vec4<f32>(0.0);
    var fraction = vec2<f32>(0.0);
    var covered = 0.0;
    for (var dy = 0u; dy < factor; dy = dy + 1u) {
        for (var dx = 0u; dx < factor; dx = dx + 1u) {
            let tx = min(out_px.x * factor + dx, dims.x - 1u);
            let ty = min(out_px.y * factor + dy, dims.y - 1u);
            let texel = vec2<i32>(i32(tx), i32(ty));
            sum = sum + textureLoad(ssaa_texture, texel, 0);
            if !viewport_depth_is_background(depth_texel(ssaa_depth, ssaa_sampler, texel)) {
                fraction = fraction + textureLoad(ssaa_fraction, texel, 0).rg;
                covered = covered + 1.0;
            }
        }
    }
    let n = f32(factor * factor);
    // Background alone keeps the clear value, wholly indirect.
    let resolved = select(vec2<f32>(1.0), fraction / covered, covered > 0.0);
    return ResolveOut(sum / n, vec4<f32>(resolved, 0.0, 1.0));
}
