// SSAA resolve pass : box-filter downsample from an N× supersampled texture to native resolution.
//
// Group 0, binding 0: ssaa_texture (Rgba16Float, w*factor × h*factor)
// Group 0, binding 1: sampler (nearest : we do the averaging manually)
// Group 0, binding 2: uniform { factor: u32, _pad: [u32;3] }
//
// The vertex shader emits a fullscreen triangle. The fragment shader reads
// factor² texels per output pixel and averages them (box filter).

struct SsaaUniform {
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var ssaa_texture: texture_2d<f32>;
@group(0) @binding(1) var ssaa_sampler: sampler;
@group(0) @binding(2) var<uniform> ssaa_uniform: SsaaUniform;

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
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let factor = ssaa_uniform.factor;
    let dims = vec2<u32>(textureDimensions(ssaa_texture));
    // Output pixel in the native-res target.
    let out_px = vec2<u32>(in.pos.xy);
    // Accumulate factor² SSAA texels.
    var sum = vec4<f32>(0.0);
    for (var dy = 0u; dy < factor; dy = dy + 1u) {
        for (var dx = 0u; dx < factor; dx = dx + 1u) {
            let tx = min(out_px.x * factor + dx, dims.x - 1u);
            let ty = min(out_px.y * factor + dy, dims.y - 1u);
            sum = sum + textureLoad(ssaa_texture, vec2<i32>(i32(tx), i32(ty)), 0);
        }
    }
    return sum / f32(factor * factor);
}
