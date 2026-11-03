// transmission_downsample.wgsl
// Builds one mip of the transmission target from the level above it: a
// bilinear tap at the centre of each 2x2 block, which averages it.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi & 1u) * 2u);
    let y = f32((vi >> 1u) * 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let src_size = vec2<f32>(textureDimensions(src));
    let dst_size = max(floor(src_size * 0.5), vec2<f32>(1.0));
    let uv = pos.xy / dst_size;
    return textureSampleLevel(src, src_sampler, uv, 0.0);
}
