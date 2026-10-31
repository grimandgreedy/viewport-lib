// transmission_copy.wgsl
// Copies the opaque HDR scene into mip 0 of the transmission target, with the
// scene depth in alpha so a transmissive surface can tell whether what it
// samples lies behind it.

@group(0) @binding(0) var scene_colour: texture_2d<f32>;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
@group(0) @binding(2) var depth_sampler: sampler;

// #include "helpers/depth_texel.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi & 1u) * 2u);
    let y = f32((vi >> 1u) * 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(pos.xy);
    let colour = textureLoad(scene_colour, coord, 0).rgb;
    let depth = depth_texel(scene_depth, depth_sampler, coord);
    return vec4<f32>(colour, depth);
}
