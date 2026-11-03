// Writes the scene depth clear value over the current viewport rect. Draws
// no colour: the pipeline's colour write mask is empty.

// #include "helpers/depth.wgsl"

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi & 1u) * 2u);
    let y = f32((vi >> 1u) * 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, VIEWPORT_DEPTH_FAR, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
