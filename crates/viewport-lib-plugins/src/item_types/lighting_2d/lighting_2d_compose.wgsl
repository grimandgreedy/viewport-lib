// Multiply the accumulated light over the scene colour: one full-screen
// triangle, the blend state doing the multiply.

@group(0) @binding(0) var light_target: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = select(-1.0, 3.0, vi == 1u);
    let y = select(-1.0, 3.0, vi == 2u);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let light = textureLoad(light_target, vec2<i32>(pos.xy), 0);
    return vec4<f32>(light.rgb, 1.0);
}
