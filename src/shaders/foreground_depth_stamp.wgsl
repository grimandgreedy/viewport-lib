// foreground_depth_stamp.wgsl
// Writes near depth into the output depth buffer wherever the foreground
// pass drew geometry, so post-tone-map passes (grid, ground plane, gizmos)
// are occluded by foreground items. The foreground depth itself may use a
// different projection than the scene, so the stamp writes the nearest
// possible depth rather than copying the value.

@group(0) @binding(0) var foreground_depth: texture_depth_2d;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi & 1u) * 2u);
    let y = f32((vi >> 1u) * 2u);
    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv       = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @builtin(frag_depth) f32 {
    let dims  = textureDimensions(foreground_depth);
    let coord = clamp(
        vec2<u32>(in.uv * vec2<f32>(dims)),
        vec2<u32>(0u),
        dims - vec2<u32>(1u),
    );
    if textureLoad(foreground_depth, coord, 0) >= 1.0 {
        discard;
    }
    return 0.0;
}
