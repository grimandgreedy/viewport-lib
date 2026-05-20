// depth_blit.wgsl
// Copies a depth texture to a larger depth-only render target using nearest-neighbor sampling.
// The render pass has no color attachment; the fragment writes @builtin(frag_depth).
// Used by the HDR render path when render_scale < 1.0 to produce a native-resolution
// depth buffer for post-tone-map overlay passes (grid, gizmos, axes, etc.) that draw
// into the native-res output surface but need occlusion testing against scene geometry.

@group(0) @binding(0) var src_depth: texture_depth_2d;

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
    // Map output UV [0,1] to source pixel coordinate.
    // textureDimensions returns the scene-resolution size; dividing output UV by that
    // naturally handles the scale difference between the source and output textures.
    let dims   = textureDimensions(src_depth);
    let coord  = clamp(
        vec2<u32>(in.uv * vec2<f32>(dims)),
        vec2<u32>(0u),
        dims - vec2<u32>(1u),
    );
    return textureLoad(src_depth, coord, 0);
}
