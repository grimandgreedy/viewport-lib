// Fullscreen blit shader: copies one texture 1:1 to the output surface.
// Used by the HDR viewport callback to transfer the rendered intermediate
// texture into the eframe render pass.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    // Fullscreen triangle: covers the entire clip-space quad with 3 vertices.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VertexOut;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv  = uvs[vi];
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(src_texture, src_sampler, in.uv);
}
