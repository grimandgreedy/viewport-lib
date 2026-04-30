// Screen-space overlay text and solid quad shader.
//
// Renders glyph quads (textured from the glyph atlas) and solid-colour quads
// (background boxes, leader lines) in a single batched draw call.
//
// Vertex attributes carry NDC position, atlas UV, RGBA tint, and a flag
// distinguishing textured vs solid fragments.

@group(0) @binding(0) var glyph_atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

struct VertexInput {
    @location(0) position:    vec2<f32>,  // NDC xy
    @location(1) uv:          vec2<f32>,  // atlas UV (ignored for solid quads)
    @location(2) color:       vec4<f32>,  // RGBA tint
    @location(3) use_texture: f32,        // 1.0 = sample atlas, 0.0 = solid
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       uv:           vec2<f32>,
    @location(1)       color:        vec4<f32>,
    @location(2)       use_texture:  f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv            = in.uv;
    out.color         = in.color;
    out.use_texture   = in.use_texture;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.use_texture > 0.5) {
        let atlas_a = textureSample(glyph_atlas, atlas_sampler, in.uv).a;
        return vec4<f32>(in.color.rgb, in.color.a * atlas_a);
    } else {
        return in.color;
    }
}
