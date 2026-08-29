// Shadow atlas depth viewer. Draws a corner quad showing the shadow depth atlas as greyscale.

struct AtlasBlitUniform {
    rect: vec4<f32>,  // NDC: xmin, ymin, xmax, ymax
};

@group(0) @binding(0) var<uniform> blit: AtlasBlitUniform;
@group(0) @binding(1) var shadow_atlas: texture_depth_2d;
@group(0) @binding(2) var atlas_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    // 2 triangles: (0,1,2), (0,2,3). Vertex order: BL, BR, TR, TL.
    let positions = array<vec2<f32>, 4>(
        vec2<f32>(blit.rect.x, blit.rect.y),
        vec2<f32>(blit.rect.z, blit.rect.y),
        vec2<f32>(blit.rect.z, blit.rect.w),
        vec2<f32>(blit.rect.x, blit.rect.w),
    );
    // UV: texture y=0 is top, NDC y+ is up, so flip V.
    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0),
    );
    let indices = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let i = indices[vi];
    var out: VertexOut;
    out.position = vec4<f32>(positions[i], 0.0, 1.0);
    out.uv = uvs[i];
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let depth = textureSampleLevel(shadow_atlas, atlas_sampler, in.uv, 0i);
    // Apply a mild gamma curve to spread values away from 0 and 1.
    let grey = pow(depth, 0.33);
    return vec4<f32>(grey, grey, grey, 0.9);
}
