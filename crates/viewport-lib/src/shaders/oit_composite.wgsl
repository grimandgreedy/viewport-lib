// OIT composite shader : blends weighted-blended accum/reveal into the HDR buffer.
//
// Reads the accum (Rgba16Float) and reveal (R8Unorm) textures produced by the OIT
// geometry pass and reconstructs the final composite colour using the McGuire & Bavoil
// formula, writing premultiplied alpha RGBA into the HDR colour target.
//
// Group 0:
//   binding 0 : accum_tex:  texture_2d<f32>   (Rgba16Float accumulation)
//   binding 1 : reveal_tex: texture_2d<f32>   (R8Unorm reveal / transmittance)
//   binding 2 : samp:       sampler           (linear, clamp-to-edge)

@group(0) @binding(0) var accum_tex:  texture_2d<f32>;
@group(0) @binding(1) var reveal_tex: texture_2d<f32>;
@group(0) @binding(2) var samp:       sampler;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle : no vertex buffer required.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    var out: VertexOut;
    out.clip_pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let accum = textureSample(accum_tex, samp, in.uv);
    let r     = textureSample(reveal_tex, samp, in.uv).r;

    // reveal starts at 1.0 (cleared) and decreases via dst*(1-alpha) per fragment.
    // r near 1.0 means no transparent geometry wrote here; discard to skip the blend.
    if r > 0.999 {
        discard;
    }

    let avg_colour = accum.rgb / max(accum.a, 1e-5);
    // Output premultiplied alpha so the HDR blend state (One / OneMinusSrcAlpha) composites correctly.
    return vec4<f32>(avg_colour * (1.0 - r), 1.0 - r);
}
