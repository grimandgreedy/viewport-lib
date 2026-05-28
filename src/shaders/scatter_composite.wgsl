// Composite the scatter-volume intermediate onto the HDR target.
//
// The scatter pass renders into a half- or full-resolution RGBA16F target with
// premultiplied alpha. This shader samples that target and lets fixed-function
// blending do the alpha-over composite. When the intermediate is half-res, the
// linear sampler does a bilinear upsample.

@group(0) @binding(0) var source_tex: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var p: vec2<f32>;
    if vi == 0u { p = vec2<f32>(-1.0, -1.0); }
    else if vi == 1u { p = vec2<f32>(3.0, -1.0); }
    else { p = vec2<f32>(-1.0, 3.0); }
    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(source_tex, source_sampler, in.uv, 0.0);
}
