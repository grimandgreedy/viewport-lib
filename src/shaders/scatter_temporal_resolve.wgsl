// Temporal accumulation pass for the scatter (participating media) pipeline.
//
// Reads the current frame's per-volume composite (raw_current) and the
// previous frame's accumulated result (history_prev), and blends. When
// temporal accumulation is disabled or the history is not valid the output
// passes raw through unchanged.
//
// Proper per-pixel reprojection using the previous-frame view-projection
// and a velocity buffer is future work tracked in the volumetric-effects
// plan. The current implementation samples history at the current pixel's
// uv, which is correct under no camera motion and produces a brief trail
// when the camera pans -- the standard temporal-accumulation tradeoff.

struct TemporalUniform {
    prev_view_proj: mat4x4<f32>,
    // x = history weight (0..1), y = history_valid (0/1),
    // z = reserved, w = reserved.
    temporal_pack: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: TemporalUniform;
@group(0) @binding(1) var raw_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var bilinear_sampler: sampler;
@group(0) @binding(4) var opaque_depth: texture_depth_2d;
@group(0) @binding(5) var depth_sampler: sampler;

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
    let raw = textureSampleLevel(raw_tex, bilinear_sampler, in.uv, 0.0);
    let blend = uniforms.temporal_pack.x;
    let history_valid = uniforms.temporal_pack.y > 0.5;

    // Phony references keep the matrix and depth bindings in the shader's
    // reflection so the pipeline layout matches the bind-group layout. The
    // depth_sampler binding stays declared at module scope but is not
    // referenced; wgpu tolerates layout bindings that the shader does not
    // consume.
    _ = uniforms.prev_view_proj[0];
    _ = textureLoad(opaque_depth, vec2<i32>(0, 0), 0);

    if !history_valid || blend <= 0.0 {
        return raw;
    }
    let hist = textureSampleLevel(history_tex, bilinear_sampler, in.uv, 0.0);
    return mix(raw, hist, blend);
}
