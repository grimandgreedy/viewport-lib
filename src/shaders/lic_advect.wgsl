// lic_advect.wgsl : Surface LIC pass 2 -- advects noise along screen-space flow directions.
//
// Reads lic_vector_texture (output of lic_surface pass): RG=encoded direction, A=coverage.
// Reads lic_noise_tex: viewport-sized R8Unorm white noise (one independent random value per
// screen pixel). Each pixel is sampled with textureLoad (nearest), so samples along a
// streamline are correlated when the streamline stays within the same pixel, and independent
// when it crosses into a different pixel. This creates the directional LIC contrast.
//
// Advection follows the true streamline by re-sampling the flow direction at each step rather
// than walking in a straight line, which keeps samples correlated along the flow path.
//
// Writes normalized LIC intensity to R8Unorm output (0.5 = neutral for non-surface pixels).

struct LicAdvectUniform {
    steps:     u32,
    step_size: f32,
    vp_width:  f32,
    vp_height: f32,
}

@group(0) @binding(0) var<uniform> params: LicAdvectUniform;
@group(0) @binding(1) var lic_vector_tex: texture_2d<f32>;
@group(0) @binding(2) var lic_noise_tex:  texture_2d<f32>;
@group(0) @binding(3) var lin_samp:       sampler;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    let uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv);
}

fn sample_noise(pos: vec2<f32>) -> f32 {
    // The noise texture is viewport-sized (one random value per screen pixel).
    // textureLoad with clamped integer coords gives nearest-neighbor access.
    let dims = textureDimensions(lic_noise_tex);
    let px = clamp(vec2<i32>(pos * vec2<f32>(dims)), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    return textureLoad(lic_noise_tex, px, 0).r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let vec_sample = textureSample(lic_vector_tex, lin_samp, in.uv);

    // Alpha < 0.5 means this pixel has no surface; output neutral 0.5.
    if vec_sample.a < 0.5 {
        return vec4<f32>(0.5, 0.0, 0.0, 1.0);
    }

    // Decode screen-space direction from [0,1] to [-1,1].
    let screen_dir = vec_sample.xy * 2.0 - vec2<f32>(1.0);
    if length(screen_dir) < 1e-5 {
        return vec4<f32>(0.5, 0.0, 0.0, 1.0);
    }
    let dir = normalize(screen_dir);

    let vp      = vec2<f32>(params.vp_width, params.vp_height);
    let step_uv = params.step_size / vp;

    var sum   = 0.0;
    var count = 0u;

    // Forward advection: follow the streamline by re-sampling the direction at each step.
    // Walking in a fixed straight line produces uncorrelated samples beyond the first few
    // steps; updating the direction gives true LIC correlation along the flow path.
    var pos = in.uv;
    var fwd = dir * step_uv;
    for (var i = 0u; i < params.steps; i++) {
        pos += fwd;
        if pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 { break; }
        let v = textureSample(lic_vector_tex, lin_samp, pos);
        if v.a < 0.5 { break; }
        sum += sample_noise(pos);
        count++;
        let local_dir = v.xy * 2.0 - vec2<f32>(1.0);
        if length(local_dir) > 1e-5 {
            fwd = normalize(local_dir) * step_uv;
        }
    }

    // Backward advection: negate the direction and update it at each step.
    pos = in.uv;
    var bwd = -dir * step_uv;
    for (var i = 0u; i < params.steps; i++) {
        pos += bwd;
        if pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 { break; }
        let v = textureSample(lic_vector_tex, lin_samp, pos);
        if v.a < 0.5 { break; }
        sum += sample_noise(pos);
        count++;
        let local_dir = v.xy * 2.0 - vec2<f32>(1.0);
        if length(local_dir) > 1e-5 {
            bwd = -normalize(local_dir) * step_uv;
        }
    }

    let intensity = select(0.5, sum / f32(count), count > 0u);
    return vec4<f32>(intensity, 0.0, 0.0, 1.0);
}
