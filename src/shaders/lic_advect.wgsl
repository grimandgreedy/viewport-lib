// lic_advect.wgsl : Surface LIC pass 2 -- advects noise along screen-space flow directions.
//
// Reads lic_vector_texture (output of lic_surface pass): RG=encoded direction, B=noise, A=coverage.
// Writes normalized LIC intensity to R8Unorm output (0.5 = neutral for non-surface pixels).

struct LicAdvectUniform {
    steps:       u32,
    step_size:   f32,
    noise_scale: f32,
    vp_width:    f32,
    vp_height:   f32,
    _pad0:       f32,
    _pad1:       f32,
    _pad2:       f32,
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

    // Convert step size from pixels to UV space.
    let pixel_step = dir * params.step_size / vec2<f32>(params.vp_width, params.vp_height);

    var sum = 0.0;
    var count = 0u;

    // Forward advection.
    var pos = in.uv;
    for (var i = 0u; i < params.steps; i++) {
        pos += pixel_step;
        if pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 {
            break;
        }
        let v = textureSample(lic_vector_tex, lin_samp, pos);
        if v.a < 0.5 {
            break;
        }
        let noise_uv = pos * params.noise_scale;
        sum += textureSample(lic_noise_tex, lin_samp, noise_uv).r;
        count++;
    }

    // Backward advection.
    pos = in.uv;
    for (var i = 0u; i < params.steps; i++) {
        pos -= pixel_step;
        if pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 {
            break;
        }
        let v = textureSample(lic_vector_tex, lin_samp, pos);
        if v.a < 0.5 {
            break;
        }
        let noise_uv = pos * params.noise_scale;
        sum += textureSample(lic_noise_tex, lin_samp, noise_uv).r;
        count++;
    }

    let intensity = select(0.5, sum / f32(count), count > 0u);
    return vec4<f32>(intensity, 0.0, 0.0, 1.0);
}
