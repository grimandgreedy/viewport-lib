// Surface LIC advect and composite: one fullscreen draw into the scene colour.
//
// Reads the vector target (RG = packed screen direction, B = strength, A =
// coverage) and a scene-sized white noise texture. Each covered pixel averages
// the noise along its streamline, re-reading the direction at every step, and
// outputs a modulation m. The pipeline blends with src * dst + dst * src, so
// the scene colour becomes colour * 2m: 0.5 leaves it alone.

struct AdvectParams {
    steps: u32,
    step_size: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: AdvectParams;
@group(0) @binding(1) var vector_tex: texture_2d<f32>;
@group(0) @binding(2) var noise_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_samp: sampler;

// Strength is written to the vector target as strength / STRENGTH_MAX.
const STRENGTH_MAX: f32 = 4.0;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
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

// One independent value per screen pixel, read without filtering: samples
// along a streamline stay correlated only while it stays inside a pixel.
fn sample_noise(pos: vec2<f32>) -> f32 {
    let dims = textureDimensions(noise_tex);
    let px = clamp(vec2<i32>(pos * vec2<f32>(dims)), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    return textureLoad(noise_tex, px, 0).r;
}

// Sum of the noise along the streamline leaving `start` in direction `sign`,
// in x, and the number of samples taken, in y.
fn advect(start: vec2<f32>, dir: vec2<f32>, sign: f32, step_uv: vec2<f32>) -> vec2<f32> {
    var sum = 0.0;
    var count = 0.0;
    var pos = start;
    var delta = sign * dir * step_uv;
    for (var i = 0u; i < params.steps; i++) {
        pos += delta;
        if pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 { break; }
        let v = textureSampleLevel(vector_tex, lin_samp, pos, 0.0);
        if v.a < 0.5 { break; }
        sum += sample_noise(pos);
        count += 1.0;
        let local_dir = v.xy * 2.0 - vec2<f32>(1.0);
        if length(local_dir) > 1e-5 {
            delta = sign * normalize(local_dir) * step_uv;
        }
    }
    return vec2<f32>(sum, count);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let centre = textureSampleLevel(vector_tex, lin_samp, in.uv, 0.0);
    // No flow surface here: leave the scene colour untouched.
    if centre.a < 0.5 {
        discard;
    }

    var intensity = 0.5;
    let screen_dir = centre.xy * 2.0 - vec2<f32>(1.0);
    if length(screen_dir) >= 1e-5 {
        let dir = normalize(screen_dir);
        let step_uv = params.step_size / vec2<f32>(textureDimensions(vector_tex));
        let total = advect(in.uv, dir, 1.0, step_uv) + advect(in.uv, dir, -1.0, step_uv);
        if total.y > 0.0 {
            intensity = total.x / total.y;
        }
    }

    // Scale the deviation from neutral by the item's strength. The clamp
    // bounds the result to between black and twice the scene colour.
    let strength = centre.b * STRENGTH_MAX;
    let m = clamp(0.5 + strength * (intensity - 0.5), 0.0, 1.0);
    return vec4<f32>(m, m, m, 1.0);
}
