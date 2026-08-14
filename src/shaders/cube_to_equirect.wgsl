// cube_to_equirect.wgsl : resolve six captured cube faces into an equirect
// panorama on the GPU.
//
// This mirrors the CPU resolve in `src/renderer/capture.rs`: for each output
// texel it forms the world direction under the consumer's convention (phi =
// atan2(y, x) around Z, theta = asin(z)), picks the face whose forward is
// nearest that direction, projects a point one unit along the direction through
// that face's view-projection, and samples the face at the projected UV. Doing
// it in a fragment pass keeps a fully on-GPU capture -> resolve -> prefilter
// path without a CPU round trip.

const PI:  f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

struct Resolve {
    // Per-face view-projection matrix, used to project a world point back onto
    // the face the same way it was rendered.
    view_proj: array<mat4x4<f32>, 6>,
    // Per-face forward direction (xyz); w unused. Face selection is nearest
    // forward, matching the CPU resolve.
    forward: array<vec4<f32>, 6>,
    // Capture eye position (xyz); w unused.
    eye: vec4<f32>,
};

@group(0) @binding(0) var faces: texture_2d_array<f32>;
@group(0) @binding(1) var samp:  sampler;
@group(0) @binding(2) var<uniform> params: Resolve;

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
    // Equirect texel -> world direction (matches dir_to_equirect_uv inverse in
    // helpers/ambient.wgsl and the CPU resolve).
    let theta = (0.5 - in.uv.y) * PI;
    let phi = (in.uv.x - 0.5) * TAU;
    let ct = cos(theta);
    let st = sin(theta);
    let cp = cos(phi);
    let sp = sin(phi);
    let d = vec3<f32>(ct * cp, ct * sp, st);

    // Pick the face whose forward is nearest the direction.
    var best: i32 = 0;
    var best_dot: f32 = -1.0e30;
    for (var i: i32 = 0; i < 6; i = i + 1) {
        let dp = dot(d, params.forward[i].xyz);
        if (dp > best_dot) {
            best_dot = dp;
            best = i;
        }
    }

    // Project a point one unit along the direction through the face camera. The
    // distance is irrelevant: the ray from the eye through that point hits the
    // same face texel regardless of how far along it sits.
    let clip = params.view_proj[best] * vec4<f32>(params.eye.xyz + d, 1.0);
    let inv_w = 1.0 / max(clip.w, 1.0e-6);
    let fu = clip.x * inv_w * 0.5 + 0.5;
    let fv = 1.0 - (clip.y * inv_w * 0.5 + 0.5);

    // Clamp-to-edge sampling handles directions that project just past a
    // 90-degree face boundary, matching the CPU edge clamp.
    return textureSampleLevel(faces, samp, vec2<f32>(fu, fv), best, 0.0);
}
