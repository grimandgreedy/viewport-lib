// Participating-media volume rendering: fullscreen ray-march over an array of
// box / sphere primitives. Writes a premultiplied RGBA fragment that the
// renderer alpha-composites onto the opaque scene.
//
// Group 0: shared camera + clip planes (matches mesh.wgsl / projected_tet.wgsl).
// Group 1: scatter-volume uniforms, opaque depth texture, depth sampler.

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
};

struct ClipPlanes {
    planes:          array<vec4<f32>, 6>,
    count:           u32,
    _pad0:           u32,
    viewport_width:  f32,
    viewport_height: f32,
};

// Must match GpuScatterVolume in src/scene/scatter_volume.rs. 64 bytes.
struct GpuScatterVolume {
    shape_kind:     u32,
    _pad0:          vec3<u32>,
    p0:             vec4<f32>,
    p1:             vec4<f32>,
    colour_density: vec4<f32>,
};

const MAX_SCATTER_VOLUMES: u32 = 16u;
const MARCH_STEPS:         u32 = 32u;

struct ScatterUniforms {
    volumes: array<GpuScatterVolume, MAX_SCATTER_VOLUMES>,
    count:   u32,
    _pad:    vec3<u32>,
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;

@group(1) @binding(0) var<uniform>            uniforms:      ScatterUniforms;
@group(1) @binding(1) var                      opaque_depth: texture_depth_2d;
@group(1) @binding(2) var                      depth_sampler: sampler;

struct VsOut {
    @builtin(position)         clip_pos: vec4<f32>,
    @location(0)               ndc_xy:   vec2<f32>,
};

// Fullscreen triangle covering the full viewport, no vertex buffer required.
// vi = 0,1,2 -> (-1,-1), (3,-1), (-1,3) in NDC.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32(i32(vi) - 1) * 2.0;     // -1, 3, -1 (after the i32(vi-1)*2 trick)
    let y = f32(i32(vi & 1u) * 2 - 1) * 2.0;
    // Simpler explicit form:
    var p: vec2<f32>;
    if vi == 0u { p = vec2<f32>(-1.0, -1.0); }
    else if vi == 1u { p = vec2<f32>(3.0, -1.0); }
    else { p = vec2<f32>(-1.0, 3.0); }
    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc_xy = p;
    return out;
}

// Reconstruct a world-space ray from an NDC xy position.
fn world_ray(ndc: vec2<f32>) -> vec4<f32> {
    // Returns origin in .xyz of the near point and a normalized direction is
    // returned separately by the caller.
    let near_h = camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    return near_h;
}

struct Interval {
    t_enter:    f32,
    t_exit:     f32,
    colour_rgb: vec3<f32>,
    density:    f32,
};

fn ray_box(p0: vec3<f32>, p1: vec3<f32>, o: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / d;
    let t0 = (p0 - o) * inv;
    let t1 = (p1 - o) * inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_enter, t_exit);
}

fn ray_sphere(c: vec3<f32>, r: f32, o: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    let oc = o - c;
    let a = dot(d, d);
    let b = 2.0 * dot(oc, d);
    let cc = dot(oc, oc) - r * r;
    let disc = b * b - 4.0 * a * cc;
    if disc < 0.0 { return vec2<f32>(1.0, 0.0); }
    let sq = sqrt(disc);
    let t0 = (-b - sq) / (2.0 * a);
    let t1 = (-b + sq) / (2.0 * a);
    return vec2<f32>(t0, t1);
}

// Convert an NDC depth in [0,1] (D3D / wgpu convention) at a given screen
// position into the world-space ray distance from `eye`.
fn opaque_distance(ndc_xy: vec2<f32>, depth: f32, ray_dir: vec3<f32>) -> f32 {
    if depth >= 1.0 {
        return 1e30;
    }
    let ndc = vec4<f32>(ndc_xy, depth, 1.0);
    let world_h = camera.inv_view_proj * ndc;
    let world_p = world_h.xyz / world_h.w;
    return dot(world_p - camera.eye_pos, ray_dir);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct view ray from NDC.
    let ndc_xy = in.ndc_xy;
    let near_h = camera.inv_view_proj * vec4<f32>(ndc_xy, 0.0, 1.0);
    let far_h  = camera.inv_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p  = far_h.xyz  / far_h.w;
    let ray_dir = normalize(far_p - near_p);
    let eye = camera.eye_pos;

    // Sample opaque depth at this fragment so volumes do not march past
    // already-rendered geometry.
    let uv = vec2<f32>(ndc_xy.x * 0.5 + 0.5, 1.0 - (ndc_xy.y * 0.5 + 0.5));
    let depth = textureSampleLevel(opaque_depth, depth_sampler, uv, 0.0);
    let t_opaque = opaque_distance(ndc_xy, depth, ray_dir);

    // Collect ray intervals against each volume.
    var intervals: array<Interval, MAX_SCATTER_VOLUMES>;
    var n_intervals: u32 = 0u;
    let count = min(uniforms.count, MAX_SCATTER_VOLUMES);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let v = uniforms.volumes[i];
        var hit: vec2<f32>;
        if v.shape_kind == 0u {
            hit = ray_box(v.p0.xyz, v.p1.xyz, eye, ray_dir);
        } else {
            hit = ray_sphere(v.p0.xyz, v.p0.w, eye, ray_dir);
        }
        let t_enter = max(hit.x, 0.0);
        let t_exit  = min(hit.y, t_opaque);
        if t_enter >= t_exit { continue; }
        var entry: Interval;
        entry.t_enter = t_enter;
        entry.t_exit = t_exit;
        entry.colour_rgb = v.colour_density.rgb;
        entry.density = v.colour_density.a;
        intervals[n_intervals] = entry;
        n_intervals = n_intervals + 1u;
    }

    if n_intervals == 0u {
        discard;
    }

    // Insertion sort intervals by t_enter ascending. n_intervals <= 16.
    for (var i: u32 = 1u; i < n_intervals; i = i + 1u) {
        let key = intervals[i];
        var j: i32 = i32(i) - 1;
        loop {
            if j < 0 { break; }
            if intervals[u32(j)].t_enter <= key.t_enter { break; }
            intervals[u32(j) + 1u] = intervals[u32(j)];
            j = j - 1;
        }
        intervals[u32(j + 1)] = key;
    }

    // Front-to-back composite. Each interval contributes uniform Beer-Lambert
    // absorption with a flat scattered colour. Future phases replace the
    // uniform contribution with per-step ray marching.
    var acc_rgb: vec3<f32> = vec3<f32>(0.0);
    var acc_a: f32 = 0.0;
    for (var k: u32 = 0u; k < n_intervals; k = k + 1u) {
        let iv = intervals[k];
        let thickness = max(iv.t_exit - iv.t_enter, 0.0);
        let alpha = 1.0 - exp(-iv.density * thickness);
        let one_minus_acc = 1.0 - acc_a;
        acc_rgb = acc_rgb + iv.colour_rgb * alpha * one_minus_acc;
        acc_a = acc_a + alpha * one_minus_acc;
        if acc_a > 0.995 { break; }
    }

    // Premultiplied alpha output. Caller blends with One / OneMinusSrcAlpha.
    return vec4<f32>(acc_rgb, acc_a);
}
