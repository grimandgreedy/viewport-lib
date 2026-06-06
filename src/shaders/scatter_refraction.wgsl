// Refractive distortion pass for `ScatterVolume`.
//
// Each refractive volume renders one instanced screen-space quad whose
// vertex shader projects the volume's world bounding box and emits a
// 2-triangle rectangle covering its on-screen footprint. The fragment
// shader samples the previously copied scene colour at a UV offset derived
// from the local density gradient and writes the distorted result back to
// the HDR target with replace blend, so the scatter pass that follows
// integrates absorption and in-scattering on top of the shimmered scene.

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct GpuRefractionVolume {
    shape_pack: vec4<u32>,
    p0:         vec4<f32>,
    p1:         vec4<f32>,
    params:     vec4<f32>, // strength, threshold, noise_scale, time
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> vol:    GpuRefractionVolume;

@group(2) @binding(0) var scene_src:     texture_2d<f32>;
@group(2) @binding(1) var scene_sampler: sampler;
@group(2) @binding(2) var opaque_depth:  texture_depth_2d;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc_xy:   vec2<f32>,
};

fn hash31(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

fn value_noise_3d(p: vec3<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);
    let w = pf * pf * (3.0 - 2.0 * pf);
    let n000 = hash31(pi);
    let n100 = hash31(pi + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(pi + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(pi + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(pi + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(pi + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(pi + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(pi + vec3<f32>(1.0, 1.0, 1.0));
    let nx00 = mix(n000, n100, w.x);
    let nx10 = mix(n010, n110, w.x);
    let nx01 = mix(n001, n101, w.x);
    let nx11 = mix(n011, n111, w.x);
    let nxy0 = mix(nx00, nx10, w.y);
    let nxy1 = mix(nx01, nx11, w.y);
    return mix(nxy0, nxy1, w.z);
}

fn project(p: vec3<f32>) -> vec4<f32> {
    return camera.view_proj * vec4<f32>(p, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var bmin: vec3<f32>;
    var bmax: vec3<f32>;
    if vol.shape_pack.x == 0u {
        bmin = vol.p0.xyz;
        bmax = vol.p1.xyz;
    } else {
        let r = vol.p0.w;
        bmin = vol.p0.xyz - vec3<f32>(r);
        bmax = vol.p0.xyz + vec3<f32>(r);
    }
    var corners: array<vec3<f32>, 8>;
    corners[0] = vec3<f32>(bmin.x, bmin.y, bmin.z);
    corners[1] = vec3<f32>(bmax.x, bmin.y, bmin.z);
    corners[2] = vec3<f32>(bmin.x, bmax.y, bmin.z);
    corners[3] = vec3<f32>(bmax.x, bmax.y, bmin.z);
    corners[4] = vec3<f32>(bmin.x, bmin.y, bmax.z);
    corners[5] = vec3<f32>(bmax.x, bmin.y, bmax.z);
    corners[6] = vec3<f32>(bmin.x, bmax.y, bmax.z);
    corners[7] = vec3<f32>(bmax.x, bmax.y, bmax.z);

    var ss_min = vec2<f32>(1.0, 1.0);
    var ss_max = vec2<f32>(-1.0, -1.0);
    var any_behind = false;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let c = project(corners[i]);
        if c.w <= 1e-4 {
            any_behind = true;
        } else {
            let nx = c.x / c.w;
            let ny = c.y / c.w;
            ss_min = min(ss_min, vec2<f32>(nx, ny));
            ss_max = max(ss_max, vec2<f32>(nx, ny));
        }
    }
    if any_behind {
        ss_min = vec2<f32>(-1.0, -1.0);
        ss_max = vec2<f32>(1.0, 1.0);
    } else {
        ss_min = clamp(ss_min, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));
        ss_max = clamp(ss_max, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));
    }

    var p: vec2<f32>;
    if vi == 0u      { p = vec2<f32>(ss_min.x, ss_min.y); }
    else if vi == 1u { p = vec2<f32>(ss_max.x, ss_min.y); }
    else if vi == 2u { p = vec2<f32>(ss_min.x, ss_max.y); }
    else if vi == 3u { p = vec2<f32>(ss_max.x, ss_min.y); }
    else if vi == 4u { p = vec2<f32>(ss_max.x, ss_max.y); }
    else             { p = vec2<f32>(ss_min.x, ss_max.y); }

    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc_xy = p;
    return out;
}

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
    let ndc_xy = in.ndc_xy;
    let near_h = camera.inv_view_proj * vec4<f32>(ndc_xy, 0.0, 1.0);
    let far_h  = camera.inv_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p  = far_h.xyz  / far_h.w;
    let ray_dir = normalize(far_p - near_p);
    let eye = camera.eye_pos;

    let dims = textureDimensions(opaque_depth, 0);
    let uv = vec2<f32>(ndc_xy.x * 0.5 + 0.5, 1.0 - (ndc_xy.y * 0.5 + 0.5));
    let coord = vec2<i32>(
        clamp(i32(uv.x * f32(dims.x)), 0, i32(dims.x) - 1),
        clamp(i32(uv.y * f32(dims.y)), 0, i32(dims.y) - 1),
    );
    let depth = textureLoad(opaque_depth, coord, 0);
    let t_opaque = opaque_distance(ndc_xy, depth, ray_dir);

    var hit: vec2<f32>;
    if vol.shape_pack.x == 0u {
        hit = ray_box(vol.p0.xyz, vol.p1.xyz, eye, ray_dir);
    } else {
        hit = ray_sphere(vol.p0.xyz, vol.p0.w, eye, ray_dir);
    }
    let t_enter = max(hit.x, 0.0);
    let t_exit  = min(hit.y, t_opaque);
    if t_enter >= t_exit { discard; }

    let strength    = vol.params.x;
    let threshold   = vol.params.y;
    let noise_scale = max(vol.params.z, 1e-4);
    let time        = vol.params.w;

    // Sample the noise field near the volume centre along the ray. The
    // density gradient there sets the per-pixel distortion vector and the
    // scalar density gates the threshold check.
    let t_mid = mix(t_enter, t_exit, 0.5);
    let p     = eye + ray_dir * t_mid;
    let warp  = vec3<f32>(
        sin(time * 0.71),
        cos(time * 0.83),
        sin(time * 0.59 + 1.7),
    );
    let q   = p * noise_scale + warp;
    let eps = 0.35;
    let n_x0 = value_noise_3d(q - vec3<f32>(eps, 0.0, 0.0));
    let n_x1 = value_noise_3d(q + vec3<f32>(eps, 0.0, 0.0));
    let n_y0 = value_noise_3d(q - vec3<f32>(0.0, eps, 0.0));
    let n_y1 = value_noise_3d(q + vec3<f32>(0.0, eps, 0.0));
    let n_z0 = value_noise_3d(q - vec3<f32>(0.0, 0.0, eps));
    let n_z1 = value_noise_3d(q + vec3<f32>(0.0, 0.0, eps));
    let grad = vec3<f32>(n_x1 - n_x0, n_y1 - n_y0, n_z1 - n_z0);
    let mid_density = (n_x0 + n_x1 + n_y0 + n_y1 + n_z0 + n_z1) / 6.0;
    if mid_density < threshold { discard; }

    // Project the world gradient into view space to drive a 2D screen-space
    // offset. Flip Y so screen-up matches the texture-up convention.
    let g_view = (camera.view * vec4<f32>(grad, 0.0)).xyz;
    let offset = vec2<f32>(g_view.x, -g_view.y) * strength;
    let sample_uv = clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let sampled = textureSampleLevel(scene_src, scene_sampler, sample_uv, 0.0).rgb;
    return vec4<f32>(sampled, 1.0);
}
