// Participating-media volume rendering, one volume per instanced draw.
//
// Vertex stage projects the volume's world bounding box, computes its
// screen-space AABB in NDC, and emits one of six corners of a 2-triangle
// quad that exactly covers that rectangle. When the volume straddles the
// camera near plane it falls back to a fullscreen quad. Pixels outside the
// rectangle never run the fragment shader.
//
// Fragment stage runs the ray-march for the single bound volume only -- no
// per-fragment volume loop, no in-shader interval sort, no temporal blend
// (temporal is handled by a separate resolve pass). The output is alpha-over
// composited against the cleared raw_current intermediate; the caller sorts
// instances back-to-front so the existing premultiplied alpha-over blend
// produces the correct ordering across overlapping volumes.

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct SingleLight {
    light_view_proj: mat4x4<f32>,
    pos_or_dir:      vec3<f32>,
    light_type:      u32,
    colour:          vec3<f32>,
    intensity:       f32,
    range:           f32,
    inner_angle:     f32,
    outer_angle:     f32,
    spot_direction:  vec3<f32>,
    _pad:            vec2<f32>,
};

struct Lights {
    count:                u32,
    shadow_bias:          f32,
    shadows_enabled:      u32,
    debug_vis_mode:       u32,
    sky_colour:           vec3<f32>,
    hemisphere_intensity: f32,
    ground_colour:        vec3<f32>,
    debug_vis_scale:      f32,
    lights:               array<SingleLight, 8>,
    ibl_enabled:          u32,
    ibl_intensity:        f32,
    ibl_rotation:         f32,
    show_skybox:          u32,
    debug_vis_split_x:    f32,
    _pad_dbg_a:           u32,
    _pad_dbg_b:           u32,
    _pad_dbg_c:           u32,
};

struct ClipPlanes {
    planes:          array<vec4<f32>, 6>,
    count:           u32,
    _pad0:           u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct ShadowAtlas {
    cascade_vp:        array<mat4x4<f32>, 4>,
    cascade_splits:    vec4<f32>,
    cascade_count:     u32,
    atlas_size:        f32,
    shadow_filter:     u32,
    pcss_light_radius: f32,
    atlas_rects:       array<vec4<f32>, 8>,
};

struct GpuScatterVolume {
    shape_pack:     vec4<u32>,
    p0:             vec4<f32>,
    p1:             vec4<f32>,
    colour_density: vec4<f32>,
    params:         vec4<f32>,
    remap_data:     vec4<f32>,
    remap_data2:    vec4<f32>,
    noise_pack:     vec4<f32>,
    noise_vel:      vec4<f32>,
};

struct ScatterFrameUniform {
    time_pack:  vec4<f32>,
    count_pack: vec4<u32>,
};

const MAX_MARCH_STEPS:          u32 = 128u;
const FLAG_UNLIT:               u32 = 1u;
const FLAG_RECEIVE_SHADOWS:     u32 = 2u;
const FLAG_USE_RAMP:            u32 = 4u;
const FLAG_USE_NOISE:           u32 = 8u;
const FLAG_USE_DENSITY_TEXTURE: u32 = 16u;
const REMAP_IDENTITY:    u32 = 0u;
const REMAP_SMOOTHSTEP:  u32 = 1u;
const REMAP_EXP_FALLOFF: u32 = 2u;
const EMISSION_NONE:      u32 = 0u;
const EMISSION_LINEAR:    u32 = 1u;
const EMISSION_POWER:     u32 = 2u;
const EMISSION_THRESHOLD: u32 = 3u;

@group(0) @binding(0) var<uniform> camera:         Camera;
@group(0) @binding(1) var          shadow_map:     texture_depth_2d;
@group(0) @binding(2) var          shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform> lights_uniform: Lights;
@group(0) @binding(4) var<uniform> clip_planes:    ClipPlanes;
@group(0) @binding(5) var<uniform> shadow_atlas:   ShadowAtlas;

@group(1) @binding(0) var<uniform> vol: GpuScatterVolume;

@group(2) @binding(0) var          colourmap_lut:    texture_2d<f32>;
@group(2) @binding(1) var          colourmap_sampler: sampler;
@group(2) @binding(2) var          density_texture:  texture_3d<f32>;
@group(2) @binding(3) var          density_sampler:  sampler;

@group(3) @binding(0) var<uniform> frame:          ScatterFrameUniform;
@group(3) @binding(1) var          opaque_depth:   texture_depth_2d;
@group(3) @binding(2) var          depth_sampler:  sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc_xy:   vec2<f32>,
};

// Project a world point through view_proj; return clip-space coords.
fn project(p: vec3<f32>) -> vec4<f32> {
    return camera.view_proj * vec4<f32>(p, 1.0);
}

// Vertex shader: compute the screen-space AABB of the volume's world bounds
// and emit one of six quad corners (two triangles). When the volume crosses
// the camera near plane we fall back to a fullscreen quad to handle the
// camera-inside-volume case.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Reconstruct the volume's world-space AABB. Box: (p0.xyz, p1.xyz).
    // Sphere: (centre - r, centre + r).
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
    // If any corner is behind the near plane, the AABB is unreliable: emit
    // a fullscreen quad. This covers the camera-inside-volume case.
    if any_behind {
        ss_min = vec2<f32>(-1.0, -1.0);
        ss_max = vec2<f32>(1.0, 1.0);
    } else {
        ss_min = clamp(ss_min, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));
        ss_max = clamp(ss_max, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));
    }

    // Vertex indices 0..5 map to two triangles covering the rect.
    //   (xmin,ymin) (xmax,ymin) (xmin,ymax) | (xmax,ymin) (xmax,ymax) (xmin,ymax)
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

// ---------------------------------------------------------------------------
// Fragment helpers (unchanged from prior pass apart from removing the per-
// fragment volume loop wrapper).
// ---------------------------------------------------------------------------

fn pseudo_blue_noise(pixel: vec2<f32>, frame_idx: u32) -> f32 {
    let p = pixel + vec2<f32>(f32(frame_idx & 0xFFFFu) * 0.61803398, f32(frame_idx >> 16u) * 0.37207937);
    let h = sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453123;
    return fract(h);
}

fn hash31(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

fn value_noise_3d(p: vec3<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);
    let w = pf * pf * (3.0 - 2.0 * pf);
    let n000 = hash31(pi + vec3<f32>(0.0, 0.0, 0.0));
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

fn fbm_value_noise(p_in: vec3<f32>, octaves_f: f32, lacunarity: f32) -> f32 {
    let octaves = u32(clamp(octaves_f, 1.0, 6.0));
    var sum = 0.0;
    var amp = 1.0;
    var amp_sum = 0.0;
    var freq = 1.0;
    var p = p_in;
    for (var i = 0u; i < octaves; i = i + 1u) {
        sum = sum + value_noise_3d(p * freq) * amp;
        amp_sum = amp_sum + amp;
        amp = amp * 0.5;
        freq = freq * lacunarity;
    }
    return sum / max(amp_sum, 1e-4);
}

fn apply_noise(
    flags: u32,
    noise_pack: vec4<f32>,
    noise_vel: vec3<f32>,
    world_pos: vec3<f32>,
    time: f32,
) -> f32 {
    if (flags & FLAG_USE_NOISE) == 0u {
        return 1.0;
    }
    let scale = max(noise_pack.x, 1e-4);
    let octaves = noise_pack.y;
    let time_scale = noise_pack.z;
    let lacunarity = max(noise_pack.w, 1.1);
    let warp = vec3<f32>(
        sin(time * time_scale * 0.71),
        cos(time * time_scale * 0.83),
        sin(time * time_scale * 0.59 + 1.7),
    );
    let p = (world_pos + noise_vel * time) * scale + warp;
    return fbm_value_noise(p, octaves, lacunarity);
}

fn sample_density_texture(
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    world_pos: vec3<f32>,
) -> f32 {
    let extent = max(aabb_max - aabb_min, vec3<f32>(1e-4));
    let uvw = clamp((world_pos - aabb_min) / extent, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSampleLevel(density_texture, density_sampler, uvw, 0.0).r;
}

fn apply_remap(
    kind: u32,
    data: vec4<f32>,
    data2: vec4<f32>,
    world_pos: vec3<f32>,
) -> f32 {
    if kind == REMAP_SMOOTHSTEP {
        let centre = data.xyz;
        let lo = data.w;
        let hi = max(data2.x, lo + 1e-4);
        let r = length(world_pos - centre);
        let t = clamp((r - lo) / (hi - lo), 0.0, 1.0);
        let s_curve = t * t * (3.0 - 2.0 * t);
        return 1.0 - s_curve;
    } else if kind == REMAP_EXP_FALLOFF {
        let centre = data.xyz;
        let falloff = max(data.w, 1e-4);
        let r = length(world_pos - centre);
        return exp(-falloff * r);
    }
    return 1.0;
}

fn emission_factor(kind: u32, param: f32, density: f32) -> f32 {
    if kind == EMISSION_LINEAR {
        return density;
    } else if kind == EMISSION_POWER {
        let exp_ = max(param, 1e-3);
        return pow(clamp(density, 0.0, 1.0), exp_);
    } else if kind == EMISSION_THRESHOLD {
        if density >= param { return 1.0; }
        return 0.0;
    }
    return 0.0;
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

fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(denom, 1e-4), 1.5));
}

fn sample_sun_shadow(world_pos: vec3<f32>) -> f32 {
    if lights_uniform.shadows_enabled == 0u || shadow_atlas.cascade_count == 0u {
        return 1.0;
    }
    let dist = dot(world_pos - camera.eye_pos, camera.forward);
    var cascade_idx = 0u;
    for (var i = 0u; i < shadow_atlas.cascade_count; i++) {
        if dist > shadow_atlas.cascade_splits[i] {
            cascade_idx = i + 1u;
        }
    }
    cascade_idx = min(cascade_idx, shadow_atlas.cascade_count - 1u);
    let light_clip = shadow_atlas.cascade_vp[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let tile_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if tile_uv.x < 0.0 || tile_uv.x > 1.0 || tile_uv.y < 0.0 || tile_uv.y > 1.0 ||
       ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let rect = shadow_atlas.atlas_rects[cascade_idx];
    let atlas_uv = vec2<f32>(
        mix(rect.x, rect.z, tile_uv.x),
        mix(rect.y, rect.w, tile_uv.y),
    );
    let depth = ndc.z - lights_uniform.shadow_bias;
    return textureSampleCompare(shadow_map, shadow_sampler, atlas_uv, depth);
}

// ---------------------------------------------------------------------------
// Fragment entry
// ---------------------------------------------------------------------------

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

    // Ray-vs-shape entry / exit for this single volume.
    var hit: vec2<f32>;
    if vol.shape_pack.x == 0u {
        hit = ray_box(vol.p0.xyz, vol.p1.xyz, eye, ray_dir);
    } else {
        hit = ray_sphere(vol.p0.xyz, vol.p0.w, eye, ray_dir);
    }
    let t_enter = max(hit.x, 0.0);
    let t_exit  = min(hit.y, t_opaque);
    if t_enter >= t_exit { discard; }

    // Per-volume constants.
    let flags = vol.shape_pack.y;
    let unlit = (flags & FLAG_UNLIT) != 0u;
    let recv_shadows = (flags & FLAG_RECEIVE_SHADOWS) != 0u;
    let use_ramp = (flags & FLAG_USE_RAMP) != 0u;
    let use_density_tex = (flags & FLAG_USE_DENSITY_TEXTURE) != 0u;
    let remap_kind = vol.shape_pack.z;
    let emission_kind = vol.shape_pack.w;
    let aniso = vol.params.x;
    let emission_strength = vol.params.y;
    let emission_param = vol.params.z;
    let step_budget = u32(vol.params.w);

    let global_steps = max(frame.count_pack.x, 1u);
    let blue_noise_on = frame.count_pack.y != 0u;
    let frame_idx = frame.count_pack.z;
    var steps = global_steps;
    if step_budget > 0u { steps = step_budget; }
    steps = min(steps, MAX_MARCH_STEPS);

    let thickness = t_exit - t_enter;
    let dt = thickness / f32(steps);

    // Lighting setup.
    var sun_dir = vec3<f32>(0.0, 0.0, 1.0);
    var sun_colour = vec3<f32>(0.0);
    if lights_uniform.count > 0u && lights_uniform.lights[0].light_type == 0u {
        sun_dir = normalize(lights_uniform.lights[0].pos_or_dir);
        sun_colour = lights_uniform.lights[0].colour * lights_uniform.lights[0].intensity;
    }
    let cos_theta = dot(ray_dir, sun_dir);
    let phase = phase_hg(cos_theta, aniso);
    let ambient_top = lights_uniform.sky_colour * lights_uniform.hemisphere_intensity;
    let ambient_bot = lights_uniform.ground_colour * lights_uniform.hemisphere_intensity;
    let zenith = clamp(sun_dir.z * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(ambient_bot, ambient_top, zenith);

    let time = frame.time_pack.x;
    var jitter = 0.5;
    if blue_noise_on {
        let pixel = in.clip_pos.xy;
        jitter = pseudo_blue_noise(pixel, frame_idx);
    }

    // World-space AABB for density-texture sampling.
    var aabb_min: vec3<f32>;
    var aabb_max: vec3<f32>;
    if vol.shape_pack.x == 0u {
        aabb_min = vol.p0.xyz;
        aabb_max = vol.p1.xyz;
    } else {
        let r = vol.p0.w;
        aabb_min = vol.p0.xyz - vec3<f32>(r);
        aabb_max = vol.p0.xyz + vec3<f32>(r);
    }

    var acc_rgb = vec3<f32>(0.0);
    var acc_a: f32 = 0.0;
    for (var s: u32 = 0u; s < MAX_MARCH_STEPS; s = s + 1u) {
        if s >= steps { break; }
        let t = t_enter + (f32(s) + jitter) * dt;
        let p = eye + ray_dir * t;
        let remap = apply_remap(remap_kind, vol.remap_data, vol.remap_data2, p);
        var modulator: f32;
        if use_density_tex {
            modulator = sample_density_texture(aabb_min, aabb_max, p);
        } else {
            modulator = apply_noise(flags, vol.noise_pack, vol.noise_vel.xyz, p, time);
        }
        let local_density = vol.colour_density.a * remap * modulator;
        let step_extinction = local_density * dt;
        let step_alpha = 1.0 - exp(-step_extinction);

        var local_colour = vol.colour_density.rgb;
        if use_ramp {
            let lut_uv = vec2<f32>(clamp(remap, 0.0, 1.0), 0.5);
            let lut_sample = textureSampleLevel(
                colourmap_lut, colourmap_sampler, lut_uv, 0.0,
            );
            local_colour = lut_sample.rgb * vol.colour_density.rgb;
        }

        var in_scatter = vec3<f32>(0.0);
        if !unlit {
            var sun_contrib = sun_colour * phase;
            if recv_shadows {
                sun_contrib = sun_contrib * sample_sun_shadow(p);
            }
            in_scatter = (sun_contrib + ambient) * local_colour;
        }

        var emission = vec3<f32>(0.0);
        if emission_kind != EMISSION_NONE {
            let f = emission_factor(emission_kind, emission_param, local_density);
            emission = local_colour * (emission_strength * f);
        }

        let one_minus_acc = 1.0 - acc_a;
        acc_rgb = acc_rgb + (in_scatter + emission) * step_alpha * one_minus_acc;
        acc_a = acc_a + step_alpha * one_minus_acc;
        if acc_a > 0.995 { break; }
    }

    return vec4<f32>(acc_rgb, acc_a);
}
