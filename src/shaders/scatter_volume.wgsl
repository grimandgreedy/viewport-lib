// Participating-media volume rendering with lighting (V2).
//
// Fullscreen ray-march over an array of box / sphere primitives. Each volume
// interval is subdivided into uniform steps; at each step the shader samples
// the directional sun (lights_uniform.lights[0] when type 0) plus the
// hemisphere ambient, applies a Henyey-Greenstein phase function, and
// accumulates in-scattering with Beer-Lambert absorption. Shadow shafts fall
// out of the cascaded-shadow-atlas sample at each step.
//
// Group 0: shared camera bind group (camera + shadow_map + shadow_sampler +
//          lights + clip_planes + shadow_atlas + IBL slots). This shader only
//          references the bindings it needs; the rest are tolerated by wgpu.
// Group 1: scatter-volume uniforms, opaque depth texture, depth sampler.

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
    shape_pack:     vec4<u32>,   // x=kind (0=box,1=sphere), y=flags, z=remap_kind, w=emission_kind
    p0:             vec4<f32>,
    p1:             vec4<f32>,
    colour_density: vec4<f32>,   // rgb=colour OR tint when FLAG_USE_RAMP, a=density
    params:         vec4<f32>,   // x=anisotropy, y=emission_strength, z=emission_param, w=reserved
    remap_data:     vec4<f32>,   // Smoothstep: (center.xyz, lo). ExpFalloff: (center.xyz, falloff).
    remap_data2:    vec4<f32>,   // Smoothstep: (hi,_,_,_). ExpFalloff: unused.
    noise_pack:     vec4<f32>,   // (scale, octaves_as_f32, time_scale, lacunarity)
    noise_vel:      vec4<f32>,   // (scroll.xyz, _)
};

const MAX_SCATTER_VOLUMES: u32 = 16u;
// Per-fragment ray-march step count. Runtime-controlled via
// `uniforms.count_pack.y` (global) and `iv.step_budget` (per volume).
const MAX_MARCH_STEPS:     u32 = 128u;
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

struct ScatterUniforms {
    volumes:    array<GpuScatterVolume, MAX_SCATTER_VOLUMES>,
    // x = count, yzw = pad (storage is vec4<u32>; first lane reused for time).
    count_pack: vec4<u32>,
    // x = elapsed seconds since renderer start, yzw reserved.
    time_pack:  vec4<f32>,
    // Previous-frame view_proj for temporal reprojection.
    prev_view_proj: mat4x4<f32>,
    // x = history blend factor (0..1), y = history valid (0/1),
    // z = temporal enabled (0/1), w = reserved.
    temporal_pack: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera:         Camera;
@group(0) @binding(1) var          shadow_map:     texture_depth_2d;
@group(0) @binding(2) var          shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform> lights_uniform: Lights;
@group(0) @binding(4) var<uniform> clip_planes:    ClipPlanes;
@group(0) @binding(5) var<uniform> shadow_atlas:   ShadowAtlas;

@group(1) @binding(0) var<uniform> uniforms:        ScatterUniforms;
@group(1) @binding(1) var          opaque_depth:    texture_depth_2d;
@group(1) @binding(2) var          depth_sampler:   sampler;
// Active colourmap LUT (256x1). Bound only when at least one volume has
// `FLAG_USE_RAMP`; otherwise a 1x1 fallback texture is bound.
@group(1) @binding(3) var          colourmap_lut:    texture_2d<f32>;
@group(1) @binding(4) var          colourmap_sampler: sampler;
// External 3D density texture. Bound only when at least one volume has
// `FLAG_USE_DENSITY_TEXTURE`; otherwise a 1x1x1 fallback (value = 1.0) is bound.
@group(1) @binding(5) var          density_texture: texture_3d<f32>;
@group(1) @binding(6) var          density_sampler: sampler;
// Previous-frame scatter result, sampled with bilinear UV reprojection for
// temporal accumulation. Bound to a 1x1 fallback texture when temporal is
// disabled or the history is not yet valid.
@group(1) @binding(7) var          history_tex:     texture_2d<f32>;
@group(1) @binding(8) var          history_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc_xy:   vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var p: vec2<f32>;
    if vi == 0u { p = vec2<f32>(-1.0, -1.0); }
    else if vi == 1u { p = vec2<f32>(3.0, -1.0); }
    else { p = vec2<f32>(-1.0, 3.0); }
    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc_xy = p;
    return out;
}

struct Interval {
    t_enter:           f32,
    t_exit:            f32,
    colour_rgb:        vec3<f32>,
    density:           f32,
    aniso:             f32,
    flags:             u32,
    remap_kind:        u32,
    emission_kind:     u32,
    emission_strength: f32,
    emission_param:    f32,
    step_budget:       u32,
    remap_data:        vec4<f32>,
    remap_data2:       vec4<f32>,
    noise_pack:        vec4<f32>,
    noise_vel:         vec4<f32>,
    tex_aabb_min:      vec3<f32>,
    tex_aabb_max:      vec3<f32>,
};

// Cheap hash-based pseudo blue-noise. Not the proper precomputed Cranley-
// Patterson sequence the plan calls for, but it has flat power spectrum at
// the screen-tile scale, which is what matters for hiding banding. Seed by
// pixel + frame index so the noise differs each frame; temporal
// accumulation (future) would integrate this out.
fn pseudo_blue_noise(pixel: vec2<f32>, frame: u32) -> f32 {
    let p = pixel + vec2<f32>(f32(frame & 0xFFFFu) * 0.61803398, f32(frame >> 16u) * 0.37207937);
    let h = sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453123;
    return fract(h);
}

// 3D value noise (no gradient table; cheaper than classic Perlin and good
// enough for fog / smoke / fire). Hash takes integer lattice coordinates.
fn hash31(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

fn value_noise_3d(p: vec3<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);
    // Smoothstep weights (Perlin's fade for nicer interpolation).
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
    return mix(nxy0, nxy1, w.z);  // [0, 1]
}

// Fractal Brownian motion: sum value-noise octaves with frequency growing by
// `lacunarity` and amplitude halving per octave. `octaves` clamped to [1, 6].
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
    return sum / max(amp_sum, 1e-4);  // [0, 1]
}

// Apply the per-volume noise driver at a world position. Returns a unitless
// multiplier in roughly [0, 1.4] (the centred mean is around 0.6-0.8). The
// shader multiplies the local density by this value, so static density is
// preserved on average.
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

// Sample the bound 3D density texture at normalized coordinates within the
// volume's world-space AABB. Returns the texture value (typically [0, 1]).
// `aabb_min` and `aabb_max` describe the volume's spatial bounds:
//   Box: shape min / max.
//   Sphere: centre +/- radius (bounding cube).
fn sample_density_texture(
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    world_pos: vec3<f32>,
) -> f32 {
    let extent = max(aabb_max - aabb_min, vec3<f32>(1e-4));
    let uvw = clamp((world_pos - aabb_min) / extent, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSampleLevel(density_texture, density_sampler, uvw, 0.0).r;
}

// Apply the per-volume density remap at a world position. Returns a unitless
// scalar in [0, 1] that the per-step extinction multiplies the base density by.
fn apply_remap(
    kind: u32,
    data: vec4<f32>,
    data2: vec4<f32>,
    world_pos: vec3<f32>,
) -> f32 {
    if kind == REMAP_SMOOTHSTEP {
        // Radial smoothstep around the volume's own centre. Density is full
        // strength up to `lo` and decays to zero at `hi`, regardless of the
        // volume's position in world space.
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

// Map a local density value through the emission curve. Result multiplies
// `emission_strength` and the volume's colour to produce emitted radiance.
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

// Henyey-Greenstein phase function. cos_theta is the angle between view ray
// and light direction; g in (-1, 1). Normalized so the integral over the
// sphere equals 1.
fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(denom, 1e-4), 1.5));
}

// Single-tap cascaded shadow sample used inside the march loop. Cheaper than
// the mesh.wgsl PCF / PCSS variants; per-step march would be prohibitively
// expensive at full quality. Returns 1.0 when lit, 0.0 when fully shadowed.
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

    // Collect ray intervals + per-volume params.
    var intervals: array<Interval, MAX_SCATTER_VOLUMES>;
    var n_intervals: u32 = 0u;
    let count = min(uniforms.count_pack.x, MAX_SCATTER_VOLUMES);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let v = uniforms.volumes[i];
        var hit: vec2<f32>;
        if v.shape_pack.x == 0u {
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
        entry.aniso = v.params.x;
        entry.flags = v.shape_pack.y;
        entry.remap_kind = v.shape_pack.z;
        entry.emission_kind = v.shape_pack.w;
        entry.emission_strength = v.params.y;
        entry.emission_param = v.params.z;
        entry.step_budget = u32(v.params.w);
        entry.remap_data = v.remap_data;
        entry.remap_data2 = v.remap_data2;
        entry.noise_pack = v.noise_pack;
        entry.noise_vel = v.noise_vel;
        if v.shape_pack.x == 0u {
            entry.tex_aabb_min = v.p0.xyz;
            entry.tex_aabb_max = v.p1.xyz;
        } else {
            let r = v.p0.w;
            entry.tex_aabb_min = v.p0.xyz - vec3<f32>(r);
            entry.tex_aabb_max = v.p0.xyz + vec3<f32>(r);
        }
        intervals[n_intervals] = entry;
        n_intervals = n_intervals + 1u;
    }
    if n_intervals == 0u {
        discard;
    }

    // Insertion sort by t_enter.
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

    // Resolve the primary directional light (lights_uniform.lights[0] if it
    // is a directional). pos_or_dir on a directional light points from the
    // surface toward the light per mesh.wgsl convention; we pass it through
    // the same way so `phase_hg(dot(ray_dir, light_dir), g)` matches V2's
    // sign convention (positive g = forward scattering toward the camera
    // when the light is behind the volume).
    var sun_dir = vec3<f32>(0.0, 0.0, 1.0);
    var sun_colour = vec3<f32>(0.0);
    if lights_uniform.count > 0u && lights_uniform.lights[0].light_type == 0u {
        sun_dir = normalize(lights_uniform.lights[0].pos_or_dir);
        sun_colour = lights_uniform.lights[0].colour * lights_uniform.lights[0].intensity;
    }
    let ambient_top = lights_uniform.sky_colour * lights_uniform.hemisphere_intensity;
    let ambient_bot = lights_uniform.ground_colour * lights_uniform.hemisphere_intensity;

    // Front-to-back composite over intervals; within each, ray-march N steps.
    var acc_rgb: vec3<f32> = vec3<f32>(0.0);
    var acc_a: f32 = 0.0;
    for (var k: u32 = 0u; k < n_intervals; k = k + 1u) {
        let iv = intervals[k];
        let unlit = (iv.flags & FLAG_UNLIT) != 0u;
        let recv_shadows = (iv.flags & FLAG_RECEIVE_SHADOWS) != 0u;
        let global_steps = max(uniforms.count_pack.y, 1u);
        let blue_noise_on = uniforms.count_pack.z != 0u;
        let frame_idx = uniforms.count_pack.w;
        // Per-volume override beats global; both are clamped at pack time.
        var steps = global_steps;
        if iv.step_budget > 0u { steps = iv.step_budget; }
        steps = min(steps, MAX_MARCH_STEPS);
        let thickness = iv.t_exit - iv.t_enter;
        let dt = thickness / f32(steps);
        let cos_theta = dot(ray_dir, sun_dir);
        let phase = phase_hg(cos_theta, iv.aniso);
        // Hemisphere ambient: a coarse split between sky / ground for the
        // sun's up direction. Volumes have no surface normal so a zenith
        // blend gives a reasonable shape.
        let zenith = clamp(sun_dir.z * 0.5 + 0.5, 0.0, 1.0);
        let ambient = mix(ambient_bot, ambient_top, zenith);
        let use_ramp = (iv.flags & FLAG_USE_RAMP) != 0u;
        let use_density_tex = (iv.flags & FLAG_USE_DENSITY_TEXTURE) != 0u;
        let time = uniforms.time_pack.x;
        // Blue-noise jitter on the march start offset hides banding at low
        // step counts. Without temporal accumulation it adds high-frequency
        // dithering, which the eye tolerates much better than banding.
        var jitter = 0.5;
        if blue_noise_on {
            let pixel = in.clip_pos.xy;
            jitter = pseudo_blue_noise(pixel, frame_idx);
        }
        // World-space AABB for density-texture sampling. For Box volumes the
        // ray-intersection uses p0..p1 directly; for Sphere we have centre
        // (p0.xyz) and radius (p0.w), so the AABB is centre +/- radius.
        var aabb_min: vec3<f32>;
        var aabb_max: vec3<f32>;
        // The original `GpuScatterVolume` for this interval is not retained
        // verbatim, but we can reconstruct the AABB from p0/p1 and the
        // shape kind. Re-fetch from the uniforms array by stable ordering
        // would be cleaner; for V4 we encode the shape kind into a single
        // bit of `remap_data.w` only when noise/texture flag is set, but
        // that conflicts with existing remap semantics. Instead we accept
        // a small redundancy: pass the AABB through with each interval.
        // (See Interval struct below.)
        aabb_min = iv.tex_aabb_min;
        aabb_max = iv.tex_aabb_max;
        for (var s: u32 = 0u; s < MAX_MARCH_STEPS; s = s + 1u) {
            if s >= steps { break; }
            let t = iv.t_enter + (f32(s) + jitter) * dt;
            let p = eye + ray_dir * t;
            let remap = apply_remap(iv.remap_kind, iv.remap_data, iv.remap_data2, p);
            var modulator: f32;
            if use_density_tex {
                modulator = sample_density_texture(aabb_min, aabb_max, p);
            } else {
                modulator = apply_noise(iv.flags, iv.noise_pack, iv.noise_vel.xyz, p, time);
            }
            let local_density = iv.density * remap * modulator;
            let step_extinction = local_density * dt;
            let step_alpha = 1.0 - exp(-step_extinction);

            // Local colour: either the volume's flat colour (tint of [1,1,1]
            // by default for Ramp volumes) or a colourmap LUT sample driven
            // by the remap value `remap` in [0, 1].
            var local_colour = iv.colour_rgb;
            if use_ramp {
                let lut_uv = vec2<f32>(clamp(remap, 0.0, 1.0), 0.5);
                let lut_sample = textureSampleLevel(
                    colourmap_lut, colourmap_sampler, lut_uv, 0.0,
                );
                local_colour = lut_sample.rgb * iv.colour_rgb;
            }

            // In-scattering: sun (shadowed) + ambient, modulated by volume
            // colour. Skipped when `unlit` so the volume reads as a flat
            // medium driven only by colour, density, and emission.
            var in_scatter = vec3<f32>(0.0);
            if !unlit {
                var sun_contrib = sun_colour * phase;
                if recv_shadows {
                    sun_contrib = sun_contrib * sample_sun_shadow(p);
                }
                in_scatter = (sun_contrib + ambient) * local_colour;
            }

            // Emission: self-emitted radiance proportional to the local
            // density mapped through the per-volume curve. Always added,
            // never shadow-attenuated.
            var emission = vec3<f32>(0.0);
            if iv.emission_kind != EMISSION_NONE {
                let f = emission_factor(iv.emission_kind, iv.emission_param, local_density);
                emission = local_colour * (iv.emission_strength * f);
            }

            let one_minus_acc = 1.0 - acc_a;
            acc_rgb = acc_rgb + (in_scatter + emission) * step_alpha * one_minus_acc;
            acc_a = acc_a + step_alpha * one_minus_acc;
            if acc_a > 0.995 { break; }
        }
        if acc_a > 0.995 { break; }
    }

    var curr = vec4<f32>(acc_rgb, acc_a);

    // Temporal accumulation: reproject the current pixel into the previous
    // frame and blend with the history result. The reprojection uses opaque
    // depth when available (so the trail follows scene surfaces) and falls
    // back to a midpoint along the ray when the pixel hits sky.
    let temporal_on = uniforms.temporal_pack.z > 0.5;
    let history_valid = uniforms.temporal_pack.y > 0.5;
    if temporal_on && history_valid {
        var depth_for_reproj = depth;
        var t_reproj: f32 = 0.0;
        if depth >= 1.0 {
            // Sky / no opaque hit: project the midpoint of the first
            // interval. Better than a fixed depth because volumes near the
            // camera get a tighter reproject.
            if n_intervals > 0u {
                t_reproj = 0.5 * (intervals[0].t_enter + intervals[0].t_exit);
            } else {
                t_reproj = 0.0;
            }
        }
        var world_for_reproj: vec3<f32>;
        if depth < 1.0 {
            let ndc = vec4<f32>(ndc_xy, depth, 1.0);
            let world_h = camera.inv_view_proj * ndc;
            world_for_reproj = world_h.xyz / world_h.w;
        } else {
            world_for_reproj = eye + ray_dir * t_reproj;
        }
        let prev_clip = uniforms.prev_view_proj * vec4<f32>(world_for_reproj, 1.0);
        if prev_clip.w > 1e-4 {
            let prev_ndc = prev_clip.xyz / prev_clip.w;
            let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));
            // Reject samples that fell off-screen, behind the previous near
            // plane, or past the previous far plane.
            if prev_uv.x >= 0.0 && prev_uv.x <= 1.0 &&
               prev_uv.y >= 0.0 && prev_uv.y <= 1.0 &&
               prev_ndc.z >= 0.0 && prev_ndc.z <= 1.0 {
                let hist = textureSampleLevel(history_tex, history_sampler, prev_uv, 0.0);
                let blend = uniforms.temporal_pack.x;
                curr = mix(curr, hist, blend);
            }
        }
    }

    return curr;
}
