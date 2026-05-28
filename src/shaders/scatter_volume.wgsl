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
    shape_pack:     vec4<u32>,   // x=kind (0=box,1=sphere), y=flags, zw=pad
    p0:             vec4<f32>,
    p1:             vec4<f32>,
    colour_density: vec4<f32>,   // rgb=colour, a=density
    params:         vec4<f32>,   // x=anisotropy, yzw reserved
};

const MAX_SCATTER_VOLUMES: u32 = 16u;
const MARCH_STEPS:         u32 = 24u;
const FLAG_UNLIT:           u32 = 1u;
const FLAG_RECEIVE_SHADOWS: u32 = 2u;

struct ScatterUniforms {
    volumes:    array<GpuScatterVolume, MAX_SCATTER_VOLUMES>,
    count_pack: vec4<u32>,
};

@group(0) @binding(0) var<uniform> camera:         Camera;
@group(0) @binding(1) var          shadow_map:     texture_depth_2d;
@group(0) @binding(2) var          shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform> lights_uniform: Lights;
@group(0) @binding(4) var<uniform> clip_planes:    ClipPlanes;
@group(0) @binding(5) var<uniform> shadow_atlas:   ShadowAtlas;

@group(1) @binding(0) var<uniform> uniforms:      ScatterUniforms;
@group(1) @binding(1) var          opaque_depth:  texture_depth_2d;
@group(1) @binding(2) var          depth_sampler: sampler;

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
    t_enter:    f32,
    t_exit:     f32,
    colour_rgb: vec3<f32>,
    density:    f32,
    aniso:      f32,
    flags:      u32,
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
        if unlit {
            // Unlit short-circuit: flat colour weighted by Beer-Lambert.
            let thickness = max(iv.t_exit - iv.t_enter, 0.0);
            let alpha = 1.0 - exp(-iv.density * thickness);
            let one_minus_acc = 1.0 - acc_a;
            acc_rgb = acc_rgb + iv.colour_rgb * alpha * one_minus_acc;
            acc_a = acc_a + alpha * one_minus_acc;
        } else {
            let thickness = iv.t_exit - iv.t_enter;
            let dt = thickness / f32(MARCH_STEPS);
            let cos_theta = dot(ray_dir, sun_dir);
            let phase = phase_hg(cos_theta, iv.aniso);
            for (var s: u32 = 0u; s < MARCH_STEPS; s = s + 1u) {
                let t = iv.t_enter + (f32(s) + 0.5) * dt;
                let p = eye + ray_dir * t;
                let step_extinction = iv.density * dt;
                let step_alpha = 1.0 - exp(-step_extinction);
                // Hemisphere ambient: a coarse split between sky / ground for
                // the sun's up direction. Volumes do not have a surface
                // normal so a zenith-blend gives a reasonable shape.
                let zenith = clamp(sun_dir.z * 0.5 + 0.5, 0.0, 1.0);
                let ambient = mix(ambient_bot, ambient_top, zenith);
                var sun_contrib = sun_colour * phase;
                if recv_shadows {
                    sun_contrib = sun_contrib * sample_sun_shadow(p);
                }
                let in_scatter = (sun_contrib + ambient) * iv.colour_rgb;
                let one_minus_acc = 1.0 - acc_a;
                acc_rgb = acc_rgb + in_scatter * step_alpha * one_minus_acc;
                acc_a = acc_a + step_alpha * one_minus_acc;
                if acc_a > 0.995 { break; }
            }
        }
        if acc_a > 0.995 { break; }
    }

    return vec4<f32>(acc_rgb, acc_a);
}
