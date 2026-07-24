// Instanced OIT (order-independent transparency) mesh shader.
//
// Identical to mesh_instanced.wgsl except the fragment shader outputs two
// weighted-blended OIT targets instead of a single HDR colour:
//   @location(0) accum  : Rgba16Float accumulation buffer
//   @location(1) reveal : R8Unorm   reveal (transmittance) buffer
//
// Group 0: Camera + shadow atlas + lights + clip planes (unchanged from mesh_instanced.wgsl).
// Group 1: Instance storage buffer + albedo + sampler + normal map + AO map.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
    forward: vec3<f32>,
    _pad1: f32,
    inv_view_proj: mat4x4<f32>,
};

// Shared light struct definitions and `lights_storage` binding 13 of group 0.
// #include "scene_lighting.wgsl"

// Frozen fragment-shading hook structs (ShadingSurface, LightSample).
// #include "shade.wgsl"

// Per-vertex deformation hook contract.
// #include "deform.wgsl"

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count: u32,
    _pad0: u32,
    viewport_width: f32,
    viewport_height: f32,
};

struct ShadowAtlas {
    cascade_vp: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    cascade_count: u32,
    atlas_size: f32,
    shadow_filter: u32,
    pcss_light_radius: f32,
    atlas_rects: array<vec4<f32>, 8>,
};

struct InstanceData {
    model: mat4x4<f32>,
    colour: vec4<f32>,
    selected: u32,
    wireframe: u32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    has_texture: u32,
    use_pbr: u32,
    metallic: f32,
    roughness: f32,
    has_normal_map: u32,
    has_ao_map: u32,
    unlit: u32,
    receive_shadows: u32,
    use_flat: u32,
    normal_strength: f32,
    uv_transform: vec4<f32>,
    ao_range: vec2<f32>,                  // (min, max) remap of AO map R sample
    alpha_cutoff: f32,                    // Mask cutoff (albedo alpha threshold)
    alpha_flag: u32,                      // 1 = alpha-test enabled, 0 = off
};

struct ClipVolumeEntry {
    volume_type: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    center: vec3<f32>,
    radius: f32,
    half_extents: vec3<f32>,
    _pad1: f32,
    col0: vec3<f32>,
    _pad2: f32,
    col1: vec3<f32>,
    _pad3: f32,
    col2: vec3<f32>,
    _pad4: f32,
}

struct ClipVolumeUB {
    count: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    volumes: array<ClipVolumeEntry, 4>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform> lights_uniform: Lights;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(5) var<uniform> shadow_atlas: ShadowAtlas;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;
@group(0) @binding(7) var ibl_irradiance: texture_2d<f32>;
@group(0) @binding(8) var ibl_prefiltered: texture_2d<f32>;
@group(0) @binding(9) var ibl_brdf_lut: texture_2d<f32>;
@group(0) @binding(10) var ibl_sampler: sampler;
@group(0) @binding(11) var ibl_skybox: texture_2d<f32>;
@group(0) @binding(12) var<storage, read_write> debug_frag_buf: array<vec4<f32>>;

// #include "clip_volume_test.wgsl"
@group(1) @binding(0) var<storage, read> instances:          array<InstanceData>;
@group(1) @binding(1) var                obj_texture:        texture_2d<f32>;
@group(1) @binding(2) var                obj_sampler:        sampler;
@group(1) @binding(3) var                normal_map:         texture_2d<f32>;
@group(1) @binding(4) var                ao_map:             texture_2d<f32>;
@group(1) @binding(5) var<storage, read> visibility_indices: array<u32>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) colour:          vec4<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) world_pos:      vec3<f32>,
    @location(3) uv:             vec2<f32>,
    @location(4) world_tangent:  vec4<f32>,
    @location(5) @interpolate(flat) instance_idx: u32,
};

struct OitOut {
    @location(0) accum:  vec4<f32>,
    @location(1) reveal: f32,
};

@vertex
fn vs_main(in: VertexIn, @builtin(instance_index) idx: u32) -> VertexOut {
    let inst = instances[idx];
    var out: VertexOut;
    var dv = DeformVertex(in.position, in.normal, in.vertex_index);
    let dctx = DeformContext(inst.model, inst.model[3].xyz, 0.0, 0u, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let model3 = mat3x3<f32>(
        inst.model[0].xyz,
        inst.model[1].xyz,
        inst.model[2].xyz,
    );
    let world_pos4 = inst.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv.normal = normalize(model3 * dv.normal);
    dv = viewport_deform_world_space(dv, dctx);
    let world_pos = vec4<f32>(dv.position, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.colour = in.colour;
    out.world_pos = world_pos.xyz;
    out.world_normal = dv.normal;
    out.world_tangent = vec4<f32>(normalize(model3 * in.tangent.xyz), in.tangent.w);
    out.uv = in.uv;
    out.instance_idx = idx;
    return out;
}

// GPU-driven cull variant: identical to vs_main but looks up the actual
// instance index from visibility_indices before reading instance data.
@vertex
fn vs_main_cull(in: VertexIn, @builtin(instance_index) idx: u32) -> VertexOut {
    let actual_idx = visibility_indices[idx];
    let inst = instances[actual_idx];
    var out: VertexOut;
    var dv = DeformVertex(in.position, in.normal, in.vertex_index);
    let dctx = DeformContext(inst.model, inst.model[3].xyz, 0.0, 0u, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let model3 = mat3x3<f32>(
        inst.model[0].xyz,
        inst.model[1].xyz,
        inst.model[2].xyz,
    );
    let world_pos4 = inst.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv.normal = normalize(model3 * dv.normal);
    dv = viewport_deform_world_space(dv, dctx);
    let world_pos = vec4<f32>(dv.position, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.colour = in.colour;
    out.world_pos = world_pos.xyz;
    out.world_normal = dv.normal;
    out.world_tangent = vec4<f32>(normalize(model3 * in.tangent.xyz), in.tangent.w);
    out.uv = in.uv;
    out.instance_idx = actual_idx;
    return out;
}

// ShadowSample stub: no shadow sampling in this shader (transparent instances skip CSM).
// Declared so debug_vis.wgsl can reference last_shadow_sample uniformly across all variants.
struct ShadowSample {
    factor: f32,
    cascade_idx: u32,
    atlas_uv: vec2<f32>,
    tile_uv: vec2<f32>,
    biased_depth: f32,
    surface_depth: f32,
    normal_bias_world: f32,
}

// ---------------------------------------------------------------------------
// PBR BRDF helpers (Cook-Torrance)
// ---------------------------------------------------------------------------
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}
fn G1_Smith(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}
fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return G1_Smith(NdotV, roughness) * G1_Smith(NdotL, roughness);
}
fn F_Schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}
// IBL helpers : canonical source: mesh.wgsl
// Keep in sync with: mesh.wgsl, mesh_instanced.wgsl, mesh_oit.wgsl
const IBL_PI: f32 = 3.14159265;
fn dir_to_equirect_uv(dir: vec3<f32>, rotation: f32) -> vec2<f32> {
    let s = sin(rotation); let c = cos(rotation);
    let d = vec3<f32>(c * dir.x - s * dir.y, s * dir.x + c * dir.y, dir.z);
    return vec2<f32>(0.5 + atan2(d.y, d.x) / (2.0 * IBL_PI), 0.5 - asin(clamp(d.z, -1.0, 1.0)) / IBL_PI);
}
fn sample_ibl_irradiance(N: vec3<f32>, rotation: f32) -> vec3<f32> {
    return textureSampleLevel(ibl_irradiance, ibl_sampler, dir_to_equirect_uv(N, rotation), 0.0).rgb;
}
fn sample_ibl_prefiltered(R: vec3<f32>, roughness: f32, rotation: f32) -> vec3<f32> {
    // Mip floored by the screen-space footprint of R (see mesh.wgsl).
    let dr = max(length(dpdx(R)), length(dpdy(R)));
    let footprint_mip = clamp(log2(max(dr * 128.0 / (2.0 * IBL_PI), 1.0)), 0.0, 4.0);
    let mip = max(roughness * 4.0, footprint_mip);
    return textureSampleLevel(ibl_prefiltered, ibl_sampler, dir_to_equirect_uv(R, rotation), mip).rgb;
}
// Geometric specular AA: widen roughness by normal variance (see mesh.wgsl).
fn specular_aa_roughness(N: vec3<f32>, roughness: f32) -> f32 {
    let du = dpdx(N);
    let dv = dpdy(N);
    let kernel = min(0.5 * (dot(du, du) + dot(dv, dv)), 0.18);
    let alpha = roughness * roughness;
    return sqrt(sqrt(clamp(alpha * alpha + kernel, 0.0, 1.0)));
}
fn sample_brdf_lut(NdotV: f32, roughness: f32) -> vec2<f32> {
    return textureSampleLevel(ibl_brdf_lut, ibl_sampler, vec2<f32>(NdotV, roughness), 0.0).rg;
}
fn F_Schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}
struct IblContrib {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

fn ibl_ambient(N: vec3<f32>, V: vec3<f32>, base_colour: vec3<f32>, metallic: f32,
               roughness: f32, F0: vec3<f32>, ao: f32, intensity: f32, rotation: f32) -> IblContrib {
    let NdotV = max(dot(N, V), 0.001);
    let F = F_Schlick_roughness(NdotV, F0, roughness);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let irradiance = sample_ibl_irradiance(N, rotation);
    let R = reflect(-V, N);
    let prefiltered = sample_ibl_prefiltered(R, roughness, rotation);
    let brdf = sample_brdf_lut(NdotV, roughness);
    let diffuse_ibl = kD * irradiance * base_colour * ao * intensity;
    let specular_ibl = prefiltered * (F * brdf.x + brdf.y) * ao * intensity;
    return IblContrib(diffuse_ibl, specular_ibl);
}

fn pbr_light_contrib(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, radiance: vec3<f32>,
    base_colour: vec3<f32>, metallic: f32, roughness: f32, F0: vec3<f32>,
) -> vec3<f32> {
    let H = normalize(L + V);
    let NdotL = max(dot(N, L), 0.0);
    if NdotL <= 0.0 { return vec3<f32>(0.0); }
    let NdotV = max(dot(N, V), 0.001);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);
    let D = D_GGX(NdotH, roughness);
    let G = G_Smith(NdotV, NdotL, roughness);
    let F = F_Schlick(HdotV, F0);
    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
    let specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.001);
    return (kD * base_colour / 3.14159265 + specular) * radiance * NdotL;
}

// ---------------------------------------------------------------------------
// OIT fragment shader : weighted blended output
// ---------------------------------------------------------------------------
struct Surface {
    resolved: bool,
    out_oit: OitOut,
    base_colour: vec3<f32>,
    normal: vec3<f32>,
    ao_factor: f32,
    mat_uv: vec2<f32>,
    alpha: f32,
    front_facing: u32,
};

// Fill the frozen plugin-facing ShadingSurface (shade.wgsl) from the resolved
// surface and the unpacked PBR terms. Called only from the shade-slot marker
// regions of plugin-composed modules; unused in the base module. The UV
// derivatives are taken here, before the light loop's non-uniform control
// flow, so hook bodies can textureSampleGrad. The instanced pipelines draw
// with backface culling and no backface policy, so `front_facing` is 1.
fn build_shading_surface(
    surface: Surface,
    in: VertexOut,
    V: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
) -> ShadingSurface {
    var surf: ShadingSurface;
    surf.base_colour = surface.base_colour;
    surf.normal = surface.normal;
    // Keep the geometric normal in the same hemisphere as the shading normal.
    var ng = normalize(in.world_normal);
    if dot(ng, surface.normal) < 0.0 { ng = -ng; }
    surf.geometric_normal = ng;
    surf.view_dir = V;
    surf.world_pos = in.world_pos;
    // Tangent frame, orthonormalised against the shading normal. A degenerate
    // mesh tangent gets a synthesised frame instead of NaNs.
    let n = surface.normal;
    var t = in.world_tangent.xyz - dot(in.world_tangent.xyz, n) * n;
    let t_len = length(t);
    if t_len > 1e-5 {
        t = t / t_len;
    } else {
        let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.9);
        t = normalize(cross(up, n));
    }
    let handedness = select(in.world_tangent.w, 1.0, in.world_tangent.w == 0.0);
    surf.tangent = t;
    surf.bitangent = cross(n, t) * handedness;
    surf.f0 = F0;
    surf.metallic = metallic;
    surf.roughness = roughness;
    surf.ao = surface.ao_factor;
    surf.alpha = surface.alpha;
    surf.uv = surface.mat_uv;
    surf.uv_ddx = dpdx(surface.mat_uv);
    surf.uv_ddy = dpdy(surface.mat_uv);
    surf.front_facing = surface.front_facing;
    // Filled from the injected varying in modules whose hook reads the
    // per-vertex extension attribute; zero everywhere else.
    surf.attr = vec4<f32>(0.0);
    return surf;
}

struct LitResult {
    rgb: vec3<f32>,
    dbg_direct_lum: f32,
    dbg_ambient_lum: f32,
    dbg_ibl_diff_lum: f32,
    dbg_ibl_spec_lum: f32,
    dbg_roughness: f32,
    dbg_metallic: f32,
    last_shadow_sample: ShadowSample,
};

// Material prep for the instanced transparent path. Unlit fully determines the
// colour and sets `resolved`; otherwise the surface fields feed compute_lit.
fn compute_surface(in: VertexOut) -> Surface {
    let inst = instances[in.instance_idx];

    var out: Surface;
    out.resolved = false;
    out.out_oit.accum = vec4<f32>(0.0);
    out.out_oit.reveal = 0.0;
    out.base_colour = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.ao_factor = 1.0;
    out.mat_uv = in.uv;
    out.alpha = 1.0;
    out.front_facing = 1u;

    for (var i = 0u; i < clip_planes.count; i++) {
        let plane = clip_planes.planes[i];
        if dot(in.world_pos, plane.xyz) + plane.w < 0.0 { discard; }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    let mat_uv = in.uv * inst.uv_transform.zw + inst.uv_transform.xy;
    out.mat_uv = mat_uv;

    var tex_colour = vec4<f32>(1.0);
    if inst.has_texture == 1u { tex_colour = textureSample(obj_texture, obj_sampler, mat_uv); }
    let obj_colour = vec4<f32>(
        inst.colour.rgb * in.colour.rgb * tex_colour.rgb,
        inst.colour.a   * in.colour.a   * tex_colour.a,
    );
    out.alpha = obj_colour.a;
    let base_colour = obj_colour.rgb;

    // Unlit: skip all lighting, return raw colour directly through OIT.
    if inst.unlit != 0u {
        let alpha = obj_colour.a;
        let w = alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + pow(abs(in.clip_pos.z / in.clip_pos.w), 4.0))));
        out.resolved = true;
        out.out_oit.accum  = vec4<f32>(base_colour * alpha, alpha) * w;
        out.out_oit.reveal = alpha;
        return out;
    }

    var N: vec3<f32>;
    if inst.use_flat != 0u {
        let dpx = dpdx(in.world_pos);
        let dpy = dpdy(in.world_pos);
        var Nf = normalize(cross(dpx, dpy));
        if dot(Nf, in.world_normal) < 0.0 { Nf = -Nf; }
        N = Nf;
    } else if inst.has_normal_map != 0u {
        let nm_sample = textureSample(normal_map, obj_sampler, mat_uv).rgb;
        var ts_unpacked = nm_sample * 2.0 - vec3<f32>(1.0);
        ts_unpacked.x = ts_unpacked.x * inst.normal_strength;
        ts_unpacked.y = ts_unpacked.y * inst.normal_strength;
        let ts_normal = normalize(ts_unpacked);
        let T = normalize(in.world_tangent.xyz);
        let Ng = normalize(in.world_normal);
        let T_orth = normalize(T - dot(T, Ng) * Ng);
        let B = cross(Ng, T_orth) * in.world_tangent.w;
        let TBN = mat3x3<f32>(T_orth, B, Ng);
        N = normalize(TBN * ts_normal);
    } else {
        N = normalize(in.world_normal);
    }

    var ao_factor = 1.0;
    if inst.has_ao_map != 0u {
        let raw_ao = textureSample(ao_map, obj_sampler, mat_uv).r;
        ao_factor = mix(inst.ao_range.x, inst.ao_range.y, raw_ao);
    }

    out.base_colour = base_colour;
    out.normal = N;
    out.ao_factor = ao_factor;
    return out;
}

// Lighting for the instanced transparent path. Skips shadow sampling.
fn compute_lit(surface: Surface, in: VertexOut) -> LitResult {
    let inst = instances[in.instance_idx];
    var base_colour = surface.base_colour;
    let ao_factor = surface.ao_factor;
    var N = surface.normal;

    let V = normalize(camera.eye_pos - in.world_pos);
    let tint = vec4<f32>(1.0);
    var last_shadow_sample = ShadowSample(1.0, 0u, vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0, 0.0);
    var final_rgb: vec3<f32>;

    var dbg_direct_lum   = 0.0;
    var dbg_ambient_lum  = 0.0;
    var dbg_ibl_diff_lum = 0.0;
    var dbg_ibl_spec_lum = 0.0;
    var dbg_roughness    = 0.5;
    var dbg_metallic     = 0.0;
    let lum_weights = vec3<f32>(0.2126, 0.7152, 0.0722);

    if inst.use_pbr != 0u {
        var metallic  = clamp(inst.metallic,  0.0, 1.0);
        var roughness = specular_aa_roughness(N, max(inst.roughness, 0.04));
        var F0 = mix(vec3<f32>(0.04), base_colour, metallic);
        // Plugin shading hooks: the composer fills the shade-slot regions in
        // plugin-composed modules; in the base module they are inert comments.
        // <viewport-shade-slot:surface>
        // </viewport-shade-slot:surface>
        var Lo = vec3<f32>(0.0);
        let pbr_range = cluster_light_range(in.world_pos, lights_uniform.count);
        for (var j = 0u; j < pbr_range.count; j++) {
            let i = cluster_light_global(pbr_range, j);
            let ev = eval_light(lights_storage[i], in.world_pos);
            if !ev.in_range { continue; }
            let L = ev.l;
            let radiance = ev.radiance;
            // Backfacing: pbr_light_contrib returns exactly zero; skip it.
            // <viewport-shade-slot:backface-cull>
            if dot(N, L) <= 0.0 { continue; }
            // </viewport-shade-slot:backface-cull>
            // Transparent surfaces: skip shadow map sampling.
            // <viewport-shade-slot:light>
            Lo += pbr_light_contrib(N, V, L, radiance, base_colour, metallic, roughness, F0);
            // </viewport-shade-slot:light>
        }
        dbg_direct_lum = dot(Lo, lum_weights);
        dbg_roughness  = roughness;
        dbg_metallic   = metallic;
        // <viewport-shade-slot:ambient>
        var ambient: vec3<f32>;
        if lights_uniform.ibl_enabled != 0u {
            let ibl = ibl_ambient(N, V, base_colour, metallic, roughness, F0,
                                  ao_factor, lights_uniform.ibl_intensity,
                                  lights_uniform.ibl_rotation);
            ambient = ibl.diffuse + ibl.specular;
            dbg_ibl_diff_lum = dot(ibl.diffuse, lum_weights);
            dbg_ibl_spec_lum = dot(ibl.specular, lum_weights);
            dbg_ambient_lum  = dbg_ibl_diff_lum + dbg_ibl_spec_lum;
        } else {
            let hemi_t = clamp(in.world_normal.z * 0.5 + 0.5, 0.0, 1.0);
            let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
            let ambient_scale = vec3<f32>(inst.ambient) + hemi_colour * lights_uniform.hemisphere_intensity;
            ambient = ambient_scale * (base_colour * (1.0 - metallic) + F0 * metallic) * ao_factor;
            dbg_ambient_lum = dot(ambient, lum_weights);
        }
        // </viewport-shade-slot:ambient>
        final_rgb = clamp((Lo + ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        // <viewport-shade-slot:recolor>
        // </viewport-shade-slot:recolor>
    } else {
        var total_colour_contrib = vec3<f32>(0.0);
        let bp_range = cluster_light_range(in.world_pos, lights_uniform.count);
        for (var j = 0u; j < bp_range.count; j++) {
            let i = cluster_light_global(bp_range, j);
            let ev = eval_light(lights_storage[i], in.world_pos);
            if !ev.in_range { continue; }
            // Transparent surfaces: skip shadow map sampling.
            let H = normalize(ev.l + V);
            let n_dot_l = max(dot(N, ev.l), 0.0);
            let n_dot_h = max(dot(N, H), 0.0);
            let diffuse_contrib  = inst.diffuse  * n_dot_l;
            let specular_contrib = inst.specular * pow(n_dot_h, inst.shininess);
            total_colour_contrib += (diffuse_contrib + specular_contrib) * ev.radiance;
        }
        let ambient_contrib = inst.ambient;
        let hemi_t = clamp(in.world_normal.z * 0.5 + 0.5, 0.0, 1.0);
        let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
        let hemi_ambient = hemi_colour * lights_uniform.hemisphere_intensity;
        let direct_rgb = base_colour * total_colour_contrib;
        dbg_direct_lum  = dot(direct_rgb, lum_weights);
        let hemi_rgb = base_colour * (ambient_contrib + hemi_ambient) * ao_factor;
        dbg_ambient_lum = dot(hemi_rgb, lum_weights);
        let lit_rgb = hemi_rgb + direct_rgb;
        final_rgb = clamp(lit_rgb * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    var res: LitResult;
    res.rgb = final_rgb;
    res.dbg_direct_lum = dbg_direct_lum;
    res.dbg_ambient_lum = dbg_ambient_lum;
    res.dbg_ibl_diff_lum = dbg_ibl_diff_lum;
    res.dbg_ibl_spec_lum = dbg_ibl_spec_lum;
    res.dbg_roughness = dbg_roughness;
    res.dbg_metallic = dbg_metallic;
    res.last_shadow_sample = last_shadow_sample;
    return res;
}

@fragment
fn fs_oit_main(in: VertexOut) -> OitOut {
    let surface = compute_surface(in);
    if surface.resolved {
        return surface.out_oit;
    }

    let lit = compute_lit(surface, in);

    // Re-bind the locals the debug-vis overlay reads before the include.
    let N = surface.normal;
    let ao_factor = surface.ao_factor;
    let last_shadow_sample = lit.last_shadow_sample;
    let dbg_direct_lum   = lit.dbg_direct_lum;
    let dbg_ambient_lum  = lit.dbg_ambient_lum;
    let dbg_ibl_diff_lum = lit.dbg_ibl_diff_lum;
    let dbg_ibl_spec_lum = lit.dbg_ibl_spec_lum;
    let dbg_roughness    = lit.dbg_roughness;
    let dbg_metallic     = lit.dbg_metallic;
    let dbg_emissive_lum = 0.0;
    var final_rgb = lit.rgb;

    // #include "debug_vis.wgsl"

    // McGuire & Bavoil weighted blended OIT output.
    let alpha = surface.alpha;
    let z = in.clip_pos.z;
    let w = alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0))));

    var out: OitOut;
    out.accum  = vec4<f32>(final_rgb * alpha * w, alpha * w);
    out.reveal = alpha;
    return out;
}
