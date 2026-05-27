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

struct SingleLight {
    light_view_proj: mat4x4<f32>,
    pos_or_dir: vec3<f32>,
    light_type: u32,
    colour: vec3<f32>,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    spot_direction: vec3<f32>,
    _pad: vec2<f32>,
};

struct Lights {
    count: u32,
    shadow_bias: f32,
    shadows_enabled: u32,
    debug_vis_mode: u32,
    sky_colour: vec3<f32>,
    hemisphere_intensity: f32,
    ground_colour: vec3<f32>,
    debug_vis_scale: f32,
    lights: array<SingleLight, 8>,
    ibl_enabled: u32,
    ibl_intensity: f32,
    ibl_rotation: f32,
    show_skybox: u32,
};

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
    _pad_inst0: u32,
    _pad_inst1: u32,
    _pad_inst2: u32,
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

fn clip_volume_test(p: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_volume.count; i = i + 1u) {
        let e = clip_volume.volumes[i];
        if e.volume_type == 2u {
            let d = p - e.center;
            let local = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
            if abs(local.x) > e.half_extents.x
                || abs(local.y) > e.half_extents.y
                || abs(local.z) > e.half_extents.z {
                return false;
            }
        } else if e.volume_type == 3u {
            let ds = p - e.center;
            if dot(ds, ds) > e.radius * e.radius { return false; }
        } else if e.volume_type == 4u {
            let axis = e.col0;
            let d = p - e.center;
            let along = dot(d, axis);
            if abs(along) > e.half_extents.x { return false; }
            let radial = d - axis * along;
            if dot(radial, radial) > e.radius * e.radius { return false; }
        }
    }
    return true;
}
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
    let world_pos = inst.model * vec4<f32>(in.position, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.colour = in.colour;
    out.world_pos = world_pos.xyz;
    let model3 = mat3x3<f32>(
        inst.model[0].xyz,
        inst.model[1].xyz,
        inst.model[2].xyz,
    );
    out.world_normal = normalize(model3 * in.normal);
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
    let world_pos = inst.model * vec4<f32>(in.position, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.colour = in.colour;
    out.world_pos = world_pos.xyz;
    let model3 = mat3x3<f32>(
        inst.model[0].xyz,
        inst.model[1].xyz,
        inst.model[2].xyz,
    );
    out.world_normal = normalize(model3 * in.normal);
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
    let d = vec3<f32>(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
    return vec2<f32>(0.5 + atan2(d.z, d.x) / (2.0 * IBL_PI), 0.5 - asin(clamp(d.y, -1.0, 1.0)) / IBL_PI);
}
fn sample_ibl_irradiance(N: vec3<f32>, rotation: f32) -> vec3<f32> {
    return textureSampleLevel(ibl_irradiance, ibl_sampler, dir_to_equirect_uv(N, rotation), 0.0).rgb;
}
fn sample_ibl_prefiltered(R: vec3<f32>, roughness: f32, rotation: f32) -> vec3<f32> {
    return textureSampleLevel(ibl_prefiltered, ibl_sampler, dir_to_equirect_uv(R, rotation), roughness * 4.0).rgb;
}
fn sample_brdf_lut(NdotV: f32, roughness: f32) -> vec2<f32> {
    return textureSampleLevel(ibl_brdf_lut, ibl_sampler, vec2<f32>(NdotV, roughness), 0.0).rg;
}
fn F_Schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}
fn ibl_ambient(N: vec3<f32>, V: vec3<f32>, base_colour: vec3<f32>, metallic: f32,
               roughness: f32, F0: vec3<f32>, ao: f32, intensity: f32, rotation: f32) -> vec3<f32> {
    let NdotV = max(dot(N, V), 0.001);
    let F = F_Schlick_roughness(NdotV, F0, roughness);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let irradiance = sample_ibl_irradiance(N, rotation);
    let R = reflect(-V, N);
    let prefiltered = sample_ibl_prefiltered(R, roughness, rotation);
    let brdf = sample_brdf_lut(NdotV, roughness);
    return (kD * irradiance * base_colour + prefiltered * (F * brdf.x + brdf.y)) * ao * intensity;
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
@fragment
fn fs_oit_main(in: VertexOut) -> OitOut {
    let inst = instances[in.instance_idx];

    for (var i = 0u; i < clip_planes.count; i++) {
        let plane = clip_planes.planes[i];
        if dot(in.world_pos, plane.xyz) + plane.w < 0.0 { discard; }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    var tex_colour = vec4<f32>(1.0);
    if inst.has_texture == 1u { tex_colour = textureSample(obj_texture, obj_sampler, in.uv); }
    let obj_colour = vec4<f32>(
        inst.colour.rgb * in.colour.rgb * tex_colour.rgb,
        inst.colour.a   * in.colour.a   * tex_colour.a,
    );
    let base_colour = obj_colour.rgb;

    // Unlit: skip all lighting, return raw colour directly through OIT.
    if inst.unlit != 0u {
        let alpha = obj_colour.a;
        let w = alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + pow(abs(in.clip_pos.z / in.clip_pos.w), 4.0))));
        var oit_out: OitOut;
        oit_out.accum  = vec4<f32>(base_colour * alpha, alpha) * w;
        oit_out.reveal = alpha;
        return oit_out;
    }

    var N: vec3<f32>;
    if inst.has_normal_map != 0u {
        let nm_sample = textureSample(normal_map, obj_sampler, in.uv).rgb;
        let ts_normal = normalize(nm_sample * 2.0 - vec3<f32>(1.0));
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
    if inst.has_ao_map != 0u { ao_factor = textureSample(ao_map, obj_sampler, in.uv).r; }

    let V = normalize(camera.eye_pos - in.world_pos);
    let tint = vec4<f32>(1.0);
    var last_shadow_sample = ShadowSample(1.0, 0u, vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0, 0.0);
    var final_rgb: vec3<f32>;

    if inst.use_pbr != 0u {
        let metallic  = clamp(inst.metallic,  0.0, 1.0);
        let roughness = max(inst.roughness, 0.04);
        let F0 = mix(vec3<f32>(0.04), base_colour, metallic);
        var Lo = vec3<f32>(0.0);
        for (var i = 0u; i < lights_uniform.count; i++) {
            let l = lights_uniform.lights[i];
            var L: vec3<f32>; var radiance: vec3<f32>;
            if l.light_type == 0u {
                L = normalize(l.pos_or_dir); radiance = l.colour * l.intensity;
            } else if l.light_type == 1u {
                let to_light = l.pos_or_dir - in.world_pos; let dist = length(to_light);
                L = to_light / max(dist, 0.0001);
                let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                radiance = l.colour * l.intensity * falloff * falloff;
            } else {
                let to_light = l.pos_or_dir - in.world_pos; let dist = length(to_light);
                L = to_light / max(dist, 0.0001);
                let dist_falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                let spot_dir = normalize(l.spot_direction);
                let cos_angle = dot(-L, spot_dir);
                let cos_outer = cos(l.outer_angle); let cos_inner = cos(l.inner_angle);
                let cone_att = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
                radiance = l.colour * l.intensity * dist_falloff * dist_falloff * cone_att;
            }
            // Transparent surfaces: skip shadow map sampling.
            Lo += pbr_light_contrib(N, V, L, radiance, base_colour, metallic, roughness, F0);
        }
        var ambient: vec3<f32>;
        if lights_uniform.ibl_enabled != 0u {
            ambient = ibl_ambient(N, V, base_colour, metallic, roughness, F0,
                                  ao_factor, lights_uniform.ibl_intensity,
                                  lights_uniform.ibl_rotation);
        } else {
            let hemi_t = clamp(in.world_normal.y * 0.5 + 0.5, 0.0, 1.0);
            let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
            let ambient_scale = vec3<f32>(inst.ambient) + hemi_colour * lights_uniform.hemisphere_intensity;
            ambient = ambient_scale * (base_colour * (1.0 - metallic) + F0 * metallic) * ao_factor;
        }
        final_rgb = clamp((Lo + ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        var total_colour_contrib = vec3<f32>(0.0);
        for (var i = 0u; i < lights_uniform.count; i++) {
            let l = lights_uniform.lights[i];
            var light_dir: vec3<f32>; var attenuation = 1.0;
            if l.light_type == 0u {
                light_dir = normalize(l.pos_or_dir);
            } else if l.light_type == 1u {
                let to_light = l.pos_or_dir - in.world_pos; let dist = length(to_light);
                light_dir = to_light / max(dist, 0.0001);
                let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                attenuation = falloff * falloff;
            } else {
                let to_light = l.pos_or_dir - in.world_pos; let dist = length(to_light);
                light_dir = to_light / max(dist, 0.0001);
                let dist_falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                let spot_dir = normalize(l.spot_direction);
                let cos_angle = dot(-light_dir, spot_dir);
                let cos_outer = cos(l.outer_angle); let cos_inner = cos(l.inner_angle);
                let cone_att = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
                attenuation = dist_falloff * dist_falloff * cone_att;
            }
            // Transparent surfaces: skip shadow map sampling.
            let H = normalize(light_dir + V);
            let n_dot_l = max(dot(N, light_dir), 0.0);
            let n_dot_h = max(dot(N, H), 0.0);
            let diffuse_contrib  = inst.diffuse  * n_dot_l * l.intensity * attenuation;
            let specular_contrib = inst.specular * pow(n_dot_h, inst.shininess)
                                 * l.intensity * attenuation;
            total_colour_contrib += (diffuse_contrib + specular_contrib) * l.colour;
        }
        let ambient_contrib = inst.ambient;
        let hemi_t = clamp(in.world_normal.y * 0.5 + 0.5, 0.0, 1.0);
        let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
        let hemi_ambient = hemi_colour * lights_uniform.hemisphere_intensity;
        let lit_rgb = base_colour * (ambient_contrib + hemi_ambient) * ao_factor
                    + base_colour * total_colour_contrib;
        final_rgb = clamp(lit_rgb * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // #include "debug_vis.wgsl"

    // McGuire & Bavoil weighted blended OIT output.
    let alpha = obj_colour.a;
    let z = in.clip_pos.z;
    let w = alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0))));

    var out: OitOut;
    out.accum  = vec4<f32>(final_rgb * alpha * w, alpha * w);
    out.reveal = alpha;
    return out;
}
