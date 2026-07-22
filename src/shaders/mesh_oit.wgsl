// OIT (order-independent transparency) mesh shader : McGuire & Bavoil weighted blended.
//
// Identical to mesh.wgsl except for the fragment output: instead of writing a
// single RGBA colour to the HDR target, this shader writes to two targets:
//   @location(0) accum  : Rgba16Float accumulation buffer
//   @location(1) reveal : R8Unorm   reveal (transmittance) buffer
//
// The weighted-blended OIT formula is applied after computing the fully-lit
// colour (same Blinn-Phong / Cook-Torrance path as mesh.wgsl).
//
// Group 0: Camera uniform, shadow atlas, lights, clip planes, shadow info (unchanged).
// Group 1: Object uniform, albedo texture, sampler, normal map, AO map, LUT, scalar buffer.

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

struct Object {
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
    has_attribute: u32,
    scalar_min: f32,
    scalar_max: f32,
    receive_shadows: u32,
    nan_colour: vec4<f32>,                  // offset 144
    use_nan_colour: u32,                    // offset 160
    use_matcap: u32,                       // offset 164
    matcap_blendable: u32,                 // offset 168
    unlit: u32,                            // offset 172
    use_face_colour: u32,                   // offset 176
    uv_vis_mode: u32,                      // offset 180 : 0=off 1=checker 2=grid 3=localcheck 4=localrad
    uv_vis_scale: f32,                     // offset 184 : tile frequency multiplier
    backface_policy: u32,                  // offset 188 : 0=Cull 1=Identical 2=DiffColour 3=Tint 4..7=Pattern
    backface_colour: vec4<f32>,             // offset 192
    has_warp: u32,                         // offset 208
    warp_scale: f32,                       // offset 212
    _pad_warp0: u32,                       // offset 216
    _pad_warp1: u32,                       // offset 220
    emissive: vec3<f32>,                   // offset 224
    use_flat: u32,                         // offset 236 : 1 = recover N from screen-space derivatives of world_pos
    alpha_mode: u32,                       // offset 240 : 0=Opaque 1=Mask 2=Blend
    alpha_cutoff: f32,                     // offset 244
    has_metallic_roughness_tex: u32,       // offset 248
    has_emissive_tex: u32,                 // offset 252
    uv_transform: vec4<f32>,               // offset 256 : (offset.xy, scale.xy)
    deform_flags: u32,                     // offset 272 : bit i set when deformer slot i is active for this draw
    normal_strength: f32,                  // offset 276 : scales tangent normal XY (also aligns next vec2)
    ao_range: vec2<f32>,                   // offset 280 : (min, max) remap of AO map R sample
    metallic_range: vec2<f32>,             // offset 288 : (min, max) remap of MR texture B channel
    roughness_range: vec2<f32>,            // offset 296 : (min, max) remap of MR texture G channel
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
@group(1) @binding(0) var<uniform> object: Object;
@group(1) @binding(1) var obj_texture: texture_2d<f32>;
@group(1) @binding(2) var obj_sampler: sampler;
@group(1) @binding(3) var normal_map: texture_2d<f32>;
@group(1) @binding(4) var ao_map: texture_2d<f32>;
@group(1) @binding(5) var lut_texture: texture_2d<f32>;
@group(1) @binding(6) var<storage, read> scalar_buffer: array<f32>;
@group(1) @binding(8) var<storage, read> face_colour_buffer: array<vec4<f32>>;
@group(1) @binding(9) var<storage, read> warp_buffer: array<f32>;
@group(1) @binding(10) var lut_sampler: sampler;
@group(1) @binding(11) var metallic_roughness_tex: texture_2d<f32>;
@group(1) @binding(12) var emissive_tex: texture_2d<f32>;

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
    @location(5) scalar_val:     f32,
    @location(6) is_nan_scalar:  f32,
    @location(7) face_colour:     vec4<f32>,
};

struct OitOut {
    @location(0) accum:  vec4<f32>,
    @location(1) reveal: f32,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    var local_pos = in.position;
    if object.has_warp != 0u {
        let wi = in.vertex_index * 3u;
        let warp_len = arrayLength(&warp_buffer);
        if wi + 2u < warp_len {
            local_pos += vec3<f32>(warp_buffer[wi], warp_buffer[wi + 1u], warp_buffer[wi + 2u]) * object.warp_scale;
        }
    }
    var dv = DeformVertex(local_pos, in.normal, in.vertex_index);
    let dctx = DeformContext(object.model, object.model[3].xyz, 0.0, object.deform_flags, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let model3 = mat3x3<f32>(
        object.model[0].xyz,
        object.model[1].xyz,
        object.model[2].xyz,
    );
    let world_pos4 = object.model * vec4<f32>(dv.position, 1.0);
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
    let buf_len = arrayLength(&scalar_buffer);
    let idx = in.vertex_index;
    let has_attr = object.has_attribute != 0u && buf_len > 0u;
    let safe_idx = min(idx, select(0u, buf_len - 1u, buf_len > 0u));
    let raw_scalar = scalar_buffer[safe_idx];
    out.scalar_val = select(0.0, raw_scalar, has_attr);
    let sv_bits = bitcast<u32>(raw_scalar);
    let sv_is_nan = has_attr && (sv_bits & 0x7F800000u) == 0x7F800000u && (sv_bits & 0x007FFFFFu) != 0u;
    out.is_nan_scalar = select(0.0, 1.0, sv_is_nan);
    let fc_len = arrayLength(&face_colour_buffer);
    let fc_idx = min(idx, select(0u, fc_len - 1u, fc_len > 0u));
    out.face_colour = select(
        vec4<f32>(1.0),
        face_colour_buffer[fc_idx],
        object.use_face_colour != 0u && fc_len > 0u,
    );
    return out;
}

// ---------------------------------------------------------------------------
// 32-sample Poisson disk (shadow sampling : identical to mesh.wgsl)
// ---------------------------------------------------------------------------


// #include "csm.wgsl"

// ---------------------------------------------------------------------------
// PBR BRDF helpers (Cook-Torrance) : identical to mesh.wgsl
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
// Keep in sync with: mesh.wgsl, mesh_instanced.wgsl, mesh_instanced_oit.wgsl
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
    return textureSampleLevel(ibl_prefiltered, ibl_sampler, dir_to_equirect_uv(R, rotation), roughness * 4.0).rgb;
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

// UV parameterization visualization : procedural RGB colour from UV coordinates.
// Matches the implementation in mesh.wgsl exactly.
fn param_vis_colour(uv: vec2<f32>, mode: u32, scale: f32) -> vec3<f32> {
    let col_a      = vec3<f32>(0.85, 0.85, 0.85);
    let col_b      = vec3<f32>(0.2,  0.2,  0.2);
    let line_col   = vec3<f32>(0.1,  0.1,  0.1);
    let bg_col     = vec3<f32>(0.85, 0.85, 0.85);
    let line_width = 0.05f;
    let su = uv.x * scale;
    let sv = uv.y * scale;
    if mode == 1u {
        let p = (i32(floor(su)) + i32(floor(sv))) & 1;
        return select(col_a, col_b, p != 0);
    } else if mode == 2u {
        let on_line = fract(su) < line_width || fract(sv) < line_width;
        return select(bg_col, line_col, on_line);
    } else if mode == 3u {
        let d      = uv - vec2<f32>(0.5);
        let r      = length(d) * scale * 2.0;
        let theta  = atan2(d.y, d.x);
        let ring   = i32(floor(r)) & 1;
        let sector = i32(floor(theta * 4.0 / 3.14159265 + 8.0)) & 1;
        return select(col_a, col_b, (ring ^ sector) != 0);
    } else {
        let r = length(uv - vec2<f32>(0.5)) * scale * 2.0;
        return select(col_a, col_b, (i32(floor(r)) & 1) != 0);
    }
}

// ---------------------------------------------------------------------------
// OIT fragment shader : writes to accum + reveal targets.
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
// flow, so hook bodies can textureSampleGrad.
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
    // Keep the geometric normal in the same hemisphere as the shading normal,
    // which compute_surface has already flipped per the backface policy.
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

// Material prep for the transparent (OIT) path. Like the opaque shader minus
// wireframe and matcap; the shading models that fully determine the colour emit
// their own weighted-blend OitOut and set `resolved`, otherwise the surface
// fields feed compute_lit.
fn compute_surface(in: VertexOut, is_front: bool) -> Surface {
    var out: Surface;
    out.resolved = false;
    out.out_oit.accum = vec4<f32>(0.0);
    out.out_oit.reveal = 0.0;
    out.base_colour = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.ao_factor = 1.0;
    out.mat_uv = in.uv;
    out.alpha = 1.0;
    out.front_facing = select(0u, 1u, is_front);

    // Section view clipping.
    for (var i = 0u; i < clip_planes.count; i++) {
        let plane = clip_planes.planes[i];
        if dot(in.world_pos, plane.xyz) + plane.w < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    // Per-material UV transform: atlas region / tiling selection. Identity
    // (offset 0,0 scale 1,1) passes the authored UV through unchanged.
    let mat_uv = in.uv * object.uv_transform.zw + object.uv_transform.xy;
    out.mat_uv = mat_uv;

    // Sample texture if one is assigned.
    var tex_colour = vec4<f32>(1.0);
    if object.has_texture == 1u {
        tex_colour = textureSample(obj_texture, obj_sampler, mat_uv);
    }
    let obj_colour = vec4<f32>(
        object.colour.rgb * in.colour.rgb * tex_colour.rgb,
        object.colour.a   * in.colour.a   * tex_colour.a,
    );
    out.alpha = obj_colour.a;

    // Alpha MASK: discard fragments whose alpha is below the cutoff.
    if object.alpha_mode == 1u && obj_colour.a < object.alpha_cutoff {
        discard;
    }

    var base_colour = obj_colour.rgb;

    // Per-face RGBA colour: use directly, bypassing all lighting and colourmap logic.
    if object.use_face_colour != 0u {
        var fc = in.face_colour;
        if object.selected != 0u {
            fc = mix(fc, vec4<f32>(1.0, 0.55, 0.1, 1.0), 0.35);
        }
        let alpha = fc.a * object.colour.a;
        let w = alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + pow(abs(in.clip_pos.z / in.clip_pos.w), 4.0))));
        out.resolved = true;
        out.out_oit.accum  = vec4<f32>(fc.rgb * alpha, alpha) * w;
        out.out_oit.reveal = alpha;
        return out;
    }

    // Scalar attribute colour override.
    if object.has_attribute != 0u {
        if in.is_nan_scalar > 0.5 {
            if object.use_nan_colour == 0u {
                discard;
            }
            let alpha = object.nan_colour.a;
            let z = in.clip_pos.z;
            let w = alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0))));
            out.resolved = true;
            out.out_oit.accum  = vec4<f32>(object.nan_colour.rgb * alpha * w, alpha * w);
            out.out_oit.reveal = alpha;
            return out;
        }
        let raw = in.scalar_val;
        let range = object.scalar_max - object.scalar_min;
        let t = clamp(
            select(0.0, (raw - object.scalar_min) / range, range > 0.0001),
            0.0, 1.0,
        );
        base_colour = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(t, 0.5), 0.0).rgb;
    }

    // Unlit: skip all lighting, return raw colour directly through OIT.
    if object.unlit != 0u {
        let alpha = obj_colour.a;
        let w = alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + pow(abs(in.clip_pos.z / in.clip_pos.w), 4.0))));
        out.resolved = true;
        out.out_oit.accum  = vec4<f32>(base_colour * alpha, alpha) * w;
        out.out_oit.reveal = alpha;
        return out;
    }

    // UV parameterization visualization: procedural pattern replaces all lighting.
    if object.uv_vis_mode != 0u {
        let vis   = param_vis_colour(in.uv, object.uv_vis_mode, object.uv_vis_scale);
        let alpha = obj_colour.a;
        let w = alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + pow(abs(in.clip_pos.z / in.clip_pos.w), 4.0))));
        out.resolved = true;
        out.out_oit.accum  = vec4<f32>(vis * alpha, alpha) * w;
        out.out_oit.reveal = alpha;
        return out;
    }

    // Shading normal. `use_flat` recovers a per-fragment geometric normal
    // from screen-space derivatives of world position and takes precedence
    // over the normal-map path.
    var N: vec3<f32>;
    if object.use_flat != 0u {
        let dpx = dpdx(in.world_pos);
        let dpy = dpdy(in.world_pos);
        var Nf = normalize(cross(dpx, dpy));
        if dot(Nf, in.world_normal) < 0.0 { Nf = -Nf; }
        N = Nf;
    } else if object.has_normal_map != 0u {
        let nm_sample = textureSample(normal_map, obj_sampler, mat_uv).rgb;
        var ts_unpacked = nm_sample * 2.0 - vec3<f32>(1.0);
        ts_unpacked.x = ts_unpacked.x * object.normal_strength;
        ts_unpacked.y = ts_unpacked.y * object.normal_strength;
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

    // Back-face policy handling: flip normal and optionally override colour for back faces.
    // 0=Cull, 1=Identical, 2=DifferentColour, 3=Tint, 4=Checker, 5=Hatching, 6=Crosshatch, 7=Stripes.
    if !is_front && object.backface_policy >= 2u {
        N = -N;
        if object.backface_policy == 2u {
            base_colour = object.backface_colour.rgb;
        } else if object.backface_policy == 3u {
            base_colour = base_colour * (1.0 - object.backface_colour.r);
        } else {
            let pattern_colour = object.backface_colour.rgb;
            let pattern_type = object.backface_policy - 4u;
            let wp = in.world_pos * object.backface_colour.w;
            var use_pattern = false;
            if pattern_type == 0u {
                // Checker: alternating squares in world XZ.
                let p = (i32(floor(wp.x)) + i32(floor(wp.z))) & 1;
                use_pattern = p != 0;
            } else if pattern_type == 1u {
                // Hatching: diagonal lines at 45 degrees.
                use_pattern = fract((wp.x + wp.z) * 0.5) < 0.4;
            } else if pattern_type == 2u {
                // Crosshatch: two sets of diagonal lines.
                use_pattern = fract((wp.x + wp.z) * 0.5) < 0.3 || fract((wp.x - wp.z) * 0.5) < 0.3;
            } else {
                // Stripes: horizontal lines in world Z.
                use_pattern = fract(wp.z * 0.5) < 0.4;
            }
            base_colour = select(base_colour, pattern_colour, use_pattern);
        }
    }

    // AO factor from AO map. Per-material `ao_range` remaps the raw sample
    // before it drives shading; identity `(0, 1)` is a no-op.
    var ao_factor = 1.0;
    if object.has_ao_map != 0u {
        let raw_ao = textureSample(ao_map, obj_sampler, mat_uv).r;
        ao_factor = mix(object.ao_range.x, object.ao_range.y, raw_ao);
    }

    out.base_colour = base_colour;
    out.normal = N;
    out.ao_factor = ao_factor;
    return out;
}

// Lighting for the transparent path. Transparent surfaces skip shadow sampling,
// so last_shadow_sample stays at its unshadowed default.
fn compute_lit(surface: Surface, in: VertexOut) -> LitResult {
    let base_colour = surface.base_colour;
    let ao_factor = surface.ao_factor;
    let N = surface.normal;

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

    if object.use_pbr != 0u {
        var metallic  = clamp(object.metallic,  0.0, 1.0);
        var roughness = max(object.roughness, 0.04);
        if object.has_metallic_roughness_tex != 0u {
            // glTF ORM texture: G=roughness factor, B=metallic factor. Per-material
            // `metallic_range` / `roughness_range` remap the raw samples before the
            // scalar factor; identity `(0, 1)` is a no-op.
            let mr = textureSample(metallic_roughness_tex, obj_sampler, surface.mat_uv);
            let m_remapped = mix(object.metallic_range.x, object.metallic_range.y, mr.b);
            let r_remapped = mix(object.roughness_range.x, object.roughness_range.y, mr.g);
            metallic  = clamp(m_remapped * metallic,  0.0, 1.0);
            roughness = max(r_remapped * roughness, 0.04);
        }
        let F0 = mix(vec3<f32>(0.04), base_colour, metallic);
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
            // Transparent surfaces do not cast/receive shadows (no CSM sampling).
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
            let ambient_scale = vec3<f32>(object.ambient) + hemi_colour * lights_uniform.hemisphere_intensity;
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
            // Transparent surfaces do not participate in shadow evaluation.
            let H = normalize(ev.l + V);
            let n_dot_l = max(dot(N, ev.l), 0.0);
            let n_dot_h = max(dot(N, H), 0.0);
            let diffuse_contrib  = object.diffuse  * n_dot_l;
            let specular_contrib = object.specular * pow(n_dot_h, object.shininess);
            total_colour_contrib += (diffuse_contrib + specular_contrib) * ev.radiance;
        }
        let ambient_contrib = object.ambient;
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
fn fs_oit_main(in: VertexOut, @builtin(front_facing) is_front: bool) -> OitOut {
    let surface = compute_surface(in, is_front);
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
    let lum_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    var final_rgb = lit.rgb;

    // Emissive term: added after lighting so it can push HDR values above 1.0.
    var emissive = object.emissive;
    if object.has_emissive_tex != 0u {
        emissive = emissive * textureSample(emissive_tex, obj_sampler, surface.mat_uv).rgb;
    }
    final_rgb += emissive;
    var dbg_emissive_lum = dot(emissive, lum_weights);

    // #include "debug_vis.wgsl"

    // ---------------------------------------------------------------------------
    // McGuire & Bavoil weighted blended OIT output.
    // ---------------------------------------------------------------------------
    let alpha = surface.alpha;
    let z = in.clip_pos.z;  // NDC depth 0..1
    let w = alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0))));

    var out: OitOut;
    out.accum  = vec4<f32>(final_rgb * alpha * w, alpha * w);
    out.reveal = alpha;
    return out;
}
