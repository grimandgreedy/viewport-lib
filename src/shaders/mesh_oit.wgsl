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
    _pad: u32,
    sky_colour: vec3<f32>,
    hemisphere_intensity: f32,
    ground_colour: vec3<f32>,
    _pad2: f32,
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
    _pad_scalar: u32,
    nan_colour: vec4<f32>,    // offset 144
    use_nan_colour: u32,      // offset 160
    use_matcap: u32,         // offset 164
    matcap_blendable: u32,   // offset 168
    unlit: u32,              // offset 172
    use_face_colour: u32,     // offset 176
    uv_vis_mode: u32,           // offset 180 : 0=off 1=checker 2=grid 3=localcheck 4=localrad
    uv_vis_scale: f32,          // offset 184 : tile frequency multiplier
    backface_policy: u32,       // offset 188 : 0=Cull 1=Identical 2=DiffColour 3=Tint 4..7=Pattern
    backface_colour: vec4<f32>,  // offset 192
    has_warp: u32,              // offset 208
    warp_scale: f32,            // offset 212
    _pad_warp0: u32,            // offset 216
    _pad_warp1: u32,            // offset 220
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
    let world_pos = object.model * vec4<f32>(local_pos, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.colour = in.colour;
    out.world_pos = world_pos.xyz;
    let model3 = mat3x3<f32>(
        object.model[0].xyz,
        object.model[1].xyz,
        object.model[2].xyz,
    );
    out.world_normal = normalize(model3 * in.normal);
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
const POISSON_DISK: array<vec2<f32>, 32> = array<vec2<f32>, 32>(
    vec2<f32>(-0.94201624, -0.39906216), vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870), vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432), vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845), vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554), vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023), vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507), vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367), vec2<f32>( 0.14383161, -0.14100790),
    vec2<f32>(-0.44451570,  0.67055830), vec2<f32>( 0.70509040, -0.15854630),
    vec2<f32>( 0.07130650, -0.64599580), vec2<f32>( 0.39881030,  0.55789810),
    vec2<f32>(-0.60554040, -0.34964830), vec2<f32>( 0.85095100,  0.47178830),
    vec2<f32>(-0.47994860,  0.08443340), vec2<f32>(-0.12494190, -0.76098760),
    vec2<f32>( 0.64839320,  0.74738240), vec2<f32>(-0.96815740, -0.12345680),
    vec2<f32>( 0.27682050, -0.80927180), vec2<f32>(-0.73016460,  0.18344200),
    vec2<f32>( 0.54754660,  0.06234570), vec2<f32>(-0.30967360, -0.61021430),
    vec2<f32>(-0.57774330,  0.80459740), vec2<f32>( 0.18238670, -0.37596540),
);

fn sample_shadow_csm(
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    surface_normal: vec3<f32>,
    light_dir: vec3<f32>,
) -> f32 {
    let dist = dot(world_pos - eye_pos, camera.forward);
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
    let texel_size = 1.0 / shadow_atlas.atlas_size;
    let n_dot_l = dot(surface_normal, light_dir);
    let offset_sign = select(-1.0, 1.0, n_dot_l >= 0.0);
    let texel_world = 2.0 / (shadow_atlas.cascade_vp[cascade_idx][0][0] * shadow_atlas.atlas_size * (rect.z - rect.x));
    let normal_bias = texel_world * mix(1.5, 0.5, clamp(abs(n_dot_l), 0.0, 1.0));
    let offset_world = world_pos + surface_normal * (offset_sign * normal_bias);
    let offset_clip = shadow_atlas.cascade_vp[cascade_idx] * vec4<f32>(offset_world, 1.0);
    let biased_depth = (offset_clip.xyz / offset_clip.w).z - lights_uniform.shadow_bias;
    let noise = fract(52.9829189 * fract(dot(world_pos.xz, vec2<f32>(0.06711056, 0.00583715))));
    let rot = noise * 6.28318530;
    let sin_r = sin(rot);
    let cos_r = cos(rot);
    if shadow_atlas.shadow_filter == 1u {
        let search_radius = shadow_atlas.pcss_light_radius * 16.0 * texel_size;
        var blocker_sum = 0.0;
        var blocker_count = 0.0;
        for (var i = 0u; i < 16u; i++) {
            let d = POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * search_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let sample_depth = textureSampleCompare(shadow_map, shadow_sampler, clamped_uv, biased_depth);
            if sample_depth < 1.0 {
                let coords = vec2<i32>(clamped_uv * shadow_atlas.atlas_size);
                let raw_depth = textureLoad(shadow_map, coords, 0);
                blocker_sum += raw_depth;
                blocker_count += 1.0;
            }
        }
        if blocker_count < 1.0 { return 1.0; }
        let avg_blocker = blocker_sum / blocker_count;
        let penumbra_width = shadow_atlas.pcss_light_radius * (biased_depth - avg_blocker) / max(avg_blocker, 0.001);
        let filter_radius = max(penumbra_width * 16.0 * texel_size, texel_size);
        var shadow = 0.0;
        for (var i = 0u; i < 32u; i++) {
            let d = POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * filter_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            shadow += textureSampleCompare(shadow_map, shadow_sampler, clamped_uv, biased_depth);
        }
        return shadow / 32.0;
    } else {
        let pcf_radius = 4.0 * texel_size;
        var shadow = 0.0;
        for (var i = 0u; i < 32u; i++) {
            let d = POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * pcf_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            shadow += textureSampleCompare(shadow_map, shadow_sampler, clamped_uv, biased_depth);
        }
        return shadow / 32.0;
    }
}

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
@fragment
fn fs_oit_main(in: VertexOut, @builtin(front_facing) is_front: bool) -> OitOut {
    // Section view clipping.
    for (var i = 0u; i < clip_planes.count; i++) {
        let plane = clip_planes.planes[i];
        if dot(in.world_pos, plane.xyz) + plane.w < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    // Sample texture if one is assigned.
    var tex_colour = vec4<f32>(1.0);
    if object.has_texture == 1u {
        tex_colour = textureSample(obj_texture, obj_sampler, in.uv);
    }
    let obj_colour = vec4<f32>(
        object.colour.rgb * in.colour.rgb * tex_colour.rgb,
        object.colour.a   * in.colour.a   * tex_colour.a,
    );
    var base_colour = obj_colour.rgb;

    // Per-face RGBA colour: use directly, bypassing all lighting and colourmap logic.
    if object.use_face_colour != 0u {
        var fc = in.face_colour;
        if object.selected != 0u {
            fc = mix(fc, vec4<f32>(1.0, 0.55, 0.1, 1.0), 0.35);
        }
        let alpha = fc.a * object.colour.a;
        let w = alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + pow(abs(in.clip_pos.z / in.clip_pos.w), 4.0))));
        var oit_out: OitOut;
        oit_out.accum  = vec4<f32>(fc.rgb * alpha, alpha) * w;
        oit_out.reveal = alpha;
        return oit_out;
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
            var nan_out: OitOut;
            nan_out.accum  = vec4<f32>(object.nan_colour.rgb * alpha * w, alpha * w);
            nan_out.reveal = alpha;
            return nan_out;
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
        var oit_out: OitOut;
        oit_out.accum  = vec4<f32>(base_colour * alpha, alpha) * w;
        oit_out.reveal = alpha;
        return oit_out;
    }

    // UV parameterization visualization: procedural pattern replaces all lighting.
    if object.uv_vis_mode != 0u {
        let vis   = param_vis_colour(in.uv, object.uv_vis_mode, object.uv_vis_scale);
        let alpha = obj_colour.a;
        let w = alpha * max(1e-2, min(3e3, 0.03 / (1e-5 + pow(abs(in.clip_pos.z / in.clip_pos.w), 4.0))));
        var oit_out: OitOut;
        oit_out.accum  = vec4<f32>(vis * alpha, alpha) * w;
        oit_out.reveal = alpha;
        return oit_out;
    }

    // Shading normal.
    var N: vec3<f32>;
    if object.has_normal_map != 0u {
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

    var ao_factor = 1.0;
    if object.has_ao_map != 0u {
        ao_factor = textureSample(ao_map, obj_sampler, in.uv).r;
    }

    let V = normalize(camera.eye_pos - in.world_pos);
    let tint = vec4<f32>(1.0);

    var final_rgb: vec3<f32>;

    if object.use_pbr != 0u {
        let metallic  = clamp(object.metallic,  0.0, 1.0);
        let roughness = max(object.roughness, 0.04);
        let F0 = mix(vec3<f32>(0.04), base_colour, metallic);
        var Lo = vec3<f32>(0.0);
        for (var i = 0u; i < lights_uniform.count; i++) {
            let l = lights_uniform.lights[i];
            var L: vec3<f32>;
            var radiance: vec3<f32>;
            if l.light_type == 0u {
                L = normalize(l.pos_or_dir);
                radiance = l.colour * l.intensity;
            } else if l.light_type == 1u {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                L = to_light / max(dist, 0.0001);
                let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                radiance = l.colour * l.intensity * falloff * falloff;
            } else {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                L = to_light / max(dist, 0.0001);
                let dist_falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                let spot_dir = normalize(l.spot_direction);
                let cos_angle = dot(-L, spot_dir);
                let cos_outer = cos(l.outer_angle);
                let cos_inner = cos(l.inner_angle);
                let cone_att = clamp(
                    (cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
                    0.0, 1.0,
                );
                radiance = l.colour * l.intensity * dist_falloff * dist_falloff * cone_att;
            }
            // Transparent surfaces do not cast/receive shadows (no CSM sampling).
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
            let ambient_scale = vec3<f32>(object.ambient) + hemi_colour * lights_uniform.hemisphere_intensity;
            ambient = ambient_scale * (base_colour * (1.0 - metallic) + F0 * metallic) * ao_factor;
        }
        final_rgb = clamp((Lo + ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        var total_colour_contrib = vec3<f32>(0.0);
        for (var i = 0u; i < lights_uniform.count; i++) {
            let l = lights_uniform.lights[i];
            var light_dir: vec3<f32>;
            var attenuation = 1.0;
            if l.light_type == 0u {
                light_dir = normalize(l.pos_or_dir);
            } else if l.light_type == 1u {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                light_dir = to_light / max(dist, 0.0001);
                let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                attenuation = falloff * falloff;
            } else {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                light_dir = to_light / max(dist, 0.0001);
                let dist_falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                let spot_dir = normalize(l.spot_direction);
                let cos_angle = dot(-light_dir, spot_dir);
                let cos_outer = cos(l.outer_angle);
                let cos_inner = cos(l.inner_angle);
                let cone_att = clamp(
                    (cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
                    0.0, 1.0,
                );
                attenuation = dist_falloff * dist_falloff * cone_att;
            }
            // Transparent surfaces do not participate in shadow evaluation.
            let H = normalize(light_dir + V);
            let n_dot_l = max(dot(N, light_dir), 0.0);
            let n_dot_h = max(dot(N, H), 0.0);
            let diffuse_contrib  = object.diffuse  * n_dot_l * l.intensity * attenuation;
            let specular_contrib = object.specular * pow(n_dot_h, object.shininess)
                                 * l.intensity * attenuation;
            total_colour_contrib += (diffuse_contrib + specular_contrib) * l.colour;
        }
        let ambient_contrib = object.ambient;
        let hemi_t = clamp(in.world_normal.y * 0.5 + 0.5, 0.0, 1.0);
        let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
        let hemi_ambient = hemi_colour * lights_uniform.hemisphere_intensity;
        let lit_rgb = base_colour * (ambient_contrib + hemi_ambient) * ao_factor
                    + base_colour * total_colour_contrib;
        final_rgb = clamp(lit_rgb * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // ---------------------------------------------------------------------------
    // McGuire & Bavoil weighted blended OIT output.
    // ---------------------------------------------------------------------------
    let alpha = obj_colour.a;
    let z = in.clip_pos.z;  // NDC depth 0..1
    let w = alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0))));

    var out: OitOut;
    out.accum  = vec4<f32>(final_rgb * alpha * w, alpha * w);
    out.reveal = alpha;
    return out;
}
