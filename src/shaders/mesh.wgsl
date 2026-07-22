// Mesh shader for the 3D viewport.
//
// Group 0: Camera uniform (view-projection, eye position)
//          + shadow atlas texture + comparison sampler
//          + Lights uniform (up to 8 light sources, shadow parameters)
//          + ClipPlanes uniform (up to 6 user-defined half-space clipping planes)
//          + ShadowAtlas uniform (CSM matrices, cascade splits, PCSS params).
// Group 1: Object uniform (per-object model matrix, material properties,
//          selection flag, wireframe flag, PBR params)
//          + Albedo texture (binding 1) + sampler (binding 2)
//          + normal map (binding 3) + AO map (binding 4)
//          + metallic-roughness texture (binding 11) + emissive texture (binding 12).
//
// Lighting: Blinn-Phong (ambient + diffuse + specular) with multi-light support.
//           Cook-Torrance PBR when object.use_pbr != 0.
// Shadow mapping: CSM with atlas-based cascade selection.
//   PCF (3x3) or PCSS (blocker search + variable-width PCF) via shadow_atlas.shadow_filter.
// Selection: orange tint when object.selected == 1u.
// Wireframe: gray colour override when object.wireframe == 1u.
// Section views: fragment discarded when world_pos fails any active clip plane.
// Normal maps: tangent-space normal mapping via TBN when object.has_normal_map != 0u.
// AO maps: ambient occlusion applied to ambient + diffuse when object.has_ao_map != 0u.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
    forward: vec3<f32>,
    _pad1: f32,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
};

// Shared light struct definitions and `lights_storage` binding 13 of group 0.
// #include "scene_lighting.wgsl"

// Per-vertex deformation hook contract.
// #include "deform.wgsl"

// Clip planes uniform : 112 bytes.
struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count: u32,
    _pad0: u32,
    viewport_width: f32,
    viewport_height: f32,
};

// Shadow atlas uniform : 416 bytes.
struct ShadowAtlas {
    cascade_vp: array<mat4x4<f32>, 4>,   // 256 bytes
    cascade_splits: vec4<f32>,            //  16 bytes
    cascade_count: u32,                   //   4 bytes
    atlas_size: f32,                      //   4 bytes
    shadow_filter: u32,                   //   4 bytes (0=PCF, 1=PCSS)
    pcss_light_radius: f32,               //   4 bytes
    atlas_rects: array<vec4<f32>, 8>,     // 128 bytes
};

// Per-object uniform : 256 bytes.
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
    has_position_override: u32,            // offset 216 : 1 when a per-vertex position storage buffer is bound at binding 13
    has_normal_override: u32,              // offset 220 : 1 when a per-vertex normal storage buffer is bound at binding 14
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
@group(1) @binding(7) var matcap_texture: texture_2d<f32>;
@group(1) @binding(8) var<storage, read> face_colour_buffer: array<vec4<f32>>;
@group(1) @binding(9) var<storage, read> warp_buffer: array<f32>;
@group(1) @binding(10) var lut_sampler: sampler;
@group(1) @binding(11) var metallic_roughness_tex: texture_2d<f32>;
@group(1) @binding(12) var emissive_tex: texture_2d<f32>;
// Optional per-vertex override storage buffers a `GpuPlugin` may bind via
// `DeviceResources::set_position_override_buffer` / `set_normal_override_buffer`.
// Flat `array<f32>` with 3 values per vertex so consumer compute shaders can
// write tight `vec3` data without WGSL's 16-byte vec3 stride padding (matches
// the warp_buffer convention). When unbound, a 12-byte zero sentinel is bound
// so the bind group layout is satisfied.
@group(1) @binding(13) var<storage, read> position_override_buffer: array<f32>;
@group(1) @binding(14) var<storage, read> normal_override_buffer:   array<f32>;

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
    // 1.0 if the source scalar vertex value was NaN, 0.0 otherwise.
    // Detected in vs_main before interpolation can corrupt the NaN bit pattern.
    @location(6) is_nan_scalar:  f32,
    @location(7) face_colour:     vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    // Override > vertex attribute. When a plugin has bound a per-vertex position
    // storage buffer, replace `in.position` outright; warp is then layered on top
    // additively. Same idea for normals further down.
    var local_pos = in.position;
    if object.has_position_override != 0u {
        let pi = in.vertex_index * 3u;
        let plen = arrayLength(&position_override_buffer);
        if pi + 2u < plen {
            local_pos = vec3<f32>(
                position_override_buffer[pi],
                position_override_buffer[pi + 1u],
                position_override_buffer[pi + 2u],
            );
        }
    }
    if object.has_warp != 0u {
        let wi = in.vertex_index * 3u;
        let warp_len = arrayLength(&warp_buffer);
        if wi + 2u < warp_len {
            local_pos += vec3<f32>(warp_buffer[wi], warp_buffer[wi + 1u], warp_buffer[wi + 2u]) * object.warp_scale;
        }
    }
    var local_normal = in.normal;
    if object.has_normal_override != 0u {
        let ni = in.vertex_index * 3u;
        let nlen = arrayLength(&normal_override_buffer);
        if ni + 2u < nlen {
            local_normal = vec3<f32>(
                normal_override_buffer[ni],
                normal_override_buffer[ni + 1u],
                normal_override_buffer[ni + 2u],
            );
        }
    }
    var dv = DeformVertex(local_pos, local_normal, in.vertex_index);
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
    // Read scalar attribute value for this vertex, guarded by has_attribute and buffer length.
    let buf_len = arrayLength(&scalar_buffer);
    let idx = in.vertex_index;
    let has_attr = object.has_attribute != 0u && buf_len > 0u;
    let safe_idx = min(idx, select(0u, buf_len - 1u, buf_len > 0u));
    let raw_scalar = scalar_buffer[safe_idx];
    out.scalar_val = select(0.0, raw_scalar, has_attr);
    // Detect NaN before interpolation can corrupt the bit pattern.
    let sv_bits = bitcast<u32>(raw_scalar);
    let sv_is_nan = has_attr && (sv_bits & 0x7F800000u) == 0x7F800000u && (sv_bits & 0x007FFFFFu) != 0u;
    out.is_nan_scalar = select(0.0, 1.0, sv_is_nan);
    // Per-face RGBA colour (FaceColour attribute kind). Indexed by vertex_index which
    // equals the sequential draw invocation counter for non-indexed face draws.
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
// 32-sample Poisson disk (first 16 used for blocker search, all 32 for filter)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CSM shadow sampling : selects cascade by eye distance, samples atlas tile
// ---------------------------------------------------------------------------

fn sample_point_shadow(light: SingleLight, world_pos: vec3<f32>) -> f32 {
    if light.point_shadow_slot < 0 {
        return 1.0;
    }
    let to_frag = world_pos - light.pos_or_dir;
    let dist = length(to_frag);
    let dir = to_frag / max(dist, 1e-5);
    let normalised = clamp(dist / max(light.range, 1e-5), 0.0, 1.0);
    let bias = 0.0015;
    return textureSampleCompare(
        point_shadow_cube_tex,
        shadow_sampler,
        dir,
        light.point_shadow_slot,
        normalised - bias,
    );
}

// #include "csm.wgsl"

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

// ---------------------------------------------------------------------------
// IBL helpers : equirectangular sampling
// This is the CANONICAL copy. Keep in sync with:
//   mesh_instanced.wgsl, mesh_oit.wgsl, mesh_instanced_oit.wgsl
// ---------------------------------------------------------------------------

const IBL_PI: f32 = 3.14159265;

/// Convert a Z-up world-space direction to equirectangular UV, applying optional
/// Z-axis rotation. The IBL panorama is sampled with its vertical axis aligned
/// to +Z; horizon HDR pixels girdle the camera in the XY plane.
fn dir_to_equirect_uv(dir: vec3<f32>, rotation: f32) -> vec2<f32> {
    let s = sin(rotation);
    let c = cos(rotation);
    let d = vec3<f32>(c * dir.x - s * dir.y, s * dir.x + c * dir.y, dir.z);
    let phi = atan2(d.y, d.x); // -PI..PI (longitude around Z)
    let theta = asin(clamp(d.z, -1.0, 1.0)); // -PI/2..PI/2 (latitude: Z is polar)
    return vec2<f32>(0.5 + phi / (2.0 * IBL_PI), 0.5 - theta / IBL_PI);
}

/// Sample the irradiance map (diffuse IBL).
fn sample_ibl_irradiance(N: vec3<f32>, rotation: f32) -> vec3<f32> {
    let uv = dir_to_equirect_uv(N, rotation);
    return textureSampleLevel(ibl_irradiance, ibl_sampler, uv, 0.0).rgb;
}

/// Sample the prefiltered specular map at a roughness-derived mip level.
fn sample_ibl_prefiltered(R: vec3<f32>, roughness: f32, rotation: f32) -> vec3<f32> {
    let uv = dir_to_equirect_uv(R, rotation);
    let max_mip = 4.0; // 5 mip levels -> max index 4
    let mip = roughness * max_mip;
    return textureSampleLevel(ibl_prefiltered, ibl_sampler, uv, mip).rgb;
}

/// Look up the BRDF integration LUT (x=NdotV, y=roughness).
fn sample_brdf_lut(NdotV: f32, roughness: f32) -> vec2<f32> {
    return textureSampleLevel(ibl_brdf_lut, ibl_sampler, vec2<f32>(NdotV, roughness), 0.0).rg;
}

struct IblContrib {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

/// Full IBL ambient: diffuse irradiance + specular split-sum.
fn ibl_ambient(
    N: vec3<f32>,
    V: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    ao: f32,
    intensity: f32,
    rotation: f32,
) -> IblContrib {
    let NdotV = max(dot(N, V), 0.001);
    let F = F_Schlick_roughness(NdotV, F0, roughness);
    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL.
    let irradiance = sample_ibl_irradiance(N, rotation);
    let diffuse_ibl = kD * irradiance * base_colour * ao * intensity;

    // Specular IBL (split-sum approximation).
    let R = reflect(-V, N);
    let prefiltered = sample_ibl_prefiltered(R, roughness, rotation);
    let brdf = sample_brdf_lut(NdotV, roughness);
    let specular_ibl = prefiltered * (F * brdf.x + brdf.y) * ao * intensity;

    return IblContrib(diffuse_ibl, specular_ibl);
}

fn F_Schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn pbr_light_contrib(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    radiance: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
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

// UV parameterization visualization : returns a procedural RGB colour from UV coordinates.
// mode: 1=checker, 2=grid, 3=localcheck (polar checker), 4=localrad (concentric rings).
// scale: tile frequency multiplier applied to uv before pattern evaluation.
fn param_vis_colour(uv: vec2<f32>, mode: u32, scale: f32) -> vec3<f32> {
    let col_a      = vec3<f32>(0.85, 0.85, 0.85);
    let col_b      = vec3<f32>(0.2,  0.2,  0.2);
    let line_col   = vec3<f32>(0.1,  0.1,  0.1);
    let bg_col     = vec3<f32>(0.85, 0.85, 0.85);
    let line_width = 0.05f;
    let su = uv.x * scale;
    let sv = uv.y * scale;
    if mode == 1u {
        // Checker: alternating squares in UV space.
        let p = (i32(floor(su)) + i32(floor(sv))) & 1;
        return select(col_a, col_b, p != 0);
    } else if mode == 2u {
        // Grid: thin lines at UV integer boundaries.
        let on_line = fract(su) < line_width || fract(sv) < line_width;
        return select(bg_col, line_col, on_line);
    } else if mode == 3u {
        // LocalChecker: polar checkerboard centred at UV (0.5, 0.5).
        let d      = uv - vec2<f32>(0.5);
        let r      = length(d) * scale * 2.0;
        let theta  = atan2(d.y, d.x);
        let ring   = i32(floor(r)) & 1;
        let sector = i32(floor(theta * 4.0 / 3.14159265 + 8.0)) & 1;
        return select(col_a, col_b, (ring ^ sector) != 0);
    } else {
        // LocalRadial: concentric rings centred at UV (0.5, 0.5).
        let r = length(uv - vec2<f32>(0.5)) * scale * 2.0;
        return select(col_a, col_b, (i32(floor(r)) & 1) != 0);
    }
}

struct Surface {
    resolved: bool,
    out_colour: vec4<f32>,
    base_colour: vec3<f32>,
    normal: vec3<f32>,
    ao_factor: f32,
    mat_uv: vec2<f32>,
    alpha: f32,
};

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

// Material prep. Samples textures, resolves the shading normal and AO, and
// handles the shading models that fully determine the colour (wireframe, face
// colour, unlit, matcap, uv-vis). When one of those applies, `resolved` is set
// and `out_colour` holds the final colour; otherwise the surface fields feed
// `compute_lit`. Clip and alpha-mask fragments discard here.
fn compute_surface(in: VertexOut, is_front: bool) -> Surface {
    var out: Surface;
    out.resolved = false;
    out.out_colour = vec4<f32>(0.0);
    out.base_colour = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.ao_factor = 1.0;
    out.mat_uv = in.uv;
    out.alpha = 1.0;

    // Section view: discard fragment if it falls on the clipped side of any plane.
    for (var i = 0u; i < clip_planes.count; i++) {
        let plane = clip_planes.planes[i];
        if dot(in.world_pos, plane.xyz) + plane.w < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    // Wireframe mode: override colour to gray, no lighting.
    if object.wireframe != 0u {
        out.resolved = true;
        out.out_colour = vec4<f32>(0.75, 0.75, 0.75, 1.0);
        return out;
    }

    // Per-material UV transform: atlas region / tiling selection. Identity
    // (offset 0,0 scale 1,1) passes the authored UV through unchanged.
    let mat_uv = in.uv * object.uv_transform.zw + object.uv_transform.xy;
    out.mat_uv = mat_uv;

    // Sample texture if one is assigned; fallback texture is 1x1 white (neutral multiply).
    var tex_colour = vec4<f32>(1.0);
    if object.has_texture == 1u {
        tex_colour = textureSample(obj_texture, obj_sampler, mat_uv);
    }
    let obj_colour = vec4<f32>(object.colour.rgb * in.colour.rgb * tex_colour.rgb,
                               object.colour.a   * in.colour.a   * tex_colour.a);
    out.alpha = obj_colour.a;

    // Alpha MASK: discard fragments whose alpha is below the cutoff.
    if object.alpha_mode == 1u && obj_colour.a < object.alpha_cutoff {
        discard;
    }

    var base_colour = obj_colour.rgb;

    // Scalar attribute colour override: sample LUT when has_attribute is set.
    if object.has_attribute != 0u {
        if in.is_nan_scalar > 0.5 {
            if object.use_nan_colour == 0u {
                discard;
            }
            out.resolved = true;
            out.out_colour = vec4<f32>(object.nan_colour.rgb, object.nan_colour.a);
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

    // Per-face RGBA colour: use directly, bypassing all lighting and colourmap logic.
    if object.use_face_colour != 0u {
        var fc = in.face_colour;
        if object.selected != 0u {
            fc = mix(fc, vec4<f32>(1.0, 0.55, 0.1, 1.0), 0.35);
        }
        out.resolved = true;
        out.out_colour = vec4<f32>(fc.rgb, fc.a * object.colour.a);
        return out;
    }

    // Unlit: skip all lighting, return raw colour directly.
    if object.unlit != 0u {
        out.resolved = true;
        out.out_colour = vec4<f32>(base_colour, obj_colour.a);
        return out;
    }

    // Resolve shading normal: TBN normal mapping, flat (screen-space
    // derivatives), or the interpolated geometric normal. `use_flat` takes
    // precedence over normal mapping: the per-face geometric normal is the
    // entire point of the flat shading model.
    var N: vec3<f32>;
    if object.use_flat != 0u {
        let dpx = dpdx(in.world_pos);
        let dpy = dpdy(in.world_pos);
        var Nf = normalize(cross(dpx, dpy));
        // Align with the authored winding so flat shading matches the
        // mesh's intended outward direction even when the cross product
        // resolves to the opposite hemisphere.
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
    // Runs before matcap/uv_vis/PBR so all downstream lighting paths use the substituted values.
    // 0=Cull, 1=Identical, 2=DifferentColour, 3=Tint, 4=Checker, 5=Hatching, 6=Crosshatch, 7=Stripes.
    if !is_front && object.backface_policy >= 2u {
        N = -N;
        if object.backface_policy == 2u {
            // DifferentColour: replace base_colour entirely.
            base_colour = object.backface_colour.rgb;
        } else if object.backface_policy == 3u {
            // Tint: darken base_colour by factor stored in backface_colour.r.
            base_colour = base_colour * (1.0 - object.backface_colour.r);
        } else {
            // Pattern modes (4..7): procedural pattern scaled to object size.
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

    // Matcap shading : the matcap texture encodes material appearance as a sphere-space lookup.
    // UV is derived from the view-space normal (x,y components).
    if object.use_matcap != 0u {
        // Transform world-space shading normal to view space (rotation only, w=0).
        let view_normal = normalize((camera.view * vec4<f32>(N, 0.0)).xyz);
        // Map view-space normal XY to UV.
        // Convention: -ny*0.5+0.5 so that normals pointing UP map to v=0 (top of
        // texture) which is where built-in matcaps place the bright region.
        //
        // Clamp the XY radius to 0.99 to stay just inside the matcap disc.
        // At grazing angles (silhouette) |view_normal.xy| -> 1, which samples the
        // transparent black border of the matcap image, producing a dark dotted band.
        let mc_len = length(view_normal.xy);
        let mc_scale = select(1.0, 0.99 / mc_len, mc_len > 0.99);
        let matcap_uv = vec2<f32>(
            view_normal.x * mc_scale * 0.5 + 0.5,
            -view_normal.y * mc_scale * 0.5 + 0.5,
        );
        let mc = textureSample(matcap_texture, obj_sampler, matcap_uv);
        out.resolved = true;
        if object.matcap_blendable != 0u {
            // Blendable: RGB is the matcap colour; A tints the base geometry colour.
            let blended = clamp(mc.rgb + mc.a * base_colour, vec3<f32>(0.0), vec3<f32>(1.0));
            out.out_colour = vec4<f32>(blended, obj_colour.a);
        } else {
            // Static: matcap RGB fully overrides the object colour.
            out.out_colour = vec4<f32>(mc.rgb, obj_colour.a);
        }
        return out;
    }

    // UV parameterization visualization: procedural pattern replaces all lighting.
    if object.uv_vis_mode != 0u {
        let vis = param_vis_colour(in.uv, object.uv_vis_mode, object.uv_vis_scale);
        out.resolved = true;
        out.out_colour = vec4<f32>(vis, obj_colour.a);
        return out;
    }

    out.base_colour = base_colour;
    out.normal = N;
    out.ao_factor = ao_factor;
    return out;
}

// Lighting. Runs the standard PBR / Blinn-Phong light loops, shadow sampling,
// and ambient/IBL on a resolved surface. Returns the pre-emissive colour plus
// the debug accumulators the debug-vis overlay reads.
fn compute_lit(surface: Surface, in: VertexOut) -> LitResult {
    let base_colour = surface.base_colour;
    let ao_factor = surface.ao_factor;
    let mat_uv = surface.mat_uv;
    let N = surface.normal;

    // Use the smooth vertex normal for shadow bias. Screen-space derivatives
    // (dpdx/dpdy) become unreliable when the surface covers few pixels (zoomed
    // out) because edge fragments include helper invocations with undefined
    // world_pos, producing garbage normals that flip offset_sign and cause
    // self-shadowing. N is correctly interpolated and stable at all distances.
    let shadow_normal = N;

    let V = normalize(camera.eye_pos - in.world_pos);
    let tint = vec4<f32>(1.0, 1.0, 1.0, 1.0);

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
        // Cook-Torrance PBR path
        var metallic  = clamp(object.metallic,  0.0, 1.0);
        var roughness = max(object.roughness, 0.04);
        if object.has_metallic_roughness_tex != 0u {
            // glTF ORM texture: G=roughness factor, B=metallic factor. Per-material
            // `metallic_range` / `roughness_range` remap the raw samples before the
            // scalar factor; identity `(0, 1)` is a no-op.
            let mr = textureSample(metallic_roughness_tex, obj_sampler, mat_uv);
            let m_remapped = mix(object.metallic_range.x, object.metallic_range.y, mr.b);
            let r_remapped = mix(object.roughness_range.x, object.roughness_range.y, mr.g);
            metallic  = clamp(m_remapped * metallic,  0.0, 1.0);
            roughness = max(r_remapped * roughness, 0.04);
        }
        let F0 = mix(vec3<f32>(0.04), base_colour, metallic);

        var Lo = vec3<f32>(0.0);
        let pbr_range = cluster_light_range(in.world_pos, lights_uniform.count);
        for (var j: u32 = 0u; j < pbr_range.count; j = j + 1u) {
            let i = cluster_light_global(pbr_range, j);
            let l = lights_storage[i];
            var L: vec3<f32>;
            var radiance: vec3<f32>;

            if l.light_type == 0u {
                L = normalize(l.pos_or_dir);
                radiance = l.colour * l.intensity;
            } else if l.light_type == 1u {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                if dist >= l.range { continue; }
                L = to_light / max(dist, 0.0001);
                let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                radiance = l.colour * l.intensity * falloff * falloff;
            } else {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                if dist >= l.range { continue; }
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
            // Backfacing: pbr_light_contrib returns exactly zero, so skip
            // the shadow samples and the BRDF outright.
            if dot(N, L) <= 0.0 { continue; }

            // Shadow factor. Directional `lights[0]` uses CSM; point lights
            // with an allocated cubemap slot sample the point shadow array.
            // Spot lights keep their existing single-face perspective path
            // (sampled inside `sample_shadow_csm` when lights[0] is a spot;
            // unshadowed otherwise). Per-receiver opt-out via
            // ItemSettings.receive_shadows skips the sample.
            var shadow_factor = 1.0;
            if lights_uniform.shadows_enabled != 0u && object.receive_shadows != 0u {
                if i == 0u && lights_storage[0].light_type != 1u {
                    last_shadow_sample = sample_shadow_csm(in.world_pos, camera.eye_pos, shadow_normal, L, select(0u, 1u, object.backface_policy != 0u));
                    shadow_factor = last_shadow_sample.factor;
                } else if l.light_type == 1u && l.point_shadow_slot >= 0 {
                    shadow_factor = sample_point_shadow(l, in.world_pos);
                }
            }
            radiance *= shadow_factor;

            Lo += pbr_light_contrib(N, V, L, radiance, base_colour,
                                    metallic, roughness, F0);
        }

        dbg_direct_lum = dot(Lo, lum_weights);
        dbg_roughness  = roughness;
        dbg_metallic   = metallic;

        // Ambient: IBL when enabled, hemisphere fallback otherwise.
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

        final_rgb = clamp((Lo + ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        // BEGIN_PBR_STRIP
    } else {
        // Multi-light Blinn-Phong path
        var total_colour_contrib = vec3<f32>(0.0);

        let bp_range = cluster_light_range(in.world_pos, lights_uniform.count);
        for (var j: u32 = 0u; j < bp_range.count; j = j + 1u) {
            let i = cluster_light_global(bp_range, j);
            let l = lights_storage[i];

            var light_dir: vec3<f32>;
            var attenuation = 1.0;

            if l.light_type == 0u {
                light_dir = normalize(l.pos_or_dir);
            } else if l.light_type == 1u {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                if dist >= l.range { continue; }
                light_dir = to_light / max(dist, 0.0001);
                let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                attenuation = falloff * falloff;
            } else {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                if dist >= l.range { continue; }
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

            var shadow = 1.0;
            if lights_uniform.shadows_enabled != 0u && object.receive_shadows != 0u {
                if i == 0u && lights_storage[0].light_type != 1u {
                    last_shadow_sample = sample_shadow_csm(in.world_pos, camera.eye_pos, shadow_normal, light_dir, select(0u, 1u, object.backface_policy != 0u));
                    shadow = last_shadow_sample.factor;
                } else if l.light_type == 1u && l.point_shadow_slot >= 0 {
                    shadow = sample_point_shadow(l, in.world_pos);
                }
            }

            let H = normalize(light_dir + V);
            let n_dot_l = max(dot(N, light_dir), 0.0);
            let n_dot_h = max(dot(N, H), 0.0);

            let diffuse_contrib  = object.diffuse  * n_dot_l * l.intensity * attenuation * shadow;
            let specular_contrib = object.specular * pow(n_dot_h, object.shininess)
                                 * l.intensity * attenuation * shadow;

            total_colour_contrib += (diffuse_contrib + specular_contrib) * l.colour;
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
        // END_PBR_STRIP
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
fn fs_main(in: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let surface = compute_surface(in, is_front);
    if surface.resolved {
        return surface.out_colour;
    }

    let lit = compute_lit(surface, in);

    // Re-bind the locals the debug-vis overlay reads before the include.
    let N = surface.normal;
    let ao_factor = surface.ao_factor;
    let mat_uv = surface.mat_uv;
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
        emissive = emissive * textureSample(emissive_tex, obj_sampler, mat_uv).rgb;
    }
    final_rgb += emissive;
    var dbg_emissive_lum = dot(emissive, lum_weights);

    // #include "debug_vis.wgsl"

    return vec4<f32>(final_rgb, surface.alpha);
}
