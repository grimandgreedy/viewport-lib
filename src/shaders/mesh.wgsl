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
//          + normal map (binding 3) + AO map (binding 4).
//
// Lighting: Blinn-Phong (ambient + diffuse + specular) with multi-light support.
//           Cook-Torrance PBR when object.use_pbr != 0.
// Shadow mapping: CSM with atlas-based cascade selection.
//   PCF (3x3) or PCSS (blocker search + variable-width PCF) via shadow_atlas.shadow_filter.
// Selection: orange tint when object.selected == 1u.
// Wireframe: gray color override when object.wireframe == 1u.
// Section views: fragment discarded when world_pos fails any active clip plane.
// Normal maps: tangent-space normal mapping via TBN when object.has_normal_map != 0u.
// AO maps: ambient occlusion applied to ambient + diffuse when object.has_ao_map != 0u.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

// Single light entry — 128 bytes.
struct SingleLight {
    light_view_proj: mat4x4<f32>,  // 64 bytes (shadow matrix, lights[0] only)
    pos_or_dir: vec3<f32>,          // 12 bytes
    light_type: u32,               //  4 bytes (0=directional, 1=point, 2=spot)
    color: vec3<f32>,              // 12 bytes
    intensity: f32,                //  4 bytes
    range: f32,                    //  4 bytes
    inner_angle: f32,              //  4 bytes
    outer_angle: f32,              //  4 bytes
    spot_direction: vec3<f32>,     // 12 bytes
    _pad: vec2<f32>,               //  8 bytes
};

struct Lights {
    count: u32,
    shadow_bias: f32,
    shadows_enabled: u32,
    _pad: u32,
    sky_color: vec3<f32>,
    hemisphere_intensity: f32,
    ground_color: vec3<f32>,
    _pad2: f32,
    lights: array<SingleLight, 8>,
};

// Clip planes uniform — 112 bytes.
struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count: u32,
    _pad0: u32,
    viewport_width: f32,
    viewport_height: f32,
};

// Shadow atlas uniform — 416 bytes.
struct ShadowAtlas {
    cascade_vp: array<mat4x4<f32>, 4>,   // 256 bytes
    cascade_splits: vec4<f32>,            //  16 bytes
    cascade_count: u32,                   //   4 bytes
    atlas_size: f32,                      //   4 bytes
    shadow_filter: u32,                   //   4 bytes (0=PCF, 1=PCSS)
    pcss_light_radius: f32,               //   4 bytes
    atlas_rects: array<vec4<f32>, 8>,     // 128 bytes
};

// Per-object uniform — 176 bytes.
struct Object {
    model: mat4x4<f32>,
    color: vec4<f32>,
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
    nan_color: vec4<f32>,    // offset 144
    use_nan_color: u32,      // offset 160
    _pad_nan0: u32,
    _pad_nan1: u32,
    _pad_nan2: u32,
};

struct ClipVolumeUB {
    volume_type: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    plane_normal: vec3<f32>,
    plane_dist: f32,
    box_center: vec3<f32>,
    _padB0: f32,
    box_half_extents: vec3<f32>,
    _padB1: f32,
    box_col0: vec3<f32>,
    _padB2: f32,
    box_col1: vec3<f32>,
    _padB3: f32,
    box_col2: vec3<f32>,
    _padB4: f32,
    sphere_center: vec3<f32>,
    sphere_radius: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform> lights_uniform: Lights;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(5) var<uniform> shadow_atlas: ShadowAtlas;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

fn clip_volume_test(p: vec3<f32>) -> bool {
    if clip_volume.volume_type == 0u { return true; }
    if clip_volume.volume_type == 1u {
        return dot(p, clip_volume.plane_normal) + clip_volume.plane_dist >= 0.0;
    }
    if clip_volume.volume_type == 2u {
        let d = p - clip_volume.box_center;
        let local = vec3<f32>(
            dot(d, clip_volume.box_col0),
            dot(d, clip_volume.box_col1),
            dot(d, clip_volume.box_col2),
        );
        return abs(local.x) <= clip_volume.box_half_extents.x
            && abs(local.y) <= clip_volume.box_half_extents.y
            && abs(local.z) <= clip_volume.box_half_extents.z;
    }
    let ds = p - clip_volume.sphere_center;
    return dot(ds, ds) <= clip_volume.sphere_radius * clip_volume.sphere_radius;
}
@group(1) @binding(0) var<uniform> object: Object;
@group(1) @binding(1) var obj_texture: texture_2d<f32>;
@group(1) @binding(2) var obj_sampler: sampler;
@group(1) @binding(3) var normal_map: texture_2d<f32>;
@group(1) @binding(4) var ao_map: texture_2d<f32>;
@group(1) @binding(5) var lut_texture: texture_2d<f32>;
@group(1) @binding(6) var<storage, read> scalar_buffer: array<f32>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color:          vec4<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) world_pos:      vec3<f32>,
    @location(3) uv:             vec2<f32>,
    @location(4) world_tangent:  vec4<f32>,
    @location(5) scalar_val:     f32,
    // 1.0 if the source scalar vertex value was NaN, 0.0 otherwise.
    // Detected in vs_main before interpolation can corrupt the NaN bit pattern.
    @location(6) is_nan_scalar:  f32,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = object.model * vec4<f32>(in.position, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.color = in.color;
    out.world_pos = world_pos.xyz;
    let model3 = mat3x3<f32>(
        object.model[0].xyz,
        object.model[1].xyz,
        object.model[2].xyz,
    );
    out.world_normal = normalize(model3 * in.normal);
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
    return out;
}

// ---------------------------------------------------------------------------
// 32-sample Poisson disk (first 16 used for blocker search, all 32 for filter)
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

// ---------------------------------------------------------------------------
// CSM shadow sampling — selects cascade by eye distance, samples atlas tile
// ---------------------------------------------------------------------------
fn sample_shadow_csm(world_pos: vec3<f32>, eye_pos: vec3<f32>, surface_normal: vec3<f32>) -> f32 {
    let dist = distance(eye_pos, world_pos);

    // Select cascade based on distance.
    var cascade_idx = 0u;
    for (var i = 0u; i < shadow_atlas.cascade_count; i++) {
        if dist > shadow_atlas.cascade_splits[i] {
            cascade_idx = i + 1u;
        }
    }
    cascade_idx = min(cascade_idx, shadow_atlas.cascade_count - 1u);

    // Project the actual surface position to get the correct shadow-map UV.
    // We must NOT offset the UV — on curved surfaces the tangential component
    // of the normal would shift the sample into a shallower shadow-map region
    // (closer to the light), causing MORE false shadows instead of fewer.
    let light_clip = shadow_atlas.cascade_vp[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;

    // NDC → tile UV [0,1].
    let tile_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    // Out-of-range check.
    if tile_uv.x < 0.0 || tile_uv.x > 1.0 || tile_uv.y < 0.0 || tile_uv.y > 1.0 ||
       ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    // Remap tile UV through atlas rect.
    let rect = shadow_atlas.atlas_rects[cascade_idx];
    let atlas_uv = vec2<f32>(
        mix(rect.x, rect.z, tile_uv.x),
        mix(rect.y, rect.w, tile_uv.y),
    );

    let texel_size = 1.0 / shadow_atlas.atlas_size;

    // Normal-offset depth bias: project world_pos + N*offset into shadow space
    // and use the resulting Z as the comparison depth. This offsets depth along
    // the surface normal direction, so lit faces (N·L > 0) get a depth that is
    // closer to the light than the actual surface — preventing self-intersection.
    // Crucially, we keep the UV from the original position (above), so curved
    // surfaces don't shift into wrong shadow-map regions.
    let offset_clip = shadow_atlas.cascade_vp[cascade_idx] * vec4<f32>(world_pos + surface_normal * 0.002, 1.0);
    let biased_depth = (offset_clip.xyz / offset_clip.w).z - lights_uniform.shadow_bias;

    // Per-fragment Poisson disk rotation — breaks up the coherent square/blob
    // pattern that results from every pixel using the same disk orientation.
    // Uses world_pos.xz as seed so adjacent pixels get different rotations.
    let noise = fract(52.9829189 * fract(dot(world_pos.xz, vec2<f32>(0.06711056, 0.00583715))));
    let rot = noise * 6.28318530;
    let sin_r = sin(rot);
    let cos_r = cos(rot);

    if shadow_atlas.shadow_filter == 1u {
        // ---------------------------------------------------------------
        // PCSS: blocker search → penumbra estimation → variable PCF
        // ---------------------------------------------------------------
        let search_radius = shadow_atlas.pcss_light_radius * 16.0 * texel_size;

        // Phase 1: Blocker search (16 Poisson samples).
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

        if blocker_count < 1.0 {
            return 1.0;
        }

        let avg_blocker = blocker_sum / blocker_count;
        let penumbra_width = shadow_atlas.pcss_light_radius * (biased_depth - avg_blocker) / max(avg_blocker, 0.001);
        let filter_radius = max(penumbra_width * 16.0 * texel_size, texel_size);

        // Phase 2: Variable-width PCF (32 Poisson samples).
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
        // ---------------------------------------------------------------
        // 32-sample Poisson-disk PCF at 4-texel radius, per-fragment rotation.
        // ---------------------------------------------------------------
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

fn pbr_light_contrib(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    radiance: vec3<f32>,
    base_color: vec3<f32>,
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
    return (kD * base_color / 3.14159265 + specular) * radiance * NdotL;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Section view: discard fragment if it falls on the clipped side of any plane.
    for (var i = 0u; i < clip_planes.count; i++) {
        let plane = clip_planes.planes[i];
        if dot(in.world_pos, plane.xyz) + plane.w < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    // Wireframe mode: override color to gray, no lighting.
    if object.wireframe != 0u {
        return vec4<f32>(0.08, 0.08, 0.08, 1.0);
    }

    // Sample texture if one is assigned; fallback texture is 1x1 white (neutral multiply).
    var tex_color = vec4<f32>(1.0);
    if object.has_texture == 1u {
        tex_color = textureSample(obj_texture, obj_sampler, in.uv);
    }
    let obj_color = vec4<f32>(object.color.rgb * in.color.rgb * tex_color.rgb,
                               object.color.a   * in.color.a   * tex_color.a);
    var base_color = obj_color.rgb;

    // Scalar attribute colour override: sample LUT when has_attribute is set.
    if object.has_attribute != 0u {
        if in.is_nan_scalar > 0.5 {
            if object.use_nan_color == 0u {
                discard;
            }
            return vec4<f32>(object.nan_color.rgb, object.nan_color.a);
        }
        let raw = in.scalar_val;
        let range = object.scalar_max - object.scalar_min;
        let t = clamp(
            select(0.0, (raw - object.scalar_min) / range, range > 0.0001),
            0.0, 1.0,
        );
        base_color = textureSampleLevel(lut_texture, obj_sampler, vec2<f32>(t, 0.5), 0.0).rgb;
    }

    // Resolve shading normal: TBN normal mapping or geometric normal.
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

    // AO factor from AO map.
    var ao_factor = 1.0;
    if object.has_ao_map != 0u {
        ao_factor = textureSample(ao_map, obj_sampler, in.uv).r;
    }

    let V = normalize(camera.eye_pos - in.world_pos);
    let tint = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    var final_rgb: vec3<f32>;

    if object.use_pbr != 0u {
        // Cook-Torrance PBR path
        let metallic  = clamp(object.metallic,  0.0, 1.0);
        let roughness = max(object.roughness, 0.04);
        let F0 = mix(vec3<f32>(0.04), base_color, metallic);

        var Lo = vec3<f32>(0.0);
        for (var i = 0u; i < lights_uniform.count; i++) {
            let l = lights_uniform.lights[i];
            var L: vec3<f32>;
            var radiance: vec3<f32>;

            if l.light_type == 0u {
                L = normalize(l.pos_or_dir);
                radiance = l.color * l.intensity;
            } else if l.light_type == 1u {
                let to_light = l.pos_or_dir - in.world_pos;
                let dist = length(to_light);
                L = to_light / max(dist, 0.0001);
                let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
                radiance = l.color * l.intensity * falloff * falloff;
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
                radiance = l.color * l.intensity * dist_falloff * dist_falloff * cone_att;
            }

            // Shadow factor (lights[0] only) — CSM.
            var shadow_factor = 1.0;
            if i == 0u && lights_uniform.shadows_enabled != 0u {
                shadow_factor = sample_shadow_csm(in.world_pos, camera.eye_pos, N);
                // Fade shadow to 1.0 near the terminator (N·L ≈ 0).
                // Shadow-map texels project at grazing angles near the terminator,
                // causing visible squares. N·L already smoothly darkens that region,
                // so we suppress the shadow map there.
                let terminator = smoothstep(0.0, 0.5, dot(N, L));
                shadow_factor = mix(1.0, shadow_factor, terminator);
            }
            radiance *= shadow_factor;

            Lo += pbr_light_contrib(N, V, L, radiance, base_color,
                                    metallic, roughness, F0);
        }

        // Ambient (material ambient + hemisphere) with AO.
        let hemi_t = clamp(in.world_normal.y * 0.5 + 0.5, 0.0, 1.0);
        let hemi_color = mix(lights_uniform.ground_color, lights_uniform.sky_color, hemi_t);
        let ambient_scale = vec3<f32>(object.ambient) + hemi_color * lights_uniform.hemisphere_intensity;
        let ambient = ambient_scale * (base_color * (1.0 - metallic) + F0 * metallic) * ao_factor;

        final_rgb = clamp((Lo + ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        // Multi-light Blinn-Phong path
        var total_color_contrib = vec3<f32>(0.0);

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

            var shadow = 1.0;
            if i == 0u && lights_uniform.shadows_enabled != 0u {
                shadow = sample_shadow_csm(in.world_pos, camera.eye_pos, N);
                // Terminator fade: suppress shadow map near N·L ≈ 0 to avoid
                // shadow-texel squares on curved surfaces.
                let terminator = smoothstep(0.0, 0.5, dot(N, light_dir));
                shadow = mix(1.0, shadow, terminator);
            }

            let H = normalize(light_dir + V);
            let n_dot_l = max(dot(N, light_dir), 0.0);
            let n_dot_h = max(dot(N, H), 0.0);

            let diffuse_contrib  = object.diffuse  * n_dot_l * l.intensity * attenuation * shadow;
            let specular_contrib = object.specular * pow(n_dot_h, object.shininess)
                                 * l.intensity * attenuation * shadow;

            total_color_contrib += (diffuse_contrib + specular_contrib) * l.color;
        }

        let ambient_contrib = object.ambient;
        let hemi_t = clamp(in.world_normal.y * 0.5 + 0.5, 0.0, 1.0);
        let hemi_color = mix(lights_uniform.ground_color, lights_uniform.sky_color, hemi_t);
        let hemi_ambient = hemi_color * lights_uniform.hemisphere_intensity;

        let lit_rgb = base_color * (ambient_contrib + hemi_ambient) * ao_factor
                    + base_color * total_color_contrib;
        final_rgb = clamp(lit_rgb * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return vec4<f32>(final_rgb, obj_color.a);
}
