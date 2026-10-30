
// @viewport-wgsl-version: 2
// Shared PBR / lit-shading helpers. Requires SHARED_BINDINGS_WGSL to be
// included first.
//
// Internal binding: the CSM uniform at @group(0) @binding(5) is declared
// here for `viewport_sample_csm`'s use. Plugin shaders must not reference
// `_viewport_csm` directly.

// The direct BRDF the lit mesh shaders evaluate (the viewport_brdf_*
// functions).
// #include "brdf.wgsl"

// The cascade shadow uniform's layout.
// #include "shadow_atlas.wgsl"

@group(0) @binding(5) var<uniform> _viewport_csm: ShadowAtlas;

const _VIEWPORT_POISSON_DISK: array<vec2<f32>, 32> = array<vec2<f32>, 32>(
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

struct PbrInputs {
    world_pos:  vec3<f32>,
    world_n:    vec3<f32>,
    view_dir:   vec3<f32>,
    albedo:     vec3<f32>,
    metallic:   f32,
    roughness:  f32,
    ao:         f32,
    emissive:   vec3<f32>,
};

// One light's direction and unshadowed radiance at a point.
struct _ViewportLightEval {
    l:        vec3<f32>,
    radiance: vec3<f32>,
};

// The attenuation the lit mesh shaders use: inverse-square falloff clamped by
// the source radius, faded to zero at `range` by a quartic window, and the
// spot cone on top. Zero radiance at or beyond `range`.
fn _viewport_eval_light(light: SingleLight, world_pos: vec3<f32>) -> _ViewportLightEval {
    var ev: _ViewportLightEval;
    if light.light_type == 0u {
        ev.l = normalize(light.pos_or_dir);
        ev.radiance = light.colour * light.intensity;
        return ev;
    }
    let to_light = light.pos_or_dir - world_pos;
    let dist = length(to_light);
    ev.l = to_light / max(dist, 0.0001);
    let r2 = max(light.radius * light.radius, 1.0e-4);
    let inv_sq = 1.0 / max(dist * dist, r2);
    let win = clamp(1.0 - pow(dist / light.range, 4.0), 0.0, 1.0);
    let atten = inv_sq * win * win;
    if light.light_type == 1u {
        ev.radiance = light.colour * light.intensity * atten;
    } else {
        let spot_dir = normalize(light.spot_direction);
        let cos_angle = dot(-ev.l, spot_dir);
        let cos_outer = cos(light.outer_angle);
        let cos_inner = cos(light.inner_angle);
        let cone_att = clamp(
            (cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
            0.0, 1.0,
        );
        ev.radiance = light.colour * light.intensity * atten * cone_att;
    }
    return ev;
}

fn viewport_apply_scene_lighting(
    normal: vec3<f32>,
    base_colour: vec3<f32>,
    two_sided: bool,
    world_pos: vec3<f32>,
) -> vec3<f32> {
    let up_weight = clamp(normal.z * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(lights.ground_colour, lights.sky_colour, up_weight)
                  * lights.hemisphere_intensity;

    var direct = vec3<f32>(0.0);
    let n_lights = lights.count;
    for (var i: u32 = 0u; i < n_lights; i = i + 1u) {
        let ev = _viewport_eval_light(lights_storage[i], world_pos);
        let L = ev.l;
        let radiance = ev.radiance;
        let raw = dot(normal, L);
        let n_dot_l = select(max(raw, 0.0), abs(raw), two_sided);
        var shadow_factor = 1.0;
        if i == 0u {
            shadow_factor = viewport_sample_csm(world_pos, normal);
        }
        direct = direct + radiance * n_dot_l * shadow_factor;
    }
    return base_colour * (ambient + direct);
}

// Cascade selection + atlas tap. Mirrors the bias scheme and filter
// kernel used by the lib's built-in mesh pipeline so plugin items composite
// consistently in the same scene. Implementation details (cascade count,
// filter, bias) are internal and may change between catalog versions.
fn viewport_sample_csm(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if lights.shadows_enabled == 0u || lights.count == 0u {
        return 1.0;
    }

    let primary = lights_storage[0];
    var light_dir: vec3<f32>;
    if primary.light_type == 0u {
        light_dir = normalize(primary.pos_or_dir);
    } else {
        let to_light = primary.pos_or_dir - world_pos;
        light_dir = to_light / max(length(to_light), 0.0001);
    }

    let eye_pos = camera.eye_pos;
    let dist = dot(world_pos - eye_pos, camera.forward);

    var cascade_idx = 0u;
    for (var i = 0u; i < _viewport_csm.cascade_count; i = i + 1u) {
        if dist > _viewport_csm.cascade_splits[i] {
            cascade_idx = i + 1u;
        }
    }
    cascade_idx = min(cascade_idx, _viewport_csm.cascade_count - 1u);

    let light_clip = _viewport_csm.cascade_vp[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let tile_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    let rect = _viewport_csm.atlas_rects[cascade_idx];
    let atlas_uv = vec2<f32>(
        mix(rect.x, rect.z, tile_uv.x),
        mix(rect.y, rect.w, tile_uv.y),
    );

    let n_dot_l = dot(world_normal, light_dir);
    let offset_sign = select(-1.0, 1.0, n_dot_l >= 0.0);
    let vp = _viewport_csm.cascade_vp[cascade_idx];
    let vp_row0 = vec3<f32>(vp[0][0], vp[1][0], vp[2][0]);
    let vp_row1 = vec3<f32>(vp[0][1], vp[1][1], vp[2][1]);
    let vp_row2 = vec3<f32>(vp[0][2], vp[1][2], vp[2][2]);
    let texel_world = 2.0 / (length(vp_row0) * _viewport_csm.atlas_size * (rect.z - rect.x));

    let primary_light_type = primary.light_type;
    var offset_world: vec3<f32>;
    if primary_light_type == 0u {
        let normal_bias = texel_world * 1.5;
        offset_world = world_pos - light_dir * normal_bias;
    } else {
        let normal_bias = texel_world * mix(1.5, 0.0, clamp(abs(n_dot_l), 0.0, 1.0));
        offset_world = world_pos + world_normal * (offset_sign * normal_bias);
    }
    let offset_clip = _viewport_csm.cascade_vp[cascade_idx] * vec4<f32>(offset_world, 1.0);
    let biased_depth = (offset_clip.xyz / offset_clip.w).z - lights.shadow_bias;

    if tile_uv.x < 0.0 || tile_uv.x > 1.0 || tile_uv.y < 0.0 || tile_uv.y > 1.0 ||
       ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let n_ndc = vec3<f32>(
        dot(vp_row0, world_normal) / dot(vp_row0, vp_row0),
        dot(vp_row1, world_normal) / dot(vp_row1, vp_row1),
        dot(vp_row2, world_normal) / dot(vp_row2, vp_row2),
    );
    let nz_sign = select(-1.0, 1.0, n_ndc.z >= 0.0);
    let nz = nz_sign * max(abs(n_ndc.z), 1e-4);
    let rp_gate = select(0.0, 1.0, primary_light_type == 0u);
    let depth_grad = vec2<f32>(
        -n_ndc.x / nz * 2.0 / (rect.z - rect.x),
         n_ndc.y / nz * 2.0 / (rect.w - rect.y),
    ) * rp_gate;

    let texel_size = 1.0 / _viewport_csm.atlas_size;
    let noise = fract(52.9829189 * fract(dot(world_pos.xz, vec2<f32>(0.06711056, 0.00583715))));
    let rot = noise * 6.28318530;
    let sin_r = sin(rot);
    let cos_r = cos(rot);

    if _viewport_csm.shadow_filter == 1u {
        // Radii in world units, as the mesh shaders size them.
        let tan_sun = _viewport_csm.tan_sun;
        let r_max = _viewport_csm.range.w;
        let depth_scale = 1.0 / max(length(vp_row2), 1e-8);
        let search_world = min(tan_sun * max(ndc.z, 0.0) * depth_scale, r_max);
        let search_radius = max(search_world / texel_world, 1.0) * texel_size;
        var blocker_sum = 0.0;
        var blocker_count = 0.0;
        for (var i = 0u; i < 16u; i = i + 1u) {
            let d = _VIEWPORT_POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * search_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let coords = vec2<i32>(clamped_uv * _viewport_csm.atlas_size);
            let raw_depth = textureLoad(shadow_atlas_tex, coords, 0);
            if raw_depth < ndc.z {
                blocker_sum = blocker_sum + raw_depth;
                blocker_count = blocker_count + 1.0;
            }
        }
        if blocker_count < 1.0 {
            return 1.0;
        }
        let avg_blocker = blocker_sum / blocker_count;
        let penumbra_world = min(tan_sun * max(biased_depth - avg_blocker, 0.0) * depth_scale, r_max);
        let filter_radius = max(penumbra_world / texel_world, 1.0) * texel_size;
        var shadow = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let d = _VIEWPORT_POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * filter_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let tap_depth = biased_depth
                + clamp(dot(depth_grad, clamped_uv - atlas_uv), -0.005, 0.005);
            shadow = shadow + textureSampleCompare(shadow_atlas_tex, shadow_atlas_sampler, clamped_uv, tap_depth);
        }
        return shadow / 32.0;
    } else {
        let pcf_radius = select(4.0, 1.5, primary_light_type == 0u) * texel_size;
        var shadow = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let d = _VIEWPORT_POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * pcf_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let tap_depth = biased_depth
                + clamp(dot(depth_grad, clamped_uv - atlas_uv), -0.005, 0.005);
            shadow = shadow + textureSampleCompare(shadow_atlas_tex, shadow_atlas_sampler, clamped_uv, tap_depth);
        }
        return shadow / 32.0;
    }
}

// Rotate a direction into the environment map's frame and project it to
// equirectangular UV. Matches the mapping used by the built-in mesh
// pipelines so plugins reflect the same environment.
fn viewport_dir_to_equirect_uv(dir: vec3<f32>, rotation: f32) -> vec2<f32> {
    let s = sin(rotation);
    let c = cos(rotation);
    let d = vec3<f32>(c * dir.x - s * dir.y, s * dir.x + c * dir.y, dir.z);
    return vec2<f32>(
        0.5 + atan2(d.y, d.x) / (2.0 * 3.14159265),
        0.5 - asin(clamp(d.z, -1.0, 1.0)) / 3.14159265,
    );
}

// Geometric specular anti-aliasing: widen perceptual roughness by the
// screen-space variance of the shading normal so detailed normal maps do
// not alias into per-pixel glints. Matches the built-in mesh pipelines.
fn viewport_specular_aa_roughness(n: vec3<f32>, roughness: f32) -> f32 {
    return viewport_brdf_specular_aa_roughness(n, roughness);
}

fn viewport_fresnel_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return viewport_brdf_f_schlick_roughness(cos_theta, f0, roughness);
}

// Image-based ambient: diffuse irradiance plus prefiltered specular
// reflection from the environment, combined through the split-sum BRDF
// LUT. This is what lets metallic and smooth surfaces reflect the sky
// instead of going dark under hemisphere ambient alone.
fn viewport_ibl_ambient(
    n: vec3<f32>,
    v: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
    ao: f32,
) -> vec3<f32> {
    let n_dot_v = max(dot(n, v), 0.001);
    let f = viewport_fresnel_roughness(n_dot_v, f0, roughness);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let rotation = lights.ibl_rotation;
    let irradiance =
        textureSampleLevel(ibl_irradiance_tex, ibl_sampler, viewport_dir_to_equirect_uv(n, rotation), 0i, 0.0).rgb;
    let r = reflect(-v, n);
    // Mip floored by the screen-space footprint of r, matching the built-in
    // mesh pipelines: magnified reflections on detailed normals would
    // otherwise point-sample hot mip-0 texels and read as speckle.
    let dr = max(length(dpdx(r)), length(dpdy(r)));
    let footprint_mip = clamp(log2(max(dr * 256.0 / (2.0 * 3.14159265), 1.0)), 0.0, 4.0);
    let prefiltered = textureSampleLevel(
        ibl_specular_tex,
        ibl_sampler,
        viewport_dir_to_equirect_uv(r, rotation),
        0i,
        max(roughness * 4.0, footprint_mip),
    ).rgb;
    let brdf = textureSampleLevel(ibl_brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
    // AO as the lit mesh shaders take it: the visibility on the diffuse, the
    // specular occlusion on the reflection, each through the multi-bounce fit
    // when the lighting enables it.
    let so = viewport_brdf_spec_occlusion(n_dot_v, ao, roughness * roughness);
    var diffuse_occ = vec3<f32>(ao);
    var spec_occ = vec3<f32>(so);
    if (lights.ao_flags & 1u) != 0u {
        diffuse_occ = viewport_brdf_multi_bounce_ao(ao, albedo * (1.0 - metallic));
        spec_occ = viewport_brdf_multi_bounce_ao(so, f0);
    }
    let diffuse = kd * irradiance * albedo * diffuse_occ * lights.ibl_diffuse_scale;
    let specular = prefiltered * (f * brdf.x + brdf.y) * spec_occ * lights.ibl_specular_scale;
    return (diffuse + specular) * lights.ibl_intensity;
}

// PBR shading with the lit mesh shaders' direct BRDF: Cook-Torrance specular
// (GGX, separable Smith, Schlick) and Lambert diffuse weighted by
// (1 - metallic). Integrates against every active scene light; IBL
// contribution is added when `lights.ibl_enabled != 0`.
fn viewport_pbr_shade(inp: PbrInputs) -> vec3<f32> {
    let N = normalize(inp.world_n);
    let V = normalize(inp.view_dir);
    let roughness = viewport_specular_aa_roughness(N, max(inp.roughness, 0.04));
    let f0 = mix(vec3<f32>(0.04), inp.albedo, inp.metallic);

    // Environment reflection when an IBL probe is active, otherwise a
    // cheap hemisphere ambient. The IBL path gives metallic and smooth
    // surfaces something bright to reflect instead of reading as black.
    var ambient: vec3<f32>;
    if lights.ibl_enabled != 0u {
        ambient = viewport_ibl_ambient(N, V, inp.albedo, inp.metallic, roughness, f0, inp.ao);
    } else {
        let up_weight = clamp(N.z * 0.5 + 0.5, 0.0, 1.0);
        var occ = vec3<f32>(inp.ao);
        if (lights.ao_flags & 1u) != 0u {
            occ = viewport_brdf_multi_bounce_ao(inp.ao, inp.albedo * (1.0 - inp.metallic));
        }
        ambient = mix(lights.ground_colour, lights.sky_colour, up_weight)
                  * lights.hemisphere_intensity * inp.albedo * occ;
    }

    var lo = vec3<f32>(0.0);
    let n_lights = lights.count;
    for (var i: u32 = 0u; i < n_lights; i = i + 1u) {
        let ev = _viewport_eval_light(lights_storage[i], inp.world_pos);
        let L = ev.l;
        let radiance = ev.radiance;

        var shadow_factor = 1.0;
        if i == 0u {
            shadow_factor = viewport_sample_csm(inp.world_pos, N);
        }
        lo = lo + viewport_brdf_light_contrib(N, V, L, radiance * shadow_factor, inp.albedo,
                                              inp.metallic, roughness, f0);
    }

    return ambient + lo + inp.emissive;
}
