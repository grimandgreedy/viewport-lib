// Instanced mesh shader for the 3D viewport.
//
// Same lighting model as mesh.wgsl but reads per-instance data from a
// storage buffer indexed by @builtin(instance_index) instead of a uniform.
//
// Group 0: Camera + shadow atlas + lights + clip planes + shadow info (unchanged from mesh.wgsl).
// Group 1: Storage buffer containing array<InstanceData> (binding 0)
//          + Albedo texture (binding 1) + sampler (binding 2)
//          + normal map (binding 3) + AO map (binding 4).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    // Upper bound on the lit (pre-emissive) colour: 1.0 on the LDR path,
    // F16_MAX on the HDR path so lit output can exceed 1.0 into the
    // Rgba16Float target ahead of tone mapping.
    lit_clamp: f32,
    forward: vec3<f32>,
    _pad1: f32,
    inv_view_proj: mat4x4<f32>,
};

// Shared light struct definitions and `lights_storage` binding 13 of group 0.
// #include "helpers/scene_lighting.wgsl"

// Frozen fragment-shading hook structs (ShadingSurface, LightSample).
// #include "helpers/shade.wgsl"

// Per-vertex deformation hook contract.
// #include "helpers/deform.wgsl"

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
    emissive: vec3<f32>,                  // self-illumination added after lighting
    _pad_emissive: f32,
    has_light_probe: u32,                 // 1 = sample light_probe_sh for indirect diffuse
    light_probe_index: u32,               // base block index into light_probe_sh
    _pad_lp: vec2<u32>,
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
@group(0) @binding(7) var ibl_irradiance: texture_2d_array<f32>;
@group(0) @binding(8) var ibl_prefiltered: texture_2d_array<f32>;
@group(0) @binding(9) var ibl_brdf_lut: texture_2d<f32>;
@group(0) @binding(10) var ibl_sampler: sampler;
@group(0) @binding(11) var ibl_skybox: texture_2d<f32>;
@group(0) @binding(12) var<storage, read_write> debug_frag_buf: array<vec4<f32>>;

// #include "helpers/clip_volume_test.wgsl"
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

// GPU-driven cull variant: `idx` is the visible-slot index written by the
// compute cull pass. Look up the actual instance index via visibility_indices,
// then run the same transform as vs_main.
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

// ---------------------------------------------------------------------------
// Poisson disk + CSM shadow sampling (mirrors mesh.wgsl)
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
    return textureSampleCompareLevel(
        point_shadow_cube_tex,
        shadow_sampler,
        dir,
        light.point_shadow_slot,
        normalised - bias,
    );
}

// #include "helpers/csm.wgsl"

// ---------------------------------------------------------------------------
// PBR BRDF helpers (Cook-Torrance) : mirrors mesh.wgsl
// ---------------------------------------------------------------------------

// Shared direct BRDF: D_GGX, G1_Smith, G_Smith, F_Schlick, pbr_light_contrib.
// #include "helpers/brdf.wgsl"

// Shared ambient / IBL helpers (equirect sampling, split-sum ambient).
// #include "helpers/ambient.wgsl"

struct Surface {
    resolved: bool,
    out_colour: vec4<f32>,
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

// Material prep for the instanced opaque path. Wireframe and unlit fully
// determine the colour and set `resolved`; otherwise the surface fields feed
// compute_lit.
fn compute_surface(in: VertexOut) -> Surface {
    let inst = instances[in.instance_idx];

    var out: Surface;
    out.resolved = false;
    out.out_colour = vec4<f32>(0.0);
    out.base_colour = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.ao_factor = 1.0;
    out.mat_uv = in.uv;
    out.alpha = 1.0;
    out.front_facing = 1u;

    // Screen-space derivatives of the interpolated inputs, taken here where
    // control flow is still uniform. The shading branches below key off
    // per-instance storage (non-uniform), where implicit derivatives are
    // rejected by strict WGSL validators; these feed explicit-gradient sampling.
    let d_uv_dx = dpdx(in.uv);
    let d_uv_dy = dpdy(in.uv);
    let d_wp_dx = dpdx(in.world_pos);
    let d_wp_dy = dpdy(in.world_pos);

    for (var i = 0u; i < clip_planes.count; i++) {
        let plane = clip_planes.planes[i];
        if dot(in.world_pos, plane.xyz) + plane.w < 0.0 { discard; }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    if inst.wireframe != 0u {
        out.resolved = true;
        out.out_colour = vec4<f32>(0.75, 0.75, 0.75, 1.0);
        return out;
    }

    let mat_uv = in.uv * inst.uv_transform.zw + inst.uv_transform.xy;
    out.mat_uv = mat_uv;
    let muv_ddx = d_uv_dx * inst.uv_transform.zw;
    let muv_ddy = d_uv_dy * inst.uv_transform.zw;

    var tex_colour = vec4<f32>(1.0);
    if inst.has_texture == 1u { tex_colour = textureSampleGrad(obj_texture, obj_sampler, mat_uv, muv_ddx, muv_ddy); }
    let obj_colour = vec4<f32>(inst.colour.rgb * in.colour.rgb * tex_colour.rgb,
                               inst.colour.a   * in.colour.a   * tex_colour.a);
    out.alpha = obj_colour.a;

    // Alpha MASK: discard fragments whose albedo alpha is below the cutoff.
    if inst.alpha_flag == 1u && inst.has_texture == 1u && obj_colour.a < inst.alpha_cutoff {
        discard;
    }

    let base_colour = obj_colour.rgb;

    // Unlit: skip all lighting, return raw colour directly.
    if inst.unlit != 0u {
        out.resolved = true;
        out.out_colour = vec4<f32>(base_colour, obj_colour.a);
        return out;
    }

    var N: vec3<f32>;
    if inst.use_flat != 0u {
        let dpx = d_wp_dx;
        let dpy = d_wp_dy;
        var Nf = normalize(cross(dpx, dpy));
        if dot(Nf, in.world_normal) < 0.0 { Nf = -Nf; }
        N = Nf;
    } else if inst.has_normal_map != 0u {
        let nm_sample = textureSampleGrad(normal_map, obj_sampler, mat_uv, muv_ddx, muv_ddy).rgb;
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
        let raw_ao = textureSampleGrad(ao_map, obj_sampler, mat_uv, muv_ddx, muv_ddy).r;
        ao_factor = mix(inst.ao_range.x, inst.ao_range.y, raw_ao);
    }

    out.base_colour = base_colour;
    out.normal = N;
    out.ao_factor = ao_factor;
    return out;
}

// Lighting for the instanced opaque path. Samples shadows like the per-object path.
fn compute_lit(surface: Surface, in: VertexOut, saa_kernel: f32, refl_dr: f32) -> LitResult {
    let inst = instances[in.instance_idx];
    var base_colour = surface.base_colour;
    let ao_factor = surface.ao_factor;
    var N = surface.normal;

    // Use the smooth vertex normal for shadow bias (see mesh.wgsl for rationale).
    let shadow_normal = N;

    let V = normalize(camera.eye_pos - in.world_pos);

    // `saa_kernel` (geometric specular AA) and `refl_dr` (IBL reflection
    // footprint) are supplied by the caller, computed in uniform control flow.
    // The PBR block below is gated on per-instance data, so evaluating the
    // underlying derivatives here would violate WGSL uniformity.

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

    if inst.use_pbr != 0u {
        var metallic  = clamp(inst.metallic,  0.0, 1.0);
        var roughness = specular_aa_roughness_kernel(max(inst.roughness, 0.04), saa_kernel);
        var F0 = mix(vec3<f32>(0.04), base_colour, metallic);
        // Plugin shading hooks: the composer fills the shade-slot regions in
        // plugin-composed modules; in the base module they are inert comments.
        // <viewport-shade-slot:surface>
        // </viewport-shade-slot:surface>
        var Lo = vec3<f32>(0.0);
        // Per-cluster light list (small-N scenes fall back to the full
        // array inside cluster_light_range).
        let pbr_range = cluster_light_range(in.world_pos, lights_uniform.count);
        for (var j = 0u; j < pbr_range.count; j++) {
            let i = cluster_light_global(pbr_range, j);
            let l = lights_storage[i];
            let ev = eval_light(l, in.world_pos);
            if !ev.in_range { continue; }
            let L = ev.l;
            var radiance = ev.radiance;
            // Backfacing: pbr_light_contrib returns exactly zero, so skip
            // the shadow samples and the BRDF outright. Plugin modules whose
            // hook wants the back hemisphere empty this region instead.
            // <viewport-shade-slot:backface-cull>
            if dot(N, L) <= 0.0 { continue; }
            // </viewport-shade-slot:backface-cull>
            // <viewport-shade-slot:shadow>
            var shadow_factor = 1.0;
            if lights_uniform.shadows_enabled != 0u && inst.receive_shadows != 0u {
                if i == 0u && lights_storage[0].light_type != 1u {
                    last_shadow_sample = sample_shadow_csm(in.world_pos, camera.eye_pos, shadow_normal, L, 0u);
                    shadow_factor = last_shadow_sample.factor;
                } else if l.light_type == 1u && l.point_shadow_slot >= 0 {
                    shadow_factor = sample_point_shadow(l, in.world_pos);
                }
            }
            // </viewport-shade-slot:shadow>
            // <viewport-shade-slot:light>
            radiance *= shadow_factor;
            Lo += pbr_light_contrib(N, V, L, radiance, base_colour, metallic, roughness, F0);
            // </viewport-shade-slot:light>
        }
        dbg_direct_lum = dot(Lo, lum_weights);
        dbg_roughness  = roughness;
        dbg_metallic   = metallic;
        // <viewport-shade-slot:ambient>
        var ambient: vec3<f32>;
        if lights_uniform.ibl_enabled != 0u {
            var ibl: IblContrib;
            if lights_uniform.env_zone_count != 0u {
                ibl = ibl_ambient_zoned(N, V, base_colour, metallic, roughness, F0,
                                        ao_factor, lights_uniform.ibl_intensity,
                                        lights_uniform.ibl_rotation, refl_dr, in.world_pos,
                                        lights_uniform.env_zone_count);
            } else {
                ibl = ibl_ambient_grad(N, V, base_colour, metallic, roughness, F0,
                                       ao_factor, lights_uniform.ibl_intensity,
                                       lights_uniform.ibl_rotation, refl_dr);
            }
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
        // Light-probe instances take their indirect diffuse from the SH field
        // sampled at the object position, replacing the global-IBL / hemisphere
        // diffuse above. SH probes carry diffuse only, so IBL specular is not
        // added here.
        if inst.has_light_probe != 0u {
            ambient = evaluate_object_indirect(inst.light_probe_index, in.world_pos, N) * base_colour * ao_factor;
            dbg_ambient_lum = dot(ambient, lum_weights);
        }
        // </viewport-shade-slot:ambient>
        final_rgb = clamp((Lo + ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(camera.lit_clamp));
        // <viewport-shade-slot:recolor>
        // </viewport-shade-slot:recolor>
        // BEGIN_PBR_STRIP
    } else {
        var total_colour_contrib = vec3<f32>(0.0);
        let bp_range = cluster_light_range(in.world_pos, lights_uniform.count);
        for (var j = 0u; j < bp_range.count; j++) {
            let i = cluster_light_global(bp_range, j);
            let l = lights_storage[i];
            let ev = eval_light(l, in.world_pos);
            if !ev.in_range { continue; }
            let light_dir = ev.l;
            var shadow = 1.0;
            if lights_uniform.shadows_enabled != 0u && inst.receive_shadows != 0u {
                if i == 0u && lights_storage[0].light_type != 1u {
                    last_shadow_sample = sample_shadow_csm(in.world_pos, camera.eye_pos, shadow_normal, light_dir, 0u);
                    shadow = last_shadow_sample.factor;
                } else if l.light_type == 1u && l.point_shadow_slot >= 0 {
                    shadow = sample_point_shadow(l, in.world_pos);
                }
            }
            let H = normalize(light_dir + V);
            let n_dot_l = max(dot(N, light_dir), 0.0);
            let n_dot_h = max(dot(N, H), 0.0);
            // Energy-normalised Blinn-Phong (matches the PBR path): 1/pi on the
            // diffuse lobe, (shininess + 8) / (8 pi) on the specular lobe.
            let diffuse_contrib  = inst.diffuse  * n_dot_l * shadow * INV_PI;
            let specular_contrib = inst.specular * pow(n_dot_h, inst.shininess)
                                 * (inst.shininess + 8.0) * INV_PI * 0.125 * shadow;
            total_colour_contrib += (diffuse_contrib + specular_contrib) * ev.radiance;
        }
        let ambient_contrib = inst.ambient;
        let hemi_t = clamp(in.world_normal.z * 0.5 + 0.5, 0.0, 1.0);
        let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
        let hemi_ambient = hemi_colour * lights_uniform.hemisphere_intensity;
        let direct_rgb = base_colour * total_colour_contrib;
        dbg_direct_lum  = dot(direct_rgb, lum_weights);
        var hemi_rgb = base_colour * (ambient_contrib + hemi_ambient) * ao_factor;
        if inst.has_light_probe != 0u {
            hemi_rgb = evaluate_object_indirect(inst.light_probe_index, in.world_pos, N) * base_colour * ao_factor;
        }
        dbg_ambient_lum = dot(hemi_rgb, lum_weights);
        let lit_rgb = hemi_rgb + direct_rgb;
        final_rgb = clamp(lit_rgb * tint.rgb, vec3<f32>(0.0), vec3<f32>(camera.lit_clamp));
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
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let surface = compute_surface(in);

    // Derivative terms for the lighting stage, taken here while control flow is
    // still uniform (before the resolved early return and compute_lit's
    // per-instance branches). Exact: uses the resolved shading normal.
    let d_n_dx = dpdx(surface.normal);
    let d_n_dy = dpdy(surface.normal);
    let saa_kernel = min(0.5 * (dot(d_n_dx, d_n_dx) + dot(d_n_dy, d_n_dy)), 0.18);
    let V_dr = normalize(camera.eye_pos - in.world_pos);
    let R_dr = reflect(-V_dr, surface.normal);
    let refl_dr = max(length(dpdx(R_dr)), length(dpdy(R_dr)));

    if surface.resolved {
        return surface.out_colour;
    }

    let lit = compute_lit(surface, in, saa_kernel, refl_dr);

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
    var final_rgb = lit.rgb;

    // Emissive term: added after lighting so it can push HDR values above 1.0.
    // The instanced path has no emissive texture, so the factor is applied flat.
    let emissive = instances[in.instance_idx].emissive;
    final_rgb += emissive;
    let dbg_emissive_lum = dot(emissive, vec3<f32>(0.2126, 0.7152, 0.0722));

    // #include "helpers/debug_vis.wgsl"

    return vec4<f32>(final_rgb, surface.alpha);
}
