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

// Per-instance data (128 bytes). Shading scalars (PBR terms, ranges, emissive,
// alpha, use_pbr/use_flat, normal_strength) live in material_gpu_buf, read via
// material_id; only per-instance fields ride here. The has_* texture flags stay
// per-instance (the explicit MeshInstanceItem path sets them at upload time).
struct InstanceData {
    model: mat4x4<f32>,                   // offset 0
    colour: vec4<f32>,                    // offset 64
    selected: u32,                        // offset 80
    wireframe: u32,                       // offset 84
    has_texture: u32,                     // offset 88
    has_normal_map: u32,                  // offset 92
    has_ao_map: u32,                      // offset 96
    unlit: u32,                           // offset 100
    receive_shadows: u32,                 // offset 104
    material_id: u32,                     // offset 108
    alpha_cutoff: f32,                    // offset 112
    alpha_flag: u32,                      // offset 116
    has_light_probe: u32,                 // offset 120
    light_probe_index: u32,               // offset 124
    ignore_clip: u32,                     // offset 128
    custom_data_id: u32,                  // offset 132
    backface_pattern_scale: f32,          // offset 136
    _pad2: u32,                           // offset 140
};

// Per-material UV transform block (group 0, binding 21). Slot order: 0 albedo,
// 1 normal, 2 AO, 3 metallic-roughness, 4 emissive. material_id 0 is identity.
struct TexTransform {
    offset_scale: vec4<f32>,   // (offset.x, offset.y, scale.x, scale.y)
    rot_tc: vec4<f32>,         // (rotation_radians, f32(uv_set), 0, 0)
}
// scalars0 = (ambient, diffuse, specular, shininess)
// scalars1 = (metallic, roughness, normal_strength, _)
// scalars2 = (emissive.rgb, ao_min)
// scalars1 = (metallic, roughness, normal_strength, has_mr_tex)
// scalars3 = (ao_max, param_vis_scale, backface_policy, has_emissive_tex)
// flags    = (use_pbr, use_flat, alpha_mode, param_vis_mode)
// backface_colour = styled back-face colour (DiffColour rgb / Tint factor in .r /
//                   Pattern rgb); Pattern world scale is per-instance
// mr_range = (metallic_min, metallic_max, roughness_min, roughness_max)
struct MaterialGpu {
    xf: array<TexTransform, 5>,
    scalars0: vec4<f32>,
    scalars1: vec4<f32>,
    scalars2: vec4<f32>,
    scalars3: vec4<f32>,
    flags: vec4<u32>,
    backface_colour: vec4<f32>,
    mr_range: vec4<f32>,
    tex_index0: vec4<u32>,   // bindless array indices: albedo, normal, ao, metallic-roughness
    tex_index1: vec4<u32>,   // bindless array indices: emissive, unused, unused, unused
}
@group(0) @binding(21) var<storage, read> material_gpu_buf: array<MaterialGpu>;

// Per-instance custom-data payload (group 0, binding 22). Raw [f32; 8] per
// instance, indexed by custom_data_id. Slots 0..3 add to emissive (nits), slot 3
// is pad, and slots 4..8 (`data1`) are the raw material-plugin channel, surfaced
// to a plugin hook as `surf.attr`. custom_data_id 0 is the zero block.
struct CustomData {
    data0: vec4<f32>,   // slots 0..4
    data1: vec4<f32>,   // slots 4..8
}
@group(0) @binding(22) var<storage, read> instance_custom_data_buf: array<CustomData>;

struct SlotUv {
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
}

// See mesh.wgsl:material_slot_uv. uv' = rotate((uv*scale+offset)-0.5, rot)+0.5,
// derivatives rotated by the same angle.
fn material_slot_uv(mid: u32, slot: u32, uv: vec2<f32>, duvdx: vec2<f32>, duvdy: vec2<f32>) -> SlotUv {
    let xf = material_gpu_buf[mid].xf[slot];
    let s = xf.offset_scale.zw;
    let o = xf.offset_scale.xy;
    let rot = xf.rot_tc.x;
    let c = cos(rot);
    let sn = sin(rot);
    let base = uv * s + o - vec2<f32>(0.5, 0.5);
    let dx = duvdx * s;
    let dy = duvdy * s;
    var r: SlotUv;
    r.uv = vec2<f32>(c * base.x - sn * base.y, sn * base.x + c * base.y) + vec2<f32>(0.5, 0.5);
    r.ddx = vec2<f32>(c * dx.x - sn * dx.y, sn * dx.x + c * dx.y);
    r.ddy = vec2<f32>(c * dy.x - sn * dy.y, sn * dy.x + c * dy.y);
    return r;
}

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
@group(1) @binding(6) var                metallic_roughness_tex: texture_2d<f32>;
@group(1) @binding(7) var                emissive_tex:           texture_2d<f32>;

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
    // On the instanced path `attr` carries the per-instance raw material
    // channel: custom_data slots 4..8 (`data1`), the block reserved for material
    // plugins (slots 0..3 add to emissive, slot 3 is pad). A plugin hook that
    // instances reads its per-instance inputs here; `custom_data_id` 0 is the
    // shared all-zero block, so instances that set no custom data read zero.
    surf.attr = instance_custom_data_buf[instances[in.instance_idx].custom_data_id].data1;
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

// Procedural UV parameterisation pattern (mirrors mesh.wgsl:param_vis_colour).
// Replaces the lit colour entirely; driven by the per-material mode/scale.
fn param_vis_colour(uv: vec2<f32>, mode: u32, scale: f32) -> vec3<f32> {
    let col_a      = vec3<f32>(1.0,  1.0,  1.0);
    let col_b      = vec3<f32>(0.0,  0.0,  0.0);
    let line_col   = vec3<f32>(0.0,  0.0,  0.0);
    let bg_col     = vec3<f32>(1.0,  1.0,  1.0);
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

// Material prep for the instanced opaque path. Wireframe and unlit fully
// determine the colour and set `resolved`; otherwise the surface fields feed
// compute_lit.
fn compute_surface(in: VertexOut, is_front: bool) -> Surface {
    let inst = instances[in.instance_idx];
    let mat = material_gpu_buf[inst.material_id];

    var out: Surface;
    out.resolved = false;
    out.out_colour = vec4<f32>(0.0);
    out.base_colour = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.front_facing = select(0u, 1u, is_front);
    out.ao_factor = 1.0;
    out.mat_uv = in.uv;
    out.alpha = 1.0;

    // Screen-space derivatives of the interpolated inputs, taken here where
    // control flow is still uniform. The shading branches below key off
    // per-instance storage (non-uniform), where implicit derivatives are
    // rejected by strict WGSL validators; these feed explicit-gradient sampling.
    let d_uv_dx = dpdx(in.uv);
    let d_uv_dy = dpdy(in.uv);
    let d_wp_dx = dpdx(in.world_pos);
    let d_wp_dy = dpdy(in.world_pos);

    if instances[in.instance_idx].ignore_clip == 0u {
        for (var i = 0u; i < clip_planes.count; i++) {
            let plane = clip_planes.planes[i];
            if dot(in.world_pos, plane.xyz) + plane.w < 0.0 { discard; }
        }
        if !clip_volume_test(in.world_pos) { discard; }
    }

    if inst.wireframe != 0u {
        out.resolved = true;
        out.out_colour = vec4<f32>(0.75, 0.75, 0.75, 1.0);
        return out;
    }

    // Per-material UV transform (slot 0 = albedo; also feeds the plugin surf.uv).
    let s0 = material_slot_uv(inst.material_id, 0u, in.uv, d_uv_dx, d_uv_dy);
    let mat_uv = s0.uv;
    out.mat_uv = mat_uv;
    let muv_ddx = s0.ddx;
    let muv_ddy = s0.ddy;
    let s_normal = material_slot_uv(inst.material_id, 1u, in.uv, d_uv_dx, d_uv_dy);
    let s_ao = material_slot_uv(inst.material_id, 2u, in.uv, d_uv_dx, d_uv_dy);

    var tex_colour = vec4<f32>(1.0);
    if inst.has_texture == 1u { tex_colour = textureSampleGrad(obj_texture, obj_sampler, mat_uv, muv_ddx, muv_ddy); }
    let obj_colour = vec4<f32>(inst.colour.rgb * in.colour.rgb * tex_colour.rgb,
                               inst.colour.a   * in.colour.a   * tex_colour.a);
    out.alpha = obj_colour.a;

    // Alpha MASK: discard fragments whose albedo alpha is below the cutoff.
    if inst.alpha_flag == 1u && inst.has_texture == 1u && obj_colour.a < inst.alpha_cutoff {
        discard;
    }

    var base_colour = obj_colour.rgb;

    // Unlit: skip all lighting, return raw colour directly.
    if inst.unlit != 0u {
        out.resolved = true;
        out.out_colour = vec4<f32>(base_colour, obj_colour.a);
        return out;
    }

    // UV parameterisation visualisation: procedural pattern replaces all lighting.
    // Mode in flags.w, tile scale in scalars3.y (per-material). Uses the raw mesh
    // UV, matching the per-object path.
    if mat.flags.w != 0u {
        let vis = param_vis_colour(in.uv, mat.flags.w, mat.scalars3.y);
        out.resolved = true;
        out.out_colour = vec4<f32>(vis, obj_colour.a);
        return out;
    }

    var N: vec3<f32>;
    if mat.flags.y != 0u {
        let dpx = d_wp_dx;
        let dpy = d_wp_dy;
        var Nf = normalize(cross(dpx, dpy));
        if dot(Nf, in.world_normal) < 0.0 { Nf = -Nf; }
        N = Nf;
    } else if inst.has_normal_map != 0u {
        let nm_sample = textureSampleGrad(normal_map, obj_sampler, s_normal.uv, s_normal.ddx, s_normal.ddy).rgb;
        var ts_unpacked = nm_sample * 2.0 - vec3<f32>(1.0);
        ts_unpacked.x = ts_unpacked.x * mat.scalars1.z;
        ts_unpacked.y = ts_unpacked.y * mat.scalars1.z;
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

    // Styled back-face policy: flip the normal and override the colour on back
    // faces. Mirrors mesh.wgsl. Policy in scalars3.z: 2 DifferentColour,
    // 3 Tint, 4..7 Pattern. Cull (0) and Identical (1) do not enter here. The
    // Pattern world scale is per-instance (`inst.backface_pattern_scale`).
    let backface_policy = u32(mat.scalars3.z);
    if !is_front && backface_policy >= 2u {
        N = -N;
        if backface_policy == 2u {
            base_colour = mat.backface_colour.rgb;
        } else if backface_policy == 3u {
            base_colour = base_colour * (1.0 - mat.backface_colour.r);
        } else {
            let pattern_colour = mat.backface_colour.rgb;
            let pattern_type = backface_policy - 4u;
            let wp = in.world_pos * inst.backface_pattern_scale;
            var use_pattern = false;
            if pattern_type == 0u {
                let p = (i32(floor(wp.x)) + i32(floor(wp.z))) & 1;
                use_pattern = p != 0;
            } else if pattern_type == 1u {
                use_pattern = fract((wp.x + wp.z) * 0.5) < 0.4;
            } else if pattern_type == 2u {
                use_pattern = fract((wp.x + wp.z) * 0.5) < 0.3 || fract((wp.x - wp.z) * 0.5) < 0.3;
            } else {
                use_pattern = fract(wp.z * 0.5) < 0.4;
            }
            base_colour = select(base_colour, pattern_colour, use_pattern);
        }
    }

    var ao_factor = 1.0;
    if inst.has_ao_map != 0u {
        let raw_ao = textureSampleGrad(ao_map, obj_sampler, s_ao.uv, s_ao.ddx, s_ao.ddy).r;
        ao_factor = mix(vec2<f32>(mat.scalars2.w, mat.scalars3.x).x, vec2<f32>(mat.scalars2.w, mat.scalars3.x).y, raw_ao);
    }

    out.base_colour = base_colour;
    out.normal = N;
    out.ao_factor = ao_factor;
    return out;
}

// Lighting for the instanced opaque path. Samples shadows like the per-object path.
fn compute_lit(surface: Surface, in: VertexOut, saa_kernel: f32, refl_dr: f32) -> LitResult {
    let inst = instances[in.instance_idx];
    let mat = material_gpu_buf[inst.material_id];
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

    // Metallic-roughness texture (slot 3), sampled here in uniform control flow
    // (the PBR branch below is per-instance / non-uniform). The fallback view is a
    // 1x1 texture, so sampling unconditionally is cheap; the result is only used
    // when `has_mr_tex` is set.
    let d_uv_dx = dpdx(in.uv);
    let d_uv_dy = dpdy(in.uv);
    let s_mr = material_slot_uv(inst.material_id, 3u, in.uv, d_uv_dx, d_uv_dy);
    let mr_sample = textureSampleGrad(metallic_roughness_tex, obj_sampler, s_mr.uv, s_mr.ddx, s_mr.ddy);

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

    if mat.flags.x != 0u {
        var metallic  = clamp(mat.scalars1.x,  0.0, 1.0);
        var roughness = max(mat.scalars1.y, 0.04);
        // glTF ORM texture: G=roughness, B=metallic. `mr_range` remaps the raw
        // sample before the scalar factor, then specular-AA, matching mesh.wgsl.
        if mat.scalars1.w != 0.0 {
            let m_remapped = mix(mat.mr_range.x, mat.mr_range.y, mr_sample.b);
            let r_remapped = mix(mat.mr_range.z, mat.mr_range.w, mr_sample.g);
            metallic  = clamp(m_remapped * metallic,  0.0, 1.0);
            roughness = max(r_remapped * roughness, 0.04);
        }
        roughness = specular_aa_roughness_kernel(roughness, saa_kernel);
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
            let ambient_scale = vec3<f32>(mat.scalars0.x) + hemi_colour * lights_uniform.hemisphere_intensity;
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
            let diffuse_contrib  = mat.scalars0.y  * n_dot_l * shadow * INV_PI;
            let specular_contrib = mat.scalars0.z * pow(n_dot_h, mat.scalars0.w)
                                 * (mat.scalars0.w + 8.0) * INV_PI * 0.125 * shadow;
            total_colour_contrib += (diffuse_contrib + specular_contrib) * ev.radiance;
        }
        let ambient_contrib = mat.scalars0.x;
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
fn fs_main(in: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let surface = compute_surface(in, is_front);

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
    // The emissive factor is modulated by the emissive texture (slot 4) when set,
    // matching mesh.wgsl. Per-instance custom data slots 0..3 then add to emissive
    // (nits); zero is a no-op.
    let e_inst = instances[in.instance_idx];
    let e_mat = material_gpu_buf[e_inst.material_id];
    var emissive = e_mat.scalars2.xyz;
    if e_mat.scalars3.w != 0.0 {
        let d_uv_dx = dpdx(in.uv);
        let d_uv_dy = dpdy(in.uv);
        let s_em = material_slot_uv(e_inst.material_id, 4u, in.uv, d_uv_dx, d_uv_dy);
        emissive = emissive * textureSampleGrad(emissive_tex, obj_sampler, s_em.uv, s_em.ddx, s_em.ddy).rgb;
    }
    let inst_emissive = instance_custom_data_buf[e_inst.custom_data_id].data0.xyz;
    emissive = emissive + inst_emissive;
    final_rgb += emissive;
    let dbg_emissive_lum = dot(emissive, vec3<f32>(0.2126, 0.7152, 0.0722));

    // #include "helpers/debug_vis.wgsl"

    return vec4<f32>(final_rgb, surface.alpha);
}
