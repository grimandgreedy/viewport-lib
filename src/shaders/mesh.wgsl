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
    // Upper bound on the lit (pre-emissive) colour: 1.0 on the LDR path,
    // F16_MAX on the HDR path so lit output can exceed 1.0 into the
    // Rgba16Float target ahead of tone mapping.
    lit_clamp: f32,
    forward: vec3<f32>,
    _pad1: f32,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
};

// Shared light struct definitions and `lights_storage` binding 13 of group 0.
// #include "helpers/scene_lighting.wgsl"

// Frozen fragment-shading hook structs (ShadingSurface, LightSample).
// #include "helpers/shade.wgsl"

// Per-vertex deformation hook contract.
// #include "helpers/deform.wgsl"

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

// Per-object uniform. Layout mirrors the renderer's `ObjectUniform`
// (320 bytes); keep the field offsets in sync with its doc table.
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
    position_override_base: u32,           // offset 304 : first vec3 element read from binding 13 (pool slicing)
    position_override_len: u32,            // offset 308 : element count of the window; 0xffffffff = whole buffer
    normal_override_base: u32,             // offset 312 : same for binding 14
    normal_override_len: u32,              // offset 316
    has_light_probe: u32,                  // offset 320 : 1 = sample light_probe_sh for indirect diffuse
    light_probe_index: u32,                // offset 324 : base block index into light_probe_sh
    lightmap_mode: u32,                    // offset 328 : 0 none, 1 Replace, 2 Add, 3 AmbientOcclusion
    lightmap_directional: u32,             // offset 332 : 1 = sample the binding-18 direction atlas
    lightmap_scale_bias: vec4<f32>,        // offset 336 : lm_uv = uv1 * .xy + .zw (scene atlas sub-rect)
    lightmap_index: u32,                   // offset 352 : base atlas layer, added to the per-vertex page
    // The vec4 above makes the struct 16-aligned, so WGSL rounds its size up to
    // 368 to match the Rust ObjectUniform (which pads with a trailing [u32; 3]).
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
// Per-vertex vec4 sidecar (binding 15). Material-plugin hooks read it as their
// vertex attribute; a baked lightmap reuses the same slot to carry UV1 in .xy
// (the two are mutually exclusive per mesh). The shared zero fallback is bound
// when neither is set. The Metal vertex buffer table is full at this slot, so
// the lightmap rides it rather than taking a new binding.
@group(1) @binding(15) var<storage, read> extension_attr_buffer: array<vec4<f32>>;
// Baked lightmap texture, sampled with the binding-2 material sampler and gated
// on object.lightmap_mode. Textures use a separate Metal table, so this is a
// fresh binding. The 1x1 fallback is bound when no lightmap is set.
@group(1) @binding(17) var lightmap_tex: texture_2d_array<f32>;
// Dominant-direction atlas for a directional lightmap: xyz = unit incoming
// direction (world space), w = directionality. Gated on object.lightmap_directional;
// the 1x1 fallback is bound (and ignored) for flat lightmaps.
@group(1) @binding(18) var lightmap_dir_tex: texture_2d_array<f32>;

// Directional-lightmap response: scale the baked radiance by how the shading
// normal (post normal-map) faces the baked dominant light, relative to the
// geometric normal the bake integrated against. Returns 1 when they match (flat
// or non-normal-mapped surface -> baked value reproduced exactly), and diverges
// only where a normal map perturbs the normal, weighted by directionality (w).
fn lightmap_directional_factor(uv: vec2<f32>, page: i32, n_pix: vec3<f32>, n_geo: vec3<f32>) -> f32 {
    let d = textureSampleLevel(lightmap_dir_tex, obj_sampler, uv, page, 0.0);
    let ndl_pix = max(dot(normalize(n_pix), d.xyz), 0.0);
    let ndl_geo = max(dot(normalize(n_geo), d.xyz), 0.0);
    return mix(1.0, ndl_pix / max(ndl_geo, 0.1), d.w);
}

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
    // Baked lightmap UV1, interpolated for the fragment lightmap sample.
    @location(9) lightmap_uv:     vec2<f32>,
    // Atlas page (array layer) for a multi-page lightmap, carried in UV1.z. Flat:
    // every vertex of a chart shares one page, so no interpolation is wanted.
    @location(10) @interpolate(flat) lightmap_page: f32,
    // Plugin vertex-attribute varying: the composer adds a @location(8)
    // member here for hooks that read the per-vertex extension attribute.
    // <viewport-shade-slot:vertex-out>
    // </viewport-shade-slot:vertex-out>
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    // Override > vertex attribute. When a plugin has bound a per-vertex position
    // storage buffer, replace `in.position` outright; warp is then layered on top
    // additively. Same idea for normals further down.
    var local_pos = in.position;
    if object.has_position_override != 0u && in.vertex_index < object.position_override_len {
        // base/len slice a window out of a pooled buffer; base = 0 and
        // len = 0xffffffff reproduce the whole-buffer behaviour.
        let pi = (object.position_override_base + in.vertex_index) * 3u;
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
    if object.has_normal_override != 0u && in.vertex_index < object.normal_override_len {
        let ni = (object.normal_override_base + in.vertex_index) * 3u;
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
    // Lightmap UV1 rides the vec4 sidecar's xy (zero for non-lightmapped meshes).
    // .z carries the atlas page for a multi-page lightmap (0 otherwise).
    let lm_len = arrayLength(&extension_attr_buffer);
    let lm_sidecar = extension_attr_buffer[min(idx, max(lm_len, 1u) - 1u)];
    out.lightmap_uv = lm_sidecar.xy;
    out.lightmap_page = lm_sidecar.z;
    // <viewport-shade-slot:vertex-fetch>
    // </viewport-shade-slot:vertex-fetch>
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

// #include "helpers/csm.wgsl"

// ---------------------------------------------------------------------------
// PBR BRDF helpers (Cook-Torrance)
// ---------------------------------------------------------------------------

// Shared direct BRDF: D_GGX, G1_Smith, G_Smith, F_Schlick, pbr_light_contrib.
// #include "helpers/brdf.wgsl"

// ---------------------------------------------------------------------------
// IBL helpers : equirectangular sampling
// ---------------------------------------------------------------------------

// Shared ambient / IBL helpers (equirect sampling, split-sum ambient).
// #include "helpers/ambient.wgsl"

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
    out.front_facing = select(0u, 1u, is_front);

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
    var base_colour = surface.base_colour;
    let ao_factor = surface.ao_factor;
    let mat_uv = surface.mat_uv;
    var N = surface.normal;

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
        roughness = specular_aa_roughness(N, roughness);
        var F0 = mix(vec3<f32>(0.04), base_colour, metallic);

        // Plugin shading hooks: the composer fills the shade-slot regions in
        // plugin-composed modules; in the base module they are inert comments.
        // See docs/issues/lighting-shader-injection-seam.md for the contract.
        // <viewport-shade-slot:surface>
        // </viewport-shade-slot:surface>

        var Lo = vec3<f32>(0.0);
        let pbr_range = cluster_light_range(in.world_pos, lights_uniform.count);
        for (var j: u32 = 0u; j < pbr_range.count; j = j + 1u) {
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

            // Shadow factor. Directional `lights[0]` uses CSM; point lights
            // with an allocated cubemap slot sample the point shadow array.
            // Spot lights keep their existing single-face perspective path
            // (sampled inside `sample_shadow_csm` when lights[0] is a spot;
            // unshadowed otherwise). Per-receiver opt-out via
            // ItemSettings.receive_shadows skips the sample.
            // <viewport-shade-slot:shadow>
            var shadow_factor = 1.0;
            if lights_uniform.shadows_enabled != 0u && object.receive_shadows != 0u {
                if i == 0u && lights_storage[0].light_type != 1u {
                    last_shadow_sample = sample_shadow_csm(in.world_pos, camera.eye_pos, shadow_normal, L, select(0u, 1u, object.backface_policy != 0u));
                    shadow_factor = last_shadow_sample.factor;
                } else if l.light_type == 1u && l.point_shadow_slot >= 0 {
                    shadow_factor = sample_point_shadow(l, in.world_pos);
                }
            }
            // </viewport-shade-slot:shadow>
            // <viewport-shade-slot:light>
            radiance *= shadow_factor;
            // Subtractive lightmap: the main directional light (index 0) is baked
            // into the lightmap, so suppress its realtime direct here to avoid
            // double counting. Its shadow (sampled above) still darkens the baked
            // term in apply_lightmap below.
            if object.lightmap_mode == 4u && i == 0u {
                radiance = vec3<f32>(0.0);
            }
            Lo += pbr_light_contrib(N, V, L, radiance, base_colour,
                                    metallic, roughness, F0);
            // </viewport-shade-slot:light>
        }

        dbg_direct_lum = dot(Lo, lum_weights);
        dbg_roughness  = roughness;
        dbg_metallic   = metallic;

        // Ambient: IBL when enabled, hemisphere fallback otherwise.
        // <viewport-shade-slot:ambient>
        var ambient: vec3<f32>;
        if lights_uniform.ibl_enabled != 0u {
            var ibl: IblContrib;
            if lights_uniform.env_zone_count != 0u {
                // Per-fragment environment selection (uniform-gated branch, so
                // the reflection derivative is valid here).
                let refl_z = reflect(-V, N);
                let dr_z = max(length(dpdx(refl_z)), length(dpdy(refl_z)));
                ibl = ibl_ambient_zoned(N, V, base_colour, metallic, roughness, F0,
                                        ao_factor, lights_uniform.ibl_intensity,
                                        lights_uniform.ibl_rotation, dr_z, in.world_pos,
                                        lights_uniform.env_zone_count);
            } else {
                ibl = ibl_ambient(N, V, base_colour, metallic, roughness, F0,
                                  ao_factor, lights_uniform.ibl_intensity,
                                  lights_uniform.ibl_rotation);
            }
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
        // Light-probe objects take their indirect diffuse from the SH field
        // sampled at the object position, replacing the global-IBL / hemisphere
        // diffuse above. SH probes carry diffuse only, so IBL specular is not
        // added here.
        if object.has_light_probe != 0u {
            ambient = evaluate_sh_probe(object.light_probe_index, N) * base_colour * ao_factor;
            dbg_ambient_lum = dot(ambient, lum_weights);
        }
        // Baked lightmap: replace, add, or occlude the ambient term. Sample the
        // base mip explicitly: a lightmap packs charts tightly, so UV
        // derivatives across a chart are large and derivative-based mip
        // selection would snap to a coarse level and show the atlas texels as
        // blocky bands on the mesh. The atlas has no detail to lose at range.
        if object.lightmap_mode != 0u {
            // Scene-atlas placement: map the object's [0,1] unwrap into its sub-rect
            // and add its base layer to the per-vertex page. Identity scale/bias and
            // layer 0 reproduce a per-mesh (single- or multi-page) lightmap exactly.
            let lm_uv = in.lightmap_uv * object.lightmap_scale_bias.xy + object.lightmap_scale_bias.zw;
            let lm_page = i32(object.lightmap_index) + i32(round(in.lightmap_page));
            var lm = textureSampleLevel(lightmap_tex, obj_sampler, lm_uv, lm_page, 0.0);
            if object.lightmap_directional == 1u {
                let f = lightmap_directional_factor(lm_uv, lm_page, N, in.world_normal);
                lm = vec4<f32>(lm.rgb * f, lm.a);
            }
            ambient = apply_lightmap(ambient, base_colour, ao_factor, lm, object.lightmap_mode, last_shadow_sample.factor);
            dbg_ambient_lum = dot(ambient, lum_weights);
        }
        // </viewport-shade-slot:ambient>

        final_rgb = clamp((Lo + ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(camera.lit_clamp));
        // <viewport-shade-slot:recolor>
        // </viewport-shade-slot:recolor>
        // BEGIN_PBR_STRIP
    } else {
        // Multi-light Blinn-Phong path
        var total_colour_contrib = vec3<f32>(0.0);

        let bp_range = cluster_light_range(in.world_pos, lights_uniform.count);
        for (var j: u32 = 0u; j < bp_range.count; j = j + 1u) {
            let i = cluster_light_global(bp_range, j);
            let l = lights_storage[i];
            let ev = eval_light(l, in.world_pos);
            if !ev.in_range { continue; }
            let light_dir = ev.l;

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

            let diffuse_contrib  = object.diffuse  * n_dot_l * shadow;
            let specular_contrib = object.specular * pow(n_dot_h, object.shininess) * shadow;

            // Subtractive lightmap: the main directional light is baked in, so skip
            // its realtime direct (its shadow still darkens the baked term below).
            if !(object.lightmap_mode == 4u && i == 0u) {
                total_colour_contrib += (diffuse_contrib + specular_contrib) * ev.radiance;
            }
        }

        let ambient_contrib = object.ambient;
        let hemi_t = clamp(in.world_normal.z * 0.5 + 0.5, 0.0, 1.0);
        let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
        let hemi_ambient = hemi_colour * lights_uniform.hemisphere_intensity;

        let direct_rgb = base_colour * total_colour_contrib;
        dbg_direct_lum  = dot(direct_rgb, lum_weights);
        var hemi_rgb = base_colour * (ambient_contrib + hemi_ambient) * ao_factor;
        // Light-probe objects take their indirect diffuse from the SH field.
        if object.has_light_probe != 0u {
            hemi_rgb = evaluate_sh_probe(object.light_probe_index, N) * base_colour * ao_factor;
        }
        // Baked lightmap: replace, add, or occlude the ambient term. Sample the
        // base mip explicitly (see the ambient branch above): derivative-based
        // mip selection over tightly-packed atlas charts reads as blocky bands.
        if object.lightmap_mode != 0u {
            // Scene-atlas placement: map the object's [0,1] unwrap into its sub-rect
            // and add its base layer to the per-vertex page. Identity scale/bias and
            // layer 0 reproduce a per-mesh (single- or multi-page) lightmap exactly.
            let lm_uv = in.lightmap_uv * object.lightmap_scale_bias.xy + object.lightmap_scale_bias.zw;
            let lm_page = i32(object.lightmap_index) + i32(round(in.lightmap_page));
            var lm = textureSampleLevel(lightmap_tex, obj_sampler, lm_uv, lm_page, 0.0);
            if object.lightmap_directional == 1u {
                let f = lightmap_directional_factor(lm_uv, lm_page, N, in.world_normal);
                lm = vec4<f32>(lm.rgb * f, lm.a);
            }
            hemi_rgb = apply_lightmap(hemi_rgb, base_colour, ao_factor, lm, object.lightmap_mode, last_shadow_sample.factor);
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
    // Surface-hook emissive: the composer adds the hook's emissive term here
    // in plugin-composed modules; inert in the base module.
    // <viewport-shade-slot:emissive>
    // </viewport-shade-slot:emissive>

    // #include "helpers/debug_vis.wgsl"

    var final_alpha = surface.alpha;
    // Surface-hook alpha: honoured only under Mask/Blend alpha modes; the
    // composer fills this in plugin-composed modules.
    // <viewport-shade-slot:alpha>
    // </viewport-shade-slot:alpha>
    return vec4<f32>(final_rgb, final_alpha);
}
