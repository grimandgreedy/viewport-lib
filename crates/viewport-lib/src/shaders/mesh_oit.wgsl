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

// #include "helpers/instance_data.wgsl"

// The per-draw view the shader body reads: assembled by `load_object` from
// the instance record and its material block. Not a GPU layout.
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
    material_id: u32,                      // offset 256 : index into material_gpu_buf
    uv1_base: u32,                         // offset 260 : mesh base_vertex for the uv1 buffer
    _pad_uv1: u32,                         // offset 264
    _pad_uv2: u32,                         // offset 268
    deform_flags: u32,                     // offset 272 : bit i set when deformer slot i is active for this draw
    normal_strength: f32,                  // offset 276 : scales tangent normal XY (also aligns next vec2)
    ao_range: vec2<f32>,                   // offset 280 : (min, max) remap of AO map R sample
    metallic_range: vec2<f32>,             // offset 288 : (min, max) remap of MR texture B channel
    roughness_range: vec2<f32>,            // offset 296 : (min, max) remap of MR texture G channel
    _pad_overrides: vec4<u32>,             // offset 304 : unused; the override windows come from the sidecar record
    has_light_probe: u32,                  // offset 320 : 1 = sample light_probe_sh for indirect diffuse
    light_probe_index: u32,                // offset 324 : base block index into light_probe_sh
    lightmap_mode: u32,                    // offset 328 : 0 none, 1 Replace, 2 Add, 3 AmbientOcclusion
    lightmap_directional: u32,             // offset 332 : 1 = sample the binding-18 direction atlas
    has_shadowmask: u32,                   // offset 356 : 1 = the binding-18 atlas is a shadowmask
    ignore_clip: u32,                      // offset 360 : 1 = exempt from clip planes/volumes
    object_mask: u32,                      // offset 364 : layer mask, AND-tested per light
    // The vec4 higher up keeps the struct 16-aligned, so its size is 368 and
    // matches the Rust ObjectUniform.
};

// Per-material UV transform block (group 0, binding 21). Slot order: 0 albedo,
// 1 normal, 2 AO, 3 metallic-roughness, 4 emissive. material_id 0 is identity.
struct TexTransform {
    offset_scale: vec4<f32>,   // (offset.x, offset.y, scale.x, scale.y)
    rot_tc: vec4<f32>,         // (rotation_radians, f32(uv_set), 0, 0)
}
// The per-object mesh path only reads the UV transforms (`xf`); the scalar and
// flag blocks are consumed by the instanced path. They are declared here so the
// WGSL array stride matches the Rust MaterialGpu (368 bytes): omitting them
// shrinks the stride and misreads every entry past material_id 0.
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
    tex_index1: vec4<u32>,   // emissive bindless index, sampler code (not read), unused, unused
    item0: vec4<f32>,        // (scalar_min, scalar_max, 0, 0)
    nan_colour: vec4<f32>,   // colour for NaN scalar attribute samples
    item_flags: vec4<u32>,   // (MAT_* bits, lightmap_mode, 0, 0)
    params: array<vec4<f32>, 16>, // material plugin parameters, declaration order
}
@group(0) @binding(21) var<storage, read> material_gpu_buf: array<MaterialGpu>;

// `item_flags.x` bits of MaterialGpu (mirrors the Rust `item_flags` module).
const MAT_USE_NAN_COLOUR: u32 = 1u;
const MAT_USE_MATCAP: u32 = 2u;
const MAT_MATCAP_BLENDABLE: u32 = 4u;
const MAT_LIGHTMAP_DIRECTIONAL: u32 = 8u;
const MAT_HAS_SHADOWMASK: u32 = 16u;
const MAT_HAS_EMISSIVE_TEX: u32 = 32u;
const MAT_UV_TRANSFORM: u32 = 128u;

struct SlotUv {
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
}

// See mesh.wgsl:material_slot_uv. uv' = rotate((uv*scale+offset)-0.5, rot)+0.5,
// derivatives rotated by the same angle. rot_tc.y picks uv0 (0) or uv1 (1).
fn material_slot_uv(
    mid: u32,
    slot: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    ddx0: vec2<f32>,
    ddy0: vec2<f32>,
    ddx1: vec2<f32>,
    ddy1: vec2<f32>,
) -> SlotUv {
    let xf = material_gpu_buf[mid].xf[slot];
    let use1 = xf.rot_tc.y > 0.5;
    let uv = select(uv0, uv1, use1);
    let duvdx = select(ddx0, ddx1, use1);
    let duvdy = select(ddy0, ddy1, use1);
    // Off when every slot of the batch's materials is at identity.
    if !FEAT_UV_TRANSFORM {
        return SlotUv(uv, duvdx, duvdy);
    }
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

// #include "helpers/clip_volume_test.wgsl"
// Indexed per-object storage array; see mesh.wgsl for the rationale. The
// selected element is copied into the `object` private global at entry.
@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;
var<private> object: Object;
@group(1) @binding(1) var obj_texture: texture_2d<f32>;
@group(1) @binding(2) var obj_sampler: sampler;
@group(1) @binding(3) var normal_map: texture_2d<f32>;
@group(1) @binding(4) var ao_map: texture_2d<f32>;
@group(1) @binding(5) var lut_texture: texture_2d<f32>;
// The mesh's sidecar record chunk (see mesh.wgsl).
@group(1) @binding(6) var<storage, read> sidecar_pool: array<u32>;
// #include "helpers/sidecar.wgsl"
// Optional features, one constant each, as in mesh.wgsl.
override FEAT_RECORD: bool = true;
override FEAT_COLOUR_STREAM: bool = true;
override FEAT_WARP: bool = true;
override FEAT_OVERRIDES: bool = true;
override FEAT_LIGHTMAP: bool = true;
override FEAT_MATCAP: bool = true;
override FEAT_DEFORM: bool = true;
override FEAT_EMISSIVE_MAP: bool = true;
override FEAT_UV_TRANSFORM: bool = true;
@group(1) @binding(10) var lut_sampler: sampler;
@group(1) @binding(11) var metallic_roughness_tex: texture_2d<f32>;
@group(1) @binding(12) var emissive_tex: texture_2d<f32>;
// Per-vertex position and normal overrides, flat `array<f32>` with 3 values
// per vertex, windowed by the sidecar record (see mesh.wgsl).
@group(1) @binding(13) var<storage, read> position_override_buffer: array<f32>;
@group(1) @binding(14) var<storage, read> normal_override_buffer:   array<f32>;
// Baked lightmap texture, sampled with the binding-2 material sampler.
@group(1) @binding(17) var lightmap_tex: texture_2d_array<f32>;
// Dominant-direction atlas for a directional lightmap (see mesh.wgsl).
@group(1) @binding(18) var lightmap_dir_tex: texture_2d_array<f32>;
// Second UV set (glTF TEXCOORD_1), parallel to the vertex buffer. See mesh.wgsl.
@group(1) @binding(19) var<storage, read> uv1_buf: array<vec2<f32>>;
// The GPU-culled draw's visible list (see mesh.wgsl).
@group(1) @binding(15) var<storage, read> visibility_indices: array<u32>;

fn lightmap_directional_factor(uv: vec2<f32>, page: i32, n_pix: vec3<f32>, n_geo: vec3<f32>) -> f32 {
    let d = textureSampleLevel(lightmap_dir_tex, obj_sampler, uv, page, 0.0);
    let ndl_pix = max(dot(normalize(n_pix), d.xyz), 0.0);
    let ndl_geo = max(dot(normalize(n_geo), d.xyz), 0.0);
    return mix(1.0, ndl_pix / max(ndl_geo, 0.1), d.w);
}

// Assemble the `Object` view the fragment body reads, as in mesh.wgsl. The
// vertex stage reads the record fields it needs directly.
fn load_object_vertex(i: u32) -> Object {
    var o: Object;
    let inst = instances[i];
    let bit = inst.flags;
    o.model = inst.model;
    o.colour = inst.colour;
    o.selected = u32((bit & INST_SELECTED) != 0u);
    o.wireframe = u32((bit & INST_WIREFRAME) != 0u);
    o.has_texture = u32((bit & INST_HAS_TEXTURE) != 0u);
    o.has_normal_map = u32((bit & INST_HAS_NORMAL_MAP) != 0u);
    o.has_ao_map = u32((bit & INST_HAS_AO_MAP) != 0u);
    o.has_attribute = u32((bit & INST_HAS_ATTRIBUTE) != 0u);
    o.use_face_colour = u32((bit & INST_USE_FACE_COLOUR) != 0u);
    o.receive_shadows = u32((bit & INST_RECEIVE_SHADOWS) != 0u);
    o.unlit = u32((bit & INST_UNLIT) != 0u);
    o.has_warp = u32((bit & INST_HAS_WARP) != 0u);
    o.warp_scale = inst.warp_scale;
    o.alpha_cutoff = inst.alpha_cutoff;
    o.material_id = inst.material_id;
    o.deform_flags = inst.deform_flags;
    o.has_light_probe = u32((bit & INST_HAS_LIGHT_PROBE) != 0u);
    o.light_probe_index = inst.light_probe_index;
    o.ignore_clip = u32((bit & INST_IGNORE_CLIP) != 0u);
    o.object_mask = inst.object_mask;
    return o;
}

fn load_object(i: u32) -> Object {
    var o = load_object_vertex(i);
    let inst = instances[i];
    let mat = material_gpu_buf[inst.material_id];
    let mbit = mat.item_flags.x;
    o.ambient = mat.scalars0.x;
    o.diffuse = mat.scalars0.y;
    o.specular = mat.scalars0.z;
    o.shininess = mat.scalars0.w;
    o.use_pbr = mat.flags.x;
    o.metallic = mat.scalars1.x;
    o.roughness = mat.scalars1.y;
    o.scalar_min = mat.item0.x;
    o.scalar_max = mat.item0.y;
    o.nan_colour = mat.nan_colour;
    o.use_nan_colour = u32((mbit & MAT_USE_NAN_COLOUR) != 0u);
    o.use_matcap = u32((mbit & MAT_USE_MATCAP) != 0u);
    o.matcap_blendable = u32((mbit & MAT_MATCAP_BLENDABLE) != 0u);
    o.uv_vis_mode = mat.flags.w;
    o.uv_vis_scale = mat.scalars3.y;
    o.backface_policy = u32(mat.scalars3.z);
    // The Pattern world scale is per instance; the block's .w is unused.
    o.backface_colour = vec4<f32>(mat.backface_colour.xyz, inst.backface_pattern_scale);
    o.emissive = mat.scalars2.xyz;
    o.use_flat = mat.flags.y;
    o.alpha_mode = mat.flags.z;
    o.has_metallic_roughness_tex = u32(mat.scalars1.w);
    o.has_emissive_tex = u32(mat.scalars3.w);
    o.normal_strength = mat.scalars1.z;
    o.ao_range = vec2<f32>(mat.scalars2.w, mat.scalars3.x);
    o.metallic_range = mat.mr_range.xy;
    o.roughness_range = mat.mr_range.zw;
    o.lightmap_mode = mat.item_flags.y;
    o.lightmap_directional = u32((mbit & MAT_LIGHTMAP_DIRECTIONAL) != 0u);
    o.has_shadowmask = u32((mbit & MAT_HAS_SHADOWMASK) != 0u);
    return o;
}

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
    @builtin(vertex_index) vertex_index: u32,
    // Per-instance record words from vertex buffer slot 1 (see mesh.wgsl).
    @location(5) w_record: u32,
    @location(6) w_colour: u32,
    @location(7) w_warp: u32,
    @location(8) w_lightmap: u32,
    @location(9) w_vertex_base: u32,
    @location(10) w_deform: u32,
};

// The vertex inputs of a pipeline whose feature set reads no record word
// (no record, colour stream, warp, override, lightmap or deform): `VertexIn`
// without slot 1. Such a pipeline declares slot 0 alone, so no instance-rate
// attribute is fetched for it.
struct VertexInLean {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
    @builtin(vertex_index) vertex_index: u32,
};

// `VertexIn` for the shared vertex body, every record word zero.
fn from_lean_in(p: VertexInLean) -> VertexIn {
    var v: VertexIn;
    v.position = p.position;
    v.normal = p.normal;
    v.colour = p.colour;
    v.uv = p.uv;
    v.tangent = p.tangent;
    v.vertex_index = p.vertex_index;
    return v;
}

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
    // Baked lightmap UV1, interpolated for the fragment lightmap sample.
    @location(9) lightmap_uv:     vec2<f32>,
    // Atlas page (array layer) for a multi-page lightmap, carried in UV1.z. Flat:
    // every vertex of a chart shares one page, so no interpolation is wanted.
    @location(10) @interpolate(flat) lightmap_page: f32,
    // Index of this draw's element in instances[], carried to the fragment stage.
    @location(11) @interpolate(flat) obj_idx: u32,
    // Second UV set (glTF TEXCOORD_1); vec2(0.0) for meshes without one.
    @location(12) uv1: vec2<f32>,
    // Kept-surface value from deformers that define `keep`; see deform.wgsl.
    // <viewport-deform-keep> @location(13) deform_keep: f32,
    // Plugin vertex-attribute varying: the composer adds a @location(8)
    // member here for hooks that read the per-vertex extension attribute.
    // <viewport-shade-slot:vertex-out>
    // </viewport-shade-slot:vertex-out>
};

// The varyings of a pipeline whose feature set reads neither a colour stream
// nor a lightmap: `VertexOut` without the five only those features read. A
// vertex entry returning `VertexOut` writes all of its members (WGSL
// zero-initialises the unwritten ones), so such a pipeline draws through the
// `_plain` entries below, whose interface is only these.
struct VertexOutPlain {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) colour:          vec4<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) world_pos:      vec3<f32>,
    @location(3) uv:             vec2<f32>,
    @location(4) world_tangent:  vec4<f32>,
    @location(11) @interpolate(flat) obj_idx: u32,
    @location(12) uv1: vec2<f32>,
    // <viewport-deform-keep> @location(13) deform_keep: f32,
    // <viewport-shade-slot:vertex-out-plain>
    // </viewport-shade-slot:vertex-out-plain>
};

fn to_plain(o: VertexOut) -> VertexOutPlain {
    var p: VertexOutPlain;
    p.clip_pos = o.clip_pos;
    p.colour = o.colour;
    p.world_normal = o.world_normal;
    p.world_pos = o.world_pos;
    p.uv = o.uv;
    p.world_tangent = o.world_tangent;
    p.obj_idx = o.obj_idx;
    p.uv1 = o.uv1;
    // <viewport-deform-keep> p.deform_keep = o.deform_keep;
    // <viewport-shade-slot:vertex-out-to-plain>
    // </viewport-shade-slot:vertex-out-to-plain>
    return p;
}

// The full varyings for the shared fragment body, the five a plain pipeline
// never reads left at zero.
fn from_plain(p: VertexOutPlain) -> VertexOut {
    var o: VertexOut;
    o.clip_pos = p.clip_pos;
    o.colour = p.colour;
    o.world_normal = p.world_normal;
    o.world_pos = p.world_pos;
    o.uv = p.uv;
    o.world_tangent = p.world_tangent;
    o.obj_idx = p.obj_idx;
    o.uv1 = p.uv1;
    // <viewport-deform-keep> o.deform_keep = p.deform_keep;
    // <viewport-shade-slot:vertex-out-from-plain>
    // </viewport-shade-slot:vertex-out-from-plain>
    return o;
}

struct OitOut {
    @location(0) accum:  vec4<f32>,
    @location(1) reveal: f32,
};

// A direct draw: the instance index is the record.
@vertex
fn vs_main(in: VertexIn, @builtin(instance_index) instance_index: u32) -> VertexOut {
    return vs_body(in, instance_index);
}

// A GPU-culled draw: the instance index is a slot in the visible list, which
// names the record. See mesh.wgsl.
@vertex
fn vs_main_cull(in: VertexIn, @builtin(instance_index) slot: u32) -> VertexOut {
    return vs_body(in, visibility_indices[slot]);
}

// `vs_main` and `vs_main_cull` for a pipeline drawing through `VertexOutPlain`.
@vertex
fn vs_main_plain(in: VertexIn, @builtin(instance_index) instance_index: u32) -> VertexOutPlain {
    return to_plain(vs_body(in, instance_index));
}

@vertex
fn vs_main_cull_plain(in: VertexIn, @builtin(instance_index) slot: u32) -> VertexOutPlain {
    return to_plain(vs_body(in, visibility_indices[slot]));
}

// The plain entries for a pipeline that also reads no record word, through
// `VertexInLean`.
@vertex
fn vs_main_lean(in: VertexInLean, @builtin(instance_index) instance_index: u32) -> VertexOutPlain {
    return to_plain(vs_body(from_lean_in(in), instance_index));
}

@vertex
fn vs_main_cull_lean(in: VertexInLean, @builtin(instance_index) slot: u32) -> VertexOutPlain {
    return to_plain(vs_body(from_lean_in(in), visibility_indices[slot]));
}

fn vs_body(in: VertexIn, record: u32) -> VertexOut {
    var out: VertexOut;
    out.obj_idx = record;
    // The vertex index within the mesh, as in mesh.wgsl.
    let vi = in.vertex_index - in.w_vertex_base;
    // The record fields every draw reads; the rest only inside the feature
    // that needs them, as in mesh.wgsl.
    let model = instances[record].model;
    let inst_flags = instances[record].flags;
    let deform_flags = instances[record].deform_flags;
    // The mesh's sidecar record and the streams this item reads, as in
    // mesh.wgsl; each read is behind the flag that needs it.
    let rec = in.w_record;
    let has_sidecar = FEAT_RECORD && (inst_flags & INST_HAS_SIDECAR) != 0u;
    deform_table = in.w_deform;
    // Override > vertex attribute, as in mesh.wgsl; warp layers on top.
    var local_pos = in.position;
    if FEAT_OVERRIDES && (inst_flags & INST_HAS_POSITION_OVERRIDE) != 0u && vi < sc_word(rec + SC_POS_LEN) {
        let pi = (sc_word(rec + SC_POS_BASE) + vi) * 3u;
        if pi + 2u < arrayLength(&position_override_buffer) {
            local_pos = vec3<f32>(
                position_override_buffer[pi],
                position_override_buffer[pi + 1u],
                position_override_buffer[pi + 2u],
            );
        }
    }
    if FEAT_WARP && has_sidecar && (inst_flags & INST_HAS_WARP) != 0u {
        let warp = in.w_warp;
        if warp != 0u {
            local_pos += sc_vec3(warp + vi * 3u) * instances[record].warp_scale;
        }
    }
    var local_normal = in.normal;
    if FEAT_OVERRIDES && (inst_flags & INST_HAS_NORMAL_OVERRIDE) != 0u && vi < sc_word(rec + SC_NRM_LEN) {
        let ni = (sc_word(rec + SC_NRM_BASE) + vi) * 3u;
        if ni + 2u < arrayLength(&normal_override_buffer) {
            local_normal = vec3<f32>(
                normal_override_buffer[ni],
                normal_override_buffer[ni + 1u],
                normal_override_buffer[ni + 2u],
            );
        }
    }
    var dv = DeformVertex(local_pos, local_normal, vi);
    let dctx = DeformContext(model, model[3].xyz, 0.0, deform_flags, 0u);
    if FEAT_DEFORM {
        dv = viewport_deform_object_space(dv, dctx);
    }
    let model3 = mat3x3<f32>(
        model[0].xyz,
        model[1].xyz,
        model[2].xyz,
    );
    let world_pos4 = model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv.normal = normalize(model3 * dv.normal);
    if FEAT_DEFORM {
        dv = viewport_deform_world_space(dv, dctx);
    }
    // <viewport-deform-keep> out.deform_keep = viewport_deform_keep(dv, dctx);
    let world_pos = vec4<f32>(dv.position, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.colour = in.colour;
    out.world_pos = world_pos.xyz;
    out.world_normal = dv.normal;
    out.world_tangent = vec4<f32>(normalize(model3 * in.tangent.xyz), in.tangent.w);
    out.uv = in.uv;
    // Second UV set from the parallel uv1 stream. See mesh.wgsl:vs_body.
    out.uv1 = vec2<f32>(0.0);
    if (inst_flags & INST_HAS_UV1) != 0u {
        let uv1_len = arrayLength(&uv1_buf);
        out.uv1 = uv1_buf[min(in.vertex_index, max(uv1_len, 1u) - 1u)];
    }
    // Read only by fragment code behind the same feature, as in mesh.wgsl.
    if FEAT_COLOUR_STREAM {
        let colour_stream = in.w_colour;
        let face = (inst_flags & INST_USE_FACE_COLOUR) != 0u;
        var has_attr = false;
        var raw_scalar = 0.0;
        if has_sidecar && (inst_flags & INST_HAS_ATTRIBUTE) != 0u && !face && colour_stream != 0u {
            has_attr = true;
            raw_scalar = sc_f32(colour_stream + vi);
        }
        out.scalar_val = raw_scalar;
        let sv_bits = bitcast<u32>(raw_scalar);
        let sv_is_nan = has_attr && (sv_bits & 0x7F800000u) == 0x7F800000u && (sv_bits & 0x007FFFFFu) != 0u;
        out.is_nan_scalar = select(0.0, 1.0, sv_is_nan);
        out.face_colour = vec4<f32>(1.0);
        if has_sidecar && face && colour_stream != 0u {
            out.face_colour = sc_vec4(colour_stream + 4u * vi);
        }
    }
    // Lightmap UV1 through the draw's placement, as in mesh.wgsl.
    let lm = in.w_lightmap;
    if FEAT_LIGHTMAP && has_sidecar && lm != 0u {
        let lm_sidecar = sc_vec4(lm + 4u * vi);
        let placement = instances[record].lightmap_scale_bias;
        out.lightmap_uv = lm_sidecar.xy * placement.xy + placement.zw;
        out.lightmap_page = lm_sidecar.z + f32(instances[record].lightmap_layer);
    }
    // <viewport-shade-slot:vertex-fetch>
    // </viewport-shade-slot:vertex-fetch>
    return out;
}

// ---------------------------------------------------------------------------
// 32-sample Poisson disk (shadow sampling : identical to mesh.wgsl)
// ---------------------------------------------------------------------------


// #include "helpers/csm.wgsl"

// ---------------------------------------------------------------------------
// PBR BRDF helpers (Cook-Torrance)
// ---------------------------------------------------------------------------
// Shared direct BRDF: D_GGX, G1_Smith, G_Smith, F_Schlick, pbr_light_contrib.
// #include "helpers/brdf.wgsl"

// Shared ambient / IBL helpers (equirect sampling, split-sum ambient).
// #include "helpers/ambient.wgsl"

// UV parameterization visualization : procedural RGB colour from UV coordinates.
// Matches the implementation in mesh.wgsl exactly.
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
    // Raw metallic-roughness sample (G = roughness, B = metallic), read only
    // when the material has the map.
    mr_sample: vec2<f32>,
    // Emissive factor times the emissive map. Sampled here rather than after
    // lighting so the UV gradients are not held through the light loop.
    emissive: vec3<f32>,
    alpha: f32,
    // The material's alpha mode (0 opaque, 1 mask, 2 blend, 3 premultiplied)
    // and mask cutoff, for the surface hook's alpha gate.
    alpha_mode: u32,
    alpha_cutoff: f32,
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
    // The instance's raw material channel: custom data slots 4..8 (`data1`),
    // the block reserved for material plugins. A module whose hook reads the
    // per-vertex extension attribute replaces it with that attribute.
    // `custom_data_id` 0 is the shared all-zero block.
    surf.attr = custom_data1(lights_uniform.custom_data_base, instances[in.obj_idx].custom_data_id);
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
    let shade_material = object.material_id;
    // <viewport-shade-slot:params>
    // </viewport-shade-slot:params>
    var out: Surface;
    out.resolved = false;
    out.out_oit.accum = vec4<f32>(0.0);
    out.out_oit.reveal = 0.0;
    out.base_colour = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.ao_factor = 1.0;
    out.mat_uv = in.uv;
    out.mr_sample = vec2<f32>(1.0);
    out.emissive = vec3<f32>(0.0);
    out.alpha = 1.0;
    out.alpha_mode = object.alpha_mode;
    out.alpha_cutoff = object.alpha_cutoff;
    out.front_facing = select(0u, 1u, is_front);

    // Screen-space derivatives of the interpolated inputs, taken here where
    // control flow is still uniform. The shading branches below key off
    // per-object storage (non-uniform), where implicit derivatives are rejected
    // by strict WGSL validators; these feed explicit-gradient sampling.
    let d_uv_dx = dpdx(in.uv);
    let d_uv_dy = dpdy(in.uv);
    let d_uv1_dx = dpdx(in.uv1);
    let d_uv1_dy = dpdy(in.uv1);
    let d_wp_dx = dpdx(in.world_pos);
    let d_wp_dy = dpdy(in.world_pos);

    // A deformer that defines `keep` removed the surface here.
    // <viewport-deform-keep> if in.deform_keep < 0.0 { discard; }

    // Section view clipping. Items with ignore_clip stay fully visible.
    if (instances[in.obj_idx].flags & INST_IGNORE_CLIP) == 0u {
        for (var i = 0u; i < clip_planes.count; i++) {
            let plane = clip_planes.planes[i];
            if dot(in.world_pos, plane.xyz) + plane.w < 0.0 {
                discard;
            }
        }
        if !clip_volume_test(in.world_pos) { discard; }
    }

    // Per-material UV transform (slot 0 = albedo; also feeds the plugin surf.uv).
    let s0 = material_slot_uv(
        object.material_id, 0u, in.uv, in.uv1, d_uv_dx, d_uv_dy, d_uv1_dx, d_uv1_dy,
    );
    let mat_uv = s0.uv;
    out.mat_uv = mat_uv;
    let muv_ddx = s0.ddx;
    let muv_ddy = s0.ddy;
    let s_normal = material_slot_uv(
        object.material_id, 1u, in.uv, in.uv1, d_uv_dx, d_uv_dy, d_uv1_dx, d_uv1_dy,
    );
    let s_ao = material_slot_uv(
        object.material_id, 2u, in.uv, in.uv1, d_uv_dx, d_uv_dy, d_uv1_dx, d_uv1_dy,
    );

    // Sample texture if one is assigned.
    var tex_colour = vec4<f32>(1.0);
    if object.has_texture == 1u {
        tex_colour = textureSampleGrad(obj_texture, obj_sampler, mat_uv, muv_ddx, muv_ddy);
    }
    // An opaque material ignores texture and vertex alpha: below full opacity
    // it fades by the opacity alone, as it draws solid at full opacity.
    let obj_colour = vec4<f32>(
        object.colour.rgb * in.colour.rgb * tex_colour.rgb,
        select(object.colour.a * in.colour.a * tex_colour.a, object.colour.a, object.alpha_mode == 0u),
    );
    out.alpha = obj_colour.a;

    // Alpha MASK: discard fragments whose alpha is below the cutoff.
    if object.alpha_mode == 1u && obj_colour.a < object.alpha_cutoff {
        discard;
    }

    var base_colour = obj_colour.rgb;

    // Per-face RGBA colour: use directly, bypassing all lighting and colourmap logic.
    if FEAT_COLOUR_STREAM && object.use_face_colour != 0u {
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
    if FEAT_COLOUR_STREAM && object.has_attribute != 0u {
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
        // alpha_mode == 3 means the colour is already premultiplied by alpha, so
        // it is weighted directly; straight blend premultiplies here.
        let premult_rgb = select(base_colour * alpha, base_colour, object.alpha_mode == 3u);
        out.out_oit.accum  = vec4<f32>(premult_rgb, alpha) * w;
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
        let dpx = d_wp_dx;
        let dpy = d_wp_dy;
        var Nf = normalize(cross(dpx, dpy));
        if dot(Nf, in.world_normal) < 0.0 { Nf = -Nf; }
        N = Nf;
    } else if object.has_normal_map != 0u {
        let nm_sample = textureSampleGrad(normal_map, obj_sampler, s_normal.uv, s_normal.ddx, s_normal.ddy).rgb;
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
        let raw_ao = textureSampleGrad(ao_map, obj_sampler, s_ao.uv, s_ao.ddx, s_ao.ddy).r;
        ao_factor = mix(object.ao_range.x, object.ao_range.y, raw_ao);
    }

    // Metallic-roughness (slot 3) and emissive (slot 4), each with its own
    // transform and uv_set.
    if object.has_metallic_roughness_tex != 0u {
        let s_mr = material_slot_uv(
            object.material_id, 3u, in.uv, in.uv1, d_uv_dx, d_uv_dy, d_uv1_dx, d_uv1_dy,
        );
        out.mr_sample = textureSampleGrad(metallic_roughness_tex, obj_sampler, s_mr.uv, s_mr.ddx, s_mr.ddy).gb;
    }
    var emissive = object.emissive;
    if FEAT_EMISSIVE_MAP && object.has_emissive_tex != 0u {
        let s_em = material_slot_uv(
            object.material_id, 4u, in.uv, in.uv1, d_uv_dx, d_uv_dy, d_uv1_dx, d_uv1_dy,
        );
        emissive = emissive * textureSampleGrad(emissive_tex, obj_sampler, s_em.uv, s_em.ddx, s_em.ddy).rgb;
    }
    out.emissive = emissive;

    out.base_colour = base_colour;
    out.normal = N;
    out.ao_factor = ao_factor;
    return out;
}

// Lighting for the transparent path. Transparent surfaces skip shadow sampling,
// so last_shadow_sample stays at its unshadowed default.
// Which shading loop a material takes. Material-plugin hooks compose onto
// the PBR loop, so the composer forces this to true in a hook module.
fn shading_is_pbr(flag: u32) -> bool {
    return flag != 0u;
}

fn compute_lit(
    surface: Surface,
    in: VertexOut,
    saa_kernel: f32,
    refl_dr: f32,
) -> LitResult {
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

    if shading_is_pbr(object.use_pbr) {
        var metallic  = clamp(object.metallic,  0.0, 1.0);
        var roughness = max(object.roughness, 0.04);
        if object.has_metallic_roughness_tex != 0u {
            // glTF ORM texture: G=roughness factor, B=metallic factor. Per-material
            // `metallic_range` / `roughness_range` remap the raw samples before the
            // scalar factor; identity `(0, 1)` is a no-op.
            let mr = surface.mr_sample;
            let m_remapped = mix(object.metallic_range.x, object.metallic_range.y, mr.y);
            let r_remapped = mix(object.roughness_range.x, object.roughness_range.y, mr.x);
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
        let pbr_range = cluster_light_range(in.world_pos, lights_uniform.count);
        // The item's light-channel mask, read once rather than per light.
        let light_mask = object.object_mask;
        for (var j = 0u; j < pbr_range.count; j++) {
            let i = cluster_light_global(pbr_range, j);
            if !light_in_channel(lights_storage[i], light_mask) { continue; }
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
            // Subtractive lightmap: the main directional light is baked in, so
            // suppress its realtime direct to avoid double counting (transparent
            // surfaces have no realtime shadow to add back).
            if !(FEAT_LIGHTMAP && object.lightmap_mode == 4u && i == 0u) {
                Lo += pbr_light_contrib(N, V, L, radiance, base_colour, metallic, roughness, F0);
            }
            // </viewport-shade-slot:light>
        }
        dbg_direct_lum = dot(Lo, lum_weights);
        dbg_roughness  = roughness;
        dbg_metallic   = metallic;
        // <viewport-shade-slot:ambient>
        var ambient: vec3<f32>;
        // The environment's specular, kept apart so a light probe or lightmap
        // that replaces the baked diffuse below leaves the reflection in place,
        // and the weight the environment's own diffuse gets.
        var ibl_specular = vec3<f32>(0.0);
        let ambient_kd = (vec3<f32>(1.0) - F_Schlick_roughness(max(dot(N, V), 0.001), F0, roughness))
            * (1.0 - metallic);
        if lights_uniform.ibl_enabled != 0u {
            var ibl: IblContrib;
            if lights_uniform.env_zone_count != 0u {
                // Per-fragment environment selection. `refl_dr` is the reflection
                // footprint computed in uniform control flow by the caller.
                ibl = ibl_ambient_zoned(N, V, base_colour, metallic, roughness, F0,
                                        ao_factor, lights_uniform.ibl_intensity,
                                        lights_uniform.ibl_rotation, refl_dr, in.world_pos,
                                        lights_uniform.env_zone_count);
            } else {
                ibl = ibl_ambient_grad(N, V, base_colour, metallic, roughness, F0,
                                       ao_factor, lights_uniform.ibl_intensity,
                                       lights_uniform.ibl_rotation, refl_dr);
            }
            ibl.diffuse *= lights_uniform.ibl_diffuse_scale;
            ibl.specular *= lights_uniform.ibl_specular_scale;
            ibl_specular = ibl.specular;
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
        // Light-probe objects take their indirect diffuse from the SH field; with
        // an environment, its reflection stays.
        if object.has_light_probe != 0u {
            let probe = evaluate_object_indirect(object.light_probe_index, in.world_pos, N) * base_colour * ao_factor;
            if lights_uniform.ibl_enabled != 0u {
                ambient = probe * ambient_kd + ibl_specular;
            } else {
                ambient = probe;
            }
            dbg_ambient_lum = dot(ambient, lum_weights);
        }
        // Baked lightmap: replace, add, or occlude the ambient term. Force the
        // base mip: derivative-based selection over tightly-packed atlas charts
        // reads as blocky bands on the mesh.
        if FEAT_LIGHTMAP && object.lightmap_mode != 0u {
            // Scene-atlas placement: map the object's [0,1] unwrap into its sub-rect
            // and add its base layer to the per-vertex page. Identity scale/bias and
            // layer 0 reproduce a per-mesh (single- or multi-page) lightmap exactly.
            let lm_uv = in.lightmap_uv;
            let lm_page = i32(round(in.lightmap_page));
            var lm = textureSampleLevel(lightmap_tex, obj_sampler, lm_uv, lm_page, 0.0);
            if object.lightmap_directional == 1u {
                let f = lightmap_directional_factor(lm_uv, lm_page, N, in.world_normal);
                lm = vec4<f32>(lm.rgb * f, lm.a);
            }
            ambient = apply_lightmap(ambient, base_colour, ao_factor, lm, object.lightmap_mode, 1.0);
            // Replace and Subtractive bake the diffuse only: weight it like the
            // environment's diffuse and keep the environment's reflection.
            if lights_uniform.ibl_enabled != 0u
                && (object.lightmap_mode == 1u || object.lightmap_mode == 4u) {
                ambient = ambient * ambient_kd + ibl_specular;
            }
            dbg_ambient_lum = dot(ambient, lum_weights);
        }
        // </viewport-shade-slot:ambient>
        final_rgb = clamp((Lo + ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(camera.lit_clamp));
        // <viewport-shade-slot:recolor>
        // </viewport-shade-slot:recolor>
    } else {
        var total_colour_contrib = vec3<f32>(0.0);
        let bp_range = cluster_light_range(in.world_pos, lights_uniform.count);
        // The item's light-channel mask, read once rather than per light.
        let light_mask = object.object_mask;
        for (var j = 0u; j < bp_range.count; j++) {
            let i = cluster_light_global(bp_range, j);
            if !light_in_channel(lights_storage[i], light_mask) { continue; }
            let ev = eval_light(lights_storage[i], in.world_pos);
            if !ev.in_range { continue; }
            // Transparent surfaces do not participate in shadow evaluation.
            let H = normalize(ev.l + V);
            let n_dot_l = max(dot(N, ev.l), 0.0);
            let n_dot_h = max(dot(N, H), 0.0);
            // Energy-normalised Blinn-Phong (matches the PBR path): 1/pi on the
            // diffuse lobe, (shininess + 8) / (8 pi) on the specular lobe.
            let diffuse_contrib  = object.diffuse  * n_dot_l * INV_PI;
            let specular_contrib = object.specular * pow(n_dot_h, object.shininess)
                                 * (object.shininess + 8.0) * INV_PI * 0.125;
            total_colour_contrib += (diffuse_contrib + specular_contrib) * ev.radiance;
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
            hemi_rgb = evaluate_object_indirect(object.light_probe_index, in.world_pos, N) * base_colour * ao_factor;
        }
        // Baked lightmap: replace, add, or occlude the ambient term. Force the
        // base mip (see the ambient branch above).
        if FEAT_LIGHTMAP && object.lightmap_mode != 0u {
            // Scene-atlas placement: map the object's [0,1] unwrap into its sub-rect
            // and add its base layer to the per-vertex page. Identity scale/bias and
            // layer 0 reproduce a per-mesh (single- or multi-page) lightmap exactly.
            let lm_uv = in.lightmap_uv;
            let lm_page = i32(round(in.lightmap_page));
            var lm = textureSampleLevel(lightmap_tex, obj_sampler, lm_uv, lm_page, 0.0);
            if object.lightmap_directional == 1u {
                let f = lightmap_directional_factor(lm_uv, lm_page, N, in.world_normal);
                lm = vec4<f32>(lm.rgb * f, lm.a);
            }
            hemi_rgb = apply_lightmap(hemi_rgb, base_colour, ao_factor, lm, object.lightmap_mode, 1.0);
        }
        dbg_ambient_lum = dot(hemi_rgb, lum_weights);
        let lit_rgb = hemi_rgb + direct_rgb;
        final_rgb = clamp(lit_rgb * tint.rgb, vec3<f32>(0.0), vec3<f32>(camera.lit_clamp));
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
    return fs_body(in, is_front);
}

// `fs_oit_main` for a pipeline drawing through `VertexOutPlain`.
@fragment
fn fs_oit_main_plain(p: VertexOutPlain, @builtin(front_facing) is_front: bool) -> OitOut {
    return fs_body(from_plain(p), is_front);
}

fn fs_body(in: VertexOut, is_front: bool) -> OitOut {
    object = load_object(in.obj_idx);
    let surface = compute_surface(in, is_front);

    // Derivative terms for the lighting stage, taken here while control flow is
    // still uniform (before the resolved early return and compute_lit's
    // object-dependent branches). Exact: uses the resolved normal and UV.
    let d_n_dx = dpdx(surface.normal);
    let d_n_dy = dpdy(surface.normal);
    let saa_kernel = min(0.5 * (dot(d_n_dx, d_n_dx) + dot(d_n_dy, d_n_dy)), 0.18);
    let V_dr = normalize(camera.eye_pos - in.world_pos);
    let R_dr = reflect(-V_dr, surface.normal);
    let refl_dr = max(length(dpdx(R_dr)), length(dpdy(R_dr)));

    if surface.resolved {
        return surface.out_oit;
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
    let lum_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    var final_rgb = lit.rgb;

    // Emissive term: added after lighting so it can push HDR values above 1.0.
    var emissive = surface.emissive;
    // Per-instance custom data slots 0..3 add to emissive (nits); zero is a
    // no-op.
    emissive += custom_data0(lights_uniform.custom_data_base, instances[in.obj_idx].custom_data_id).xyz;
    final_rgb += emissive;
    var dbg_emissive_lum = dot(emissive, lum_weights);
    // Surface-hook emissive: the composer adds the hook's emissive term here
    // in plugin-composed modules; inert in the base module.
    // <viewport-shade-slot:emissive>
    // </viewport-shade-slot:emissive>

    // #include "helpers/debug_vis.wgsl"

    // ---------------------------------------------------------------------------
    // McGuire & Bavoil weighted blended OIT output.
    // ---------------------------------------------------------------------------
    var final_alpha = surface.alpha;
    // Surface-hook alpha: honoured only under Mask/Blend alpha modes; the
    // composer fills this in plugin-composed modules.
    // <viewport-shade-slot:alpha>
    // </viewport-shade-slot:alpha>
    let alpha = final_alpha;
    let z = in.clip_pos.z;  // NDC depth 0..1
    let w = alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0))));

    var out: OitOut;
    // alpha_mode == 3 (BlendPremultiplied): final_rgb already carries the alpha
    // premultiply, so weight it as-is. Straight blend applies `* alpha` here.
    let premult_rgb = select(final_rgb * alpha, final_rgb, object.alpha_mode == 3u);
    out.accum  = vec4<f32>(premult_rgb * w, alpha * w);
    out.reveal = alpha;
    return out;
}
