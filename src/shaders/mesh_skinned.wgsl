// Skinned vertex stage for the standard lit material.
//
// Replaces only the vertex stage of `mesh.wgsl`. The fragment stage in
// `mesh.wgsl` (entry point `fs_main`) is reused unchanged. Vertex output
// layout must stay in sync with `VertexOut` in `mesh.wgsl`.
//
// Skin data lives in storage buffers at group 2: a per-vertex weights buffer
// (joint indices + weights) and a per-instance joint palette. The mesh's
// bind-pose vertex buffer is unchanged. Linear blend skinning is applied to
// position and normal in this shader before the standard model transform.
//
// ---------------------------------------------------------------------------
// Bindings (only those the vertex stage needs are declared; the rest of
// `mesh.wgsl`'s bindings remain available to `fs_main` via the shared bind
// group layouts).
// ---------------------------------------------------------------------------

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
    forward: vec3<f32>,
    _pad1: f32,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

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
    nan_colour: vec4<f32>,
    use_nan_colour: u32,
    use_matcap: u32,
    matcap_blendable: u32,
    unlit: u32,
    use_face_colour: u32,
    uv_vis_mode: u32,
    uv_vis_scale: f32,
    backface_policy: u32,
    backface_colour: vec4<f32>,
    has_warp: u32,
    warp_scale: f32,
    _pad_warp0: u32,
    _pad_warp1: u32,
    emissive: vec3<f32>,
    _pad_emissive: u32,
    alpha_mode: u32,
    alpha_cutoff: f32,
    has_metallic_roughness_tex: u32,
    has_emissive_tex: u32,
};

@group(1) @binding(0) var<uniform> object: Object;
// Per-vertex storage buffers consumed by the vertex stage. Bindings match
// `mesh.wgsl`'s declarations exactly so the same `object_bgl` works for both
// pipeline variants. Texture bindings (1..5, 7, 11, 12) are read by
// `mesh.wgsl`'s `fs_main` and not declared here.
@group(1) @binding(6) var<storage, read> scalar_buffer: array<f32>;
@group(1) @binding(8) var<storage, read> face_colour_buffer: array<vec4<f32>>;
@group(1) @binding(9) var<storage, read> warp_buffer: array<f32>;

// One entry per vertex: 4 weights, then joint indices packed as two u32s
// (low/high u16 pair). Field order matches `PackedSkinVertex` in
// `src/resources/skin.rs` so the struct is naturally 24 bytes with no
// padding under WGSL std430 layout (vec4<f32> aligns to 16, two u32s pack
// immediately after).
struct SkinVertex {
    weights: vec4<f32>,
    joints_01: u32,
    joints_23: u32,
};

@group(2) @binding(0) var<storage, read> skin_weights: array<SkinVertex>;
@group(2) @binding(1) var<storage, read> skin_palette: array<mat4x4<f32>>;

// ---------------------------------------------------------------------------
// Vertex IO
// ---------------------------------------------------------------------------

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:   vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) colour:         vec4<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) world_pos:      vec3<f32>,
    @location(3) uv:             vec2<f32>,
    @location(4) world_tangent:  vec4<f32>,
    @location(5) scalar_val:     f32,
    @location(6) is_nan_scalar:  f32,
    @location(7) face_colour:    vec4<f32>,
};

// Unpack one of four 16-bit joint indices from the two packed u32s.
fn unpack_joint(skin: SkinVertex, slot: u32) -> u32 {
    if slot == 0u { return skin.joints_01 & 0xFFFFu; }
    if slot == 1u { return (skin.joints_01 >> 16u) & 0xFFFFu; }
    if slot == 2u { return skin.joints_23 & 0xFFFFu; }
    return (skin.joints_23 >> 16u) & 0xFFFFu;
}

// Blend the four palette matrices for this vertex into a single skinning
// matrix. Zero-weight slots fall out naturally because the matrix is scaled
// by zero before summing.
fn skin_matrix(vertex_index: u32) -> mat4x4<f32> {
    let s = skin_weights[vertex_index];
    let j0 = unpack_joint(s, 0u);
    let j1 = unpack_joint(s, 1u);
    let j2 = unpack_joint(s, 2u);
    let j3 = unpack_joint(s, 3u);
    return skin_palette[j0] * s.weights.x
         + skin_palette[j1] * s.weights.y
         + skin_palette[j2] * s.weights.z
         + skin_palette[j3] * s.weights.w;
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    // Warp is applied to the bind-pose position before skinning so a warped
    // vertex follows its joint cleanly. Mirrors mesh.wgsl's vs_main behaviour
    // for warp + skinning composition.
    var bind_local = in.position;
    if object.has_warp != 0u {
        let wi = in.vertex_index * 3u;
        let warp_len = arrayLength(&warp_buffer);
        if wi + 2u < warp_len {
            bind_local += vec3<f32>(
                warp_buffer[wi],
                warp_buffer[wi + 1u],
                warp_buffer[wi + 2u],
            ) * object.warp_scale;
        }
    }

    let skin = skin_matrix(in.vertex_index);
    let local_pos = (skin * vec4<f32>(bind_local, 1.0)).xyz;

    let skin3 = mat3x3<f32>(
        skin[0].xyz,
        skin[1].xyz,
        skin[2].xyz,
    );
    let local_normal = skin3 * in.normal;
    let local_tangent = skin3 * in.tangent.xyz;

    let world_pos = object.model * vec4<f32>(local_pos, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.colour = in.colour;

    let model3 = mat3x3<f32>(
        object.model[0].xyz,
        object.model[1].xyz,
        object.model[2].xyz,
    );
    out.world_normal = normalize(model3 * local_normal);
    out.world_tangent = vec4<f32>(normalize(model3 * local_tangent), in.tangent.w);
    out.uv = in.uv;

    // Per-vertex scalar attribute, guarded by has_attribute + buffer length.
    // Detect NaN before interpolation can corrupt the bit pattern. Mirrors
    // mesh.wgsl::vs_main.
    let buf_len = arrayLength(&scalar_buffer);
    let idx = in.vertex_index;
    let has_attr = object.has_attribute != 0u && buf_len > 0u;
    let safe_idx = min(idx, select(0u, buf_len - 1u, buf_len > 0u));
    let raw_scalar = scalar_buffer[safe_idx];
    out.scalar_val = select(0.0, raw_scalar, has_attr);
    let sv_bits = bitcast<u32>(raw_scalar);
    let sv_is_nan = has_attr
        && (sv_bits & 0x7F800000u) == 0x7F800000u
        && (sv_bits & 0x007FFFFFu) != 0u;
    out.is_nan_scalar = select(0.0, 1.0, sv_is_nan);

    // Per-face RGBA colour (FaceColour attribute kind). Indexed by
    // vertex_index which equals the sequential draw invocation counter for
    // non-indexed face draws.
    let fc_len = arrayLength(&face_colour_buffer);
    let fc_idx = min(idx, select(0u, fc_len - 1u, fc_len > 0u));
    out.face_colour = select(
        vec4<f32>(1.0),
        face_colour_buffer[fc_idx],
        object.use_face_colour != 0u && fc_len > 0u,
    );

    return out;
}
