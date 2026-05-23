// Skinned depth-only vertex stage for the shadow cast pipeline.
//
// Replaces the vertex stage of `shadow.wgsl` for skinned meshes. The pipeline
// has no fragment stage. Same skin data layout as `mesh_skinned.wgsl`.

struct ShadowVP {
    light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> shadow_vp: ShadowVP;

// Only the model matrix from the per-object uniform is referenced here. The
// buffer bound at group 1 binding 0 is the full ObjectUniform; this shader's
// struct is allowed to be smaller because min_binding_size is None on the
// object bind group layout.
struct Object {
    model: mat4x4<f32>,
};

@group(1) @binding(0) var<uniform> object: Object;

// Field order matches `PackedSkinVertex` in `src/resources/skin.rs` (and the
// `SkinVertex` struct in `mesh_skinned.wgsl`): weights first so the vec4
// sits on a 16-byte boundary with no leading padding, joint pairs after.
struct SkinVertex {
    weights: vec4<f32>,
    joints_01: u32,
    joints_23: u32,
};

@group(2) @binding(0) var<storage, read> skin_weights: array<SkinVertex>;
@group(2) @binding(1) var<storage, read> skin_palette: array<mat4x4<f32>>;

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
};

fn unpack_joint(skin: SkinVertex, slot: u32) -> u32 {
    if slot == 0u { return skin.joints_01 & 0xFFFFu; }
    if slot == 1u { return (skin.joints_01 >> 16u) & 0xFFFFu; }
    if slot == 2u { return skin.joints_23 & 0xFFFFu; }
    return (skin.joints_23 >> 16u) & 0xFFFFu;
}

fn skin_matrix(vertex_index: u32) -> mat4x4<f32> {
    let s = skin_weights[vertex_index];
    return skin_palette[unpack_joint(s, 0u)] * s.weights.x
         + skin_palette[unpack_joint(s, 1u)] * s.weights.y
         + skin_palette[unpack_joint(s, 2u)] * s.weights.z
         + skin_palette[unpack_joint(s, 3u)] * s.weights.w;
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let skin = skin_matrix(in.vertex_index);
    let local_pos = (skin * vec4<f32>(in.position, 1.0)).xyz;
    let world = object.model * vec4<f32>(local_pos, 1.0);
    out.clip_pos = shadow_vp.light_view_proj * world;
    return out;
}
