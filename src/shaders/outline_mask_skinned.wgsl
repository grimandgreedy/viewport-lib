// Skinned variant of outline_mask.wgsl.
//
// Renders selected skinned objects as white (r=1.0) into an R8 mask texture,
// matching outline_mask.wgsl, but applies LBS to the bind-pose position in the
// vertex stage so the outline tracks the deformed silhouette on the GPU
// skinning path. Without this the outline mask is drawn from the unmodified
// bind-pose vertex buffer and the selection halo lags the rendered character.
//
// Group 0: Camera uniform (only view_proj is used).
// Group 1: OutlineUniform (only model is used).
// Group 2: skin bind group : weights storage buffer + per-instance joint palette.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

struct OutlineUniform {
    model: mat4x4<f32>,
    colour: vec4<f32>,
    pixel_offset: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct SkinVertex {
    weights: vec4<f32>,
    joints_01: u32,
    joints_23: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> outline: OutlineUniform;
@group(2) @binding(0) var<storage, read> skin_weights: array<SkinVertex>;
@group(2) @binding(1) var<storage, read> skin_palette: array<mat4x4<f32>>;

fn unpack_joint(skin: SkinVertex, slot: u32) -> u32 {
    if slot == 0u { return skin.joints_01 & 0xFFFFu; }
    if slot == 1u { return (skin.joints_01 >> 16u) & 0xFFFFu; }
    if slot == 2u { return skin.joints_23 & 0xFFFFu; }
    return (skin.joints_23 >> 16u) & 0xFFFFu;
}

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
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    let skin = skin_matrix(vertex_index);
    let local = (skin * vec4<f32>(position, 1.0)).xyz;
    let world = outline.model * vec4<f32>(local, 1.0);
    return camera.view_proj * world;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
