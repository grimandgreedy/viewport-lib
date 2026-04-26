// GPU object-ID picking shader.
//
// Renders each instance as a flat u32 object ID into an R32Uint target.
// A second R32Float target captures the clip-space depth value so the caller
// can reconstruct world position without an additional depth-texture copy.
//
// No lighting, no textures : just position transform + flat ID output.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

// Per-instance pick data: model matrix + object_id.
// Stored in a separate compact storage buffer (not reusing the 128-byte InstanceData).
struct PickInstance {
    model_c0: vec4<f32>,
    model_c1: vec4<f32>,
    model_c2: vec4<f32>,
    model_c3: vec4<f32>,
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> pick_instances: array<PickInstance>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    // Other vertex attributes (normal, color, uv, tangent) are present in the
    // buffer but ignored : the 64-byte stride handles them automatically.
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) object_id: u32,
};

@vertex
fn vs_main(in: VertexIn, @builtin(instance_index) idx: u32) -> VertexOut {
    let inst = pick_instances[idx];
    let model = mat4x4<f32>(inst.model_c0, inst.model_c1, inst.model_c2, inst.model_c3);
    var out: VertexOut;
    out.clip_pos = camera.view_proj * model * vec4<f32>(in.position, 1.0);
    out.object_id = inst.object_id;
    return out;
}

struct FragOut {
    @location(0) object_id: u32,
    @location(1) depth: f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    var out: FragOut;
    out.object_id = in.object_id;
    // clip_pos.z in the fragment stage is the depth buffer value in [0, 1].
    out.depth = in.clip_pos.z;
    return out;
}
