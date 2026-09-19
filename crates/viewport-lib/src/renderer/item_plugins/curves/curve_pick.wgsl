// GPU object-ID pick shader for the curve mesh item types (streamtube, tube,
// ribbon).
//
// All three generate a connected triangle mesh CPU-side and upload it as one
// owned vertex + index buffer pair, so one pick shader covers them: read the
// position, apply the item's model matrix, write the item's object id. The
// vertex stride is the lib's 64-byte `Vertex`; only position is declared.
//
// Group 0: the shared scene bind group (Camera at binding 0, ClipVolume at
//          binding 6).
// Group 1: per-draw model matrix + object id (one item = one draw).
//
// On a device with the primitive-index feature the pipeline builder rewrites
// `fs_main` to report `@builtin(primitive_index)` in the primitive channel, so
// a SEGMENT / STRIP pick can map the hit triangle back to a curve segment.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
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

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

// #include "helpers/clip_volume_test.wgsl"

@group(1) @binding(0) var<uniform> pick: PickInstance;

struct VertexIn {
    @location(0) position: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let model = mat4x4<f32>(pick.model_c0, pick.model_c1, pick.model_c2, pick.model_c3);
    let world = model * vec4<f32>(in.position, 1.0);
    var out: VertexOut;
    out.clip_pos  = camera.view_proj * world;
    out.world_pos = world.xyz;
    return out;
}

struct FragOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    if !clip_volume_test(in.world_pos) { discard; }
    var out: FragOut;
    out.object_id = pick.object_id;
    out.primitive_id = 0u;
    out.depth = in.clip_pos.z;
    return out;
}
