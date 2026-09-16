// GPU POLY_NODE pick shader for the curve mesh item types.
//
// Draws the same connected mesh as `curve_pick.wgsl` but writes the global node
// index of the segment endpoint nearer the fragment into the primitive channel,
// so a POLY_NODE pick reads the final node id straight off the target with no
// CPU refinement, at both point and rect.
//
// Reads the hit triangle from `@builtin(primitive_index)` to index the
// per-triangle node payload, so this variant is only built on a device with the
// primitive-index feature.
//
// Group 0: the shared scene bind group.
// Group 1: per-draw model matrix + object id.
// Group 2: the item's per-triangle segment-endpoint payload.

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

// One triangle's segment endpoints: the two control points and their global
// node indices. 32-byte stride (the vec3 keeps 16-byte alignment).
struct NodePair {
    p0: vec3<f32>,
    i0: u32,
    p1: vec3<f32>,
    i1: u32,
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

// #include "helpers/clip_volume_test.wgsl"

@group(1) @binding(0) var<uniform> pick: PickInstance;
@group(2) @binding(0) var<storage, read> tri_nodes: array<NodePair>;

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
fn fs_main(in: VertexOut, @builtin(primitive_index) prim: u32) -> FragOut {
    if !clip_volume_test(in.world_pos) { discard; }
    let model = mat4x4<f32>(pick.model_c0, pick.model_c1, pick.model_c2, pick.model_c3);

    let np = tri_nodes[prim];
    let p0 = (model * vec4<f32>(np.p0, 1.0)).xyz;
    let p1 = (model * vec4<f32>(np.p1, 1.0)).xyz;
    let d0 = distance(p0, in.world_pos);
    let d1 = distance(p1, in.world_pos);

    var out: FragOut;
    out.object_id = pick.object_id;
    out.primitive_id = select(np.i1, np.i0, d0 <= d1);
    out.depth = in.clip_pos.z;
    return out;
}
