// GPU surface EDGE pick shader.
//
// Like pick_vertex.wgsl, but the fragment writes the id of the hit triangle's
// nearest edge (by distance from the edge segment to the fragment position) into
// the primitive channel. The edge id is `primitive_index * 3 + local_edge`, where
// local edge 0 is corner (0,1), 1 is (1,2), 2 is (2,0). An EDGE pick reads the
// final edge id straight from the channel with no CPU refinement, at point and
// rect.
//
// Reads the hit triangle from `@builtin(primitive_index)`, so this variant is
// only built on devices with SHADER_PRIMITIVE_INDEX.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
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

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;
@group(1) @binding(0) var<storage, read> pick_instances: array<PickInstance>;
// The mesh vertex buffer viewed as raw f32s (64-byte / 16-float stride, position
// in the first three floats) and its triangle index buffer.
@group(2) @binding(0) var<storage, read> mesh_floats: array<f32>;
@group(2) @binding(1) var<storage, read> mesh_indices: array<u32>;

// #include "clip_volume_test.wgsl"

struct VertexIn {
    @location(0) position: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) object_id: u32,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) instance_id: u32,
};

@vertex
fn vs_main(in: VertexIn, @builtin(instance_index) idx: u32) -> VertexOut {
    let inst = pick_instances[idx];
    let model = mat4x4<f32>(inst.model_c0, inst.model_c1, inst.model_c2, inst.model_c3);
    let world = model * vec4<f32>(in.position, 1.0);
    var out: VertexOut;
    out.clip_pos = camera.view_proj * world;
    out.object_id = inst.object_id;
    out.world_pos = world.xyz;
    out.instance_id = idx;
    return out;
}

fn vertex_position(vi: u32) -> vec3<f32> {
    let o = vi * 16u;
    return vec3<f32>(mesh_floats[o], mesh_floats[o + 1u], mesh_floats[o + 2u]);
}

// Distance from point `p` to the segment `a`-`b`.
fn edge_distance(a: vec3<f32>, b: vec3<f32>, p: vec3<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return distance(a + t * ab, p);
}

struct FragOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
};

@fragment
fn fs_main(in: VertexOut, @builtin(primitive_index) prim: u32) -> FragOut {
    if !clip_volume_test(in.world_pos) { discard; }
    let inst = pick_instances[in.instance_id];
    let model = mat4x4<f32>(inst.model_c0, inst.model_c1, inst.model_c2, inst.model_c3);

    let base = prim * 3u;
    let p0 = (model * vec4<f32>(vertex_position(mesh_indices[base]), 1.0)).xyz;
    let p1 = (model * vec4<f32>(vertex_position(mesh_indices[base + 1u]), 1.0)).xyz;
    let p2 = (model * vec4<f32>(vertex_position(mesh_indices[base + 2u]), 1.0)).xyz;

    let d0 = edge_distance(p0, p1, in.world_pos);
    let d1 = edge_distance(p1, p2, in.world_pos);
    let d2 = edge_distance(p2, p0, in.world_pos);

    var best = 0u;
    var best_d = d0;
    if d1 < best_d { best = 1u; best_d = d1; }
    if d2 < best_d { best = 2u; best_d = d2; }

    var out: FragOut;
    out.object_id = in.object_id;
    out.primitive_id = base + best;
    out.depth = in.clip_pos.z;
    return out;
}
