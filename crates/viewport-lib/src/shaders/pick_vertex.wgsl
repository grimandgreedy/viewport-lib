// GPU surface VERTEX pick shader.
//
// Like pick_id.wgsl, but the fragment writes the global index of the hit
// triangle's nearest corner (by world distance to the fragment position) into
// the primitive channel. A VERTEX pick then reads the final vertex id straight
// from that channel with no CPU refinement, at both point and rect. The nearest
// corner is chosen exactly as the CPU `nearest_triangle_vertex` helper does.
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

// #include "helpers/clip_volume_test.wgsl"

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
    let i0 = mesh_indices[base];
    let i1 = mesh_indices[base + 1u];
    let i2 = mesh_indices[base + 2u];

    let p0 = (model * vec4<f32>(vertex_position(i0), 1.0)).xyz;
    let p1 = (model * vec4<f32>(vertex_position(i1), 1.0)).xyz;
    let p2 = (model * vec4<f32>(vertex_position(i2), 1.0)).xyz;

    let d0 = distance(p0, in.world_pos);
    let d1 = distance(p1, in.world_pos);
    let d2 = distance(p2, in.world_pos);

    var best = i0;
    var best_d = d0;
    if d1 < best_d { best = i1; best_d = d1; }
    if d2 < best_d { best = i2; best_d = d2; }

    var out: FragOut;
    out.object_id = in.object_id;
    out.primitive_id = best;
    out.depth = in.clip_pos.z;
    return out;
}
