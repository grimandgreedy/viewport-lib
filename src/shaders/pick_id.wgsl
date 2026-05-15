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
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;
@group(1) @binding(0) var<storage, read> pick_instances: array<PickInstance>;

fn clip_volume_test(p: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_volume.count; i = i + 1u) {
        let e = clip_volume.volumes[i];
        if e.volume_type == 2u {
            let d = p - e.center;
            let local = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
            if abs(local.x) > e.half_extents.x
                || abs(local.y) > e.half_extents.y
                || abs(local.z) > e.half_extents.z {
                return false;
            }
        } else if e.volume_type == 3u {
            let ds = p - e.center;
            if dot(ds, ds) > e.radius * e.radius { return false; }
        } else if e.volume_type == 4u {
            let axis = e.col0;
            let d = p - e.center;
            let along = dot(d, axis);
            if abs(along) > e.half_extents.x { return false; }
            let radial = d - axis * along;
            if dot(radial, radial) > e.radius * e.radius { return false; }
        }
    }
    return true;
}

struct VertexIn {
    @location(0) position: vec3<f32>,
    // Other vertex attributes (normal, colour, uv, tangent) are present in the
    // buffer but ignored : the 64-byte stride handles them automatically.
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0) @interpolate(flat) object_id: u32,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn, @builtin(instance_index) idx: u32) -> VertexOut {
    let inst = pick_instances[idx];
    let model = mat4x4<f32>(inst.model_c0, inst.model_c1, inst.model_c2, inst.model_c3);
    let world = model * vec4<f32>(in.position, 1.0);
    var out: VertexOut;
    out.clip_pos  = camera.view_proj * world;
    out.object_id = inst.object_id;
    out.world_pos = world.xyz;
    return out;
}

struct FragOut {
    @location(0) object_id: u32,
    @location(1) depth: f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    if !clip_volume_test(in.world_pos) { discard; }
    var out: FragOut;
    out.object_id = in.object_id;
    // clip_pos.z in the fragment stage is the depth buffer value in [0, 1].
    out.depth = in.clip_pos.z;
    return out;
}
