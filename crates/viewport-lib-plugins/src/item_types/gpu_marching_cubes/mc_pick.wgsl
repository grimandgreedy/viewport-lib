// GPU marching-cubes object-pick shader.
//
// The marching-cubes surface is generated into a storage/vertex buffer of
// world-space 24-byte vertices (position + normal) and drawn indirectly. This
// pick shader reuses that vertex buffer, reads only the position, and writes the
// job's pick id and the fragment depth into the three pick targets. Object-level:
// the primitive channel is 0.
//
// Group 0: the shared scene bind group, whose pick camera bind group the
//          surface pick pipeline binds too. The clip volume at binding 6 is
//          present in the layout but unused here.
// Group 1: per-job pick id (binding 0).

@group(1) @binding(0) var<uniform> pick_id: vec4<u32>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    // Normal (location 1) is present in the 24-byte vertex but unused here.
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.position, 1.0);
    return out;
}

struct PickOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
};

@fragment
fn fs_main(in: VertexOut) -> PickOut {
    var out: PickOut;
    out.object_id = pick_id.x;
    out.primitive_id = 0u;
    out.depth = in.clip_pos.z;
    return out;
}
