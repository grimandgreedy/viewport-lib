// GPU object-ID pick shader for volume surface slices.
//
// The vertex stage mirrors volume_surface_slice.wgsl's vs_main exactly: it
// reads the same mesh vertex buffer and applies the same model matrix from
// VolumeSurfaceSliceUniform (group 1, reused unchanged from the render bind
// group), so the pick surface matches the rendered slice mesh. Object-level
// only: the primitive channel is a constant 0.
//
// Group 0: minimal pick camera (binding 0 camera, binding 6 clip volume).
// Group 1: VolumeSurfaceSliceUniform (binding 0), reused unchanged from the
//          render bind group. Bindings 1-4 (volume texture, samplers, LUT)
//          are present but unused here.
// Group 2: per-item pick id (binding 0).

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

struct VolumeSurfaceSliceUniform {
    model:      mat4x4<f32>,  // offset   0, size 64
    bbox_min:   vec3<f32>,    // offset  64, size 12  (align 16, fits at 64)
    scalar_min: f32,           // offset  76, size  4
    bbox_max:   vec3<f32>,    // offset  80, size 12  (align 16, fits at 80)
    scalar_max: f32,           // offset  92, size  4
    opacity:    f32,           // offset  96, size  4
    // struct size = roundUp(100, 16) = 112 -- matches the Rust repr(C) layout
};

// Per-item pick id.
struct PickId {
    object_id: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

@group(1) @binding(0) var<uniform> slice_ub: VolumeSurfaceSliceUniform;

@group(2) @binding(0) var<uniform> pick: PickId;

// #include "clip_volume_test.wgsl"

// Matches the standard mesh Vertex layout; only position is read here.
struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = (slice_ub.model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
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
    out.object_id    = pick.object_id;
    out.primitive_id = 0u;
    out.depth        = in.clip_pos.z;
    return out;
}
