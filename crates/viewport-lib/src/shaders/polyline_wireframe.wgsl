// Polyline wireframe shader: thin 1px LineList representation.
//
// Reads segment endpoints directly from the segment storage buffer using
// instance_index and vertex_index. Each instance is one segment; each segment
// draws two vertices (LineList topology), producing a single 1px GPU line.
//
// The storage buffer holds the same 112-byte segment layout used by the thick
// polyline vertex buffer (28 floats per segment):
//   floats 0-2 : pos_a (segment start, world space)
//   floats 3-5 : pos_b (segment end, world space)
//   floats 6+  : ignored by this shader
//
// Group 0: camera uniform + ClipPlanes + ClipVolume (matching camera_bgl
// layout, same as polyline.wgsl's group 0 -- the pipeline layout is shared
// with the thick-line pipeline, this module just did not declare or use the
// clip bindings until now).
// Group 1: binding 0 = segment storage buffer.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:  u32,
    _pad0:  u32,
    viewport_width:  f32,
    viewport_height: f32,
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

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

// #include "helpers/clip_volume_test.wgsl"

@group(1) @binding(0) var<storage, read> seg_data: array<f32>;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index)   vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    // 112 bytes per segment = 28 floats.
    let base = iid * 28u;
    let is_b = vid == 1u;
    let pos = select(
        vec3<f32>(seg_data[base],       seg_data[base + 1u], seg_data[base + 2u]),
        vec3<f32>(seg_data[base + 3u],  seg_data[base + 4u], seg_data[base + 5u]),
        is_b,
    );
    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Half-space clip-plane culling (section views), matching polyline.wgsl's
    // thick-line fragment stage. Previously absent: a wireframe polyline
    // silently ignored every active clip plane and clip volume regardless of
    // `ItemSettings.ignore_clip`.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }
    return vec4<f32>(0.75, 0.75, 0.75, 1.0);
}

// Clip-exempt variant: used for clip object overlays (box/sphere/cylinder
// wireframes) and any polyline with `ItemSettings.ignore_clip = true`.
@fragment
fn fs_main_no_clip(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.75, 0.75, 0.75, 1.0);
}
