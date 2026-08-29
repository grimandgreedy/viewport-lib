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
// Group 0: camera uniform (same layout as polyline.wgsl group 0).
// Group 1: binding 0 = segment storage buffer.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<storage, read> seg_data: array<f32>;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
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
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.75, 0.75, 0.75, 1.0);
}
