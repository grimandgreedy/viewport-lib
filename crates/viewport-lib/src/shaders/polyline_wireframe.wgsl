// Polyline wireframe shader: thin 1px LineList representation.
//
// Each instance is one segment and draws two vertices (LineList topology),
// producing a single 1px GPU line. The segment endpoints arrive as instance
// attributes from the same 112-byte segment buffer the thick polyline
// pipeline reads: pos_a (offset 0) at location 0 and pos_b (offset 12) at
// location 1. No storage buffer, so the same pipeline runs on every tier.
//
// Group 0: camera uniform + ClipPlanes + ClipVolume (matching the scene group
// 0 layout, as polyline.wgsl does).

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
// #include "helpers/scene_fragment.wgsl"

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

// One instance per segment, from the same 112-byte segment records the thick
// line pipeline reads: only the two endpoints are bound.
@vertex
fn vs_main(
    @builtin(vertex_index) vid:   u32,
    @location(0)           pos_a: vec3<f32>,
    @location(1)           pos_b: vec3<f32>,
) -> VertexOut {
    let pos = select(pos_a, pos_b, vid == 1u);
    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    return out;
}

fn fs_main_body(in: VertexOut) -> vec4<f32> {
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

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return fs_main_body(in);
}

// `fs_main` for the HDR scene pass, with the indirect fraction.
@fragment
fn fs_scene(in: VertexOut) -> SceneFragment {
    let colour = fs_main_body(in);
    return scene_fragment(colour, scene_indirect_fraction);
}

// Clip-exempt variant: used for clip object overlays (box/sphere/cylinder
// wireframes) and any polyline with `ItemSettings.ignore_clip = true`.
fn fs_main_no_clip_body(in: VertexOut) -> vec4<f32> {
    return vec4<f32>(0.75, 0.75, 0.75, 1.0);
}

@fragment
fn fs_main_no_clip(in: VertexOut) -> @location(0) vec4<f32> {
    return fs_main_no_clip_body(in);
}

// `fs_main_no_clip` for the HDR scene pass, with the indirect fraction.
@fragment
fn fs_scene_no_clip(in: VertexOut) -> SceneFragment {
    let colour = fs_main_no_clip_body(in);
    return scene_fragment(colour, scene_indirect_fraction);
}
