// GPU object-ID pick shader for polylines.
//
// The vertex stage is a copy of polyline.wgsl's vs_main: it reads the same
// 112-byte per-segment instance buffer and performs the identical screen-space
// miter/cap expansion, so the picked ribbon matches the rendered ribbon exactly.
// The only difference is the fragment output: instead of a colour it writes the
// per-item object id (from a small group-2 uniform) and the clip-space depth,
// matching the pick pass's two render targets (R32Uint id + R32Float depth).
//
// Group 0: pick camera bind group (Camera at binding 0, ClipVolume at binding 6),
//          matching the surface pick pipeline's group 0.
// Group 1: the polyline render item's own bind group (PolylineUniform + LUT +
//          sampler). Reused unchanged so the model matrix and viewport dims that
//          drive the expansion are exactly those the render path used.
// Group 2: per-draw pick id uniform (one polyline item = one draw).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

// Polyline per-item uniform : 112 bytes. Layout matches polyline.wgsl.
struct PolylineUniform {
    model:           mat4x4<f32>,
    default_colour:   vec4<f32>,
    line_width:      f32,
    scalar_min:      f32,
    scalar_max:      f32,
    has_scalar:      u32,
    viewport_width:  f32,
    viewport_height: f32,
    _pad:            vec2<f32>,
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

struct PickId {
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

// #include "clip_volume_test.wgsl"

@group(1) @binding(0) var<uniform> pl_uniform:  PolylineUniform;
@group(1) @binding(1) var          lut_texture: texture_2d<f32>;
@group(1) @binding(2) var          lut_sampler: sampler;

@group(2) @binding(0) var<uniform> pick: PickId;

// Per-segment instance data (112 bytes). Layout matches polyline.wgsl.
struct SegmentIn {
    @location(0)  pos_a:            vec3<f32>,
    @location(1)  pos_b:            vec3<f32>,
    @location(2)  prev_pos:         vec3<f32>,
    @location(3)  next_pos:         vec3<f32>,
    @location(4)  scalar_a:         f32,
    @location(5)  scalar_b:         f32,
    @location(6)  has_prev:         u32,
    @location(7)  has_next:         u32,
    @location(8)  colour_a:          vec4<f32>,
    @location(9)  colour_b:          vec4<f32>,
    @location(10) radius_a:         f32,
    @location(11) radius_b:         f32,
    @location(12) use_direct_colour: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

// Apply the per-item model matrix to a position in the consumer's input space.
fn apply_model(p: vec3<f32>) -> vec3<f32> {
    return (pl_uniform.model * vec4<f32>(p, 1.0)).xyz;
}

// Project a position to screen pixels after applying the model matrix.
fn to_screen(p: vec3<f32>) -> vec2<f32> {
    let world = apply_model(p);
    let clip = camera.view_proj * vec4<f32>(world, 1.0);
    let w = max(clip.w, 0.0001f);
    let ndc = clip.xy / w;
    return ndc * vec2<f32>(pl_uniform.viewport_width * 0.5f,
                           pl_uniform.viewport_height * 0.5f);
}

// Miter extrusion vector, identical to polyline.wgsl.
fn miter_extrusion(dir_in: vec2<f32>, dir_out: vec2<f32>) -> vec2<f32> {
    let perp_in  = vec2<f32>(-dir_in.y,  dir_in.x);
    let perp_out = vec2<f32>(-dir_out.y, dir_out.x);
    let bisect   = normalize(perp_in + perp_out);
    let cos_half = max(dot(bisect, perp_out), 0.25f);
    return bisect / cos_half;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    seg: SegmentIn,
) -> VertexOut {
    let use_b   = (vid == 1u || vid == 3u || vid == 4u);
    let use_right = (vid == 2u || vid == 4u || vid == 5u);
    let pos    = select(seg.pos_a, seg.pos_b, use_b);
    let side   = select(-1.0f, 1.0f, use_right);

    let screen_prev = to_screen(seg.prev_pos);
    let screen_a    = to_screen(seg.pos_a);
    let screen_b    = to_screen(seg.pos_b);
    let screen_next = to_screen(seg.next_pos);

    let ab_vec = screen_b - screen_a;
    let ab_len = length(ab_vec);
    let dir_ab = select(vec2<f32>(1.0f, 0.0f), ab_vec / ab_len, ab_len > 0.001f);

    // --- Miter / cap at A ---
    var extrusion_a: vec2<f32>;
    if seg.has_prev != 0u {
        let pa_vec = screen_a - screen_prev;
        let pa_len = length(pa_vec);
        let dir_pa = select(dir_ab, pa_vec / pa_len, pa_len > 0.001f);
        extrusion_a = miter_extrusion(dir_pa, dir_ab);
    } else {
        extrusion_a = vec2<f32>(-dir_ab.y, dir_ab.x);
    }

    // --- Miter / cap at B ---
    var extrusion_b: vec2<f32>;
    if seg.has_next != 0u {
        let bn_vec = screen_next - screen_b;
        let bn_len = length(bn_vec);
        let dir_bn = select(dir_ab, bn_vec / bn_len, bn_len > 0.001f);
        extrusion_b = miter_extrusion(dir_ab, dir_bn);
    } else {
        extrusion_b = vec2<f32>(-dir_ab.y, dir_ab.x);
    }

    let extrusion = select(extrusion_a, extrusion_b, use_b);

    let world_pos = apply_model(pos);
    var clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0f);

    let radius = select(seg.radius_a, seg.radius_b, use_b);

    let half_w = radius * 0.5f;
    let ndc_offset = side * half_w * extrusion
        * vec2<f32>(2.0f / pl_uniform.viewport_width, 2.0f / pl_uniform.viewport_height);
    clip_pos.x += ndc_offset.x * clip_pos.w;
    clip_pos.y += ndc_offset.y * clip_pos.w;

    var out: VertexOut;
    out.clip_pos  = clip_pos;
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
