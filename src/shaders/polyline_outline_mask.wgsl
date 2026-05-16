// polyline_outline_mask.wgsl : renders polyline segments as screen-space quads
// into the R8 mask texture.  Replicates the geometry expansion from polyline.wgsl
// but outputs white rather than interpolated colour.  No clip plane testing.
//
// Group 0: camera_bind_group_layout (binding 0 only).
// Group 1: polyline_bgl (binding 0 only; texture and sampler bindings unused).
//
// Instance input: same 112-byte per-segment format as polyline.wgsl.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct PolylineUniform {
    default_colour:   vec4<f32>,
    line_width:      f32,
    scalar_min:      f32,
    scalar_max:      f32,
    has_scalar:      u32,
    viewport_width:  f32,
    viewport_height: f32,
    _pad:            vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera:     Camera;
@group(1) @binding(0) var<uniform> pl_uniform: PolylineUniform;

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

fn to_screen(p: vec3<f32>) -> vec2<f32> {
    let clip = camera.view_proj * vec4<f32>(p, 1.0);
    let w = max(clip.w, 0.0001f);
    let ndc = clip.xy / w;
    return ndc * vec2<f32>(pl_uniform.viewport_width * 0.5f,
                           pl_uniform.viewport_height * 0.5f);
}

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
) -> @builtin(position) vec4<f32> {
    let use_b     = (vid == 1u || vid == 3u || vid == 4u);
    let use_right = (vid == 2u || vid == 4u || vid == 5u);
    let pos  = select(seg.pos_a, seg.pos_b, use_b);
    let side = select(-1.0f, 1.0f, use_right);

    let screen_prev = to_screen(seg.prev_pos);
    let screen_a    = to_screen(seg.pos_a);
    let screen_b    = to_screen(seg.pos_b);
    let screen_next = to_screen(seg.next_pos);

    let ab_vec = screen_b - screen_a;
    let ab_len = length(ab_vec);
    let dir_ab = select(vec2<f32>(1.0f, 0.0f), ab_vec / ab_len, ab_len > 0.001f);

    var extrusion_a: vec2<f32>;
    if seg.has_prev != 0u {
        let pa_vec = screen_a - screen_prev;
        let pa_len = length(pa_vec);
        let dir_pa = select(dir_ab, pa_vec / pa_len, pa_len > 0.001f);
        extrusion_a = miter_extrusion(dir_pa, dir_ab);
    } else {
        extrusion_a = vec2<f32>(-dir_ab.y, dir_ab.x);
    }

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
    var clip_pos = camera.view_proj * vec4<f32>(pos, 1.0f);
    let radius = select(seg.radius_a, seg.radius_b, use_b);

    let half_w = radius * 0.5f;
    let ndc_offset = side * half_w * extrusion
        * vec2<f32>(2.0f / pl_uniform.viewport_width, 2.0f / pl_uniform.viewport_height);
    clip_pos.x += ndc_offset.x * clip_pos.w;
    clip_pos.y += ndc_offset.y * clip_pos.w;

    return clip_pos;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
