// Polyline shader for the 3D viewport : screen-space thick lines with miter joints.
//
// Each draw call covers all segments for one polyline item.  The vertex buffer
// is stepped per-instance (one entry per segment); 6 vertices are drawn per
// instance to form a screen-aligned quad (two triangles).
//
// Miter joints: at each interior vertex (shared by two consecutive segments),
// the extrusion direction is the bisector of the two segment perpendiculars.
// The extrusion length is scaled by 1/cos(half_angle) so the joint stays at
// constant width. Miters are clamped to avoid excessive length on sharp corners.
// At strip endpoints (has_prev=0 or has_next=0), a square cap is used instead.
//
// Group 0: Camera uniform + ClipPlanes + ClipVolume (matching camera_bgl layout).
// Group 1: PolylineUniform (line_width, scalar mapping, viewport dims, color)
//          + LUT texture (256×1 Rgba8Unorm)
//          + LUT sampler.
//
// Instance input (per segment, VertexStepMode::Instance, 112 bytes):
//   location  0 : pos_a             vec3   segment start (world space)
//   location  1 : pos_b             vec3   segment end   (world space)
//   location  2 : prev_pos          vec3   point before A (== pos_a if strip start)
//   location  3 : next_pos          vec3   point after  B (== pos_b if strip end)
//   location  4 : scalar_a          f32    scalar at A
//   location  5 : scalar_b          f32    scalar at B
//   location  6 : has_prev          u32    1 = interior join at A, 0 = square cap
//   location  7 : has_next          u32    1 = interior join at B, 0 = square cap
//   location  8 : color_a           vec4   direct RGBA at A
//   location  9 : color_b           vec4   direct RGBA at B
//   location 10 : radius_a          f32    line width in px at A
//   location 11 : radius_b          f32    line width in px at B
//   location 12 : use_direct_color  u32    1 = use color_a/b, 0 = LUT / default

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

// Polyline per-item uniform : 48 bytes.
struct PolylineUniform {
    default_color:   vec4<f32>,  // offset  0 : 16 bytes
    line_width:      f32,        // offset 16 :  4 bytes (screen pixels)
    scalar_min:      f32,        // offset 20 :  4 bytes
    scalar_max:      f32,        // offset 24 :  4 bytes
    has_scalar:      u32,        // offset 28 :  4 bytes
    viewport_width:  f32,        // offset 32 :  4 bytes
    viewport_height: f32,        // offset 36 :  4 bytes
    _pad:            vec2<f32>,  // offset 40 :  8 bytes
};                               // total 48 bytes

struct ClipVolumeUB {
    volume_type: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    plane_normal: vec3<f32>,
    plane_dist: f32,
    box_center: vec3<f32>,
    _padB0: f32,
    box_half_extents: vec3<f32>,
    _padB1: f32,
    box_col0: vec3<f32>,
    _padB2: f32,
    box_col1: vec3<f32>,
    _padB3: f32,
    box_col2: vec3<f32>,
    _padB4: f32,
    sphere_center: vec3<f32>,
    sphere_radius: f32,
};

@group(0) @binding(0) var<uniform> camera:     Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

fn clip_volume_test(p: vec3<f32>) -> bool {
    if clip_volume.volume_type == 0u { return true; }
    if clip_volume.volume_type == 1u {
        return dot(p, clip_volume.plane_normal) + clip_volume.plane_dist >= 0.0;
    }
    if clip_volume.volume_type == 2u {
        let d = p - clip_volume.box_center;
        let local = vec3<f32>(
            dot(d, clip_volume.box_col0),
            dot(d, clip_volume.box_col1),
            dot(d, clip_volume.box_col2),
        );
        return abs(local.x) <= clip_volume.box_half_extents.x
            && abs(local.y) <= clip_volume.box_half_extents.y
            && abs(local.z) <= clip_volume.box_half_extents.z;
    }
    let ds = p - clip_volume.sphere_center;
    return dot(ds, ds) <= clip_volume.sphere_radius * clip_volume.sphere_radius;
}

@group(1) @binding(0) var<uniform> pl_uniform:  PolylineUniform;
@group(1) @binding(1) var          lut_texture: texture_2d<f32>;
@group(1) @binding(2) var          lut_sampler: sampler;

// Per-segment instance data (112 bytes).
struct SegmentIn {
    @location(0)  pos_a:            vec3<f32>,
    @location(1)  pos_b:            vec3<f32>,
    @location(2)  prev_pos:         vec3<f32>,
    @location(3)  next_pos:         vec3<f32>,
    @location(4)  scalar_a:         f32,
    @location(5)  scalar_b:         f32,
    @location(6)  has_prev:         u32,
    @location(7)  has_next:         u32,
    @location(8)  color_a:          vec4<f32>,
    @location(9)  color_b:          vec4<f32>,
    @location(10) radius_a:         f32,
    @location(11) radius_b:         f32,
    @location(12) use_direct_color: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       color:     vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
};

// Project a world-space point to screen pixels.
// Returns vec2 in pixel coordinates (origin at viewport center).
fn to_screen(p: vec3<f32>) -> vec2<f32> {
    let clip = camera.view_proj * vec4<f32>(p, 1.0);
    let w = max(clip.w, 0.0001f);
    let ndc = clip.xy / w;
    return ndc * vec2<f32>(pl_uniform.viewport_width * 0.5f,
                           pl_uniform.viewport_height * 0.5f);
}

// Compute the miter extrusion vector for a junction between two directions.
// dir_in and dir_out are the normalized screen-space directions of the incoming
// and outgoing segments. Returns a vector whose length accounts for the miter
// scale factor (1/cos(half_angle)), clamped to avoid extreme extension.
// The returned vector points to the "left" side of the outgoing direction.
fn miter_extrusion(dir_in: vec2<f32>, dir_out: vec2<f32>) -> vec2<f32> {
    let perp_in  = vec2<f32>(-dir_in.y,  dir_in.x);
    let perp_out = vec2<f32>(-dir_out.y, dir_out.x);
    let bisect   = normalize(perp_in + perp_out);
    // dot(bisect, perp_out) = cos(half_angle).  Clamp to 0.25 to cap miter at 4× width.
    let cos_half = max(dot(bisect, perp_out), 0.25f);
    return bisect / cos_half;
}

// Corner layout : 6 vertices per segment (TriangleList):
//   Triangle 0: v0=A-left,  v1=B-left,  v2=A-right
//   Triangle 1: v3=B-left,  v4=B-right, v5=A-right
//
//   vid  0 1 2 3 4 5
//   useB F T F T T F   (use pos_b / scalar_b endpoint)
//   right F F T F T T  (offset to the right of the direction)

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    seg: SegmentIn,
) -> VertexOut {
    let use_b   = (vid == 1u || vid == 3u || vid == 4u);
    let use_right = (vid == 2u || vid == 4u || vid == 5u);
    let pos    = select(seg.pos_a, seg.pos_b, use_b);
    let scalar = select(seg.scalar_a, seg.scalar_b, use_b);
    let side   = select(-1.0f, 1.0f, use_right);

    // Project all relevant world-space points to screen pixels.
    let screen_prev = to_screen(seg.prev_pos);
    let screen_a    = to_screen(seg.pos_a);
    let screen_b    = to_screen(seg.pos_b);
    let screen_next = to_screen(seg.next_pos);

    // Normalized screen-space direction of this segment.
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
        // Square cap: just the perpendicular to AB.
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
        // Square cap: just the perpendicular to AB.
        extrusion_b = vec2<f32>(-dir_ab.y, dir_ab.x);
    }

    // Select the extrusion for this corner's endpoint.
    let extrusion = select(extrusion_a, extrusion_b, use_b);

    // Base clip-space position.
    var clip_pos = camera.view_proj * vec4<f32>(pos, 1.0f);

    // Per-vertex radius: interpolate between endpoint radii (baked from node_radii or line_width).
    let radius = select(seg.radius_a, seg.radius_b, use_b);

    // Apply half-width offset in screen space, converted to clip-space offset.
    // NDC offset = (pixels / viewport_px) * 2.  Clip offset = NDC offset * w.
    let half_w = radius * 0.5f;
    let ndc_offset = side * half_w * extrusion
        * vec2<f32>(2.0f / pl_uniform.viewport_width, 2.0f / pl_uniform.viewport_height);
    clip_pos.x += ndc_offset.x * clip_pos.w;
    clip_pos.y += ndc_offset.y * clip_pos.w;

    var out: VertexOut;
    out.clip_pos  = clip_pos;
    out.world_pos = pos;

    // Color priority: direct RGBA > scalar LUT > default_color.
    if seg.use_direct_color != 0u {
        out.color = select(seg.color_a, seg.color_b, use_b);
    } else if pl_uniform.has_scalar != 0u {
        let range = pl_uniform.scalar_max - pl_uniform.scalar_min;
        let t = select(0.0f, (scalar - pl_uniform.scalar_min) / range, range > 0.0f);
        let u = clamp(t, 0.0f, 1.0f);
        out.color = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(u, 0.5f), 0.0f);
    } else {
        out.color = pl_uniform.default_color;
    }

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Half-space clip-plane culling (section views).
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0f), plane) < 0.0f {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }
    return in.color;
}
