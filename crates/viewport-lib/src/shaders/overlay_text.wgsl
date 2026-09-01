// Screen-space overlay text and solid quad shader.
//
// Renders glyph quads (textured from the glyph atlas) and solid-colour quads
// (background boxes, leader lines) in a single batched draw call.
//
// Vertex attributes carry NDC position, atlas UV, RGBA tint, a flag distinguishing
// textured vs solid fragments, and an optional clip: a framebuffer-pixel bounding
// box plus an index into a shared clip-shape buffer for SDF and nested clipping.
// The SDF functions mirror overlay_shape.wgsl so text clips to the same masks as
// shapes.

@group(0) @binding(0) var glyph_atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

// One clip-mask shape (framebuffer pixels). `params` = (shape_type, rotation,
// parent_index, invert). Matches ClipShapeGpu on the CPU and overlay_shape.wgsl.
struct ClipShape {
    center:    vec2<f32>,
    half_size: vec2<f32>,
    radii:     vec4<f32>,
    params:    vec4<f32>,
    pivot:     vec2<f32>,
    pad:       vec2<f32>,
};

@group(0) @binding(2) var<storage, read> clip_shapes: array<ClipShape>;

// Viewport size in logical pixels (xy); zw padding. Maps local-pixel vertex
// positions to NDC so overlay geometry is independent of the viewport size.
struct Viewport {
    size: vec2<f32>,
    pad:  vec2<f32>,
};
@group(0) @binding(3) var<uniform> viewport: Viewport;

struct VertexInput {
    @location(0) position:    vec2<f32>,  // local logical pixels
    @location(1) uv:          vec2<f32>,  // atlas UV (ignored for solid quads)
    @location(2) colour:      vec4<f32>,  // RGBA tint
    @location(3) use_texture: f32,        // 1.0 = sample atlas, 0.0 = solid
    @location(4) clip_index:  f32,        // clip-shape index, or -1 for none
    @location(5) clip_rect:   vec4<f32>,  // framebuffer clip bbox (x0,y0,x1,y1); all zero = none
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       uv:           vec2<f32>,
    @location(1)       colour:       vec4<f32>,
    @location(2)       use_texture:  f32,
    @location(3) @interpolate(flat) clip_index: f32,
    @location(4) @interpolate(flat) clip_rect:  vec4<f32>,
};

// Map a local logical-pixel position to NDC using the viewport size.
fn px_to_ndc(px: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(px.x / viewport.size.x * 2.0 - 1.0, 1.0 - px.y / viewport.size.y * 2.0);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(px_to_ndc(in.position), 0.0, 1.0);
    out.uv            = in.uv;
    out.colour        = in.colour;
    out.use_texture   = in.use_texture;
    out.clip_index    = in.clip_index;
    out.clip_rect     = in.clip_rect;
    return out;
}

// --- SDF functions (mirror of overlay_shape.wgsl) --------------------------------

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    var rs = r;
    if (p.x > 0.0) {
        rs = vec4<f32>(rs.x, rs.y, rs.z, rs.w);
    } else {
        rs = vec4<f32>(rs.w, rs.z, rs.y, rs.x);
    }
    if (p.y > 0.0) {
        rs.x = rs.y;
    }
    let q = abs(p) - b + rs.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - rs.x;
}

fn sd_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sd_ellipse(p: vec2<f32>, ab: vec2<f32>) -> f32 {
    let pa = abs(p);
    let ei = 1.0 / ab;
    let e2 = ab * ab;
    let ve = ei * vec2<f32>(e2.x - e2.y, e2.y - e2.x);
    var t = vec2<f32>(0.70710678118);
    for (var i = 0; i < 3; i = i + 1) {
        let v = ve * t * t * t;
        let u = normalize(pa - v) * length(t * ab - v);
        let w = ei * (v + u);
        t = normalize(clamp(w, vec2<f32>(0.0), vec2<f32>(1.0)));
    }
    let nearest = t * ab;
    let d = length(pa - nearest);
    let np = pa / ab;
    let inside = dot(np, np);
    return select(d, -d, inside < 1.0);
}

fn sd_capsule(p: vec2<f32>, half_size: vec2<f32>) -> f32 {
    let r = min(half_size.x, half_size.y);
    var q = abs(p);
    if (half_size.x > half_size.y) {
        q.x = q.x - (half_size.x - r);
    } else {
        q.y = q.y - (half_size.y - r);
    }
    return length(max(q, vec2<f32>(0.0))) - r;
}

fn sd_ring(p: vec2<f32>, outer_r: f32, inner_frac: f32) -> f32 {
    let wall = outer_r * (1.0 - inner_frac) * 0.5;
    let mid_r = outer_r - wall;
    return abs(length(p) - mid_r) - wall;
}

fn sd_arc(p: vec2<f32>, outer_r: f32, inner_frac: f32, sa: f32, ea: f32) -> f32 {
    let d_ring = sd_ring(p, outer_r, inner_frac);
    let angle = atan2(p.y, p.x);
    let two_pi = 6.28318530718;
    let sweep = ((ea - sa) % two_pi + two_pi) % two_pi;
    let a = ((angle - sa) % two_pi + two_pi) % two_pi;
    if (a <= sweep) {
        return d_ring;
    }
    let wall = outer_r * (1.0 - inner_frac) * 0.5;
    let mid_r = outer_r - wall;
    let inner_r = mid_r - wall;
    let outer_edge = mid_r + wall;
    let cs = vec2<f32>(cos(sa), sin(sa));
    let ce = vec2<f32>(cos(ea), sin(ea));
    let proj_s = clamp(dot(p, cs), inner_r, outer_edge);
    let proj_e = clamp(dot(p, ce), inner_r, outer_edge);
    let ds = length(p - cs * proj_s);
    let de = length(p - ce * proj_e);
    return min(ds, de);
}

fn sd_triangle(p: vec2<f32>, hs: vec2<f32>) -> f32 {
    let q = vec2<f32>(abs(p.x), p.y);
    let e = vec2<f32>(hs.x, 2.0 * hs.y);
    let en = normalize(e);
    let n = vec2<f32>(en.y, -en.x);
    let d_edge = dot(q - vec2<f32>(0.0, -hs.y), n);
    let d_base = q.y - hs.y;
    return max(d_edge, d_base);
}

fn sd_line(p: vec2<f32>, hs: vec2<f32>, r: f32, square: bool) -> f32 {
    if (square) {
        let seg_len = length(hs);
        if (seg_len < 0.0001) {
            return length(p) - r;
        }
        let d = hs / seg_len;
        let along = dot(p, d);
        let perp  = dot(p, vec2<f32>(-d.y, d.x));
        let q = abs(vec2<f32>(along, perp)) - vec2<f32>(seg_len, r);
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
    } else {
        let ba = 2.0 * hs;
        let pa = p + hs;
        let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h) - r;
    }
}

fn sd_star_n(p: vec2<f32>, r: f32, n: f32, rf: f32) -> f32 {
    let ri = r * rf;
    let an = 3.14159265 / n;
    let two_an = 2.0 * an;
    let a = atan2(p.y, p.x);
    let a_mod = ((a % two_an) + two_an) % two_an;
    let a_abs = select(a_mod, two_an - a_mod, a_mod > an);
    let rp = length(p);
    let q = rp * vec2<f32>(cos(a_abs), sin(a_abs));
    let outer = vec2<f32>(r, 0.0);
    let inner = vec2<f32>(ri * cos(an), ri * sin(an));
    let ba = inner - outer;
    let qa = q - outer;
    let t = clamp(dot(qa, ba) / dot(ba, ba), 0.0, 1.0);
    let d = length(qa - ba * t);
    let cross_val = qa.x * ba.y - qa.y * ba.x;
    return d * select(1.0, -1.0, cross_val < 0.0);
}

fn sd_ngon(p: vec2<f32>, r: f32, n: f32) -> f32 {
    let an = 3.14159265 / n;
    let two_an = 2.0 * an;
    let a = atan2(p.y, p.x) + an;
    let a_mod = ((a % two_an) + two_an) % two_an;
    let a_abs = select(a_mod, two_an - a_mod, a_mod > an);
    let rp = length(p);
    let q = rp * vec2<f32>(cos(a_abs), sin(a_abs));
    let he = r * cos(an);
    let hv = r * sin(an);
    let dx = q.x - he;
    let dy = max(q.y - hv, 0.0);
    return select(dx, sqrt(dx * dx + dy * dy), dy > 0.0);
}

fn sd_cross(p: vec2<f32>, hs: vec2<f32>, arm_frac: f32) -> f32 {
    let arm_w = arm_frac * min(hs.x, hs.y);
    let q_h = abs(p) - vec2<f32>(hs.x, arm_w);
    let q_v = abs(p) - vec2<f32>(arm_w, hs.y);
    let d_h = length(max(q_h, vec2<f32>(0.0))) + min(max(q_h.x, q_h.y), 0.0);
    let d_v = length(max(q_v, vec2<f32>(0.0))) + min(max(q_v.x, q_v.y), 0.0);
    return min(d_h, d_v);
}

fn eval_sdf(p: vec2<f32>, hs: vec2<f32>, shape_type: f32, radii: vec4<f32>) -> f32 {
    let st = i32(shape_type + 0.5);
    switch (st) {
        case 1: {
            return sd_circle(p, min(hs.x, hs.y));
        }
        case 2: {
            return sd_ellipse(p, hs);
        }
        case 3: {
            return sd_capsule(p, hs);
        }
        case 4: {
            return sd_ring(p, min(hs.x, hs.y), radii.x);
        }
        case 5: {
            return sd_arc(p, min(hs.x, hs.y), radii.x, radii.y, radii.z);
        }
        case 6: {
            let dir = i32(radii.x + 0.5);
            var tp = p;
            if (dir == 1) {
                tp.y = -tp.y;
            } else if (dir == 2) {
                tp = vec2<f32>(tp.y, tp.x);
            } else if (dir == 3) {
                tp = vec2<f32>(tp.y, -tp.x);
            }
            var ths = hs;
            if (dir >= 2) {
                ths = vec2<f32>(hs.y, hs.x);
            }
            return sd_triangle(tp, ths);
        }
        case 7: {
            return sd_line(p, hs, radii.x, radii.y > 0.5);
        }
        case 8: {
            return sd_star_n(p, min(hs.x, hs.y), radii.x, radii.y);
        }
        case 9: {
            return sd_ngon(p, min(hs.x, hs.y), radii.x);
        }
        case 10: {
            return sd_cross(p, hs, radii.x);
        }
        default: {
            return sd_rounded_box(p, hs, radii);
        }
    }
}

// True if `fp` is outside the clip mask at `idx0` or any ancestor (chain intersect).
fn clip_outside(fp: vec2<f32>, idx0: f32) -> bool {
    var idx = i32(idx0);
    var guard = 0;
    loop {
        if (idx < 0 || guard >= 8) {
            break;
        }
        let c = clip_shapes[idx];
        let rc = cos(-c.params.y);
        let rs = sin(-c.params.y);
        let pd = (fp - c.center) - c.pivot;
        let p = vec2<f32>(rc * pd.x - rs * pd.y, rs * pd.x + rc * pd.y) + c.pivot;
        if (eval_sdf(p, c.half_size, c.params.x, c.radii) > 0.5) {
            return true;
        }
        idx = i32(c.params.z);
        guard = guard + 1;
    }
    return false;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clip: cheap bounding-box reject (exact for rectangular masks) then the mask's
    // SDF and parent chain. `clip_rect`/`clip_position.xy` are framebuffer pixels.
    let cr = in.clip_rect;
    if (cr.z > cr.x && cr.w > cr.y) {
        let fp = in.clip_position.xy;
        if (fp.x < cr.x || fp.x > cr.z || fp.y < cr.y || fp.y > cr.w) {
            discard;
        }
    }
    if (in.clip_index >= 0.0 && clip_outside(in.clip_position.xy, in.clip_index)) {
        discard;
    }

    if (in.use_texture > 1.5) {
        // Colour glyph (emoji): draw the atlas RGBA as-is. The run colour only
        // carries opacity through its alpha; the tint RGB is ignored.
        let c = textureSample(glyph_atlas, atlas_sampler, in.uv);
        return vec4<f32>(c.rgb, c.a * in.colour.a);
    } else if (in.use_texture > 0.5) {
        // Coverage glyph: tint the atlas alpha by the run colour.
        let atlas_a = textureSample(glyph_atlas, atlas_sampler, in.uv).a;
        return vec4<f32>(in.colour.rgb, in.colour.a * atlas_a);
    } else {
        return in.colour;
    }
}
