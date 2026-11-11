// Screen-space overlay layer composite.
//
// A layer's members are drawn into a scratch target of the viewport's size;
// this draws the result as one quad over the layer's bounds. The content is
// read texel for texel, optionally replaced by a blurred copy, filtered by a
// 4x5 colour matrix, given a drop shadow from a blurred copy of its alpha, and
// faded by the layer's opacity. Every value here is premultiplied linear
// colour.
//
// `fs_composite` writes the result for a blend state to combine with the
// destination. `fs_composite_dst` reads the destination from a copy of the
// target and applies a W3C separable or non-separable blend mode, writing
// `(X, Sa)` for a premultiplied source-over state, which leaves the
// destination untouched where the layer is transparent. `fs_backdrop` draws the
// blurred, filtered destination inside the layer's clip, under its content.

// Soft clip coverage over `clip_shapes` and `eval_sdf` below.
// #include "helpers/overlay_clip.wgsl"

// The layer's members, drawn at their screen positions.
@group(0) @binding(0) var t_content: texture_2d<f32>;
// The content blurred at half resolution (a blank texel when not blurred).
@group(0) @binding(1) var t_blur: texture_2d<f32>;
// The content blurred for the drop shadow, at half resolution.
@group(0) @binding(2) var t_shadow: texture_2d<f32>;
// A copy of the destination as drawn before this layer's rank.
@group(0) @binding(3) var t_dst: texture_2d<f32>;
// The destination blurred, at half resolution, for the backdrop.
@group(0) @binding(4) var t_backdrop: texture_2d<f32>;
@group(0) @binding(5) var s_linear: sampler;

// One clip-mask shape (framebuffer pixels). `params` = (shape_type, rotation,
// parent_index, invert). Matches ClipShapeGpu on the CPU and the other overlay
// shaders.
struct ClipShape {
    centre:    vec2<f32>,
    half_size: vec2<f32>,
    radii:     vec4<f32>,
    params:    vec4<f32>,
    pivot:     vec2<f32>,
    pad:       vec2<f32>,
}

// A uniform array of the renderer's `MAX_OVERLAY_CLIP_SHAPES`, shared with the
// other overlay pipelines.
@group(1) @binding(0) var<uniform> clip_shapes: array<ClipShape, 256>;

// Viewport size in logical pixels and the frame's pixels per point.
struct Viewport {
    size: vec2<f32>,
    ppp:  f32,
    pad:  f32,
};
@group(1) @binding(1) var<uniform> viewport: Viewport;

// Layer flags in `params.z`.
const FLAG_BLUR: u32 = 1u;
const FLAG_SHADOW: u32 = 2u;
const FLAG_MATRIX: u32 = 4u;

// Blend mode codes in `params.y`, as the renderer numbers them.
const MODE_MULTIPLY: i32 = 1;
const MODE_SCREEN: i32 = 2;
const MODE_OVERLAY: i32 = 3;
const MODE_DARKEN: i32 = 4;
const MODE_LIGHTEN: i32 = 5;
const MODE_DODGE: i32 = 6;
const MODE_BURN: i32 = 7;
const MODE_HARD_LIGHT: i32 = 8;
const MODE_SOFT_LIGHT: i32 = 9;
const MODE_DIFFERENCE: i32 = 10;
const MODE_EXCLUSION: i32 = 11;
const MODE_HUE: i32 = 12;
const MODE_SATURATION: i32 = 13;
const MODE_COLOUR: i32 = 14;
const MODE_LUMINOSITY: i32 = 15;

struct InstanceInput {
    @location(0) rect:          vec4<f32>,  // logical pixels x0, y0, x1, y1
    @location(1) params:        vec4<f32>,  // opacity, mode, flags, unused
    @location(2) m0:            vec4<f32>,  // colour matrix columns
    @location(3) m1:            vec4<f32>,
    @location(4) m2:            vec4<f32>,
    @location(5) m3:            vec4<f32>,
    @location(6) m4:            vec4<f32>,
    @location(7) shadow_colour: vec4<f32>,  // straight linear RGBA
    @location(8) shadow:        vec4<f32>,  // offset xy in logical pixels
    @location(9) backdrop:      vec4<f32>,  // saturation, brightness, hue shift
    @location(10) clip_rect:    vec4<f32>,  // framebuffer pixels; all zero = none
    @location(11) clip:         vec4<f32>,  // x = clip-mask index or -1
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // The screen position over the viewport, for every texture read.
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) params: vec4<f32>,
    @location(2) @interpolate(flat) m0: vec4<f32>,
    @location(3) @interpolate(flat) m1: vec4<f32>,
    @location(4) @interpolate(flat) m2: vec4<f32>,
    @location(5) @interpolate(flat) m3: vec4<f32>,
    @location(6) @interpolate(flat) m4: vec4<f32>,
    @location(7) @interpolate(flat) shadow_colour: vec4<f32>,
    // The shadow offset in uv units.
    @location(8) @interpolate(flat) shadow_uv: vec2<f32>,
    @location(9) @interpolate(flat) backdrop: vec4<f32>,
    @location(10) @interpolate(flat) clip_rect: vec4<f32>,
    @location(11) @interpolate(flat) clip: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: InstanceInput) -> VertexOutput {
    // Two triangles over the rect, corners computed from the vertex index.
    let i = vi % 6u;
    let t = vec2<f32>(
        select(0.0, 1.0, i == 1u || i == 3u || i == 4u),
        select(0.0, 1.0, i == 2u || i == 4u || i == 5u),
    );
    let px = mix(inst.rect.xy, inst.rect.zw, t);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(
        px.x / viewport.size.x * 2.0 - 1.0,
        1.0 - px.y / viewport.size.y * 2.0,
        0.0,
        1.0,
    );
    out.uv = px / viewport.size;
    out.params = inst.params;
    out.m0 = inst.m0;
    out.m1 = inst.m1;
    out.m2 = inst.m2;
    out.m3 = inst.m3;
    out.m4 = inst.m4;
    out.shadow_colour = inst.shadow_colour;
    out.shadow_uv = inst.shadow.xy / viewport.size;
    out.backdrop = inst.backdrop;
    out.clip_rect = inst.clip_rect;
    out.clip = inst.clip;
    return out;
}

// The clip-mask distance fields, as the other overlay shaders define them.
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

fn sd_line(p: vec2<f32>, hs: vec2<f32>, r: f32, cap: f32) -> f32 {
    if (cap > 0.5) {
        // Rotate into segment frame, then use a box SDF.
        let seg_len = length(hs);
        if (seg_len < 0.0001) {
            return length(p) - r;
        }
        let d = hs / seg_len;
        let along = dot(p, d);
        let perp  = dot(p, vec2<f32>(-d.y, d.x));
        let reach = seg_len + select(0.0, r, cap > 1.5);
        let q = abs(vec2<f32>(along, perp)) - vec2<f32>(reach, r);
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
            return sd_line(p, hs, radii.x, radii.y);
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

// How much of the pixel the layer's clip keeps.
// `clip_rect` and `fp` are framebuffer pixels from the viewport's top-left
// corner, found from `uv` rather than `@builtin(position)`, which counts from
// the framebuffer's.
fn layer_clip(in: VertexOutput) -> f32 {
    let fp = overlay_pixel_centre(in.uv * viewport.size * viewport.ppp);
    var c = overlay_rect_coverage(fp, in.clip_rect);
    if (in.clip.x >= 0.0 && c > 0.0) {
        c = c * overlay_mask_coverage(fp, in.clip.x);
    }
    return c;
}

// The texel of `t` under the fragment, for a target of any size.
fn load_at(t: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(t));
    let p = clamp(vec2<i32>(floor(uv * vec2<f32>(dims))), vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(t, p, 0);
}

fn unpremultiply(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(c.rgb / c.a, c.a);
}

// The layer's content, premultiplied: blurred, filtered, shadowed, faded and
// clipped.
fn layer_content(in: VertexOutput) -> vec4<f32> {
    let flags = u32(in.params.z + 0.5);
    let sharp = load_at(t_content, in.uv);
    // Explicit-level samples, legal in any control flow.
    let blurred = textureSampleLevel(t_blur, s_linear, in.uv, 0.0);
    let shadow_src = textureSampleLevel(t_shadow, s_linear, in.uv - in.shadow_uv, 0.0);
    var p = select(sharp, blurred, (flags & FLAG_BLUR) != 0u);
    if ((flags & FLAG_MATRIX) != 0u) {
        let s = unpremultiply(p);
        let f = clamp(
            in.m0 * s.r + in.m1 * s.g + in.m2 * s.b + in.m3 * s.a + in.m4,
            vec4<f32>(0.0),
            vec4<f32>(1.0),
        );
        p = vec4<f32>(f.rgb * f.a, f.a);
    }
    if ((flags & FLAG_SHADOW) != 0u) {
        let sa = shadow_src.a * in.shadow_colour.a;
        p = p + vec4<f32>(in.shadow_colour.rgb * sa, sa) * (1.0 - p.a);
    }
    return p * in.params.x * layer_clip(in);
}

// Premultiplied result for a blend state: source-over, or one of the
// fixed-function modes.
@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    return layer_content(in);
}

fn lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.3, 0.59, 0.11));
}

fn clip_colour(c: vec3<f32>) -> vec3<f32> {
    let l = lum(c);
    let n = min(min(c.r, c.g), c.b);
    let x = max(max(c.r, c.g), c.b);
    var o = c;
    if (n < 0.0) {
        o = l + (o - l) * l / max(l - n, 1.0e-6);
    }
    if (x > 1.0) {
        o = l + (o - l) * (1.0 - l) / max(x - l, 1.0e-6);
    }
    return o;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return clip_colour(c + (l - lum(c)));
}

fn sat(c: vec3<f32>) -> f32 {
    return max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
}

// Rescale `c` so its saturation is `s`, keeping which channel is the largest
// and which the smallest.
fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    let range = mx - mn;
    if (range <= 0.0) {
        return vec3<f32>(0.0);
    }
    return (c - mn) * s / range;
}

fn soft_light_d(b: f32) -> f32 {
    if (b <= 0.25) {
        return ((16.0 * b - 12.0) * b + 4.0) * b;
    }
    return sqrt(b);
}

fn blend_channel(mode: i32, s: f32, b: f32) -> f32 {
    switch (mode) {
        case 1: { return s * b; }
        case 2: { return s + b - s * b; }
        case 3: {
            // Hard light with the operands swapped.
            if (b <= 0.5) {
                return s * 2.0 * b;
            }
            let t = 2.0 * b - 1.0;
            return s + t - s * t;
        }
        case 4: { return min(s, b); }
        case 5: { return max(s, b); }
        case 6: {
            if (b <= 0.0) {
                return 0.0;
            }
            if (s >= 1.0) {
                return 1.0;
            }
            return min(1.0, b / (1.0 - s));
        }
        case 7: {
            if (b >= 1.0) {
                return 1.0;
            }
            if (s <= 0.0) {
                return 0.0;
            }
            return 1.0 - min(1.0, (1.0 - b) / s);
        }
        case 8: {
            if (s <= 0.5) {
                return b * 2.0 * s;
            }
            let t = 2.0 * s - 1.0;
            return b + t - b * t;
        }
        case 9: {
            if (s <= 0.5) {
                return b - (1.0 - 2.0 * s) * b * (1.0 - b);
            }
            return b + (2.0 * s - 1.0) * (soft_light_d(b) - b);
        }
        case 10: { return abs(s - b); }
        case 11: { return s + b - 2.0 * s * b; }
        default: { return s; }
    }
}

// The blended colour of straight source `s` over straight backdrop `b`.
fn blend_colour(mode: i32, s: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    if (mode == MODE_HUE) {
        return set_lum(set_sat(s, sat(b)), lum(b));
    }
    if (mode == MODE_SATURATION) {
        return set_lum(set_sat(b, sat(s)), lum(b));
    }
    if (mode == MODE_COLOUR) {
        return set_lum(s, lum(b));
    }
    if (mode == MODE_LUMINOSITY) {
        return set_lum(b, lum(s));
    }
    return vec3<f32>(
        blend_channel(mode, s.r, b.r),
        blend_channel(mode, s.g, b.g),
        blend_channel(mode, s.b, b.b),
    );
}

// The backdrop's colour filters, as the backdrop-blur shapes apply them.
fn filter_backdrop(rgb: vec3<f32>, f: vec4<f32>) -> vec3<f32> {
    var c = rgb;
    if (abs(f.z) > 0.0001) {
        let k = vec3<f32>(0.57735026919);
        let ca = cos(f.z);
        let sa = sin(f.z);
        c = c * ca + cross(k, c) * sa + k * dot(k, c) * (1.0 - ca);
    }
    let luma = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    c = mix(vec3<f32>(luma), c, f.x);
    return max(c * f.y, vec3<f32>(0.0));
}

// The layer blended with the destination it was drawn over. Writes
// `Cs (1 - Ab) + As Ab B(Cs, Cb)` with the source alpha, which a
// premultiplied source-over state completes with `(1 - As)` of the
// destination.
@fragment
fn fs_composite_dst(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = layer_content(in);
    var d = load_at(t_dst, in.uv);
    // With a backdrop the layer was drawn over it, not over the copy.
    let bd = textureSampleLevel(t_backdrop, s_linear, in.uv, 0.0);
    if (in.backdrop.w > 0.5) {
        // Premultiplied: a backdrop inside a layer keeps the layer's alpha,
        // one over the frame is opaque.
        let m = layer_clip(in) * bd.a;
        let b = filter_backdrop(bd.rgb, in.backdrop);
        d = vec4<f32>(b * layer_clip(in), m) + d * (1.0 - m);
    }
    let mode = i32(in.params.y + 0.5);
    let s = unpremultiply(p);
    let b = unpremultiply(d);
    let mixed = blend_colour(mode, s.rgb, b.rgb);
    let x = p.rgb * (1.0 - d.a) + p.a * d.a * mixed;
    return vec4<f32>(x, p.a);
}

// The blurred, filtered destination inside the layer's clip.
@fragment
fn fs_backdrop(in: VertexOutput) -> @location(0) vec4<f32> {
    let bd = textureSampleLevel(t_backdrop, s_linear, in.uv, 0.0);
    let m = layer_clip(in);
    return vec4<f32>(filter_backdrop(bd.rgb, in.backdrop) * m, bd.a * m);
}
