// Screen-space overlay text and solid quad shader.
//
// Renders glyph quads (textured from the glyph atlas's coverage or colour
// page), vector-shape quads
// (textured from the coverage atlas) and solid-colour triangles (background
// boxes, polylines, tessellated fills) in a single batched draw call. Solid
// edges are anti-aliased by a one-device-pixel fringe the vertex stage places.
//
// Vertex attributes carry NDC position, atlas UV, RGBA tint, a flag distinguishing
// textured vs solid fragments, and an optional clip: a framebuffer-pixel bounding
// box plus an index into a shared clip-shape buffer for SDF and nested clipping.
// The SDF functions mirror overlay_shape.wgsl so text clips to the same masks as
// shapes. An item's fill (a gradient or an image) is the shared overlay fill
// module evaluated per fragment over the item's box and multiplied into the
// vertex colour; a solid fill is carried by the vertex colour alone.

// The fill module: group 1 (fill records, fill image, sampler) and
// `overlay_fill`.
// #include "helpers/overlay_fill.wgsl"

// Soft clip coverage over `clip_shapes` and `eval_sdf` below.
// #include "helpers/overlay_clip.wgsl"

// The glyph atlas's coverage page: one channel per texel.
@group(0) @binding(0) var glyph_atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;
// Vector-shape coverage cells, one channel.
@group(0) @binding(4) var coverage_atlas: texture_2d<f32>;
// The glyph atlas's colour page: straight-alpha RGBA.
@group(0) @binding(5) var glyph_colour: texture_2d<f32>;

// One clip-mask shape (framebuffer pixels). `params` = (shape_type, rotation,
// parent_index, invert). Matches ClipShapeGpu on the CPU and overlay_shape.wgsl.
struct ClipShape {
    centre:    vec2<f32>,
    half_size: vec2<f32>,
    radii:     vec4<f32>,
    params:    vec4<f32>,
    pivot:     vec2<f32>,
    pad:       vec2<f32>,
};

// A uniform array of the renderer's `MAX_OVERLAY_CLIP_SHAPES` (256 at 64 bytes
// fills a 16 KiB uniform binding); a frame's later masks are dropped.
@group(0) @binding(2) var<uniform> clip_shapes: array<ClipShape, 256>;

// Viewport size in logical pixels and the frame's pixels per point. The size
// maps local-pixel vertex positions to NDC so overlay geometry is independent
// of the viewport size; `ppp` sizes anti-aliasing fringes in device pixels.
struct Viewport {
    size: vec2<f32>,
    ppp:  f32,
    pad:  f32,
};
@group(0) @binding(3) var<uniform> viewport: Viewport;

// Per-draw instance: a group's per-frame transform, opacity, tint and outer
// clip, from the instance stream; slot 0 is the identity immediate draws use.
struct OverlayInstance {
    translate:  vec2<f32>,
    opacity:    f32,
    clip_index: f32,        // per-frame clip-mask index, or -1 for none
    clip_rect:  vec4<f32>,
    tint:       vec4<f32>,  // per-frame colour multiplier, identity [1,1,1,1]
    linear:     vec4<f32>,  // the linear map about the pivot, row-major a b c d
    pivot:      vec2<f32>,
    aa_scale:   f32,        // mean scale of `linear`, for the shape shader
    pad:        f32,
};

// One level of the composition contract: the linear map about the pivot, then
// the translation. With the identity instance this returns `p`.
fn group_apply(p: vec2<f32>, inst: OverlayInstance) -> vec2<f32> {
    let d = p - inst.pivot;
    let m = inst.linear;
    return inst.pivot + vec2<f32>(m.x * d.x + m.y * d.y, m.z * d.x + m.w * d.y) + inst.translate;
}
// The per-draw instance as a per-instance vertex stream (see `OverlayInstance`).
struct InstanceInput {
    @location(7) translate_opacity_clip: vec4<f32>,
    @location(8) clip_rect: vec4<f32>,
    @location(9) tint: vec4<f32>,
    @location(10) linear: vec4<f32>,
    @location(11) pivot_aa: vec4<f32>,
};

fn read_instance(i: InstanceInput) -> OverlayInstance {
    return OverlayInstance(
        i.translate_opacity_clip.xy,
        i.translate_opacity_clip.z,
        i.translate_opacity_clip.w,
        i.clip_rect,
        i.tint,
        i.linear,
        i.pivot_aa.xy,
        i.pivot_aa.z,
        i.pivot_aa.w,
    );
}

struct VertexInput {
    @location(0) position:    vec2<f32>,  // local logical pixels
    @location(1) uv:          vec2<f32>,  // atlas UV (ignored for solid quads)
    @location(2) colour:      vec4<f32>,  // RGBA tint
    @location(3) use_texture: f32,        // 0 = solid, 1 = glyph coverage, 2 = colour glyph, 3 = coverage cell
    @location(4) clip_index:  f32,        // clip-shape index, or -1 for none
    @location(5) clip_rect:   vec4<f32>,  // framebuffer clip bbox (x0,y0,x1,y1); all zero = none
    @location(6) group_tint:  f32,        // 1 = the group tint applies here, 0 = shadow geometry
    @location(12) fill:       vec4<f32>,  // xy = pixels from the fill box centre, z = fill record or -1
    @location(13) aa:         vec4<f32>,  // fringe: xy = outward direction, z = side (-1 inner, 0 none, 1 outer), w = max inset
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       uv:           vec2<f32>,
    @location(1)       colour:       vec4<f32>,
    @location(2)       use_texture:  f32,
    @location(3) @interpolate(flat) clip_index:  f32,
    @location(4) @interpolate(flat) clip_rect:   vec4<f32>,
    @location(5) @interpolate(flat) opacity:     f32,
    @location(6) @interpolate(flat) outer_clip:  vec4<f32>,
    @location(7) @interpolate(flat) tint:        vec4<f32>,
    @location(8)                    fill_local:  vec2<f32>,
    @location(9) @interpolate(flat) fill_index:  f32,
    // Edge coverage from an anti-aliasing fringe: 1 inside, ramping to 0
    // across the outer device pixel.
    @location(10)                   coverage:    f32,
};

// Map a local logical-pixel position to NDC using the viewport size.
fn px_to_ndc(px: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(px.x / viewport.size.x * 2.0 - 1.0, 1.0 - px.y / viewport.size.y * 2.0);
}

// Place a fringe vertex. One device pixel is `1 / s` local units, `s` being
// the frame's pixels per point times the group's mean scale, so the fringe is
// one device pixel wide whatever the transform. An inner-row vertex moves half
// a pixel in (no further than its max inset, where the opposite edge's fringe
// starts) at full coverage; a hairline narrower than a pixel keeps its
// integrated weight by scaling that coverage down. An outer-row vertex moves
// half a pixel out at zero coverage. Returns the position and coverage.
//
// A negative max inset marks a hairline: the inner row stays on the line at
// full coverage and the outer row moves one device pixel out along the
// screen-space normal, through the group's full linear map rather than its
// mean scale, so the line is one pixel of weight under a stretch too.
fn place_fringe(p: vec2<f32>, aa: vec4<f32>, inst: OverlayInstance) -> vec3<f32> {
    let ppp = select(1.0, viewport.ppp, viewport.ppp > 0.0);
    let s = max(ppp * inst.aa_scale, 1.0e-6);
    if (aa.w < 0.0) {
        if (aa.z > 0.5) {
            return vec3<f32>(p + hairline_offset(aa.xy, inst.linear, ppp), 0.0);
        }
        return vec3<f32>(p, 1.0);
    }
    if (aa.z < -0.5) {
        let m = aa.w * s;
        let inset = min(0.5, m);
        let c = select(1.0, 2.0 * m / (m + 0.5), m < 0.5);
        return vec3<f32>(p - aa.xy * (inset / s), c);
    }
    if (aa.z > 0.5) {
        return vec3<f32>(p + aa.xy * (0.5 / s), 0.0);
    }
    return vec3<f32>(p, 1.0);
}

// The local offset that a hairline's outer row moves by: `dir`'s length in
// device pixels along the screen normal of `dir` under the linear map `m`
// (row-major a b c d), taken back into local units. A map with no inverse
// moves nothing.
fn hairline_offset(dir: vec2<f32>, m: vec4<f32>, ppp: f32) -> vec2<f32> {
    let det = m.x * m.w - m.y * m.z;
    if (abs(det) < 1.0e-12) {
        return vec2<f32>(0.0);
    }
    // Normals map by the inverse transpose.
    let nt = vec2<f32>(dir.x * m.w - dir.y * m.z, dir.y * m.x - dir.x * m.y) / det;
    let len = length(nt);
    if (len < 1.0e-12) {
        return vec2<f32>(0.0);
    }
    let ns = nt * (length(dir) / (len * ppp));
    return vec2<f32>(m.w * ns.x - m.y * ns.y, m.x * ns.y - m.z * ns.x) / det;
}

@vertex
fn vs_main(in: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    let inst = read_instance(instance);
    // Apply the group transform (identity for immediate draws). Text-stream
    // geometry is the vertex position itself, so this moves glyphs, polylines,
    // and vector fills together.
    let placed = place_fringe(in.position, in.aa, inst);
    out.coverage      = placed.z;
    out.clip_position = vec4<f32>(px_to_ndc(group_apply(placed.xy, inst)), 0.0, 1.0);
    out.uv            = in.uv;
    out.colour        = in.colour;
    out.use_texture   = in.use_texture;
    // Prefer the per-frame group clip mask (retained groups) over the baked
    // per-vertex mask (immediate items); the identity instance is -1, so
    // immediate draws keep their baked clip.
    out.clip_index    = select(in.clip_index, inst.clip_index, inst.clip_index >= 0.0);
    out.clip_rect     = in.clip_rect;
    out.opacity       = inst.opacity;
    out.outer_clip    = inst.clip_rect;
    // A group's tint reaches its content and not its shadow layers, which are
    // baked into this same stream, so the vertex says which it is. Opacity is
    // the other way round and applies to both: a fading group takes its shadows
    // with it.
    out.tint          = mix(vec4<f32>(1.0, 1.0, 1.0, 1.0), inst.tint, in.group_tint);
    out.fill_local    = in.fill.xy;
    out.fill_index    = in.fill.z;
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

// The fragment colour in straight alpha.
fn shade(in: VertexOutput) -> vec4<f32> {
    // Clip: the item's box (exact for rectangular masks), the mask's SDF and
    // parent chain, and a retained group's per-frame outer box (the scroll
    // viewport; zero for immediate draws, which is no clip). Each is a
    // coverage, so a clip edge is anti-aliased; a pixel none of it keeps is
    // dropped. `clip_rect`/`clip_position.xy` are framebuffer pixels.
    let fp = in.clip_position.xy;
    var clip = overlay_rect_coverage(fp, in.clip_rect) * overlay_rect_coverage(fp, in.outer_clip);
    if (in.clip_index >= 0.0 && clip > 0.0) {
        clip = clip * overlay_mask_coverage(fp, in.clip_index);
    }
    if (clip <= 0.0) {
        discard;
    }

    // Sampled with an explicit level rather than `textureSample`, which computes
    // its own derivatives and is therefore only legal in uniform control flow.
    // `use_texture` varies per vertex, so the branches below are not uniform,
    // and a strict WGSL front end rejects the whole module: Tint does, naga
    // does not, so the text pipeline failed to build in Chrome while working in
    // Firefox. Each atlas has one mip level, so level zero is what an implicit
    // lookup would have chosen anyway.
    let glyph = textureSampleLevel(glyph_atlas, atlas_sampler, in.uv, 0.0).r;
    let emoji = textureSampleLevel(glyph_colour, atlas_sampler, in.uv, 0.0);
    let cell = textureSampleLevel(coverage_atlas, atlas_sampler, in.uv, 0.0).r;
    // The item's fill multiplies into the vertex colour; with no record it is
    // white and the vertex colour passes through. Evaluated outside any
    // branch, because an image fill samples a texture.
    let raw = in.colour * overlay_fill(in.fill_local, i32(in.fill_index));
    // The item's colour filter, if its record carries one. A colour glyph
    // filters the page's colour under the run's alpha.
    let is_emoji = in.use_texture > 1.5 && in.use_texture <= 2.5;
    let colour = overlay_colour_matrix(
        select(raw, vec4<f32>(emoji.rgb, raw.a), is_emoji),
        i32(in.fill_index),
    );

    var result: vec4<f32>;
    if (in.use_texture > 2.5) {
        // Vector-shape coverage cell: tint its coverage by the fill.
        result = vec4<f32>(colour.rgb, colour.a * cell);
    } else if (in.use_texture > 1.5) {
        // Colour glyph (emoji): draw the colour page's RGBA as-is. The run
        // colour only carries opacity through its alpha; the tint RGB is
        // ignored.
        result = vec4<f32>(colour.rgb, emoji.a * colour.a);
    } else if (in.use_texture > 0.5) {
        // Coverage glyph: tint the coverage page by the run colour.
        result = vec4<f32>(colour.rgb, colour.a * glyph);
    } else {
        result = colour;
    }
    // Per-draw tint and opacity (identity for immediate draws), then the
    // fringe and clip coverage.
    let coverage = clamp(in.coverage, 0.0, 1.0) * clip;
    return vec4<f32>(result.rgb * in.tint.rgb, result.a * in.opacity * in.tint.a * coverage);
}

// Straight alpha, for the source-over blend state.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return shade(in);
}

// Premultiplied alpha, for the blend states whose factors expect a
// premultiplied source.
@fragment
fn fs_premul(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = shade(in);
    return vec4<f32>(c.rgb * c.a, c.a);
}
