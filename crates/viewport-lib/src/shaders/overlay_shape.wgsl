// Screen-space SDF overlay shape shader.
//
// Each shape is one instance: the vertex stage expands its bounding quad from
// the vertex index, and the fragment stage maps each pixel back into the
// shape's own frame through the inverse of the item's linear map and evaluates
// a signed-distance function there, producing an anti-aliased fill with
// stacked outer and inner shadow layers around it. The fill is the shared
// overlay fill module evaluated in the shape's own frame (a gradient or an
// image), or a flat colour.

// The fill module: group 1 (fill records, fill image, sampler) and
// `overlay_fill`.
// #include "helpers/overlay_fill.wgsl"

// Soft clip coverage over `clip_shapes` and `eval_sdf` below.
// #include "helpers/overlay_clip.wgsl"

// Fill coverage at the outline, anti-aliased or crisp.
// #include "helpers/overlay_edge.wgsl"

// Nine instance attributes and no vertex stream, inside the 16 WebGL2 allows.
struct ShapeInstance {
    @location(0) quad:        vec4<f32>,  // xy = quad centre (logical px), zw = quad half-extents
    @location(1) pivot_half:  vec4<f32>,  // xy = pivot (logical px), zw = shape half-extents
    @location(2) linv:        vec4<f32>,  // inverse item linear map, row-major a b c d
    @location(3) fill_colour: vec4<f32>,  // flat colour, drawn when there is no fill record
    @location(4) radii:       vec4<f32>,  // shape-specific params
    @location(5) shadow_meta: vec4<f32>,  // base_index, outer_ct, inner_ct, shape_type
    @location(6) clip_rect:   vec4<f32>,  // framebuffer-pixel clip bbox (x0,y0,x1,y1); all zero = no box clip
    @location(7) misc:        vec4<f32>,  // clip-shape index (or -1), pivot in the shape frame xy, item scale
    @location(8) fill:        vec4<f32>,  // x = fill record index, or -1 for the flat colour; y = 1 for a crisp edge
};

// One stacked shadow layer. `params` = (blur, offset_x, offset_y, is_inner),
// `params2` = (spread, falloff, unused, unused).
struct ShadowLayer {
    colour: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
};

// Shadow coverage for a point whose signed distance to the (already offset)
// silhouette is `sd`. `spread` grows the silhouette, `blur` is the fade
// distance outward from that grown edge, and `falloff` shapes the fade curve:
// 1.0 is the plain fade, above concentrates it against the item with a longer
// light tail, below broadens it towards a glow. `aa` keeps a zero-blur edge
// (a contour) antialiased instead of degenerate.
fn shadow_coverage(sd: f32, spread: f32, blur: f32, falloff: f32, aa: f32) -> f32 {
    let d = sd - spread;
    let w = max(blur, aa);
    let base = 1.0 - smoothstep(0.0, w, d);
    if (falloff == 1.0) {
        return base;
    }
    return pow(base, falloff);
}

// The inner (inset) counterpart: coverage rises from 0 at the shape edge to 1
// across the blur band, with `spread` starting the band further inside.
fn inner_shadow_coverage(sd: f32, spread: f32, blur: f32, falloff: f32, aa: f32) -> f32 {
    let w = max(blur, aa);
    let base = smoothstep(0.0, w, sd + spread);
    if (falloff == 1.0) {
        return base;
    }
    return pow(base, falloff);
}

@group(0) @binding(0) var shadow_layers: texture_2d<f32>;

// The stacked shadow layers, as a data texture rather than a storage buffer so
// the overlay family runs on devices without storage buffers. Layer `i` is
// three `Rgba32Float` texels side by side, `SHADOW_LAYERS_PER_ROW` layers to a
// row (the renderer's constant of the same name).
const SHADOW_LAYERS_PER_ROW: i32 = 256;
fn shadow_layer(i: i32) -> ShadowLayer {
    let at = vec2<i32>((i % SHADOW_LAYERS_PER_ROW) * 3, i / SHADOW_LAYERS_PER_ROW);
    return ShadowLayer(
        textureLoad(shadow_layers, at, 0),
        textureLoad(shadow_layers, at + vec2<i32>(1, 0), 0),
        textureLoad(shadow_layers, at + vec2<i32>(2, 0), 0),
    );
}

// One clip-mask shape (framebuffer pixels). `params` = (shape_type, rotation,
// parent_index, invert). An instance's `clip_index` selects an entry;
// `parent_index` (or -1) chains to an enclosing mask so nested clips intersect.
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
@group(0) @binding(1) var<uniform> clip_shapes: array<ClipShape, 256>;

// Viewport size in logical pixels (xy); zw padding. Maps local-pixel vertex
// positions to NDC so overlay geometry is independent of the viewport size.
struct Viewport {
    size: vec2<f32>,
    pad:  vec2<f32>,
};
@group(0) @binding(2) var<uniform> viewport: Viewport;

// Per-draw instance: a retained group's per-frame transform, opacity, tint and
// outer clip, bound as a uniform at a dynamic offset; offset 0 is the identity
// immediate draws use. The instance is consumed in the vertex stage: the
// transform on the quad, opacity and tint folded into the flat colour, and the
// outer clip intersected with the baked clip into the single `clip_rect`
// varying (two bbox rejects are equivalent to one reject against their
// intersection). Opacity and tint ride flat varyings for the fill record and
// the shadow layers, which are read in the fragment stage.
struct OverlayInstance {
    translate:  vec2<f32>,
    opacity:    f32,
    clip_index: f32,        // per-frame clip-mask index, or -1 for none
    clip_rect:  vec4<f32>,
    tint:       vec4<f32>,  // per-frame colour multiplier, identity [1,1,1,1]
    linear:     vec4<f32>,  // the group's linear map about the pivot, row-major a b c d
    pivot:      vec2<f32>,
    aa_scale:   f32,        // mean scale of `linear`: how many pixels one shape unit covers
    pad:        f32,
};
@group(0) @binding(3) var<uniform> group: OverlayInstance;

// One level of the composition contract: the linear map about the pivot, then
// the translation. With the identity instance this returns `p`.
fn group_apply(p: vec2<f32>, inst: OverlayInstance) -> vec2<f32> {
    let d = p - inst.pivot;
    let m = inst.linear;
    return inst.pivot + vec2<f32>(m.x * d.x + m.y * d.y, m.z * d.x + m.w * d.y) + inst.translate;
}

fn px_to_ndc(px: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(px.x / viewport.size.x * 2.0 - 1.0, 1.0 - px.y / viewport.size.y * 2.0);
}

// Combine two bounding-box clips in framebuffer pixels. An all-zero rect means
// "no clip", so the other is returned; two real boxes intersect (a fragment must
// lie inside both). A non-overlapping intersection returns an off-screen box so
// nothing survives. Immediate shapes read the identity instance (zero clip) and
// so keep exactly their baked clip.
fn combine_clip(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let a_valid = a.z > a.x && a.w > a.y;
    let b_valid = b.z > b.x && b.w > b.y;
    if (!a_valid) { return b; }
    if (!b_valid) { return a; }
    let r = vec4<f32>(max(a.x, b.x), max(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
    if (r.z <= r.x || r.w <= r.y) {
        // Off the top-left, where no fragment falls. Not far off to the
        // right: 1e9 + 1 rounds to 1e9 in f32 and the box would lose its area.
        return vec4<f32>(-2.0, -2.0, -1.0, -1.0);
    }
    return r;
}

// The quad's six corners (two triangles), in units of the half-extents.
fn quad_corner(vi: u32) -> vec2<f32> {
    switch (vi) {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>( 1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0, -1.0); }
        case 4u: { return vec2<f32>( 1.0,  1.0); }
        default: { return vec2<f32>(-1.0,  1.0); }
    }
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Pixel offset from the pivot in the item's own pixel space, before the
    // group transform: the one interpolated varying.
    @location(0) local_pos: vec2<f32>,
    @location(1) @interpolate(flat) fill_colour: vec4<f32>,
    @location(2) @interpolate(flat) radii:       vec4<f32>,
    @location(3) @interpolate(flat) shadow_meta: vec4<f32>,
    @location(4) @interpolate(flat) linv:        vec4<f32>,
    // clip-shape index, pivot in the shape frame xy, the group's opacity (which
    // reaches the shadow layers, read from a texture in the fragment stage).
    @location(5) @interpolate(flat) misc:        vec4<f32>,
    // shape half-extents xy, the anti-aliasing band in shape units, item scale.
    @location(6) @interpolate(flat) half_aa:     vec4<f32>,
    @location(7) @interpolate(flat) clip_rect:   vec4<f32>,
    // x = fill record index, or -1 for `fill_colour`; y = 1 for a crisp edge.
    @location(8) @interpolate(flat) fill:        vec4<f32>,
    // The group's tint, applied to a fill record's colour.
    @location(9) @interpolate(flat) group_tint:  vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, in: ShapeInstance) -> VertexOutput {
    var out: VertexOutput;
    let inst = group;
    let a = inst.opacity;
    let t = inst.tint;
    // The quad in the item's pixel space, then the group transform on it.
    // `local_pos` stays in the item's space, so the fragment maps it into the
    // shape's own frame through `linv` and evaluates the SDF there, and the
    // group transform carries the result. Edge anti-aliasing follows: the
    // band is `1 / aa_scale` shape units, one framebuffer pixel after the
    // group's mean scale.
    let q = in.quad.xy + quad_corner(vi) * in.quad.zw;
    out.clip_position = vec4<f32>(px_to_ndc(group_apply(q, inst)), 0.0, 1.0);
    out.local_pos     = q - in.pivot_half.xy;
    // Fold the per-draw opacity and colour tint into the flat colour (identity
    // for immediate draws). The tint does not reach the SDF shadow layers (read
    // in the fragment stage), so a tinted shape keeps its authored shadow
    // colour.
    out.fill_colour   = vec4<f32>(in.fill_colour.rgb * t.rgb, in.fill_colour.a * a * t.a);
    out.radii         = in.radii;
    out.shadow_meta   = in.shadow_meta;
    out.linv          = in.linv;
    // Prefer the per-frame group clip mask (retained groups) over the baked mask;
    // the identity instance is -1, so immediate draws keep their baked clip.
    // Opacity reaches the shadow layers; the tint does not. Fading a group out
    // has to take its shadows and its border band with it, and a border band is
    // a shadow layer.
    out.misc          = vec4<f32>(
        select(in.misc.x, inst.clip_index, inst.clip_index >= 0.0),
        in.misc.y,
        in.misc.z,
        a,
    );
    out.half_aa       = vec4<f32>(in.pivot_half.zw, 1.0 / max(inst.aa_scale, 1.0e-6), in.misc.w);
    // Intersect the baked clip with the per-frame group clip (identity for
    // immediate draws), so a retained shape group clips to its outer rect.
    out.clip_rect     = combine_clip(in.clip_rect, inst.clip_rect);
    out.fill          = in.fill;
    out.group_tint    = t;
    return out;
}

// Signed distance to a rounded box with per-corner radii.
// radii: x = top-right, y = bottom-right, z = bottom-left, w = top-left
// (iq convention, rearranged from the item's [tl, tr, br, bl] on the CPU).
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    var rs = r;
    if (p.x > 0.0) {
        rs = vec4<f32>(rs.x, rs.y, rs.z, rs.w);  // right side: top-right, bottom-right
    } else {
        rs = vec4<f32>(rs.w, rs.z, rs.y, rs.x);  // left side: top-left, bottom-left
    }
    if (p.y > 0.0) {
        // bottom half
        rs.x = rs.y;
    }
    let q = abs(p) - b + rs.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - rs.x;
}

// Signed distance to a circle centred at the origin.
fn sd_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// Approximate signed distance to an axis-aligned ellipse.
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
    // sign: inside if normalised ellipse equation < 1
    let np = pa / ab;
    let inside = dot(np, np);
    return select(d, -d, inside < 1.0);
}

// Signed distance to a capsule (pill) along the longer axis.
fn sd_capsule(p: vec2<f32>, half_size: vec2<f32>) -> f32 {
    let r = min(half_size.x, half_size.y);
    var q = abs(p);
    if (half_size.x > half_size.y) {
        // horizontal capsule: clamp x to the straight segment
        q.x = q.x - (half_size.x - r);
    } else {
        // vertical capsule: clamp y to the straight segment
        q.y = q.y - (half_size.y - r);
    }
    return length(max(q, vec2<f32>(0.0))) - r;
}

// Signed distance to a ring (annulus).
// outer_r: outer radius, inner_frac: inner radius as fraction of outer.
fn sd_ring(p: vec2<f32>, outer_r: f32, inner_frac: f32) -> f32 {
    let wall = outer_r * (1.0 - inner_frac) * 0.5;
    let mid_r = outer_r - wall;
    return abs(length(p) - mid_r) - wall;
}

// Signed distance to an arc (annular sector).
// outer_r: outer radius, inner_frac: inner radius fraction,
// sa/ea: start/end angles in radians (CCW from +x).
fn sd_arc(p: vec2<f32>, outer_r: f32, inner_frac: f32, sa: f32, ea: f32) -> f32 {
    // Ring distance.
    let d_ring = sd_ring(p, outer_r, inner_frac);

    // Angular mask: compute the angle of the fragment and check if it
    // falls inside the swept range [sa, ea] (CCW).
    let angle = atan2(p.y, p.x); // -pi..pi

    // Normalise sweep so we can do a single range check.
    // Map angle into [0, 2*pi) relative to start_angle.
    let two_pi = 6.28318530718;
    let sweep = ((ea - sa) % two_pi + two_pi) % two_pi; // positive sweep length
    let a = ((angle - sa) % two_pi + two_pi) % two_pi;  // angle relative to start

    if (a <= sweep) {
        return d_ring;
    }

    // Outside the angular range: distance to the two end-cap edges.
    let wall = outer_r * (1.0 - inner_frac) * 0.5;
    let mid_r = outer_r - wall;
    let inner_r = mid_r - wall;
    let outer_edge = mid_r + wall;

    // End-cap line segments at start_angle and end_angle.
    let cs = vec2<f32>(cos(sa), sin(sa));
    let ce = vec2<f32>(cos(ea), sin(ea));

    // Closest point on each cap segment (between inner_r and outer_edge).
    let proj_s = clamp(dot(p, cs), inner_r, outer_edge);
    let proj_e = clamp(dot(p, ce), inner_r, outer_edge);

    let ds = length(p - cs * proj_s);
    let de = length(p - ce * proj_e);

    return min(ds, de);
}

// Signed distance to an isoceles triangle pointing up, fitted to half_size.
// The apex is at (0, -hs.y) and the base spans (-hs.x, hs.y) to (hs.x, hs.y).
fn sd_triangle(p: vec2<f32>, hs: vec2<f32>) -> f32 {
    // Mirror to the right half.
    let q = vec2<f32>(abs(p.x), p.y);
    // Edge from apex (0, -hs.y) to base corner (hs.x, hs.y).
    let e = vec2<f32>(hs.x, 2.0 * hs.y);
    let en = normalize(e);
    // Signed distance to the slanted edge (normal points outward to the right).
    let n = vec2<f32>(en.y, -en.x);
    let d_edge = dot(q - vec2<f32>(0.0, -hs.y), n);
    // Signed distance to the base (bottom edge).
    let d_base = q.y - hs.y;
    return max(d_edge, d_base);
}

// Signed distance to a line segment from -hs to +hs with stroke radius r.
// `cap`: 0 round, 1 butt (flush with the end points), 2 square (half the
// width past them).
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
        // Capsule: segment from (-hs.x, -hs.y) to (hs.x, hs.y).
        let ba = 2.0 * hs;
        let pa = p + hs;
        let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h) - r;
    }
}

// Signed distance to an N-pointed star with outer radius r and inner/outer ratio rf.
// radii.x = n (number of points), radii.y = rf.
fn sd_star_n(p: vec2<f32>, r: f32, n: f32, rf: f32) -> f32 {
    let ri = r * rf;
    let an = 3.14159265 / n;
    let two_an = 2.0 * an;

    let a = atan2(p.y, p.x);
    let a_mod = ((a % two_an) + two_an) % two_an;
    let a_abs = select(a_mod, two_an - a_mod, a_mod > an);

    let rp = length(p);
    let q = rp * vec2<f32>(cos(a_abs), sin(a_abs));

    // Edge from outer tip (r, 0) to inner valley.
    let outer = vec2<f32>(r, 0.0);
    let inner = vec2<f32>(ri * cos(an), ri * sin(an));
    let ba = inner - outer;
    let qa = q - outer;
    let t = clamp(dot(qa, ba) / dot(ba, ba), 0.0, 1.0);
    let d = length(qa - ba * t);
    let cross_val = qa.x * ba.y - qa.y * ba.x;
    return d * select(1.0, -1.0, cross_val < 0.0);
}

// Signed distance to a regular N-gon with circumradius r.
// radii.x = n (number of sides).
fn sd_ngon(p: vec2<f32>, r: f32, n: f32) -> f32 {
    let an = 3.14159265 / n;
    let two_an = 2.0 * an;
    let a = atan2(p.y, p.x) + an;
    let a_mod = ((a % two_an) + two_an) % two_an;
    let a_abs = select(a_mod, two_an - a_mod, a_mod > an);

    let rp = length(p);
    let q = rp * vec2<f32>(cos(a_abs), sin(a_abs));

    let he = r * cos(an); // apothem
    let hv = r * sin(an); // half vertex extent

    let dx = q.x - he;
    let dy = max(q.y - hv, 0.0);
    return select(dx, sqrt(dx * dx + dy * dy), dy > 0.0);
}

// Signed distance to a plus/cross shape.
// radii.x = arm_width_frac (fraction of min(hs.x, hs.y)).
fn sd_cross(p: vec2<f32>, hs: vec2<f32>, arm_frac: f32) -> f32 {
    let arm_w = arm_frac * min(hs.x, hs.y);
    let q_h = abs(p) - vec2<f32>(hs.x, arm_w);
    let q_v = abs(p) - vec2<f32>(arm_w, hs.y);
    let d_h = length(max(q_h, vec2<f32>(0.0))) + min(max(q_h.x, q_h.y), 0.0);
    let d_v = length(max(q_v, vec2<f32>(0.0))) + min(max(q_v.x, q_v.y), 0.0);
    return min(d_h, d_v);
}

// Evaluate the SDF for the current shape type at position p.
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
            // Line: radii.x = stroke radius, radii.y = cap (0 round, 1 butt,
            // 2 square).
            let r = radii.x;
            return sd_line(p, hs, r, radii.y);
        }
        case 8: {
            // Star: radii.x = n (points), radii.y = inner_radius_frac.
            let r = min(hs.x, hs.y);
            return sd_star_n(p, r, radii.x, radii.y);
        }
        case 9: {
            // RegularPolygon: radii.x = n (sides). The top-level
            // `rotation` on the item rotates p before this case is reached.
            let r = min(hs.x, hs.y);
            return sd_ngon(p, r, radii.x);
        }
        case 10: {
            // Cross: radii.x = arm_width_frac.
            return sd_cross(p, hs, radii.x);
        }
        default: {
            return sd_rounded_box(p, hs, radii);
        }
    }
}

// Map a pixel offset from the pivot (item pixel space) into the shape's own
// frame: the inverse linear map, then the pivot's place in that frame.
fn to_shape_frame(local: vec2<f32>, linv: vec4<f32>, pivot: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(linv.x * local.x + linv.y * local.y, linv.z * local.x + linv.w * local.y) + pivot;
}

// The fragment colour in straight alpha.
fn shade(in: VertexOutput) -> vec4<f32> {
    // Clip: the bounding box (exact for rectangular masks), then the mask's SDF
    // and its parent chain for shape-accurate and nested clipping, each as a
    // coverage so the clip edge is anti-aliased. `clip_rect`/`clip_position.xy`
    // are in framebuffer pixels (top-left origin).
    let fp = in.clip_position.xy;
    var clip = overlay_rect_coverage(fp, in.clip_rect);
    let clip_index = in.misc.x;
    if (clip_index >= 0.0 && clip > 0.0) {
        clip = clip * overlay_mask_coverage(fp, clip_index);
    }
    if (clip <= 0.0) {
        discard;
    }
    // Into the shape's own frame: the SDF evaluates unrotated, unscaled and
    // unsheared, and the item's linear map (folded into `linv`) carries the
    // result. With no transform this is the identity about the pivot.
    let pivot = in.misc.yz;
    let p = to_shape_frame(in.local_pos, in.linv, pivot);
    let hs = in.half_aa.xy;
    let shape_type = in.shadow_meta.w;

    let d = eval_sdf(p, hs, shape_type, in.radii);

    // Anti-aliasing: a one framebuffer pixel band at the boundary, in shape
    // units: one unit covers `aa_scale` pixels after the group transform. An
    // immediate shape and an unscaled group both land on exactly 1.0. A crisp
    // edge steps at pixel centres instead, and so do its zero-blur shadow
    // layers (`band`).
    let aa = in.half_aa.z;
    let crisp = in.fill.y > 0.5;
    let band = overlay_edge_band(aa, crisp);
    // Shadow offsets are authored in screen-aligned shape units scaled by the
    // item's uniform scale, so they are taken off in the item's pixel space
    // before the map into the shape frame.
    let item_scale = in.half_aa.w;
    let group_opacity = in.misc.w;

    // shadow_meta: (base_index, outer_count, inner_count, shape_type).
    let base_index = i32(in.shadow_meta.x + 0.5);
    let outer_count = i32(in.shadow_meta.y + 0.5);
    let inner_count = i32(in.shadow_meta.z + 0.5);

    // Accumulate the stacked outer shadow layers (drawn behind the fill).
    // Each layer is composited src-over the previous, first layer furthest
    // back. Sample the SDF at the layer's offset in the shape frame.
    var shadow_col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < outer_count; i = i + 1) {
        let layer = shadow_layer(base_index + i);
        let sr = layer.params.x;
        let soff = layer.params.yz;
        let sspread = layer.params2.x;
        let sfall = layer.params2.y;
        if ((sr > 0.0 || sspread > 0.0) && layer.colour.a > 0.0) {
            let sp = to_shape_frame(in.local_pos - soff * item_scale, in.linv, pivot);
            let sd = eval_sdf(sp, hs, shape_type, in.radii);
            let a = layer.colour.a
                * group_opacity
                * shadow_coverage(sd, sspread, sr, sfall, band);
            let src = vec4<f32>(layer.colour.rgb, a);
            shadow_col = vec4<f32>(
                mix(shadow_col.rgb, src.rgb, src.a),
                src.a + shadow_col.a * (1.0 - src.a),
            );
        }
    }

    let fill_alpha = overlay_fill_coverage(d, p, hs, shape_type, aa, crisp);

    // Clip the outer shadow to outside the silhouette, the way CSS clips an
    // outer box-shadow to outside the border box. Without this a translucent
    // shape shows its own shadow through itself: the layers are drawn behind
    // the fill, and a dilated silhouette covers the whole interior, not just
    // the ring outside it. Inner shadows are the inside counterpart and are
    // composited separately below.
    shadow_col = vec4<f32>(shadow_col.rgb, shadow_col.a * (1.0 - fill_alpha));

    // Nothing to draw where neither the fill nor an outer layer reaches.
    if (fill_alpha <= 0.0 && shadow_col.a <= 0.0) {
        discard;
    }

    // The fill: a record from the shared fill module (a gradient or an image)
    // evaluated in the shape's own frame, with the group's opacity and tint on
    // it, or the flat colour. Evaluated here, outside any branch, because an
    // image fill samples a texture.
    let record = i32(in.fill.x);
    let fill_sample = overlay_fill(p, record);
    var fill_col = in.fill_colour;
    if (record >= 0) {
        let gt = in.group_tint;
        fill_col = vec4<f32>(fill_sample.rgb * gt.rgb, fill_sample.a * group_opacity * gt.a);
        fill_col = overlay_colour_matrix(fill_col, record);
    }

    // Start with the accumulated outer shadow layers.
    var colour = shadow_col;

    // Composite fill on top of shadow, as a proper source-over on
    // un-premultiplied colours. `mix(dst.rgb, src.rgb, src.a)` is only correct
    // when the destination is opaque: with a cleared or clipped shadow behind
    // it, the destination's colour still leaks through at full strength even
    // though its alpha is zero. That is what made a translucent shape look
    // tinted by its own clipped-away shadow.
    if (fill_alpha > 0.0) {
        let fc = vec4<f32>(fill_col.rgb, fill_col.a * fill_alpha);
        let out_a = fc.a + colour.a * (1.0 - fc.a);
        let rgb = (fc.rgb * fc.a + colour.rgb * colour.a * (1.0 - fc.a)) / max(out_a, 1.0e-5);
        colour = vec4<f32>(select(fc.rgb, rgb, out_a > 0.0), out_a);
    }

    // Inner shadow layers: composite on top of the fill, so they tint the
    // body. Sample the SDF at the offset point in the shape frame; positive sd
    // means the offset point lies outside the shape, so the current fragment is
    // in the shadow band, fading over the layer radius. Only contributes where
    // the fragment is inside the shape.
    if (d < 0.0) {
        for (var j = 0; j < inner_count; j = j + 1) {
            let layer = shadow_layer(base_index + outer_count + j);
            let sr = layer.params.x;
            let soff = layer.params.yz;
            let sspread = layer.params2.x;
            let sfall = layer.params2.y;
            if ((sr > 0.0 || sspread > 0.0) && layer.colour.a > 0.0) {
                let sp = to_shape_frame(in.local_pos - soff * item_scale, in.linv, pivot);
                let inner_sd = eval_sdf(sp, hs, shape_type, in.radii);
                // Inner shadows ramp the other way: alpha rises as the offset
                // point falls further outside the shape. Spread starts the band
                // further in, blur is its fade width, falloff shapes the ramp
                // the same way it does an outer layer.
                let inner_alpha = layer.colour.a
                    * group_opacity
                    * inner_shadow_coverage(inner_sd, sspread, sr, sfall, band);
                if (inner_alpha > 0.0) {
                    let ic = vec4<f32>(layer.colour.rgb, inner_alpha);
                    colour = vec4<f32>(
                        mix(colour.rgb, ic.rgb, ic.a),
                        ic.a + colour.a * (1.0 - ic.a),
                    );
                }
            }
        }
    }

    return vec4<f32>(colour.rgb, colour.a * clip);
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
