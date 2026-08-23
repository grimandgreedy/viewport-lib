// Screen-space SDF overlay shape shader.
//
// Each shape is a bounding quad whose vertices carry the shape parameters.
// The fragment shader evaluates a signed-distance function to produce
// anti-aliased fill and border regions, with an optional outer shadow/glow.

struct VertexInput {
    @location(0) position:        vec2<f32>,  // NDC xy
    @location(1) local_pos:       vec2<f32>,  // pixels from shape centre
    @location(2) fill_colour:     vec4<f32>,  // start colour (or solid colour)
    @location(3) border_colour:   vec4<f32>,
    @location(4) half_size:       vec2<f32>,  // shape half-extents in pixels
    @location(5) radii:           vec4<f32>,  // shape-specific params
    @location(6) shape_meta:      vec2<f32>, // x=border_width, y=shape_type
    @location(7) stop_positions:  vec4<f32>, // positions in [0,1] for stops a..d
    @location(8) fill_colour2:    vec4<f32>,  // 2nd colour stop (equals fill_colour for solid)
    @location(9) gradient_params: vec4<f32>,  // x=type, y=angle/offset, z=stop_count, w=pad
    @location(10) shadow_index:   vec4<f32>,  // base_index, outer_ct, inner_ct, border_mode
    @location(11) rotation_pivot: vec4<f32>,  // rotation, pivot_x, pivot_y, pad
    @location(12) clip_rect:      vec4<f32>,  // framebuffer-pixel clip bbox (x0,y0,x1,y1); all zero = no box clip
    @location(13) clip_index:     f32,        // clip-shape index, or -1 for none
    @location(14) stop_colour_c:  vec4<f32>,  // 3rd colour stop (multi-stop gradients)
    @location(15) stop_colour_d:  vec4<f32>,  // 4th colour stop
};

// One stacked shadow layer. `params` = (radius, offset_x, offset_y, is_inner).
struct ShadowLayer {
    colour: vec4<f32>,
    params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> shadow_layers: array<ShadowLayer>;

// One clip-mask shape (framebuffer pixels). `params` = (shape_type, rotation,
// parent_index, invert). A vertex's `clip_index` selects an entry; `parent_index`
// (or -1) chains to an enclosing mask so nested clips intersect.
struct ClipShape {
    center:    vec2<f32>,
    half_size: vec2<f32>,
    radii:     vec4<f32>,
    params:    vec4<f32>,
    pivot:     vec2<f32>,
    pad:       vec2<f32>,
};

@group(0) @binding(1) var<storage, read> clip_shapes: array<ClipShape>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos:       vec2<f32>,
    @location(1) fill_colour:     vec4<f32>,
    @location(2) border_colour:   vec4<f32>,
    @location(3) half_size:       vec2<f32>,
    @location(4) radii:           vec4<f32>,
    @location(5) border_width:    f32,
    @location(6) shape_type:      f32,
    @location(7) fill_colour2:    vec4<f32>,
    @location(8) gradient_params: vec4<f32>,
    @location(9) @interpolate(flat) shadow_index:   vec4<f32>,
    @location(10) @interpolate(flat) rotation_pivot: vec4<f32>,
    @location(11) @interpolate(flat) clip_rect: vec4<f32>,
    @location(12) @interpolate(flat) clip_index: f32,
    @location(13) @interpolate(flat) stop_colour_c:  vec4<f32>,
    @location(14) @interpolate(flat) stop_colour_d:  vec4<f32>,
    @location(15) @interpolate(flat) stop_positions: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position   = vec4<f32>(in.position, 0.0, 1.0);
    out.local_pos       = in.local_pos;
    out.fill_colour     = in.fill_colour;
    out.border_colour   = in.border_colour;
    out.half_size       = in.half_size;
    out.radii           = in.radii;
    out.border_width    = in.shape_meta.x;
    out.shape_type      = in.shape_meta.y;
    out.fill_colour2    = in.fill_colour2;
    out.gradient_params = in.gradient_params;
    out.shadow_index    = in.shadow_index;
    out.rotation_pivot  = in.rotation_pivot;
    out.clip_rect       = in.clip_rect;
    out.clip_index      = in.clip_index;
    out.stop_colour_c   = in.stop_colour_c;
    out.stop_colour_d   = in.stop_colour_d;
    out.stop_positions  = in.stop_positions;
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
// radii.x = stroke radius, radii.y = 0 (round cap) or 1 (square cap).
fn sd_line(p: vec2<f32>, hs: vec2<f32>, r: f32, square: bool) -> f32 {
    if (square) {
        // Rotate into segment frame, then use a box SDF.
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
            // Line: radii.x = stroke radius, radii.y = 0 (round) or 1 (square).
            let r = radii.x;
            let square = radii.y > 0.5;
            return sd_line(p, hs, r, square);
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

// True if the framebuffer point `fp` is outside the clip mask at `idx0` or any of
// its ancestors (masks intersect down the parent chain). Each mask is evaluated by
// its SDF in its own rotated frame. A short guard bounds the chain length.
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
    // Clip: first a cheap bounding-box reject (exact for rectangular masks), then
    // the mask's SDF and its parent chain for shape-accurate and nested clipping.
    // `clip_rect`/`clip_position.xy` are in framebuffer pixels (top-left origin).
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
    // Rotate local position by -rotation around the pivot so the SDF
    // evaluates in the shape's unrotated frame; the visible result is the
    // shape rotated by `rotation` about the pivot. With a zero pivot this is
    // rotation about the shape centre.
    let _rc = cos(-in.rotation_pivot.x);
    let _rs = sin(-in.rotation_pivot.x);
    let pivot = in.rotation_pivot.yz;
    let _pd = in.local_pos - pivot;
    let p = vec2<f32>(
        _rc * _pd.x - _rs * _pd.y,
        _rs * _pd.x + _rc * _pd.y,
    ) + pivot;
    let hs = in.half_size;

    let d = eval_sdf(p, hs, in.shape_type, in.radii);

    // Anti-aliasing: 1 pixel smoothstep at the boundary.
    let aa = 1.0;

    // shadow_index: (base_index, outer_count, inner_count, border_mode).
    let base_index = i32(in.shadow_index.x + 0.5);
    let outer_count = i32(in.shadow_index.y + 0.5);
    let inner_count = i32(in.shadow_index.z + 0.5);
    let border_mode = i32(in.shadow_index.w + 0.5);

    // Accumulate the stacked outer shadow layers (drawn behind the fill).
    // Each layer is composited src-over the previous, first layer furthest
    // back. Sample the SDF at the layer's offset in the rotated frame.
    var shadow_col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < outer_count; i = i + 1) {
        let layer = shadow_layers[base_index + i];
        let sr = layer.params.x;
        let soff = layer.params.yz;
        if (sr > 0.0 && layer.colour.a > 0.0) {
            let spd = (in.local_pos - soff) - pivot;
            let sp_r = vec2<f32>(_rc * spd.x - _rs * spd.y, _rs * spd.x + _rc * spd.y) + pivot;
            let sd = eval_sdf(sp_r, hs, in.shape_type, in.radii);
            let a = layer.colour.a * (1.0 - smoothstep(0.0, sr, sd));
            let src = vec4<f32>(layer.colour.rgb, a);
            shadow_col = vec4<f32>(
                mix(shadow_col.rgb, src.rgb, src.a),
                src.a + shadow_col.a * (1.0 - src.a),
            );
        }
    }

    let fill_alpha = 1.0 - smoothstep(-aa, 0.0, d);

    // If neither fill nor any outer shadow contributes, discard.
    if (fill_alpha <= 0.0 && shadow_col.a <= 0.0) {
        discard;
    }

    // Compute fill colour: solid, linear gradient, radial gradient, or
    // conical (sweep) gradient. gradient_params.x selects the mode:
    //   0 = solid
    //   1 = linear  (y = angle in radians)
    //   2 = radial  (centre -> edge over max half-extent)
    //   3 = conical (sweep around origin, y = offset_angle in radians)
    // stop_count selects the number of active stops (0 for solid, 2..4 for
    // multi-stop). 2-stop fills use only stops a and b at positions 0,1.
    var fill_col: vec4<f32>;
    let gtype = i32(in.gradient_params.x + 0.5);
    if (gtype == 0) {
        fill_col = in.fill_colour;
    } else {
        // Compute the gradient parameter t in [0, 1] for the chosen mode.
        var t: f32 = 0.0;
        if (gtype == 1) {
            let angle = in.gradient_params.y;
            let dir = vec2<f32>(cos(angle), sin(angle));
            let max_proj = abs(hs.x * dir.x) + abs(hs.y * dir.y);
            t = clamp(dot(p, dir) / max(max_proj, 0.001) * 0.5 + 0.5, 0.0, 1.0);
        } else if (gtype == 2) {
            let max_half = max(hs.x, hs.y);
            t = clamp(length(p) / max(max_half, 0.001), 0.0, 1.0);
        } else if (gtype == 3) {
            let two_pi = 6.28318530717958647692;
            let a = atan2(p.y, p.x) - in.gradient_params.y;
            t = fract(a / two_pi + 1.0);
        }
        // Pack the 4 stops into arrays so we can loop. WGSL doesn't index
        // vec components by a non-const i, so we copy into temps.
        var sc0 = in.fill_colour;
        var sc1 = in.fill_colour2;
        var sc2 = in.stop_colour_c;
        var sc3 = in.stop_colour_d;
        let p0 = in.stop_positions.x;
        let p1 = in.stop_positions.y;
        let p2 = in.stop_positions.z;
        let p3 = in.stop_positions.w;
        let n = i32(in.gradient_params.z + 0.5);
        // Default: clamp before first stop / after last stop.
        fill_col = sc0;
        if (t >= p3 && n >= 4) {
            fill_col = sc3;
        } else if (t >= p2 && n >= 3) {
            // Between stop 2 and stop 3.
            let span = max(p3 - p2, 0.000001);
            fill_col = mix(sc2, sc3, clamp((t - p2) / span, 0.0, 1.0));
        } else if (t >= p1) {
            // Between stop 1 and stop 2 (or end for 2-stop).
            let end = select(p1, p2, n >= 3);
            let end_col = select(sc1, sc2, n >= 3);
            let span = max(end - p1, 0.000001);
            fill_col = mix(sc1, end_col, clamp((t - p1) / span, 0.0, 1.0));
        } else if (t >= p0) {
            // Between stop 0 and stop 1.
            let span = max(p1 - p0, 0.000001);
            fill_col = mix(sc0, sc1, clamp((t - p0) / span, 0.0, 1.0));
        }
    }

    // Start with the accumulated outer shadow layers.
    var colour = shadow_col;

    // Composite fill on top of shadow.
    if (fill_alpha > 0.0) {
        let fc = vec4<f32>(fill_col.rgb, fill_col.a * fill_alpha);
        colour = vec4<f32>(
            mix(colour.rgb, fc.rgb, fc.a),
            fc.a + colour.a * (1.0 - fc.a),
        );
    }

    // Inner shadow layers: composite on top of the fill (so they tint the
    // body) but under the border. Sample the SDF at (p - offset) in the
    // rotated frame; positive sd means the offset point lies outside the
    // shape, so the current fragment is in the shadow band, fading over the
    // layer radius. Only contributes where the fragment is inside the shape.
    if (d < 0.0) {
        for (var j = 0; j < inner_count; j = j + 1) {
            let layer = shadow_layers[base_index + outer_count + j];
            let sr = layer.params.x;
            let soff = layer.params.yz;
            if (sr > 0.0 && layer.colour.a > 0.0) {
                let spd = (in.local_pos - soff) - pivot;
                let sp_ir = vec2<f32>(_rc * spd.x - _rs * spd.y, _rs * spd.x + _rc * spd.y) + pivot;
                let inner_sd = eval_sdf(sp_ir, hs, in.shape_type, in.radii);
                let inner_alpha = layer.colour.a * smoothstep(0.0, sr, inner_sd);
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

    // Border: blend border colour in a band near d = 0.
    // border_mode (shadow_index.w): 0=inset, 1=outer, 2=center.
    if (in.border_width > 0.0) {
        let bw = in.border_width;
        let bm = border_mode;
        var lo: f32;
        var hi: f32;
        if (bm == 1) {
            // Outer: band [0, bw]
            lo = 0.0;
            hi = bw;
        } else if (bm == 2) {
            // Center: band [-bw/2, bw/2]
            lo = -bw * 0.5;
            hi = bw * 0.5;
        } else {
            // Inset: band [-bw, 0]
            lo = -bw;
            hi = 0.0;
        }
        let border_alpha = (1.0 - smoothstep(hi, hi + aa, d)) * smoothstep(lo - aa, lo, d);
        let border_ref_alpha = 1.0 - smoothstep(-aa, 0.0, d - hi);
        colour = mix(colour, vec4<f32>(in.border_colour.rgb, in.border_colour.a * border_ref_alpha), border_alpha);
    }

    return colour;
}
