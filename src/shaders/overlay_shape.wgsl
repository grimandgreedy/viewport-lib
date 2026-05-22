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
    @location(6) border_width:    f32,
    @location(7) shape_type:      f32,        // 0=rect, 1=circle, 2=ellipse, 3=capsule, 4=ring, 5=arc, 6=triangle, 7=line, 8=star, 9=ngon, 10=cross
    @location(8) fill_colour2:    vec4<f32>,  // end colour for gradient (equals fill_colour for solid)
    @location(9) gradient_params: vec2<f32>,  // x=type (0=solid, 1=linear), y=angle radians
    @location(10) shadow_colour:  vec4<f32>,  // RGBA shadow colour
    @location(11) shadow_params:  vec4<f32>,  // x=radius, y=offset_x, z=offset_y
};

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
    @location(8) gradient_params: vec2<f32>,
    @location(9) shadow_colour:   vec4<f32>,
    @location(10) shadow_params:  vec4<f32>,
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
    out.border_width    = in.border_width;
    out.shape_type      = in.shape_type;
    out.fill_colour2    = in.fill_colour2;
    out.gradient_params = in.gradient_params;
    out.shadow_colour   = in.shadow_colour;
    out.shadow_params   = in.shadow_params;
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
                tp = vec2<f32>(-tp.y, tp.x);
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
            // RegularPolygon: radii.x = n (sides).
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.local_pos;
    let hs = in.half_size;

    let d = eval_sdf(p, hs, in.shape_type, in.radii);

    // Anti-aliasing: 1 pixel smoothstep at the boundary.
    let aa = 1.0;

    // Shadow/glow: draw behind the fill using a separate SDF evaluation at
    // the offset position. The shadow fades from shadow_colour at the shape
    // edge to transparent at shadow_radius pixels out.
    let shadow_r = in.shadow_params.x;
    let shadow_off = vec2<f32>(in.shadow_params.y, in.shadow_params.z);
    var shadow_a = 0.0;
    if (shadow_r > 0.0 && in.shadow_colour.a > 0.0) {
        let sd = eval_sdf(p - shadow_off, hs, in.shape_type, in.radii);
        shadow_a = in.shadow_colour.a * (1.0 - smoothstep(0.0, shadow_r, sd));
    }

    let fill_alpha = 1.0 - smoothstep(-aa, 0.0, d);

    // If neither fill nor shadow contributes, discard.
    if (fill_alpha <= 0.0 && shadow_a <= 0.0) {
        discard;
    }

    // Compute fill colour: solid or linear gradient.
    var fill_col: vec4<f32>;
    if (in.gradient_params.x > 0.5) {
        // Linear gradient: project local_pos onto gradient direction and
        // map to [0, 1] over the bounding box extent along that axis.
        let angle = in.gradient_params.y;
        let dir = vec2<f32>(cos(angle), sin(angle));
        let max_proj = abs(hs.x * dir.x) + abs(hs.y * dir.y);
        let t = clamp(dot(p, dir) / max(max_proj, 0.001) * 0.5 + 0.5, 0.0, 1.0);
        fill_col = mix(in.fill_colour, in.fill_colour2, t);
    } else {
        fill_col = in.fill_colour;
    }

    // Start with the shadow layer.
    var colour = vec4<f32>(in.shadow_colour.rgb, shadow_a);

    // Composite fill on top of shadow.
    if (fill_alpha > 0.0) {
        let fc = vec4<f32>(fill_col.rgb, fill_col.a * fill_alpha);
        colour = vec4<f32>(
            mix(colour.rgb, fc.rgb, fc.a),
            fc.a + colour.a * (1.0 - fc.a),
        );
    }

    // Border: blend border colour in a band near d = 0.
    // border_mode (shadow_params.w): 0=inset, 1=outer, 2=center.
    if (in.border_width > 0.0) {
        let bw = in.border_width;
        let bm = i32(in.shadow_params.w + 0.5);
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
