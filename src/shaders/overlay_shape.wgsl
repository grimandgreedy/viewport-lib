// Screen-space SDF overlay shape shader.
//
// Each shape is a bounding quad whose vertices carry the shape parameters.
// The fragment shader evaluates a signed-distance function to produce
// anti-aliased fill and border regions.

struct VertexInput {
    @location(0) position:      vec2<f32>,  // NDC xy
    @location(1) local_pos:     vec2<f32>,  // pixels from shape centre
    @location(2) fill_colour:   vec4<f32>,
    @location(3) border_colour: vec4<f32>,
    @location(4) half_size:     vec2<f32>,  // shape half-extents in pixels
    @location(5) radii:         vec4<f32>,  // shape-specific params
    @location(6) border_width:  f32,
    @location(7) shape_type:    f32,        // 0 = rounded rect, 1 = circle, 2 = ellipse, 3 = capsule
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos:     vec2<f32>,
    @location(1) fill_colour:   vec4<f32>,
    @location(2) border_colour: vec4<f32>,
    @location(3) half_size:     vec2<f32>,
    @location(4) radii:         vec4<f32>,
    @location(5) border_width:  f32,
    @location(6) shape_type:    f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.local_pos     = in.local_pos;
    out.fill_colour   = in.fill_colour;
    out.border_colour = in.border_colour;
    out.half_size     = in.half_size;
    out.radii         = in.radii;
    out.border_width  = in.border_width;
    out.shape_type    = in.shape_type;
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.local_pos;
    let hs = in.half_size;

    var d: f32;
    let st = i32(in.shape_type + 0.5);

    switch (st) {
        case 1: {
            // Circle
            d = sd_circle(p, min(hs.x, hs.y));
        }
        case 2: {
            // Ellipse
            d = sd_ellipse(p, hs);
        }
        case 3: {
            // Capsule
            d = sd_capsule(p, hs);
        }
        default: {
            // Rounded rect (type 0 and fallback)
            d = sd_rounded_box(p, hs, in.radii);
        }
    }

    // Anti-aliasing: 1 pixel smoothstep at the boundary.
    let aa = 1.0;

    let fill_alpha = 1.0 - smoothstep(-aa, 0.0, d);
    if (fill_alpha <= 0.0) {
        discard;
    }

    var colour = in.fill_colour;
    colour = vec4<f32>(colour.rgb, colour.a * fill_alpha);

    // Border: blend border colour in the band near d = 0.
    if (in.border_width > 0.0) {
        let bw = in.border_width;
        // Border band occupies [-bw, 0] in SDF space (inside the fill edge).
        let border_alpha = (1.0 - smoothstep(-aa, 0.0, d)) * smoothstep(-bw - aa, -bw, d);
        colour = mix(colour, vec4<f32>(in.border_colour.rgb, in.border_colour.a * fill_alpha), border_alpha);
    }

    return colour;
}
