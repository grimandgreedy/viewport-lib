// Screen-space SDF overlay shape shader with texture fill.
//
// Same SDF logic as overlay_shape.wgsl, but the interior samples from a
// bound texture instead of a solid fill colour. fill_colour acts as a tint
// multiplied with each texel. Supports optional outer shadow/glow.

@group(0) @binding(0) var t_fill: texture_2d<f32>;
@group(0) @binding(1) var s_fill: sampler;

struct VertexInput {
    @location(0) position:      vec2<f32>,  // NDC xy
    @location(1) local_pos:     vec2<f32>,  // pixels from shape centre
    @location(2) fill_colour:   vec4<f32>,  // tint (multiplied with texture sample)
    @location(3) border_colour: vec4<f32>,
    @location(4) half_size:     vec2<f32>,  // shape half-extents in pixels
    @location(5) radii:         vec4<f32>,  // shape-specific params
    @location(6) border_width:  f32,
    @location(7) shape_type:    f32,        // 0=rounded rect, 1=circle, 2=ellipse, 3=capsule, 4=ring, 5=arc, 6=triangle
    @location(8) uv:            vec2<f32>,  // texture UV: (0,0)=top-left, (1,1)=bottom-right
    @location(9) shadow_colour: vec4<f32>,  // RGBA shadow colour
    @location(10) shadow_params: vec4<f32>, // x=radius, y=offset_x, z=offset_y
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
    @location(7) uv:            vec2<f32>,
    @location(8) shadow_colour: vec4<f32>,
    @location(9) shadow_params: vec4<f32>,
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
    out.uv            = in.uv;
    out.shadow_colour = in.shadow_colour;
    out.shadow_params = in.shadow_params;
    return out;
}

// Signed distance to a rounded box with per-corner radii.
// radii: x = top-right, y = bottom-right, z = bottom-left, w = top-left
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

    let aa = 1.0;

    // Shadow/glow behind the fill.
    let shadow_r = in.shadow_params.x;
    let shadow_off = vec2<f32>(in.shadow_params.y, in.shadow_params.z);
    var shadow_a = 0.0;
    if (shadow_r > 0.0 && in.shadow_colour.a > 0.0) {
        let sd = eval_sdf(p - shadow_off, hs, in.shape_type, in.radii);
        shadow_a = in.shadow_colour.a * (1.0 - smoothstep(0.0, shadow_r, sd));
    }

    let fill_alpha = 1.0 - smoothstep(-aa, 0.0, d);

    if (fill_alpha <= 0.0 && shadow_a <= 0.0) {
        discard;
    }

    // Start with the shadow layer.
    var colour = vec4<f32>(in.shadow_colour.rgb, shadow_a);

    // Composite textured fill on top of shadow.
    if (fill_alpha > 0.0) {
        let tex_sample = textureSample(t_fill, s_fill, in.uv);
        let tinted = tex_sample * in.fill_colour;
        let fc = vec4<f32>(tinted.rgb, tinted.a * fill_alpha);
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
            lo = 0.0;
            hi = bw;
        } else if (bm == 2) {
            lo = -bw * 0.5;
            hi = bw * 0.5;
        } else {
            lo = -bw;
            hi = 0.0;
        }
        let border_alpha = (1.0 - smoothstep(hi, hi + aa, d)) * smoothstep(lo - aa, lo, d);
        let border_ref_alpha = 1.0 - smoothstep(-aa, 0.0, d - hi);
        colour = mix(colour, vec4<f32>(in.border_colour.rgb, in.border_colour.a * border_ref_alpha), border_alpha);
    }

    return colour;
}
