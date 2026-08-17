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
    @location(10) shadow_params: vec4<f32>, // x=radius, y=offset_x, z=offset_y, w=border_mode
    @location(11) extras:        vec4<f32>, // x=blur, y=ns_centre_mode, z=ns_edge_mode, w=ns_enabled
    @location(12) nine_slice_uv:   vec4<f32>, // texture-uv insets: top,right,bottom,left
    @location(13) nine_slice_frac: vec4<f32>, // shape-fraction insets: top,right,bottom,left
    @location(14) texture_transform_a: vec4<f32>, // offset.xy, scale.xy
    @location(15) texture_transform_b: vec4<f32>, // rotation, tile_mode, flip_x, flip_y
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
    @location(10) extras:       vec4<f32>,
    @location(11) @interpolate(flat) nine_slice_uv:   vec4<f32>,
    @location(12) @interpolate(flat) nine_slice_frac: vec4<f32>,
    @location(13) @interpolate(flat) texture_transform_a: vec4<f32>,
    @location(14) @interpolate(flat) texture_transform_b: vec4<f32>,
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
    out.extras        = in.extras;
    out.nine_slice_uv   = in.nine_slice_uv;
    out.nine_slice_frac = in.nine_slice_frac;
    out.texture_transform_a = in.texture_transform_a;
    out.texture_transform_b = in.texture_transform_b;
    return out;
}

// Remap one component of the shape UV through the 9-slice piecewise function.
//   shape_uv : 0..1 across the shape's bounding box along this axis
//   start_frac, end_frac : insets as shape fraction (left/right or top/bottom)
//   start_uv, end_uv     : matching insets in texture UV
//   edge_mode            : 0=stretch, 1=tile (for the corner/edge bands)
//   centre_mode          : 0=stretch, 1=tile (for the centre band)
// Returns the texture UV component to sample.
fn ninepatch_axis(
    shape_uv: f32,
    start_frac: f32,
    end_frac: f32,
    start_uv: f32,
    end_uv: f32,
    edge_mode: f32,
    centre_mode: f32,
) -> f32 {
    // Three regions along this axis: [0, start_frac), [start_frac, 1 - end_frac), [1 - end_frac, 1].
    if (shape_uv < start_frac) {
        // Leading edge band: shape_uv * (start_uv / start_frac) for stretch,
        // or (shape_uv * (1/start_frac) modulo 1) * start_uv for tile.
        let s = max(start_frac, 0.00001);
        let t = shape_uv / s;
        let tiled = fract(t);
        let used_t = mix(t, tiled, edge_mode);
        return used_t * start_uv;
    } else if (shape_uv > 1.0 - end_frac) {
        let e = max(end_frac, 0.00001);
        let t = (shape_uv - (1.0 - end_frac)) / e;
        let tiled = fract(t);
        let used_t = mix(t, tiled, edge_mode);
        return (1.0 - end_uv) + used_t * end_uv;
    } else {
        // Centre band.
        let span_shape = max(1.0 - start_frac - end_frac, 0.00001);
        let span_tex = max(1.0 - start_uv - end_uv, 0.00001);
        let t = (shape_uv - start_frac) / span_shape;
        let tiled = fract(t * (span_shape / span_tex));
        let used_t = mix(t, tiled, centre_mode);
        return start_uv + used_t * span_tex;
    }
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

// Apply saturation, brightness, and hue-rotation to a backdrop colour.
// sat/bright are multipliers (1.0 = unchanged); hue is in radians. Hue
// rotation uses Rodrigues rotation around the grey axis (1,1,1)/sqrt(3).
fn apply_backdrop_filters(rgb: vec3<f32>, sat: f32, bright: f32, hue: f32) -> vec3<f32> {
    var c = rgb;
    if (abs(hue) > 0.0001) {
        let k = vec3<f32>(0.57735026919);
        let ca = cos(hue);
        let sa = sin(hue);
        c = c * ca + cross(k, c) * sa + k * dot(k, c) * (1.0 - ca);
    }
    let luma = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    c = mix(vec3<f32>(luma, luma, luma), c, sat);
    c = c * bright;
    return max(c, vec3<f32>(0.0, 0.0, 0.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.local_pos;
    let hs = in.half_size;

    let d = eval_sdf(p, hs, in.shape_type, in.radii);

    let aa = 1.0;

    // Decode shadow_params.w: combined = border_mode + 3 * inset_shadow.
    let combined_w = i32(in.shadow_params.w + 0.5);
    let inset_shadow = combined_w >= 3;
    let border_mode = combined_w % 3;

    // Outer shadow behind the fill (only when inset_shadow is off).
    let shadow_r = in.shadow_params.x;
    let shadow_off = vec2<f32>(in.shadow_params.y, in.shadow_params.z);
    var shadow_a = 0.0;
    if (!inset_shadow && shadow_r > 0.0 && in.shadow_colour.a > 0.0) {
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
        // 9-slice remap when enabled: rebuild the sample UV by remapping
        // each axis through the piecewise function so corners stay at
        // their authored size and edges/centre tile or stretch per mode.
        var sample_uv = in.uv;
        if (in.extras.w > 0.5) {
            let u = ninepatch_axis(
                in.uv.x,
                in.nine_slice_frac.w,   // left frac
                in.nine_slice_frac.y,   // right frac
                in.nine_slice_uv.w,     // left uv
                in.nine_slice_uv.y,     // right uv
                in.extras.z,            // edge mode
                in.extras.y,            // centre mode
            );
            let v = ninepatch_axis(
                in.uv.y,
                in.nine_slice_frac.x,   // top frac
                in.nine_slice_frac.z,   // bottom frac
                in.nine_slice_uv.x,     // top uv
                in.nine_slice_uv.z,     // bottom uv
                in.extras.z,
                in.extras.y,
            );
            sample_uv = vec2<f32>(u, v);
        } else {
            // Texture transform path: scale + rotate around (0.5, 0.5),
            // translate, flip, then apply tile_mode. Identity transform
            // (scale=1, no rotation/offset/flip, Stretch) is a no-op.
            let off = in.texture_transform_a.xy;
            let scl = in.texture_transform_a.zw;
            let rot = in.texture_transform_b.x;
            let tile = i32(in.texture_transform_b.y + 0.5);
            let flip_x = in.texture_transform_b.z > 0.5;
            let flip_y = in.texture_transform_b.w > 0.5;
            var uvc = in.uv - vec2<f32>(0.5, 0.5);
            // Scale: multiply the UV by `scale`. `scale = 1.0` is 1:1.
            // Larger scale widens the sampled range (more tiles with Tile
            // mode, or sees more of the texture with Stretch/Mirror).
            uvc = uvc * scl;
            // Rotate around the centre.
            if (abs(rot) > 0.000001) {
                let c = cos(rot);
                let s = sin(rot);
                uvc = vec2<f32>(c * uvc.x - s * uvc.y, s * uvc.x + c * uvc.y);
            }
            var uvt = uvc + vec2<f32>(0.5, 0.5) + off;
            if (flip_x) { uvt.x = 1.0 - uvt.x; }
            if (flip_y) { uvt.y = 1.0 - uvt.y; }
            // Apply tile mode for sample lookup.
            if (tile == 1) {
                // Tile: wrap.
                uvt = fract(uvt - floor(uvt));
            } else if (tile == 2) {
                // Mirror: ping-pong.
                let m = uvt - 2.0 * floor(uvt * 0.5);
                uvt = vec2<f32>(
                    select(m.x, 2.0 - m.x, m.x > 1.0),
                    select(m.y, 2.0 - m.y, m.y > 1.0),
                );
            } else {
                // Stretch: clamp.
                uvt = clamp(uvt, vec2<f32>(0.0), vec2<f32>(1.0));
            }
            sample_uv = uvt;
        }
        let tex_sample = textureSample(t_fill, s_fill, sample_uv);
        var fc: vec4<f32>;
        if (in.extras.x > 0.5) {
            // Backdrop blur: bound texture is the scene-blur output (opaque
            // RGBA). Apply the backdrop colour filters (extras.yzw =
            // saturation, brightness, hue-shift), overlay the tint on top,
            // then mask by the SDF.
            let tint = in.fill_colour;
            let filtered = apply_backdrop_filters(
                tex_sample.rgb,
                in.extras.y,
                in.extras.z,
                in.extras.w,
            );
            let blended_rgb = mix(filtered, tint.rgb, tint.a);
            fc = vec4<f32>(blended_rgb, fill_alpha);
        } else {
            // Regular texture fill: multiply by tint colour. Works correctly
            // for opaque PNGs (logos, icons, screenshots)
            let tinted = tex_sample * in.fill_colour;
            fc = vec4<f32>(tinted.rgb, tinted.a * fill_alpha);
        }
        colour = vec4<f32>(
            mix(colour.rgb, fc.rgb, fc.a),
            fc.a + colour.a * (1.0 - fc.a),
        );
    }

    // Inner shadow: composite on top of the textured fill, under the border.
    if (inset_shadow && shadow_r > 0.0 && in.shadow_colour.a > 0.0 && d < 0.0) {
        let inner_sd = eval_sdf(p - shadow_off, hs, in.shape_type, in.radii);
        let inner_alpha = in.shadow_colour.a * smoothstep(0.0, shadow_r, inner_sd);
        if (inner_alpha > 0.0) {
            let ic = vec4<f32>(in.shadow_colour.rgb, inner_alpha);
            colour = vec4<f32>(
                mix(colour.rgb, ic.rgb, ic.a),
                ic.a + colour.a * (1.0 - ic.a),
            );
        }
    }

    // Border: blend border colour in a band near d = 0.
    // border_mode (low part of shadow_params.w): 0=inset, 1=outer, 2=center.
    if (in.border_width > 0.0) {
        let bw = in.border_width;
        let bm = border_mode;
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
