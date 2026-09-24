// Screen-space SDF overlay shape shader with texture fill.
//
// Same SDF logic as overlay_shape.wgsl, but the interior samples from a
// bound texture instead of a solid fill colour. fill_colour acts as a tint
// multiplied with each texel. Reads the same stacked shadow-layer buffer the
// solid path does, so a textured shape takes the same outer and inner layers.

@group(0) @binding(0) var t_fill: texture_2d<f32>;
@group(0) @binding(1) var s_fill: sampler;

struct VertexInput {
    @location(0) position:      vec2<f32>,  // NDC xy
    @location(1) local_pos:     vec2<f32>,  // pixels from shape centre
    @location(2) fill_colour:   vec4<f32>,  // tint (multiplied with texture sample)
    @location(3) half_size:     vec2<f32>,  // shape half-extents in pixels
    @location(4) radii:         vec4<f32>,  // shape-specific params
    @location(5) shape_meta:    vec2<f32>,  // x=shape_type (0=rounded rect, 1=circle, 2=ellipse, 3=capsule, 4=ring, 5=arc, 6=triangle), y=clip_index (or -1)
    @location(6) clip_rect:     vec4<f32>,  // framebuffer-pixel clip bbox (x0,y0,x1,y1); all zero = no box clip
    @location(7) uv:            vec2<f32>,  // texture UV: (0,0)=top-left, (1,1)=bottom-right
    @location(8) shadow_index:  vec3<f32>,  // base_index, outer_count, inner_count
    @location(9) extras:         vec4<f32>, // x=blur, y=ns_centre_mode, z=ns_edge_mode, w=ns_enabled
    @location(10) nine_slice_uv:   vec4<f32>, // texture-uv insets: top,right,bottom,left
    @location(11) nine_slice_frac: vec4<f32>, // shape-fraction insets: top,right,bottom,left
    @location(12) texture_transform_a: vec4<f32>, // offset.xy, scale.xy
    @location(13) texture_transform_b: vec4<f32>, // rotation, tile_mode, flip_x, flip_y
}

// One stacked shadow layer. `params` = (blur, offset_x, offset_y, is_inner),
// `params2` = (spread, falloff, unused, unused). Same layout as the solid
// path's buffer, which is the same buffer.
struct ShadowLayer {
    colour: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
};

// Coverage of an outer layer at signed distance `sd`, matching
// `shadow_coverage` in overlay_shape.wgsl.
fn shadow_coverage(sd: f32, spread: f32, blur: f32, falloff: f32, aa: f32) -> f32 {
    let dd = sd - spread;
    let w = max(blur, aa);
    let base = 1.0 - smoothstep(0.0, w, dd);
    if (falloff == 1.0) {
        return base;
    }
    return pow(base, falloff);
}

// The inset counterpart: coverage rises from 0 at the shape edge to 1 across
// the blur band, with `spread` starting the band further inside.
fn inner_shadow_coverage(sd: f32, spread: f32, blur: f32, falloff: f32, aa: f32) -> f32 {
    let w = max(blur, aa);
    let base = smoothstep(0.0, w, sd + spread);
    if (falloff == 1.0) {
        return base;
    }
    return pow(base, falloff);
}

// One clip-mask shape (framebuffer pixels). `params` = (shape_type, rotation,
// parent_index, invert). A vertex's `clip_index` selects an entry; `parent_index`
// (or -1) chains to an enclosing mask so nested clips intersect. Matches the
// solid overlay_shape.wgsl layout so both pipelines share the clip registry.
struct ClipShape {
    center:    vec2<f32>,
    half_size: vec2<f32>,
    radii:     vec4<f32>,
    params:    vec4<f32>,
    pivot:     vec2<f32>,
    pad:       vec2<f32>,
}

@group(1) @binding(0) var<storage, read> clip_shapes: array<ClipShape>;

// Viewport size in logical pixels (xy); zw padding. Maps local-pixel vertex
// positions to NDC so overlay geometry is independent of the viewport size.
struct Viewport {
    size: vec2<f32>,
    pad:  vec2<f32>,
};
@group(1) @binding(1) var<uniform> viewport: Viewport;

// The stacked shadow layers, shared with the solid overlay-shape pass.
@group(1) @binding(2) var<storage, read> shadow_layers: array<ShadowLayer>;

fn px_to_ndc(px: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(px.x / viewport.size.x * 2.0 - 1.0, 1.0 - px.y / viewport.size.y * 2.0);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos:     vec2<f32>,
    @location(1) fill_colour:   vec4<f32>,
    @location(2) half_size:     vec2<f32>,
    @location(3) radii:         vec4<f32>,
    @location(4) shape_type:    f32,
    @location(5) uv:            vec2<f32>,
    @location(6) @interpolate(flat) shadow_index: vec3<f32>,
    @location(7) extras:       vec4<f32>,
    @location(8) @interpolate(flat) nine_slice_uv:   vec4<f32>,
    @location(9) @interpolate(flat) nine_slice_frac: vec4<f32>,
    @location(10) @interpolate(flat) texture_transform_a: vec4<f32>,
    @location(11) @interpolate(flat) texture_transform_b: vec4<f32>,
    @location(12) @interpolate(flat) clip_rect:  vec4<f32>,
    @location(13) @interpolate(flat) clip_index: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(px_to_ndc(in.position), 0.0, 1.0);
    out.local_pos     = in.local_pos;
    out.fill_colour   = in.fill_colour;
    out.half_size     = in.half_size;
    out.radii         = in.radii;
    out.shape_type    = in.shape_meta.x;
    out.uv            = in.uv;
    out.shadow_index = in.shadow_index;
    out.extras        = in.extras;
    out.nine_slice_uv   = in.nine_slice_uv;
    out.nine_slice_frac = in.nine_slice_frac;
    out.texture_transform_a = in.texture_transform_a;
    out.texture_transform_b = in.texture_transform_b;
    out.clip_rect  = in.clip_rect;
    out.clip_index = in.shape_meta.y;
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

// True if the framebuffer point `fp` is outside the clip mask at `idx0` or any of
// its ancestors (masks intersect down the parent chain). Each mask is evaluated by
// its SDF in its own rotated frame. A short guard bounds the chain length. Mirrors
// the solid overlay_shape.wgsl clip test.
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
    // Clip: cheap bounding-box reject first (exact for rectangular masks), then
    // the mask SDF and its parent chain. `clip_rect`/`clip_position.xy` are in
    // framebuffer pixels (top-left origin), same as the solid shape path.
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

    let p = in.local_pos;
    let hs = in.half_size;

    let d = eval_sdf(p, hs, in.shape_type, in.radii);

    let aa = 1.0;

    // shadow_index: (base_index, outer_count, inner_count).
    let base_index = i32(in.shadow_index.x + 0.5);
    let outer_count = i32(in.shadow_index.y + 0.5);
    let inner_count = i32(in.shadow_index.z + 0.5);

    // Stacked outer layers behind the fill, first layer furthest back.
    var shadow_col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < outer_count; i = i + 1) {
        let layer = shadow_layers[base_index + i];
        let sr = layer.params.x;
        let soff = layer.params.yz;
        let sspread = layer.params2.x;
        let sfall = layer.params2.y;
        if ((sr > 0.0 || sspread > 0.0) && layer.colour.a > 0.0) {
            let sd = eval_sdf(p - soff, hs, in.shape_type, in.radii);
            let a = layer.colour.a * shadow_coverage(sd, sspread, sr, sfall, aa);
            let src = vec4<f32>(layer.colour.rgb, a);
            shadow_col = vec4<f32>(
                mix(shadow_col.rgb, src.rgb, src.a),
                src.a + shadow_col.a * (1.0 - src.a),
            );
        }
    }

    let fill_alpha = 1.0 - smoothstep(-aa, 0.0, d);

    // Clip the outer layers to outside the silhouette, as the solid path does.
    shadow_col = vec4<f32>(shadow_col.rgb, shadow_col.a * (1.0 - fill_alpha));

    if (fill_alpha <= 0.0 && shadow_col.a <= 0.0) {
        discard;
    }

    // Start with the shadow layers.
    var colour = shadow_col;

    // The sample is hoisted out of the `fill_alpha > 0.0` branch below, with
    // the UV computation it needs. `textureSample` derives its own LOD, so
    // WGSL only permits it in uniform control flow, and `fill_alpha` comes
    // from the vertex stage. Tint enforces that and rejects the whole module;
    // naga does not, which is why this drew in Firefox and not in Chrome.
    // Sampling unconditionally costs a fetch on fully transparent fills, which
    // is the cheap half of the trade: an image fill can be mipped, so pinning
    // it to level zero instead would alias whenever it is drawn smaller than
    // its source.
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

    // Composite textured fill on top of shadow.
    if (fill_alpha > 0.0) {
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
        // Source-over on un-premultiplied colours: weight the destination by
        // its own alpha. A plain mix() would let a fully clipped shadow's
        // colour bleed through the fill, since its alpha is zero but its rgb
        // is not.
        let out_a = fc.a + colour.a * (1.0 - fc.a);
        let rgb = (fc.rgb * fc.a + colour.rgb * colour.a * (1.0 - fc.a)) / max(out_a, 1.0e-5);
        colour = vec4<f32>(select(fc.rgb, rgb, out_a > 0.0), out_a);
    }

    // Stacked inner layers over the textured fill.
    if (d < 0.0) {
        for (var j = 0; j < inner_count; j = j + 1) {
            let layer = shadow_layers[base_index + outer_count + j];
            let sr = layer.params.x;
            let soff = layer.params.yz;
            let sspread = layer.params2.x;
            let sfall = layer.params2.y;
            if ((sr > 0.0 || sspread > 0.0) && layer.colour.a > 0.0) {
                let inner_sd = eval_sdf(p - soff, hs, in.shape_type, in.radii);
                let inner_alpha =
                    layer.colour.a * inner_shadow_coverage(inner_sd, sspread, sr, sfall, aa);
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

    return colour;
}
