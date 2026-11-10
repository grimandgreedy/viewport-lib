// Screen-space overlay backdrop shader: frosted glass.
//
// Draws a backdrop-blur shape over the blurred scene, sampled at the screen
// position, with the backdrop colour filters and a tint laid over it. An
// analytic shape is a six-vertex quad evaluated by its distance field; a
// vector path is its triangles, encoded as a sharp rect grown past the path
// so the distance test reads fully inside. Reads the same stacked
// shadow-layer texture the shape pass does, so a backdrop shape takes the
// same outer and inner layers.

@group(0) @binding(0) var t_backdrop: texture_2d<f32>;
@group(0) @binding(1) var s_backdrop: sampler;

// Soft clip coverage over `clip_shapes` and `eval_sdf` below.
// #include "helpers/overlay_clip.wgsl"

struct VertexInput {
    @location(0) position:     vec2<f32>,  // logical pixels
    @location(1) local_pos:    vec2<f32>,  // pixels from the shape pivot (or the path's centre)
    @location(2) tint:         vec4<f32>,  // laid over the filtered backdrop, alpha the mix
    @location(3) half_size:    vec2<f32>,  // shape half-extents in pixels
    @location(4) radii:        vec4<f32>,  // shape-specific params
    @location(5) shape_meta:   vec4<f32>,  // x=shape_type, y=clip_index (or -1), zw=pivot in the shape frame
    @location(6) clip_rect:    vec4<f32>,  // framebuffer-pixel clip bbox (x0,y0,x1,y1); all zero = no box clip
    @location(7) shadow_index: vec3<f32>,  // base_index, outer_count, inner_count
    @location(8) filters:      vec4<f32>,  // y=saturation, z=brightness, w=hue shift
    @location(9) linv:         vec4<f32>,  // inverse item linear map, row-major a b c d
    @location(10) item_params: vec4<f32>,  // x=item scale (shadow offsets)
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
    centre:    vec2<f32>,
    half_size: vec2<f32>,
    radii:     vec4<f32>,
    params:    vec4<f32>,
    pivot:     vec2<f32>,
    pad:       vec2<f32>,
}

// A uniform array of the renderer's `MAX_OVERLAY_CLIP_SHAPES` (256 at 64 bytes
// fills a 16 KiB uniform binding); a frame's later masks are dropped.
@group(1) @binding(0) var<uniform> clip_shapes: array<ClipShape, 256>;

// Viewport size in logical pixels (xy); zw padding. Maps local-pixel vertex
// positions to NDC so overlay geometry is independent of the viewport size.
struct Viewport {
    size: vec2<f32>,
    pad:  vec2<f32>,
};
@group(1) @binding(1) var<uniform> viewport: Viewport;

// The stacked shadow layers, shared with the solid overlay-shape pass.
@group(1) @binding(2) var shadow_layers: texture_2d<f32>;

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

fn px_to_ndc(px: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(px.x / viewport.size.x * 2.0 - 1.0, 1.0 - px.y / viewport.size.y * 2.0);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos:     vec2<f32>,
    @location(1) @interpolate(flat) tint: vec4<f32>,
    @location(2) @interpolate(flat) half_size: vec2<f32>,
    @location(3) @interpolate(flat) radii: vec4<f32>,
    // The blurred scene's coordinate: the screen position over the viewport.
    @location(4) uv:            vec2<f32>,
    @location(5) @interpolate(flat) shadow_index: vec3<f32>,
    @location(6) @interpolate(flat) filters: vec4<f32>,
    @location(7) @interpolate(flat) clip_rect:  vec4<f32>,
    // shape_type, clip_index, pivot in the shape frame xy
    @location(8) @interpolate(flat) shape_meta: vec4<f32>,
    @location(9) @interpolate(flat) linv:       vec4<f32>,
    @location(10) @interpolate(flat) item_params: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(px_to_ndc(in.position), 0.0, 1.0);
    out.local_pos     = in.local_pos;
    out.tint          = in.tint;
    out.half_size     = in.half_size;
    out.radii         = in.radii;
    out.uv            = in.position / viewport.size;
    out.shadow_index  = in.shadow_index;
    out.filters       = in.filters;
    out.clip_rect     = in.clip_rect;
    out.shape_meta    = in.shape_meta;
    out.linv          = in.linv;
    out.item_params   = in.item_params;
    return out;
}

// Map a pixel offset from the pivot into the shape's own frame: the inverse
// linear map, then the pivot's place in that frame.
fn to_shape_frame(local: vec2<f32>, linv: vec4<f32>, pivot: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(linv.x * local.x + linv.y * local.y, linv.z * local.x + linv.w * local.y) + pivot;
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
    // Clip: the bounding box (exact for rectangular masks), then the mask SDF
    // and its parent chain, each as a coverage. `clip_rect`/`clip_position.xy`
    // are in framebuffer pixels (top-left origin), same as the solid shape path.
    let fp = in.clip_position.xy;
    var clip = overlay_rect_coverage(fp, in.clip_rect);
    let clip_index = in.shape_meta.y;
    if (clip_index >= 0.0 && clip > 0.0) {
        clip = clip * overlay_mask_coverage(fp, clip_index);
    }
    if (clip <= 0.0) {
        discard;
    }

    // Into the shape's own frame, as the solid path does; a tessellated fill
    // carries the identity and its triangles are already placed.
    let pivot = in.shape_meta.zw;
    let p = to_shape_frame(in.local_pos, in.linv, pivot);
    let hs = in.half_size;
    let shape_type = in.shape_meta.x;
    let item_scale = in.item_params.x;

    let d = eval_sdf(p, hs, shape_type, in.radii);

    let aa = 1.0;

    // shadow_index: (base_index, outer_count, inner_count).
    let base_index = i32(in.shadow_index.x + 0.5);
    let outer_count = i32(in.shadow_index.y + 0.5);
    let inner_count = i32(in.shadow_index.z + 0.5);

    // Stacked outer layers behind the fill, first layer furthest back.
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

    // Sampled before the branch below: `textureSample` derives its own level
    // of detail, which WGSL allows only in uniform control flow, and
    // `fill_alpha` is not uniform. Tint enforces that; naga does not.
    let backdrop = textureSample(t_backdrop, s_backdrop, in.uv);

    // The filtered backdrop with the tint laid over it, masked by the SDF.
    if (fill_alpha > 0.0) {
        let tint = in.tint;
        let filtered = apply_backdrop_filters(
            backdrop.rgb,
            in.filters.y,
            in.filters.z,
            in.filters.w,
        );
        // The backdrop is premultiplied: opaque over the frame, and inside a
        // layer only as covered as the layer is under it.
        var fc = vec4<f32>(mix(filtered, tint.rgb, tint.a), fill_alpha);
        if (backdrop.a < 1.0) {
            let covered = backdrop.a * (1.0 - tint.a) + tint.a;
            fc = vec4<f32>(
                (filtered * (1.0 - tint.a) + tint.rgb * tint.a) / max(covered, 1.0e-5),
                fill_alpha * covered,
            );
        }
        // Source-over on un-premultiplied colours: weight the destination by
        // its own alpha. A plain mix() would let a fully clipped shadow's
        // colour bleed through the fill, since its alpha is zero but its rgb
        // is not.
        let out_a = fc.a + colour.a * (1.0 - fc.a);
        let rgb = (fc.rgb * fc.a + colour.rgb * colour.a * (1.0 - fc.a)) / max(out_a, 1.0e-5);
        colour = vec4<f32>(select(fc.rgb, rgb, out_a > 0.0), out_a);
    }

    // Stacked inner layers over the backdrop.
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

    return vec4<f32>(colour.rgb, colour.a * clip);
}
