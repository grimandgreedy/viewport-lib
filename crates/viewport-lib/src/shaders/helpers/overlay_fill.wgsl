// Overlay fill evaluation, shared by the overlay shape and text pipelines.
//
// An item's fill is a record in a data texture of `Rgba32Float` texels, read
// with `textureLoad`, so the overlay family runs on devices without storage
// buffers. A record is an eight-texel header followed by the gradient stops,
// any number of them; an item names its record by the index of its first
// texel, and -1 is no fill, which evaluates to white so the caller's own
// colour passes through unchanged.
//
// Header texels:
//   0: kind, stop count, colour-matrix index (or -1), nine-slice flag
//   1: box half-extents xy; z = angle (linear), seam (conical) or arc start
//      (sweep); w = arc span (sweep)
//   2: sweep centre xy in pixels from the box centre; nine-slice centre and
//      edge tile modes zw; x = the registered function (custom)
//   3: a solid colour, or the tint a texture sample or a custom fill is
//      multiplied by
//   4: texture transform: offset xy, scale zw
//   5: texture transform: rotation, tile mode, flip x, flip y
//   6: nine-slice insets in texture UV (top, right, bottom, left)
//   7: nine-slice insets as box fractions (top, right, bottom, left)
// then `count` stop colours, then the stop positions four to a texel. A
// custom fill has no stops: its sixteen parameters follow the header as four
// texels. The
// renderer's `overlay_fill` module writes the same layout. A colour matrix is
// five texels anywhere in the table, its columns: the weights of red, green,
// blue and alpha, then the offsets.
//
// The including shader binds group 1 as the fill records, the image a texture
// fill samples (a blank texel for any other draw) and its sampler, and calls
// `overlay_fill` with a local coordinate: pixels from the centre of the box
// the fill spans, in the item's own frame. It multiplies the result into its
// coverage. `overlay_fill` samples the image, so it has to be called from
// uniform control flow.

@group(1) @binding(0) var overlay_fills: texture_2d<f32>;
@group(1) @binding(1) var fill_image: texture_2d<f32>;
@group(1) @binding(2) var fill_sampler: sampler;

// Texels per row of the fill-record texture (the renderer's constant of the
// same name).
const FILL_TEXELS_PER_ROW: i32 = 1024;
const FILL_HEADER_TEXELS: i32 = 8;

const FILL_LINEAR: i32 = 1;
const FILL_RADIAL: i32 = 2;
const FILL_CONICAL: i32 = 3;
const FILL_SWEEP: i32 = 4;
const FILL_TEXTURE: i32 = 5;
const FILL_SOLID: i32 = 6;
const FILL_CUSTOM: i32 = 7;

// What a registered fill function receives besides its local coordinate: the
// half-extents of the box the fill spans, in the same pixels, and the item's
// sixteen parameters.
struct OverlayFillParams {
    half_size: vec2<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
};

// Calls the registered fill function `id`. The renderer replaces this
// function with one that switches over the registered functions when the
// overlay pipelines are built; with none registered every custom fill is
// transparent.
fn overlay_custom_fill(id: i32, local: vec2<f32>, params: OverlayFillParams) -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn fill_texel(i: i32) -> vec4<f32> {
    return textureLoad(
        overlay_fills,
        vec2<i32>(i % FILL_TEXELS_PER_ROW, i / FILL_TEXELS_PER_ROW),
        0,
    );
}

// Position of stop `i` of the record at `base`, which has `count` stops.
fn fill_stop_position(base: i32, count: i32, i: i32) -> f32 {
    let t = fill_texel(base + FILL_HEADER_TEXELS + count + i / 4);
    switch (i % 4) {
        case 0: { return t.x; }
        case 1: { return t.y; }
        case 2: { return t.z; }
        default: { return t.w; }
    }
}

// Interpolate the sorted stops of the record at `base` at `t`, holding the end
// stops outside their range.
fn fill_gradient(base: i32, count: i32, t: f32) -> vec4<f32> {
    let colours = base + FILL_HEADER_TEXELS;
    var prev_colour = fill_texel(colours);
    var prev_pos = fill_stop_position(base, count, 0);
    if (t <= prev_pos) {
        return prev_colour;
    }
    for (var i = 1; i < count; i = i + 1) {
        let colour = fill_texel(colours + i);
        let pos = fill_stop_position(base, count, i);
        if (t < pos) {
            let span = max(pos - prev_pos, 0.000001);
            return mix(prev_colour, colour, clamp((t - prev_pos) / span, 0.0, 1.0));
        }
        prev_colour = colour;
        prev_pos = pos;
    }
    return prev_colour;
}

// Remap one component of the box UV through the nine-slice piecewise function.
//   box_uv               : 0..1 across the box along this axis
//   start_frac, end_frac : insets as box fractions (left/right or top/bottom)
//   start_uv, end_uv     : matching insets in texture UV
//   edge_mode            : 0 = stretch, 1 = tile, for the corner and edge bands
//   centre_mode          : 0 = stretch, 1 = tile, for the centre band
fn fill_ninepatch_axis(
    box_uv: f32,
    start_frac: f32,
    end_frac: f32,
    start_uv: f32,
    end_uv: f32,
    edge_mode: f32,
    centre_mode: f32,
) -> f32 {
    if (box_uv < start_frac) {
        let s = max(start_frac, 0.00001);
        let t = box_uv / s;
        let used_t = mix(t, fract(t), edge_mode);
        return used_t * start_uv;
    } else if (box_uv > 1.0 - end_frac) {
        let e = max(end_frac, 0.00001);
        let t = (box_uv - (1.0 - end_frac)) / e;
        let used_t = mix(t, fract(t), edge_mode);
        return (1.0 - end_uv) + used_t * end_uv;
    } else {
        let span_box = max(1.0 - start_frac - end_frac, 0.00001);
        let span_tex = max(1.0 - start_uv - end_uv, 0.00001);
        let t = (box_uv - start_frac) / span_box;
        let tiled = fract(t * (span_box / span_tex));
        let used_t = mix(t, tiled, centre_mode);
        return start_uv + used_t * span_tex;
    }
}

// The image coordinate a texture fill samples at `local`: the box mapped to
// 0..1, then the nine-slice remap or the texture transform.
fn fill_texture_uv(local: vec2<f32>, base: i32) -> vec2<f32> {
    let head = fill_texel(base);
    let hs = fill_texel(base + 1).xy;
    let box_uv = (local + hs) / (2.0 * max(hs, vec2<f32>(0.5, 0.5)));
    if (head.w > 0.5) {
        let modes = fill_texel(base + 2);
        let ns_uv = fill_texel(base + 6);
        let ns_frac = fill_texel(base + 7);
        let u = fill_ninepatch_axis(box_uv.x, ns_frac.w, ns_frac.y, ns_uv.w, ns_uv.y, modes.w, modes.z);
        let v = fill_ninepatch_axis(box_uv.y, ns_frac.x, ns_frac.z, ns_uv.x, ns_uv.z, modes.w, modes.z);
        return vec2<f32>(u, v);
    }
    // Scale and rotate about the middle, translate, flip, then the tile mode.
    // The identity transform is a no-op.
    let tt_a = fill_texel(base + 4);
    let tt_b = fill_texel(base + 5);
    var uvc = (box_uv - vec2<f32>(0.5, 0.5)) * tt_a.zw;
    let rot = tt_b.x;
    if (abs(rot) > 0.000001) {
        let c = cos(rot);
        let s = sin(rot);
        uvc = vec2<f32>(c * uvc.x - s * uvc.y, s * uvc.x + c * uvc.y);
    }
    var uvt = uvc + vec2<f32>(0.5, 0.5) + tt_a.xy;
    if (tt_b.z > 0.5) { uvt.x = 1.0 - uvt.x; }
    if (tt_b.w > 0.5) { uvt.y = 1.0 - uvt.y; }
    let tile = i32(tt_b.y + 0.5);
    if (tile == 1) {
        uvt = fract(uvt - floor(uvt));
    } else if (tile == 2) {
        let m = uvt - 2.0 * floor(uvt * 0.5);
        uvt = vec2<f32>(
            select(m.x, 2.0 - m.x, m.x > 1.0),
            select(m.y, 2.0 - m.y, m.y > 1.0),
        );
    } else {
        uvt = clamp(uvt, vec2<f32>(0.0), vec2<f32>(1.0));
    }
    return uvt;
}

// The colour of fill record `fill` at `local`, in linear RGBA, straight alpha.
fn overlay_fill(local: vec2<f32>, fill: i32) -> vec4<f32> {
    let base = max(fill, 0);
    // Sampled whatever the kind: the kind is per item, and `textureSample`
    // derives its own level of detail, which WGSL allows only in uniform
    // control flow. Tint rejects the module otherwise; naga does not.
    let image = textureSample(fill_image, fill_sampler, fill_texture_uv(local, base));
    if (fill < 0) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let head = fill_texel(base);
    let kind = i32(head.x + 0.5);
    let colour = fill_texel(base + 3);
    if (kind == FILL_SOLID) {
        return colour;
    }
    if (kind == FILL_TEXTURE) {
        return image * colour;
    }
    let params = fill_texel(base + 1);
    let hs = params.xy;
    if (kind == FILL_CUSTOM) {
        let p = OverlayFillParams(
            hs,
            fill_texel(base + FILL_HEADER_TEXELS),
            fill_texel(base + FILL_HEADER_TEXELS + 1),
            fill_texel(base + FILL_HEADER_TEXELS + 2),
            fill_texel(base + FILL_HEADER_TEXELS + 3),
        );
        let id = i32(fill_texel(base + 2).x + 0.5);
        return overlay_custom_fill(id, local, p) * colour;
    }
    let two_pi = 6.28318530717958647692;
    var t = 0.0;
    if (kind == FILL_LINEAR) {
        let dir = vec2<f32>(cos(params.z), sin(params.z));
        let max_proj = abs(hs.x * dir.x) + abs(hs.y * dir.y);
        t = clamp(dot(local, dir) / max(max_proj, 0.001) * 0.5 + 0.5, 0.0, 1.0);
    } else if (kind == FILL_RADIAL) {
        let max_half = max(hs.x, hs.y);
        t = clamp(length(local) / max(max_half, 0.001), 0.0, 1.0);
    } else if (kind == FILL_CONICAL) {
        let a = atan2(local.y, local.x) - params.z;
        t = fract(a / two_pi + 1.0);
    } else if (kind == FILL_SWEEP) {
        let d = local - fill_texel(base + 2).xy;
        let a = atan2(d.y, d.x) - params.z;
        t = clamp(fract(a / two_pi + 1.0) * two_pi / max(params.w, 0.000001), 0.0, 1.0);
    }
    return fill_gradient(base, i32(head.y + 0.5), t);
}

// The colour filter the record at `fill` names, applied to the straight-alpha
// `colour` the item draws: a 4x5 matrix over `[r, g, b, a, 1]`, clamped. With
// no record or no matrix the colour passes through untouched.
fn overlay_colour_matrix(colour: vec4<f32>, fill: i32) -> vec4<f32> {
    if (fill < 0) {
        return colour;
    }
    let m = fill_texel(fill).z;
    if (m < 0.0) {
        return colour;
    }
    let k = i32(m + 0.5);
    let out = fill_texel(k) * colour.r
        + fill_texel(k + 1) * colour.g
        + fill_texel(k + 2) * colour.b
        + fill_texel(k + 3) * colour.a
        + fill_texel(k + 4);
    return clamp(out, vec4<f32>(0.0), vec4<f32>(1.0));
}
