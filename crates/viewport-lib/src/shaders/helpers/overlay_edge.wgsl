// Fill coverage at an analytic overlay shape's outline. The including shader
// declares nothing extra: everything comes in as arguments.

// Whether the pixel centre at `p` (shape frame) lies in the box of
// half-extents `hs`, half-open: a centre on the low edge is in, one on the high
// edge is out, so two boxes sharing an edge take each pixel on it exactly once.
// The edge distances are rounded to 1/256 pixel (`aa` shape units is one pixel)
// before the test, so float error in the two neighbours' frames cannot leave
// the pixel to neither.
fn overlay_crisp_box(p: vec2<f32>, hs: vec2<f32>, aa: f32) -> bool {
    let q = 256.0 / aa;
    let lo = round((p + hs) * q);
    let hi = round((hs - p) * q);
    return all(lo >= vec2<f32>(0.0)) && all(hi > vec2<f32>(0.0));
}

// How much of the pixel the fill covers, from the signed distance `d` at its
// centre. Anti-aliased, a one-pixel ramp inside the outline. Crisp, full
// coverage when the centre is inside and none otherwise; a box (shape type 0,
// rects and rounded rects) resolves a centre on its edge with
// `overlay_crisp_box`, its rounded corners by the distance.
fn overlay_fill_coverage(d: f32, p: vec2<f32>, hs: vec2<f32>, shape_type: f32, aa: f32, crisp: bool) -> f32 {
    if (!crisp) {
        return 1.0 - smoothstep(-aa, 0.0, d);
    }
    if (i32(shape_type + 0.5) == 0) {
        return select(0.0, 1.0, overlay_crisp_box(p, hs, aa) && d <= aa / 256.0);
    }
    return select(0.0, 1.0, d <= 0.0);
}

// The band a zero-blur shadow layer's edge ramps over: one pixel, or for a
// crisp edge a sliver small enough to read as a step.
fn overlay_edge_band(aa: f32, crisp: bool) -> f32 {
    return select(aa, aa / 256.0, crisp);
}
