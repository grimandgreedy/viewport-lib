// Clip coverage for the overlay shaders. A clip is soft: each test returns
// how much of the pixel at the framebuffer point `fp` it keeps, and the
// shaders multiply that into the fragment's alpha instead of discarding at a
// hard edge.
//
// The including shader declares `clip_shapes` (the frame's mask array, with
// `ClipShape` entries in framebuffer pixels) and `eval_sdf`.

// How much of the one-pixel square at `fp` the box `r` (x0, y0, x1, y1) covers.
// A box without area is no clip and keeps everything. An edge on a pixel
// boundary keeps or drops whole pixels, as a hard test at pixel centres would.
fn overlay_rect_coverage(fp: vec2<f32>, r: vec4<f32>) -> f32 {
    if (!(r.z > r.x && r.w > r.y)) {
        return 1.0;
    }
    let lo = max(fp - vec2<f32>(0.5), r.xy);
    let hi = min(fp + vec2<f32>(0.5), r.zw);
    let c = clamp(hi - lo, vec2<f32>(0.0), vec2<f32>(1.0));
    return c.x * c.y;
}

// How much of the pixel at `fp` the clip mask at `idx0` and its ancestors keep
// (the parent chain intersects, so coverages multiply). Each mask is evaluated
// by its SDF in its own rotated frame, with a one-pixel ramp centred on the
// edge. A short guard bounds the chain length.
fn overlay_mask_coverage(fp: vec2<f32>, idx0: f32) -> f32 {
    var idx = i32(idx0);
    var guard = 0;
    var coverage = 1.0;
    loop {
        if (idx < 0 || guard >= 8 || coverage <= 0.0) {
            break;
        }
        let c = clip_shapes[idx];
        let rc = cos(-c.params.y);
        let rs = sin(-c.params.y);
        let pd = (fp - c.centre) - c.pivot;
        let p = vec2<f32>(rc * pd.x - rs * pd.y, rs * pd.x + rc * pd.y) + c.pivot;
        coverage = coverage * clamp(0.5 - eval_sdf(p, c.half_size, c.params.x, c.radii), 0.0, 1.0);
        idx = i32(c.params.z);
        guard = guard + 1;
    }
    return coverage;
}
