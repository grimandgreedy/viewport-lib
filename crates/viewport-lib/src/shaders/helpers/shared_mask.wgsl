
// @viewport-wgsl-version: 1
// Outline-mask fragment helper. Returns a single R8 value of 1.0 for any
// covered pixel; the composite reads the mask and draws the outline edge.

@fragment
fn viewport_mask_fs() -> @location(0) f32 {
    return 1.0;
}
