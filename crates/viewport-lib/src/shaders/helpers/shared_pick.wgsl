
// @viewport-wgsl-version: 1
// Pick-id fragment helper for the three-target pick pass. The vertex stage must
// provide a flat-interpolated pick_id at @location(0) of the fragment input; see
// your pipeline's vertex shader for the matching declaration.
//
// The pass targets: @location(0) R32Uint object id, @location(1) R32Uint
// sub-object index (0 here; use viewport_pick_prim_fs from
// SHARED_PICK_PRIM_WGSL to write the triangle index for sub-object picking),
// and @location(2) R32Float depth (the framebuffer z, so the renderer can
// reconstruct world position on read-back). A pipeline built with
// build_pick_pipeline must write all three, which this helper does.

struct ViewportPickOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
};

@fragment
fn viewport_pick_fs(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) @interpolate(flat) pick_id: u32,
) -> ViewportPickOut {
    var out: ViewportPickOut;
    out.object_id = pick_id;
    out.primitive_id = 0u;
    out.depth = frag_pos.z;
    return out;
}
