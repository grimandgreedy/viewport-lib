
// @viewport-wgsl-version: 1
// Pick-id fragment helper for the three-target pick pass, primitive-index
// variant. The vertex stage must provide a flat-interpolated pick_id at
// @location(0) of the fragment input, exactly as for viewport_pick_fs.
//
// Targets: @location(0) R32Uint object id, @location(1) R32Uint triangle
// index from @builtin(primitive_index), @location(2) R32Float depth.

struct ViewportPickPrimOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
};

@fragment
fn viewport_pick_prim_fs(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(primitive_index) prim_index: u32,
    @location(0) @interpolate(flat) pick_id: u32,
) -> ViewportPickPrimOut {
    var out: ViewportPickPrimOut;
    out.object_id = pick_id;
    out.primitive_id = prim_index;
    out.depth = frag_pos.z;
    return out;
}
