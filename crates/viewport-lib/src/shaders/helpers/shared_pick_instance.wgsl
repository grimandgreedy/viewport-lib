
// @viewport-wgsl-version: 1
// Pick-id fragment helper for the three-target pick pass, instance variant.
// The vertex stage must provide a flat-interpolated pick_id at @location(0)
// and a flat-interpolated instance index at @location(1) of the fragment
// input (forward @builtin(instance_index) from the vertex input).
//
// Targets: @location(0) R32Uint object id, @location(1) R32Uint instance
// index, @location(2) R32Float depth.

struct ViewportPickInstOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
};

@fragment
fn viewport_pick_instance_fs(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) @interpolate(flat) pick_id: u32,
    @location(1) @interpolate(flat) instance_id: u32,
) -> ViewportPickInstOut {
    var out: ViewportPickInstOut;
    out.object_id = pick_id;
    out.primitive_id = instance_id;
    out.depth = frag_pos.z;
    return out;
}
