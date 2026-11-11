// Present a reference target: draw it into the whole output, each output
// pixel taking the reference texel under its centre (nearest sampling), with
// the bar colour where the scaled target does not reach.
//
// Group 0, binding 0 : the reference target
// Group 0, binding 1 : where it lands in the output

struct Present {
    // The scaled target's top-left corner, in output pixels.
    origin_px: vec2<f32>,
    // Output pixels per reference pixel.
    scale: vec2<f32>,
    // The reference target's size in pixels.
    reference_size: vec2<f32>,
    // 1.0 when output pixels outside the target take `bar_colour`.
    bars: f32,
    _pad: f32,
    // Premultiplied linear colour of the bars.
    bar_colour: vec4<f32>,
};

@group(0) @binding(0) var t_reference: texture_2d<f32>;
@group(0) @binding(1) var<uniform> present: Present;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle: (-1,-1), (3,-1), (-1,3).
    let x = f32((vi & 1u) * 2u);
    let y = f32((vi >> 1u) * 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // The output pass covers the whole target, so the fragment position is
    // the pixel centre in output pixels.
    let r = (pos.xy - present.origin_px) / present.scale;
    let outside = r.x < 0.0 || r.y < 0.0 || r.x >= present.reference_size.x
        || r.y >= present.reference_size.y;
    let texel = clamp(
        vec2<i32>(floor(r)),
        vec2<i32>(0, 0),
        vec2<i32>(present.reference_size) - vec2<i32>(1, 1),
    );
    let colour = textureLoad(t_reference, texel, 0);
    return select(colour, present.bar_colour, outside && present.bars > 0.5);
}
