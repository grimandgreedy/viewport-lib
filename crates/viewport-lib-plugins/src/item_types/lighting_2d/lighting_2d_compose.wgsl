// Multiply the accumulated light over what the painter sequence has drawn so
// far: one full-screen triangle in the painter pass, the blend state doing the
// multiply.
//
// Group 0 is the pass's camera group, left as it is. Group 1: the light
// layer, read at the texel under the fragment, so the layer need not match
// the target's size.

@group(1) @binding(0) var light_target: texture_2d<f32>;

struct ComposeOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> ComposeOut {
    let x = select(-1.0, 3.0, vi == 1u);
    let y = select(-1.0, 3.0, vi == 2u);
    var out: ComposeOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

fn light_at(ndc: vec2<f32>) -> vec3<f32> {
    let dims = textureDimensions(light_target);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let size = vec2<i32>(dims);
    let texel = clamp(vec2<i32>(floor(uv * vec2<f32>(dims))), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(light_target, texel, 0).rgb;
}

// The layer as stored.
@fragment
fn fs_main(in: ComposeOut) -> @location(0) vec4<f32> {
    return vec4<f32>(light_at(in.ndc), 1.0);
}

// Half the layer, for a full-range layer over an LDR target, whose blend
// doubles what it is given.
@fragment
fn fs_half(in: ComposeOut) -> @location(0) vec4<f32> {
    return vec4<f32>(light_at(in.ndc) * 0.5, 1.0);
}
