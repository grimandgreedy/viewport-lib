// Ported from the viewport-lib-vfx kit (src/shaders/edge_detect.wgsl), unchanged.

struct VfxUniform {
    viewport_size: vec2<f32>,
    inv_viewport_size: vec2<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(1) var input_depth: texture_depth_2d;
@group(0) @binding(2) var input_sampler: sampler;
@group(0) @binding(3) var<uniform> u: VfxUniform;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let p = positions[vertex_index];
    var out: VsOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color_strength = u.params0.x;
    let depth_strength = u.params0.y;
    let threshold = u.params0.z;
    let edge_color = u.params1.rgb;

    let texel = u.inv_viewport_size;
    let center = textureSample(input_color, input_sampler, in.uv);
    let left = textureSample(input_color, input_sampler, in.uv + vec2<f32>(-texel.x, 0.0));
    let right = textureSample(input_color, input_sampler, in.uv + vec2<f32>(texel.x, 0.0));
    let up = textureSample(input_color, input_sampler, in.uv + vec2<f32>(0.0, -texel.y));
    let down = textureSample(input_color, input_sampler, in.uv + vec2<f32>(0.0, texel.y));

    let pixel = vec2<i32>(in.uv * u.viewport_size);
    let px = max(pixel.x, 1);
    let py = max(pixel.y, 1);
    let sx = i32(u.viewport_size.x) - 2;
    let sy = i32(u.viewport_size.y) - 2;
    let p = vec2<i32>(min(px, sx), min(py, sy));
    let d_left = textureLoad(input_depth, p + vec2<i32>(-1, 0), 0);
    let d_right = textureLoad(input_depth, p + vec2<i32>(1, 0), 0);
    let d_up = textureLoad(input_depth, p + vec2<i32>(0, -1), 0);
    let d_down = textureLoad(input_depth, p + vec2<i32>(0, 1), 0);

    let color_edge = abs(luminance(left.rgb) - luminance(right.rgb))
        + abs(luminance(up.rgb) - luminance(down.rgb));
    let depth_edge = abs(d_left - d_right) + abs(d_up - d_down);
    let edge = smoothstep(threshold, threshold * 2.0, color_edge * color_strength + depth_edge * depth_strength);
    return vec4<f32>(mix(center.rgb, edge_color, edge), center.a);
}
