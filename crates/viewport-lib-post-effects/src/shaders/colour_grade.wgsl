// Ported from the viewport-lib-vfx kit (src/shaders/color_grade.wgsl), unchanged.

struct VfxUniform {
    viewport_size: vec2<f32>,
    inv_viewport_size: vec2<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
};

@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> u: VfxUniform;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let exposure = u.params0.x;
    let contrast = u.params0.y;
    let saturation = u.params0.z;
    let tint = u.params1.rgb;

    let src = textureSample(input_color, input_sampler, in.uv);
    let exposed = src.rgb * exp2(exposure);
    let contrasted = (exposed - vec3<f32>(0.5)) * contrast + vec3<f32>(0.5);
    let luma = dot(contrasted, vec3<f32>(0.2126, 0.7152, 0.0722));
    let saturated = mix(vec3<f32>(luma), contrasted, saturation);
    return vec4<f32>(max(saturated * tint, vec3<f32>(0.0)), src.a);
}
