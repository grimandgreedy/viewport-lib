// Surface LIC vector pass: draws each flow surface into the vector target.
//
// Output Rgba8Unorm: (dir_x, dir_y, strength, 1). The direction is the flow
// vector's direction on screen at the surface point, in NDC (y up, both axes
// spanning -1 to 1), packed into [0, 1] per channel. The advect pass turns it
// into pixels, since only it knows the target size. Alpha is
// 0 wherever no flow surface is the visible one, because the target is cleared
// to transparent and the pass is depth-tested against the scene.

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_vec: vec3<f32>,
    @location(1) strength: f32,
    @location(2) clip_pos: vec4<f32>,
    @location(3) world_pos: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) flow: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    // x: the item's strength, normalised to fit the 8-bit blue channel.
    @location(6) params: vec4<f32>,
) -> VertexOutput {
    let model = mat4x4<f32>(model_0, model_1, model_2, model_3);
    // World position first, as the mesh colour shaders do, so the depth this
    // pass tests with is the depth they wrote.
    let world_pos = (model * vec4<f32>(position, 1.0)).xyz;

    var out: VertexOutput;
    out.pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.clip_pos = out.pos;
    out.world_vec = (model * vec4<f32>(flow, 0.0)).xyz;
    out.strength = params.x;
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clipped with the surface it sits on. A discarded fragment leaves the
    // target transparent, which the advect pass reads as no flow surface.
    if !viewport_clip_test(in.world_pos) {
        discard;
    }
    // Which way the point moves on screen when it moves along the flow: the
    // derivative of clip.xy / clip.w, less its positive 1 / w^2 factor.
    let clip_vec = camera.view_proj * vec4<f32>(in.world_vec, 0.0);
    let ndc_dir = clip_vec.xy * in.clip_pos.w - in.clip_pos.xy * clip_vec.w;
    // The offset keeps a zero vector from normalising to NaN.
    let screen_dir = normalize(ndc_dir + vec2<f32>(0.0001));
    let encoded = screen_dir * 0.5 + vec2<f32>(0.5);
    return vec4<f32>(encoded.x, encoded.y, in.strength, 1.0);
}
