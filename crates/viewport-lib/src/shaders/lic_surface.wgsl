// lic_surface.wgsl : Surface LIC pass 1 -- renders each LIC mesh into lic_vector_texture.
//
// Output Rgba8Unorm: (encoded_vec_x, encoded_vec_y, encoded_strength, 1.0)
// Non-surface pixels have alpha = 0 (cleared before this pass).
// The encoded direction is the screen-space projection of the world-space flow vector,
// packed into [0,1] per channel. The advect pass decodes and follows these directions.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye:       vec4<f32>,
    near:      f32,
    far:       f32,
    _pad:      vec2<f32>,
}

struct LicObjectUniform {
    model:    mat4x4<f32>,
    // Per-item LIC strength, pre-normalised by LIC_STRENGTH_ENCODE_MAX on
    // the CPU so it fits the 8-bit blue channel.
    strength: f32,
    _pad0:    f32,
    _pad1:    f32,
    _pad2:    f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<uniform> object: LicObjectUniform;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_vec: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position:  vec3<f32>,
    @location(1) flow_attr: vec3<f32>,  // per-vertex flow vector, vertex buffer 1
) -> VertexOutput {
    let model     = object.model;
    let world_pos = (model * vec4<f32>(position, 1.0)).xyz;
    // Transform direction (w=0) to world space using the model rotation.
    let world_vec = (model * vec4<f32>(flow_attr, 0.0)).xyz;

    var out: VertexOutput;
    out.pos       = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_vec = world_vec;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Project world-space flow vector to clip space (directional, w=0).
    let clip_vec = camera.view_proj * vec4<f32>(in.world_vec, 0.0);
    // Encode screen-space xy direction into [0,1] (0.5 = zero component).
    let screen_dir = normalize(clip_vec.xy + vec2<f32>(0.0001)); // avoid zero-length
    let encoded    = screen_dir * 0.5 + vec2<f32>(0.5);

    // Blue carries the item's normalised strength; the advect pass decodes
    // it so per-item strength reaches the composite without a global knob.
    return vec4<f32>(encoded.x, encoded.y, object.strength, 1.0);
}
