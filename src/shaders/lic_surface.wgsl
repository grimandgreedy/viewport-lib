// lic_surface.wgsl : Surface LIC pass 1 -- renders each LIC mesh into lic_vector_texture.
//
// Output Rgba8Unorm: (encoded_vec_x, encoded_vec_y, noise, 1.0)
// Non-surface pixels have alpha = 0 (cleared before this pass).

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye:       vec4<f32>,
    near:      f32,
    far:       f32,
    _pad:      vec2<f32>,
}

struct LicObjectUniform {
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<uniform> object: LicObjectUniform;
@group(1) @binding(1) var<storage, read> vector_buffer: array<f32>; // 3 f32 per vertex
@group(1) @binding(2) var noise_tex: texture_2d<f32>;
@group(1) @binding(3) var noise_samp: sampler;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_vec: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vid: u32,
) -> VertexOutput {
    let i = vid * 3u;
    // Read per-vertex vector from flat storage (3 f32 per vertex).
    let local_vec = vec3<f32>(vector_buffer[i], vector_buffer[i + 1u], vector_buffer[i + 2u]);

    let model = object.model;
    let world_pos = (model * vec4<f32>(position, 1.0)).xyz;
    // Transform direction (w=0) to world space using the model rotation.
    let world_vec = (model * vec4<f32>(local_vec, 0.0)).xyz;

    var out: VertexOutput;
    out.pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_vec = world_vec;
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Project world-space vector to clip space (directional, w=0).
    let clip_vec = camera.view_proj * vec4<f32>(in.world_vec, 0.0);
    // Encode screen-space xy direction as [0,1] (0.5 = zero component).
    let screen_dir = normalize(clip_vec.xy + vec2<f32>(0.0001)); // avoid zero-vector
    let encoded = screen_dir * 0.5 + vec2<f32>(0.5);

    // Sample noise from a world-space UV to get variation.
    let noise_uv = in.world_pos.xy * 0.5;
    let noise = textureSample(noise_tex, noise_samp, noise_uv).r;

    return vec4<f32>(encoded.x, encoded.y, noise, 1.0);
}
