// skybox.wgsl : fullscreen equirectangular environment map background.
// Renders a fullscreen triangle, reconstructs world-space ray from inverse VP,
// and samples the equirect skybox texture.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
    forward: vec3<f32>,
    _pad1: f32,
    inv_view_proj: mat4x4<f32>,
};

struct Lights {
    count: u32,
    shadow_bias: f32,
    shadows_enabled: u32,
    debug_vis_mode: u32,
    sky_colour: vec3<f32>,
    hemisphere_intensity: f32,
    ground_colour: vec3<f32>,
    debug_vis_scale: f32,
    ibl_enabled: u32,
    ibl_intensity: f32,
    ibl_rotation: f32,
    show_skybox: u32,
    debug_vis_split_x: f32,
    _pad_dbg_a: u32,
    _pad_dbg_b: u32,
    _pad_dbg_c: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(3) var<uniform> lights_uniform: Lights;
@group(0) @binding(10) var ibl_sampler: sampler;
@group(0) @binding(11) var skybox_texture: texture_2d<f32>;

struct VertexOutput {
    // The skybox draws with depth_compare Equal against the cleared far
    // plane; @invariant guarantees the constant z = 1.0 written below is
    // bit-exact so the comparison cannot artifact.
    @builtin(position) @invariant pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices cover the entire screen.
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertexOutput;
    out.pos = vec4<f32>(positions[vi], 1.0, 1.0);  // depth = 1.0 (far plane)
    out.uv = positions[vi] * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;  // flip Y for UV
    return out;
}

const PI: f32 = 3.14159265;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct clip-space position.
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, 1.0, 1.0);

    // Unproject to world space.
    let world_pos = camera.inv_view_proj * ndc;
    let dir = normalize(world_pos.xyz / world_pos.w - camera.eye_pos);

    // Apply Z-axis rotation (viewport-lib is Z-up; rotating the panorama spins
    // it around the world up axis).
    let rotation = lights_uniform.ibl_rotation;
    let s = sin(rotation);
    let c = cos(rotation);
    let d = vec3<f32>(c * dir.x - s * dir.y, s * dir.x + c * dir.y, dir.z);

    // Convert direction to equirectangular UV: longitude phi around +Z, latitude
    // theta with +Z polar.
    let phi = atan2(d.y, d.x);
    let theta = asin(clamp(d.z, -1.0, 1.0));
    let uv = vec2<f32>(0.5 + phi / (2.0 * PI), 0.5 - theta / PI);

    let colour = textureSampleLevel(skybox_texture, ibl_sampler, uv, 0.0).rgb;
    return vec4<f32>(colour * lights_uniform.ibl_intensity, 1.0);
}
