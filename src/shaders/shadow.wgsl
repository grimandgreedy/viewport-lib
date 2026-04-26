// Shadow depth pass : renders scene from the light's point of view.
//
// Depth-only: no fragment shader needed. The GPU writes depth automatically.
// Group 0: Light uniform (light-space view-projection matrix).
// Group 1: Object uniform (model matrix : reuses the same object bind group as the main pass).

struct Light {
    view_proj: mat4x4<f32>,
};

struct Object {
    model: mat4x4<f32>,
    color: vec4<f32>,
    selected: u32,
    wireframe: u32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    has_texture: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> light: Light;
@group(1) @binding(0) var<uniform> object: Object;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return light.view_proj * object.model * vec4<f32>(position, 1.0);
}
