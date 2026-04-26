// Instanced shadow depth pass : renders scene from the light's POV using
// per-instance model matrices from a storage buffer.
//
// Group 0: Light uniform (view-projection).
// Group 1: Storage buffer containing array<InstanceData>.

struct Light {
    view_proj: mat4x4<f32>,
};

struct InstanceData {
    model: mat4x4<f32>,
    color: vec4<f32>,
    selected: u32,
    wireframe: u32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    has_texture: u32,
    use_pbr: u32,
    metallic: f32,
    roughness: f32,
    has_normal_map: u32,
    has_ao_map: u32,
};

@group(0) @binding(0) var<uniform> light: Light;
@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;

@vertex
fn vs_main(@location(0) position: vec3<f32>, @builtin(instance_index) idx: u32) -> @builtin(position) vec4<f32> {
    return light.view_proj * instances[idx].model * vec4<f32>(position, 1.0);
}
