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
// binding 5: visibility_indices — only present in the GPU-culling cull variant pipeline.
@group(1) @binding(5) var<storage, read> visibility_indices: array<u32>;

@vertex
fn vs_main(@location(0) position: vec3<f32>, @builtin(instance_index) idx: u32) -> @builtin(position) vec4<f32> {
    return light.view_proj * instances[idx].model * vec4<f32>(position, 1.0);
}

// GPU-driven culling variant: reads the actual instance index from the per-cascade
// visibility index buffer written by the cull compute pass.
@vertex
fn vs_shadow_cull(@location(0) position: vec3<f32>, @builtin(instance_index) idx: u32) -> @builtin(position) vec4<f32> {
    return light.view_proj * instances[visibility_indices[idx]].model * vec4<f32>(position, 1.0);
}
