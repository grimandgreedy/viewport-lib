// Instanced shadow depth pass : renders scene from the light's POV using
// per-instance model matrices from a storage buffer.
//
// Group 0: Light uniform (view-projection).
// Group 1: Storage buffer containing array<InstanceData>.

struct Light {
    view_proj: mat4x4<f32>,
};

// Layout must match the WGSL `InstanceData` struct in `mesh_instanced.wgsl`
// and the Rust `InstanceData` in `resources/types.rs`. The shadow vertex
// stage only reads `model`, but the storage-buffer stride must agree with
// the CPU upload size so subsequent instances align correctly.
struct InstanceData {
    model: mat4x4<f32>,
    colour: vec4<f32>,
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
    unlit: u32,
    receive_shadows: u32,
    use_flat: u32,
    normal_strength: f32,
    uv_transform: vec4<f32>,
    ao_range: vec2<f32>,
    alpha_cutoff: f32,                    // Mask cutoff (albedo alpha threshold)
    alpha_flag: u32,                      // 1 = alpha-test enabled, 0 = off
};

@group(0) @binding(0) var<uniform> light: Light;
@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;
// binding 5: visibility_indices, only present in the GPU-culling cull variant pipeline.
@group(1) @binding(5) var<storage, read> visibility_indices: array<u32>;

// Albedo texture + sampler, co-located in group 1 (bindings 1-2, matching the
// instance/cull BGLs). Bound only for the alpha-cutout pipelines.
@group(1) @binding(1) var obj_texture: texture_2d<f32>;
@group(1) @binding(2) var obj_sampler: sampler;

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

// Alpha-cutout variants: carry the UV and instance index to the fragment stage so
// the depth pass can discard cut-out fragments (leaf gaps) instead of casting a
// solid silhouette.
struct CutoutOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) inst_idx: u32,
};

@vertex
fn vs_cutout(
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) idx: u32,
) -> CutoutOut {
    let inst = instances[idx];
    var out: CutoutOut;
    out.clip_pos = light.view_proj * inst.model * vec4<f32>(position, 1.0);
    out.uv = uv * inst.uv_transform.zw + inst.uv_transform.xy;
    out.inst_idx = idx;
    return out;
}

@vertex
fn vs_cutout_cull(
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) idx: u32,
) -> CutoutOut {
    let real_idx = visibility_indices[idx];
    let inst = instances[real_idx];
    var out: CutoutOut;
    out.clip_pos = light.view_proj * inst.model * vec4<f32>(position, 1.0);
    out.uv = uv * inst.uv_transform.zw + inst.uv_transform.xy;
    out.inst_idx = real_idx;
    return out;
}

@fragment
fn fs_cutout(in: CutoutOut) {
    let inst = instances[in.inst_idx];
    if inst.alpha_flag == 1u && inst.has_texture == 1u {
        let a = textureSample(obj_texture, obj_sampler, in.uv).a;
        if a < inst.alpha_cutoff { discard; }
    }
}
