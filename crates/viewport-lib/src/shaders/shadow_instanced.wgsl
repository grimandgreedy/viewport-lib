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
// Matches the 144-byte InstanceData in mesh_instanced.wgsl. Only model,
// has_texture, alpha_cutoff, and alpha_flag are read here (alpha-tested shadows);
// the shading scalars live in material_gpu_buf, which the shadow pass does not
// bind, so alpha_cutoff / alpha_flag / has_texture stay per-instance.
struct InstanceData {
    model: mat4x4<f32>,                   // offset 0
    colour: vec4<f32>,                    // offset 64
    selected: u32,                        // offset 80
    wireframe: u32,                       // offset 84
    has_texture: u32,                     // offset 88
    has_normal_map: u32,                  // offset 92
    has_ao_map: u32,                      // offset 96
    unlit: u32,                           // offset 100
    receive_shadows: u32,                 // offset 104
    material_id: u32,                     // offset 108
    alpha_cutoff: f32,                    // offset 112
    alpha_flag: u32,                      // offset 116
    has_light_probe: u32,                 // offset 120
    light_probe_index: u32,               // offset 124
    ignore_clip: u32,                     // offset 128
    _pad0: u32,                           // offset 132
    _pad1: u32,                           // offset 136
    _pad2: u32,                           // offset 140
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
    out.uv = uv;  // shadow alpha-test uses raw UV; the material UV transform lives in material_gpu_buf, not bound here
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
    out.uv = uv;  // shadow alpha-test uses raw UV; the material UV transform lives in material_gpu_buf, not bound here
    out.inst_idx = real_idx;
    return out;
}

@fragment
fn fs_cutout(in: CutoutOut) {
    let inst = instances[in.inst_idx];
    // UV gradients taken here in uniform control flow; the alpha-cutout sample
    // below sits inside a per-instance branch (non-uniform), where an implicit
    // derivative is rejected by strict WGSL validators.
    let uv_ddx = dpdx(in.uv);
    let uv_ddy = dpdy(in.uv);
    if inst.alpha_flag == 1u && inst.has_texture == 1u {
        let a = textureSampleGrad(obj_texture, obj_sampler, in.uv, uv_ddx, uv_ddy).a;
        if a < inst.alpha_cutoff { discard; }
    }
}
