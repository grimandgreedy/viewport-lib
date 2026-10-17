// Instanced shadow depth pass : renders scene from the light's POV using
// per-instance model matrices from a storage buffer.
//
// Group 0: Light uniform (view-projection).
// Group 1: Storage buffer containing array<InstanceData>.

struct Light {
    view_proj: mat4x4<f32>,
};

// #include "helpers/instance_data.wgsl"

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
    // Vertex-colour alpha, one of the mask's alpha sources with the texture.
    @location(2) vertex_alpha: f32,
};

@vertex
fn vs_cutout(
    @location(0) position: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) idx: u32,
) -> CutoutOut {
    let inst = instances[idx];
    var out: CutoutOut;
    out.clip_pos = light.view_proj * inst.model * vec4<f32>(position, 1.0);
    out.uv = uv;  // shadow alpha-test uses raw UV; the material UV transform lives in material_gpu_buf, not bound here
    out.vertex_alpha = colour.a;
    out.inst_idx = idx;
    return out;
}

@vertex
fn vs_cutout_cull(
    @location(0) position: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) idx: u32,
) -> CutoutOut {
    let real_idx = visibility_indices[idx];
    let inst = instances[real_idx];
    var out: CutoutOut;
    out.clip_pos = light.view_proj * inst.model * vec4<f32>(position, 1.0);
    out.uv = uv;  // shadow alpha-test uses raw UV; the material UV transform lives in material_gpu_buf, not bound here
    out.vertex_alpha = colour.a;
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
    // Same alpha sources as the surface: vertex colour and albedo. Opacity is
    // not one, because a transparent batch casts no shadow at all.
    if (inst.flags & INST_ALPHA_MASK) != 0u {
        var a = in.vertex_alpha;
        if (inst.flags & INST_HAS_TEXTURE) != 0u {
            a = a * textureSampleGrad(obj_texture, obj_sampler, in.uv, uv_ddx, uv_ddy).a;
        }
        if a < inst.alpha_cutoff { discard; }
    }
}
