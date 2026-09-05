// Shadow depth pass : renders scene from the light's point of view.
//
// Depth-only for the plain pipelines: no fragment shader needed, the GPU
// writes depth automatically. The cutout pipelines add a fragment stage
// (`fs_cutout`) that samples the caster's albedo alpha and discards below
// its cutoff, so an `AlphaMode::Mask` material punches holes in the shadow
// instead of casting a solid silhouette.
// Group 0: Light uniform (light-space view-projection matrix).
// Group 1: Object uniform (model matrix : reuses the same object bind group as the main pass).

struct Light {
    view_proj: mat4x4<f32>,
};

// Mirrors the real `Object` layout from mesh.wgsl field-for-field (same
// storage buffer, same struct on the Rust side), naming only the fields the
// shadow pass reads and padding the rest so offsets stay aligned. `_pad`
// fields are never read; their names record which real field occupies that
// slot for anyone diffing this against mesh.wgsl.
struct Object {
    model: mat4x4<f32>,                    // offset   0
    _pad_colour: vec4<f32>,                // offset  64
    _pad_selected: u32,                    // offset  80
    _pad_wireframe: u32,                   // offset  84
    _pad_ambient: f32,                     // offset  88
    _pad_diffuse: f32,                     // offset  92
    _pad_specular: f32,                    // offset  96
    _pad_shininess: f32,                   // offset 100
    has_texture: u32,                      // offset 104 : 1 when the albedo texture (binding 1) is bound
    _pad_use_pbr: u32,                     // offset 108
    _pad_metallic: f32,                    // offset 112
    _pad_roughness: f32,                   // offset 116
    _pad_has_normal_map: u32,              // offset 120
    _pad_has_ao_map: u32,                  // offset 124
    _pad_has_attribute: u32,               // offset 128
    _pad_scalar_min: f32,                  // offset 132
    _pad_scalar_max: f32,                  // offset 136
    _pad_receive_shadows: u32,             // offset 140
    _pad_nan_colour: vec4<f32>,            // offset 144
    _pad_use_nan_colour: u32,              // offset 160
    _pad_use_matcap: u32,                  // offset 164
    _pad_matcap_blendable: u32,            // offset 168
    _pad_unlit: u32,                       // offset 172
    _pad_use_face_colour: u32,             // offset 176
    _pad_uv_vis_mode: u32,                 // offset 180
    _pad_uv_vis_scale: f32,                // offset 184
    _pad_backface_policy: u32,             // offset 188
    _pad_backface_colour: vec4<f32>,       // offset 192
    _pad_has_warp: u32,                    // offset 208
    _pad_warp_scale: f32,                  // offset 212
    has_position_override: u32,            // offset 216 : 1 when a per-vertex position storage buffer is bound at binding 13
    has_normal_override: u32,              // offset 220
    _pad_emissive: vec3<f32>,              // offset 224
    _pad_use_flat: u32,                    // offset 236
    alpha_mode: u32,                       // offset 240 : 0=Opaque 1=Mask 2=Blend
    alpha_cutoff: f32,                     // offset 244
    _pad_has_metallic_roughness_tex: u32,  // offset 248
    _pad_has_emissive_tex: u32,            // offset 252
    uv_transform: vec4<f32>,               // offset 256 : (offset.xy, scale.xy)
    deform_flags: u32,                     // offset 272 : bit i set when deformer slot i is active for this draw
    _pad_normal_strength: f32,             // offset 276
    _pad_ao_range: vec2<f32>,              // offset 280
    _pad_metallic_range: vec2<f32>,        // offset 288
    _pad_roughness_range: vec2<f32>,       // offset 296
    position_override_base: u32,           // offset 304 : first vec3 element read from binding 13 (pool slicing)
    position_override_len: u32,            // offset 308 : element count of the window; 0xffffffff = whole buffer
};

@group(0) @binding(0) var<uniform> light: Light;
// Indexed per-object storage array (see mesh.wgsl); the caster's element is
// selected by @builtin(instance_index) and copied into `object` at entry.
@group(1) @binding(0) var<storage, read> objects: array<Object>;
var<private> object: Object;
// Albedo texture + sampler, only bound for the cutout pipelines (the plain
// pipelines never reference them, and naga strips unused bindings).
@group(1) @binding(1) var obj_texture: texture_2d<f32>;
@group(1) @binding(2) var obj_sampler: sampler;
// Per-vertex position override (binding 13 of the shared object bind group).
// When a `GpuPlugin` drives positions through `set_position_override_buffer`
// the mesh's own vertex buffer still holds the rest pose, so the shadow
// caster has to read the override here or it rasterises stale geometry into
// the atlas (a flat plane for a displaced wave, all instances stacked at the
// rest origin for a replicated mesh). Same flat `array<f32>`, 3 per vertex,
// as the main pass. The normal override (binding 14) is not needed for a
// depth-only pass.
@group(1) @binding(13) var<storage, read> position_override_buffer: array<f32>;

// Per-vertex deformation hook contract.
// #include "helpers/deform.wgsl"

fn shadow_clip_position(position: vec3<f32>, vertex_index: u32, instance_index: u32) -> vec4<f32> {
    object = objects[instance_index];
    // Override replaces the vertex-buffer position outright, matching the main
    // mesh pass; deformers layer on top.
    var local_pos = position;
    if object.has_position_override != 0u && vertex_index < object.position_override_len {
        let pi = (object.position_override_base + vertex_index) * 3u;
        let plen = arrayLength(&position_override_buffer);
        if pi + 2u < plen {
            local_pos = vec3<f32>(
                position_override_buffer[pi],
                position_override_buffer[pi + 1u],
                position_override_buffer[pi + 2u],
            );
        }
    }
    var dv = DeformVertex(local_pos, vec3<f32>(0.0, 0.0, 1.0), vertex_index);
    let dctx = DeformContext(object.model, object.model[3].xyz, 0.0, object.deform_flags, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let world_pos4 = object.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv = viewport_deform_world_space(dv, dctx);
    return light.view_proj * vec4<f32>(dv.position, 1.0);
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> @builtin(position) vec4<f32> {
    return shadow_clip_position(position, vertex_index, instance_index);
}

// Alpha-cutout variant: carries the UV and object index to the fragment
// stage so it can sample the caster's albedo alpha and discard cut-out
// fragments (leaf gaps) instead of casting a solid silhouette.
struct CutoutOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) obj_idx: u32,
};

@vertex
fn vs_cutout(
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> CutoutOut {
    var out: CutoutOut;
    out.clip_pos = shadow_clip_position(position, vertex_index, instance_index);
    out.uv = uv;
    out.obj_idx = instance_index;
    return out;
}

@fragment
fn fs_cutout(in: CutoutOut) {
    // UV gradients taken unconditionally, in uniform control flow; the
    // alpha-cutout sample below sits inside a per-object branch
    // (non-uniform, `in.obj_idx` varies per instance), where an implicit
    // derivative is rejected by strict WGSL validators.
    let uv_ddx = dpdx(in.uv);
    let uv_ddy = dpdy(in.uv);
    let obj = objects[in.obj_idx];
    if obj.alpha_mode == 1u && obj.has_texture == 1u {
        let mat_uv = in.uv * obj.uv_transform.zw + obj.uv_transform.xy;
        let muv_ddx = uv_ddx * obj.uv_transform.zw;
        let muv_ddy = uv_ddy * obj.uv_transform.zw;
        let a = textureSampleGrad(obj_texture, obj_sampler, mat_uv, muv_ddx, muv_ddy).a;
        if a < obj.alpha_cutoff {
            discard;
        }
    }
}
