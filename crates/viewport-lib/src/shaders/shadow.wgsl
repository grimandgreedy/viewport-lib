// Shadow depth pass : renders scene from the light's point of view.
//
// Depth-only for the plain pipelines: no fragment shader needed, the GPU
// writes depth automatically. The cutout pipelines add a fragment stage
// (`fs_cutout`) that samples the caster's albedo alpha and discards below
// its cutoff, so an `AlphaMode::Mask` material punches holes in the shadow
// instead of casting a solid silhouette.
// Group 0: Light uniform (light-space view-projection matrix).
// Group 1: the mesh group 1 the colour pass binds: the record buffer, the
// albedo for cutouts, the overrides and, for the culled entry points, the
// cascade's visible list at binding 15.

struct Light {
    view_proj: mat4x4<f32>,
};

// #include "helpers/instance_data.wgsl"

@group(0) @binding(0) var<uniform> light: Light;
// Indexed per-object storage array (see mesh.wgsl); the caster's element is
// selected by @builtin(instance_index) and copied into `object` at entry.
@group(1) @binding(0) var<storage, read> objects: array<InstanceData>;
var<private> object: InstanceData;
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
// The mesh's sidecar record chunk, for the override window.
@group(1) @binding(6) var<storage, read> sidecar_pool: array<u32>;
// #include "helpers/sidecar.wgsl"
// Record index of each visible slot, written by the cascade cull. Read only
// by the `_cull` entry points, which draw from the cull's indirect args.
@group(1) @binding(15) var<storage, read> visibility_indices: array<u32>;

// Per-vertex deformation hook contract.
// #include "helpers/deform.wgsl"

// The keep value `shadow_clip_position` found, for the cutout entry point to
// pass on. Only written when some deformer defines `keep`.
var<private> shadow_keep: f32 = 1.0;

fn shadow_clip_position(position: vec3<f32>, vertex_index: u32, record: u32) -> vec4<f32> {
    object = objects[record];
    // The draw binds at the record's base vertex, so this is the mesh-local
    // index the overrides and deformers read by.
    let vi = vertex_index - object.vertex_base;
    // Override replaces the vertex-buffer position outright, matching the main
    // mesh pass; deformers layer on top.
    var local_pos = position;
    let rec = object.sidecar_record;
    if (object.flags & INST_HAS_POSITION_OVERRIDE) != 0u && vi < sc_word(rec + SC_POS_LEN) {
        let pi = (sc_word(rec + SC_POS_BASE) + vi) * 3u;
        let plen = arrayLength(&position_override_buffer);
        if pi + 2u < plen {
            local_pos = vec3<f32>(
                position_override_buffer[pi],
                position_override_buffer[pi + 1u],
                position_override_buffer[pi + 2u],
            );
        }
    }
    // With no deformer the world position is not needed, so the matrices
    // multiply first. The order changes the depth's rounding, and the shadow
    // reference images are blessed with this one.
    if object.deform_flags == 0u {
        return light.view_proj * object.model * vec4<f32>(local_pos, 1.0);
    }
    var dv = DeformVertex(local_pos, vec3<f32>(0.0, 0.0, 1.0), vi);
    let dctx = DeformContext(object.model, object.model[3].xyz, 0.0, object.deform_flags, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let world_pos4 = object.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv = viewport_deform_world_space(dv, dctx);
    // <viewport-deform-keep> shadow_keep = viewport_deform_keep(dv, dctx);
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

@vertex
fn vs_main_cull(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> @builtin(position) vec4<f32> {
    return shadow_clip_position(position, vertex_index, visibility_indices[instance_index]);
}

// Alpha-cutout variant: carries the UV and object index to the fragment
// stage so it can sample the caster's albedo alpha and discard cut-out
// fragments (leaf gaps) instead of casting a solid silhouette.
struct CutoutOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) obj_idx: u32,
    // Vertex-colour alpha, one of the mask's alpha sources with the texture.
    @location(3) vertex_alpha: f32,
    // Kept-surface value from deformers that define `keep`. A caster that can
    // be cut draws through this cutout path, since the plain one has no
    // fragment stage to discard in.
    // <viewport-deform-keep> @location(2) deform_keep: f32,
};

fn cutout_vertex(position: vec3<f32>, colour: vec4<f32>, uv: vec2<f32>, vertex_index: u32, record: u32) -> CutoutOut {
    var out: CutoutOut;
    out.clip_pos = shadow_clip_position(position, vertex_index, record);
    out.uv = uv;
    out.obj_idx = record;
    out.vertex_alpha = colour.a;
    // <viewport-deform-keep> out.deform_keep = shadow_keep;
    return out;
}

@vertex
fn vs_cutout(
    @location(0) position: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> CutoutOut {
    return cutout_vertex(position, colour, uv, vertex_index, instance_index);
}

@vertex
fn vs_cutout_cull(
    @location(0) position: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> CutoutOut {
    return cutout_vertex(position, colour, uv, vertex_index, visibility_indices[instance_index]);
}

@fragment
fn fs_cutout(in: CutoutOut) {
    // UV gradients taken unconditionally, in uniform control flow; the
    // alpha-cutout sample below sits inside a per-object branch
    // (non-uniform, `in.obj_idx` varies per instance), where an implicit
    // derivative is rejected by strict WGSL validators.
    let uv_ddx = dpdx(in.uv);
    let uv_ddy = dpdy(in.uv);
    // <viewport-deform-keep> if in.deform_keep < 0.0 { discard; }
    let obj = objects[in.obj_idx];
    // Same alpha sources as the surface: vertex colour and albedo. Opacity is
    // not one, because a caster below full opacity casts no shadow at all.
    if (obj.flags & INST_ALPHA_MASK) != 0u {
        var a = in.vertex_alpha;
        if (obj.flags & INST_HAS_TEXTURE) != 0u {
            let mat_uv = in.uv;
            let muv_ddx = uv_ddx;
            let muv_ddy = uv_ddy;
            a = a * textureSampleGrad(obj_texture, obj_sampler, mat_uv, muv_ddx, muv_ddy).a;
        }
        if a < obj.alpha_cutoff {
            discard;
        }
    }
}
