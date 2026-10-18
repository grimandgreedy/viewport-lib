// Shadow depth pass for point lights : renders one cubemap face per pass.
//
// Writes linear distance-to-light (normalised by light range) into
// @builtin(frag_depth). The lit pass then compares fragment distance against
// this linear depth, sidestepping the inverse perspective non-linearity that
// makes biasing point shadows awkward.
//
// Group 0: per-face uniform (view_proj + light_pos + range, dynamic offset).
// Group 1: object uniform (model + deform_flags, shared with main pass).
// Group 2: deformer data (shared with main pass).

struct PointFace {
    view_proj: mat4x4<f32>,
    // Packed as vec4 to avoid vec3 + f32 uniform-buffer layout ambiguity:
    // `light_pos.xyz` is the light's world position, `light_pos.w` is the
    // light range (used to normalise frag_depth into [0, 1]).
    light_pos: vec4<f32>,
};

// #include "helpers/instance_data.wgsl"

@group(0) @binding(0) var<uniform> face: PointFace;
// Indexed per-object storage array (see mesh.wgsl); the caster's element is
// selected by @builtin(instance_index) and copied into `object` at entry.
@group(1) @binding(0) var<storage, read> objects: array<InstanceData>;
var<private> object: InstanceData;
// Albedo texture + sampler, read only by the cutout pipeline.
@group(1) @binding(1) var obj_texture: texture_2d<f32>;
@group(1) @binding(2) var obj_sampler: sampler;
// Per-vertex position override (binding 13 of the shared object bind group). Read here for the
// same reason shadow.wgsl reads it: a `GpuPlugin` driving positions through
// `set_position_override_buffer` leaves the mesh's own vertex buffer at the rest pose, so a
// caster that does not read the override rasterises stale geometry into the cube face. The
// normal override (binding 14) is not needed for a depth-only pass.
@group(1) @binding(13) var<storage, read> position_override_buffer: array<f32>;
// The mesh's sidecar record chunk, for the override window.
@group(1) @binding(6) var<storage, read> sidecar_pool: array<u32>;
// #include "helpers/sidecar.wgsl"

// Per-vertex deformation hook contract.
// #include "helpers/deform.wgsl"

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos:      vec3<f32>,
    // Kept-surface value from deformers that define `keep`; see deform.wgsl.
    // <viewport-deform-keep> @location(1) deform_keep: f32,
};

// World position of a caster vertex: the position override and deformers,
// as the main pass applies them. Also returns the deformers' keep value.
struct CasterVertex {
    world_pos: vec3<f32>,
    keep: f32,
};

fn caster_vertex(position: vec3<f32>, vertex_index: u32, instance_index: u32) -> CasterVertex {
    object = objects[instance_index];
    // The draw binds at the record's base vertex, so this is the mesh-local
    // index the overrides and deformers read by.
    let vi = vertex_index - object.vertex_base;
    // Override replaces the vertex-buffer position outright, matching the main mesh pass and the
    // cascade pass; deformers layer on top.
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
    var dv = DeformVertex(local_pos, vec3<f32>(0.0, 0.0, 1.0), vi);
    let dctx = DeformContext(object.model, object.model[3].xyz, 0.0, object.deform_flags, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let world_pos4 = object.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv = viewport_deform_world_space(dv, dctx);
    var out: CasterVertex;
    out.world_pos = dv.position;
    out.keep = 0.0;
    // <viewport-deform-keep> out.keep = viewport_deform_keep(dv, dctx);
    return out;
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    let v = caster_vertex(position, vertex_index, instance_index);
    var out: VsOut;
    // <viewport-deform-keep> out.deform_keep = v.keep;
    out.world_pos = v.world_pos;
    out.clip_pos = face.view_proj * vec4<f32>(v.world_pos, 1.0);
    return out;
}

fn light_distance(world_pos: vec3<f32>) -> f32 {
    let d = length(world_pos - face.light_pos.xyz) / max(face.light_pos.w, 1e-5);
    return clamp(d, 0.0, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @builtin(frag_depth) f32 {
    // <viewport-deform-keep> if in.deform_keep < 0.0 { discard; }
    return light_distance(in.world_pos);
}

// Alpha-cutout variant for `AlphaMode::Mask` casters: carries the UV, vertex
// alpha and object index to the fragment stage, which discards below the
// cutoff so the cubemap gets the same holes as the cascade atlas.
struct CutoutOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    // <viewport-deform-keep> @location(1) deform_keep: f32,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) obj_idx: u32,
    @location(4) vertex_alpha: f32,
};

@vertex
fn vs_cutout(
    @location(0) position: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> CutoutOut {
    let v = caster_vertex(position, vertex_index, instance_index);
    var out: CutoutOut;
    // <viewport-deform-keep> out.deform_keep = v.keep;
    out.world_pos = v.world_pos;
    out.clip_pos = face.view_proj * vec4<f32>(v.world_pos, 1.0);
    out.uv = uv;
    out.obj_idx = instance_index;
    out.vertex_alpha = colour.a;
    return out;
}

@fragment
fn fs_cutout(in: CutoutOut) -> @builtin(frag_depth) f32 {
    // Gradients in uniform control flow; the sample below is per-object.
    let uv_ddx = dpdx(in.uv);
    let uv_ddy = dpdy(in.uv);
    // <viewport-deform-keep> if in.deform_keep < 0.0 { discard; }
    let obj = objects[in.obj_idx];
    // The same alpha sources as the surface: vertex colour and albedo.
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
    return light_distance(in.world_pos);
}
