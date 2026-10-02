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

// The per-object layout the main pass writes, reproduced field for field the way shadow.wgsl
// does rather than padded to the two fields this pass happens to read. A depth pass that
// gets an offset wrong reads someone else's field and fails silently, and the padded form
// has to be re-derived by hand every time a field is added between the ones it names.
struct Object {
    model: mat4x4<f32>,                    // offset   0
    _pad_colour: vec4<f32>,                // offset  64
    _pad_selected: u32,                    // offset  80
    _pad_wireframe: u32,                   // offset  84
    _pad_ambient: f32,                     // offset  88
    _pad_diffuse: f32,                     // offset  92
    _pad_specular: f32,                    // offset  96
    _pad_shininess: f32,                   // offset 100
    _pad_has_texture: u32,                 // offset 104
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
    _pad_has_normal_override: u32,         // offset 220
    _pad_emissive: vec3<f32>,              // offset 224
    _pad_use_flat: u32,                    // offset 236
    _pad_alpha_mode: u32,                  // offset 240
    _pad_alpha_cutoff: f32,                // offset 244
    _pad_has_metallic_roughness_tex: u32,  // offset 248
    _pad_has_emissive_tex: u32,            // offset 252
    _pad_uv_transform: vec4<f32>,          // offset 256
    deform_flags: u32,                     // offset 272 : bit i set when deformer slot i is active for this draw
    _pad_normal_strength: f32,             // offset 276
    _pad_ao_range: vec2<f32>,              // offset 280
    _pad_metallic_range: vec2<f32>,        // offset 288
    _pad_roughness_range: vec2<f32>,       // offset 296
    position_override_base: u32,           // offset 304 : first vec3 element read from binding 13 (pool slicing)
    position_override_len: u32,            // offset 308 : element count of the window; 0xffffffff = whole buffer
};

@group(0) @binding(0) var<uniform> face: PointFace;
// Indexed per-object storage array (see mesh.wgsl); the caster's element is
// selected by @builtin(instance_index) and copied into `object` at entry.
@group(1) @binding(0) var<storage, read> objects: array<Object>;
var<private> object: Object;
// Per-vertex position override (binding 13 of the shared object bind group). Read here for the
// same reason shadow.wgsl reads it: a `GpuPlugin` driving positions through
// `set_position_override_buffer` leaves the mesh's own vertex buffer at the rest pose, so a
// caster that does not read the override rasterises stale geometry into the cube face. The
// normal override (binding 14) is not needed for a depth-only pass.
@group(1) @binding(13) var<storage, read> position_override_buffer: array<f32>;

// Per-vertex deformation hook contract.
// #include "helpers/deform.wgsl"

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos:      vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    object = objects[instance_index];
    // Override replaces the vertex-buffer position outright, matching the main mesh pass and the
    // cascade pass; deformers layer on top.
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
    var out: VsOut;
    out.world_pos = dv.position;
    out.clip_pos = face.view_proj * vec4<f32>(dv.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @builtin(frag_depth) f32 {
    let d = length(in.world_pos - face.light_pos.xyz) / max(face.light_pos.w, 1e-5);
    return clamp(d, 0.0, 1.0);
}
