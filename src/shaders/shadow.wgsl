// Shadow depth pass : renders scene from the light's point of view.
//
// Depth-only: no fragment shader needed. The GPU writes depth automatically.
// Group 0: Light uniform (light-space view-projection matrix).
// Group 1: Object uniform (model matrix : reuses the same object bind group as the main pass).

struct Light {
    view_proj: mat4x4<f32>,
};

// Shadow needs `object.model`, the position-override flag, and
// `object.deform_flags`. The struct names the interleaved fields the shadow
// pass reads and pads across the rest to match the renderer's
// `ObjectUniform` layout.
struct Object {
    model: mat4x4<f32>,                    // offset   0
    _pad0: array<vec4<u32>, 9>,            // offsets 64..208
    _pad1: u32,                            // offset 208
    _pad2: u32,                            // offset 212
    has_position_override: u32,            // offset 216 : 1 when a per-vertex position storage buffer is bound at binding 13
    has_normal_override: u32,              // offset 220
    _pad3: array<vec4<u32>, 3>,            // offsets 224..272
    deform_flags: u32,                     // offset 272 : bit i set when deformer slot i is active for this draw
    // Individual u32 pads across 276..304: a uniform-space array would need a
    // 16-byte element stride and 28 bytes is not a multiple of 16.
    _pad4a: u32,                           // offset 276
    _pad4b: u32,                           // offset 280
    _pad4c: u32,                           // offset 284
    _pad4d: u32,                           // offset 288
    _pad4e: u32,                           // offset 292
    _pad4f: u32,                           // offset 296
    _pad4g: u32,                           // offset 300
    position_override_base: u32,           // offset 304 : first vec3 element read from binding 13 (pool slicing)
    position_override_len: u32,            // offset 308 : element count of the window; 0xffffffff = whole buffer
};

@group(0) @binding(0) var<uniform> light: Light;
// Indexed per-object storage array (see mesh.wgsl); the caster's element is
// selected by @builtin(instance_index) and copied into `object` at entry.
@group(1) @binding(0) var<storage, read> objects: array<Object>;
var<private> object: Object;
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

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> @builtin(position) vec4<f32> {
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
