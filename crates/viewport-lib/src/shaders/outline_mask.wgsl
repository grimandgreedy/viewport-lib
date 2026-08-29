// outline_mask.wgsl : renders selected objects as white (r=1.0) into an R8
// mask texture. The edge detection pass reads this mask to produce a smooth
// anti-aliased outline.
//
// Group 0: Camera bind group (only view_proj is used).
// Group 1: OutlineUniform (only model is used).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

struct OutlineUniform {
    model: mat4x4<f32>,
    colour: vec4<f32>,
    pixel_offset: f32,
    has_position_override: u32,
    position_override_base: u32,
    position_override_len: u32,
    deform_flags: u32,
    _deform_pad0: u32,
    _deform_pad1: u32,
    _deform_pad2: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> outline: OutlineUniform;
// Per-vertex position override, mirroring the main mesh pass (binding 13
// there). Without it the mask rasterises the bind pose and the halo detaches
// from geometry driven through `set_position_override_buffer`. Bound to a
// small fallback buffer when the outlined item has no override.
@group(1) @binding(1) var<storage, read> position_override_buffer: array<f32>;

// Per-vertex deformation hook contract.
// #include "helpers/deform.wgsl"

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    // Override replaces the vertex-buffer position outright, matching
    // mesh.wgsl; deformers layer on top.
    var local_pos = position;
    if outline.has_position_override != 0u && vertex_index < outline.position_override_len {
        let pi = (outline.position_override_base + vertex_index) * 3u;
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
    let dctx = DeformContext(outline.model, outline.model[3].xyz, 0.0, outline.deform_flags, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let world_pos4 = outline.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv = viewport_deform_world_space(dv, dctx);
    return camera.view_proj * vec4<f32>(dv.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
