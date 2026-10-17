// Instanced shadow depth pass for point lights: one cubemap face per pass, with
// per-instance model matrices from the instance storage buffer.
//
// Writes linear distance-to-light (normalised by light range) into
// @builtin(frag_depth), as shadow_point.wgsl does for per-object casters.
//
// Group 0: per-face uniform (view_proj + light_pos + range, dynamic offset).
// Group 1: the instanced colour pass's per-batch group (instance storage at
//          binding 0, albedo texture and sampler at bindings 1-2).

struct PointFace {
    view_proj: mat4x4<f32>,
    // `light_pos.xyz` is the light's world position, `light_pos.w` its range.
    light_pos: vec4<f32>,
};

// Matches the 144-byte InstanceData in mesh_instanced.wgsl. Only model,
// has_texture, alpha_cutoff and alpha_flag are read here.
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

@group(0) @binding(0) var<uniform> face: PointFace;
@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(1) var obj_texture: texture_2d<f32>;
@group(1) @binding(2) var obj_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

fn light_distance(world_pos: vec3<f32>) -> f32 {
    let d = length(world_pos - face.light_pos.xyz) / max(face.light_pos.w, 1e-5);
    return clamp(d, 0.0, 1.0);
}

@vertex
fn vs_main(@location(0) position: vec3<f32>, @builtin(instance_index) idx: u32) -> VsOut {
    let world = instances[idx].model * vec4<f32>(position, 1.0);
    var out: VsOut;
    out.world_pos = world.xyz;
    out.clip_pos = face.view_proj * world;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @builtin(frag_depth) f32 {
    return light_distance(in.world_pos);
}

// Alpha-cutout variant: carries the UV, vertex alpha and instance index to the
// fragment stage so masked casters punch holes in the cubemap.
struct CutoutOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) inst_idx: u32,
    @location(3) vertex_alpha: f32,
};

@vertex
fn vs_cutout(
    @location(0) position: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) idx: u32,
) -> CutoutOut {
    let world = instances[idx].model * vec4<f32>(position, 1.0);
    var out: CutoutOut;
    out.world_pos = world.xyz;
    out.clip_pos = face.view_proj * world;
    out.uv = uv;
    out.inst_idx = idx;
    out.vertex_alpha = colour.a;
    return out;
}

@fragment
fn fs_cutout(in: CutoutOut) -> @builtin(frag_depth) f32 {
    let inst = instances[in.inst_idx];
    // Gradients in uniform control flow; the sample below is per-instance.
    let uv_ddx = dpdx(in.uv);
    let uv_ddy = dpdy(in.uv);
    // The same alpha sources as the directional cutout: vertex colour and
    // albedo. Opacity is not one, because a transparent caster casts nothing.
    if inst.alpha_flag == 1u {
        var a = in.vertex_alpha;
        if inst.has_texture == 1u {
            a = a * textureSampleGrad(obj_texture, obj_sampler, in.uv, uv_ddx, uv_ddy).a;
        }
        if a < inst.alpha_cutoff {
            discard;
        }
    }
    return light_distance(in.world_pos);
}
