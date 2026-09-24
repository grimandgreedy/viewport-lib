// GPU object-ID pick shader for scatter volumes.
//
// A scatter volume is participating media with no surface to rasterise, so it
// answers object picks with its shape: a box volume draws the unit cube under
// a translate-and-scale, a sphere volume draws a unit icosphere the same way.
// Both shapes are world-space and unrotated, which is what lets one model
// matrix stand in for either and keeps this in step with the CPU analytic
// pick.
//
// Group 0: the shared scene bind group (camera at binding 0, clip volumes at
//          binding 6). Only clip volumes are tested, matching the renderer's
//          own object-id pass, which does not apply clip planes.
// Group 1: per-volume proxy uniform (model matrix + pick id).

/// The volume's unit-shape-to-world transform plus the id it answers with.
struct ProxyUniform {
    model:     mat4x4<f32>,
    object_id: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

@group(1) @binding(0) var<uniform> proxy: ProxyUniform;

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOut {
    var out: VertexOut;
    let world = (proxy.model * vec4<f32>(position, 1.0)).xyz;
    out.world_pos = world;
    out.clip_pos = camera.view_proj * vec4<f32>(world, 1.0);
    return out;
}

struct FragOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    if !viewport_pass_clip_volumes(in.world_pos) { discard; }

    var out: FragOut;
    out.object_id    = proxy.object_id;
    out.primitive_id = 0u;
    out.depth        = in.clip_pos.z;
    return out;
}
