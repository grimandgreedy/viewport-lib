// GPU object-ID pick shader for decal projection boxes.
//
// A decal has no geometry of its own: it shades whatever surfaces its
// projection box covers. For picking it stands in for itself with that box,
// so a click near a decal but off its receiver still selects it, matching the
// CPU decal pick. The box silhouette can therefore extend past the shaded
// footprint into empty space, which is deliberate.
//
// Group 0: the shared scene bindings. Only clip volumes are tested, matching
//          the lib's own object-id pass, which does not apply clip planes.
// Group 1: per-decal proxy uniform (model matrix + pick id).

/// The decal's unit-cube-to-world transform plus the id it answers picks with.
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
