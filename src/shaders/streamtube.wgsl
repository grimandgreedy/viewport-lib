// Streamtube shader : connected tube mesh renderer.
//
// The CPU generates a full connected tube mesh (parallel-transport frame, SIDES=12)
// with world-space positions and outward-facing normals baked in.  This shader
// simply transforms the mesh into clip space and applies Blinn-Phong shading.
//
// Group 0: Camera uniform (view-projection, eye position) + ClipPlanes + ClipVolume.
// Group 1: StreamtubeUniform : color (vec4) + radius (f32, unused here : mesh already scaled).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:  u32,
    _pad0:  u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct StreamtubeUniform {
    color:  vec4<f32>,
    radius: f32,
    _pad:   vec3<f32>,
};

struct ClipVolumeUB {
    volume_type: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    plane_normal: vec3<f32>,
    plane_dist: f32,
    box_center: vec3<f32>,
    _padB0: f32,
    box_half_extents: vec3<f32>,
    _padB1: f32,
    box_col0: vec3<f32>,
    _padB2: f32,
    box_col1: vec3<f32>,
    _padB3: f32,
    box_col2: vec3<f32>,
    _padB4: f32,
    sphere_center: vec3<f32>,
    sphere_radius: f32,
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;
@group(1) @binding(0) var<uniform> tube:        StreamtubeUniform;

fn clip_volume_test(p: vec3<f32>) -> bool {
    if clip_volume.volume_type == 0u { return true; }
    if clip_volume.volume_type == 1u {
        return dot(p, clip_volume.plane_normal) + clip_volume.plane_dist >= 0.0;
    }
    if clip_volume.volume_type == 2u {
        let d = p - clip_volume.box_center;
        let local = vec3<f32>(
            dot(d, clip_volume.box_col0),
            dot(d, clip_volume.box_col1),
            dot(d, clip_volume.box_col2),
        );
        return abs(local.x) <= clip_volume.box_half_extents.x
            && abs(local.y) <= clip_volume.box_half_extents.y
            && abs(local.z) <= clip_volume.box_half_extents.z;
    }
    let ds = p - clip_volume.sphere_center;
    return dot(ds, ds) <= clip_volume.sphere_radius * clip_volume.sphere_radius;
}

struct VertexIn {
    // Vertex layout (64-byte stride): position, normal, color, uv, tangent.
    // Only position and normal are used; the rest are stride padding.
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,  // stride pad
    @location(3) uv:       vec2<f32>,  // stride pad
    @location(4) tangent:  vec4<f32>,  // stride pad
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       world_nrm: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    // World-space positions and normals are baked into the mesh by the CPU generator.
    out.clip_pos  = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_nrm = normalize(in.normal);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Section-plane clipping.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    // Blinn-Phong shading : single directional key light.
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let n         = normalize(in.world_nrm);
    let n_dot_l   = max(dot(n, light_dir), 0.0);
    let shading   = 0.2 + 0.8 * n_dot_l;

    return vec4<f32>(tube.color.rgb * shading, tube.color.a);
}
