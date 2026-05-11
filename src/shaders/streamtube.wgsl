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
    color:            vec4<f32>,
    radius:           f32,
    use_vertex_color: u32,
    _pad:             vec2<f32>,
};

struct ClipVolumeEntry {
    volume_type: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    center: vec3<f32>,
    radius: f32,
    half_extents: vec3<f32>,
    _pad1: f32,
    col0: vec3<f32>,
    _pad2: f32,
    col1: vec3<f32>,
    _pad3: f32,
    col2: vec3<f32>,
    _pad4: f32,
}

struct ClipVolumeUB {
    count: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    volumes: array<ClipVolumeEntry, 4>,
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;
@group(1) @binding(0) var<uniform> tube:        StreamtubeUniform;

fn clip_volume_test(p: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_volume.count; i = i + 1u) {
        let e = clip_volume.volumes[i];
        if e.volume_type == 2u {
            let d = p - e.center;
            let local = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
            if abs(local.x) > e.half_extents.x
                || abs(local.y) > e.half_extents.y
                || abs(local.z) > e.half_extents.z {
                return false;
            }
        } else if e.volume_type == 3u {
            let ds = p - e.center;
            if dot(ds, ds) > e.radius * e.radius { return false; }
        } else if e.volume_type == 4u {
            let axis = e.col0;
            let d = p - e.center;
            let along = dot(d, axis);
            if abs(along) > e.half_extents.x { return false; }
            let radial = d - axis * along;
            if dot(radial, radial) > e.radius * e.radius { return false; }
        }
    }
    return true;
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
    @location(2)       vert_col:  vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    // World-space positions and normals are baked into the mesh by the CPU generator.
    out.clip_pos  = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_nrm = normalize(in.normal);
    out.vert_col  = in.color;
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

    // Use per-vertex color when the flag is set (TubeItem), else use the uniform color.
    let base_color = select(tube.color, in.vert_col, tube.use_vertex_color != 0u);
    return vec4<f32>(base_color.rgb * shading, base_color.a);
}
