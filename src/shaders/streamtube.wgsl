// Streamtube (cylinder-instanced) shader for the 3D viewport.
//
// Renders polyline paths as 3D tubes with configurable radius.
//
// Group 0: Camera uniform (view-projection, eye position) — same as glyph.wgsl.
//          + ClipPlanes uniform (binding 4).
// Group 1: StreamtubeUniform — color (vec4) + radius (f32).
// Group 2: Per-instance storage buffer
//          (StreamtubeInstance: position vec3, half_len f32, direction vec3, _pad f32).
//
// Vertex input: 8-sided cylinder mesh (local Y from -1 to +1, XZ radius = 1.0).
//
// Each instance corresponds to one consecutive segment of a polyline strip.
// The cylinder local +Y axis is aligned to the segment direction vector.
// Scale applied: (radius, half_len, radius) in (X, Y, Z).

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

// 32-byte aligned uniform (color 16 + radius 4 + pad 12).
struct StreamtubeUniform {
    color:  vec4<f32>,   // 16 bytes
    radius: f32,         //  4 bytes
    _pad:   vec3<f32>,   // 12 bytes
};

// 32 bytes per instance.
struct StreamtubeInstance {
    position:  vec3<f32>,  // segment midpoint — 12 bytes
    half_len:  f32,        // half segment length —  4 bytes
    direction: vec3<f32>,  // normalized direction — 12 bytes
    _pad:      f32,        //  4 bytes
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

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(4) var<uniform>       clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform>       clip_volume: ClipVolumeUB;

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

@group(1) @binding(0) var<uniform>       tube:        StreamtubeUniform;

@group(2) @binding(0) var<storage, read> instances:   array<StreamtubeInstance>;

struct VertexIn {
    // Full Vertex layout (64-byte stride); only position + normal used here.
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,  // unused — stride padding
    @location(3) uv:       vec2<f32>,  // unused — stride padding
    @location(4) tangent:  vec4<f32>,  // unused — stride padding
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       color:     vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
    @location(2)       world_nrm: vec3<f32>,
};

// Build a rotation matrix that rotates local +Y to align with `dir`.
// Identical to glyph.wgsl for consistency.
fn rotation_to_align_y(dir: vec3<f32>) -> mat3x3<f32> {
    let up = normalize(dir);
    var ref_v: vec3<f32>;
    if abs(up.y) < 0.99 {
        ref_v = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        ref_v = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(ref_v, up));
    let fwd   = cross(up, right);
    return mat3x3<f32>(right, up, fwd);
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let inst = instances[in.instance_index];

    // Build orientation matrix (local +Y -> segment direction).
    var rot = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    if length(inst.direction) > 0.0001 {
        rot = rotation_to_align_y(normalize(inst.direction));
    }

    // Non-uniform scale: tube cross-section (radius) × tube length (half_len).
    let scaled_pos = vec3<f32>(
        in.position.x * tube.radius,
        in.position.y * inst.half_len,
        in.position.z * tube.radius,
    );
    let world_pos = rot * scaled_pos + inst.position;

    // Normal transformed by rotation only (ignores non-uniform scale shear,
    // acceptable for tubes where radius << segment length in most use cases).
    let world_nrm = normalize(rot * in.normal);

    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_nrm = world_nrm;
    out.color     = tube.color;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Clip-plane culling.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    // Blinn-Phong lighting — single directional light (same as glyph.wgsl).
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let n_dot_l   = max(dot(in.world_nrm, light_dir), 0.0);
    let ambient   = 0.2;
    let diffuse   = 0.8 * n_dot_l;
    let shading   = ambient + diffuse;

    return vec4<f32>(in.color.rgb * shading, in.color.a);
}
