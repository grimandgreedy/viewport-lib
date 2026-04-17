// Polyline shader for the 3D viewport.
//
// Group 0: Camera uniform (view-projection, eye position)
//          + shadow atlas texture + comparison sampler
//          + Lights uniform
//          + ClipPlanes uniform (up to 6 half-space clipping planes)
//          + ShadowAtlas uniform (unused here, but layout must match camera_bgl).
// Group 1: Polyline uniform (line_width, scalar mapping params, default_color, has_scalar)
//          + LUT texture (256x1, Rgba8Unorm)
//          + LUT sampler
//
// Vertex input: position vec3 (location 0), scalar f32 (location 1).
//
// Uses hardware LineStrip topology — each draw call covers one streamline strip.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

// Clip planes uniform — must match mesh.wgsl group 0 binding 4.
struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:  u32,
    _pad0:  u32,
    viewport_width:  f32,
    viewport_height: f32,
};

// Polyline per-item uniform — 48 bytes.
struct PolylineUniform {
    default_color: vec4<f32>,  // 16 bytes
    line_width:    f32,        //  4 bytes (hardware line width; may be clamped to 1 by drivers)
    scalar_min:    f32,        //  4 bytes
    scalar_max:    f32,        //  4 bytes
    has_scalar:    u32,        //  4 bytes (1 = use per-vertex scalar + LUT)
    _pad:          vec4<f32>,  // 16 bytes padding to 48 bytes
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

@group(0) @binding(0) var<uniform> camera:     Camera;
// Bindings 1-5 of group 0 are shadow/light uniforms present in the layout but unused here.
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

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

@group(1) @binding(0) var<uniform> pl_uniform:  PolylineUniform;
@group(1) @binding(1) var          lut_texture: texture_2d<f32>;
@group(1) @binding(2) var          lut_sampler: sampler;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) scalar:   f32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       color:     vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let world_pos = in.position;
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;

    if pl_uniform.has_scalar != 0u {
        let range = pl_uniform.scalar_max - pl_uniform.scalar_min;
        let t = select(0.0, (in.scalar - pl_uniform.scalar_min) / range, range > 0.0);
        let u = clamp(t, 0.0, 1.0);
        out.color = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(u, 0.5), 0.0);
    } else {
        out.color = pl_uniform.default_color;
    }

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Clip-plane culling (section views).
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }
    return in.color;
}
