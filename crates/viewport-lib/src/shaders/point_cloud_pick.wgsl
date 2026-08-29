// GPU object-ID pick shader for point clouds.
//
// The vertex stage mirrors point_cloud.wgsl's vs_main: it reads the same
// per-instance position buffer and PointCloudUniform (group 1), and expands
// the same screen-space quad (using the per-point radius buffer when the item
// has one), so the pick silhouette matches the rendered circle exactly. The
// fragment stage clips to the unit circle the same way the render fragment
// does, then writes the item's object id (group 2) plus the point's instance
// index (for CLOUD_POINT sub-object picking) and clip-space depth.
//
// Group 0: full camera bind group (binding 0 camera, binding 4 clip planes,
//          binding 6 clip volume) : the screen-space quad expansion needs the
//          viewport size carried in clip_planes.
// Group 1: PointCloudUniform (binding 0) + radius buffer (binding 5), reused
//          unchanged from the render bind group. Bindings 1-4 and 6 (LUT,
//          scalar, colour, transparency) are present but unused here.
// Group 2: per-item pick id (binding 0).

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct ClipPlanes {
    planes:          array<vec4<f32>, 6>,
    count:           u32,
    _pad0:           u32,
    viewport_width:  f32,
    viewport_height: f32,
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

// Trailing fields after render_mode (a u32 _pad[3] in the Rust struct) are not
// declared: this struct is a prefix of the real 128-byte buffer, and WGSL only
// requires the declared prefix to fit within the bound buffer's size.
struct PointCloudUniform {
    model:            mat4x4<f32>,
    default_colour:   vec4<f32>,
    point_size:       f32,
    has_scalars:      u32,
    scalar_min:       f32,
    scalar_max:       f32,
    has_colours:      u32,
    has_radius:       u32,
    has_transparency: u32,
    gaussian:         u32,
    render_mode:      u32,
};

// Per-draw pick id (one point cloud item = one draw of all its points).
struct PickId {
    object_id: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

@group(0) @binding(0) var<uniform>      camera:      Camera;
@group(0) @binding(4) var<uniform>      clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform>      clip_volume: ClipVolumeUB;

@group(1) @binding(0) var<uniform>      pc_uniform:    PointCloudUniform;
@group(1) @binding(5) var<storage, read> radius_buffer: array<f32>;

@group(2) @binding(0) var<uniform>      pick: PickId;

// #include "helpers/clip_volume_test.wgsl"

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)                    world_pos: vec3<f32>,
    @location(1)                    uv:        vec2<f32>,
    // The instance index, forwarded flat so the fragment can write it into the
    // primitive-id channel for CLOUD_POINT sub-object picking.
    @location(2) @interpolate(flat) instance_index: u32,
};

fn quad_corner(vi: u32) -> vec2<f32> {
    switch vi {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0,  1.0); }
        case 4u: { return vec2<f32>( 1.0, -1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let world_pos = (pc_uniform.model * vec4<f32>(in.position, 1.0)).xyz;
    let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);

    let idx = in.instance_index;
    let point_size = select(
        pc_uniform.point_size,
        radius_buffer[idx],
        pc_uniform.has_radius != 0u,
    );
    let half_size  = point_size * 0.5;
    let corner     = quad_corner(in.vertex_index);
    let ndc_offset = corner * half_size
                     / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);

    out.clip_pos = vec4<f32>(
        center.x + ndc_offset.x * center.w,
        center.y + ndc_offset.y * center.w,
        center.z,
        center.w,
    );
    out.world_pos = world_pos;
    out.uv = corner;
    out.instance_index = in.instance_index;
    return out;
}

struct FragOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    // Clip to the circle, matching the render fragment (uv is [-1,1]^2).
    if dot(in.uv, in.uv) > 1.0 { discard; }

    var out: FragOut;
    out.object_id    = pick.object_id;
    out.primitive_id = in.instance_index;
    out.depth        = in.clip_pos.z;
    return out;
}
