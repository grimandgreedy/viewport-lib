// Sprite shader: textured billboards with per-instance color, size, rotation, and UV rect.
//
// Group 0: Camera uniform (binding 0), ClipPlanes (binding 4), ClipVolume (binding 6).
// Group 1: SpriteUniform (binding 0), sprite texture (binding 1), sampler (binding 2),
//           per-instance storage buffer (binding 3).
//
// Each sprite is rendered as 6 vertices (2 CCW triangles). The position vertex buffer
// uses Instance stepping (one vec3 per sprite). All other per-instance data comes from
// the instance storage buffer at index `instance_index`.
//
// Size modes:
//   world_space == 0 : sizes are in screen-space pixels (same as point_cloud.wgsl).
//   world_space != 0 : sizes are in world-space units; quads expand along camera right/up.

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

struct ClipVolumeUB {
    volume_type:      u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    plane_normal:     vec3<f32>,
    plane_dist:       f32,
    box_center:       vec3<f32>,
    _padB0:           f32,
    box_half_extents: vec3<f32>,
    _padB1:           f32,
    box_col0:         vec3<f32>,
    _padB2:           f32,
    box_col1:         vec3<f32>,
    _padB3:           f32,
    box_col2:         vec3<f32>,
    _padB4:           f32,
    sphere_center:    vec3<f32>,
    sphere_radius:    f32,
};

// Per-batch uniform (80 bytes):
//   model:       mat4x4<f32>  (64 bytes at offset  0)
//   world_space: u32          ( 4 bytes at offset 64) -- 0 = screen pixels, 1 = world units
//   has_texture: u32          ( 4 bytes at offset 68) -- 0 = solid color, 1 = sample texture
//   _pad0/1:     u32, u32     ( 8 bytes at offset 72)
struct SpriteUniform {
    model:       mat4x4<f32>,
    world_space: u32,
    has_texture: u32,
    _pad0:       u32,
    _pad1:       u32,
};

// Per-sprite instance data (48 bytes):
//   color:    vec4<f32>  (16 bytes at offset  0)
//   size:     f32        ( 4 bytes at offset 16)
//   rotation: f32        ( 4 bytes at offset 20) -- radians, CCW around camera-forward axis
//   _pad0/1:  f32, f32   ( 8 bytes at offset 24) -- alignment before uv_rect
//   uv_rect:  vec4<f32>  (16 bytes at offset 32) -- [u0, v0, u1, v1]
struct SpriteInstance {
    color:    vec4<f32>,
    size:     f32,
    rotation: f32,
    _pad0:    f32,
    _pad1:    f32,
    uv_rect:  vec4<f32>,
};

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(4) var<uniform>       clip_planes:   ClipPlanes;
@group(0) @binding(6) var<uniform>       clip_volume:   ClipVolumeUB;

@group(1) @binding(0) var<uniform>       sprite_ub:     SpriteUniform;
@group(1) @binding(1) var               sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var               sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:  array<SpriteInstance>;

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
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       color:     vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
    @location(2)       uv:        vec2<f32>,
};

// Unit quad corners (two CCW triangles, matching point_cloud.wgsl winding).
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
    let inst = instance_buf[in.instance_index];

    let world_pos = (sprite_ub.model * vec4<f32>(in.position, 1.0)).xyz;
    let corner    = quad_corner(in.vertex_index);

    // Apply per-instance rotation around the camera-forward axis.
    let c = cos(inst.rotation);
    let s = sin(inst.rotation);
    let rotated = vec2<f32>(
        c * corner.x - s * corner.y,
        s * corner.x + c * corner.y,
    );

    if sprite_ub.world_space != 0u {
        // World-space sizing: expand along camera right/up before projection.
        // The view matrix rows give the camera axes in world space.
        // view is column-major in WGSL: view[col][row].
        // Row 0 of view = camera right in world space.
        // Row 1 of view = camera up in world space.
        let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
        let cam_up    = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
        let half      = inst.size * 0.5;
        let ws_pos    = world_pos
                      + cam_right * (rotated.x * half)
                      + cam_up    * (rotated.y * half);
        out.clip_pos  = camera.view_proj * vec4<f32>(ws_pos, 1.0);
    } else {
        // Screen-space sizing: expand in NDC after projection (same as point_cloud.wgsl).
        let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let half_px   = inst.size * 0.5;
        let ndc_off   = rotated * half_px
                      / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        out.clip_pos  = vec4<f32>(
            center.x + ndc_off.x * center.w,
            center.y + ndc_off.y * center.w,
            center.z,
            center.w,
        );
    }

    out.world_pos = world_pos;
    out.color     = inst.color;

    // Map corner [-1, 1] to the per-instance UV rect [u0, v0] -> [u1, v1].
    let u  = mix(inst.uv_rect.x, inst.uv_rect.z, (corner.x + 1.0) * 0.5);
    let v  = mix(inst.uv_rect.y, inst.uv_rect.w, (corner.y + 1.0) * 0.5);
    out.uv = vec2<f32>(u, v);

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Section-view clip planes.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    var color = in.color;
    if sprite_ub.has_texture != 0u {
        color = color * textureSample(sprite_texture, sprite_sampler, in.uv);
    }
    // Discard fully transparent fragments.
    if color.a <= 0.001 { discard; }
    return color;
}
