// Sprite shader: textured billboards with per-instance colour, size, rotation, and UV rect.
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

// Per-batch uniform (112 bytes):
//   model:                  mat4x4<f32>  (64 bytes at offset  0)
//   world_space:            u32          ( 4 bytes at offset 64)
//   has_texture:            u32          ( 4 bytes at offset 68)
//   soft_particle_distance: f32          ( 4 bytes at offset 72)
//   orientation:            u32          ( 4 bytes at offset 76) -- 0=CameraFacing, 1=VelocityStretched, 2=AxisLocked
//   axis:                   vec3<f32>    (12 bytes at offset 80) -- AxisLocked direction
//   refraction_strength:    f32          ( 4 bytes at offset 92) -- 0 disables refractive draw
struct SpriteUniform {
    model:                  mat4x4<f32>,
    world_space:            u32,
    has_texture:            u32,
    soft_particle_distance: f32,
    orientation:            u32,
    axis:                   vec3<f32>,
    refraction_strength:    f32,
};

// Per-sprite instance data (64 bytes):
//   colour:         vec4<f32>  (16 bytes at offset  0)
//   size:           f32        ( 4 bytes at offset 16)
//   rotation:       f32        ( 4 bytes at offset 20)
//   soft_distance:  f32        ( 4 bytes at offset 24) -- per-instance soft-fade distance; 0 falls back to the batch default
//   _pad1:          f32        ( 4 bytes at offset 28)
//   uv_rect:        vec4<f32>  (16 bytes at offset 32)
//   velocity:       vec3<f32>  (12 bytes at offset 48) -- VelocityStretched direction; zero disables stretch
//   _pad2:          f32        ( 4 bytes at offset 60)
struct SpriteInstance {
    colour:        vec4<f32>,
    size:          f32,
    rotation:      f32,
    soft_distance: f32,
    _pad1:         f32,
    uv_rect:       vec4<f32>,
    velocity:      vec3<f32>,
    _pad2:         f32,
};

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(4) var<uniform>       clip_planes:   ClipPlanes;
@group(0) @binding(6) var<uniform>       clip_volume:   ClipVolumeUB;

@group(1) @binding(0) var<uniform>       sprite_ub:     SpriteUniform;
@group(1) @binding(1) var               sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var               sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:  array<SpriteInstance>;

// Group 2: scene depth resolve used for soft-particle fade.
// Bound to the scene's depth-aspect view when soft particles are active;
// otherwise bound to a placeholder. The shader only samples this when
// soft_particle_distance > 0, so the placeholder contents do not matter.
@group(2) @binding(0) var scene_depth_tex: texture_depth_2d;
@group(2) @binding(1) var scene_depth_samp: sampler;

// #include "helpers/clip_volume_test.wgsl"

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:      vec4<f32>,
    @location(0)       colour:        vec4<f32>,
    @location(1)       world_pos:     vec3<f32>,
    @location(2)       uv:            vec2<f32>,
    @location(3) @interpolate(flat) soft_distance: f32,
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

    // Camera basis vectors in world space (rows of the view matrix).
    let cam_right_default = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up_default    = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let cam_forward       = vec3<f32>(camera.view[0][2], camera.view[1][2], camera.view[2][2]);

    // Pick the quad's local right and up axes based on the orientation mode.
    var local_right = cam_right_default;
    var local_up    = cam_up_default;
    var stretch_x   = 1.0;

    if sprite_ub.orientation == 1u {
        // VelocityStretched: align local right with the projected velocity.
        let v = inst.velocity;
        let speed = length(v);
        if speed > 1e-4 {
            // Project velocity onto the plane perpendicular to the camera forward.
            let v_screen = v - cam_forward * dot(v, cam_forward);
            let s_len    = length(v_screen);
            if s_len > 1e-4 {
                local_right = v_screen / s_len;
                local_up    = normalize(cross(cam_forward, local_right));
                stretch_x   = 1.0 + speed * 0.25;
            }
        }
    } else if sprite_ub.orientation == 2u {
        // AxisLocked: long axis follows the supplied world-space direction;
        // local right sits in the plane perpendicular to both the axis and
        // the camera forward so the card stays facing toward the camera.
        let axis = normalize(sprite_ub.axis);
        local_up = axis;
        let right = cross(axis, cam_forward);
        let r_len = length(right);
        if r_len > 1e-4 {
            local_right = right / r_len;
        } else {
            // Camera looking straight along the axis: fall back to a stable basis.
            local_right = cam_right_default;
        }
    }

    if sprite_ub.world_space != 0u {
        let half = inst.size * 0.5;
        let ws_pos = world_pos
                   + local_right * (rotated.x * half * stretch_x)
                   + local_up    * (rotated.y * half);
        out.clip_pos = camera.view_proj * vec4<f32>(ws_pos, 1.0);
    } else {
        // Screen-space sizing: project the world axes to NDC so the screen
        // extent uses the chosen orientation rather than the camera basis.
        let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let right_clip = camera.view_proj * vec4<f32>(local_right, 0.0);
        let up_clip    = camera.view_proj * vec4<f32>(local_up,    0.0);
        let half_px    = inst.size * 0.5;
        let inv_vp     = vec2<f32>(1.0, 1.0)
                       / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        let offset_clip = right_clip * (rotated.x * half_px * stretch_x * inv_vp.x)
                        + up_clip    * (rotated.y * half_px * inv_vp.y);
        if sprite_ub.orientation == 0u {
            // Camera-facing falls back to the original screen-space NDC offset
            // so the size matches the prior behaviour exactly.
            let ndc_off = rotated * half_px * inv_vp;
            out.clip_pos = vec4<f32>(
                center.x + ndc_off.x * center.w,
                center.y + ndc_off.y * center.w,
                center.z,
                center.w,
            );
        } else {
            out.clip_pos = center + offset_clip * center.w;
        }
    }

    out.world_pos = world_pos;
    out.colour     = inst.colour;
    out.soft_distance = inst.soft_distance;

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

    var colour = in.colour;
    if sprite_ub.has_texture != 0u {
        colour = colour * textureSample(sprite_texture, sprite_sampler, in.uv);
    }

    // Soft-particle fade: when the sprite fragment approaches opaque scene
    // geometry behind it, ramp alpha to zero over a configurable world-space
    // distance. Per-instance overrides the batch default; a non-positive
    // per-instance value falls back to the batch value. Disabled when both
    // are non-positive.
    var soft_dist = sprite_ub.soft_particle_distance;
    if in.soft_distance > 0.0 {
        soft_dist = in.soft_distance;
    }
    if soft_dist > 0.0 {
        let viewport_size = vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        let screen_uv     = in.clip_pos.xy / viewport_size;
        let scene_ndc_z   = textureSample(scene_depth_tex, scene_depth_samp, screen_uv);
        let ndc = vec4<f32>(
            screen_uv.x * 2.0 - 1.0,
            1.0 - screen_uv.y * 2.0,
            scene_ndc_z,
            1.0,
        );
        let world_h     = camera.inv_view_proj * ndc;
        let scene_world = world_h.xyz / world_h.w;
        let scene_view_z = -(camera.view * vec4<f32>(scene_world, 1.0)).z;
        let part_view_z  = -(camera.view * vec4<f32>(in.world_pos, 1.0)).z;
        let fade = smoothstep(0.0, soft_dist, scene_view_z - part_view_z);
        colour.a = colour.a * fade;
    }

    // Discard fully transparent fragments.
    if colour.a <= 0.001 { discard; }
    return colour;
}
