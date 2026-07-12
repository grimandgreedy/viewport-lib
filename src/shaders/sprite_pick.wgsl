// GPU object-ID pick shader for sprite billboards.
//
// The vertex stage is a copy of sprite.wgsl's vs_main: it reads the same
// camera bind group (group 0), the same SpriteUniform + per-instance storage
// buffer (group 1), and expands the same camera-facing quad. Keeping the
// expansion identical means the pick silhouette matches the rendered sprite
// exactly and tracks any future change to the sprite billboard math.
//
// The fragment stage writes the item's object id (supplied per draw through a
// small uniform at group 2) into the R32Uint id target and clip-space depth
// into the R32Float depth target. The pick_id is per item (one sprite item =
// one draw of all its instances), so it does not vary per instance.

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

struct SpriteUniform {
    model:                  mat4x4<f32>,
    world_space:            u32,
    has_texture:            u32,
    soft_particle_distance: f32,
    orientation:            u32,
    axis:                   vec3<f32>,
    refraction_strength:    f32,
};

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

// Per-draw pick id (one sprite item = one draw of all its instances).
struct PickId {
    object_id: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(4) var<uniform>       clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform>       clip_volume: ClipVolumeUB;

@group(1) @binding(0) var<uniform>       sprite_ub:   SpriteUniform;
// Bindings 1 (texture) and 2 (sampler) are present in the sprite bind group
// but unused here; the pick fragment does not sample the texture.
@group(1) @binding(3) var<storage, read> instance_buf: array<SpriteInstance>;

@group(2) @binding(0) var<uniform>       pick: PickId;

// #include "clip_volume_test.wgsl"

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

// Unit quad corners (two CCW triangles, matching sprite.wgsl winding).
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
            let v_screen = v - cam_forward * dot(v, cam_forward);
            let s_len    = length(v_screen);
            if s_len > 1e-4 {
                local_right = v_screen / s_len;
                local_up    = normalize(cross(cam_forward, local_right));
                stretch_x   = 1.0 + speed * 0.25;
            }
        }
    } else if sprite_ub.orientation == 2u {
        // AxisLocked: long axis follows the supplied world-space direction.
        let axis = normalize(sprite_ub.axis);
        local_up = axis;
        let right = cross(axis, cam_forward);
        let r_len = length(right);
        if r_len > 1e-4 {
            local_right = right / r_len;
        } else {
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
        let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let right_clip = camera.view_proj * vec4<f32>(local_right, 0.0);
        let up_clip    = camera.view_proj * vec4<f32>(local_up,    0.0);
        let half_px    = inst.size * 0.5;
        let inv_vp     = vec2<f32>(1.0, 1.0)
                       / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        let offset_clip = right_clip * (rotated.x * half_px * stretch_x * inv_vp.x)
                        + up_clip    * (rotated.y * half_px * inv_vp.y);
        if sprite_ub.orientation == 0u {
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
    return out;
}

struct FragOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    // Section-view clip planes, matching the sprite render fragment.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    var out: FragOut;
    out.object_id = pick.object_id;
    out.primitive_id = 0u;
    out.depth = in.clip_pos.z;
    return out;
}
