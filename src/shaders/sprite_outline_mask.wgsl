// sprite_outline_mask.wgsl : renders selected sprite billboards as solid
// quads into the R8 outline mask texture.  Uses the same bind group layout
// and vertex transform as sprite.wgsl but outputs a flat mask value instead
// of textured/colored fragments.
//
// Group 0: Camera (view_proj, view matrix for cam_right/cam_up, viewport dims).
// Group 1: SpriteUniform + texture + sampler + instance storage buffer
//          (texture/sampler unused, but layout must match).

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

struct SpriteUniform {
    model:       mat4x4<f32>,
    world_space: u32,
    has_texture: u32,
    _pad0:       u32,
    _pad1:       u32,
};

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

@group(1) @binding(0) var<uniform>       sprite_ub:     SpriteUniform;
@group(1) @binding(1) var               sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var               sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:  array<SpriteInstance>;

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
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
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
    let inst = instance_buf[in.instance_index];

    let world_pos = (sprite_ub.model * vec4<f32>(in.position, 1.0)).xyz;
    let corner    = quad_corner(in.vertex_index);

    let c = cos(inst.rotation);
    let s = sin(inst.rotation);
    let rotated = vec2<f32>(
        c * corner.x - s * corner.y,
        s * corner.x + c * corner.y,
    );

    if sprite_ub.world_space != 0u {
        let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
        let cam_up    = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
        let half      = inst.size * 0.5;
        let ws_pos    = world_pos
                      + cam_right * (rotated.x * half)
                      + cam_up    * (rotated.y * half);
        return camera.view_proj * vec4<f32>(ws_pos, 1.0);
    } else {
        let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let half_px   = inst.size * 0.5;
        let ndc_off   = rotated * half_px
                      / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        return vec4<f32>(
            center.x + ndc_off.x * center.w,
            center.y + ndc_off.y * center.w,
            center.z,
            center.w,
        );
    }
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
