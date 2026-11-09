// sprite_outline_mask.wgsl : renders selected sprite billboards as solid
// quads into the R8 outline mask texture.  Uses the same bind group layout
// and vertex transform as sprite.wgsl but outputs a flat mask value instead
// of textured/coloured fragments.
//
// Group 0: Camera (view_proj, view matrix for cam_right/cam_up, viewport dims).
// Group 1: SpriteUniform + texture + sampler + instance storage buffer
//          (texture/sampler unused, but layout must match).

@group(1) @binding(0) var<uniform>       sprite_ub:     SpriteUniform;
@group(1) @binding(1) var               sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var               sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:  array<SpriteInstance>;

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

@vertex
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
    let inst = instance_buf[in.instance_index];

    let world_pos = (sprite_ub.model * vec4<f32>(in.position, 1.0)).xyz;
    let corner    = quad_corner(in.vertex_index);

    let offset = sprite_offset(corner, inst);

    var clip: vec4<f32>;
    if sprite_ub.world_space != 0u {
        let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
        let cam_up    = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
        let ws_pos    = world_pos
                      + cam_right * offset.x
                      + cam_up    * offset.y;
        clip = camera.view_proj * vec4<f32>(ws_pos, 1.0);
    } else {
        let centre    = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let ndc_off   = offset
                      / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        clip = vec4<f32>(
            centre.x + ndc_off.x * centre.w,
            centre.y + ndc_off.y * centre.w,
            centre.z,
            centre.w,
        );
    }
    if (sprite_ub.flags & 1u) != 0u {
        clip = sprite_pixel_snap(
            clip,
            camera.view_proj * vec4<f32>(world_pos, 1.0),
            vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height),
        );
    }
    return clip;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
