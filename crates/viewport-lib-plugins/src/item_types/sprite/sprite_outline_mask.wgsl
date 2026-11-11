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
    return sprite_corner(in.position, quad_corner(in.vertex_index), inst).clip;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
