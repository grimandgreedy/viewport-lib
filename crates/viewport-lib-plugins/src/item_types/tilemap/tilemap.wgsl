// Tilemap colour shader: unlit atlas tiles, alpha-tested for a cutout layer
// and alpha-blended for a painter layer. The vertex stage and the bindings
// are in tilemap_common.wgsl.

@vertex
fn vs_main(in: TileIn) -> TileOut {
    return tile_vertex(in);
}

fn cutout_body(in: TileOut) -> vec4<f32> {
    let colour = tile_colour(in);
    if !viewport_clip_test(in.world_pos) {
        discard;
    }
    if colour.a < layer.alpha_threshold {
        discard;
    }
    return vec4<f32>(colour.rgb, 1.0);
}

// The painter masks, read by blended layers only (the cutout entries leave
// group 3 out of their layout).
@group(3) @binding(0) var painter_mask_tex: texture_2d<f32>;
@group(3) @binding(1) var<uniform> painter_masks: PainterMasks;

fn blended_body(in: TileOut) -> vec4<f32> {
    var colour = tile_colour(in);
    if !viewport_clip_test(in.world_pos) {
        discard;
    }
    let texel = textureLoad(
        painter_mask_tex,
        painter_mask_texel(in.mask_clip.xy / in.mask_clip.w, textureDimensions(painter_mask_tex)),
        0,
    );
    colour.a = colour.a * painter_mask_keep(texel, painter_masks, layer.mask, layer.painter_layer.x);
    if colour.a <= 0.001 {
        discard;
    }
    return colour;
}

@fragment
fn fs_cutout(in: TileOut) -> @location(0) vec4<f32> {
    return cutout_body(in);
}

// `fs_cutout` for the HDR scene pass, with the indirect fraction.
@fragment
fn fs_cutout_scene(in: TileOut) -> SceneFragment {
    return scene_fragment(cutout_body(in));
}

@fragment
fn fs_blended(in: TileOut) -> @location(0) vec4<f32> {
    return blended_body(in);
}
