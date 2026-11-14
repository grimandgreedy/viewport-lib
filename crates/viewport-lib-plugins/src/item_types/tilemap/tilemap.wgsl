// Tilemap colour shader: unlit atlas tiles, alpha-tested for a cutout layer
// and alpha-blended for a painter layer, lit by the 2D light layer of their
// painter range (layer 0 for a cutout layer). The vertex stage and the
// bindings are in tilemap_common.wgsl.

@vertex
fn vs_main(in: TileIn) -> TileOut {
    return tile_vertex(in);
}

// The painter masks, read by blended layers, and the 2D light layers.
@group(3) @binding(0) var painter_mask_tex: texture_2d<f32>;
@group(3) @binding(1) var<uniform> painter_masks: PainterMasks;
@group(3) @binding(2) var painter_light_tex: texture_2d_array<f32>;
@group(3) @binding(3) var<uniform> painter_lights: PainterLights;

// `colour` lit by the light layer of the layer's painter range under the
// fragment; a layer no range holds keeps its colour. With a normal atlas and
// a range that carries light directions, shaded by the tile's normal.
fn tile_light(colour: vec4<f32>, in: TileOut) -> vec4<f32> {
    let slice = painter_light_slice(painter_lights, layer.painter_layer.x);
    if slice < 0 {
        return colour;
    }
    let texel = painter_mask_texel(
        in.mask_clip.xy / in.mask_clip.w,
        textureDimensions(painter_light_tex),
    );
    let light = textureLoad(painter_light_tex, texel, slice, 0);
    let dir_slice = painter_light_dir_slice(painter_lights, layer.painter_layer.x);
    if dir_slice < 0 || tileset.has_normal_atlas == 0u {
        return vec4<f32>(painter_light_apply(colour.rgb, light, painter_lights), colour.a);
    }
    let dir = textureLoad(painter_light_tex, texel, dir_slice, 0);
    let n = tile_normal(in);
    return vec4<f32>(painter_light_apply_normal(colour.rgb, light, dir, n, painter_lights), colour.a);
}

fn cutout_body(in: TileOut) -> vec4<f32> {
    let colour = tile_colour(in);
    if !viewport_clip_test(in.world_pos) {
        discard;
    }
    if colour.a < layer.alpha_threshold {
        discard;
    }
    return vec4<f32>(tile_light(colour, in).rgb, 1.0);
}

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
    return tile_light(colour, in);
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

// The selection-outline mask: every texel of the layer a pick would see.
@fragment
fn fs_mask(in: TileOut) -> @location(0) vec4<f32> {
    let colour = tile_colour(in);
    if !viewport_clip_test(in.world_pos) {
        discard;
    }
    if colour.a < max(layer.alpha_threshold, 0.001) {
        discard;
    }
    return vec4<f32>(1.0);
}
