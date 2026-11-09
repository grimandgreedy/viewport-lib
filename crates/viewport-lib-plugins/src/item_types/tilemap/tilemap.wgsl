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

fn blended_body(in: TileOut) -> vec4<f32> {
    let colour = tile_colour(in);
    if !viewport_clip_test(in.world_pos) {
        discard;
    }
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
