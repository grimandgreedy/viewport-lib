// Tilemap GPU pick: the colour shader's vertex stage, writing the layer's
// pick id and the cell under the cursor (column low 16 bits, row high 16)
// where the tile is opaque enough to be seen.

struct PickOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        u32,
};

@vertex
fn vs_main(in: TileIn) -> TileOut {
    return tile_vertex(in);
}

@fragment
fn fs_main(in: TileOut) -> PickOut {
    let colour = tile_colour(in);
    if !viewport_clip_test(in.world_pos) {
        discard;
    }
    if colour.a < max(layer.alpha_threshold, 0.001) {
        discard;
    }
    var out: PickOut;
    out.object_id = layer.pick_id;
    out.primitive_id = in.cell;
    out.depth = bitcast<u32>(in.clip_pos.z);
    return out;
}
