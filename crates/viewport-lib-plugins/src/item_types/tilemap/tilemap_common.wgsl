// Declarations and the vertex stage every tilemap shader shares, spliced in
// front of each body. The structs must match `TilesetUniformData` and
// `LayerUniformData` in `store.rs` exactly.
//
// Group 1: the uploaded map's tileset uniform, atlas, sampler, normal atlas
//          and the per-tile user values a tile material's vertex hook reads.
// Group 2: the drawn layer's uniform, at a dynamic offset.
//
// One instance per occupied cell, six vertices each (two triangles):
//   location 0: cell, column in the low 16 bits and row in the high 16, each
//               two's complement
//   location 1: tileset tile index, or animation slot when flag bit 3 is set
//   location 2: flags, bit 0 horizontal flip, 1 vertical, 2 diagonal, 3 animated,
//               4 a 120 degree turn on a hexagonal grid, bits 8-15 the size
//               slot (0 for one cell)
//   location 3: tint, unorm8x4 linear RGBA

struct TilesetUniform {
    tile_size:   vec2<f32>,  // texels
    columns:     u32,
    has_texture: u32,
    spacing:     f32,
    margin:      f32,
    // Bits 0-1 the grid: 0 orthogonal, 1 isometric, 2 columns staggered,
    // 3 rows staggered. Bit 2: the even columns or rows are shifted. Bit 3:
    // hexagonal, where the diagonal flag and flag bit 4 turn the tile.
    orientation: u32,
    // 1 when the normal atlas is the tileset's own, not the flat fallback.
    has_normal_atlas: u32,
    // A staggered grid's column step, row step and shift, in cells.
    grid:        vec4<f32>,
    // Per size slot: the image size in texels (xy) and where the quad's
    // bottom-left corner sits in cells from the cell's (zw). Slot 0 is unused.
    sizes:       array<vec4<f32>, 256>,
};

struct LayerUniform {
    // Plane position of cell (0, 0) relative to the world origin, with the
    // world origin's share of the parallax shift folded in.
    translation:     vec2<f32>,
    cell_size:       vec2<f32>,
    // 1 - parallax: how much of the camera's plane position the layer follows.
    follow:          vec2<f32>,
    // Elevation relative to the world origin.
    elevation:       f32,
    plane:           u32,        // 0 XY, 1 XZ
    tint:            vec4<f32>,
    alpha_threshold: f32,
    flags:           u32,        // bit 0 pixel snap
    pick_id:         u32,
    // The painter mask test (`MaskTest::word`), read by a blended layer.
    mask:            u32,
    // x: the blended layer's painter layer. y: the sprite deformers the layer
    // runs, one bit each.
    painter_layer:   vec4<u32>,
    // The tile each animation slot shows this frame, four slots per entry.
    anim:            array<vec4<u32>, 64>,
    // A tile material's parameter lanes; lane 0 is reserved.
    lanes:           array<vec4<f32>, 8>,
};

@group(1) @binding(0) var<uniform> tileset:       TilesetUniform;
@group(1) @binding(1) var          atlas:         texture_2d<f32>;
@group(1) @binding(2) var          atlas_sampler: sampler;
@group(1) @binding(3) var          normal_atlas:  texture_2d<f32>;
// `Tileset::user_values`, row by row; 1x1 zero when the tileset sets none.
@group(1) @binding(4) var          tile_user:     texture_2d<u32>;
@group(2) @binding(0) var<uniform> layer:         LayerUniform;

struct TileIn {
    @location(0)           cell:         u32,
    @location(1)           tile:         u32,
    @location(2)           flags:        u32,
    @location(3)           tint:         vec4<f32>,
    @builtin(vertex_index) vertex_index: u32,
};

struct TileOut {
    @builtin(position)              clip_pos:  vec4<f32>,
    // Atlas position in texels, y down from the top of the image.
    @location(0)                    texel:     vec2<f32>,
    // The tile's texel rectangle: min xy, max zw.
    @location(1) @interpolate(flat) rect:      vec4<f32>,
    @location(2) @interpolate(flat) tint:      vec4<f32>,
    @location(3)                    world_pos: vec3<f32>,
    @location(4) @interpolate(flat) cell:      u32,
    // The clip position again, for the fragment's place on the painter mask.
    @location(5)                    mask_clip: vec4<f32>,
    // The instance's flip and turn flags.
    @location(6) @interpolate(flat) flags:     u32,
};

// Corner of the unit cell, 0..1 with y up the plane (two CCW triangles).
fn tile_corner(vi: u32) -> vec2<f32> {
    switch vi {
        case 0u: { return vec2<f32>(0.0, 0.0); }
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u: { return vec2<f32>(0.0, 1.0); }
        case 3u: { return vec2<f32>(0.0, 1.0); }
        case 4u: { return vec2<f32>(1.0, 0.0); }
        default: { return vec2<f32>(1.0, 1.0); }
    }
}

// The plane offset, in cells, of the bottom-left corner of the bounding box
// of `cell`; `packed` is the cell as the instance carries it, for the
// parity of its column and row.
fn tile_cell_origin(cell: vec2<f32>, packed: u32) -> vec2<f32> {
    let kind = tileset.orientation & 3u;
    if kind == 0u {
        return cell;
    }
    if kind == 1u {
        return vec2<f32>(cell.x + cell.y, cell.y - cell.x) * 0.5;
    }
    let even = (tileset.orientation >> 2u) & 1u;
    if kind == 2u {
        let shifted = (packed & 1u) != even;
        return vec2<f32>(
            cell.x * tileset.grid.x,
            cell.y * tileset.grid.y - select(0.0, tileset.grid.z, shifted),
        );
    }
    let shifted = ((packed >> 16u) & 1u) != even;
    return vec2<f32>(
        cell.x * tileset.grid.x + select(0.0, tileset.grid.z, shifted),
        cell.y * tileset.grid.y,
    );
}

// The world point (relative to the world origin) at plane position `p`.
fn tile_world(p: vec2<f32>) -> vec3<f32> {
    if layer.plane == 1u {
        return vec3<f32>(p.x, -layer.elevation, p.y);
    }
    return vec3<f32>(p.x, p.y, layer.elevation);
}

// The camera's position on the plane, relative to the world origin.
fn tile_eye_on_plane() -> vec2<f32> {
    if layer.plane == 1u {
        return camera.eye_pos.xz;
    }
    return camera.eye_pos.xy;
}

// The tileset's user value for tile `index`, zero past the table.
fn tile_user_value(index: u32) -> u32 {
    let dims = textureDimensions(tile_user, 0);
    let at = vec2<u32>(index % dims.x, index / dims.x);
    if at.y >= dims.y {
        return 0u;
    }
    return textureLoad(tile_user, at, 0).r;
}

// Shift `clip` so that `anchor` lands on a whole device pixel.
fn tile_pixel_snap(clip: vec4<f32>, anchor: vec4<f32>, viewport: vec2<f32>) -> vec4<f32> {
    if anchor.w <= 0.0 {
        return clip;
    }
    let px = (anchor.xy / anchor.w * 0.5 + 0.5) * viewport;
    let shift = (round(px) - px) / viewport * 2.0;
    return vec4<f32>(clip.xy + shift * clip.w, clip.zw);
}

fn tile_vertex(in: TileIn) -> TileOut {
    var out: TileOut;
    let corner = tile_corner(in.vertex_index);
    // Column and row, sign-extended from 16 bits.
    let cell = vec2<f32>(
        f32(bitcast<i32>(in.cell << 16u) >> 16u),
        f32(bitcast<i32>(in.cell) >> 16u),
    );

    // A tile with a size slot covers its own image size, scaled as one
    // tileset tile is to one cell, from its anchor in the cell.
    var texels = tileset.tile_size;
    var quad_min = vec2<f32>(0.0);
    var quad_size = vec2<f32>(1.0);
    let size_slot = (in.flags >> 8u) & 0xffu;
    if size_slot != 0u {
        let s = tileset.sizes[size_slot];
        texels = s.xy;
        quad_min = s.zw;
        quad_size = s.xy / max(tileset.tile_size, vec2<f32>(1.0));
    }

    let cell_origin = tile_cell_origin(cell, in.cell);
    var offset = corner * quad_size;
    let hexagonal = (tileset.orientation & 8u) != 0u;
    if hexagonal {
        // Clockwise turns of 60 degrees about the quad's centre, measured in
        // texels: one for the diagonal flag, two for the 120 degree flag.
        let turns = select(0.0, 1.0, (in.flags & 4u) != 0u)
            + select(0.0, 2.0, (in.flags & 16u) != 0u);
        if turns > 0.0 {
            let angle = -turns * 1.04719755;
            let texel = max(tileset.tile_size, vec2<f32>(1.0));
            let half = quad_size * 0.5;
            let d = (offset - half) * texel;
            let c = cos(angle);
            let s = sin(angle);
            offset = half + vec2<f32>(c * d.x - s * d.y, s * d.x + c * d.y) / texel;
        }
    }

    // Parallax: the layer follows `follow` of the camera across the plane.
    let shift = layer.translation + tile_eye_on_plane() * layer.follow;
    var world = tile_world((cell_origin + quad_min + offset) * layer.cell_size + shift);
    var index = in.tile;
    if (in.flags & 8u) != 0u {
        index = layer.anim[(in.tile >> 2u) & 63u][in.tile & 3u];
    }
    // <tile-deform:vertex>
    // <tile-hook:vertex>
    var clip = camera.view_proj * vec4<f32>(world, 1.0);
    if (layer.flags & 1u) != 0u {
        // The whole layer moves by one shift, so neighbouring tiles stay
        // edge to edge.
        clip = tile_pixel_snap(
            clip,
            camera.view_proj * vec4<f32>(tile_world(shift), 1.0),
            vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height),
        );
    }

    let columns = max(tileset.columns, 1u);
    let origin = vec2<f32>(tileset.margin)
        + vec2<f32>(f32(index % columns), f32(index / columns))
        * (tileset.tile_size + vec2<f32>(tileset.spacing));

    // The drawn corner in image coordinates (y down), taken back through the
    // flips to the point of the source tile it shows: vertical, horizontal,
    // then diagonal, the reverse of the order Tiled applies them. On a
    // hexagonal grid the diagonal flag turned the quad instead.
    var q = vec2<f32>(corner.x, 1.0 - corner.y);
    if (in.flags & 2u) != 0u {
        q.y = 1.0 - q.y;
    }
    if (in.flags & 1u) != 0u {
        q.x = 1.0 - q.x;
    }
    if (in.flags & 4u) != 0u && !hexagonal {
        q = q.yx;
    }

    out.clip_pos = clip;
    out.mask_clip = clip;
    out.texel = origin + q * texels;
    out.rect = vec4<f32>(origin, origin + texels);
    out.tint = in.tint * layer.tint;
    out.world_pos = world;
    out.cell = in.cell;
    out.flags = in.flags;
    return out;
}

// The position under this fragment in an atlas `dims` texels across,
// clamped half a texel inside the tile so a linear filter never reaches a
// neighbouring tile.
fn tile_atlas_uv(in: TileOut, dims: vec2<u32>) -> vec2<f32> {
    let t = clamp(in.texel, in.rect.xy + vec2<f32>(0.5), in.rect.zw - vec2<f32>(0.5));
    return t / vec2<f32>(dims);
}

// The world normal of the tile's normal atlas at this fragment, taken
// through the tile's flips and turns the way its texels were, then onto the
// layer's plane.
fn tile_normal(in: TileOut) -> vec3<f32> {
    let mapped = textureSampleLevel(
        normal_atlas,
        atlas_sampler,
        tile_atlas_uv(in, textureDimensions(normal_atlas, 0)),
        0.0,
    ).rgb;
    let n = normalize(mapped * 2.0 - vec3<f32>(1.0));
    // In image axes (y down), reflected back through the flips: diagonal,
    // horizontal, then vertical, the reverse of the texel lookup.
    var v = vec2<f32>(n.x, -n.y);
    let hexagonal = (tileset.orientation & 8u) != 0u;
    if (in.flags & 4u) != 0u && !hexagonal {
        v = v.yx;
    }
    if (in.flags & 1u) != 0u {
        v.x = -v.x;
    }
    if (in.flags & 2u) != 0u {
        v.y = -v.y;
    }
    // Plane axes, y up, then a hexagonal tile's turns as its quad turned.
    var p = vec2<f32>(v.x, -v.y);
    if hexagonal {
        let turns = select(0.0, 1.0, (in.flags & 4u) != 0u)
            + select(0.0, 2.0, (in.flags & 16u) != 0u);
        let angle = -turns * 1.04719755;
        let c = cos(angle);
        let s = sin(angle);
        p = vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
    }
    if layer.plane == 1u {
        return normalize(vec3<f32>(p.x, -n.z, p.y));
    }
    return normalize(vec3<f32>(p.x, p.y, n.z));
}

// The tile's colour at this fragment: the atlas texel times the tints. The
// atlas is read at its top level.
fn tile_colour(in: TileOut) -> vec4<f32> {
    let sampled = textureSampleLevel(
        atlas,
        atlas_sampler,
        tile_atlas_uv(in, textureDimensions(atlas, 0)),
        0.0,
    );
    if tileset.has_texture != 0u {
        return in.tint * sampled;
    }
    return in.tint;
}
