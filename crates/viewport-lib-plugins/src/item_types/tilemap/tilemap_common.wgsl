// Declarations and the vertex stage every tilemap shader shares, spliced in
// front of each body. The structs must match `TilesetUniformData` and
// `LayerUniformData` in `store.rs` exactly.
//
// Group 1: the uploaded map's tileset uniform, atlas and sampler.
// Group 2: the drawn layer's uniform, at a dynamic offset.
//
// One instance per occupied cell, six vertices each (two triangles):
//   location 0: cell, column in the low 16 bits and row in the high 16
//   location 1: tileset tile index, or animation slot when flag bit 3 is set
//   location 2: flags, bit 0 horizontal flip, 1 vertical, 2 diagonal, 3 animated
//   location 3: tint, unorm8x4 linear RGBA

struct TilesetUniform {
    tile_size:   vec2<f32>,  // texels
    columns:     u32,
    has_texture: u32,
    spacing:     f32,
    margin:      f32,
    _pad0:       u32,
    _pad1:       u32,
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
    // x: the blended layer's painter layer.
    painter_layer:   vec4<u32>,
    // The tile each animation slot shows this frame, four slots per entry.
    anim:            array<vec4<u32>, 64>,
    // A tile material's parameter lanes; lane 0 is reserved.
    lanes:           array<vec4<f32>, 8>,
};

@group(1) @binding(0) var<uniform> tileset:       TilesetUniform;
@group(1) @binding(1) var          atlas:         texture_2d<f32>;
@group(1) @binding(2) var          atlas_sampler: sampler;
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
    let cell = vec2<f32>(f32(in.cell & 0xffffu), f32(in.cell >> 16u));

    // Parallax: the layer follows `follow` of the camera across the plane.
    let shift = layer.translation + tile_eye_on_plane() * layer.follow;
    var world = tile_world((cell + corner) * layer.cell_size + shift);
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

    var index = in.tile;
    if (in.flags & 8u) != 0u {
        index = layer.anim[(in.tile >> 2u) & 63u][in.tile & 3u];
    }
    let columns = max(tileset.columns, 1u);
    let origin = vec2<f32>(tileset.margin)
        + vec2<f32>(f32(index % columns), f32(index / columns))
        * (tileset.tile_size + vec2<f32>(tileset.spacing));

    // The drawn corner in image coordinates (y down), taken back through the
    // flips to the point of the source tile it shows: vertical, horizontal,
    // then diagonal, the reverse of the order Tiled applies them.
    var q = vec2<f32>(corner.x, 1.0 - corner.y);
    if (in.flags & 2u) != 0u {
        q.y = 1.0 - q.y;
    }
    if (in.flags & 1u) != 0u {
        q.x = 1.0 - q.x;
    }
    if (in.flags & 4u) != 0u {
        q = q.yx;
    }

    out.clip_pos = clip;
    out.mask_clip = clip;
    out.texel = origin + q * tileset.tile_size;
    out.rect = vec4<f32>(origin, origin + tileset.tile_size);
    out.tint = in.tint * layer.tint;
    out.world_pos = world;
    out.cell = in.cell;
    return out;
}

// The tile's colour at this fragment: the atlas texel times the tints. The
// atlas is read at its top level, clamped half a texel inside the tile, so a
// linear filter never reaches a neighbouring tile.
fn tile_colour(in: TileOut) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(atlas, 0));
    let t = clamp(in.texel, in.rect.xy + vec2<f32>(0.5), in.rect.zw - vec2<f32>(0.5));
    let sampled = textureSampleLevel(atlas, atlas_sampler, t / dims, 0.0);
    if tileset.has_texture != 0u {
        return in.tint * sampled;
    }
    return in.tint;
}
