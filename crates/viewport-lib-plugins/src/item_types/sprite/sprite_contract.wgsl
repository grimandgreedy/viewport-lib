// The structs and the lanes a sprite material's body reads, whatever draws
// it: spliced before the body in the sprite shaders (after
// `sprite_hook.wgsl`) and in the tilemap shaders of a tile material. The
// drawing shader fills `hook_params` before calling the body.

struct HookParams {
    lanes: array<vec4<f32>, 8>,
};
// The material's lanes as its body reads them, lane 0 carrying the time.
var<private> hook_params: HookParams;

// What a custom material's `shade_sprite` receives. The first seven fields
// are the particle hook's `ParticleFragment`, so one body draws both.
struct SpriteFragment {
    // Atlas UV of this fragment.
    uv:          vec2<f32>,
    // The sprite's texture at `uv`, palette-swapped when the batch swaps
    // palettes; white for an untextured batch.
    texel:       vec4<f32>,
    // The sprite's tint.
    colour:      vec4<f32>,
    // A particle's life; always 0 for a sprite.
    age:         f32,
    age_sec:     f32,
    world:       vec3<f32>,
    // Seconds, from the frame time.
    time:        f32,
    // Position across the quad, 0 to 1 from its bottom-left corner. Outside
    // that range in the margin an outline adds.
    local_uv:    vec2<f32>,
    // The sprite's atlas rect, flips applied.
    uv_rect:     vec4<f32>,
    flash:       f32,
    dissolve:    f32,
    palette_row: u32,
    // Screen-space derivatives of `uv`, for `textureSampleGrad`.
    uv_ddx:      vec2<f32>,
    uv_ddy:      vec2<f32>,
    // The sprite's user value (`SpriteItem::user_values`), as the vertex
    // hook reads it.
    user:        u32,
};

// What a custom material's `displace_sprite` receives, for one corner of the
// sprite's quad. It returns where the corner goes, in world space.
//
// Every position is relative to the renderer's world origin, as the mesh
// deformers' are. A body that samples a world-anchored pattern takes the
// origin through its own parameters.
struct SpriteVertex {
    // Which corner: 0 bottom left, 1 bottom right, 2 top left, 3 top right.
    corner:   u32,
    // The corner's position across the quad, 0 to 1 from its bottom-left
    // corner. Outside that range in the margin an outline adds.
    local_uv: vec2<f32>,
    // The corner in world space (for a screen-space size, on the plane
    // through the sprite's position facing the camera).
    world:    vec3<f32>,
    // The sprite's position in world space. For a tile, the bottom-left
    // corner of its cell's bounding box.
    anchor:   vec3<f32>,
    // Seconds, from the frame time.
    time:     f32,
    // The midpoint of the quad's bottom edge in world space, before any
    // displacement: where a plant bends from.
    base:     vec3<f32>,
    // Where `anchor` sits across the quad, 0 to 1 from its bottom-left corner.
    pivot:    vec2<f32>,
    // The quad's width and height in world units.
    size:     vec2<f32>,
    // The quad's unit axes in world space, after rotation: `right` along its
    // bottom edge, `up` from its bottom edge to its top.
    right:    vec3<f32>,
    up:       vec3<f32>,
    // The world size of one source texel along `right` and `up`; zero for an
    // untextured sprite or tile.
    texel:    vec2<f32>,
    // The sprite's user value (`SpriteItem::user_values`), or for a tile the
    // tileset's value for the tile it shows (`Tileset::user_values`). Zero
    // when none is set.
    user:     u32,
    // For a tile, the tileset tile it shows this frame; zero for a sprite.
    tile:     u32,
    // For a tile, the tile as placed: the tile index, or for an animated tile
    // its animation slot. Zero for a sprite.
    tile_placed: u32,
};
