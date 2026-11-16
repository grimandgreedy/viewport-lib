// The declarations a sprite material reads, spliced before its body: in the
// effect shaders after the body of `sprite.wgsl` or `sprite_lit.wgsl`, and in
// the pick and outline-mask shaders of a material with a vertex hook.
//
// Group 3: this batch's effect block (2), the dissolve noise (3) and the
// palette (4), then the sampler (5) and the textures (6..) a custom material
// reads, declared by the composer, one per texture it uses. Bindings 0-1 are
// unused: the lit body reads group 1's normal map. A material body follows
// the particle hook contract: `sprite_samp`, `htex0..htex3` and
// `hook_params.lanes` (lane 0 reserved, `.w` the time).

// Per-batch effect block. Must match `SpriteFxUniform` in `effects.rs`.
struct SpriteFx {
    flash_colour:   vec4<f32>,
    edge_colour:    vec4<f32>,
    outline_colour: vec4<f32>,
    edge_width:     f32,
    noise_scale:    f32,
    // Outline width in device pixels; 0 draws no outline.
    outline_px:     f32,
    // Bit 0: a noise texture is bound. Bit 1: palette swap. Bit 2: the
    // palette index is the texel's luminance rather than its red channel.
    flags:          u32,
    // A custom material's parameter lanes. Lane 0 is reserved; parameter i
    // is lane 1 + i.
    lanes:          array<vec4<f32>, 8>,
};

@group(3) @binding(2) var<uniform> sprite_fx:         SpriteFx;
@group(3) @binding(3) var          sprite_fx_noise:   texture_2d<f32>;
@group(3) @binding(4) var          sprite_fx_palette: texture_2d<f32>;
@group(3) @binding(5) var          sprite_samp:       sampler;
