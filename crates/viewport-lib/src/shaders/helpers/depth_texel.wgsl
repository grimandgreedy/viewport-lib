// One texel of a depth texture, for a pass that needs the stored value.
//
// Bind the depth view as `texture_2d<f32>` (an unfilterable-float layout entry
// accepts a depth view) with a non-filtering, clamp-to-edge sampler. Sampling
// at the texel centre with a nearest filter returns the texel `textureLoad`
// would. `textureLoad` itself does not work on GL: naga cannot translate it on
// a `texture_depth_2d`, and on `texture_2d<f32>` it reads 0 on GLES, where the
// depth texture keeps GL's default filters and is incomplete without a
// `NEAREST` sampler. Out-of-range coordinates read the edge texel.
fn depth_texel(tex: texture_2d<f32>, samp: sampler, coord: vec2<i32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(tex));
    return textureSampleLevel(tex, samp, (vec2<f32>(coord) + 0.5) / dims, 0.0).r;
}

// One texel of a stencil aspect, bound as `texture_2d<u32>` with the same kind
// of non-filtering, clamp-to-edge sampler. A gather with the sampler rather
// than `textureLoad`: on GL the stencil reads as an integer texture, which
// is incomplete under wgpu's default filters unless a `NEAREST` sampler is
// bound. Sampled a quarter texel past the centre, the gather's last component
// is texel `coord`.
fn stencil_texel(tex: texture_2d<u32>, samp: sampler, coord: vec2<i32>) -> u32 {
    let dims = vec2<f32>(textureDimensions(tex));
    return textureGather(0, tex, samp, (vec2<f32>(coord) + 0.75) / dims).w;
}
