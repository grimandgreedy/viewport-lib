// @viewport-wgsl-version: 1
// Painter mask test. Declares no bindings: the drawing shader declares the
// mask texture and the `PainterMasks` uniform in a group it owns, loads the
// texel under the fragment and passes it in.
//
// The mask texture holds one coverage channel per slot (red slot 0 to alpha
// slot 3) over the viewport. The uniform says, per slot, which painter
// layers the slot's mask applies to and whether a mask drew into it this
// frame.

struct PainterMasks {
    // Per slot: first layer, last layer, 1 when a mask drew into the slot
    // this frame, unused.
    slots: array<vec4<u32>, 4>,
};

// The texel of a `dims`-sized mask texture under a fragment at `ndc`.
fn painter_mask_texel(ndc: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let size = vec2<i32>(dims);
    return clamp(vec2<i32>(floor(uv * vec2<f32>(dims))), vec2<i32>(0), size - vec2<i32>(1));
}

// How much of a fragment on painter `layer` a mask test keeps, given the
// mask texel under it. `test` packs the slot in bits 0-1 and the side in bits
// 2-3 (1 visible inside, 2 visible outside), as `MaskTest::word` does. A
// fragment with no test, testing a slot no mask drew into, or on a layer
// outside the slot's range keeps everything.
fn painter_mask_keep(texel: vec4<f32>, masks: PainterMasks, test: u32, layer: u32) -> f32 {
    let side = (test >> 2u) & 3u;
    let slot = masks.slots[test & 3u];
    if side == 0u || slot.z == 0u || layer < slot.x || layer > slot.y {
        return 1.0;
    }
    let coverage = texel[test & 3u];
    return select(1.0 - coverage, coverage, side == 1u);
}
