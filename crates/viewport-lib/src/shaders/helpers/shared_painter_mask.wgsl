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

// The 2D light layers: a `texture_2d_array<f32>` the drawing shader declares
// beside the masks, one array layer of light per run of painter layers. Lit
// content multiplies its colour by the light under it in the array layer of
// the range its own painter layer lies in, so each range is lit by its own
// layer only.
struct PainterLights {
    // Ranges in use.
    count: u32,
    // Multiplied into a sampled light: 2 for a layer holding half the light.
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    // Per range: first layer, last layer, array layer, and one more than the
    // array layer holding the range's light directions (0 for none).
    ranges: array<vec4<u32>, 16>,
};

// The array layer lighting painter `layer`, or -1 when no range holds it.
fn painter_light_slice(lights: PainterLights, layer: u32) -> i32 {
    for (var i = 0u; i < min(lights.count, 16u); i = i + 1u) {
        let r = lights.ranges[i];
        if layer >= r.x && layer <= r.y {
            return i32(r.z);
        }
    }
    return -1;
}

// `rgb` lit by the light layer texel `light` (loaded from the array layer
// `painter_light_slice` named).
fn painter_light_apply(rgb: vec3<f32>, light: vec4<f32>, lights: PainterLights) -> vec3<f32> {
    return rgb * light.rgb * lights.scale;
}

// The array layer holding the light directions for painter `layer`, or -1
// when its range has none (or no range holds it).
fn painter_light_dir_slice(lights: PainterLights, layer: u32) -> i32 {
    for (var i = 0u; i < min(lights.count, 16u); i = i + 1u) {
        let r = lights.ranges[i];
        if layer >= r.x && layer <= r.y {
            return i32(r.w) - 1;
        }
    }
    return -1;
}

// `rgb` with the world-space normal `n` lit by the light layer texel `light`
// and the direction texel `dir` under it. `dir` holds the luminance-weighted
// sum of the unit directions to the lights in `rgb` and the summed weight in
// `a`; the part of the light they account for is shaded by `n` against
// their weighted mean, the rest (ambient, global and polygon lights) stays
// flat. Exact while `n` faces every light; a light behind the normal is
// averaged in rather than clamped on its own.
fn painter_light_apply_normal(
    rgb: vec3<f32>,
    light: vec4<f32>,
    dir: vec4<f32>,
    n: vec3<f32>,
    lights: PainterLights,
) -> vec3<f32> {
    let total = light.rgb * lights.scale;
    let lum = dot(total, vec3<f32>(0.2126, 0.7152, 0.0722));
    let w = max(dir.a, 0.0);
    let direct = clamp(w / max(lum, 1e-6), 0.0, 1.0);
    let shade = max(dot(n, dir.rgb / max(w, 1e-6)), 0.0);
    return rgb * total * (1.0 - direct + direct * shade);
}
