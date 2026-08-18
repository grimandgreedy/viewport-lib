// exposure.wgsl : GPU auto-exposure. A log-luminance histogram over the HDR
// scene target, reduced to a percentile-clipped average, converted to a target
// EV100 and eased into a persistent exposure buffer. Three entry points share
// one bind group and run clear -> build -> resolve between the HDR pass and the
// tone map, in the same submission, so a single dirty render is correctly
// exposed on its own frame (no CPU readback, no cross-frame dependency).

const HISTOGRAM_BINS: u32 = 256u;

// Per-frame metering + adaptation parameters (see `ExposureParams` in Rust).
struct ExposureParams {
    min_log_lum:       f32,
    inv_log_lum_range: f32,
    log_lum_range:     f32,
    k_factor:          f32,
    min_ev:            f32,
    max_ev:            f32,
    compensation:      f32,
    exposure_boost:    f32,
    speed_up:          f32,
    speed_down:        f32,
    dt:                f32,
    low_percent:       f32,
    high_percent:      f32,
    tex_width:         f32,
    tex_height:        f32,
    center_weight:     f32,
    adaptation:        f32,
    _pad0:             f32,
    _pad1:             f32,
    _pad2:             f32,
};

// Persistent adaptation state; `exposure` is the linear multiplier the tone map
// reads (see `ExposureState` in Rust and `tone_map.wgsl`).
struct ExposureState {
    exposure:   f32,
    current_ev: f32,
    target_ev:  f32,
    adapting:   f32,
};

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: ExposureParams;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, HISTOGRAM_BINS>;
@group(0) @binding(3) var<storage, read_write> state: ExposureState;
@group(0) @binding(4) var depth_texture: texture_depth_2d;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Map an HDR luminance to a histogram bin. Near-black pixels collapse to bin 0
// and are excluded from the average by `resolve_main`.
fn bin_for_lum(lum: f32) -> u32 {
    if lum < 1e-4 {
        return 0u;
    }
    let t = clamp((log2(lum) - params.min_log_lum) * params.inv_log_lum_range, 0.0, 1.0);
    return u32(t * f32(HISTOGRAM_BINS - 1u) + 0.5);
}

// Zero the histogram before the build pass.
@compute @workgroup_size(256)
fn clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < HISTOGRAM_BINS {
        atomicStore(&histogram[gid.x], 0u);
    }
}

// Center-weight for a texel, in units of the quantised histogram weight. Pixels
// near the frame centre count more, so a centred subject drives exposure and
// zoom/pan barely move it. `center_weight = 0` gives a flat full-frame meter.
fn center_weight_for(gid: vec2<u32>, w: u32, h: u32) -> u32 {
    let uv = (vec2<f32>(vec2<u32>(gid)) + vec2<f32>(0.5)) / vec2<f32>(f32(w), f32(h));
    // Normalised radius from centre: 0 at centre, ~1 at the frame edges.
    let r = length((uv - vec2<f32>(0.5)) * 2.0) * 0.70710677;
    let falloff = smoothstep(1.0, 0.15, r); // 1 at centre -> 0 at the corners
    let weight = mix(1.0, falloff, params.center_weight);
    // Quantise to an integer weight; keep a floor so edge pixels still register.
    return max(u32(weight * 64.0 + 0.5), 1u);
}

// One invocation per HDR texel: bin its log-luminance with a center weight.
@compute @workgroup_size(16, 16, 1)
fn build_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(params.tex_width);
    let h = u32(params.tex_height);
    if gid.x >= w || gid.y >= h {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    // Skip background (far-plane) texels: the HDR target is cleared to the flat
    // background fill, which is not scene content and must not bias the meter.
    // (Matches the tone map's own background test.)
    if textureLoad(depth_texture, coord, 0) >= 0.999999 {
        return;
    }
    let texel = textureLoad(hdr_texture, coord, 0);
    let lum = luminance(texel.rgb);
    let bin = bin_for_lum(lum);
    // Bin 0 is a raw (unweighted) count of lit texels, used by resolve_main as a
    // confidence signal (how much of the frame is scene content vs excluded
    // background). Near-black texels land in bin 0 too but are never part of the
    // luminance average (the resolve loop starts at bin 1), so counting here does
    // not bias metering.
    atomicAdd(&histogram[0], 1u);
    if bin >= 1u {
        atomicAdd(&histogram[bin], center_weight_for(gid.xy, w, h));
    }
}

// Single invocation: reduce the histogram to an average log-luminance with a
// low/high percentile clip, convert to a target EV, ease the persistent EV
// toward it (dt<=0 snaps), then write the exposure multiplier the tone map reads.
@compute @workgroup_size(1)
fn resolve_main() {
    // Total lit pixels (bin 0 is near-black background / unlit, excluded).
    var total: u32 = 0u;
    for (var i: u32 = 1u; i < HISTOGRAM_BINS; i = i + 1u) {
        total = total + atomicLoad(&histogram[i]);
    }

    var avg_log: f32 = params.min_log_lum + params.log_lum_range * 0.5;
    if total > 0u {
        let lo = f32(total) * params.low_percent;
        let hi = f32(total) * params.high_percent;
        var seen: f32 = 0.0;
        var weighted: f32 = 0.0;
        var used: f32 = 0.0;
        for (var i: u32 = 1u; i < HISTOGRAM_BINS; i = i + 1u) {
            let count = f32(atomicLoad(&histogram[i]));
            if count <= 0.0 {
                continue;
            }
            let lo_i = seen;
            let hi_i = seen + count;
            seen = hi_i;
            // Fraction of this bin lying inside the [lo, hi] percentile window.
            let frac = max(0.0, min(hi_i, hi) - max(lo_i, lo));
            if frac <= 0.0 {
                continue;
            }
            let t = f32(i) / f32(HISTOGRAM_BINS - 1u);
            let log_lum = params.min_log_lum + t * params.log_lum_range;
            weighted = weighted + log_lum * frac;
            used = used + frac;
        }
        if used > 0.0 {
            avg_log = weighted / used;
        }
    }

    let l_avg = exp2(avg_log);
    // The metered value renders exactly to `middle_grey`. We meter an upper-mid
    // luminance band (low/high_percent ~0.65/0.95) for framing stability, so
    // `l_avg` is the well-lit key content, not a whole-frame average. Exposing
    // that key to the bare 18% grey card value would read ~1 stop dark (the band
    // is ~1 stop above the frame mean), so the target is raised to keep lit
    // surfaces looking lit while the exposure stays orbit-stable. Manual and
    // physical-camera exposure are unaffected; this only sets the auto target.
    let middle_grey = 0.36;
    let middle_grey_stops = log2(middle_grey / (params.k_factor / 100.0));
    // Partial adaptation. `target_ev` reduces to `log2(l_avg / middle_grey)` - the
    // full exposure that would render the metered value to `middle_grey`. Scaling
    // it by ADAPT (<1) makes the exposure only partially follow the scene: at the
    // reference level (l_avg == middle_grey) it is unchanged, and away from it the
    // image keeps some of its real brightness difference instead of being driven
    // all the way to grey. This is more eye-like (a dim view stays somewhat dim)
    // AND proportionally shrinks every framing-driven swing - orbit, zoom onto a
    // bright/dark object, and the flare when the camera faces a dimly-lit
    // underside - since those are all `l_avg` moving.
    var target_ev = params.adaptation * (log2(max(l_avg, 1e-5) * 100.0 / params.k_factor) - middle_grey_stops);
    target_ev = clamp(target_ev, params.min_ev, params.max_ev);

    // Persistent EV; snap on first use (CPU seeds a non-finite value) or dt<=0.
    var cur = state.current_ev;
    if !(cur > -1e30 && cur < 1e30) {
        cur = target_ev;
    }

    // Low-confidence guard. When only a small fraction of the frame is lit scene
    // content - the camera has orbited past the ground and is looking at mostly
    // empty (metering-excluded) background - the meter has lost its bright anchor
    // and the raw average would flare the exposure up. Ease the target back
    // toward the held EV in proportion to how little of the frame is lit, so
    // orbiting over the horizon holds steady instead of suddenly brightening.
    let raw_lit = f32(atomicLoad(&histogram[0]));
    let frame_px = max(params.tex_width * params.tex_height, 1.0);
    let lit_fraction = raw_lit / frame_px;
    // Gate only the BRIGHTENING direction: a darker meter reading because scene
    // content has left the frame (camera orbiting past the ground into mostly
    // excluded background) must not flare the exposure up, but a brighter reading
    // (scene genuinely got brighter) is always honoured so highlights stay
    // protected. When smoothing (dt>0) the gate is a soft ramp so brightening
    // eases in as content fills the frame. In snap mode (dt<=0) a partial gate
    // would instead ratchet `cur` toward the target one frame at a time - a
    // gradual change where the caller asked for an instant one - so there the
    // gate is a hard decision: honour the brighter target once enough of the
    // frame is lit scene, otherwise fully freeze the held EV (the flare guard).
    let gate_soft = smoothstep(0.15, 0.40, lit_fraction);
    let brighten_gate = select(gate_soft, step(0.275, lit_fraction), params.dt <= 0.0);
    if target_ev < cur {
        target_ev = mix(cur, target_ev, brighten_gate);
    }

    var new_ev: f32;
    if params.dt <= 0.0 {
        new_ev = target_ev;
    } else {
        let rate = select(params.speed_down, params.speed_up, target_ev > cur);
        let factor = clamp(1.0 - exp(-params.dt * rate), 0.0, 1.0);
        new_ev = cur + (target_ev - cur) * factor;
    }

    let ev_used = new_ev - params.compensation;
    let max_lum = 1.2 * exp2(ev_used);
    state.exposure = params.exposure_boost / max_lum;
    state.current_ev = new_ev;
    state.target_ev = target_ev;
    state.adapting = select(0.0, 1.0, abs(target_ev - new_ev) > 0.01);
}
