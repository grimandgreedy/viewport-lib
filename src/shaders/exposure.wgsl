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
    let texel = textureLoad(hdr_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let lum = luminance(texel.rgb);
    let bin = bin_for_lum(lum);
    atomicAdd(&histogram[bin], center_weight_for(gid.xy, w, h));
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
    var target_ev = log2(max(l_avg, 1e-5) * 100.0 / params.k_factor);
    target_ev = clamp(target_ev, params.min_ev, params.max_ev);

    // Persistent EV; snap on first use (CPU seeds a non-finite value) or dt<=0.
    var cur = state.current_ev;
    if !(cur > -1e30 && cur < 1e30) {
        cur = target_ev;
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
