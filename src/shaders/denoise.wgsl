// ---------------------------------------------------------------------------
// Edge-aware a-trous denoiser for the path tracer accumulation buffer.
//
// A spatial-only SVGF-style filter: an a-trous wavelet with a growing step size
// (dispatched several times with step 1, 2, 4, ...), edge-stopped on the
// first-hit normal and the local luminance so it smooths noise without crossing
// silhouettes. Colour is demodulated by the first-hit albedo before filtering
// and remodulated after, so texture detail survives the blur. Sky pixels (the
// albedo guide marks them with w = 0) are copied through untouched.
// ---------------------------------------------------------------------------

struct DenoiseParams {
    dims:    vec2<u32>,
    step:    u32,
    _pad:    u32,
    sigma_n: f32,
    sigma_l: f32,
    _pad2:   vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: DenoiseParams;
@group(0) @binding(1) var<storage, read> gbuf_albedo: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gbuf_normal: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4<f32>>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Demodulated (irradiance) colour at a pixel: radiance divided by its albedo.
fn irradiance(idx: u32) -> vec3<f32> {
    return src[idx].rgb / max(gbuf_albedo[idx].rgb, vec3<f32>(0.02));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.dims.x;
    let h = params.dims.y;
    if gid.x >= w || gid.y >= h { return; }
    let idx = gid.y * w + gid.x;

    // Sky and other primary-miss pixels have no surface to filter against; keep
    // the accumulated value.
    if gbuf_albedo[idx].w < 0.5 {
        dst[idx] = src[idx];
        return;
    }

    let n_c = gbuf_normal[idx].xyz;
    let irr_c = irradiance(idx);
    let l_c = luma(irr_c);

    // Separable a-trous kernel { 1, 4, 6, 4, 1 } / 16 applied as a 5x5 outer
    // product, sampled at the current step size.
    let kernel = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    let step = i32(params.step);

    var sum = vec3<f32>(0.0);
    var sum_w = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let sx = i32(gid.x) + dx * step;
            let sy = i32(gid.y) + dy * step;
            if sx < 0 || sy < 0 || sx >= i32(w) || sy >= i32(h) { continue; }
            let q = u32(sy) * w + u32(sx);
            // Only filter across surface pixels; skip sky neighbours.
            if gbuf_albedo[q].w < 0.5 { continue; }

            let irr_q = irradiance(q);
            let n_q = gbuf_normal[q].xyz;
            let w_n = pow(max(dot(n_c, n_q), 0.0), params.sigma_n);
            let w_l = exp(-abs(l_c - luma(irr_q)) / (params.sigma_l + 1.0e-3));
            let h_ij = kernel[dx + 2] * kernel[dy + 2];
            let weight = h_ij * w_n * w_l;

            sum = sum + irr_q * weight;
            sum_w = sum_w + weight;
        }
    }

    let filtered = select(irr_c, sum / sum_w, sum_w > 0.0);
    dst[idx] = vec4<f32>(filtered * gbuf_albedo[idx].rgb, 1.0);
}
