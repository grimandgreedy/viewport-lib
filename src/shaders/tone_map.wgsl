// tone_map.wgsl : fullscreen post-process composite: tone mapping, bloom, SSAO, contact shadows.
// Renders a fullscreen triangle (no vertex buffer) using vertex_index.

struct ToneMapUniform {
    exposure:                f32,
    mode:                    u32,  // 0=Reinhard, 1=ACES, 2=KhronosNeutral
    bloom_enabled:           u32,
    ssao_enabled:            u32,
    contact_shadows_enabled: u32,
    edl_enabled:             u32,
    edl_radius:              f32,
    edl_strength:            f32,
    background_colour:        vec4<f32>,
    near_plane:              f32,
    far_plane:               f32,
    lic_enabled:             u32,
    lic_strength:            f32,
    foreground_enabled:      u32,
    _pad0:                   u32,
    _pad1:                   u32,
    _pad2:                   u32,
}

@group(0) @binding(0) var hdr_texture:  texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler:  sampler;
@group(0) @binding(2) var<uniform> params: ToneMapUniform;
@group(0) @binding(3) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var ao_texture:    texture_2d<f32>;
@group(0) @binding(5) var cs_texture:    texture_2d<f32>;
@group(0) @binding(6) var depth_texture: texture_depth_2d;
@group(0) @binding(7) var lic_texture:   texture_2d<f32>;
// Foreground pass coverage mask: depth < 1.0 where foreground geometry was
// drawn. The screen-space terms below (AO, contact shadows, EDL, LIC) are
// computed from scene depth, which is unrelated to what a covered pixel
// shows, so they are skipped there.
@group(0) @binding(8) var foreground_depth: texture_depth_2d;

// Exposure state buffer (see exposure.wgsl / ExposureState in Rust). `exposure`
// is the pre-tone-map linear multiplier, written CPU-side for manual/physical
// exposure or by the auto-exposure resolve compute pass. Replaces the old
// `ToneMapUniform.exposure` scalar so every exposure mode shares one path.
struct ExposureState {
    exposure:   f32,
    current_ev: f32,
    target_ev:  f32,
    adapting:   f32,
}
@group(0) @binding(9) var<storage, read> exposure_state: ExposureState;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    let uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv);
}

fn reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (x + vec3<f32>(1.0));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

// Khronos PBR Neutral tone mapper.
// Passes values below ~0.76 through with only a small black-point offset,
// then compresses highlights. Designed to preserve hand-authored SDR colours.
fn khronos_neutral(colour: vec3<f32>) -> vec3<f32> {
    let start_compression: f32 = 0.8 - 0.04;
    let desaturation: f32 = 0.15;

    let x = min(colour.r, min(colour.g, colour.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    let c = colour - offset;

    let peak = max(c.r, max(c.g, c.b));
    if peak < start_compression {
        return c;
    }

    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    let scaled = c * (new_peak / peak);
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(scaled, vec3<f32>(new_peak), g);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth_dims = textureDimensions(depth_texture);
    let depth_uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(0.99999994));
    let depth_coord = vec2<i32>(vec2<u32>(depth_uv * vec2<f32>(depth_dims)));
    let depth = textureLoad(depth_texture, depth_coord, 0);
    // With no opaque geometry at this pixel (depth at the far plane), the HDR
    // buffer holds only premultiplied *transparent* contributions (additive /
    // OIT particles) blended over a transparent-black clear; the flat background
    // colour is composited under them after tone mapping, below. The buffer is
    // cleared with alpha=0 and transparent draws raise alpha, so alpha ~ 0 means
    // nothing was drawn here and this is a pure background pixel.
    var covered = false;
    if params.foreground_enabled != 0u {
        let fg_dims = textureDimensions(foreground_depth);
        let fg_coord = vec2<i32>(vec2<u32>(depth_uv * vec2<f32>(fg_dims)));
        covered = textureLoad(foreground_depth, fg_coord, 0) < 1.0;
    }
    let is_background = depth >= 0.999999 && !covered;
    let hdr = textureSample(hdr_texture, hdr_sampler, in.uv);

    // Bloom spreads (via blur) beyond the pixels that produced it, so it must
    // survive over empty background too. Sample it up front and only take the
    // pure-background fast path when nothing contributes here, neither a
    // transparent draw nor bloom. Otherwise a glow would be clipped hard at the
    // silhouette of whatever cast it.
    var bloom = vec3<f32>(0.0);
    if params.bloom_enabled != 0u {
        bloom = textureSample(bloom_texture, hdr_sampler, in.uv).rgb;
    }
    if is_background && hdr.a < 0.001 && dot(bloom, vec3<f32>(1.0)) < 0.0003 {
        return params.background_colour;
    }

    // Add bloom additively before tone mapping.
    var colour = hdr.rgb + bloom;

    // Screen-space geometry terms (AO, contact shadows) apply only where geometry
    // was shaded. Over empty background there is no geometry, so skipping them
    // both avoids modulating transparent particles / bloom by stale values in
    // non-geometry regions and keeps the bloom-over-background path independent of
    // those textures' clear values.
    if !is_background && !covered {
        // Multiply by AO before tone mapping.
        if params.ssao_enabled != 0u {
            let ao = textureSampleLevel(ao_texture, hdr_sampler, in.uv, 0.0).r;
            colour = colour * ao;
        }

        // Multiply by contact shadow factor before tone mapping.
        if params.contact_shadows_enabled != 0u {
            let cs = textureSampleLevel(cs_texture, hdr_sampler, in.uv, 0.0).r;
            colour = colour * cs;
        }
    }

    // Eye-Dome Lighting: darken pixels at depth discontinuities.
    // Depth is linearized (z_eye / far) before the log comparison so that the
    // log differences are large enough to produce a visible effect regardless
    // of the near/far plane ratio.
    if params.edl_enabled != 0u && !covered {
        let n = params.near_plane;
        let f = params.far_plane;
        // Linear depth in [near/far, 1]: z_eye/far = n / (f - d*(f-n))
        let lin_dc  = n / (f - depth * (f - n));
        let log_ldc = log(lin_dc);
        let dims_i  = vec2<i32>(depth_dims);
        let edl_r   = i32(max(1.0, round(params.edl_radius)));
        var edl_nc: vec2<i32>;
        var edl_sum = 0.0;
        edl_nc = clamp(depth_coord + vec2<i32>( edl_r,      0), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(n / (f - textureLoad(depth_texture, edl_nc, 0) * (f - n))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>( edl_r,  edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(n / (f - textureLoad(depth_texture, edl_nc, 0) * (f - n))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(      0,  edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(n / (f - textureLoad(depth_texture, edl_nc, 0) * (f - n))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(-edl_r,  edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(n / (f - textureLoad(depth_texture, edl_nc, 0) * (f - n))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(-edl_r,      0), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(n / (f - textureLoad(depth_texture, edl_nc, 0) * (f - n))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(-edl_r, -edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(n / (f - textureLoad(depth_texture, edl_nc, 0) * (f - n))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(      0, -edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(n / (f - textureLoad(depth_texture, edl_nc, 0) * (f - n))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>( edl_r, -edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(n / (f - textureLoad(depth_texture, edl_nc, 0) * (f - n))) - log_ldc);
        // Normalize by sample count then apply exponential response so strength=1
        // gives moderate edge darkening and strength=5 gives near-complete darkening.
        let edl_factor = 1.0 - exp(-params.edl_strength * edl_sum / 8.0);
        colour = colour * (1.0 - edl_factor);
    }

    // Surface LIC: modulate colour by LIC intensity (0.5 = neutral, no change).
    // A surface effect, so it only applies where geometry was shaded.
    if !is_background && !covered && params.lic_enabled != 0u {
        let lic_val = textureSampleLevel(lic_texture, hdr_sampler, in.uv, 0.0).r;
        let lic_factor = 1.0 + params.lic_strength * (lic_val * 2.0 - 1.0);
        colour = colour * max(0.0, lic_factor);
    }

    // Pre-tone-mapping exposure (from the exposure state buffer).
    colour = colour * exposure_state.exposure;

    // Tone mapping.
    if params.mode == 0u {
        colour = reinhard(colour);
    } else if params.mode == 1u {
        colour = aces(colour);
    } else {
        colour = khronos_neutral(colour);
    }

    // Composite transparent HDR content over the flat background colour. Without
    // this, a faint particle over empty scene replaces the background with its
    // own dim premultiplied value (reading as near-black) instead of adding to
    // it. Skipped when opaque geometry is present (it already supplied the base
    // colour tone-mapped above). Done in display space, after tone mapping, so
    // pure-background pixels match the early-out above exactly. Alpha is the
    // transparent coverage (>1 possible from stacked additive draws), clamped so
    // saturated regions fully replace the background.
    if is_background {
        colour = colour + params.background_colour.rgb * (1.0 - clamp(hdr.a, 0.0, 1.0));
    }

    return vec4<f32>(colour, 1.0);
}
