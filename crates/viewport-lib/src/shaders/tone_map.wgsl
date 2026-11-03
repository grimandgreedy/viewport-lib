// tone_map.wgsl : fullscreen post-process composite: tone mapping, bloom, SSAO, contact shadows.
// Renders a fullscreen triangle (no vertex buffer) using vertex_index.

struct ToneMapUniform {
    exposure:                f32,
    // 0 Reinhard, 1 ACES fit, 2 Khronos Neutral, 3 AgX, 4 ACES RRT, 5 generic,
    // 6 view LUT.
    mode:                    u32,
    bloom_enabled:           u32,
    ssao_enabled:            u32,
    contact_shadows_enabled: u32,
    edl_enabled:             u32,
    edl_radius:              f32,
    edl_strength:            f32,
    background_colour:        vec4<f32>,
    // Near and the far distance EDL normalises against.
    near_plane:              f32,
    far_plane:               f32,
    // Non-zero for an orthographic projection.
    orthographic:            u32,
    // The projection's far plane, 0 when it has none.
    projection_far:          f32,
    foreground_enabled:      u32,
    vignette_amount:         f32,
    vignette_radius:         f32,
    vignette_softness:       f32,
    // 0 = no grading, 1 = display-referred, 2 = scene-referred.
    grade_mode:              u32,
    grade_lut_size:          f32,
    // 0 = Rec.709, 1 = Rec.2020.
    working_space:           u32,
    // 0 = none, 1 = punchy, 2 = golden.
    agx_look:                u32,
    // Generic curve: contrast, input scale, output scale; w is the view LUT's
    // slice count.
    generic_curve:           vec4<f32>,
}

@group(0) @binding(0) var hdr_texture:  texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler:  sampler;
@group(0) @binding(2) var<uniform> params: ToneMapUniform;
@group(0) @binding(3) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var ao_texture:    texture_2d<f32>;
@group(0) @binding(5) var cs_texture:    texture_2d<f32>;
@group(0) @binding(6) var depth_texture: texture_2d<f32>;
@group(0) @binding(7) var depth_sampler: sampler;
// Foreground pass coverage mask: not background where foreground geometry was
// drawn. The screen-space terms below (AO, contact shadows, EDL) are
// computed from scene depth, which is unrelated to what a covered pixel
// shows, so they are skipped there.
@group(0) @binding(8) var foreground_depth: texture_2d<f32>;

// Exposure state buffer (see exposure.wgsl / ExposureState in Rust). The
// composite does not apply it: the scene shaders multiply every radiance
// source by the camera's exposure where they read it, so the HDR colour here
// is already exposed. Bound so the layout matches the exposure producer's.
struct ExposureState {
    exposure:   f32,
    current_ev: f32,
    target_ev:  f32,
    adapting:   f32,
}
@group(0) @binding(9) var<storage, read> exposure_state: ExposureState;

// Colour-grading strip LUT: n slices of n x n laid out horizontally (width
// n*n, height n = params.grade_lut_size). Red indexes across a slice, green
// down it, blue selects the slice. Neutral placeholder when grading is off.
@group(0) @binding(10) var grade_lut: texture_2d<f32>;

// View-transform strip LUT for mode 6, same layout as the grade LUT, holding
// display values over the scene-referred shaper. Neutral placeholder
// otherwise.
@group(0) @binding(11) var view_lut: texture_2d<f32>;

// The scene pass's light split: red is the indirect share of each opaque
// pixel's light, green the direct share, 1 for both where nothing wrote it.
// AO scales the indirect share and contact shadows the direct share, so
// neither darkens emission. A white placeholder when neither is on. Under
// SSAA the resolve has already averaged it down.
@group(0) @binding(12) var indirect_fraction_texture: texture_2d<f32>;

// #include "helpers/depth_texel.wgsl"
// #include "helpers/depth.wgsl"
// #include "helpers/tone_operators.wgsl"

const GRADE_DISPLAY: u32 = 1u;
const GRADE_SCENE: u32 = 2u;

const WORKING_REC2020: u32 = 1u;

const MODE_VIEW_LUT: u32 = 6u;

// The working-space colour in Rec.2020, for the operators that run there.
fn to_rec2020(colour: vec3<f32>, rec2020: bool) -> vec3<f32> {
    if rec2020 {
        return colour;
    }
    return SRGB_TO_REC2020 * colour;
}

// An operator's Rec.2020 result back in the working space, negatives at 0.
fn from_rec2020(colour: vec3<f32>, rec2020: bool) -> vec3<f32> {
    if rec2020 {
        return max(colour, vec3<f32>(0.0));
    }
    return max(REC2020_TO_SRGB * colour, vec3<f32>(0.0));
}

// Scene-referred shaper: log2 of exposed radiance over [2^-14, 2^10] mapped
// to [0, 1]. Must equal the types crate's `VIEW_LUT_SHAPER`.
const GRADE_LOG2_MIN: f32 = -14.0;
const GRADE_LOG2_RANGE: f32 = 24.0;

// Sample a strip LUT of n slices at a colour in [0, 1], interpolating between
// the two nearest blue slices.
fn sample_strip(lut: texture_2d<f32>, n: f32, colour: vec3<f32>) -> vec3<f32> {
    let c = clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0));
    let b = c.b * (n - 1.0);
    let slice0 = floor(b);
    let slice1 = min(slice0 + 1.0, n - 1.0);
    let f = b - slice0;
    let u0 = (slice0 * n + c.r * (n - 1.0) + 0.5) / (n * n);
    let u1 = (slice1 * n + c.r * (n - 1.0) + 0.5) / (n * n);
    let v = (c.g * (n - 1.0) + 0.5) / n;
    let s0 = textureSampleLevel(lut, hdr_sampler, vec2<f32>(u0, v), 0.0).rgb;
    let s1 = textureSampleLevel(lut, hdr_sampler, vec2<f32>(u1, v), 0.0).rgb;
    return mix(s0, s1, f);
}

// The grade LUT at a colour in [0, 1].
fn grade(colour: vec3<f32>) -> vec3<f32> {
    return sample_strip(grade_lut, params.grade_lut_size, colour);
}

// Exposed radiance through the log shaper, before the clamp to [0, 1].
fn shape_scene(colour: vec3<f32>) -> vec3<f32> {
    let floor_value = vec3<f32>(exp2(GRADE_LOG2_MIN));
    return (log2(max(colour, floor_value)) - GRADE_LOG2_MIN) / GRADE_LOG2_RANGE;
}

// Grade exposed radiance through the log shaper and back.
fn grade_scene(colour: vec3<f32>) -> vec3<f32> {
    let graded = grade(shape_scene(colour));
    return exp2(graded * GRADE_LOG2_RANGE + GRADE_LOG2_MIN);
}

// Final display-space treatment shared by every exit: display-referred
// grading, then the vignette. Applied on the pure-background fast path too,
// so both cover empty regions.
fn display_finish(colour_in: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    var colour = colour_in;
    if params.grade_mode == GRADE_DISPLAY {
        colour = grade(colour);
    }
    if params.vignette_amount > 0.0 {
        // Distance normalised so 1.0 is the centre-to-corner distance.
        let d = length(uv - vec2<f32>(0.5)) * 1.4142135;
        let falloff = smoothstep(
            params.vignette_radius,
            params.vignette_radius + max(params.vignette_softness, 0.001),
            d,
        );
        colour = colour * (1.0 - params.vignette_amount * falloff);
    }
    return colour;
}

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

// View distance over the EDL reference far, capped at 1 so background and
// anything past the reference read as the far end.
fn edl_linear_depth(d: f32) -> f32 {
    let clip = vec4<f32>(
        params.near_plane,
        params.projection_far,
        f32(params.orthographic),
        0.0,
    );
    return min(viewport_depth_to_view_z(d, clip) / params.far_plane, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth_dims = textureDimensions(depth_texture);
    let depth_uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(0.99999994));
    let depth_coord = vec2<i32>(vec2<u32>(depth_uv * vec2<f32>(depth_dims)));
    let depth = depth_texel(depth_texture, depth_sampler, depth_coord);
    // With no opaque geometry at this pixel (depth at the far plane), the HDR
    // buffer holds only premultiplied *transparent* contributions (additive /
    // OIT particles) over a transparent-black clear; the background is
    // composited under them after tone mapping, below, and nowhere else. The
    // buffer is cleared to zero in both colour and alpha and transparent draws
    // raise alpha, so alpha ~ 0 means nothing was drawn here and this is a pure
    // background pixel.
    var covered = false;
    if params.foreground_enabled != 0u {
        let fg_dims = textureDimensions(foreground_depth);
        let fg_coord = vec2<i32>(vec2<u32>(depth_uv * vec2<f32>(fg_dims)));
        covered = !viewport_depth_is_background(
            depth_texel(foreground_depth, depth_sampler, fg_coord),
        );
    }
    let is_background = viewport_depth_is_background(depth) && !covered;
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
        // The uniform is premultiplied, so this is already the right pair: an
        // opaque background unchanged, a transparent one zero in every channel
        // rather than a colour at zero coverage (which no compositor can use).
        return vec4<f32>(
            display_finish(params.background_colour.rgb, in.uv),
            params.background_colour.a,
        );
    }

    var colour = hdr.rgb;

    // Screen-space geometry terms (AO, contact shadows) apply only where geometry
    // was shaded. Over empty background there is no geometry, so skipping them
    // both avoids modulating transparent particles by stale values in
    // non-geometry regions and keeps the bloom-over-background path independent of
    // those textures' clear values. Each darkens its share of the light only,
    // `1 - (1 - term) * share`: AO the indirect light, contact shadows (the key
    // light's screen-space shadow) the direct light.
    if !is_background && !covered {
        var split = vec2<f32>(1.0);
        if params.ssao_enabled != 0u || params.contact_shadows_enabled != 0u {
            split = textureSampleLevel(indirect_fraction_texture, hdr_sampler, in.uv, 0.0).rg;
        }
        if params.ssao_enabled != 0u {
            let ao = textureSampleLevel(ao_texture, hdr_sampler, in.uv, 0.0).r;
            colour = colour * (1.0 - (1.0 - ao) * split.x);
        }
        if params.contact_shadows_enabled != 0u {
            let cs = textureSampleLevel(cs_texture, hdr_sampler, in.uv, 0.0).r;
            colour = colour * (1.0 - (1.0 - cs) * split.y);
        }
    }

    // Add bloom after the occlusion terms, so they do not dim the glow.
    colour = colour + bloom;

    // Eye-Dome Lighting: darken pixels at depth discontinuities.
    // Depth is linearized (z_eye / far, capped at 1 for background) before the
    // log comparison so that the log differences are large enough to produce a
    // visible effect regardless of the near/far plane ratio.
    if params.edl_enabled != 0u && !covered {
        let log_ldc = log(edl_linear_depth(depth));
        let dims_i  = vec2<i32>(depth_dims);
        let edl_r   = i32(max(1.0, round(params.edl_radius)));
        var edl_nc: vec2<i32>;
        var edl_sum = 0.0;
        edl_nc = clamp(depth_coord + vec2<i32>( edl_r,      0), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(edl_linear_depth(depth_texel(depth_texture, depth_sampler, edl_nc))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>( edl_r,  edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(edl_linear_depth(depth_texel(depth_texture, depth_sampler, edl_nc))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(      0,  edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(edl_linear_depth(depth_texel(depth_texture, depth_sampler, edl_nc))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(-edl_r,  edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(edl_linear_depth(depth_texel(depth_texture, depth_sampler, edl_nc))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(-edl_r,      0), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(edl_linear_depth(depth_texel(depth_texture, depth_sampler, edl_nc))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(-edl_r, -edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(edl_linear_depth(depth_texel(depth_texture, depth_sampler, edl_nc))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>(      0, -edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(edl_linear_depth(depth_texel(depth_texture, depth_sampler, edl_nc))) - log_ldc);
        edl_nc = clamp(depth_coord + vec2<i32>( edl_r, -edl_r), vec2<i32>(0), dims_i - vec2<i32>(1));
        edl_sum += max(0.0, log(edl_linear_depth(depth_texel(depth_texture, depth_sampler, edl_nc))) - log_ldc);
        // Normalize by sample count then apply exponential response so strength=1
        // gives moderate edge darkening and strength=5 gives near-complete darkening.
        let edl_factor = 1.0 - exp(-params.edl_strength * edl_sum / 8.0);
        colour = colour * (1.0 - edl_factor);
    }

    // Into the working space for the grade and the operator.
    let rec2020 = params.working_space == WORKING_REC2020;
    if rec2020 {
        colour = SRGB_TO_REC2020 * colour;
    }

    // Scene-referred grading, on exposed radiance before the operator.
    if params.grade_mode == GRADE_SCENE {
        colour = grade_scene(colour);
    }

    // Tone mapping. An unknown mode leaves the colour unchanged rather than
    // picking an operator.
    if params.mode == 0u {
        colour = reinhard(colour);
    } else if params.mode == 1u {
        colour = aces(colour);
    } else if params.mode == 2u {
        colour = khronos_neutral(colour);
    } else if params.mode == 3u {
        colour = from_rec2020(agx(to_rec2020(colour, rec2020), params.agx_look), rec2020);
    } else if params.mode == 4u {
        colour = from_rec2020(aces_rrt(to_rec2020(colour, rec2020)), rec2020);
    } else if params.mode == 5u {
        let g = params.generic_curve;
        colour = generic_curve(colour, g.x, g.y, g.z);
    } else if params.mode == MODE_VIEW_LUT {
        colour = sample_strip(view_lut, params.generic_curve.w, shape_scene(colour));
    }

    // Back to Rec.709 for the display-space composite.
    if rec2020 {
        colour = max(REC2020_TO_SRGB * colour, vec3<f32>(0.0));
    }

    // Composite transparent HDR content over the background. Without this, a
    // faint particle over empty scene replaces the background with its own dim
    // premultiplied value (reading as near-black) instead of adding to it.
    // Skipped when opaque geometry is present (it already supplied the base
    // colour tone-mapped above). Done in display space, after tone mapping, so
    // pure-background pixels match the early-out above exactly. This is the only
    // place the background enters: the HDR buffer is cleared to zero, so the
    // quantity added here is added once. Coverage is the transparent alpha (>1
    // possible from stacked additive draws), clamped so saturated regions fully
    // replace the background.
    var coverage = 1.0;
    if is_background {
        let scene = clamp(hdr.a, 0.0, 1.0);
        colour = colour + params.background_colour.rgb * (1.0 - scene);
        // Scene over background, so the two coverages compose. An opaque
        // background still gives 1 here, which is what every consumer that
        // renders to a window wants and what the alpha-1 case has always been.
        // Bloom deliberately contributes none of this: it rides in RGB with no
        // coverage of its own, which composites as the additive glow it is
        // rather than occluding whatever is behind the viewport.
        coverage = scene + params.background_colour.a * (1.0 - scene);
    }

    return vec4<f32>(display_finish(colour, in.uv), coverage);
}
