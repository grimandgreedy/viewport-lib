// Surface contours: lines where a per-vertex scalar crosses a set of levels,
// drawn over a surface that has already been drawn.
//
// The mesh is drawn again, depth-tested against the scene. Each fragment finds
// the distance from its interpolated scalar to the nearest level, converts it
// to pixels through the scalar's screen-space rate of change, and covers the
// pixel by how far inside the line's half width it is.

struct Contour {
    model: mat4x4<f32>,
    // Linear RGBA.
    colour: vec4<f32>,
    // Half the line width, in logical pixels.
    half_width: f32,
    // Viewport width in logical pixels, to turn pass pixels into logical
    // ones under supersampling.
    viewport_width: f32,
    // 0: the listed levels, 1: evenly spaced levels.
    mode: u32,
    // How many of `levels` are in use.
    count: u32,
    origin: f32,
    interval: f32,
    // Smallest gap between levels: what a scalar change is measured against
    // to call a region flat.
    spacing: f32,
    _pad: f32,
    levels: array<vec4<f32>, 8>,
}

@group(1) @binding(0) var<uniform> contour: Contour;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) scalar: f32,
    // NDC x, interpolated linearly on screen so its rate of change is the
    // pass's NDC per pixel.
    @location(1) @interpolate(linear) ndc_x: f32,
    @location(2) world_pos: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) scalar: f32,
) -> VertexOutput {
    // World position first, as the mesh colour shaders do, so the depth this
    // pass tests with is the depth they wrote.
    let world_pos = (contour.model * vec4<f32>(position, 1.0)).xyz;
    var out: VertexOutput;
    out.pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.scalar = scalar;
    out.ndc_x = out.pos.x / out.pos.w;
    out.world_pos = world_pos;
    return out;
}

fn level(i: u32) -> f32 {
    return contour.levels[i / 4u][i % 4u];
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Derivatives first, in uniform control flow.
    let s = in.scalar;
    let ds = fwidth(s);
    // 2 / pass width, constant across the pass.
    let dndc = fwidth(in.ndc_x);

    // Clipped with the surface it sits on.
    if !viewport_clip_test(in.world_pos) {
        discard;
    }
    // NaN never compares equal to itself.
    if s != s {
        discard;
    }

    var d = 3.4e38;
    if contour.mode == 1u {
        let k = round((s - contour.origin) / contour.interval);
        d = abs(s - (contour.origin + k * contour.interval));
    } else {
        for (var i = 0u; i < contour.count; i++) {
            d = min(d, abs(s - level(i)));
        }
    }

    // A plateau sitting on a level would otherwise fill in, since every pixel
    // of it is at distance zero. A field this flat draws no line.
    if ds <= contour.spacing * 1e-4 {
        discard;
    }

    // Pass pixels per logical pixel: above 1 under supersampling.
    let pass_scale = 2.0 / max(dndc * contour.viewport_width, 1e-6);
    let half_width = contour.half_width * pass_scale;
    let dist = d / ds;
    let coverage = 1.0 - smoothstep(half_width - 0.5, half_width + 0.5, dist);
    if coverage <= 0.0 {
        discard;
    }
    return vec4<f32>(contour.colour.rgb, contour.colour.a * coverage);
}
