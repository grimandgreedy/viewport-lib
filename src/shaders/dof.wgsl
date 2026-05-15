// dof.wgsl : depth-of-field post-process pass.
//
// Reads the HDR scene texture and the depth buffer. For each pixel, computes
// a circle of confusion (CoC) radius based on how far the pixel's depth is
// from the focal plane, then gathers the HDR colour by sampling a Vogel disc
// of that radius. Near and far regions both blur; pixels inside focal_range
// of focal_distance receive no blur.

struct DofUniform {
    // Linear (view-space) depth of the in-focus plane, in the same units as
    // the scene (positive, distance from camera).
    focal_distance: f32,
    // Half-width of the sharp band around focal_distance (view-space units).
    focal_range:    f32,
    // Maximum blur kernel radius in pixels.
    max_blur_radius: f32,
    // Camera near/far planes, used to linearize the depth buffer.
    near_plane:     f32,
    far_plane:      f32,
    // Viewport dimensions, used to convert pixel offsets to UV offsets.
    viewport_width:  f32,
    viewport_height: f32,
    _pad: f32,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var depth_tex:   texture_depth_2d;
@group(0) @binding(3) var<uniform> params: DofUniform;

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

// Linearize the hardware depth buffer value to view-space eye distance.
// Standard NDC depth: z_eye = near * far / (far - d * (far - near)).
fn linearize_depth(d: f32) -> f32 {
    let n = params.near_plane;
    let f = params.far_plane;
    return n * f / (f - d * (f - n));
}

// Vogel disc sample: golden angle spiral, unit disc radius.
// index in [0, count), returns a 2D offset in [-1, 1]^2 with unit-disc extent.
fn vogel_disc(index: u32, count: u32, rotation: f32) -> vec2<f32> {
    let golden_angle: f32 = 2.399963;
    let r = sqrt(f32(index) + 0.5) / sqrt(f32(count));
    let theta = f32(index) * golden_angle + rotation;
    return r * vec2<f32>(cos(theta), sin(theta));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    let depth = textureLoad(depth_tex, pixel, 0);

    // Background or sky: copy HDR directly without blurring.
    if depth >= 0.9999 {
        return textureSample(hdr_texture, hdr_sampler, in.uv);
    }

    let z_eye = linearize_depth(depth);

    // Circle of confusion: how far outside the sharp band are we?
    let dist_from_focus = abs(z_eye - params.focal_distance) - params.focal_range;
    let coc_pixels = clamp(
        dist_from_focus / max(params.focal_range, 0.001) * params.max_blur_radius,
        0.0,
        params.max_blur_radius,
    );

    // No blur inside the focal range.
    if coc_pixels < 0.5 {
        return textureSample(hdr_texture, hdr_sampler, in.uv);
    }

    // Gather HDR colour with a Vogel disc kernel of radius coc_pixels.
    // Use 16 samples; enough for smooth-looking bokeh at modest radii.
    let num_samples: u32 = 16u;
    let inv_w = 1.0 / params.viewport_width;
    let inv_h = 1.0 / params.viewport_height;
    // Per-pixel rotation based on screen position to break up the disc pattern.
    let rotation = fract(sin(dot(in.uv, vec2<f32>(127.1, 311.7))) * 43758.5453) * 6.28318;

    var colour_sum = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < num_samples; i = i + 1u) {
        let offset = vogel_disc(i, num_samples, rotation) * coc_pixels;
        let sample_uv = in.uv + vec2<f32>(offset.x * inv_w, offset.y * inv_h);
        colour_sum = colour_sum + textureSample(hdr_texture, hdr_sampler, sample_uv).rgb;
    }
    let colour = colour_sum / f32(num_samples);
    return vec4<f32>(colour, 1.0);
}
