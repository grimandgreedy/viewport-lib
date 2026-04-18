// Full-screen analytical grid shader.
//
// No vertex buffer — triangle positions are hardcoded in vs_main (full-screen triangle).
// The fragment shader unprojects each pixel to a world-space ray, intersects with the
// horizontal grid plane y = grid_y, and writes both an analytically anti-aliased grid
// color and the correct clip-space depth via @builtin(frag_depth).
//
// Horizon fade: lines fade to transparent as the viewing angle approaches horizontal,
// eliminating the clipping and mangle artifacts that line-primitive grids suffer from.

struct GridUniform {
    view_proj:    mat4x4<f32>,   // offset   0 — for clip-space depth output
    cam_to_world: mat3x3<f32>,   // offset  64 — camera-to-world rotation (no translation)
    tan_half_fov: f32,           // offset 112 — tan(fov_y/2)
    aspect:       f32,           // offset 116 — viewport width/height
    _pad_ivp:     vec2<f32>,     // offset 120 — padding
    eye_pos:      vec3<f32>,     // offset 128
    grid_y:       f32,           // offset 140
    spacing_minor: f32,          // offset 144
    spacing_major: f32,          // offset 148
    snap_origin:  vec2<f32>,     // offset 152
    color_minor:  vec4<f32>,     // offset 160
    color_major:  vec4<f32>,     // offset 176
}

@group(0) @binding(0) var<uniform> grid: GridUniform;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc:      vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Hardcoded full-screen triangle covering [-1, 1] NDC on both axes.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), p);
}

struct FragOut {
    @location(0)         color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    // Compute world-space ray direction analytically (no matrix inversion).
    // Camera-space direction: (nx * aspect * tan_half, ny * tan_half, -1).
    // This is exact regardless of near/far planes — no ill-conditioning at large distances.
    let dir_cam = vec3<f32>(
        in.ndc.x * grid.aspect * grid.tan_half_fov,
        in.ndc.y * grid.tan_half_fov,
        -1.0,
    );
    // Rotate to world space using the camera orientation (pure rotation, no translation).
    let ray_dir = grid.cam_to_world * dir_cam;

    // Intersect the ray with the horizontal grid plane y = grid_y.
    if abs(ray_dir.y) < 1e-6 { discard; }
    let t = (grid.grid_y - grid.eye_pos.y) / ray_dir.y;
    if t <= 0.0 { discard; }

    let hit = grid.eye_pos + ray_dir * t;

    // Compute the correct clip-space depth for the grid intersection point
    // so that hardware depth testing occludes the grid with geometry drawn later.
    let hit_clip = grid.view_proj * vec4<f32>(hit, 1.0);
    let grid_depth = clamp(hit_clip.z / hit_clip.w, 0.0, 1.0);

    // Horizon fade — |sin| of angle between ray and grid plane.
    // 0 at horizon (ray parallel to plane), 1 looking straight down.
    // Fades lines to transparent near the horizon to eliminate clipping artifacts.
    let angle_sin = abs(ray_dir.y) / length(ray_dir);
    let fade = smoothstep(0.02, 0.10, angle_sin);
    if fade < 0.001 { discard; }

    // Analytical grid lines with fwidth anti-aliasing.
    // fwidth gives the rate of change per pixel, enabling sub-pixel AA without MSAA.
    //
    // Work in snap-origin-relative coordinates so that fract() operates on small numbers.
    // snap_origin = floor(eye.xz / spacing_major) * spacing_major (computed on CPU).
    // Since spacing_major is a power of 10, snap_origin is exactly representable in f32.
    // hit.xz - snap_origin is always within [-spacing_major, +spacing_major].
    let pos = hit.xz - grid.snap_origin;

    // Minor grid lines.
    // smoothstep(0, fw, d): 0 at line center, 1 at one pixel away → 1 - result = line coverage.
    let c_minor  = pos / grid.spacing_minor;
    let d_minor  = abs(fract(c_minor - 0.5) - 0.5);
    let fw_minor = max(fwidth(c_minor), vec2<f32>(1e-4));
    let line_minor  = 1.0 - smoothstep(vec2<f32>(0.0), fw_minor, d_minor);
    let alpha_minor = max(line_minor.x, line_minor.y) * grid.color_minor.a * fade;

    // Major grid lines (every spacing_major units, typically 10x minor).
    let c_major  = pos / grid.spacing_major;
    let d_major  = abs(fract(c_major - 0.5) - 0.5);
    let fw_major = max(fwidth(c_major), vec2<f32>(1e-4));
    let line_major  = 1.0 - smoothstep(vec2<f32>(0.0), fw_major, d_major);
    let alpha_major = max(line_major.x, line_major.y) * grid.color_major.a * fade;

    let final_alpha = clamp(alpha_minor + alpha_major, 0.0, 1.0);
    if final_alpha < 0.001 { discard; }

    // Blend major color over minor color proportional to their contributions.
    let t_blend = clamp(alpha_major / (alpha_minor + alpha_major + 1e-5), 0.0, 1.0);
    let rgb = mix(grid.color_minor.rgb, grid.color_major.rgb, t_blend);

    return FragOut(vec4<f32>(rgb, final_alpha), grid_depth);
}
