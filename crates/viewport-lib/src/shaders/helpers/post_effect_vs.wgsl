
// @viewport-wgsl-version: 1
// Fullscreen triangle for post-effect passes: three vertices, no vertex
// buffer, uv in [0, 1] with the origin at the top-left.

struct ViewportPostVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> ViewportPostVsOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    let uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return ViewportPostVsOut(vec4<f32>(p, 0.0, 1.0), uv);
}
