// @viewport-wgsl-version: 1
//
// The scene depth convention. Depth runs from VIEWPORT_DEPTH_NEAR at the near
// plane to VIEWPORT_DEPTH_FAR at the far plane (or at infinity), and the
// depth buffer clears to the far value.

const VIEWPORT_DEPTH_NEAR: f32 = 0.0;
const VIEWPORT_DEPTH_FAR: f32 = 1.0;

// True where the depth buffer still holds its clear value: no geometry drew.
fn viewport_depth_is_background(d: f32) -> bool {
    return d >= VIEWPORT_DEPTH_FAR;
}

// True when depth `a` is nearer the camera than depth `b`.
fn viewport_depth_nearer(a: f32, b: f32) -> bool {
    return a < b;
}

// The nearer of two depths.
fn viewport_depth_nearest(a: f32, b: f32) -> f32 {
    return min(a, b);
}

// The farther of two depths.
fn viewport_depth_farthest(a: f32, b: f32) -> f32 {
    return max(a, b);
}

// Positive view-space distance in front of the camera for depth `d`.
// `clip` is the camera's (near, far, is_ortho, _) lane; far is 0 for a
// perspective projection with no far plane.
fn viewport_depth_to_view_z(d: f32, clip: vec4<f32>) -> f32 {
    let n = clip.x;
    let f = clip.y;
    if clip.z != 0.0 {
        return n + d * (f - n);
    }
    return n * f / (f - d * (f - n));
}
