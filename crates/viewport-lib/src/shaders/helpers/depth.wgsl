// @viewport-wgsl-version: 1
//
// The scene depth convention. Depth is reversed: VIEWPORT_DEPTH_NEAR (1) at
// the near plane and VIEWPORT_DEPTH_FAR (0) at the far plane, which for a
// perspective camera is at infinity. The depth buffer clears to the far
// value. Shadow maps use standard depth and do not follow these helpers.

const VIEWPORT_DEPTH_NEAR: f32 = 1.0;
const VIEWPORT_DEPTH_FAR: f32 = 0.0;

// True where the depth buffer still holds its clear value: no geometry drew.
fn viewport_depth_is_background(d: f32) -> bool {
    return d <= VIEWPORT_DEPTH_FAR;
}

// True when depth `a` is nearer the camera than depth `b`.
fn viewport_depth_nearer(a: f32, b: f32) -> bool {
    return a > b;
}

// The nearer of two depths.
fn viewport_depth_nearest(a: f32, b: f32) -> f32 {
    return max(a, b);
}

// The farther of two depths.
fn viewport_depth_farthest(a: f32, b: f32) -> f32 {
    return min(a, b);
}

// `d` held off the far end, so a position reconstructed from it stays finite:
// the far end of a perspective projection is at infinity. Background then
// reads as 10,000 near-plane distances away, where a camera with the
// library's default near/far ratio has its far plane. For reconstructions
// that feed screen-space derivatives, which mix in neighbouring background
// pixels.
fn viewport_depth_finite(d: f32) -> f32 {
    return max(d, 1.0e-4);
}

// `d` moved `amount` toward the camera in depth units.
fn viewport_depth_toward_near(d: f32, amount: f32) -> f32 {
    return d + amount;
}

// Depth as a 0..1 value that grows away from the camera: 0 at the near plane
// and 1 at the far plane. For weights and fades written against standard
// depth.
fn viewport_depth_from_near(d: f32) -> f32 {
    return 1.0 - d;
}

// Positive view-space distance in front of the camera for depth `d`.
// `clip` is the camera's (near, far, is_ortho, _) lane; far is 0 for a
// perspective projection with no far plane, where background depth reads
// as a very large distance.
fn viewport_depth_to_view_z(d: f32, clip: vec4<f32>) -> f32 {
    let n = clip.x;
    let f = clip.y;
    if clip.z != 0.0 {
        return n + (1.0 - d) * (f - n);
    }
    if f == 0.0 {
        return n / max(d, 1.0e-30);
    }
    return n * f / (d * (f - n) + n);
}

// The world-space ray through a screen point, from an inverse
// view-projection in this convention: `origin` on the near plane and a unit
// `dir`. Stays finite when the far plane is at infinity.
struct ViewportRay {
    origin: vec3<f32>,
    dir:    vec3<f32>,
}

fn viewport_ndc_ray(inv_view_proj: mat4x4<f32>, ndc_xy: vec2<f32>) -> ViewportRay {
    let near_h = inv_view_proj * vec4<f32>(ndc_xy, VIEWPORT_DEPTH_NEAR, 1.0);
    let far_h = inv_view_proj * vec4<f32>(ndc_xy, VIEWPORT_DEPTH_FAR, 1.0);
    let origin = near_h.xyz / near_h.w;
    // A point at infinity has w = 0 and its xyz is the direction.
    return ViewportRay(origin, normalize(far_h.xyz - origin * far_h.w));
}
