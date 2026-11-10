// 2D light accumulation: per light, the shadow geometry cast by every occluder
// edge stamps the light's index into the stencil, then the light's disc adds
// its falloff wherever the stencil does not hold that index.
//
// Group 0: one light's block, at a dynamic offset per light.
//
// Positions are plane coordinates; the block carries the plane's origin and
// axes relative to the camera eye, and a view-projection relative to the eye,
// so nothing large reaches f32.

struct LightBlock {
    view_proj: mat4x4<f32>,
    origin:    vec4<f32>,
    x_axis:    vec4<f32>,
    y_axis:    vec4<f32>,
    // Plane position, radius, unused.
    light:     vec4<f32>,
    // Linear RGB scaled by the light's intensity.
    colour:    vec4<f32>,
};

@group(0) @binding(0) var<uniform> block: LightBlock;

fn plane_clip(p: vec2<f32>) -> vec4<f32> {
    let w = block.origin.xyz + block.x_axis.xyz * p.x + block.y_axis.xyz * p.y;
    return block.view_proj * vec4<f32>(w, 1.0);
}

struct EdgeIn {
    // Edge ends `a` and `b`, plane coordinates.
    @location(0)           ends:      vec4<f32>,
    // 1 for an edge of a closed outline, wound counter-clockwise: it casts
    // only when it faces away from the light, so the occluder's lit side
    // stays lit. 0 for a free segment, which casts from both sides.
    @location(1)           one_sided: u32,
    @builtin(vertex_index) vi:        u32,
};

// The quad an edge casts: the edge, and its two ends pushed out along the
// rays from the light far enough that the far side lies beyond the light's
// reach. Mirrors `shadow_quad` in `mod.rs`.
@vertex
fn vs_shadow(in: EdgeIn) -> @builtin(position) vec4<f32> {
    let l = block.light.xy;
    let a = in.ends.xy;
    let b = in.ends.zw;
    let outward = vec2<f32>(b.y - a.y, a.x - b.x);
    let faces_light = dot(outward, l - a) > 0.0;
    let da = a - l;
    let db = b - l;
    let la = length(da);
    let lb = length(db);
    if (in.one_sided != 0u && faces_light) || la < 1e-6 || lb < 1e-6 {
        // Degenerate: every corner at the same point outside the view.
        return vec4<f32>(2.0, 2.0, 0.5, 1.0);
    }
    let ua = da / la;
    let ub = db / lb;
    let cos_half = sqrt(max((1.0 + dot(ua, ub)) * 0.5, 0.0));
    let far = max(block.light.z / max(cos_half, 1e-3), max(la, lb));
    var p = a;
    switch in.vi {
        case 1u: { p = b; }
        case 2u, 4u: { p = l + ub * far; }
        case 5u: { p = l + ua * far; }
        default: { p = a; }
    }
    return plane_clip(p);
}

@fragment
fn fs_shadow() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

struct LightOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       p:   vec2<f32>,
};

// The square around the light's reach, two triangles.
@vertex
fn vs_light(@builtin(vertex_index) vi: u32) -> LightOut {
    var c = vec2<f32>(-1.0, -1.0);
    switch vi {
        case 1u, 4u: { c = vec2<f32>(1.0, -1.0); }
        case 2u, 3u: { c = vec2<f32>(-1.0, 1.0); }
        case 5u: { c = vec2<f32>(1.0, 1.0); }
        default: { c = vec2<f32>(-1.0, -1.0); }
    }
    var out: LightOut;
    out.p = block.light.xy + c * block.light.z;
    out.pos = plane_clip(out.p);
    return out;
}

// Smooth falloff to zero at the light's reach, added to the light target.
@fragment
fn fs_light(in: LightOut) -> @location(0) vec4<f32> {
    let d = distance(in.p, block.light.xy) / max(block.light.z, 1e-6);
    let f = clamp(1.0 - d * d, 0.0, 1.0);
    return vec4<f32>(block.colour.rgb * (f * f), 0.0);
}
