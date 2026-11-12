// 2D light accumulation: per light, the shadow geometry cast by every occluder
// edge stamps the light's index into the stencil, then the light's shape adds
// its light wherever the stencil does not hold that index. A light with soft
// shadows instead draws its shadow geometry into a mask, which is blurred,
// and adds its light through the mask.
//
// With normals on, a second target takes the direction to the light per
// fragment, in world space and weighted by the luminance the light adds.
//
// Group 0: one light's block, at a dynamic offset per light.
// Group 1: the light's texture, white for a light without one.
// Group 2: the blurred shadow mask, for a light with soft shadows.
//
// Positions are plane coordinates; the block carries the plane's origin and
// axes relative to the camera eye, and a view-projection relative to the eye,
// so nothing large reaches f32.

struct LightBlock {
    view_proj: mat4x4<f32>,
    origin:    vec4<f32>,
    // The plane's x axis, and in `w` the light's height above the plane.
    x_axis:    vec4<f32>,
    y_axis:    vec4<f32>,
    // Plane position, shadow reach, radius.
    light:     vec4<f32>,
    // Linear RGB scaled by the light's intensity, and the falloff exponent.
    colour:    vec4<f32>,
    // Cone: direction, cos of the inner and outer half-angles.
    // Directional: direction of travel.
    // Textured: cos and sin of the rotation, half the size.
    params:    vec4<f32>,
    // The plane rectangle the light's shape covers: centre, half size.
    bounds:    vec4<f32>,
    // Kind, falloff, polygon point count, unused.
    flags:     vec4<u32>,
    // Polygon outline relative to the light, two points per element.
    points:    array<vec4<f32>, 32>,
};

const KIND_POINT: u32 = 0u;
const KIND_CONE: u32 = 1u;
const KIND_DIRECTIONAL: u32 = 2u;
const KIND_GLOBAL: u32 = 3u;
const KIND_TEXTURED: u32 = 4u;
const KIND_POLYGON: u32 = 5u;

@group(0) @binding(0) var<uniform> block: LightBlock;
@group(1) @binding(0) var light_texture: texture_2d<f32>;
@group(1) @binding(1) var light_sampler: sampler;
@group(2) @binding(0) var shadow_mask: texture_2d<f32>;

fn plane_clip(p: vec2<f32>) -> vec4<f32> {
    let w = block.origin.xyz + block.x_axis.xyz * p.x + block.y_axis.xyz * p.y;
    return block.view_proj * vec4<f32>(w, 1.0);
}

fn covers_plane() -> bool {
    return block.flags.x == KIND_DIRECTIONAL || block.flags.x == KIND_GLOBAL;
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

// The quad an edge casts: the edge, and its two ends pushed away from the
// light far enough that the far side lies beyond the light's reach. From a
// light at a point the ends move along the rays from it, mirroring
// `shadow_quad` in `mod.rs`; from a directional light both move along its
// direction of travel.
@vertex
fn vs_shadow(in: EdgeIn) -> @builtin(position) vec4<f32> {
    let a = in.ends.xy;
    let b = in.ends.zw;
    let outward = vec2<f32>(b.y - a.y, a.x - b.x);
    let reach = block.light.z;
    var far_a = a;
    var far_b = b;
    var faces_light = false;
    var degenerate = false;
    if block.flags.x == KIND_DIRECTIONAL {
        let d = block.params.xy;
        faces_light = dot(outward, -d) > 0.0;
        far_a = a + d * reach;
        far_b = b + d * reach;
    } else {
        let l = block.light.xy;
        faces_light = dot(outward, l - a) > 0.0;
        let da = a - l;
        let db = b - l;
        let la = length(da);
        let lb = length(db);
        degenerate = la < 1e-6 || lb < 1e-6;
        let ua = da / max(la, 1e-6);
        let ub = db / max(lb, 1e-6);
        let cos_half = sqrt(max((1.0 + dot(ua, ub)) * 0.5, 0.0));
        let far = max(reach / max(cos_half, 1e-3), max(la, lb));
        far_a = l + ua * far;
        far_b = l + ub * far;
    }
    if (in.one_sided != 0u && faces_light) || degenerate {
        // Every corner at the same point outside the view.
        return vec4<f32>(2.0, 2.0, 0.5, 1.0);
    }
    var p = a;
    switch in.vi {
        case 1u: { p = b; }
        case 2u, 4u: { p = far_b; }
        case 5u: { p = far_a; }
        default: { p = a; }
    }
    return plane_clip(p);
}

@fragment
fn fs_shadow() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

// The shadow geometry into a soft light's mask: one where shadowed.
@fragment
fn fs_shadow_mask() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

struct LightOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       p:   vec2<f32>,
};

// The rectangle the light's shape covers, two triangles; the whole view for
// a light over the whole plane.
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
    if covers_plane() {
        out.p = vec2<f32>(0.0);
        out.pos = vec4<f32>(c, 0.5, 1.0);
        return out;
    }
    out.p = block.bounds.xy + c * block.bounds.zw;
    out.pos = plane_clip(out.p);
    return out;
}

// The falloff at `d`, the distance over the radius: one at the centre, zero
// from one out.
fn falloff(d: f32) -> f32 {
    let x = clamp(d, 0.0, 1.0);
    switch block.flags.y {
        case 0u: { return 1.0 - x; }
        case 2u: {
            let k = 16.0;
            return max((1.0 / (1.0 + k * x * x) - 1.0 / (1.0 + k)) * (1.0 + k) / k, 0.0);
        }
        case 3u: { return select(pow(1.0 - x, max(block.colour.w, 1e-3)), 0.0, x >= 1.0); }
        default: {
            let f = 1.0 - x * x;
            return f * f;
        }
    }
}

fn polygon_point(i: u32) -> vec2<f32> {
    let v = block.points[i / 2u];
    return select(v.xy, v.zw, (i & 1u) == 1u);
}

// Signed distance from `p` (relative to the light) to the polygon outline,
// negative inside.
fn polygon_distance(p: vec2<f32>) -> f32 {
    let n = block.flags.z;
    let first = p - polygon_point(0u);
    var d = dot(first, first);
    var s = 1.0;
    var j = n - 1u;
    for (var i = 0u; i < n; i = i + 1u) {
        let vi = polygon_point(i);
        let vj = polygon_point(j);
        let e = vj - vi;
        let w = p - vi;
        let t = clamp(dot(w, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
        let q = w - e * t;
        d = min(d, dot(q, q));
        let c = vec3<bool>((p.y >= vi.y), (p.y < vj.y), (e.x * w.y > e.y * w.x));
        if all(c) || all(!c) {
            s = -s;
        }
        j = i;
    }
    return s * sqrt(d);
}

// The light's colour and shape at a plane point.
fn light_at(in: LightOut) -> vec3<f32> {
    let kind = block.flags.x;
    let l = block.light.xy;
    let radius = max(block.light.w, 1e-6);
    let rel = in.p - l;

    // A textured light's sample, taken for every kind so the sample stays
    // out of the branches.
    let cs = block.params.xy;
    let half_size = max(block.params.zw, vec2<f32>(1e-6));
    let local = vec2<f32>(cs.x * rel.x + cs.y * rel.y, -cs.y * rel.x + cs.x * rel.y) / half_size;
    let uv = vec2<f32>(local.x * 0.5 + 0.5, 0.5 - local.y * 0.5);
    let texel = textureSampleLevel(light_texture, light_sampler, uv, 0.0);

    var f = 1.0;
    var tint = vec3<f32>(1.0);
    if kind == KIND_POINT {
        f = falloff(length(rel) / radius);
    } else if kind == KIND_CONE {
        let len = length(rel);
        let cos_angle = select(1.0, dot(rel / max(len, 1e-6), block.params.xy), len > 1e-6);
        f = falloff(len / radius) * smoothstep(block.params.w, block.params.z, cos_angle);
    } else if kind == KIND_TEXTURED {
        let inside = all(abs(local) <= vec2<f32>(1.0));
        f = select(0.0, texel.a, inside);
        tint = texel.rgb;
    } else if kind == KIND_POLYGON {
        let sd = polygon_distance(rel);
        if sd > 0.0 {
            f = select(0.0, falloff(sd / radius), block.light.w > 0.0);
        }
    }
    return block.colour.rgb * tint * f;
}

// The direction texel for `colour` added at a plane point: the unit world
// direction from the point to the light, raised by the light's height, times
// the luminance added, and that luminance. Global and polygon lights add no
// direction, so what they light is shaded flat.
fn light_direction(in: LightOut, colour: vec3<f32>) -> vec4<f32> {
    let kind = block.flags.x;
    if kind == KIND_GLOBAL || kind == KIND_POLYGON {
        return vec4<f32>(0.0);
    }
    let h = block.x_axis.w;
    var v = vec3<f32>(block.light.xy - in.p, h);
    if kind == KIND_DIRECTIONAL {
        v = vec3<f32>(-block.params.xy, h);
    }
    let normal = cross(block.x_axis.xyz, block.y_axis.xyz);
    let world = block.x_axis.xyz * v.x + block.y_axis.xyz * v.y + normal * v.z;
    let len = length(world);
    // A fragment right under a light with no height: light from straight
    // above.
    let l = select(normal, world / max(len, 1e-12), len > 1e-6);
    let w = dot(colour, vec3<f32>(0.2126, 0.7152, 0.0722));
    return vec4<f32>(l * w, w);
}

// How much of a soft light reaches the fragment: one minus its blurred
// shadow mask.
fn unshadowed(pos: vec4<f32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(shadow_mask));
    let texel = clamp(vec2<i32>(floor(pos.xy)), vec2<i32>(0), dims - vec2<i32>(1));
    return 1.0 - textureLoad(shadow_mask, texel, 0).r;
}

struct LightTargets {
    @location(0) colour: vec4<f32>,
    @location(1) direction: vec4<f32>,
};

// The light added to the light target.
@fragment
fn fs_light(in: LightOut) -> @location(0) vec4<f32> {
    return vec4<f32>(light_at(in), 0.0);
}

// `fs_light` through the soft light's mask.
@fragment
fn fs_light_soft(in: LightOut) -> @location(0) vec4<f32> {
    return vec4<f32>(light_at(in) * unshadowed(in.pos), 0.0);
}

// `fs_light` with the direction target.
@fragment
fn fs_light_dir(in: LightOut) -> LightTargets {
    let c = light_at(in);
    return LightTargets(vec4<f32>(c, 0.0), light_direction(in, c));
}

// `fs_light_soft` with the direction target.
@fragment
fn fs_light_dir_soft(in: LightOut) -> LightTargets {
    let c = light_at(in) * unshadowed(in.pos);
    return LightTargets(vec4<f32>(c, 0.0), light_direction(in, c));
}
