// mc_isosurface.wgsl - GPU marching cubes without compute: the isosurface
// ray-marched through the scalar field in a fragment shader, shaded by
// `shade_mc` (mc_shading.wgsl, prepended).
//
// Used where the device has no compute stage or storage buffers (WebGL2). The
// draw is the grid's bounding box, back faces only, so a ray starts at the eye
// when the eye is inside the box. Each fragment marches its ray from the box
// entry at half a cell per step, refines the first isovalue crossing by
// bisection, writes the hit's depth, and shades it with the field's gradient
// as the normal, as the extracted triangles' normals are.
//
// Group 1 : per-draw
//   binding 0 : McSurfaceUniform (mc_shading.wgsl)
//   binding 1 : the scalar field, one R32Float texel per grid node. Read with
//               textureLoad and interpolated here, since 32-bit float
//               filtering is optional on these devices.
//   binding 2 : McIsoUniform

struct McIsoUniform {
    // World position of grid node (0, 0, 0).
    origin:   vec3<f32>,
    isovalue: f32,
    // World distance between neighbouring nodes along each axis.
    spacing:  vec3<f32>,
    // World distance per march step.
    step:     f32,
    // Node count along each axis.
    dims:     vec3<u32>,
    // Upper bound on march steps per ray.
    max_steps: u32,
};

@group(1) @binding(1) var field: texture_3d<f32>;
@group(1) @binding(2) var<uniform> iso: McIsoUniform;

struct IsoVertexOutput {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

// Corner `vi` (0-35) of the unit cube's twelve triangles, wound so the
// outward faces are front-facing.
//
// Two triangles per face, as corner indices into the eight cube corners
// (bit 0 = x, bit 1 = y, bit 2 = z), in face order -z, +z, -y, +y, -x, +x:
//   0 2 3 0 3 1 | 4 5 7 4 7 6 | 0 1 5 0 5 4 | 2 6 7 2 7 3 | 0 4 6 0 6 2 | 1 3 7 1 7 5
// packed three bits an index, ten indices a word. Selected arithmetically
// rather than from a local array, which some GLES compilers (Adreno)
// miscompile when it is indexed by the vertex index.
fn cube_corner(vi: u32) -> vec3<f32> {
    let word = vi / 10u;
    let packed = select(
        select(0x27b0b0d0u, 0x32945237u, word == 1u),
        select(0x161a07d7u, 0x0002f3d9u, word == 3u),
        word >= 2u,
    );
    let c = (packed >> (3u * (vi % 10u))) & 7u;
    return vec3<f32>(f32(c & 1u), f32((c >> 1u) & 1u), f32((c >> 2u) & 1u));
}

fn box_min() -> vec3<f32> {
    return iso.origin;
}

fn box_max() -> vec3<f32> {
    return iso.origin + iso.spacing * vec3<f32>(iso.dims - vec3<u32>(1u));
}

@vertex
fn vs_iso(@builtin(vertex_index) vi: u32) -> IsoVertexOutput {
    let world = mix(box_min(), box_max(), cube_corner(vi));
    var out: IsoVertexOutput;
    out.clip_pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    return out;
}

fn node(i: vec3<i32>) -> f32 {
    let hi = vec3<i32>(iso.dims) - vec3<i32>(1);
    return textureLoad(field, clamp(i, vec3<i32>(0), hi), 0).r;
}

// The field at a world position, trilinear between the eight nodes around it.
fn sample_field(p: vec3<f32>) -> f32 {
    let g = (p - iso.origin) / iso.spacing;
    let hi = vec3<f32>(iso.dims) - vec3<f32>(2.0);
    let base = clamp(floor(g), vec3<f32>(0.0), max(hi, vec3<f32>(0.0)));
    let f = clamp(g - base, vec3<f32>(0.0), vec3<f32>(1.0));
    let i = vec3<i32>(base);
    let c00 = mix(node(i), node(i + vec3<i32>(1, 0, 0)), f.x);
    let c10 = mix(node(i + vec3<i32>(0, 1, 0)), node(i + vec3<i32>(1, 1, 0)), f.x);
    let c01 = mix(node(i + vec3<i32>(0, 0, 1)), node(i + vec3<i32>(1, 0, 1)), f.x);
    let c11 = mix(node(i + vec3<i32>(0, 1, 1)), node(i + vec3<i32>(1, 1, 1)), f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

// The field's gradient at a world position, by central differences half a
// cell either side.
fn field_gradient(p: vec3<f32>) -> vec3<f32> {
    let h = iso.spacing * 0.5;
    return vec3<f32>(
        sample_field(p + vec3<f32>(h.x, 0.0, 0.0)) - sample_field(p - vec3<f32>(h.x, 0.0, 0.0)),
        sample_field(p + vec3<f32>(0.0, h.y, 0.0)) - sample_field(p - vec3<f32>(0.0, h.y, 0.0)),
        sample_field(p + vec3<f32>(0.0, 0.0, h.z)) - sample_field(p - vec3<f32>(0.0, 0.0, h.z)),
    ) / iso.spacing;
}

struct IsoHit {
    found: bool,
    pos:   vec3<f32>,
};

// March the eye ray through `world_pos` (a back face of the box) to the first
// isovalue crossing.
fn march_isosurface(world_pos: vec3<f32>) -> IsoHit {
    var hit = IsoHit(false, vec3<f32>(0.0));
    let eye = camera.eye_pos;
    let dir = normalize(world_pos - eye);
    let t0 = (box_min() - eye) / dir;
    let t1 = (box_max() - eye) / dir;
    let t_near = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), 0.0);
    let t_far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if t_near >= t_far {
        return hit;
    }
    var t_prev = t_near;
    var prev = sample_field(eye + dir * t_prev) - iso.isovalue;
    for (var i = 0u; i < iso.max_steps; i = i + 1u) {
        if t_prev >= t_far {
            break;
        }
        let t = min(t_prev + iso.step, t_far);
        let cur = sample_field(eye + dir * t) - iso.isovalue;
        // A NaN on either side compares false and is stepped over.
        if (prev <= 0.0 && cur > 0.0) || (prev > 0.0 && cur <= 0.0) {
            var lo = t_prev;
            var hi = t;
            let lo_below = prev <= 0.0;
            for (var b = 0u; b < 6u; b = b + 1u) {
                let mid = (lo + hi) * 0.5;
                let below = sample_field(eye + dir * mid) - iso.isovalue <= 0.0;
                if below == lo_below {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            hit = IsoHit(true, eye + dir * ((lo + hi) * 0.5));
            break;
        }
        t_prev = t;
        prev = cur;
    }
    return hit;
}

fn hit_depth(p: vec3<f32>) -> f32 {
    let clip = camera.view_proj * vec4<f32>(p, 1.0);
    return clip.z / clip.w;
}

struct IsoFragmentOutput {
    @location(0)          colour: vec4<f32>,
    @builtin(frag_depth)  depth:  f32,
};

@fragment
fn fs_iso(in: IsoVertexOutput) -> IsoFragmentOutput {
    let hit = march_isosurface(in.world_pos);
    if !hit.found {
        discard;
    }
    // A flat field has no gradient; face the viewer instead.
    let g = field_gradient(hit.pos);
    let n = select(camera.eye_pos - hit.pos, g, dot(g, g) > 1e-20);
    var out: IsoFragmentOutput;
    out.colour = shade_mc(hit.pos, n);
    out.depth = hit_depth(hit.pos);
    return out;
}

// The selection mask: the same march, writing coverage.
@fragment
fn fs_iso_mask(in: IsoVertexOutput) -> IsoFragmentOutput {
    let hit = march_isosurface(in.world_pos);
    if !hit.found {
        discard;
    }
    var out: IsoFragmentOutput;
    out.colour = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    out.depth = hit_depth(hit.pos);
    return out;
}
